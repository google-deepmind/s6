// Copyright 2021 The s6 Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "classes/class.h"

#include <cstdint>
#include <deque>
#include <mutex>  // NOLINT

#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "classes/attribute.h"
#include "classes/class_manager.h"
#include "core_util.h"
#include "event_counters.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {
Class::AttributeMap CalculateAttributes(
    const Class::AttributeMap& base_attributes,
    const Class::AttributeMap& instance_attributes) {
  Class::AttributeMap attributes = base_attributes;
  for (const auto& [k, v] : instance_attributes) {
    auto it = attributes.find(k);
    if (it == attributes.end()) {
      attributes[k] = v;
      continue;
    }

    if (it->second->kind() == Attribute::kDataDescriptor)
      // Data descriptors take precedence over instance attributes.
      continue;
    it->second = v;
  }
  return attributes;
}
}  // namespace

absl::StatusOr<Class*> Class::Create(ClassManager& mgr, absl::string_view name,
                                     PyTypeObject* type) {
  AttributeMap attributes;
  int64_t dictoffset = -1;
  if (type) {
    S6_ASSIGN_OR_RETURN(attributes, GetTypeAttributesAsMap(mgr, type));
    dictoffset = type->tp_dictoffset;
  }
  AttributeMap instance_attributes;
  absl::flat_hash_map<Attribute*, int64_t> instance_slots;

  return mgr.CreateClass(mgr, name, attributes, instance_attributes,
                         instance_slots, nullptr, 0, dictoffset, type,
                         DictKind::kEmpty);
}

absl::StatusOr<Class*> Class::CreateType(ClassManager& mgr,
                                         absl::string_view name,
                                         PyTypeObject* type) {
  S6_ASSIGN_OR_RETURN(AttributeMap attributes,
                      GetMetaTypeAttributesAsMap(mgr, type));
  PyTypeObject* metatype = Py_TYPE(type);
  int64_t dictoffset = metatype->tp_dictoffset;
  AttributeMap instance_attributes;
  absl::flat_hash_map<Attribute*, int64_t> instance_slots;

  // Iterate over tp_dict to find the instance slots.
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(type->tp_dict, &pos, &key, &value)) {
    if (!PyUnicode_CheckExact(key)) {
      return absl::FailedPreconditionError("Attribute key was not unicode");
    }
    AttributeDescription desc(mgr, key, value, AttributeDescription::kType);
    std::shared_ptr<Attribute> attr = attributes.at(desc.name());
    if (*attr == desc) {
      // `pos` is the index within the dict values storage, but has already been
      // advanced.
      instance_slots[attr.get()] = pos - 1;
    }
  }

  DictKind dict_kind = DictKind::kEmpty;
  if (type->tp_dict &&
      _PyDict_HasSplitTable(reinterpret_cast<PyDictObject*>(type->tp_dict))) {
    dict_kind = DictKind::kSplit;
  } else if (type->tp_dict) {
    dict_kind = DictKind::kCombined;
  }

  return mgr.CreateClass(mgr, absl::StrCat("type(", name, ")"), attributes,
                         instance_attributes, instance_slots, nullptr, 0,
                         dictoffset, metatype, dict_kind,
                         /*type_class_type=*/type);
}

absl::StatusOr<Class*> Class::CreateGlobals(ClassManager& mgr,
                                            PyDictObject* globals) {
  S6_ASSIGN_OR_RETURN(AttributeMap instance_attributes,
                      GetGlobalsAsAttributeMap(mgr, globals));

  // Try and give the class a reasonable name. This only helps with debugging.
  // Most globals dicts are actually the dicts of Modules, so will contain
  // __name__.
  std::string name = "globals";
  if (auto it = instance_attributes.find(mgr.InternString("__name__"));
      it != instance_attributes.end() && it->second->value()) {
    absl::string_view s = GetObjectAsCheapString(it->second->value());
    if (!s.empty()) {
      name = absl::StrCat("globals(", s, ")");
    }
  }

  DictKind dict_kind =
      (_PyDict_HasSplitTable(globals)) ? DictKind::kSplit : DictKind::kCombined;
  absl::flat_hash_map<Attribute*, int64_t> instance_slots;
  S6_ASSIGN_OR_RETURN(
      Class * cls,
      mgr.CreateClass(mgr, name, instance_attributes, instance_attributes,
                      instance_slots, dict_kind));
  mgr.globals_classes_.emplace(globals, cls);
  return cls;
}

Class* Class::LookupTransition(ClassTransition transition) const {
  absl::MutexLock lock(&transition_table_mu_);
  auto it = transition_table_.find(transition);
  if (it != transition_table_.end()) {
    return it->second;
  }
  return nullptr;
}

absl::StatusOr<Class*> Class::ApplyTransition(
    ClassTransition transition, const AttributeDescription& attribute,
    DictKind dict_kind) {
  absl::MutexLock lock(&transition_table_mu_);
  auto it = transition_table_.find(transition);
  if (it != transition_table_.end()) {
    // Transition existed, but if this is a non-behavioral attribute the values
    // may differ.
    Class* new_cls = it->second;
    if (transition.kind() != ClassTransition::kDelete) {
      Attribute* new_attribute =
          new_cls->instance_attributes_.at(attribute.name()).get();
      if (new_attribute->value() &&
          new_attribute->value() != attribute.value()) {
        new_attribute->clear_value();
      }
    }
    return new_cls;
  }

  if (transition_chain_length_ + 1 >= kMaxTransitionChainLength) {
    return absl::FailedPreconditionError(
        "Transition chain length would become too long");
  }

  // Create the new instance attribute list.
  AttributeMap instance_attributes = instance_attributes_;
  absl::flat_hash_map<Attribute*, int64_t> instance_slots = instance_slots_;
  switch (transition.kind()) {
    case ClassTransition::kAdd: {
      auto attr = attribute.CreateAttribute(*mgr_);

      instance_slots[attr.get()] = instance_slots.size();
      instance_attributes[attribute.name()] = attr;
      break;
    }

    case ClassTransition::kMutate: {
      auto attr = attribute.CreateAttribute(*mgr_);
      auto& previous_attr = instance_attributes.at(attribute.name());
      instance_slots[attr.get()] = instance_slots[previous_attr.get()];
      instance_slots.erase(previous_attr);
      instance_attributes[attribute.name()] = attr;
      break;
    }

    case ClassTransition::kDelete: {
      instance_attributes.erase(transition.attribute().name());
      instance_slots.clear();
      dict_kind = DictKind::kNonContiguous;
      break;
    }
  }

  AttributeMap attributes =
      CalculateAttributes(base_class_->attributes_, instance_attributes);
  S6_ASSIGN_OR_RETURN(
      Class * cls,
      mgr_->CreateClass(*mgr_, absl::StrCat(name_, transition.description()),
                        std::move(attributes), std::move(instance_attributes),
                        std::move(instance_slots), base_class_,
                        transition_chain_length_ + 1, dictoffset_, type_,
                        dict_kind));
  transition_table_.emplace(std::move(transition), cls);
  return cls;
}

absl::StatusOr<Class*> Class::Transition(const AttributeDescription& attribute,
                                         DictKind dict_kind) {
  auto it = instance_attributes_.find(attribute.name());
  if (it == instance_attributes_.end()) {
    // Addition of a new attribute always takes a transition.
    return ApplyTransition(ClassTransition::Add(attribute, dict_kind),
                           attribute, dict_kind);
  }

  auto& previous_instance_attribute = it->second;

  // There is a behavioral change (and thus a required transition) if either
  // the old or new attribute are behavioral and the attributes are not
  // equivalent.
  bool behavioral_change =
      previous_instance_attribute->behavioral() || attribute.behavioral();
  if (behavioral_change && *previous_instance_attribute != attribute) {
    return ApplyTransition(ClassTransition::Mutate(attribute, dict_kind),
                           attribute, dict_kind);
  }

  // If the previous attribute had a value and this one is different, set
  // the value to nullptr.
  // TODO: This should raise some kind of event for the compiler
  // to catch if it relied on this value.
  if (attribute.value() != previous_instance_attribute->value()) {
    previous_instance_attribute->clear_value();
  }

  // Otherwise there is no behavioral change so there is no transition.
  return this;
}

void Class::UnderlyingTypeHasBeenModified(PyTypeObject* type) {
  S6_VLOG(1) << "Modification of underlying type of class " << name();
  S6_CHECK_EQ(base_class_, this) << "UnderlyingTypeHasBeenModified *MUST* be "
                                    "called on the base class of a type.";
  // We tolerate nullptr types for unit testing.
  if (type) {
    auto attributes_or = GetTypeAttributesAsMap(*mgr_, type);
    if (!attributes_or.ok()) {
      S6_VLOG(1) << "Class became INVALID during type transition!";
      invalid_ = true;
    } else {
      AttributeMap new_attributes = std::move(attributes_or.value());

      for (absl::string_view attr :
           {"__getattr__", "__setattr__", "__delattr__"}) {
        auto before_it = attributes_.find(mgr_->InternString(attr));
        auto after_it = new_attributes.find(mgr_->InternString(attr));
        if ((after_it == new_attributes.end()) ^
            (before_it == attributes_.end())) {
          S6_VLOG(1)
              << "Class is INVALID because it modifies the overriding of "
              << attr;
          invalid_ = true;
        } else if (after_it != new_attributes.end() &&
                   before_it != attributes_.end() &&
                   before_it->second->value() != after_it->second->value()) {
          S6_VLOG(1) << "Class is INVALID because it overrides attribute "
                     << attr << " with a differing value.";
          invalid_ = true;
        }
      }

      attributes_ = std::move(new_attributes);
    }
  }
  // There really had better not be cycles in the transition graph, otherwise
  // we're going to deadlock hard.
  absl::MutexLock lock(&transition_table_mu_);
  for (auto& [transition, cls] : transition_table_) {
    cls->UnderlyingTypeHasBeenModified();
  }

  CallListeners();
}

void Class::UnderlyingTypeHasBeenModified() {
  // The instance attributes haven't changed, so just re-merge with the base
  // attributes.
  attributes_ =
      CalculateAttributes(base_class_->attributes(), instance_attributes_);
  {
    absl::MutexLock lock(&transition_table_mu_);
    for (auto& [transition, cls] : transition_table_) {
      cls->UnderlyingTypeHasBeenModified();
    }
  }

  CallListeners();
}

ClassModificationListener* Class::AddListener(
    ClassModificationListener::Callback callback) {
  listeners_.push_back(
      new ClassModificationListener(std::move(callback), listeners_));
  return &listeners_.back();
}

void Class::CallListeners() {
  for (auto it = listeners_.begin(); it != listeners_.end();) {
    // Early-increment "it" in case Call() destroys the listener.
    ClassModificationListener& listener = *it++;
    listener.Call();
  }
}

void ClassModificationListener::Call() {
  std::move(callback_)();
  Destroy();
}

void ClassModificationListener::Destroy() {
  parent_.erase(this);
  delete this;
}

std::string ClassTransition::description() const {
  switch (transition_kind_) {
    case kAdd:
      return absl::StrCat("+", attribute_.name());
    case kDelete:
      return absl::StrCat("-", attribute_.name());
    case kMutate:
      return absl::StrCat("!", attribute_.name());
  }
  S6_UNREACHABLE();
}

absl::Status MergeNewEntriesIntoAttributeMap(
    ClassManager& mgr, PyObject* dict,
    AttributeDescription::InstanceOrType kind, Class::AttributeMap& map) {
  S6_RET_CHECK(dict && PyDict_Check(dict));

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    if (!PyUnicode_CheckExact(key)) {
      return absl::FailedPreconditionError("Attribute key was not unicode");
    }
    AttributeDescription attr(mgr, key, value, kind);
    // Note that we iterate in MRO order, so already-emplaced items should not
    // be overwritten.
    if (!map.contains(attr.name()))
      map.emplace(attr.name(), attr.CreateAttribute(mgr));
  }
  return absl::OkStatus();
}

absl::StatusOr<Class::AttributeMap> GetTypeAttributesAsMap(ClassManager& mgr,
                                                           PyTypeObject* type) {
  if ((type->tp_flags & Py_TPFLAGS_READY) == 0)
    return absl::FailedPreconditionError(
        "A non-ready type will not have correct MRO");
  PyObject* mro = type->tp_mro;
  Class::AttributeMap result;
  if (!mro) {
    return absl::FailedPreconditionError("Type has an empty MRO");
  }
  S6_RET_CHECK(PyTuple_CheckExact(mro));

  if (type->tp_dict) {
    S6_RETURN_IF_ERROR(MergeNewEntriesIntoAttributeMap(
        mgr, type->tp_dict, AttributeDescription::kType, result));
    S6_VLOG(1)
        << "Merging tp_dict for type " << type->tp_name << " has "
        << result.size() << " elems and ma_version "
        << reinterpret_cast<PyDictObject*>(type->tp_dict)->ma_version_tag;
  }

  Py_INCREF(mro);
  for (int64_t i = 0, n = PyTuple_GET_SIZE(mro); i < n; ++i) {
    PyObject* base = PyTuple_GET_ITEM(mro, i);
    S6_CHECK(PyType_Check(base));
    PyObject* dict = reinterpret_cast<PyTypeObject*>(base)->tp_dict;
    S6_RETURN_IF_ERROR(MergeNewEntriesIntoAttributeMap(
        mgr, dict, AttributeDescription::kType, result));
  }
  Py_DECREF(mro);
  return result;
}

absl::StatusOr<Class::AttributeMap> GetMetaTypeAttributesAsMap(
    ClassManager& mgr, PyTypeObject* type) {
  // The lookup algorithm in type_getattro() does two _PyType_Lookup()s - first
  // on the metatype and secondly on the type itself. It prioritizes data
  // descriptors on the metatype, then the local _PyType_Lookup(), then any
  // other attribute on the metatype.

  S6_ASSIGN_OR_RETURN(Class::AttributeMap metatype_attrs,
                      GetTypeAttributesAsMap(mgr, Py_TYPE(type)));
  S6_ASSIGN_OR_RETURN(Class::AttributeMap type_attrs,
                      GetTypeAttributesAsMap(mgr, type));

  // Take type_attrs as our result, and replace any data descriptors found in
  // metatype_attrs.
  for (auto& [name, attr] : metatype_attrs) {
    if (attr->kind() == Attribute::kDataDescriptor) {
      // Data descriptors in the metaclass shadow anything else.
      type_attrs[name] = attr;
    } else {
      // If this exists in type_attrs already, that takes priority.
      type_attrs.emplace(name, attr);
    }
  }
  return type_attrs;
}

absl::StatusOr<Class::AttributeMap> GetGlobalsAsAttributeMap(
    ClassManager& mgr, PyDictObject* globals) {
  Class::AttributeMap attributes;
  S6_RETURN_IF_ERROR(MergeNewEntriesIntoAttributeMap(
      mgr, (PyObject*)globals, AttributeDescription::kInstance, attributes));

  // Globals may have __builtins__ that should also have its contents merged.
  // Globals shadows builtins; any keys within globals are not overwritten by
  // those in builtins. Note that `builtins` is a borrowed reference.
  PyObject* builtins = PyDict_GetItemString(
      reinterpret_cast<PyObject*>(globals), "__builtins__");
  if (builtins && PyModule_Check(builtins)) {
    S6_RETURN_IF_ERROR(MergeNewEntriesIntoAttributeMap(
        mgr, PyModule_GetDict(builtins), AttributeDescription::kInstance,
        attributes));
  } else if (builtins && PyDict_Check(builtins)) {
    S6_RETURN_IF_ERROR(MergeNewEntriesIntoAttributeMap(
        mgr, builtins, AttributeDescription::kInstance, attributes));
  } else {
    return absl::FailedPreconditionError("No builtins");
  }
  return attributes;
}

void Class::SetItemForGlobals(PyObject* name, PyObject* value) {
  AttributeDescription attr(*mgr_, name, value,
                            AttributeDescription::kInstance);
  auto it = attributes_.find(attr.name());
  if (it == attributes_.end()) {
    // Newly added attribute.
    auto a = attr.CreateAttribute(*mgr_);
    attributes_.emplace(attr.name(), a);
    instance_attributes_.emplace(attr.name(), a);
    return;
  }

  // Attribute already existed.
  Attribute* existing_attr = it->second.get();
  if (existing_attr->value() == nullptr ||
      existing_attr->value() == attr.value()) {
    // No change to behavior.
    return;
  }

  // Behavior has changed. We must safely drain the compilation queue, set
  // value() to nullptr, and dispose of any code that relied on the prior value.
  mgr_->SafelyRunTypeHasBeenModified([&]() {
    existing_attr->clear_value();
    CallListeners();
  });
}

void Class::ReinitializeGlobals(PyDictObject* dict) {
  auto new_attributes_or = GetGlobalsAsAttributeMap(*mgr_, dict);
  if (!new_attributes_or.ok()) {
    invalid_ = true;
    return;
  }
  Class::AttributeMap new_attributes = std::move(new_attributes_or.value());

  std::vector<InternedString> attributes_to_clear;
  std::vector<InternedString> attributes_to_remove;

  for (auto old_it = attributes_.begin(); old_it != attributes_.end();
       ++old_it) {
    auto new_it = new_attributes.find(old_it->first);
    if (new_it == new_attributes.end()) {
      // This attribute has been deleted.
      attributes_to_remove.push_back(old_it->first);
      continue;
    }
    if (old_it->second->value() &&
        new_it->second->value() != old_it->second->value()) {
      // This attribute now has a different value (and the value was non-nullptr
      // before).
      attributes_to_clear.push_back(new_it->first);
    }
    // Erase from new_attributes such that at the end of the loop,
    // new_attributes just contains all attributes to add.
    new_attributes.erase(new_it);
  }

  if (!attributes_to_clear.empty() || !attributes_to_remove.empty() ||
      !new_attributes.empty()) {
    mgr_->SafelyRunTypeHasBeenModified([&]() {
      for (const InternedString& s : attributes_to_clear) {
        attributes_.find(s)->second->clear_value();
      }
      for (const InternedString& s : attributes_to_remove) {
        attributes_.erase(s);
      }
      for (const auto& [name, attr] : new_attributes) {
        attributes_.emplace(name, attr);
      }
    });
    CallListeners();
  }
  instance_attributes_ = attributes_;
}

}  // namespace deepmind::s6
