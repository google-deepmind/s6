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

#ifndef THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_H_
#define THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_H_

#include <Python.h>

#include <cstdint>
#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "classes/attribute.h"
#include "classes/class_manager.h"
#include "classes/util.h"
#include "utils/intrusive_list.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

// The format that a dictionary can be in. A dictionary can be in one of two
// modes: split and combined. It takes more work to load or store a value in
// a combined dict.
//
// Once a dict has had an item deleted, we transition to an noncontiguous state
// where we don't guarantee any more the slot indices of instance attributes.
enum class DictKind {
  // There are no instance attributes, so the dictionary could be in any state.
  kEmpty,
  // The dictionary is split (ma_keys is the hashtable, ma_values is the value
  // list).
  kSplit,
  // The dictionary is combined (ma_keys has both the hashtable and values
  // list).
  kCombined,
  // The dictionary has had an item deleted.
  kNonContiguous
};

// A transition between hidden classes. This is a lightweight description of
// the addition, mutation or deletion of attributes.
class ClassTransition {
 public:
  enum Kind { kAdd, kDelete, kMutate };

  // We take `dict_kind` here and treat it as a key for the transition
  // because deleting an attribute from an object deletes the cached keys object
  // completely, meaning all new instances of the type (or any additions to any
  // existing instances) will revert to a dict in combined form.
  static ClassTransition Add(AttributeDescription attribute,
                             DictKind dict_kind) {
    return ClassTransition(kAdd, std::move(attribute), dict_kind);
  }
  static ClassTransition Delete(InternedString attribute_name) {
    return ClassTransition(kDelete, attribute_name);
  }
  static ClassTransition Mutate(AttributeDescription attribute,
                                DictKind dict_kind) {
    return ClassTransition(kMutate, std::move(attribute), dict_kind);
  }

  Kind kind() const { return transition_kind_; }
  const AttributeDescription& attribute() const { return attribute_; }

  // Returns a description of this transition suitable for appending to a class
  // name.
  std::string description() const;

  template <typename H>
  friend H AbslHashValue(H h, const ClassTransition& c) {
    return H::combine(std::move(h), c.attribute_, c.transition_kind_,
                      c.dict_kind_);
  }

  bool operator==(const ClassTransition& other) const {
    return std::tie(attribute_, transition_kind_, dict_kind_) ==
           std::tie(other.attribute_, other.transition_kind_, other.dict_kind_);
  }

 private:
  ClassTransition(Kind kind, AttributeDescription attribute, DictKind dict_kind)
      : attribute_(attribute), transition_kind_(kind), dict_kind_(dict_kind) {
    // The value is important if the attribute is behavioral. Otherwise
    // two additions of the same attribute with different values should
    // result in the same transition.
    if (!attribute_.behavioral()) attribute_.clear_value();
  }

  ClassTransition(Kind kind, InternedString attribute_name)
      : attribute_(AttributeDescription::CreateUnknown(attribute_name, false,
                                                       nullptr)),
        transition_kind_(kind),
        dict_kind_(DictKind::kNonContiguous) {
    S6_CHECK_EQ(kind, kDelete);
  }

  AttributeDescription attribute_;
  Kind transition_kind_;
  DictKind dict_kind_;
};

// A ClassModificationListener is notified whenever the behavior of a class
// changes. This happens if the underlying type has been modified.
class ClassModificationListener
    : public IntrusiveLink<ClassModificationListener> {
 public:
  using list_type = IntrusiveList<ClassModificationListener>;
  // The type of a listener callback. Once the callback returns, the Listener
  // object is destroyed.
  //
  // This function is guaranteed to be called no more than once.
  using Callback = std::function<void()>;

  // Destroys the listener without calling the callback. Deletes this
  // object.
  void Destroy();

 private:
  friend class Class;
  // Default destructor, but private.
  ~ClassModificationListener() {}

  // Only Class can construct a listener object. List objects are always
  // constructed with operator new.
  ClassModificationListener(Callback callback, list_type& parent)
      : callback_(std::move(callback)), parent_(parent) {}

  // Only callable by Class.
  void Call();

  Callback callback_;
  list_type& parent_;
};

// A Class is a hidden class. It forms a parallel type system that provides more
// immutability guarantees than a PyTypeObject.
//
class Class {
 public:
  using AttributeMap =
      absl::flat_hash_map<InternedString, std::shared_ptr<Attribute>>;

  // Creates a new Class. It will receive an automatically generated ID.
  //
  // The name is purely descriptive; it does not have to be unique.
  //
  // If a PyTypeObject is given, it is scanned and attributes are extracted.
  static absl::StatusOr<Class*> Create(ClassManager& mgr,
                                       absl::string_view name,
                                       PyTypeObject* type = nullptr);

  // Creates a new Class for a globals dict. All dicts used as globals() by
  // code objects get a unique Class.
  static absl::StatusOr<Class*> CreateGlobals(ClassManager& mgr,
                                              PyDictObject* globals);

  // Creates a new Class for a type object. This differs from a normal class in
  // that the class represents attributes on the type itself, not on instances
  // of that type.
  //
  //   class C(object): pass
  //   classof(C).is_type_class() -> True
  //   classof(C()).is_type_class() -> False
  //
  static absl::StatusOr<Class*> CreateType(ClassManager& mgr,
                                           absl::string_view name,
                                           PyTypeObject* type);

  // The ID is a dense integer unique to this class. The identifier zero is
  // reserved and belongs to no class.
  int64_t id() const { return id_; }

  // The name of this class.
  absl::string_view name() const { return name_; }

  // The attributes within instances of this class.
  const AttributeMap& attributes() const { return attributes_; }

  // True if the class is invalid. A class can become invalid if __getattr__
  // or __setattr__ are defined on it dynamically.
  bool invalid() const { return invalid_; }

  // Applies the given transition and returns the new Class. A new Class is
  // created if required.
  //
  // Returns failure if a new Class could not be created or if the transition
  // chain becomes too long.
  //
  // The returned Class is owned by the ClassManager.
  //
  // This function is thread safe.
  absl::StatusOr<Class*> ApplyTransition(ClassTransition transition,
                                         const AttributeDescription& attribute,
                                         DictKind dict_kind);

  // Determines the ClassTransition to apply, given `attribute` is being set on
  // an instance of this class.
  //
  // The returned Class is owned by the ClassManager.
  //
  // If this returns OK, it may return `this` (if the attribute set does NOT
  // cause a transition) or a new Class to transition to.
  absl::StatusOr<Class*> Transition(const AttributeDescription& attribute,
                                    DictKind dict_kind);

  // Looks up an existing class transition. If the transition exists, returns
  // the transitioned-to Class. Otherwise returns nullptr.
  Class* LookupTransition(ClassTransition transition) const;

  // Called when the type underlying this class has been modified. The class
  // must re-analyze.
  //
  // This is thread-antagonistic. It must only be called when no other threads
  // could be accessing Class data.
  void UnderlyingTypeHasBeenModified(PyTypeObject* type);

  // Given an Attribute, attempts to find its slot in the instance dictionary.
  // This slot is an index into the dict's `ma_values` list (for split dicts).
  //
  // Returns -1 if no slot index is known for the given Attribute.
  int64_t GetInstanceSlot(const Attribute& attr) const {
    if (auto it = instance_slots_.find(&attr); it != instance_slots_.end()) {
      return it->second;
    }
    return -1;
  }

  // The tp_dictoffset of the underlying type.
  // REQUIRES: !is_globals_class().
  int64_t dictoffset() const {
    S6_CHECK(!is_globals_class());
    return dictoffset_;
  }

  // The underlying type.
  // REQUIRES: !is_globals_class().
  PyTypeObject* type() const {
    S6_CHECK(!is_globals_class());
    return type_;
  }

  // Returns true if this class does not have any instance attributes.
  // REQUIRES: !is_globals_class(). Base classes do not make sense in the
  // context of globals.
  bool is_base_class() const { return base_class_ == this; }

  // Returns true if this class is a globals class. A globals class is a special
  // class created to represent the globals dict of a function. A globals class
  // has only instance attributes.
  bool is_globals_class() const { return is_globals_class_; }

  // Returns true if this class is a type class. A type class defines the
  // properties of a Type rather than an *instance of that type*. For example:
  //
  //   class C(object): pass
  //   classof(C).is_type_class() -> True
  //   classof(C()).is_type_class() -> False
  //
  // The difference is that that a Type subclass has a different attribute
  // lookup algorithm; it checks the metaclass first whereas normal objects do
  // not.
  bool is_type_class() const { return type_class_type_ != nullptr; }

  // Returns the underlying type for this type class. Note that for a type
  // class, type() returns the *metatype*. This returns the actual type object.
  PyTypeObject* type_class_type() const {
    S6_CHECK(is_type_class());
    return type_class_type_;
  }

  // Adds a new ClassModificationListener. The returned object is owned by
  // this Class. It is guaranteed to be valid until Listener::Call() returns
  // true or Listener::Destroy() is called.
  //
  // Listener::Call() is not guaranteed to be be called on program termination.
  //
  // This function is thread-hostile with respect to
  // UnderlyingTypeHasBeenModified.
  ClassModificationListener* AddListener(
      ClassModificationListener::Callback callback);

  // This class represents `dict`, which is used a a globals dict. Reinitialize
  // the class.
  void ReinitializeGlobals(PyDictObject* dict);

  // This class represents a globals dict. Modify the class definition to
  // reflect PyDict_SetItem(name, value).
  void SetItemForGlobals(PyObject* name, PyObject* value);

  DictKind dict_kind() const { return dict_kind_; }

  // This is a private constructor that is called from a
  // std::deque<>::emplace_back from ClassManager, so cannot be friended.
  Class(int64_t id, ClassManager& mgr, absl::string_view name,
        AttributeMap attributes, AttributeMap instance_attributes,
        absl::flat_hash_map<Attribute*, int64_t> instance_slots,
        Class* base_class, int64_t transition_chain_length, int64_t dictoffset,
        PyTypeObject* type, DictKind dict_kind,
        PyTypeObject* type_class_type = nullptr)
      : mgr_(&mgr),
        id_(id),
        name_(name),
        transition_table_(),
        attributes_(std::move(attributes)),
        instance_attributes_(std::move(instance_attributes)),
        instance_slots_(std::move(instance_slots)),
        base_class_(base_class),
        transition_chain_length_(transition_chain_length),
        dictoffset_(dictoffset),
        type_(type),
        type_class_type_(type_class_type),
        dict_kind_(dict_kind) {
    if (!base_class_) {
      base_class_ = this;
    }
    S6_CHECK(base_class_->instance_attributes_.empty());
  }

  // This is a private constructor that is called from a
  // std::deque<>::emplace_back from ClassManager, so cannot be friended.
  //
  // This constructor creates a globals class.
  Class(int64_t id, ClassManager& mgr, absl::string_view name,
        AttributeMap attributes, AttributeMap instance_attributes,
        absl::flat_hash_map<Attribute*, int64_t> instance_slots,
        DictKind dict_kind)
      : mgr_(&mgr),
        id_(id),
        name_(name),
        transition_table_(),
        attributes_(std::move(attributes)),
        instance_attributes_(std::move(instance_attributes)),
        instance_slots_(std::move(instance_slots)),
        base_class_(nullptr),
        transition_chain_length_(0),
        is_globals_class_(true),
        dict_kind_(dict_kind) {}

 private:
  static constexpr int64_t kMaxTransitionChainLength = 50;

  void UnderlyingTypeHasBeenModified();

  // Calls all listeners. The listener list will be empty after this function
  // returns.
  void CallListeners();

  ClassManager* mgr_;
  int64_t id_;
  std::string name_;
  absl::flat_hash_map<ClassTransition, Class*> transition_table_
      ABSL_GUARDED_BY(transition_table_mu_);
  mutable absl::Mutex transition_table_mu_;

  // The list of attributes. Note that this is not guarded because it is
  // immutable except under UnderlyingTypeHasBeenModified.
  AttributeMap attributes_;

  // The list of instance attributes. This is used to correctly recalculate
  // `attributes` when the underlying type changes.
  AttributeMap instance_attributes_;

  // The map of instance attribute to slot within the dict's `ma_values` list.
  // Deletions and too many additions will force a dict into a "combined" state
  // where ma_values is nullptr; If this is the case, this map will be empty.
  absl::flat_hash_map<Attribute*, int64_t> instance_slots_;

  // The Class that is the beginning of the transition chain; it has no instance
  // attributes.
  Class* base_class_;

  // The number of transitions that have happened since the original class was
  // created. We track this so that we can set an upper bound on the number of
  // Classes created for a single object and avoid antagonistic code:
  //   for i in infinite:
  //     def f(): pass
  //     x.f = f  # A function is a behavioral attribute so mutations force a
  //              # transition.
  int64_t transition_chain_length_;

  // The tp_dictoffset of the underlying type.
  int64_t dictoffset_ = -1;

  // The underlying type.
  PyTypeObject* type_ = nullptr;

  // All listeners that will be notified when a class changes.
  IntrusiveList<ClassModificationListener> listeners_;

  // Is this class a type class? if so, the type object itself (type_ contains
  // the metatype).
  PyTypeObject* type_class_type_ = nullptr;

  // True if the class is invalid. A class can become invalid if __getattr__
  // or __setattr__ are defined on it dynamically.
  bool invalid_ = false;

  // Is this class a globals class? Globals classes only have instance
  // attributes and type/dictoffset queries do not make sense.
  bool is_globals_class_ = false;

  DictKind dict_kind_;
};

// Scans the MRO of `type` and finds all attributes, returning a flattened set.
//
// Note: this is tested in classes_test.py.
absl::StatusOr<Class::AttributeMap> GetTypeAttributesAsMap(ClassManager& mgr,
                                                           PyTypeObject* type);

// Scans the MRO of `type` and its metatype. This mirrors the attribute lookup
// protocol for attributes on a type subclass.
absl::StatusOr<Class::AttributeMap> GetMetaTypeAttributesAsMap(
    ClassManager& mgr, PyTypeObject* type);

// Scans all members of `globals` and globals.__builtins__, returning a
// flattened set of attributes.
absl::StatusOr<Class::AttributeMap> GetGlobalsAsAttributeMap(
    ClassManager& mgr, PyDictObject* globals);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_H_
