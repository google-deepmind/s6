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

#ifndef THIRD_PARTY_DEEPMIND_S6_CLASSES_ATTRIBUTE_H_
#define THIRD_PARTY_DEEPMIND_S6_CLASSES_ATTRIBUTE_H_

#include <Python.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "absl/strings/string_view.h"
#include "classes/class_manager.h"
#include "classes/util.h"
#include "core_util.h"

namespace deepmind::s6 {

// An attribute within a hidden class. For example:
//   class C(object): pass
//   C().x = 2
//
// The object's Class will now have an Attribute called "x". The value() may be
// set to `2`. When `C().x = 3` is executed, the value() field will be
// transitioned to nullptr because there is not a known constant value.
//
// Some attributes are *behavioral*; modifications to their value changes an
// object's behavior. Examples of this are descriptors. These have the
// `behavioral()` flag set.
//
// To create an Attribute, first create an AttributeDescription, then turn it
// into a full-fat Attribute with AttributeDescription::CreateAttribute().
//
// TODO: This is a stub right now but will end up being a class
// hierarchy. Can we use the same mechanisms as Value?
class Attribute {
 public:
  // The kind of an attribute. An attribute's kind determines its behavior.
  enum Kind {
    // TODO: More kinds here: getset, methoddef descriptors, normal
    // attributes. Remove unknown.

    // A generic data descriptor. Data descriptors take precedence over instance
    // attributes that shadow them.
    kDataDescriptor,

    // A generic non-data descriptor.
    kNonDataDescriptor,

    // A PyFunctionObject, which may either act as a descriptor or not.
    kFunction,

    // A PyCFunctionObject. This is never a descriptor.
    kCFunction,

    // A PyStaticMethod object which has a function callable. This is a function
    // that does not have `self` prepended.
    kStaticMethod,

    // We don't know anything about this attribute.
    kUnknown,
  };

  InternedString name() const { return name_; }

  // True if modifications to the value of this attribute affects the behavior
  // of the object and thus require a class transition.
  bool behavioral() const { return behavioral_; }

  // True if the attribute is a descriptor.
  virtual bool IsDescriptor() const {
    return kind_ == kDataDescriptor || kind_ == kNonDataDescriptor;
  }

  // Non-nullptr if this attribute has a known value. This is a borrowed
  // reference.
  //
  // TODO: Return something like a handle from this to allow users to
  // be notified if the attribute transitions to nullptr, for example during
  // compilation.
  PyObject* value() const { return value_.load(); }

  // Sets the value to nullptr.
  void clear_value() { value_ = nullptr; }

  // The value kind. This should always be used instead of querying value()
  // where possible.
  Kind kind() const { return kind_; }

  // If value() is non-nullptr, the class of the value.
  Class* value_class() const { return value_class_; }

  // TODO: Add type information lattice.

  Attribute(ClassManager& mgr, InternedString name, bool behavioral,
            PyObject* value, Kind kind)
      : name_(name),
        value_(value),
        value_class_(nullptr),
        kind_(kind),
        behavioral_(behavioral) {
    if (value_) {
      if (int64_t id = GetClassId(value); id != 0) {
        value_class_ = mgr.GetClassById(id);
      }
    }
  }

  virtual ~Attribute();

 private:
  friend bool operator==(const Attribute& self, const Attribute& other) {
    return std::make_tuple(self.name_, self.behavioral_, self.value_.load(),
                           self.kind_) ==
           std::make_tuple(other.name_, other.behavioral_, other.value_.load(),
                           other.kind_);
  }
  friend bool operator!=(const Attribute& self, const Attribute& other) {
    return !(self == other);
  }

  InternedString name_;
  std::atomic<PyObject*> value_;
  Class* value_class_;
  Kind kind_;
  bool behavioral_;
};

// An attribute that is a PyFunctionObject. This can either be `bound`, in which
// the function is a descriptor and prepends `self` to all calls, or `unbound`,
// in which the function is called as a simple (instance) attribute.
class FunctionAttribute : public Attribute {
 public:
  explicit FunctionAttribute(ClassManager& mgr, InternedString name,
                             PyObject* function, bool bound);

  ~FunctionAttribute() override;

  bool IsDescriptor() const override { return bound(); }

  // If this is a bound function, returns true. A bound function is a method;
  // where the function acts as a descriptor when called and prepends `self`.
  bool bound() const { return bound_; }

  // Returns the code object.
  PyCodeObject* code() const { return code_; }

  // Returns the defaults tuple.
  absl::Span<PyObject* const> defaults() const { return defaults_; }

  // Returns the argument names.
  const absl::flat_hash_map<absl::string_view, int64_t>& argument_names()
      const {
    return argument_names_;
  }

 private:
  // Note that all objects here are borrowed references.
  PyCodeObject* code_;
  std::vector<PyObject*> defaults_;
  absl::flat_hash_map<absl::string_view, int64_t> argument_names_;
  bool bound_;
};

class StaticMethodAttribute : public FunctionAttribute {
 public:
  explicit StaticMethodAttribute(ClassManager& mgr, InternedString name,
                                 PyObject* staticmethod);
};

// An attribute that is a PyCFunctionObject. This is a wrapper around a native
// C function.
//
// It may have a `self`, but it does not implement the descriptor protocol, so
// the number of arguments is always constant (it takes them as a tuple anyway).
class CFunctionAttribute : public Attribute {
 public:
  CFunctionAttribute(ClassManager& mgr, InternedString name,
                     PyObject* function);

  ~CFunctionAttribute() override;

  // The `self` argument to the C function. This can be nullptr.
  PyObject* self() const { return self_; }

  // The flags, which are a combination of METH_* flags from methodobject.h.
  int64_t flags() const { return flags_; }

  // The implementing C function.
  PyCFunction method() const { return method_; }

 private:
  // Note that all objects here are borrowed references.
  PyObject* self_;
  int64_t flags_;
  PyCFunction method_;
};

// A "staging post" for creation of Attributes. An AttributeDescription performs
// initial analysis of a name/value PyObject pair and determines the Attribute
// Kind and whether it is behavioral. This is a plain-old-data object that is
// cheap to construct.
//
// If it is later determined that a full-fat Attribute is needed, the
// AttributeDescription creates it.
class AttributeDescription {
 public:
  enum InstanceOrType {
    // This attribute is set on a Type. Attributes conforming to the descriptor
    // protocol have their tp_descr_get/set function called on attribute
    // lookup/set.
    kType,

    // This attribute is set on an instance of a type. Descriptors are never
    // invoked; tp_descr_get/set are never called on attribute lookup/set.
    kInstance,
  };
  AttributeDescription(ClassManager& mgr, PyObject* name, PyObject* value,
                       InstanceOrType instance_or_type);
  AttributeDescription(InternedString name, PyObject* value,
                       InstanceOrType instance_or_type);

  // For testing purposes, create an attribute with unknown kind.
  static AttributeDescription CreateUnknown(ClassManager& mgr,
                                            absl::string_view name,
                                            bool behavioral, PyObject* value);
  static AttributeDescription CreateUnknown(InternedString name,
                                            bool behavioral, PyObject* value);

  // Converts the AttributeDescription into a full-fat Attribute.
  std::shared_ptr<Attribute> CreateAttribute(ClassManager& mgr) const;

  InternedString name() const { return name_; }
  bool behavioral() const { return behavioral_; }
  PyObject* value() const { return value_; }
  Attribute::Kind kind() const { return kind_; }

  void clear_value() { value_ = nullptr; }

  template <typename H>
  friend H AbslHashValue(H h, const AttributeDescription& a) {
    return H::combine(std::move(h), a.name_, a.behavioral_, a.value_, a.kind_);
  }

  bool operator==(const AttributeDescription& other) const {
    return std::make_tuple(name_, behavioral_, value_, other.instance_or_type_,
                           kind_, extra_object1_, extra_object2_) ==
           std::make_tuple(other.name_, other.behavioral_, other.value_,
                           other.instance_or_type_, other.kind_,
                           other.extra_object1_, other.extra_object2_);
  }

  AttributeDescription() {}

 private:
  AttributeDescription(InternedString name, bool behavioral, PyObject* value,
                       Attribute::Kind kind)
      : name_(name), behavioral_(behavioral), value_(value), kind_(kind) {}
  InternedString name_;
  bool behavioral_ = false;
  PyObject* value_ = nullptr;
  InstanceOrType instance_or_type_ = kType;
  Attribute::Kind kind_ = Attribute::kUnknown;

  // Extra PyObjects whose identity should be part of the description; for
  // example PyCFunction's `self` and `method` members.
  PyObject* extra_object1_ = nullptr;
  PyObject* extra_object2_ = nullptr;
};

// Allow comparison between Attribute and AttributeDescription.
inline bool operator==(const Attribute& a, const AttributeDescription& b) {
  return std::make_tuple(a.name(), a.behavioral(), a.kind()) ==
         std::make_tuple(b.name(), b.behavioral(), b.kind());
}
inline bool operator!=(const Attribute& a, const AttributeDescription& b) {
  return !(a == b);
}
inline bool operator==(const AttributeDescription& a, const Attribute& b) {
  return std::make_tuple(a.name(), a.behavioral(), a.kind()) ==
         std::make_tuple(b.name(), b.behavioral(), b.kind());
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CLASSES_ATTRIBUTE_H_
