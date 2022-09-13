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

#include "classes/attribute.h"

#include <Python.h>

#include <cstdint>

#include "core_util.h"

namespace deepmind::s6 {

namespace {
struct StaticMethodImpl {
  PyObject_HEAD;
  PyObject* sm_callable;
  PyObject* sm_dict;
};
}  // namespace

AttributeDescription::AttributeDescription(ClassManager& mgr, PyObject* name,
                                           PyObject* value,
                                           InstanceOrType instance_or_type)
    : AttributeDescription(mgr.InternString(name), value, instance_or_type) {}

AttributeDescription::AttributeDescription(InternedString name, PyObject* value,
                                           InstanceOrType instance_or_type) {
  name_ = name;
  value_ = value;
  instance_or_type_ = instance_or_type;

  if (value && Py_TYPE(value) == &PyFunction_Type) {
    behavioral_ = true;
    kind_ = Attribute::kFunction;
    return;
  }

  if (value && Py_TYPE(value) == &PyStaticMethod_Type) {
    PyObject* callable =
        reinterpret_cast<StaticMethodImpl*>(value)->sm_callable;
    if (PyFunction_Check(callable)) {
      behavioral_ = true;
      extra_object1_ = callable;
      kind_ = Attribute::kStaticMethod;
      return;
    }
  }

  if (instance_or_type == kType) {
    descrgetfunc descr_get = Py_TYPE(value)->tp_descr_get;
    descrsetfunc descr_set = Py_TYPE(value)->tp_descr_set;
    if (descr_get || descr_set) {
      kind_ = PyDescr_IsData(value) ? Attribute::kDataDescriptor
                                    : Attribute::kNonDataDescriptor;
      behavioral_ = true;
      return;
    }
  }

  if (value && Py_TYPE(value) == &PyCFunction_Type) {
    behavioral_ = true;
    extra_object1_ = PyCFunction_GET_SELF(value);
    extra_object2_ =
        reinterpret_cast<PyObject*>(PyCFunction_GET_FUNCTION(value));
    kind_ = Attribute::kCFunction;
    return;
  }

  // Apart from descriptors, attributes set on a type or an instance behave the
  // same.
  behavioral_ = false;
  kind_ = Attribute::kUnknown;
}

AttributeDescription AttributeDescription::CreateUnknown(ClassManager& mgr,
                                                         absl::string_view name,
                                                         bool behavioral,
                                                         PyObject* value) {
  return AttributeDescription(mgr.InternString(name), behavioral, value,
                              Attribute::kUnknown);
}

AttributeDescription AttributeDescription::CreateUnknown(InternedString name,
                                                         bool behavioral,
                                                         PyObject* value) {
  return AttributeDescription(name, behavioral, value, Attribute::kUnknown);
}

std::shared_ptr<Attribute> AttributeDescription::CreateAttribute(
    ClassManager& mgr) const {
  if (kind_ == Attribute::kFunction) {
    bool is_descriptor = instance_or_type_ == kType;
    return std::make_shared<FunctionAttribute>(mgr, name_, value_,
                                               is_descriptor);
  }
  if (kind_ == Attribute::kCFunction) {
    return std::make_shared<CFunctionAttribute>(mgr, name_, value_);
  }
  if (kind_ == Attribute::kStaticMethod) {
    return std::make_shared<StaticMethodAttribute>(mgr, name_, value_);
  }
  return std::make_shared<Attribute>(mgr, name_, behavioral_, value_, kind_);
}

FunctionAttribute::FunctionAttribute(ClassManager& mgr, InternedString name,
                                     PyObject* function, bool bound)
    : Attribute(mgr, name, true, function, Attribute::kFunction),
      bound_(bound) {
  S6_CHECK(PyFunction_Check(function));
  code_ = reinterpret_cast<PyCodeObject*>(PyFunction_GET_CODE(function));
  Py_INCREF(code_);

  PyObject* defaults_tuple = PyFunction_GET_DEFAULTS(function);
  if (defaults_tuple) {
    S6_CHECK(PyTuple_CheckExact(defaults_tuple));
    for (int64_t i = 0; i < PyTuple_GET_SIZE(defaults_tuple); ++i) {
      defaults_.push_back(PyTuple_GET_ITEM(defaults_tuple, i));
    }
  }

  // Obtain the names for all arguments. This is best-effort - if for some
  // reason an argument name is not an interned unicode it will partially fail,
  // but we only optimize based on positive facts anyway, so the worst damage
  // that a missing argument could do is inhibit an optimization.
  PyObject* varnames = code_->co_varnames;
  if (varnames) {
    S6_CHECK(PyTuple_CheckExact(varnames));
    for (int64_t i = 0; i < PyTuple_GET_SIZE(varnames); ++i) {
      PyObject* obj = PyTuple_GET_ITEM(varnames, i);
      argument_names_[GetObjectAsCheapString(obj)] = i;
    }
  }
}

StaticMethodAttribute::StaticMethodAttribute(ClassManager& mgr,
                                             InternedString name,
                                             PyObject* staticmethod)
    : FunctionAttribute(
          mgr, name,
          reinterpret_cast<StaticMethodImpl*>(staticmethod)->sm_callable,
          /*bound=*/false) {}

CFunctionAttribute::CFunctionAttribute(ClassManager& mgr, InternedString name,
                                       PyObject* function)
    : Attribute(mgr, name, true, function, Attribute::kCFunction) {
  S6_CHECK(PyCFunction_Check(function));
  self_ = PyCFunction_GET_SELF(function);
  flags_ = PyCFunction_GET_FLAGS(function);
  method_ = PyCFunction_GET_FUNCTION(function);
}

Attribute::~Attribute() {}

FunctionAttribute::~FunctionAttribute() { Py_XDECREF(code_); }

CFunctionAttribute::~CFunctionAttribute() {}

}  // namespace deepmind::s6
