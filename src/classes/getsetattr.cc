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

#include <Python.h>

#include "classes/attribute.h"
#include "classes/class_manager.h"
#include "classes/object.h"
#include "event_counters.h"

namespace deepmind::s6 {

PyObject* GenericGetAttrUsingClasses(PyObject* obj, PyObject* name) {
  // Quick bailouts: if obj doesn't have a class, or name is not an ascii
  // unicode object.
  InternedString name_str = ClassManager::Instance().InternString(name);
  Class* cls = GetClass(obj);
  if (!cls || name_str.empty() || cls->invalid()) {
    return PyObject_GenericGetAttr(obj, name);
  }

  auto it = cls->attributes().find(name_str);
  if (it == cls->attributes().end()) {
    // Let CPython deal with raising exceptions.
    return PyObject_GenericGetAttr(obj, name);
  }

  const Attribute& attribute = *it->second;

  // Descriptors must be called. They always have a valid value().
  if (attribute.IsDescriptor()) {
    PyObject* descriptor = attribute.value();
    S6_CHECK(descriptor);
    auto descr_get = Py_TYPE(descriptor)->tp_descr_get;
    if (descr_get) {
      PyObject* value =
          descr_get(descriptor, obj, reinterpret_cast<PyObject*>(Py_TYPE(obj)));
      return value;
    }
  }

  // Attributes with a non-nullptr value() can just return that.
  if (PyObject* value = attribute.value()) {
    Py_INCREF(value);
    return value;
  }

  // Attributes with nullptr value must be instance variables and therefore
  // live in the dict.
  PyDictObject* dict = GetObjectDict(obj);
  // We *must* have a dict here; there is no other way of finding an attribute
  // with non-constant value over all instances of a type.
  S6_CHECK(dict);
  PyObject* result = PyDict_GetItem(reinterpret_cast<PyObject*>(dict), name);
  S6_CHECK(result);
  // PyDict_GetItem returns a borrowed reference.
  Py_INCREF(result);
  return result;
}

int GenericSetAttrUsingClasses(PyObject* obj, PyObject* name, PyObject* value) {
  // Quick bailouts: if obj doesn't have a class, or name is not an ascii
  // unicode object.
  S6_CHECK(!PyErr_Occurred());
  InternedString name_str = ClassManager::Instance().InternString(name);
  Class* cls = GetClass(obj);
  if (!cls || name_str.empty() || cls->invalid()) {
    return PyObject_GenericSetAttr(obj, name, value);
  }

  auto it = cls->attributes().find(name_str);
  const Attribute* attribute =
      it == cls->attributes().end() ? nullptr : it->second.get();

  // Descriptors must be called. They always have a valid value(). We only care
  // about data descriptors here.
  if (attribute && attribute->IsDescriptor()) {
    PyObject* descriptor = attribute->value();
    S6_CHECK(descriptor);
    auto descr_set = Py_TYPE(descriptor)->tp_descr_set;
    if (descr_set) {
      return descr_set(descriptor, obj, value);
    }
  }

  // We're going to either add or mutate an instance variable.
  PyObject** dict_ptr = _PyObject_GetDictPtr(obj);
  if (!dict_ptr) {
    // Object cannot have a dict.
    return PyObject_GenericSetAttr(obj, name, value);
  }

  int result = _PyObjectDict_SetItem(Py_TYPE(obj), dict_ptr, name, value);
  if (result != 0) {
    // Let's assume _PyObjectDict_SetItem is idempotent on failure.
    PyErr_Clear();
    return PyObject_GenericSetAttr(obj, name, value);
  }
  PyDictObject* dict = reinterpret_cast<PyDictObject*>(*dict_ptr);
  DictKind dict_kind = (dict != nullptr && _PyDict_HasSplitTable(dict))
                           ? DictKind::kSplit
                           : DictKind::kCombined;

  AttributeDescription attr(name_str, value, AttributeDescription::kInstance);
  absl::StatusOr<Class*> new_class_or;
  if (!value) {
    // This was an attribute deletion.
    new_class_or = cls->ApplyTransition(ClassTransition::Delete(name_str),
                                        AttributeDescription(), dict_kind);
  } else {
    new_class_or = cls->Transition(attr, dict_kind);
  }
  if (!new_class_or.ok()) return 0;
  Class* new_class = new_class_or.value();

  if (dict) {
    SetClassId(dict, new_class->id());
  }
  return 0;
}

int SetAttrForGlobalsDict(PyObject* dict, PyObject* name, PyObject* value) {
  ClassManager& mgr = ClassManager::Instance();
  InternedString name_str = mgr.InternString(name);
  Class* cls = mgr.GetClassById(
      GetGlobalsClassId(reinterpret_cast<PyDictObject*>(dict)));
  int result = PyDict_SetItem(dict, name, value);

  if (!cls || name_str.empty() || cls->invalid() || !value || result != 0) {
    return result;
  }

  cls->SetItemForGlobals(name, value);
  if (!cls->invalid()) {
    SetClassId(reinterpret_cast<PyDictObject*>(dict), cls->id());
  }
  return 0;
}

}  // namespace deepmind::s6
