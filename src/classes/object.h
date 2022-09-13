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

#ifndef THIRD_PARTY_DEEPMIND_S6_CLASSES_OBJECT_H_
#define THIRD_PARTY_DEEPMIND_S6_CLASSES_OBJECT_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "classes/class.h"

namespace deepmind::s6 {

// Defines functions for interaction between PyObjects and Classes. Note that
// this module is tested from Python in classes/python/classes_test.py.

// Obtains the Class of a PyObject. Returns nullptr if the object does not have
// a class.
Class* GetClass(PyObject* object);

////////////////////////////////////////////////////////////////////////////////
// Adoption
//
// Adoption refers to the process of taking an existing Python object and
// ensuring that S6 can determine its behavior and track changes to its
// behavior.
//
// Adopting an object merely sets the object's class ID to the type's class ID.
// Adopting an object will ensure that its type is adopted.
//
// Adopting a type involves setting the type's class ID (to a new Class) and
// overwriting the type's tp_getattro/tp_setattro fields such that S6 can
// intercept attribute reads and writes. This allows S6 to track behavioral
// changes of objects of that type.

// Returns an OK status if a type is adoptable. A type is adoptable if:
//  It has no custom getter/setter
//  It has tp_dictoffset >= 0.
absl::Status IsAdoptableType(PyTypeObject* type);

// Returns an OK status if an object is adoptable. An object is adoptable if:
//  Its type is adoptable.
//  Its object dictionary does not exist, can not exist, or is empty. Objects
//    with existing instance attributes cannot be adopted (yet).
absl::Status IsAdoptable(PyObject* object);

// Adopts this object.
//
// Adoption determines the Class for the object and creates one if necessary.
// Adoption also sets up callbacks on the type to determine if the object was
// modified.
//
// If this succeeds then GetClass(object) should return non-nullptr.
absl::Status Adopt(PyObject* object);

// Adopts this type.
//
// Sets up callbacks on the type to determine if an object of this type is
// modified.
//
// Because new instances of types have an empty __dict__, and we have set the
// class ID of this type, all new instances of this type are automatically
// adopted.
absl::Status AdoptType(PyTypeObject* type);

// Overrides PyType_Type.tp_new to attempt to detect new types being created
// and to adopt them automatically.
void AdoptNewTypes();

// Undoes AdoptNewTypes().
void StopAdoptingNewTypes();

// Finds all existing subclasses (transitively) of Type and adopts them if
// possible.
void AdoptExistingTypes();

////////////////////////////////////////////////////////////////////////////////
// Globals
//
// Every code object has a globals() dictionary. All global variables are looked
// up from this dictionary. We assign a class to dictionaries used as globals,
// so we can track the behavior of global loads and stores.
//
// Every dictionary used as a globals() dict gets its own class. In general this
// means one class per Module. A Module's dict is actually the same as the
// globals() dict for functions bound within the module.

// Adopts a globals dict. We keep a map of adopted globals dicts, so that if
// they get invalidated we don't try to create new classes.
void AdoptGlobalsDict(PyDictObject* globals);

// Obtains the class ID of a globals dict.
int64_t GetGlobalsClassId(PyDictObject* globals);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CLASSES_OBJECT_H_
