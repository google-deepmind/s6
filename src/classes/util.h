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

#ifndef THIRD_PARTY_DEEPMIND_S6_CLASSES_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_CLASSES_UTIL_H_

#include <Python.h>

#include <cstdint>

namespace deepmind::s6 {

// Attempts to get the __dict__ of an object. Returns nullptr if the type has
// tp_dictoffset <= 0.
PyDictObject* GetObjectDict(PyObject* object);

////////////////////////////////////////////////////////////////////////////////
// Class of an object
//
// A `Class ID` is a numeric value that represents a particular Class object.
// We categorize objects into two categories:
//  * Objects that are behaviorally the same as any other Object of their Type.
//  * Objects that are behaviorally different from other Objects of their Type.
//
// The only supported way of an object becoming behaviorally different from its
// Type is addition of attributes in the object's __dict__. For this reason
// we only support adoption (see below) of objects whose types have the default
// tp_setattro/tp_getattro values (PyObject_Generic[GS]etAttr).
//
// We need to store the class ID of an object somewhere. There is no room in
// PyObject, so we store it in *two* locations: the object's type AND
// the object's __dict__.
//
// If an object has a __dict__, its class ID is found in the `ma_version_tag`
// field (the upper 20 bits of) of PyDictObject.
//
// If an object does NOT have a dict, and it is adoptable by S6, then it is
// behaviourally identical to its type, and therefore the class ID can be looked
// up on the type. We use a spare 32-bits in PyTypeObject (after tp_version_tag)
// to store the class ID.
//
// There are interesting properties of using these storage locations.
//   1) A PyTypeObject is not guaranteed to zero these unused bits on
//      construction, so we use a Py_TPFLAGS_ flag on the type to check if S6
//      has initialized them yet.
//   2) If an object's __dict__ is modified, it will receive a new
//      `ma_version_tag`. This will overwrite the upper 20 bits with zeroes*
//      and therefore set the class ID to zero. This allows us to with zero
//      effort determine if our assumptions are broken.
//
//      * ma_version_tag is bumped at every dict creation or modification.
//        Assuming events occur at 100kHz, 44 bits would take 5.5 years to
//        wrap.
//
// Therefore the algorithm for obtaining an object's class ID is:
//   if (type(obj).tp_flags & Py_TPFLAGS_S6_Adopted == 0)
//     return NO_CLASS // Type has not been adopted, hidden_32_bits is undefined
//   if type(obj).tp_dictoffset <= 0:
//     return NO_CLASS // Negative dictoffset unsupported.
//   if type(obj).tp_dictoffset == 0:
//     return type(obj).hidden_32_bits
//   PyDictObject** dict_ptr = obj + tp_dictoffset;
//   if (!*dict_ptr):
//     return type(obj).hidden_32_bits // Object does not have a dict yet.
//   // Object has a dict, and class ID is in the uppermost 20 bits.
//   return dict_ptr->ma_version_tag >> 44;

// The number of bits in a class ID.
constexpr int64_t kNumClassIdBits = 20;

// Obtains the class ID of a PyObject. Returns zero if the object does not have
// a class.
int64_t GetClassId(PyObject* object);

// Sets the class ID of a PyObject, given its dict as returned by
// GetObjectDict().
void SetClassId(PyDictObject* object_dict, int64_t class_id);

inline int64_t GetClassIdFromType(PyTypeObject* type) {
  // The class ID is stored in the upper 20 bits of tp_flags.
  return type->tp_flags >> (64 - kNumClassIdBits);
}

inline void SetClassIdOnType(PyTypeObject* type, int64_t class_id) {
  type->tp_flags |= class_id << (64 - kNumClassIdBits);
}

// Given an object's __dict__, returns the class ID.
int64_t GetClassIdFromObjectDict(PyDictObject* dict);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CLASSES_UTIL_H_
