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

#include "classes/util.h"

#include <Python.h>

#include <cstdint>

#include "absl/base/casts.h"

namespace deepmind::s6 {
int64_t GetClassIdFromObjectDict(PyDictObject* dict) {
  // The class ID is stored in the upper 20 bits of ma_version_tag. All
  // ma_version_tags that were not set by S6 will have these bits clear (unless
  // the global ma_version_tag has reached 2**44, but that would take 5 years
  // at 100kHz).
  return dict->ma_version_tag >> (64 - kNumClassIdBits);
}

PyDictObject* GetObjectDict(PyObject* object) {
  PyTypeObject* type = Py_TYPE(object);
  if (type->tp_dictoffset <= 0) {
    return nullptr;
  }

  PyDictObject** dict = absl::bit_cast<PyDictObject**>(
      absl::bit_cast<uint8_t*>(object) + type->tp_dictoffset);
  return *dict;
}

// Sets the class ID of a PyObject, given its dict as returned by
// GetObjectDict().
void SetClassId(PyDictObject* object_dict, int64_t class_id) {
  object_dict->ma_version_tag = static_cast<uint64_t>(class_id)
                                << (64 - kNumClassIdBits);
}

int64_t GetClassId(PyObject* object) {
  PyTypeObject* type = Py_TYPE(object);
  if (type->tp_dictoffset < 0) {
    // Classes are not handled for this type.
    return 0;
  }

  if (type->tp_dictoffset > 0) {
    PyDictObject** dict = absl::bit_cast<PyDictObject**>(
        absl::bit_cast<uint8_t*>(object) + type->tp_dictoffset);
    if (*dict) {
      return GetClassIdFromObjectDict(*dict);
    }
  }
  // We get here if an object cannot have a dict OR if it doesn't have a dict.
  return GetClassIdFromType(type);
}

}  // namespace deepmind::s6
