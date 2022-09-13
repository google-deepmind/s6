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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_UTIL_H_

#include <Python.h>

#include <cstdint>

namespace deepmind::s6 {

// The type of an address within generated code.
using ProgramAddress = uint64_t;

// The implementation of PyDictKeysObject, which is private to CPython.
struct _PyDictKeysObject {
  Py_ssize_t dk_refcnt;
  Py_ssize_t dk_size;
  void* dk_lookup;
  Py_ssize_t dk_usable;
  Py_ssize_t dk_nentries;
  char dk_indices[];
};

// An entry in a combined dictionary. This is an implementation detail of the
// dict implementation that we copy here so the code generator can use it.
struct _PyDictKeyEntry {
  PyObject* hash;
  PyObject* key;
  PyObject* value;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_UTIL_H_
