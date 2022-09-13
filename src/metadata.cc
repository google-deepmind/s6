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

#include "metadata.h"

#include "utils/logging.h"
namespace deepmind::s6 {

Metadata* Metadata::Get(PyCodeObject* co) {
  int index = GetPyCodeExtraOffset();

  void* meta = nullptr;
  S6_CHECK_EQ(_PyCode_GetExtra(reinterpret_cast<PyObject*>(co), index, &meta),
              0);
  if (meta == nullptr) {
    meta = new Metadata(co);
    S6_CHECK_EQ(_PyCode_SetExtra(reinterpret_cast<PyObject*>(co), index, meta),
                0);
  }
  return static_cast<Metadata*>(meta);
}

void Metadata::FreeMetadata(void* meta) {
  if (meta) {
    delete static_cast<Metadata*>(meta);
  }
}

int Metadata::GetPyCodeExtraOffset() {
  // TODO: Request an index in s6::GlobalEnable.
  static const int kIndex = _PyEval_RequestCodeExtraIndex(FreeMetadata);
  S6_DCHECK_GE(kIndex, 0);
  return kIndex;
}

}  // namespace deepmind::s6
