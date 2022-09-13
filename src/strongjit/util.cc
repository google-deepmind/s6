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

#include "strongjit/util.h"

#include <Python.h>
#include <opcode.h>

#include <cstdint>

#include "utils/logging.h"

namespace deepmind::s6 {

std::vector<BytecodeInstruction> ExtractInstructions(PyCodeObject* co) {
  absl::Span<const _Py_CODEUNIT> insts(
      reinterpret_cast<_Py_CODEUNIT*>(PyBytes_AS_STRING(co->co_code)),
      PyBytes_Size(co->co_code) / sizeof(_Py_CODEUNIT));
  std::vector<BytecodeInstruction> bytecode_insts;

  int64_t index = 0;
  for (_Py_CODEUNIT inst : insts) {
    bytecode_insts.emplace_back(index++ * 2, _Py_OPCODE(inst), _Py_OPARG(inst));
  }

  return bytecode_insts;
}

// Converts a TryHandler::Kind into its CPython equivalent.
int TryHandlerKindToOpcode(TryHandler::Kind kind) {
  switch (kind) {
    case TryHandler::kExcept:
      return SETUP_EXCEPT;
    case TryHandler::kLoop:
      return SETUP_LOOP;
    case TryHandler::kFinally:
      return SETUP_FINALLY;
    case TryHandler::kExceptHandler:
      return EXCEPT_HANDLER;
    case TryHandler::kFinallyHandler:
      S6_CHECK(false) << "kFinallyHandler has no opcode";
      return -1;  // Unreachable.
  }
  S6_UNREACHABLE();
}

}  // namespace deepmind::s6
