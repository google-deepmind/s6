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

#include "core_util.h"

#include <Python.h>

#include <cstdint>
#include <string>

namespace deepmind::s6 {

std::string PyObjectToString(PyObject* obj) {
  PyObject* str = PyObject_Str(obj);
  if (!str) return "<null>";
  PyObject* encoded_str = PyUnicode_AsUTF8String(str);
  if (!encoded_str) return "<null>";
  Py_DECREF(str);
  if (!PyBytes_CheckExact(encoded_str)) return "<non-utf8-string>";
  std::string result = PyBytes_AS_STRING(encoded_str);
  Py_DECREF(encoded_str);
  return result;
}

namespace {
// List of {opcode, name} pairs for all CPython interpreter opcodes. Sorted by
// opcode.
const std::pair<int64_t, absl::string_view> kOpcodeNames[] = {
    {1, "POP_TOP"},
    {2, "ROT_TWO"},
    {3, "ROT_THREE"},
    {4, "DUP_TOP"},
    {5, "DUP_TOP_TWO"},
    {9, "NOP"},
    {10, "UNARY_POSITIVE"},
    {11, "UNARY_NEGATIVE"},
    {12, "UNARY_NOT"},
    {15, "UNARY_INVERT"},
    {16, "BINARY_MATRIX_MULTIPLY"},
    {17, "INPLACE_MATRIX_MULTIPLY"},
    {19, "BINARY_POWER"},
    {20, "BINARY_MULTIPLY"},
    {22, "BINARY_MODULO"},
    {23, "BINARY_ADD"},
    {24, "BINARY_SUBTRACT"},
    {25, "BINARY_SUBSCR"},
    {26, "BINARY_FLOOR_DIVIDE"},
    {27, "BINARY_TRUE_DIVIDE"},
    {28, "INPLACE_FLOOR_DIVIDE"},
    {29, "INPLACE_TRUE_DIVIDE"},
    {50, "GET_AITER"},
    {51, "GET_ANEXT"},
    {52, "BEFORE_ASYNC_WITH"},
    {55, "INPLACE_ADD"},
    {56, "INPLACE_SUBTRACT"},
    {57, "INPLACE_MULTIPLY"},
    {59, "INPLACE_MODULO"},
    {60, "STORE_SUBSCR"},
    {61, "DELETE_SUBSCR"},
    {62, "BINARY_LSHIFT"},
    {63, "BINARY_RSHIFT"},
    {64, "BINARY_AND"},
    {65, "BINARY_XOR"},
    {66, "BINARY_OR"},
    {67, "INPLACE_POWER"},
    {68, "GET_ITER"},
    {69, "GET_YIELD_FROM_ITER"},
    {70, "PRINT_EXPR"},
    {71, "LOAD_BUILD_CLASS"},
    {72, "YIELD_FROM"},
    {73, "GET_AWAITABLE"},
    {75, "INPLACE_LSHIFT"},
    {76, "INPLACE_RSHIFT"},
    {77, "INPLACE_AND"},
    {78, "INPLACE_XOR"},
    {79, "INPLACE_OR"},
    {80, "BREAK_LOOP"},
    {81, "WITH_CLEANUP_START"},
    {82, "WITH_CLEANUP_FINISH"},
    {83, "RETURN_VALUE"},
    {84, "IMPORT_STAR"},
    {85, "SETUP_ANNOTATIONS"},
    {86, "YIELD_VALUE"},
    {87, "POP_BLOCK"},
    {88, "END_FINALLY"},
    {89, "POP_EXCEPT"},
    {90, "STORE_NAME"},
    {91, "DELETE_NAME"},
    {92, "UNPACK_SEQUENCE"},
    {93, "FOR_ITER"},
    {94, "UNPACK_EX"},
    {95, "STORE_ATTR"},
    {96, "DELETE_ATTR"},
    {97, "STORE_GLOBAL"},
    {98, "DELETE_GLOBAL"},
    {100, "LOAD_CONST"},
    {101, "LOAD_NAME"},
    {102, "BUILD_TUPLE"},
    {103, "BUILD_LIST"},
    {104, "BUILD_SET"},
    {105, "BUILD_MAP"},
    {106, "LOAD_ATTR"},
    {107, "COMPARE_OP"},
    {108, "IMPORT_NAME"},
    {109, "IMPORT_FROM"},
    {110, "JUMP_FORWARD"},
    {111, "JUMP_IF_FALSE_OR_POP"},
    {112, "JUMP_IF_TRUE_OR_POP"},
    {113, "JUMP_ABSOLUTE"},
    {114, "POP_JUMP_IF_FALSE"},
    {115, "POP_JUMP_IF_TRUE"},
    {116, "LOAD_GLOBAL"},
    {119, "CONTINUE_LOOP"},
    {120, "SETUP_LOOP"},
    {121, "SETUP_EXCEPT"},
    {122, "SETUP_FINALLY"},
    {124, "LOAD_FAST"},
    {125, "STORE_FAST"},
    {126, "DELETE_FAST"},
    {127, "STORE_ANNOTATION"},
    {130, "RAISE_VARARGS"},
    {131, "CALL_FUNCTION"},
    {132, "MAKE_FUNCTION"},
    {133, "BUILD_SLICE"},
    {135, "LOAD_CLOSURE"},
    {136, "LOAD_DEREF"},
    {137, "STORE_DEREF"},
    {138, "DELETE_DEREF"},
    {141, "CALL_FUNCTION_KW"},
    {142, "CALL_FUNCTION_EX"},
    {143, "SETUP_WITH"},
    {144, "EXTENDED_ARG"},
    {145, "LIST_APPEND"},
    {146, "SET_ADD"},
    {147, "MAP_ADD"},
    {148, "LOAD_CLASSDEREF"},
    {149, "BUILD_LIST_UNPACK"},
    {150, "BUILD_MAP_UNPACK"},
    {151, "BUILD_MAP_UNPACK_WITH_CALL"},
    {152, "BUILD_TUPLE_UNPACK"},
    {153, "BUILD_SET_UNPACK"},
    {154, "SETUP_ASYNC_WITH"},
    {155, "FORMAT_VALUE"},
    {156, "BUILD_CONST_KEY_MAP"},
    {157, "BUILD_STRING"},
    {158, "BUILD_TUPLE_UNPACK_WITH_CALL"},
    {160, "LOAD_METHOD"},
    {161, "CALL_METHOD"},
    {257, "EXCEPT_HANDLER"}};
}  // namespace

const absl::string_view BytecodeOpcodeToString(int64_t opcode) {
  auto it = std::lower_bound(std::begin(kOpcodeNames), std::end(kOpcodeNames),
                             std::make_pair(opcode, absl::string_view()));
  S6_CHECK(it != std::end(kOpcodeNames));
  S6_CHECK_EQ(it->first, opcode);
  return it->second;
}

std::string Location::ToString() const {
  if (kind_ == kUndefined) {
    return "undef";
  }
  if (kind_ == kImmediate) {
    return "imm";
  }
  if (kind_ == kFrameSlot) {
    return absl::StrCat("stack[", index_, "]");
  }
  if (kind_ == kCallStackSlot) {
    return absl::StrCat("callstack[", index_, "]");
  }
  asmjit::String s;
  asmjit::Logging::formatRegister(s, /*flags=*/0, /*emitter=*/nullptr,
                                  asmjit::ArchInfo::kIdX64, reg_->type(),
                                  reg_->id());
  return std::string(s.data(), s.size());
}

// If `constant` is an interned PyUnicodeObject, returns its data. Otherwise
// returns the empty string.
absl::string_view GetObjectAsCheapString(PyObject* constant) {
  if (!constant) {
    return {};
  }
  if (!PyUnicode_CHECK_INTERNED(constant)) {
    return {};
  }

  Py_ssize_t size;
  const char* data = PyUnicode_AsUTF8AndSize(constant, &size);
  if (!data) {
    return {};
  }
  return absl::string_view(data, size);
}

absl::string_view GetObjectAsCheapStringRequiringGil(PyObject* constant) {
  if (!constant) {
    return {};
  }

  Py_ssize_t size;
  const char* data = PyUnicode_AsUTF8AndSize(constant, &size);
  if (!data) {
    return {};
  }
  return absl::string_view(data, size);
}
}  // namespace deepmind::s6
