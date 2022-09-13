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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CALL_STUB_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CALL_STUB_H_

#include <array>
#include <cstdint>

#include "asmjit/asmjit.h"
#include "runtime/runtime.h"

namespace deepmind::s6 {

// A CallStub is parameterized by either being for a Caller or a Callee. For
// register arguments this makes no difference, but for arguments passed on the
// stack, a callee picks them up from the caller's stack frame (thus they are
// referenced with respect to RBP), whereas the caller references them with
// respect to RSP.
enum class CallStubType { kCaller, kCallee };

static constexpr std::array<asmjit::x86::Gp, 6> kAbiRegisters = {
    asmjit::x86::rdi, asmjit::x86::rsi, asmjit::x86::rdx,
    asmjit::x86::rcx, asmjit::x86::r8,  asmjit::x86::r9};

// Returns the ABI-defined location for argument `Index`. This will either
// return an x86::Gp or an x86::Mem. The return type is dependent on the Index
// argument. This allows the result to be passed directly to an asmjit emitter.
//
// Stack arguments are addressed relative to the callee or caller frame
// depending on Type.
template <int64_t Index, CallStubType Type = CallStubType::kCaller>
constexpr auto AbiLocation() {
  if constexpr (Index < kAbiRegisters.size()) {
    return kAbiRegisters[Index];
  } else {
    int64_t slot_index = Index - kAbiRegisters.size();
    // If this is a kCaller slot type, stack slots are indexed with respect to
    // RSP. [RSP + 0] is the first stack slot.
    if (Type == CallStubType::kCaller) {
      return asmjit::x86::qword_ptr(asmjit::x86::rsp,
                                    slot_index * sizeof(void*));
    } else {
      // Otherwise this is a kCallee slot type, so stack slots are indexed with
      // respect to the base pointer RBP.
      //  [rbp + 0] = previous rbp
      //  [rbp + 8] = caller address
      //  [rbp + 16] = first caller stack slot
      return asmjit::x86::qword_ptr(asmjit::x86::rbp,
                                    16 + (slot_index * sizeof(void*)));
    }
  }
}

// Returns the ABI-defined location for argument `Index`. This variant takes
// `index` as an argument rather than a template argument, so returns an
// asmjit::Operand.
constexpr asmjit::Operand AbiLocation(
    int64_t index, CallStubType type = CallStubType::kCaller) {
  if (index < kAbiRegisters.size()) {
    return kAbiRegisters[index];
  } else {
    int64_t slot_index = index - kAbiRegisters.size();
    if (type == CallStubType::kCaller) {
      return asmjit::x86::qword_ptr(asmjit::x86::rsp,
                                    slot_index * sizeof(void*));
    } else {
      return asmjit::x86::qword_ptr(asmjit::x86::rbp,
                                    16 + (slot_index * sizeof(void*)));
    }
  }
}

// A CallStub describes the argument locations for a function call. It allows
// symbolic references to arguments instead of naming the physical register
// locations or numeric argument indices.
//
// Use by specializing over the intended runtime call:
//   CallStub<MyRuntimeFunction> call;
//   e.mov(call.my_arg(), asmjit::imm(0));
//   e.call(call.imm());
template <auto F>
struct CallStub {
  // Only specializations are useful. The primary template is useless.
  CallStub() = delete;
};

// Specializations per runtime function.
template <>
struct CallStub<CleanupStackFrame> {
  constexpr static auto stack_frame() { return AbiLocation<0>(); }

  static asmjit::Imm imm() { return asmjit::imm(CleanupStackFrame); }
};

template <>
struct CallStub<SetUpStackFrameForGenerator> {
  constexpr static auto pyframe() { return AbiLocation<0>(); }
  constexpr static auto stack_frame() { return AbiLocation<1>(); }
  constexpr static auto code_object() { return AbiLocation<2>(); }
  constexpr static auto num_spill_slots() { return AbiLocation<3>(); }

  static asmjit::Imm imm() { return asmjit::imm(SetUpStackFrameForGenerator); }
};

template <>
struct CallStub<SetUpStackFrame> {
  constexpr static auto pyframe() { return AbiLocation<0>(); }
  constexpr static auto stack_frame() { return AbiLocation<1>(); }
  constexpr static auto code_object() { return AbiLocation<2>(); }

  static asmjit::Imm imm() { return asmjit::imm(SetUpStackFrame); }
};

template <>
struct CallStub<SetupFreeVars> {
  constexpr static auto func() { return AbiLocation<0>(); }
  constexpr static auto pyframe() { return AbiLocation<1>(); }

  static asmjit::Imm imm() { return asmjit::imm(SetupFreeVars); }
};

template <>
struct CallStub<PyObject_GetAttr> {
  constexpr static auto object() { return AbiLocation<0>(); }
  constexpr static auto name() { return AbiLocation<1>(); }

  static asmjit::Imm imm() { return asmjit::imm(PyObject_GetAttr); }
};

template <>
struct CallStub<PyObject_SetAttr> {
  constexpr static auto object() { return AbiLocation<0>(); }
  constexpr static auto name() { return AbiLocation<1>(); }
  constexpr static auto value() { return AbiLocation<2>(); }

  static asmjit::Imm imm() { return asmjit::imm(PyObject_SetAttr); }
};

template <>
struct CallStub<SetUpForYieldValue> {
  constexpr static auto result() { return AbiLocation<0>(); }
  constexpr static auto stack_frame() { return AbiLocation<1>(); }
  constexpr static auto yi() { return AbiLocation<2>(); }

  static asmjit::Imm imm() { return asmjit::imm(SetUpForYieldValue); }
};

template <>
struct CallStub<CallAttribute> {
  constexpr static auto arg_count() { return AbiLocation<0>(); }
  constexpr static auto args() { return AbiLocation<1>(); }
  constexpr static auto names() { return AbiLocation<2>(); }
  constexpr static auto stack_frame() { return AbiLocation<3>(); }
  constexpr static auto attr_str() { return AbiLocation<4>(); }
  constexpr static auto call_python_bytecode_offset() {
    return AbiLocation<5>();
  }

  static asmjit::Imm imm() { return asmjit::imm(CallAttribute); }
};

template <>
struct CallStub<Except> {
  constexpr static auto bytecode_offset() { return AbiLocation<0>(); }
  constexpr static auto args() { return AbiLocation<1>(); }

  static asmjit::Imm imm() { return asmjit::imm(Except); }
};

// Similar to CallStub but defines the arguments expected by a *callee* using
// the PyFrame ABI.
struct PyFrameCalleeStub {
  constexpr static auto pyframe() {
    return AbiLocation<0, CallStubType::kCallee>();
  }
  constexpr static auto profile_counter() {
    return AbiLocation<1, CallStubType::kCallee>();
  }
  constexpr static auto code_object() {
    return AbiLocation<2, CallStubType::kCallee>();
  }
};

// Similar to CallStub but defines the arguments expected by a *callee* using
// the Fast ABI.
struct FastCalleeStub {
  constexpr static auto py_function_object() {
    return AbiLocation<0, CallStubType::kCallee>();
  }

  constexpr static asmjit::Operand argument(int64_t i) {
    return AbiLocation(i + 1, CallStubType::kCallee);
  }
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CALL_STUB_H_
