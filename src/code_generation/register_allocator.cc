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

#include "code_generation/register_allocator.h"

#include <array>
#include <cstdint>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "asmjit/asmjit.h"
#include "code_generation/live_interval.h"
#include "core_util.h"
#include "strongjit/instruction.h"
#include "strongjit/instructions.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"

namespace deepmind::s6 {
Location GetAbiLocation(int64_t operand_index) {
  static const std::array<asmjit::x86::Reg, 6> kAbiRegisters = {
      {asmjit::x86::rdi, asmjit::x86::rsi, asmjit::x86::rdx, asmjit::x86::rcx,
       asmjit::x86::r8, asmjit::x86::r9}};
  if (operand_index >= kAbiRegisters.size()) {
    return Location::CallStackSlot(operand_index - kAbiRegisters.size());
  }
  return Location::Register(kAbiRegisters[operand_index]);
}

// Returns the ABI register requirement for operand `operand_index` of `v`.
LocationRequirement GetRegisterRequirement(const Instruction* inst,
                                           int64_t operand_index) {
  if (isa<CallNativeInst>(inst) || isa<DecrefInst>(inst)) {
    return LocationRequirement(GetAbiLocation(operand_index));
  }
  if (isa<CallNativeIndirectInst>(inst)) {
    if (operand_index == 0)
      // Make sure the compiler doesn't select one of the callee operand
      // registers for the callee by forcing it to be `rax`.
      return LocationRequirement(Location::Register(asmjit::x86::rax));
    return LocationRequirement(GetAbiLocation(operand_index - 1));
  }
  if (const CallPythonInst* c = dyn_cast<CallPythonInst>(inst)) {
    if (isa<ConstantAttributeInst>(c->callee()) && c->fastcall()) {
      // A CallPythonInst's hot path uses variadic arguments. On the hot path,
      // the PyFunctionObject target comes first, then all call arguments.
      int64_t call_argument_index = c->GetCallArgumentIndex(operand_index);
      if (call_argument_index < 0) return LocationRequirement::InRegister();
      return LocationRequirement(GetAbiLocation(call_argument_index + 1));
    }
    // Otherwise this is the slow path of CallPythonInst where we will call a
    // runtime function.
    // Note that this is also the case for CallAttributeInst.
    if (operand_index == CallPythonInst::kNamesOperandIndex)
      return LocationRequirement(GetAbiLocation(2));

    // Pass all call arguments on the stack sequentially.
    if (operand_index == CallPythonInst::kCalleeOperandIndex) {
      return LocationRequirement(Location::CallStackSlot(0));
    }
    return LocationRequirement(Location::CallStackSlot(operand_index - 1));
  }
  if (const CallVectorcallInst* c = dyn_cast<CallVectorcallInst>(inst)) {
    // The vectorcall ABI is: (self, args, nargs, names)
    if (operand_index == CallVectorcallInst::kCalleeOperandIndex)
      // Put the callee in RAX, so it doesn't accidentally get allocated to RSI
      // which would clobber `args`.
      return LocationRequirement(Location::Register(asmjit::x86::rax));
    if (operand_index == CallVectorcallInst::kSelfOperandIndex)
      return LocationRequirement(GetAbiLocation(0));
    if (operand_index == CallVectorcallInst::kNamesOperandIndex)
      return LocationRequirement(GetAbiLocation(3));

    // Pass all call arguments on the stack sequentially.
    int64_t call_argument_index = c->GetCallArgumentIndex(operand_index);
    return LocationRequirement(Location::CallStackSlot(call_argument_index));
  }
  if (isa<BoxInst>(inst) || isa<UnboxInst>(inst)) {
    // The sole operand will be the argument to a function call.
    return LocationRequirement(GetAbiLocation(operand_index));
  }
  if (isa<BytecodeBeginInst>(inst) || isa<RematerializeInst>(inst)) {
    // Required values do not require a register; hint it to be in memory.
    return LocationRequirement::Anywhere();
  }
  if (const DeoptimizeIfInst* di = dyn_cast<DeoptimizeIfInst>(inst)) {
    auto first_required_value_operand = di->required_values().data();
    if (&di->operands()[operand_index] >= first_required_value_operand) {
      // Required values do not require a register; hint it to be in memory.
      return LocationRequirement::Anywhere();
    }
  }
  if (isa<DeoptimizeIfSafepointInst>(inst)) {
    if (operand_index > 0) {
      // Required values do not require a register; hint it to be in memory.
      return LocationRequirement::Anywhere();
    }
  }
  if (const YieldValueInst* yi = dyn_cast<YieldValueInst>(inst)) {
    if (operand_index >= 1) {
      // Value stack items do not require a register; hint it to be in memory.
      return LocationRequirement::InFrameSlot();
    }
  }
  if (isa<JmpInst>(inst)) {
    return LocationRequirement::Anywhere();
  }
  if (isa<BrInst>(inst) && operand_index > 0) {
    return LocationRequirement::Anywhere();
  }
  if (isa<ExceptInst>(inst)) {
    // Arguments to excepts must use slots 6+ on the call stack (0-5 are filled
    // in by the exception handler).
    //
    // Note that in ExceptInst, operand 0 is the successor.
    return LocationRequirement::InLocation(
        Location::CallStackSlot(6 + operand_index - 1));
  }
  return LocationRequirement::InRegister();
}

// Returns the ABI register requirement for `v`.
LocationRequirement GetRegisterRequirement(const Value* v) {
  if (isa<CallNativeInst>(v) || isa<CallNativeIndirectInst>(v) ||
      (!isa<CallAttributeInst>(v) && isa<CallPythonInst>(v)) ||
      isa<YieldValueInst>(v) || isa<CallVectorcallInst>(v)) {
    // All calls take their return in rax.
    return LocationRequirement::InLocation(
        Location::Register(asmjit::x86::rax));
  }
  if (const DivideInst* di = dyn_cast<DivideInst>(v)) {
    if (di->type() == NumericInst::kInt64) {
      // The idiv instruction always returns the quotient in rax.
      return LocationRequirement::InLocation(
          Location::Register(asmjit::x86::rax));
    }
  }
  if (const RemainderInst* di = dyn_cast<RemainderInst>(v)) {
    if (di->type() == NumericInst::kInt64) {
      // The idiv instruction always returns the remainder in rdx.
      return LocationRequirement::InLocation(
          Location::Register(asmjit::x86::rdx));
    }
  }
  if (isa<RematerializeInst>(v)) {
    // Rematerialize doesn't need a register.
    return LocationRequirement::InLocation(Location::Immediate(-1));
  }
  return LocationRequirement::InRegister();
}

RegisterAllocation::~RegisterAllocation() {}

}  // namespace deepmind::s6
