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

#include "code_generation/code_generation_context.h"

#include <cstdint>

#include "asmjit/x86/x86operand.h"
#include "code_generation/register_allocator.h"

namespace deepmind::s6 {
namespace x86 = ::asmjit::x86;

CodeGenerationContext::CodeGenerationContext(
    asmjit::x86::Emitter& emitter, PyCodeObject* py_code_object,
    void* s6_code_object_storage, const RegisterAllocation& register_allocation,
    JitAllocator& jit_allocator, int64_t num_spill_slots,
    absl::flat_hash_set<const Instruction*> optimizable_conditions,
    const Function& function)
    : emitter_(emitter),
      py_code_object_(py_code_object),
      s6_code_object_storage_(s6_code_object_storage),
      register_allocation_(register_allocation),
      jit_allocator_(jit_allocator),
      function_(function),
      num_spill_slots_(num_spill_slots),
      optimizable_conditions_(std::move(optimizable_conditions)) {
  num_fastlocals_ = py_code_object->co_nlocals +
                    PyTuple_GET_SIZE(py_code_object->co_cellvars) +
                    PyTuple_GET_SIZE(py_code_object->co_freevars);
  num_expected_arguments_ = py_code_object->co_argcount;
  if (py_code_object->co_flags & CO_VARARGS) ++num_expected_arguments_;
  if (py_code_object->co_flags & CO_VARKEYWORDS) ++num_expected_arguments_;
  has_free_or_cell_vars_ = num_fastlocals_ != py_code_object->co_nlocals;
  is_generator_ = ((py_code_object->co_flags & CO_GENERATOR) != 0);
  stack_frame_layout_is_finalized_ = false;
  cleanup_point_ = emitter_.newLabel();
  epilog_point_ = emitter_.newLabel();

  for (const Block& block : function) {
    block_labels_[&block] = emitter_.newLabel();
  }
}

JitStub<StackFrame> CodeGenerationContext::stack_frame() const {
  // The StackFrame object is always located immediately below the base pointer
  // such that &stack_frame + sizeof(StackFrame) == base_pointer.
  //
  // This is relied upon by StackFrame::GetFromBasePointer().
  return JitStub<StackFrame>(
      x86::ptr(x86::rbp, -static_cast<int32_t>(sizeof(StackFrame))));
}

JitStub<PyFrameObject> CodeGenerationContext::pyframe() const {
  return JitStub<PyFrameObject>(x86::ptr(pyframe_reg()));
}

void CodeGenerationContext::FinalizeStackFrameLayout(
    std::vector<asmjit::x86::Gp> callee_saved_registers,
    int64_t num_call_stack_slots) {
  callee_saved_registers.push_back(x86::r12);
  callee_saved_registers.push_back(x86::rbx);
  if (is_generator()) {
    callee_saved_registers.push_back(generator_spill_slot_reg());
  }

  // The stack frame is layed out as:
  //   Prior RBP          <-- high addresses, RBP
  //   StackFrame             Fixed size, sizeof(StackFrame)
  //   spill_slot[n-1]
  //   ..
  //   spill_slot[0]
  //   ... padding? ...
  //   callee_saved_register[n-1]
  //   ...
  //   callee_saved_register[0]
  //   call_stack_slot[n-1]
  //   ...
  //   call_stack_slot[0] <-- low addresses, RSP

  num_call_stack_slots =
      std::max<int64_t>(num_call_stack_slots, num_requested_call_stack_slots_);
  // The stack pointer must always be aligned to a multiple of 16 bytes, which
  // is why we have an optional extra padding. The padding appears in the middle
  // because the call stack must always start at RSP+0 for ABI reasons.
  //
  // Note all offsets are values to *add* from RBP.
  int64_t spill_slots_offset =
      -sizeof(StackFrame) - num_spill_slots_ * sizeof(void*);
  int64_t stack_size_in_bytes_before_padding =
      sizeof(void*) * (num_spill_slots_ + num_call_stack_slots +
                       callee_saved_registers.size()) +
      sizeof(StackFrame);
  int64_t padding = (stack_size_in_bytes_before_padding % 16 == 0) ? 0 : 8;
  int64_t callee_saved_registers_offset =
      spill_slots_offset - padding -
      callee_saved_registers.size() * sizeof(void*);
  int64_t rsp_offset =
      callee_saved_registers_offset - num_call_stack_slots * sizeof(void*);

  stack_frame_layout_ = StackFrameLayout(
      spill_slots_offset, num_spill_slots_, callee_saved_registers_offset,
      callee_saved_registers.size(), rsp_offset);
  callee_saved_registers_ = callee_saved_registers;
  stack_frame_layout_is_finalized_ = true;
}

void CodeGenerationContext::BindPyFrameEntryPoint() {
  pyframe_entry_point_ = emitter_.newLabel();
  emitter_.bind(pyframe_entry_point_);
}

void CodeGenerationContext::BindFastEntryPoint() {
  fast_entry_point_ = emitter_.newLabel();
  emitter_.bind(fast_entry_point_);
}

void CodeGenerationContext::BindCleanupPoint() {
  emitter_.bind(cleanup_point_);
}

void CodeGenerationContext::BindEpilogPoint() { emitter_.bind(epilog_point_); }

x86::Mem CodeGenerationContext::spill_slot(int64_t index) const {
  if (is_generator()) {
    return x86::qword_ptr(generator_spill_slot_reg(), index * sizeof(void*));
  }
  // Offset to the start of spill slots is rbp - StackFrame - spill slots.
  int64_t spill_slots_offset =
      -sizeof(StackFrame) - num_spill_slots_ * sizeof(void*);
  return x86::qword_ptr(x86::rbp, spill_slots_offset + index * sizeof(void*));
}

x86::Mem CodeGenerationContext::call_stack_slot(int64_t index) {
  num_requested_call_stack_slots_ =
      std::max<int64_t>(num_requested_call_stack_slots_, index + 1);
  return x86::qword_ptr(x86::rsp, index * sizeof(void*));
}

void CodeGenerationContext::SetCurrentInstruction(
    const Instruction* instruction) {
  current_instruction_ = instruction;
}

absl::StatusOr<x86::Gp> CodeGenerationContext::Destination(const Value& value) {
  Location loc = register_allocation_.DestinationLocation(value);
  if (!loc.IsInRegister())
    return absl::UnimplementedError(
        "destination operands in memory not handled!");
  return loc.Register().as<x86::Gp>();
}

asmjit::Operand CodeGenerationContext::Operand(const Location& loc) {
  S6_CHECK(loc.IsDefined());
  if (loc.IsImmediate()) return asmjit::Imm(loc.ImmediateValue());
  if (loc.IsInRegister()) return loc.Register();
  if (loc.IsFrameSlot()) {
    return spill_slot(loc.FrameSlot());
  }
  S6_CHECK(loc.IsCallStackSlot()) << loc.ToString();
  return call_stack_slot(loc.CallStackSlot());
}

asmjit::Operand CodeGenerationContext::Operand(const Value* value,
                                               const Instruction* instruction) {
  if (!instruction) {
    instruction = current_instruction_;
  }
  return Operand(register_allocation_.OperandLocation(*instruction, *value));
}

x86::Gp CodeGenerationContext::OperandInRegister(
    const Value* value, const Instruction* instruction) {
  asmjit::Operand op = Operand(value, instruction);
  if (op.isReg()) return op.as<x86::Gp>();
  x86::Gp gp = scratch_reg();
  emitter_.emit(x86::Inst::kIdMov, gp, op);
  return gp;
}

void CodeGenerationContext::BindBlock(const Block* block) {
  emitter_.bind(block_labels_[block]);
}

}  // namespace deepmind::s6
