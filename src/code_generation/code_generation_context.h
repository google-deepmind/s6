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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATION_CONTEXT_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATION_CONTEXT_H_

#include <cstdint>

#include "code_generation/jit_stub.h"
#include "runtime/stack_frame.h"

namespace deepmind::s6 {

// Holds all state used during code generation.
class CodeGenerationContext {
 public:
  CodeGenerationContext(
      asmjit::x86::Emitter& emitter, PyCodeObject* py_code_object,
      void* s6_code_object_storage,
      const RegisterAllocation& register_allocation,
      JitAllocator& jit_allocator, int64_t num_spill_slots,
      absl::flat_hash_set<const Instruction*> optimizable_conditions,
      const Function& function);

  // Returns the emitter in use for code generation.
  asmjit::x86::Emitter& emitter() { return emitter_; }

  // Returns the RegisterAllocation.
  const RegisterAllocation& register_allocation() const {
    return register_allocation_;
  }

  // Returns the JitAllocator.
  JitAllocator& jit_allocator() const { return jit_allocator_; }

  // Returns a JitStub for the StackFrame object. This is statically sized, so
  // is known throughout code generation.
  JitStub<StackFrame> stack_frame() const;

  // Returns a JitStub for the PyFrameObject.
  JitStub<PyFrameObject> pyframe() const;

  // Returns true if code is being emitted for a generator function.
  bool is_generator() const { return is_generator_; }

  // Returns the scratch register, which is never register-allocated and is
  // always available for code generation to use.
  asmjit::x86::Gp scratch_reg() const { return asmjit::x86::r11; }

  // Returns the dedicated register that holds the profile counter.
  asmjit::x86::Gp profile_counter_reg() const { return asmjit::x86::r12; }

  // Returns the dedicated register that holds the PyFrameObject.
  asmjit::x86::Gp pyframe_reg() const { return asmjit::x86::rbx; }

  // Returns the dedicated register that holds the spill slots for generator
  // functions.
  // REQUIRES: is_generator()
  asmjit::x86::Gp generator_spill_slot_reg() const {
    S6_CHECK(is_generator_);
    return asmjit::x86::r13;
  }

  // Returns the Memory operand of the given spill slot.
  asmjit::x86::Mem spill_slot(int64_t index) const;

  // Returns the Memory operand of the given call stack slot. Note that this
  // implicitly extends the number of call stack slots and is safe to call
  // before StackFrameLayoutIsFinalized().
  asmjit::x86::Mem call_stack_slot(int64_t index);

  // Returns the destination operand for `value` as a register.
  absl::StatusOr<asmjit::x86::Gp> Destination(const Value& value);

  // Returns the given Location as an asmjit::Operand.
  asmjit::Operand Operand(const Location& loc);

  // Returns `value` as an operand to the currently being visited instruction.
  // This could be an immediate, register or stack value.
  //
  // If `instruction` is non-nullptr, the query is performed on entry to
  // `instruction` rather than the current instruction.
  asmjit::Operand Operand(const Value* value,
                          const Instruction* instruction = nullptr);

  // Returns `value` as an operand to the currently being visited instruction.
  // If the operand is not in a register already, a scratch register is used.
  //
  // If `instruction` is non-nullptr, the query is performed on entry to
  // `instruction` rather than the current instruction.
  asmjit::x86::Gp OperandInRegister(const Value* value,
                                    const Instruction* instruction = nullptr);

  // Finalizes and sets the stack frame layout by declaring the number of spill
  // slots, callee-saved registers and call stack slots.
  void FinalizeStackFrameLayout(
      std::vector<asmjit::x86::Gp> callee_saved_registers,
      int64_t num_call_stack_slots);

  // Queries if the stack frame layout has been finalized and thus is valid.
  bool StackFrameLayoutIsFinalized() const {
    return stack_frame_layout_is_finalized_;
  }

  // Returns the list of callee-saved registers in use in this function.
  // REQUIRES: StackFrameLayoutIsFinalized()
  const std::vector<asmjit::x86::Gp>& callee_saved_registers() const {
    S6_CHECK(StackFrameLayoutIsFinalized());
    return callee_saved_registers_;
  }

  // Returns a JitStub for the StackFrameWithLayout object. This is dynamically
  // sized, so requires the stack frame layout to be finalized.
  // REQUIRES: StackFrameLayoutIsFinalized().
  JitStub<StackFrameWithLayout> stack_frame_with_layout() const {
    S6_CHECK(StackFrameLayoutIsFinalized());
    return JitStub<StackFrameWithLayout>(stack_frame().Mem(),
                                         stack_frame_layout_);
  }

  // Returns the actual StackFrameLayout object.
  // REQUIRES: StackFrameLayoutIsFinalized();
  const StackFrameLayout& stack_frame_layout() {
    S6_CHECK(StackFrameLayoutIsFinalized());
    return stack_frame_layout_;
  }

  // Returns the s6::CodeObject that is being generated.
  // This object is not constructed, it is just a block of plain memory of the
  // right size in which the CodeObject will be constructed
  void* s6_code_object_storage() const { return s6_code_object_storage_; }
  // Returns the PyCodeObject that is being generated.
  PyCodeObject* py_code_object() const { return py_code_object_; }
  // Returns the number of fastlocals. This includes free and cell variables.
  int64_t num_fastlocals() const { return num_fastlocals_; }
  // Returns the number of arguments expected by the function.
  int64_t num_expected_arguments() const { return num_expected_arguments_; }
  // Returns true if the function has free or cell variables.
  bool has_free_or_cell_vars() const { return has_free_or_cell_vars_; }

  // Binds the PyFrame ABI entry point to the current emitter cursor.
  void BindPyFrameEntryPoint();

  // Binds the Fast ABI entry point to the current emitter cursor.
  void BindFastEntryPoint();

  // Binds the label for stack cleanup to the current emitter cursor.
  void BindCleanupPoint();

  // Binds the label for the frame epilog to the current emitter cursor.
  void BindEpilogPoint();

  // Returns the entry points.
  asmjit::Label pyframe_entry_point() const { return pyframe_entry_point_; }
  asmjit::Label fast_entry_point() const { return fast_entry_point_; }

  // Returns the label to jump to to clean up the stack frame. This falls
  // through to epilog_point().
  asmjit::Label cleanup_point() const { return cleanup_point_; }

  // Returns the label to jump to to run the raw epilog without cleaning any
  // Python objects up.
  asmjit::Label epilog_point() const { return epilog_point_; }

  // Returns the label for the start of the given block.
  asmjit::Label block_label(const Block* block) const {
    return block_labels_.at(block);
  }

  // Returns true if `condition` is within `optimizable_conditions` given to
  // the constructor.
  bool ConditionIsOptimizable(const Instruction* condition) const {
    return optimizable_conditions_.contains(condition);
  }

  // Sets the currently being visited instruction. This affects the Operand*
  // queries.
  void SetCurrentInstruction(const Instruction* instruction);

  // Binds `block`'s label to the current cursor.
  void BindBlock(const Block* block);

  // Returns the list of deoptimization labels.
  std::vector<std::pair<const Instruction*, asmjit::Label>>& deopt_labels() {
    return deopt_labels_;
  }

  const Function& function() const { return function_; }

 private:
  asmjit::x86::Emitter& emitter_;
  PyCodeObject* py_code_object_;
  void* s6_code_object_storage_;
  const RegisterAllocation& register_allocation_;
  JitAllocator& jit_allocator_;
  const Function& function_;
  StackFrameLayout stack_frame_layout_;
  int64_t num_fastlocals_;
  int64_t num_expected_arguments_;
  bool has_free_or_cell_vars_;
  bool is_generator_;
  bool stack_frame_layout_is_finalized_;
  std::vector<asmjit::x86::Gp> callee_saved_registers_;
  asmjit::Label pyframe_entry_point_;
  asmjit::Label fast_entry_point_;
  asmjit::Label cleanup_point_;
  asmjit::Label epilog_point_;
  int64_t num_spill_slots_;
  int64_t num_requested_call_stack_slots_ = 0;
  const Instruction* current_instruction_;
  absl::flat_hash_set<const Instruction*> optimizable_conditions_;
  absl::flat_hash_map<const Block*, asmjit::Label> block_labels_;
  std::vector<std::pair<const Instruction*, asmjit::Label>> deopt_labels_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATION_CONTEXT_H_
