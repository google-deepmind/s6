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

#include "code_generation/prolog_epilog_insertion.h"

#include <cstdint>

#include "absl/status/status.h"
#include "code_generation/asmjit_util.h"
#include "code_generation/call_stub.h"
#include "code_generation/jit_stub.h"
#include "code_generation/register_allocator.h"
#include "metadata.h"
#include "runtime/generator.h"
#include "runtime/pyframe_object_cache.h"
#include "runtime/runtime.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace x86 = ::asmjit::x86;

namespace {
// Emits the very first instructions in a function:
//   * Saves RBP
//   * Reserves a region for the local stack frame.
//   * Saves all callee-saved registers.
void GenerateInitialSetup(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();
  e.push(x86::rbp);
  e.mov(x86::rbp, x86::rsp);
  // The subtraction amount is exactly the same as the return address offset.
  e.sub(x86::rsp, -ctx.stack_frame_layout().return_address_offset());

  JitStub<StackFrameWithLayout> stack_frame_stub =
      ctx.stack_frame_with_layout();
  int64_t index = 0;
  for (const x86::Gp& gp : ctx.callee_saved_registers()) {
    stack_frame_stub.callee_saved_register_slot(index++).Store(gp, e);
  }
}

// Emits code that copies all function arguments from f_localsplus to wherever
// the register allocator wants them. f_localsplus is zeroed.
void SetUpFunctionArguments(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();

  int64_t i = 0;
  for (const BlockArgument* arg : ctx.function().entry().block_arguments()) {
    x86::Mem fastlocal = ctx.pyframe().fastlocals(i++).Mem();
    const Location& loc = ctx.register_allocation().DestinationLocation(*arg);
    asmjit::Operand operand = ctx.Operand(loc);

    if (operand.isReg()) {
      x86::Gp gp = operand.as<x86::Gp>();
      e.mov(gp, fastlocal);
      e.mov(fastlocal, asmjit::imm(0));
    } else {
      x86::Gp scratch = ctx.scratch_reg();
      e.mov(scratch, fastlocal);
      e.mov(fastlocal, asmjit::imm(0));
      e.mov(operand.as<x86::Mem>(), scratch);
    }
  }
}

// Emits function prolog code for the ABI that accepts a PyFrameObject for a
// function that is a generator.
absl::Status GenerateGeneratorAbiProlog(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();

  if (ctx.stack_frame_layout().spill_slots_count() >
      PyGeneratorFrameObjectCache::kNumSpillSlots) {
    return absl::ResourceExhaustedError(
        "Generator function uses more spill slots than are available in the "
        "generator frame cache.");
  }

  PyFrameCalleeStub incoming;
  CallStub<SetUpStackFrameForGenerator> call;

  e.mov(ctx.profile_counter_reg(), incoming.profile_counter());
  e.mov(ctx.pyframe_reg(), incoming.pyframe());

  ctx.stack_frame().Load(call.stack_frame(), e);
  e.mov(call.num_spill_slots(),
        asmjit::imm(ctx.stack_frame_layout().spill_slots_count()));
  S6_RET_CHECK_EQ(incoming.pyframe(), call.pyframe());
  e.mov(call.code_object(), asmjit::imm(ctx.s6_code_object_storage()));

  e.call(call.imm());

  // A std::pair is returned in rax:rdx (first element in rax, second in rdx).
  JitStub<GeneratorState> gen_state(x86::ptr(x86::rdx));
  gen_state.spill_slots().Load(ctx.generator_spill_slot_reg(), e);

  // Obtain gen_state->resume_pc() into rdx. This will be zero if this
  // generator is just starting, or an address within a YieldValueInst
  // otherwise.
  gen_state.resume_pc().Load(x86::rdx, e);

  // If the resume PC is nonzero, jump to it.
  e.test(x86::rdx, x86::rdx);
  asmjit::Label carry_on = e.newLabel();
  e.jz(carry_on);
  // Jump to code within YieldValueInst. Rax is the value to yield, rdx is
  // nonzero.

  e.jmp(x86::rdx);
  e.bind(carry_on);

  return absl::OkStatus();
}

// Emits function prolog code for the ABI that accepts a PyFrameObject.
absl::Status GeneratePyFrameAbiProlog(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();

  PyFrameCalleeStub incoming;
  CallStub<SetUpStackFrame> call;

  e.mov(ctx.profile_counter_reg(), incoming.profile_counter());
  e.mov(ctx.pyframe_reg(), incoming.pyframe());

  ctx.stack_frame().Load(call.stack_frame(), e);
  e.mov(call.code_object(), asmjit::imm(ctx.s6_code_object_storage()));
  S6_RET_CHECK_EQ(incoming.pyframe(), call.pyframe());

  e.call(call.imm());

  return absl::OkStatus();
}

// Emits function prolog code for the fast ABI entry point.
absl::Status GenerateFastAbiProlog(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();
  PyFrameObjectCache* cache = GetPyFrameObjectCache();
  int64_t num_entries_required =
      ctx.py_code_object()->co_stacksize + ctx.num_fastlocals();
  if (num_entries_required > PyFrameObjectCache::kNumFrameSlots) {
    return absl::UnimplementedError(
        absl::StrCat("Required ", num_entries_required,
                     " frame entries, but the frame cache only holds ",
                     PyFrameObjectCache::kNumFrames));
  }

  FastCalleeStub incoming;

  // The pyframe_reg register currently holds the previous frame.
  JitStub<PyFrameObject> prev_frame(x86::ptr(ctx.pyframe_reg()));

  // We need two free registers. We use r11 and r10, which should both be free.
  x86::Gp head = x86::r10;
  x86::Gp end = x86::r11;
  e.mov(end, asmjit::imm(cache->GetEnd()));
  e.mov(head, asmjit::imm(cache->GetHeadPointer()));

  // Is the value at the head pointer the same as end?
  asmjit::Label take_fast_path = e.newLabel();
  asmjit::Label done = e.newLabel();
  e.cmp(x86::qword_ptr(head), end);
  e.jne(take_fast_path);

  e.call(asmjit::imm(AllocateFrameOnHeap));
  // AllocateFrameOnHeap returns its result in r11.
  e.mov(x86::rax, x86::r11);
  e.jmp(done);

  e.bind(take_fast_path);
  // Dereference `head` into rax, and increment the pointer.
  e.mov(x86::rax, x86::qword_ptr(head));
  e.add(x86::qword_ptr(head),
        asmjit::imm(sizeof(PyFrameObjectCache::SizedFrameObject)));
  e.bind(done);

  // Now the old PyFrameObject is still in rbx, and the new one is in rax.
  JitStub<PyFrameObject> this_frame(x86::ptr(x86::rax));

  // r10 = prev_frame->f_builtins
  // current_frame->f_builtins = r10
  prev_frame.f_builtins().Load(x86::r10, e);
  this_frame.f_builtins().Store(x86::r10, e);

  // current_frame->f_back = prev_frame
  this_frame.f_back().Store(x86::rbx, e);

  // Now move rax into rbx, where we keep our frame register.
  e.mov(x86::rbx, x86::rax);
  this_frame = prev_frame;

  // stack_frame->pyframe_ = current_frame
  ctx.stack_frame().pyframe().Store(x86::rbx, e);

  // Copy all arguments to where they should be. Arguments that are already in
  // the correct place stay where they are, arguments in stack slots get moved,
  // and arguments anywhere else are an error that the register allocator should
  // have sorted out.

  auto block_arguments = ctx.function().entry().block_arguments();
  S6_CHECK_EQ(block_arguments.size(), ctx.num_expected_arguments());
  for (int64_t i = 0; i < ctx.num_expected_arguments(); ++i) {
    asmjit::Operand src = incoming.argument(i);
    const Location& loc =
        ctx.register_allocation().DestinationLocation(*block_arguments[i]);
    asmjit::Operand dst = ctx.Operand(loc);

    if (ctx.has_free_or_cell_vars()) {
      // SetupFreeVars below requires the original arguments in f_fastlocals. So
      // put them there now, and we'll reload them later.
      dst = this_frame.fastlocals(i).Mem();
    }

    if (src == dst) continue;
    S6_CHECK(ctx.has_free_or_cell_vars() || loc.IsFrameSlot());

    if (src.isReg()) {
      e.mov(dst.as<x86::Mem>(), src.as<x86::Gp>());
    } else {
      e.mov(ctx.scratch_reg(), src.as<x86::Gp>());
      e.mov(dst.as<x86::Mem>(), ctx.scratch_reg());
    }
  }

  // Copy state from the previous StackFrame into this StackFrame. The
  // previous StackFrame can be found relative to the previous RBP.
  e.mov(x86::rax, x86::ptr(x86::rbp));
  JitStub<StackFrame> this_stack_frame = ctx.stack_frame();
  JitStub<StackFrame> prev_stack_frame =
      this_stack_frame.CloneWithNewBase(x86::rax);

  {
    JitStub<PyThreadState> tstate =
        prev_stack_frame.thread_state().Load(ctx.scratch_reg(), e);
    this_stack_frame.thread_state().Store(tstate.Reg(), e);

    // tstate->frame = current_frame
    tstate.frame().Store(ctx.pyframe_reg(), e);
  }  // tstate now out of scope. scratch_reg is free.

  JitStub<PyFunctionObject> py_func_object(
      x86::ptr(incoming.py_function_object()));
  // current_frame->f_globals = func_obj->func_globals
  JitStub<PyObject> globals =
      py_func_object.func_globals().Load(ctx.scratch_reg(), e);
  this_frame.f_globals().Store(globals.Reg(), e);

  // current_frame->f_locals = nullptr
  this_frame.f_locals().Store(asmjit::imm(0), e);

  // current_frame->f_code = code
  e.mov(ctx.scratch_reg(), asmjit::imm(ctx.py_code_object()));
  this_frame.f_code().Store(ctx.scratch_reg(), e);

  // StackFrame::magic_ = StackFrame::kMagic.
  e.mov(ctx.scratch_reg(), asmjit::imm(StackFrame::kMagic));
  this_stack_frame.magic().Store(ctx.scratch_reg(), e);

  // StackFrame::s6_code_object = ...
  e.mov(ctx.scratch_reg(), asmjit::imm(ctx.s6_code_object_storage()));
  this_stack_frame.s6_code_object().Store(ctx.scratch_reg(), e);

  // StackFrame::called_with_fast_calling_convention = true.
  this_stack_frame.called_with_fast_calling_convention().Store(asmjit::imm(1),
                                                               e);

  if (ctx.has_free_or_cell_vars()) {
    CallStub<SetupFreeVars> call;
    e.mov(call.pyframe(), ctx.pyframe_reg());
    S6_RET_CHECK_EQ(incoming.py_function_object(), call.func());
    e.call(call.imm());
    SetUpFunctionArguments(ctx);
  }

  return absl::OkStatus();
}
}  // namespace

absl::Status GenerateProlog(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();
  // We can't currently generate a fast ABI entry point for generator functions.
  bool has_fast_abi = !ctx.is_generator();

  ctx.BindPyFrameEntryPoint();

  asmjit::Label after_prolog = e.newLabel();
  GenerateInitialSetup(ctx);
  if (ctx.is_generator()) {
    S6_RETURN_IF_ERROR(GenerateGeneratorAbiProlog(ctx));
    SetUpFunctionArguments(ctx);
    // No jmp here because we don't emit a fast ABI so we can fall through.
    S6_CHECK(!has_fast_abi);
  } else {
    S6_RETURN_IF_ERROR(GeneratePyFrameAbiProlog(ctx));
    SetUpFunctionArguments(ctx);
    e.jmp(after_prolog);
  }

  if (has_fast_abi) {
    e.align(asmjit::AlignMode::kAlignCode, 16);
    ctx.BindFastEntryPoint();

    GenerateInitialSetup(ctx);
    S6_RETURN_IF_ERROR(GenerateFastAbiProlog(ctx));
  }
  e.bind(after_prolog);

  // Inlines Py_EnterRecursiveCall
  // TODO: Bail out when recursion depth is too great.
  JitStub<PyThreadState> tstate =
      ctx.stack_frame().thread_state().Load(ctx.scratch_reg(), e);
  e.inc(tstate.recursion_depth().Mem());
  return absl::OkStatus();
}

absl::Status GenerateCleanup(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();
  ctx.BindCleanupPoint();
  // Inlines Py_LeaveRecursiveCall.
  JitStub<PyThreadState> tstate =
      ctx.stack_frame().thread_state().Load(ctx.scratch_reg(), e);
  e.dec(tstate.recursion_depth().Mem());

  asmjit::Label out = e.newLabel();
  if (!ctx.has_free_or_cell_vars() && !ctx.is_generator()) {
    // We can inline the fast path.

    // tstate->frame = frame->back
    JitStub<PyThreadState> tstate =
        ctx.stack_frame().thread_state().Load(ctx.scratch_reg(), e);
    JitStub<PyFrameObject> back = ctx.pyframe().f_back().Load(x86::rcx, e);
    tstate.frame().Store(back.Reg(), e);

    asmjit::Label fail = e.newLabel();
    // If the frame wasn't called with the fast calling convention, a caller
    // function owns a reference to the PyFrameObject, so just do nothing here.
    // Everything will be deallocated when the frame destructor fires and we
    // hold a borrowed reference.
    e.cmp(ctx.stack_frame().called_with_fast_calling_convention().Mem(),
          asmjit::imm(1));
    e.jnz(out);

    // This frame was created by the fast calling convention, so it used the
    // PyFrameObjectCache to borrow a reference to a PyFrameObject. We must
    // deallocate all the members we used and release it back to the frame
    // object cache.

    // Inline GetPyFrameObjectCache::Finished.
    // Note we can assume all caller-saves are available here.
    x86::Gp head = x86::rcx;
    PyFrameObjectCache* cache = GetPyFrameObjectCache();
    e.mov(head, asmjit::imm(cache->GetHeadPointer()));

    // Is this object at the top of stack?
    JitStub<PyFrameObject> pyframe(x86::ptr(ctx.pyframe_reg()));
    e.mov(ctx.scratch_reg(), x86::ptr(head));
    e.sub(ctx.scratch_reg(), sizeof(PyFrameObjectCache::SizedFrameObject));
    e.cmp(pyframe.Reg(), ctx.scratch_reg());
    e.jnz(fail);

    // Is the object freeable?
    e.cmp(pyframe.ob_refcnt().Mem(), asmjit::imm(1));
    e.jnz(fail);

    // Is this a cached frame?
    e.mov(ctx.scratch_reg(), asmjit::imm(cache->GetType()));
    e.cmp(pyframe.ob_type().Mem(), ctx.scratch_reg());
    e.jnz(fail);

    // Back up the head pointer. Note we don't bother freeing tombstones on this
    // fast path. Because we know pyframe == GetHead() - 1 already, we can
    // just store pyframe into GetHead().
    e.mov(x86::ptr(head), pyframe.Reg());
    e.jmp(out);

    e.bind(fail);
  }

  // We call a cleanup function here so shelve the function result in r12.
  // We know r12 is unused because it is the profile counter register.
  // TODO: Clean this up.
  e.mov(x86::r12, x86::rax);

  ctx.stack_frame().Load(CallStub<CleanupStackFrame>::stack_frame(), e);
  if (ctx.is_generator()) {
    e.call(asmjit::imm(CleanupStackFrameForGenerator));
  } else {
    e.call(asmjit::imm(CleanupStackFrame));
  }

  e.mov(x86::rax, x86::r12);
  e.bind(out);
  return absl::OkStatus();
}

absl::Status GenerateEpilog(CodeGenerationContext& ctx) {
  asmjit::x86::Emitter& e = ctx.emitter();
  ctx.BindEpilogPoint();
  JitStub<StackFrameWithLayout> layout = ctx.stack_frame_with_layout();

  // Clear stack frame magic cookie.
  ctx.stack_frame().magic().Store(asmjit::imm(0), e);

  // Because we don't use the standard frame layout (dirty registers at the
  // bottom of the stack) we cannot use asmjit's FuncFrame to emit the epilog.
  int64_t index = 0;
  for (const x86::Gp& gp : ctx.callee_saved_registers()) {
    layout.callee_saved_register_slot(index++).Load(gp, e);
  }

  // Skip over the rest of the stack content.
  e.mov(x86::rsp, x86::rbp);
  e.pop(x86::rbp);
  e.ret();
  return absl::OkStatus();
}

}  // namespace deepmind::s6
