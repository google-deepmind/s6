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

#include "runtime/deoptimization_runtime.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "code_object.h"
#include "core_util.h"
#include "event_counters.h"
#include "global_intern_table.h"
#include "interpreter.h"
#include "metadata.h"
#include "runtime/callee_address.h"
#include "runtime/evaluator.h"
#include "runtime/generator.h"
#include "runtime/pyframe_object_cache.h"
#include "runtime/stack_frame.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"
#include "strongjit/value_casts.h"
#include "utils/logging.h"
#include "utils/path.h"

namespace deepmind::s6 {
namespace {
// When runtime::Deoptimize() is called, we must save the value of all
// allocatable registers, because they may be required by the deoptimization
// map.
//
// Because DeoptimizeData contains the entire stack frame created by Deoptimize,
// including the pushed base pointer and return address, the stack pointer
// in the caller's frame at the time Deoptimize() was called can be obtained via
//
//   (void*)(deoptimize_data + 1).
struct DeoptimizeData {
  // LINT.IfChange
  // The value of all allocatable registers. These are pushed by Deoptimize() in
  // reverse order.
  uint64_t rax, rcx, rdx, rsi, rdi, r8, r9, r10, rbx, r14, r15, r13;
  // LINT.ThenChange(code_generation/register_allocator.h)

  // The value of the base pointer as pushed by Deoptimize().
  uint64_t rbp;
  // The address to which Deoptimize() will return.
  uint64_t return_address;

  // Computes the RSP value within the caller frame.
  int64_t* ComputeCallerRsp() { return reinterpret_cast<int64_t*>(this + 1); }
};

// Converts a Location as recorded by the code generator to a real value. A
// Location is either a register, stack spill slot or immediate.
int64_t LookupLocation(const Location& l, int64_t* spill_slots,
                       DeoptimizeData& data) {
  if (l.IsImmediate()) {
    return l.ImmediateValue();
  }
  if (l.IsOnStack()) {
    return spill_slots[l.FrameSlot()];
  }
  if (l.IsCallStackSlot()) {
    return data.ComputeCallerRsp()[l.CallStackSlot()];
  }
  S6_CHECK(l.IsInRegister()) << l.ToString();
  auto reg = l.Register();
  if (reg == asmjit::x86::rax) return data.rax;
  if (reg == asmjit::x86::rcx) return data.rcx;
  if (reg == asmjit::x86::rdx) return data.rdx;
  if (reg == asmjit::x86::rsi) return data.rsi;
  if (reg == asmjit::x86::rdi) return data.rdi;
  if (reg == asmjit::x86::r8) return data.r8;
  if (reg == asmjit::x86::r9) return data.r9;
  if (reg == asmjit::x86::r10) return data.r10;
  if (reg == asmjit::x86::rbx) return data.rbx;
  if (reg == asmjit::x86::r14) return data.r14;
  if (reg == asmjit::x86::r15) return data.r15;
  if (reg == asmjit::x86::r13) return data.r13;
  S6_LOG(FATAL) << "Unknown reg";
}

void ReplaceRematerializeInsts(const SafepointInst* safepoint,
                               EvaluatorContext& evaluator_ctx) {
  absl::flat_hash_set<const RematerializeInst*> rematerialized;
  for (const Value* operand : safepoint->operands()) {
    if (const RematerializeInst* r = dyn_cast<RematerializeInst>(operand)) {
      if (!rematerialized.insert(r).second) {
        // This is a duplicate rematerialize in the operand list. Just incref
        // the existing value.
        Py_INCREF(evaluator_ctx.Get<PyObject*>(r));
        continue;
      }
      std::vector<int64_t> arguments;
      arguments.reserve(r->operands().size());
      for (const Value* v : r->operands()) {
        arguments.push_back(evaluator_ctx.Get(v));
      }
      if (r->callee() == Callee::kPyFloat_FromDouble) {
        // The argument is a double, so casting the function to one
        // taking int64_t args will not work.
        evaluator_ctx.Set(
            r, PyFloat_FromDouble(absl::bit_cast<double>(arguments[0])));
      } else {
        void* address = *GetCalleeSymbolAddress(r->callee());
        evaluator_ctx.Set(r, CallNative(address, arguments));
      }
    }
  }
}
}  // namespace

int64_t LookupLocation(const Location& l, int64_t* spill_slots) {
  if (l.IsImmediate()) {
    return l.ImmediateValue();
  }
  if (l.IsOnStack()) {
    return spill_slots[l.FrameSlot()];
  }
  S6_LOG(FATAL) << "Location cannot be a register or call stack slot!"
                << l.ToString();
}

// The implementation of deoptimization. This is called by the Deoptimize()
// trampoline with:
//   `data`: Contains the current register state.
extern "C" PyObject* S6_DeoptimizeImpl(DeoptimizeData& data) {
  StackFrame* stack_frame = StackFrame::GetFromBasePointer(data.rbp);
  CodeObject* code_object = stack_frame->s6_code_object();
  S6_CHECK(code_object);

  // Stage 0 of deoptimization: Mark this function as deoptimized so that no
  // else attempts to run its compiled code again.
  code_object->MarkDeoptimized();
  code_object->metadata()->Deoptimize();
  EventCounters::Instance().Add("runtime.deoptimization_count", 1);

  StackFrameWithLayout layed_out_stack_frame(*stack_frame,
                                             code_object->stack_frame_layout());
  int64_t* spill_slots = layed_out_stack_frame.GetSpillSlots<int64_t>().data();

  GeneratorState* gen_state = GeneratorState::Get(stack_frame->pyframe());
  if (gen_state) {
    spill_slots = reinterpret_cast<int64_t*>(gen_state->spill_slots());
  }

  // Stage 1 of deoptimization is to use the Strongjit evaluator to pick up
  // and transition to a safe point (a BytecodeBeginInst).
  const DeoptimizationMap& deopt_map = code_object->deoptimization_map();
  const Instruction* inst =
      deopt_map.GetInstructionAtAddress(data.return_address);
  const Function& f = code_object->function();
  PyFrameObject* pyframe = stack_frame->pyframe();

  EvaluatorContext evaluator_ctx(f, pyframe);

  S6_VLOG(1) << "Deoptimize from " << f.name() << " at " << FormatOrDie(*inst);

  const SafepointInst* safepoint = nullptr;
  if (const DeoptimizeIfInst* deopt_inst = dyn_cast<DeoptimizeIfInst>(inst)) {
    const Block& b = *deopt_inst->true_successor();
    // This is a heuristic; deoptimize_if only currently gets generated for
    // exception paths.
    code_object->metadata()->set_except_observed(true);

    // Prime `evaluator_ctx` with values from the stack and registers.
    for (const auto& [v, loc] : deopt_map.live_values(deopt_inst)) {
      int64_t l = LookupLocation(loc, spill_slots, data);
      evaluator_ctx.Set(v, l);
    }

    // Invoke the evaluator. This will complete either with a ReturnInst,
    // ExceptInst, YieldInst or BytecodeBeginInst.
    FunctionResult result = *EvaluateFunction(*b.begin(), evaluator_ctx);
    // If the function finished and the last instruction wasn't a YieldInst, the
    // function is truly complete and we can return.
    safepoint = dyn_cast<SafepointInst>(result.final_instruction);
    if (!safepoint) return result.return_value;

  } else {
    safepoint = cast<SafepointInst>(inst);

    if (const DeoptimizeIfSafepointInst* deopt_safepoint =
            dyn_cast<DeoptimizeIfSafepointInst>(safepoint)) {
      std::string text = absl::StrCat("deoptimize.reason(",
                                      deopt_safepoint->description_str(), ")");
      EventCounters::Instance().Add(text, 1);
    }

    for (const auto& [v, loc] : deopt_map.live_values(inst)) {
      int64_t l = LookupLocation(loc, spill_slots, data);
      evaluator_ctx.Set(v, l);
    }
  }

  // Stage 1.5: run any Rematerialize instructions referred to by the safepoint.
  ReplaceRematerializeInsts(safepoint, evaluator_ctx);

  // Stage 2: transition into the CPython interpreter.

  // The frame is no longer executing optimized code and may now outlive the
  // frame instantiation. As a result, the references f_code, f_back, f_globals
  // and f_builtins should no longer be borrowed, if they ever were.
  if (stack_frame->called_with_fast_calling_convention()) {
    Py_XINCREF(pyframe->f_back);
    Py_XINCREF(pyframe->f_code);
    Py_XINCREF(pyframe->f_builtins);
    Py_XINCREF(pyframe->f_globals);
  }
  // Mark the stack frame as no longer live.
  stack_frame->ClearMagic();

  // If we're a generator, we need to remove the generator state from the value
  // stack now.
  if (pyframe->f_stacktop > pyframe->f_valuestack) {
    S6_CHECK_EQ(pyframe->f_stacktop, &pyframe->f_valuestack[1]);
    Py_CLEAR(pyframe->f_valuestack[0]);
    pyframe->f_stacktop = pyframe->f_valuestack;
    S6_CHECK_EQ(pyframe->f_stacktop, &pyframe->f_valuestack[0]);
  }

  if (f.is_traced()) {
  }
  // Invoke the interpreter to finish up.
  PyObject* obj = InvokeBytecodeInterpreter(*safepoint, evaluator_ctx,
                                            code_object->program());

  if (stack_frame->called_with_fast_calling_convention()) {
    Py_XDECREF(pyframe->f_back);
    Py_XDECREF(pyframe->f_code);
    Py_XDECREF(pyframe->f_builtins);
    Py_XDECREF(pyframe->f_globals);

    if (Py_REFCNT(pyframe) == 1) {
      // This frame will be recycled, so clear out all fastlocals and ensure
      // the value stack is fully cleared.
      // Decref all fastlocals (localsplus before the value stack starts).
      PyCodeObject* co = pyframe->f_code;
      for (PyObject** it = &pyframe->f_localsplus[0];
           it != pyframe->f_valuestack; ++it) {
        Py_CLEAR(*it);
      }

      // And ensure the value stack is cleaned up. All references here will have
      // been decreffed, but the values may not be nullptr. Just set them to
      // nullptr.
      for (int64_t i = 0; i < co->co_stacksize; ++i) {
        pyframe->f_valuestack[i] = nullptr;
      }
    }
  }

  // EvalFrame will end by setting tstate->frame = nullptr. Set it back to
  // our materialized frame so that the epilog code (runtime::CleanupStackFrame)
  // can correctly mop up.
  stack_frame->thread_state()->frame = pyframe;
  return obj;
}

// The entry point for deoptimization. The caller has set r11 to the
// DeoptimizeInfo* for this deoptimization point. All allocatable registers may
// contain valid data and must be saved.
//
// Because of this custom calling convention, this function is written in
// assembly.
__attribute__((__naked__)) PyObject* Deoptimize() {
  asm volatile(R"(
  # Use the standard prolog so GDB can correctly unwind the stack.
  pushq %rbp
  movq %rsp, %rbp

  # Create the DeoptimizeData object on the stack, by pushing all the contents
  # in reverse order.
  pushq %r13
  pushq %r15
  pushq %r14
  pushq %rbx
  pushq %r10
  pushq %r9
  pushq %r8
  pushq %rdi
  pushq %rsi
  pushq %rdx
  pushq %rcx
  pushq %rax

  # Finally set rdi (first argument) to the current stack pointer (DeoptimizeData*).
  movq %rsp, %rdi
  call S6_DeoptimizeImpl

  addq $(12 * 8), %rsp
  popq %rbp
  ret
)");
}

StackFrame* FindStackFrameForPyFrameObject(PyFrameObject* pyframe,
                                           int64_t max_frames) {
  // Walk back through the base pointer chain looking for the stack frame magic.
  ProgramAddress rbp =
      reinterpret_cast<ProgramAddress>(__builtin_frame_address(0));
  PyThreadState* thread_state = PyThreadState_GET();
  while (max_frames-- && rbp) {
    StackFrame* maybe_stack_frame = StackFrame::GetFromBasePointer(rbp);
    ProgramAddress next_rbp = *reinterpret_cast<ProgramAddress*>(rbp);
    if (maybe_stack_frame->HasValidMagic() &&
        maybe_stack_frame->pyframe() == pyframe &&
        maybe_stack_frame->thread_state() == thread_state) {
      S6_DCHECK_EQ(maybe_stack_frame->thread_state(), PyThreadState_GET())
          << "maybe_stack_frame at " << maybe_stack_frame
          << " has invalid thread_state";
      return maybe_stack_frame;
    }
    rbp = next_rbp;
  }
  return nullptr;
}

void DeoptimizePausedGenerator(GeneratorState* gen_state,
                               PyFrameObject* pyframe) {
  const YieldValueInst* yi = gen_state->yield_value_inst();
  S6_CHECK(yi) << "Deoptimizing generator that isn't paused?";
  S6_VLOG(2) << "Deoptimizing paused generator at " << FormatOrDie(*yi);
  ValueMap* values = gen_state->value_map();

  PyObject* gen_state_object = pyframe->f_valuestack[0];
  // We always deoptimize when a generator is about to be invoked, so there
  // must be a value sent back to the yielded instruction.
  PyObject* pushed_value = nullptr;
  if (pyframe->f_stacktop == &pyframe->f_valuestack[2]) {
    pushed_value = pyframe->f_valuestack[1];
    pyframe->f_valuestack[1] = nullptr;
  } else {
    S6_CHECK_EQ(pyframe->f_stacktop, &pyframe->f_valuestack[1]);
  }

  // Erase the generator state object from the value stack. We keep a
  // reference (in gen_state_object) and will decref it later.
  pyframe->f_valuestack[0] = nullptr;
  pyframe->f_stacktop = pyframe->f_valuestack;

  // Prime the frame's value stack, bytecode offset and try handler stack.
  EvaluatorContext ctx(gen_state->code_object()->function(), pyframe);
  if (values) {
    ctx.CopyValues(*values);
  } else {
    // We're in generated code.
    const DeoptimizationMap& deopt_map =
        gen_state->code_object()->deoptimization_map();

    for (const auto& [v, loc] :
         deopt_map.live_values(gen_state->yield_value_inst())) {
      if (loc.IsInRegister()) {
        // The yielded value is in a register. We don't need this here, and
        // no other values should be in registers because YieldValueInst is
        // marked kClobberAllRegisters.
        continue;
      }
      int64_t l = LookupLocation(
          loc, reinterpret_cast<int64_t*>(gen_state->spill_slots()));
      ctx.Set(v, l);
    }
  }
  ReplaceRematerializeInsts(yi, ctx);

  // The generator state is no longer paused.
  gen_state->set_yield_value_inst(nullptr);

  PrepareForBytecodeInterpreter(*yi, ctx, gen_state->code_object()->program());
  // The top of stack must always be the value sent back to the generator.
  if (pushed_value) {
    *pyframe->f_stacktop++ = pushed_value;
  }
  // Resume at the next instruction after the YIELD_VALUE, so lasti =
  // addressof(YIELD_VALUE).
  pyframe->f_lasti = yi->bytecode_offset();

  // Okay, the frame is now set up and good to go. We don't need the generator
  // state any more.
  Py_DECREF(gen_state_object);
  S6_CHECK(IsDeoptimizedGeneratorFrame(pyframe));
}

}  // namespace deepmind::s6
