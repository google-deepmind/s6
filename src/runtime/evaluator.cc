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

#include "runtime/evaluator.h"

#include <Python.h>
#include <code.h>
#include <methodobject.h>

#include <cstdint>
#include <cstring>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core_util.h"
#include "interpreter.h"
#include "runtime/generator.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/util.h"
#include "utils/logging.h"
#include "utils/path.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

EvaluatorContext::EvaluatorContext(const Function& f, PyFrameObject* pyframe)
    : values_(f), pyframe_(pyframe) {
  PyCodeObject* co = reinterpret_cast<PyCodeObject*>(pyframe->f_code);
  for (int64_t i = 0; i < PyTuple_GET_SIZE(co->co_names); ++i) {
    PyObject* unicode = PyTuple_GET_ITEM(co->co_names, i);
    names_table_[PyUnicode_AsUTF8(unicode)] = unicode;
  }
}

absl::StatusOr<FunctionResult> EvaluateFunction(const Instruction& begin_inst,
                                                EvaluatorContext& ctx) {
  const Instruction* inst = &begin_inst;
  while (!ctx.IsFinished()) {
    if (const BytecodeBeginInst* bi = dyn_cast<BytecodeBeginInst>(inst);
        bi && ctx.finish_on_bytecode_begin()) {
      return FunctionResult{nullptr, bi};
    }

    EVLOG(2) << FormatOrDie(*inst);
    S6_RETURN_IF_ERROR(Evaluate(*inst, ctx));

    if (!ctx.IsFinished()) inst = ctx.GetNextInstruction();
  }
  return FunctionResult{ctx.GetReturnValue(), inst};
}

absl::StatusOr<PyObject*> EvaluateFunction(const Function& f,
                                           PyFrameObject* frame) {
  if (Py_EnterRecursiveCall(""))
    // Recursion limit reached.
    return nullptr;
  PyThreadState* tstate = PyThreadState_GET();
  EVLOG(1) << "Executing function " << f.name();

  if (frame->f_gen) {
    // This is a generator.
#if PY_MINOR_VERSION < 7
    // On entry to a generator's or coroutine's frame we have to save the
    // caller's exception state and possibly restore the generator's exception
    // state.
    if (!frame->f_exc_type || frame->f_exc_type == Py_None) {
      SaveExceptionState(tstate, frame);
    } else {
      SwapExceptionState(tstate, frame);
    }
#endif

    GeneratorState* generator_state = GeneratorState::Get(frame);
    if (!generator_state) {
      // Create a new generator state. The evaluator does not use spill slots.
      generator_state = GeneratorState::Create(frame, /*num_spill_slots=*/0);
    }
  }

  frame->f_executing = 0;
  tstate->frame = frame;

  const Instruction* inst = &*f.entry().begin();

  bool resuming_generator = false;
  EvaluatorContext ctx(f, frame);
  GeneratorState* gen_state = GeneratorState::Get(frame);
  if (gen_state) {
    gen_state->EnsureValueMapCreated(f);
    ctx.CopyValues(*gen_state->value_map());
    if (gen_state->yield_value_inst()) {
      const YieldValueInst* yi = gen_state->yield_value_inst();
      S6_CHECK(yi) << "Resuming generator without yield?";
      S6_VLOG(1) << "Resuming generator from " << FormatOrDie(*yi);
      S6_CHECK(frame->f_stacktop != nullptr);
      // The result of the YieldValueInst is the top of the value stack.
      ctx.Set(yi, *--frame->f_stacktop);
      inst = &*std::next(yi->GetIterator());

      // Now we've resumed, the generator is no longer paused.
      gen_state->set_yield_value_inst(nullptr);
      resuming_generator = true;
    }
  }

  // Unless we're resuming a generator, move the arguments from fastlocals into
  // SSA values.
  if (!resuming_generator) {
    const Block& entry = f.entry();
    int64_t i = 0;
    for (const BlockArgument* arg : entry.block_arguments()) {
      ctx.Set(arg, frame->f_localsplus[i]);
      frame->f_localsplus[i++] = nullptr;
    }
  }

  auto cleanup_frame = absl::MakeCleanup([&]() {
    EVLOG(1) << "Finished executing function" << f.name();

    Py_LeaveRecursiveCall();
    frame->f_executing = 0;
    tstate->frame = frame->f_back;
  });

  if (S6_VLOG_IS_ON(1)) {
    S6_LOG_LINES(INFO, FormatOrDie(f));
  }

  S6_ASSIGN_OR_RETURN(FunctionResult result, EvaluateFunction(*inst, ctx));

  if (const ReturnInst* ri = dyn_cast<ReturnInst>(result.final_instruction)) {
    if (gen_state) {
      DeallocateGeneratorState(frame);
#if PY_MINOR_VERSION < 7
      RestoreAndClearExceptionState(tstate, frame);
#endif
    }
    return result.return_value;
  } else if (isa<ExceptInst>(result.final_instruction)) {
    if (gen_state) {
      DeallocateGeneratorState(frame);
#if PY_MINOR_VERSION < 7
      RestoreAndClearExceptionState(tstate, frame);
#endif
    }
    return nullptr;
  } else if (const YieldValueInst* yi =
                 dyn_cast<YieldValueInst>(result.final_instruction)) {
    // Yielding a value should *not* tear down the frame contents.
    std::move(cleanup_frame).Cancel();
    S6_CHECK(gen_state);
    gen_state->set_yield_value_inst(yi);
    *gen_state->value_map() = ctx.value_map();
#if PY_MINOR_VERSION < 7
    if (!yi->try_handlers().empty()) {
      SwapExceptionState(tstate, frame);
    } else {
      RestoreAndClearExceptionState(tstate, frame);
    }
#endif

    Py_LeaveRecursiveCall();

    tstate->frame->f_executing = 0;
    tstate->frame = frame->f_back;

    return result.return_value;
  } else if (const SafepointInst* si =
                 dyn_cast<SafepointInst>(result.final_instruction)) {
    return InvokeBytecodeInterpreter(*si, ctx, /*program=*/{});
  } else {
    S6_RET_CHECK(isa<ExceptInst>(result.final_instruction));
    return nullptr;
  }
}

class EvaluatorValueStack {
 public:
  EvaluatorValueStack(const EvaluatorContext& ctx, const SafepointInst& inst)
      : ctx_(ctx), inst_(inst), height_(inst.value_stack().size()) {}

  template <typename T = int64_t>
  T Pop() {
    S6_CHECK_GE(height_, 1);
    return ctx_.Get<T>(inst_.value_stack()[--height_]);
  }

  // Peek starts at one to follow the Python convention.
  template <typename T = int64_t>
  T Peek(int64_t n) {
    S6_CHECK_GE(n, 1);
    S6_CHECK_GE(height_ - n, 0);
    return ctx_.Get<T>(inst_.value_stack()[height_ - n]);
  }

  void Drop(int64_t n) {
    S6_CHECK_GE(n, 0);
    S6_CHECK_GE(height_ - n, 0);
    height_ -= n;
  }

  int64_t height() { return height_; }

 private:
  const EvaluatorContext& ctx_;
  const SafepointInst& inst_;
  int64_t height_;
};

template <typename T>
class PushFrontSpan {
 public:
  using value_type = T;
  using iterator = typename absl::Span<T>::iterator;

  explicit PushFrontSpan(absl::Span<T> span) : span_(span), pos_(span.size()) {}

  void push_front(T t) {
    S6_CHECK_GE(pos_, 1);
    span_[--pos_] = t;
  }

  iterator begin() { return span_.begin() + pos_; }
  iterator end() { return span_.end(); }

  size_t size() { return span_.size() - pos_; }

  size_t pos() { return pos_; }

 private:
  absl::Span<T> span_;
  size_t pos_;
};

void PrepareForBytecodeInterpreter(
    const SafepointInst& safepoint, EvaluatorContext& ctx,
    absl::Span<BytecodeInstruction const> program) {
  PyFrameObject* pyframe = ctx.pyframe();
  // Prime the frame's value stack, bytecode offset, fastlocals and try handler
  // stack.
  PyCodeObject* co = pyframe->f_code;
  int64_t num_fastlocals = co->co_nlocals + PyTuple_GET_SIZE(co->co_cellvars) +
                           PyTuple_GET_SIZE(co->co_freevars);
  pyframe->f_valuestack = &pyframe->f_localsplus[num_fastlocals];

  // Unwind the stack and block stack simultaneously.
  // This is required to undo the padding added to finally handlers to make the
  // stack static. We need to restore the original CPython stack organisation.
  // This organisation is described by the END_FINALLY comment
  // in strongjit/ingestion_handlers.cc.
  // We need to unwind from the top to the bottom of the stacks otherwise
  // it gets nasty because of `finally_fallthrough_popped_handler` (explained
  // in the END_FINALLY comment)

  // The sources:
  EvaluatorValueStack stack(ctx, safepoint);
  absl::Span<TryHandler const> handler_stack = safepoint.try_handlers();
  auto pop_handler = [&]() {
    const TryHandler& handler = handler_stack.back();
    handler_stack.remove_suffix(1);
    return handler;
  };

  // The destinations: they are filled from the top to the bottom.
  // First we compute the height that we would reach if there weren't any
  // finally blocks. If there was one, the stack_target will not get to the
  // bottom of f_valuestack and this will be corrected by a std::copy after the
  // main loop.
  PushFrontSpan<PyObject*> stack_target(absl::MakeSpan(
      pyframe->f_valuestack,
      std::min<size_t>(pyframe->f_code->co_stacksize, stack.height())));
  PushFrontSpan<PyTryBlock> block_target(absl::MakeSpan(
      pyframe->f_blockstack, std::min<size_t>(std::size(pyframe->f_blockstack),
                                              handler_stack.size())));

  while (!handler_stack.empty()) {
    const TryHandler& handler = pop_handler();
    if (handler.kind() != TryHandler::kFinallyHandler) {
      // The handler is not a finally handler so the unwinding is
      // straightforward. Just copy the stack slot one by one
      // and then copy the block handler.
      while (stack.height() > handler.stack_height()) {
        stack_target.push_front(stack.Pop<PyObject*>());
      }
      block_target.push_front({TryHandlerKindToOpcode(handler.kind()),
                               handler.pc_value().AsOffset(),
                               static_cast<int>(stack_target.pos())});
      continue;
    }

    // Here, the handler is a finally handler so the unwinding is more complex.
    // In case of a finally handler the strongJIT stack and CPython stack do
    // not match. The stacks format are explained in the END_FINALLY comment
    // in ingestion_handlers. The important bit is that the value that is 6
    // slots above the handler stack height is a discriminator that determines
    // which stack shape we are in. If the LSB of the discriminator is
    // 0, the discriminator is a PyObject pointer to the exception type of
    // the exception we are handling. If it is one, we are coming from
    // an other source and the discriminator value is the result
    // of WhyToDiscriminator which just does an bitwiseor of the Why value
    // with 1.
    while (stack.height() > handler.stack_height() + 6) {
      stack_target.push_front(stack.Pop<PyObject*>());
    }
    int64_t discriminator = stack.Peek(1);
    if ((discriminator & 1) == 0) {
      // The finally handler handles an exception, because the discriminator is
      // a pointer (LSB is 0).
      while (stack.height() > handler.stack_height()) {
        stack_target.push_front(stack.Pop<PyObject*>());
      }
      block_target.push_front(
          {TryHandlerKindToOpcode(TryHandler::kExceptHandler),
           handler.pc_value().AsOffset(),
           static_cast<int>(stack_target.pos())});
      continue;
    }
    switch (discriminator) {
      case WhyToDiscriminator(Why::kNot):
        // The finally handler handles a fallthrough.
        if (handler.finally_fallthrough_popped_handler()) {
          // If this flag is set and we are in a fallthrough case, the except
          // handler below was already popped and the corresponding stack values
          // are all zeros so we can skip it.
          const TryHandler& handler = pop_handler();
          S6_CHECK_EQ(handler.kind(), TryHandler::kExceptHandler);
          stack.Drop(3);
        }
        stack_target.push_front(Py_None);
        Py_INCREF(Py_None);
        break;
      case WhyToDiscriminator(Why::kBreak):
        // The finally handler handles a break.
        stack_target.push_front(PyLong_FromWhy(Why::kBreak));
        break;
      case WhyToDiscriminator(Why::kContinue):
        // The finally handler handles a continue.
        stack_target.push_front(PyLong_FromWhy(Why::kContinue));
        S6_CHECK(handler.pc_continue());
        stack_target.push_front(
            PyLong_FromLong(handler.pc_continue()->AsOffset()));
        break;
      case WhyToDiscriminator(Why::kReturn):
        // The finally handler handles a return.
        stack_target.push_front(PyLong_FromWhy(Why::kReturn));
        stack_target.push_front(stack.Peek<PyObject*>(2));
        break;
      default:
        S6_CHECK(false) << "Invalid discriminator for finally handler";
    }
    stack.Drop(6);
  }
  // Finish to copy the stack that is below the bottom block.
  while (stack.height() > 0) {
    stack_target.push_front(stack.Pop<PyObject*>());
  }

  // Now we need to move the stack and block stack to the start if required.
  // std::copy supports overlap if the destination is below the source.
  if (stack_target.pos() != 0) {
    std::copy(stack_target.begin(), stack_target.end(), pyframe->f_valuestack);
    for (PyTryBlock& tb : block_target) {
      tb.b_level -= stack_target.pos();
    }
  }
  if (block_target.pos() != 0) {
    std::copy(block_target.begin(), block_target.end(), pyframe->f_blockstack);
  }
  pyframe->f_stacktop = pyframe->f_valuestack + stack_target.size();
  pyframe->f_iblock = block_target.size();
  // The main stack and block stack translation is finished.

  PcValue pc = PcValue::FromOffset(safepoint.bytecode_offset());

  // If we're about to execute an instruction that was preceded by EXTENDED_ARG,
  // ensure we back up over all the EXTENDED_ARGs otherwise the interpreter will
  // misparse this instruction.
  //
  // Hardcode EXTENDED_ARG here because the header it is defined in is awkward
  // to access.
  constexpr int64_t kExtendedArg = 144;
  while (!program.empty() && pc.AsIndex() > 0 &&
         program[pc.Prev().AsIndex()].opcode() == kExtendedArg) {
    pc = pc.Prev();
  }
  // Note "lasti" is not the offset to start back up at, it's the last
  // successfully executed op (one prior to the new start point).
  pyframe->f_lasti = pc.Prev().AsOffset();

  int64_t i = 0;
  for (const Value* v : safepoint.fastlocals()) {
    pyframe->f_localsplus[i++] = ctx.Get<PyObject*>(v);
  }

  // Update the reference counts to their expected CPython Value.
  for (const Value* v : safepoint.increfs()) Py_XINCREF(ctx.Get<PyObject*>(v));
  for (const Value* v : safepoint.decrefs()) Py_XDECREF(ctx.Get<PyObject*>(v));
}

PyObject* InvokeBytecodeInterpreter(
    const SafepointInst& safepoint, EvaluatorContext& ctx,
    absl::Span<BytecodeInstruction const> program) {
  PrepareForBytecodeInterpreter(safepoint, ctx, program);
  if (const auto* yi = dyn_cast<YieldValueInst>(&safepoint)) {
    *ctx.pyframe()->f_stacktop++ = ctx.Get<PyObject*>(yi->yielded_value());
  }
  // Invoke the interpreter to finish up.
  return EvalFrame(ctx.pyframe(), PyErr_Occurred() ? 1 : 0);
}

}  // namespace deepmind::s6
