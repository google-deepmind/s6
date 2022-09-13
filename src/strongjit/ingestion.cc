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

#include "strongjit/ingestion.h"

#include <Python.h>
#include <opcode.h>

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "code_object.h"
#include "core_util.h"
#include "cppitertools/reversed.hpp"
#include "interpreter.h"
#include "strongjit/builder.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion_handlers.h"
#include "strongjit/instruction.h"
#include "strongjit/instructions.h"
#include "strongjit/ssa.h"
#include "strongjit/util.h"
#include "type_feedback.h"
#include "utils/logging.h"
#include "utils/path.h"
#include "utils/range.h"
#include "utils/status_macros.h"

// TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION >= 7
#define EXC_INFO(x) exc_state.x
#else
#define EXC_INFO(x) x
#endif

namespace deepmind::s6 {

// Stores information related to a CPython bytecode instruction during analysis
// and translation. This is not exposed to Analyze/Translate handler functions.
class BytecodeInfo {
 public:
  explicit BytecodeInfo(Block* block, PcValue pc) : block_(block), pc_(pc) {}

  // Returns the stack height on entry to this bytecode.
  int64_t entry_stack_height() { return entry_stack_height_; }
  void set_entry_stack_height(int64_t height) {
    S6_CHECK(entry_stack_height_ == -1 || entry_stack_height_ == height)
        << "Multiple predecessors had differing stack heights! previous "
           "height: "
        << entry_stack_height_ << ", new height: " << height << " for offset "
        << pc_.AsOffset();
    entry_stack_height_ = height;
  }

  // Returns the entry block.
  Block* block() { return block_; }

  // Returns the list of bindings used by AnalysisContext::Bind and
  // TranslateContext::Bind.
  absl::InlinedVector<Value**, 2>& bindings() { return bindings_; }

  // Returns the try handler stack.
  absl::InlinedVector<TryHandler, 4>& try_handlers() { return try_handlers_; }
  void set_try_handlers(
      const absl::InlinedVector<TryHandler, 4>& try_handlers) {
    if (try_handlers_.empty()) {
      try_handlers_ = try_handlers;
    } else {
      // Cheap equality check. It would be better but costly to
      // check for the equality of everything.
      S6_CHECK_EQ(try_handlers_.size(), try_handlers.size());
      // try_handler_ is not empty and they both have the same size
      // so both backs can be accessed safely
      S6_CHECK_EQ(try_handlers_.back().kind(), try_handlers.back().kind());
      S6_CHECK_EQ(try_handlers_.back().pc_value().AsOffset(),
                  try_handlers.back().pc_value().AsOffset());
      S6_CHECK_EQ(try_handlers_.back().stack_height(),
                  try_handlers.back().stack_height());
    }
  }

  // Returns whether this block is the start of an exception handler.
  bool is_exception_handler() const { return is_exception_handler_; }
  void set_is_exception_handler() { is_exception_handler_ = true; }

  // Returns whether this block is the start of an finally handler.
  bool is_finally_handler() const { return static_cast<bool>(finally_why_); }

  // Adds new possible reasons for which the finally handler starting at this
  // bytecode may be entered.
  void AddFinallyWhy(WhyFlags why) { finally_why_ |= why; }

  // Returns all the possible reasons for which the finally handler starting
  // at this bytecode may be entered
  WhyFlags finally_why() const { return finally_why_; }

  // Returns whether this block always falls through to the next - there is no
  // explicit control flow.
  bool only_falls_through() const { return only_falls_through_; }
  void set_only_falls_through(bool b) { only_falls_through_ = b; }

  PcValue pc() const { return pc_; }

 private:
  Block* block_;

  // Default to -1, as zero is a valid stack height and we should be able to
  // distinguish between blocks that do not yet have a valid stack height and
  // those with zero stack on entry.
  int64_t entry_stack_height_ = -1;

  absl::InlinedVector<Value**, 2> bindings_;
  absl::InlinedVector<TryHandler, 4> try_handlers_;

  PcValue pc_;
  bool is_exception_handler_ = false;
  bool only_falls_through_ = true;
  WhyFlags finally_why_ = {};
};

// Holds state for all bytecode instructions being translated.
class IngestionContext {
 public:
  explicit IngestionContext(std::vector<BytecodeInfo> bytecode_infos,
                            int64_t num_fastlocals,
                            const Function::TypeFeedback& type_feedback,
                            bool except_observed)
      : bytecode_infos_(std::move(bytecode_infos)),
        num_fastlocals_(num_fastlocals),
        type_feedback_(type_feedback),
        except_observed_(except_observed) {
    for (BytecodeInfo& info : bytecode_infos_) {
      info_for_block_[info.block()] = &info;
    }
  }

  Block* BlockAt(PcValue target) {
    return bytecode_infos_[target.AsIndex()].block();
  }

  BytecodeInfo& InfoAt(PcValue target) {
    return bytecode_infos_[target.AsIndex()];
  }

  BytecodeInfo* InfoForBlock(Block* block) {
    return info_for_block_.contains(block) ? info_for_block_.at(block)
                                           : nullptr;
  }

  int64_t num_fastlocals() const { return num_fastlocals_; }
  const Function::TypeFeedback& type_feedback() const { return type_feedback_; }

  bool except_observed() const { return except_observed_; }

 private:
  std::vector<BytecodeInfo> bytecode_infos_;
  absl::flat_hash_map<Block*, BytecodeInfo*> info_for_block_;
  int64_t num_fastlocals_;
  const Function::TypeFeedback& type_feedback_;
  bool except_observed_;
};

AnalysisContext::AnalysisContext(IngestionContext* ctx, BytecodeInfo* info,
                                 SsaBuilder::BitVector* defines)
    : ctx_(*ctx),
      info_(*info),
      defines_(*defines),
      stack_height_(info->entry_stack_height()),
      try_handlers_(info->try_handlers()) {}

AnalysisContext& AnalysisContext::Push() {
  int64_t value = ctx_.num_fastlocals() + stack_height_++;
  if (defines_.size() <= value) defines_.resize(value + 1);
  defines_.set_bit(value);
  return *this;
}

AnalysisContext& AnalysisContext::Pop() {
  --stack_height_;
  return *this;
}

AnalysisContext& AnalysisContext::Drop(int64_t n) {
  S6_CHECK_GE(n, 0) << "Cannot drop a negative number of elements";
  stack_height_ -= n;
  return *this;
}

AnalysisContext& AnalysisContext::DefFastLocal(int64_t index) {
  if (defines_.size() <= index) defines_.resize(index + 1);
  defines_.set_bit(index);
  return *this;
}

AnalysisContext& AnalysisContext::DoesNotFallthrough() {
  falls_through_ = false;
  only_falls_through_ = false;
  return *this;
}

AnalysisContext& AnalysisContext::MayJumpTo(PcValue target) {
  BytecodeInfo& info = ctx_.InfoAt(target);
  S6_VLOG(2) << "  MayJumpTo " << target.AsOffset() << " with height "
             << stack_height_ << " and blocks " << Format(try_handlers());
  S6_CHECK(!info.is_exception_handler())
      << "Except handler should not be directly jumped to";
  S6_CHECK(!info.is_finally_handler())
      << "Finally handler must not be jumped to "
         "with MayJumpTo. Use MayJumpToFinally";
  info.set_entry_stack_height(stack_height_);
  info.set_try_handlers(try_handlers());
  only_falls_through_ = false;
  return *this;
}

AnalysisContext& AnalysisContext::MayJumpToFinally(PcValue target,
                                                   WhyFlags why) {
  BytecodeInfo& info = ctx_.InfoAt(target);
  S6_CHECK(info.is_finally_handler())
      << "MayJumpToFinally needs to jump to a finally handler";
  S6_VLOG(2) << "  MayJumpToFinally " << target.AsOffset() << " with height "
             << stack_height_ << " and blocks " << Format(try_handlers());
  S6_CHECK_EQ(info.entry_stack_height(), stack_height_)
      << "Mismatch of stack heights when jumping to a finally handler.";
  info.AddFinallyWhy(why);
  only_falls_through_ = false;
  return *this;
}

AnalysisContext& AnalysisContext::Bind(Value** value_ptr) {
  info_.bindings().push_back(value_ptr);
  return *this;
}

Block* AnalysisContext::BlockAt(PcValue target) { return ctx_.BlockAt(target); }

void AnalysisContext::PushExcept(PcValue target) {
  S6_VLOG(2) << "  Pushing Except at " << target.AsOffset() << " with height "
             << stack_height_ << " and blocks " << Format(try_handlers());
  BytecodeInfo& info = ctx_.InfoAt(target);
  S6_CHECK(info.try_handlers().empty());
  info.try_handlers() = try_handlers();
  info.set_is_exception_handler();
  info.block()->SetExceptHandler();
  for (int64_t i = 0; i < 6; ++i) {
    info.block()->CreateBlockArgument();
  }
  info.set_entry_stack_height(stack_height_);
  info.try_handlers().emplace_back(TryHandler::kExceptHandler, target,
                                   stack_height_);
  try_handlers_.emplace_back(TryHandler::kExcept, target, stack_height_);
}

void AnalysisContext::PushFinally(PcValue target) {
  S6_VLOG(2) << "  Pushing Finally at " << target.AsOffset() << " with height "
             << stack_height_ << " and blocks " << Format(try_handlers());
  BytecodeInfo& info = ctx_.InfoAt(target);
  S6_CHECK(info.try_handlers().empty());
  info.try_handlers() = try_handlers();
  info.AddFinallyWhy(Why::kException);
  info.block()->SetFinallyHandler();
  for (int64_t i = 0; i < 6; ++i) {
    info.block()->CreateBlockArgument();
  }
  info.set_entry_stack_height(stack_height_);
  info.try_handlers().emplace_back(TryHandler::kFinallyHandler, target,
                                   stack_height_);
  try_handlers_.emplace_back(TryHandler::kFinally, target, stack_height_);
}

void AnalysisContext::PushLoop(PcValue exit) {
  S6_VLOG(2) << "  Pushing Loop exiting to " << exit.AsOffset()
             << " with height " << stack_height_ << " and blocks "
             << Format(try_handlers());
  try_handlers_.emplace_back(TryHandler::kLoop, exit, stack_height_);
}

void AnalysisContext::SetContinueTarget(const TryHandler& handler,
                                        PcValue target) {
  S6_CHECK(handler.kind() == TryHandler::kFinally);
  BytecodeInfo& info = ctx_.InfoAt(handler.pc_value());
  S6_CHECK(info.is_finally_handler());
  auto& finally_handler = info.try_handlers().back();
  S6_CHECK_EQ(finally_handler.kind(), TryHandler::kFinallyHandler);
  finally_handler.set_pc_continue(target);
}

absl::optional<PcValue> AnalysisContext::GetContinueTarget() {
  const TryHandler& handler = GetTopHandler();
  S6_CHECK_EQ(handler.kind(), TryHandler::kFinallyHandler);
  return handler.pc_continue();
}

WhyFlags AnalysisContext::GetWhyFinally() {
  const TryHandler& handler = GetTopHandler();
  S6_CHECK_EQ(handler.kind(), TryHandler::kFinallyHandler);
  BytecodeInfo& info = ctx_.InfoAt(handler.pc_value());
  S6_CHECK(info.is_finally_handler());
  return info.finally_why();
}

TryHandler AnalysisContext::PopHandler() {
  S6_CHECK(!try_handlers_.empty());
  TryHandler handler = try_handlers_.back();
  try_handlers_.pop_back();
  stack_height_ = handler.stack_height();
  return handler;
}

absl::optional<TryHandler> AnalysisContext::UnwindHandlerStack(Why why) {
  S6_CHECK(why == Why::kBreak || why == Why::kContinue || why == Why::kReturn)
      << "unhandled analysis unwind kind!";
  // Break and Continue unwind until a loop or finally handler is found.
  // Return unwinds until a finally handler is found, or just the bottom of the
  // block stack.

  while (!try_handlers_.empty()) {
    TryHandler& handler = try_handlers_.back();

    if (handler.kind() == TryHandler::kFinally) return PopHandler();

    if (handler.kind() == TryHandler::kLoop) {
      if (why == Why::kContinue) return handler;
      if (why == Why::kBreak) return PopHandler();
    }

    PopHandler();
  }

  return absl::nullopt;
}

absl::optional<TryHandler> AnalysisContext::GetTopExceptionHandler() {
  for (const TryHandler& handler : iter::reversed(try_handlers_)) {
    if (handler.kind() == TryHandler::kExcept ||
        handler.kind() == TryHandler::kFinally) {
      return handler;
    }
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////
// TranslateContext

void TranslateContext::PopExceptHandler(Builder& b, const TryHandler& handler) {
  PopAndDecrefStackTo(b, handler.stack_height() + 3);
  S6_CHECK_GE(stack_height_, 3);
  Value* tstate =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kThreadState);
  Value* old_type =
      b.LoadSteal(tstate, offsetof(PyThreadState, EXC_INFO(exc_type)));
  Value* old_value =
      b.LoadSteal(tstate, offsetof(PyThreadState, EXC_INFO(exc_value)));
  Value* old_traceback =
      b.LoadSteal(tstate, offsetof(PyThreadState, EXC_INFO(exc_traceback)));
  b.StoreDonate(Pop(), tstate, offsetof(PyThreadState, EXC_INFO(exc_type)));
  b.StoreDonate(Pop(), tstate, offsetof(PyThreadState, EXC_INFO(exc_value)));
  b.StoreDonate(Pop(), tstate,
                offsetof(PyThreadState, EXC_INFO(exc_traceback)));
  b.DecrefOrNull({old_type, old_value, old_traceback});
}

bool TranslateContext::PopFinallyHandler(Builder& b,
                                         const TryHandler& handler) {
  // The finally handler expects six values on the stack. The sixth (TOS) is a
  // dicriminator that contains the reason for which we entered the handler.
  // The stack layout depends of the exact reason the finally handler was taken.
  // To see a detailled explanation of the stack layout, see the comment about
  // END_FINALLY in ingestion_handlers.cc.
  PopAndDecrefStackTo(b, handler.stack_height() + 6);
  S6_CHECK_GE(stack_height_, 6);
  Value* discriminator = Peek(1);
  Value* mayberetval = Peek(2);  // The returned value in case of return.

  if (!handler.finally_fallthrough_popped_handler()) {
    // The flag is unset so this is the normal unwinding of a finally handler.
    // We check if we are in an exceptional case or not by checking the LSB of
    // the discriminator.
    b.Conditional(
        b.And(discriminator, b.Int64(1)),
        [&](Builder b) {
          // Non exceptional case: Doing another branch to know whether this is
          // a return or not would cost more than just doing a DecrefOrNull of
          // the potential returned value.
          b.DecrefOrNull(mayberetval);
          return Builder::ValueList{};
        },
        [&](Builder b) {
          // Exceptional case: This finally handler represents an except handler
          // so unwind it as such.
          PopExceptHandler(b, handler);
          return Builder::ValueList{};
        });
    S6_CHECK_EQ(handler.stack_height(), stack_height_);
    return false;
  }
  // The `handler.finally_fallthrough_popped_handler` flag is set. This is the
  // complex case. The semantics is that in the case of a fallthrough to the
  // finally handler, the except handler below has already been popped.
  // Since we don't know if the current finally block was entered
  // by fallthrough or not at ingestion time, we need to gate the unwinding
  // of the except handler below behind a check we are not falling through.
  // For more detail about this situation, see the END_FINALLY comment
  // in ingestion_handlers.cc.
  // In any case there should be at least 3 extra values below the handler
  // stack height for the except handler that is below it.
  S6_CHECK_GE(handler.stack_height(), 3);
  b.Conditional(
      b.IsNotEqual(discriminator, b.Int64(WhyToDiscriminator(Why::kNot))),
      [&](Builder b) {
        // Not in fallthrough case. Emit code to check if we are handling an
        // exception to know which values to decref.
        // We check that by checking if the LSB of the discriminator is 1
        b.Conditional(
            b.And(discriminator, b.Int64(1)),
            [&](Builder b) {
              // Non exceptional, non fallthrough case:
              // Only break, continue and return cases can reach here.
              // Doing another branch to know whether this is a return or not
              // would more costly than just doing a DecrefOrNull of the
              // potentiel returned value.
              b.DecrefOrNull(mayberetval);
              return Builder::ValueList{};
            },
            [&](Builder b) {
              // Exceptional case: Do not unwind this except handler with
              // PopExceptHandler since the except handler below will
              // immediately overwrite the tstate with new values
              PopBlock(b, handler);
              return Builder::ValueList{};
            });
        // Pop the except handler 3 slots below this finally handler since
        // we are not in a fallthrough case.
        PopExceptHandler(
            b, TryHandler(TryHandler::kExceptHandler, handler.pc_value(),
                          handler.stack_height() - 3));
        return Builder::ValueList{};
      });

  S6_CHECK_EQ(handler.stack_height() - 3, stack_height_);
  // Return true to tell that the except handler below has already been poped.
  // UnwindHandlerStack will skip the handler below.
  return true;
}

void TranslateContext::PopAndDecrefStackTo(Builder& b, int64_t stack_height) {
  S6_VLOG(2) << "PopAndDecrefStackTo: " << stack_height;
  std::vector<Value*> values;
  while (stack_height_ > stack_height) {
    values.push_back(Pop());
  }
  if (!values.empty()) {
    b.DecrefOrNull(values);
  }
}

absl::optional<TryHandler> TranslateContext::UnwindHandlerStack(Builder& b,
                                                                Why why) {
  S6_CHECK(why == Why::kException || why == Why::kReturn ||
           why == Why::kBreak || why == Why::kContinue)
      << "unhandled translation unwind kind!";
  // Except unwinds until an exception or finally handler is found.
  // Return unwinds until a finally handler is found.
  // Break and Continue unwind until a loop or finally handler is found.
  S6_VLOG(2) << "UnwindHandlerStack for reason: " << static_cast<int64_t>(why);
  for (auto it = info_.try_handlers().rbegin();
       it != info_.try_handlers().rend(); ++it) {
    TryHandler& handler = *it;
    S6_VLOG(2) << "  Unwind: handler.stack_height(): " << handler.stack_height()
               << ", stack_height_: " << stack_height_;
    if (handler.stack_height() > stack_height_) continue;
    switch (handler.kind()) {
      case TryHandler::kExcept:
        PopBlock(b, handler);
        if (why == Why::kException) return handler;
        continue;
      case TryHandler::kLoop:
        if (why == Why::kContinue) return handler;
        PopBlock(b, handler);
        if (why == Why::kBreak) return handler;
        continue;
      case TryHandler::kFinally:
        PopBlock(b, handler);
        return handler;
      case TryHandler::kExceptHandler:
        if (handler.stack_height() >= stack_height_) continue;
        PopExceptHandler(b, handler);
        continue;
      case TryHandler::kFinallyHandler:
        if (handler.stack_height() >= stack_height_) continue;
        if (PopFinallyHandler(b, handler)) {
          ++it;
          S6_CHECK(it != info_.try_handlers().rend());
          S6_CHECK(it->kind() == TryHandler::kExceptHandler);
        }
        continue;
    }
  }

  PopAndDecrefStackTo(b, 0);
  return absl::nullopt;
}

WhyFlags TranslateContext::GetWhyFinally() {
  const TryHandler& handler = GetTopHandler();
  S6_CHECK_EQ(handler.kind(), TryHandler::kFinallyHandler);
  BytecodeInfo& info = ctx_.InfoAt(handler.pc_value());
  S6_CHECK(info.is_finally_handler());
  return info.finally_why();
}

const TryHandler& TranslateContext::GetTopHandler() const {
  return info_.try_handlers().back();
}

bool TranslateContext::Except(Builder& b) {
  int64_t stack_height = stack_height_;
  absl::optional<TryHandler> handler = UnwindHandlerStack(b, Why::kException);
  stack_height_ = stack_height;

  // We deoptimize exception returns, but we don't deoptimize jumps to
  // exception handlers.
  if (handler.has_value()) {
    b.Except(BlockAt(handler->pc_value()));
    return false;
  } else {
    DecrefFastLocals(b);
    b.Except(nullptr);
    // If exceptions have been observed during interpretation, don't deoptimize.
    return !ctx_.except_observed();
  }
}

// Conditionally unwinds the stack and takes an exception.
void TranslateContext::ExceptIf(Builder& b, Value* condition) {
  bool should_deoptimize = false;
  BrInst* br = b.Conditional(condition, [&](Builder b) {
    should_deoptimize = Except(b);
    return Builder::DoesNotReturn{};
  });
  br->set_true_deoptimized(should_deoptimize);
}

void TranslateContext::ExceptConditional(
    Builder& b, Value* condition,
    absl::FunctionRef<Builder::DoesNotReturn(Builder& b)> cleanup) {
  int64_t stack_height = stack_height_;
  bool should_deoptimize = false;
  BrInst* br = b.Conditional(condition, [&](Builder b) {
    cleanup(b);
    should_deoptimize = Except(b);

    // Some users rely on cleanup being able to pop values without changing the
    // stack_height_.
    stack_height_ = stack_height;
    return Builder::DoesNotReturn();
  });
  br->set_true_deoptimized(should_deoptimize);
}

void TranslateContext::DecrefFastLocals(Builder& b) {
  if (ctx_.num_fastlocals() == 0) return;
  std::vector<Value*> values;
  for (int64_t i = 0; i < ctx_.num_fastlocals(); ++i) {
    values.push_back(ssa_.Use(i, info_.block()));
  }
  b.DecrefOrNull(values);
}

ClassDistributionSummary TranslateContext::GetTypeFeedback() const {
  auto it = ctx_.type_feedback().find({info_.pc(), 0});
  if (it == ctx_.type_feedback().end()) {
    return ClassDistributionSummary();
  }
  return it->second;
}

TranslateContext::TranslateContext(IngestionContext* ctx, BytecodeInfo* info,
                                   SsaBuilder* ssa)
    : ctx_(*ctx),
      info_(*info),
      ssa_(*ssa),
      stack_height_(info_.entry_stack_height()) {}

Value* TranslateContext::Pop() {
  S6_VLOG(3) << "Use " << stack_height_ - 1;
  S6_CHECK_GT(stack_height_, 0) << "Value stack underflow";
  return ssa_.Use(ctx_.num_fastlocals() + --stack_height_, info_.block());
}

void TranslateContext::Drop(int64_t n) {
  S6_CHECK_GE(n, 0) << "Cannot drop a negative number of elements";
  S6_VLOG(3) << "Dropping from " << stack_height_ << " to "
             << stack_height_ - n;
  stack_height_ -= n;
}

// The top of stack (TOS) pointer points one past the stored top value. So the
// TOS element exists at Peak(1). Thus Peak(0) is invalid.
//
// REQUIREMENT: i >= 1.
Value* TranslateContext::Peek(int64_t i) {
  S6_CHECK_GE(i, 1) << "i must be positive; Peek(0) does not return "
                    << "top-of-stack, use Peek(1).";
  S6_VLOG(3) << "Use stack[" << stack_height_ - i << "]";
  return ssa_.Use(ctx_.num_fastlocals() + stack_height_ - i, info_.block());
}

void TranslateContext::Push(Value* v) {
  S6_VLOG(3) << "Def stack[" << stack_height_ << "]";
  ssa_.Def(ctx_.num_fastlocals() + stack_height_++, v, info_.block());
}

void TranslateContext::DefFastLocal(int64_t index, Value* v) {
  S6_VLOG(3) << "Def fastlocal[" << index << "]";
  ssa_.Def(index, v, info_.block());
}

Value* TranslateContext::UseFastLocal(int64_t index) {
  S6_VLOG(3) << "Use fastlocal[" << index << "]";
  return ssa_.Use(index, info_.block());
}

Block* TranslateContext::BlockAt(PcValue target) {
  return ctx_.BlockAt(target);
}

void TranslateContext::Bind(Value* v) {
  *info_.bindings()[next_binding_++] = v;
}

////////////////////////////////////////////////////////////////////////////////
// IngestProgram

absl::Status IngestProgram(absl::Span<const BytecodeInstruction> program,
                           Function& f, int64_t num_fastlocals,
                           int64_t num_arguments, bool except_observed) {
  SsaBuilder ssa;

  const int64_t kMaxProgramSize = 2000;
  if (program.size() > kMaxProgramSize) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Function is too large: ", program.size(), " > ",
                     kMaxProgramSize, " bytecodes"));
  }

  S6_VLOG(2) << "Ingesting " << f.name();
  // Start by creating one block for every bytecode instruction plus one at
  // the end which will be unreachable. This allows us to not special-case the
  // fallthrough logic for the last instruction.
  std::vector<BytecodeInfo> infos;
  infos.reserve(program.size() + 1);
  for (int64_t i = 0; i < program.size() + 1; ++i) {
    infos.emplace_back(f.CreateBlock(), PcValue::FromIndex(i));
  }
  for (const auto& instr : program) {
    S6_VLOG(3) << instr.ToString();
  }
  infos.back().block()->Create<UnreachableInst>();

  // Populate the ingestion context which will own all bytecode infos.
  IngestionContext ctx(std::move(infos), num_fastlocals, f.type_feedback(),
                       except_observed);

  // Stores the set of stack values, indexed absolutely (from zero upwards)
  // defined by a bytecode instruction.
  SsaBuilder::BitVector defines;
  // Similarly for all stack values live-in to a bytecode instruction.
  SsaBuilder::BitVector live_ins;

  // Holds the carried bits from a prior EXTENDED_ARG instruction.
  int64_t arg_carry = 0;

  // Holds extra added CFG edges during analysis, so that exception handlers
  // are modelled correctly.
  std::vector<std::pair<Block*, Block*>> added_edges;

  // The program starts with a stack height of zero.
  ctx.InfoAt(PcValue::FromIndex(0)).set_entry_stack_height(0);

  // Do the analysis on each instruction.
  for (BytecodeInstruction inst : program) {
    defines.clear();
    live_ins.clear();
    BytecodeInfo& info = ctx.InfoAt(inst.pc_value());
    AnalysisContext analysis_ctx(&ctx, &info, &defines);

    inst.set_argument(inst.argument() | (arg_carry << 8));
    arg_carry = 0;
    if (inst.opcode() == EXTENDED_ARG) {
      arg_carry = inst.argument();
    }

    S6_VLOG(2) << "Analyzing " << inst.ToString() << " with stack height "
               << info.entry_stack_height() << " and blocks "
               << Format(info.try_handlers());

    if (info.entry_stack_height() == -1) {
      // This is an unreachable instruction.
      S6_VLOG(3) << "  Dropping this unreachable instruction";
      info.block()->Create<UnreachableInst>();
      continue;
    }
    if (info.is_exception_handler() || info.is_finally_handler()) {
      // Exception and finally handlers have six arguments, which are pushed
      // in order onto the value stack.
      for (int64_t i = 0; i < 6; ++i) {
        analysis_ctx.Push();
      }
      S6_VLOG(3) << "  This is an "
                 << (info.is_exception_handler() ? "exception" : "finally")
                 << " handler";
    }

    // Run the Analyze function.
    S6_ASSIGN_OR_RETURN(AnalyzeFunction fn, GetAnalyzeFunction(inst.opcode()));
    fn(inst, Builder(info.block()), &analysis_ctx);

    // If this bytecode is within an exception scope, assume it could jump to
    // the exception handler. We model this here (before translation) by adding
    // a synthetic predecessor to the except/finally handler block. This allows
    // SSA formation to understand that the except handler can be jumped to.
    if (absl::optional<TryHandler> handler =
            analysis_ctx.GetTopExceptionHandler();
        handler.has_value()) {
      BytecodeInfo& exception_handler_info = ctx.InfoAt(handler->pc_value());
      exception_handler_info.block()->AddPredecessor(info.block());
      added_edges.emplace_back(info.block(), exception_handler_info.block());
    }

    BytecodeInfo& next_info = ctx.InfoAt(inst.pc_value().Next());
    // Discover all the added blocks as the blocks between info.block() and
    // next_info.block().
    auto blocks_begin = info.block()->GetIterator();
    auto blocks_end = next_info.block()->GetIterator();

    // Treat all blocks within the bytecode's region as if they used and defined
    // all possible values.
    int64_t num_values = num_fastlocals + info.entry_stack_height();
    if (live_ins.size() < num_values) live_ins.resize(num_values);
    live_ins.SetBits(0, num_values);
    for (Block& b : MakeRange(blocks_begin, blocks_end)) {
      ssa.SetLiveInValues(&b, live_ins);
      ssa.SetDefinedValues(&b, defines);
    }

    // Handle falls throughs.
    info.set_only_falls_through(analysis_ctx.only_falls_through());
    if (analysis_ctx.falls_through()) {
      S6_CHECK(!next_info.is_exception_handler())
          << "Falling through to exception handler should not be possible";
      Block& last_block = *std::prev(blocks_end);
      Builder b(&last_block);
      if (!next_info.is_finally_handler()) {
        // This is a normal fall trough.
        analysis_ctx.MayJumpTo(inst.pc_value().Next());
        if (!last_block.GetTerminator()) {
          // If the Analyze function didn't add a terminator, fall through.
          b.Jmp(next_info.block());
        } else {
          // A terminator was added, so sanity check it did actually fall
          // through!
          S6_RET_CHECK(absl::c_linear_search(
              last_block.GetTerminator()->successors(), next_info.block()));
        }
      } else {
        // Falling through to a finally block. Care must be taken to match
        // the stack height of the fallthrough path and the exceptional path
        // into the finally handler.
        // Currently we only support two cases:
        // - The fallthrough is done at the same stack height than the
        //   exceptional path.
        // - The fallthrough path is 3 stack slots below the exceptional path
        //   and has one except handler right below missing. That means that
        //   the except handler must be poped only when exiting a finally
        //   block coming from an exception path.
        //
        // No other cases occurred when finally handler compilation support
        // was added. But that may change, hence all the checks
        // in the following code.
        // Which of the two case we're in is marked by
        // `finally_fallthrough_popped_handler` in TryHandler.

        // The fallthrough code will push Py_None on top of the stack which
        // is not wanted.
        analysis_ctx.Pop();

        auto& nth = next_info.try_handlers();
        S6_CHECK(!nth.empty());
        S6_CHECK_EQ(nth.back().kind(), TryHandler::kFinallyHandler);

        if (analysis_ctx.stack_height() == next_info.entry_stack_height()) {
          // Simple case: they are at the same stack height.

          // Check there is an additional finally handler on top.
          S6_CHECK_EQ(analysis_ctx.try_handlers().size() + 1, nth.size());
        } else if (analysis_ctx.stack_height() ==
                   next_info.entry_stack_height() - 3) {
          // Special handled case: The fallthrough path has already poped
          // an except handler while the exceptional case need to pop it at the
          // end of the finally handler block.

          // Check we have an additional except handler below the finally
          // handler.
          if (analysis_ctx.try_handlers().size() + 2 != nth.size() ||
              nth[nth.size() - 2].kind() != TryHandler::kExceptHandler) {
            return absl::UnimplementedError(
                "Can't optimize finally handler: Unexpected case when falling "
                "through into a finally handler.");
          }

          // Declare that the fallthrough case has already poped the
          // ExceptHandler.
          nth.back().set_finally_fallthrough_popped_handler();
          // Push 3 zeroes to pad the stack to the right height.
          analysis_ctx.Push().Push().Push();
        } else {
          // Other case than the supported cases.
          return absl::UnimplementedError(
              "Can't optimize finally handler: fallthrough and exceptional "
              "stack height difference is not supported");
        }
        analysis_ctx.MayJumpToFinally(inst.pc_value().Next(), Why::kNot);
        Value* zero = b.Zero();
        Value* discriminator = b.Int64(WhyToDiscriminator(Why::kNot));
        b.Jmp(next_info.block(), {zero, zero, zero, zero, zero, discriminator});
      }
    }
  }

  // Analysis stage complete; produce pruned SSA.
  ssa.InsertBlockArguments(&f);
  ssa.SetBlockResolverFunction(
      [](const DominatorTree& domtree, const Block* b) {
        // This is called if `b` has been constructed during Translation - it
        // doesn't appear in the dominator tree that is constructed at the end
        // of Analysis.
        //
        // Walk backwards in block layout order until we find the entry of this
        // bytecode instruction. This will exist in the dominator tree.
        while (!domtree.contains(b)) b = &*std::prev(b->GetIterator());
        return b;
      });

  // Undo the CFG modifications we added to ensure except/finally handlers were
  // correctly accounted for by SSA construction.
  for (const auto [pred, succ] : added_edges) {
    succ->RemovePredecessor(pred);
  }

  S6_CHECK_EQ(arg_carry, 0);
  // Phase 2: translation.

  // Prime the start of the function with BlockArguments for all arguments and
  // the constant zero (nullptr) for all non-argument fastlocals.
  int64_t i;
  Block& entry = f.entry();
  BlockInserter inserter(&entry, entry.begin());
  ConstantInst* zero = inserter.Create<ConstantInst>(0);
  for (i = 0; i < num_arguments; ++i) {
    Value* v = entry.CreateBlockArgument();
    ssa.Def(i, v, &entry);
  }
  for (; i < num_fastlocals; ++i) {
    ssa.Def(i, zero, &entry);
  }

  // The number of bytecodes since the last AdvanceProfileCounterInst was
  // inserted. We don't insert AdvanceProfileCounterInsts on bytecodes that
  // only fall through.
  int64_t profile_counter_advance = 0;

  for (BytecodeInstruction inst : program) {
    BytecodeInfo& info = ctx.InfoAt(inst.pc_value());
    TranslateContext translate_ctx(&ctx, &info, &ssa);

    inst.set_argument(inst.argument() | (arg_carry << 8));
    arg_carry = 0;
    if (inst.opcode() == EXTENDED_ARG) {
      arg_carry = inst.argument();
    }

    if (info.entry_stack_height() == -1) {
      // This was unreachable.
      continue;
    }
    S6_VLOG(2) << "Translating " << inst.ToString() << " with stack height "
               << info.entry_stack_height() << " and blocks "
               << Format(info.try_handlers());

    // Obtain the current value stack, before any translation runs.
    std::vector<Value*> entry_value_stack;
    for (int64_t i = 0; i < info.entry_stack_height(); ++i) {
      entry_value_stack.push_back(ssa.Use(num_fastlocals + i, info.block()));
    }
    std::vector<Value*> entry_fastlocals;
    for (int64_t i = 0; i < num_fastlocals; ++i) {
      entry_fastlocals.push_back(ssa.Use(i, info.block()));
    }

    // If this is an except or finally handler, push the 6 entry block values.
    if (info.is_exception_handler() || info.is_finally_handler()) {
      for (BlockArgument* arg : info.block()->block_arguments().subspan(0, 6)) {
        translate_ctx.Push(arg);
        entry_value_stack.push_back(
            ssa.Use(num_fastlocals + entry_value_stack.size(), info.block()));
      }
    }

    // Run the Translate function.
    S6_ASSIGN_OR_RETURN(TranslateFunction fn,
                        GetTranslateFunction(inst.opcode()));
    fn(inst, Builder::FromStart(info.block()), &translate_ctx);

    BytecodeInfo& next_info = ctx.InfoAt(inst.pc_value().Next());
    auto blocks_begin = info.block()->GetIterator();
    auto blocks_end = next_info.block()->GetIterator();

    // Special case for YIELD_VALUE: if the first bytecode block contains a
    // `yield_value`, add the value stack contents minus the top of stack, which
    // is already an operand to yield_value.
    if (auto* yi = dyn_cast<YieldValueInst>(&*info.block()->begin())) {
      *yi->mutable_bytecode_offset() = inst.program_offset();
      auto operands = yi->mutable_value_stack();
      absl::Span<Value*> value_stack = absl::MakeSpan(entry_value_stack);
      value_stack.remove_suffix(1);
      absl::c_copy(value_stack, std::back_inserter(operands));
      for (Value* v : entry_fastlocals) {
        yi->mutable_fastlocals().push_back(v);
      }
      absl::c_copy(info.try_handlers(),
                   std::back_inserter(*yi->mutable_try_handlers()));
    }

    // Insert a bytecode_begin with the current value stack.
    BlockInserter inserter(info.block(), info.block()->begin());
    inserter.Create<BytecodeBeginInst>(inst.program_offset(), entry_value_stack,
                                       entry_fastlocals, info.try_handlers());

    // Insert special code when falling through in a finally handler.
    if (next_info.is_finally_handler()) {
      auto block_last = blocks_end;
      --block_last;
      if (JmpInst* jmp = dyn_cast<JmpInst>(block_last->GetTerminator());
          jmp && jmp->unique_successor() == next_info.block()) {
        S6_CHECK(jmp->arguments().size() >= 6)
            << "Jumping to finally handler with less than 6 arguments";
        if (auto constant = dyn_cast<ConstantInst>(jmp->arguments()[5]);
            constant && constant->value() == WhyToDiscriminator(Why::kNot)) {
          // We are indeed in a finally handler falltrough case.
          Builder b(&*block_last, --block_last->end());
          Value* none = translate_ctx.Pop();
          b.DecrefNotNull(none);
          if (next_info.try_handlers()
                  .back()
                  .finally_fallthrough_popped_handler()) {
            Value* zero = b.Zero();
            translate_ctx.Push(zero);
            translate_ctx.Push(zero);
            translate_ctx.Push(zero);
          }
        }
      }
    }

    // And fill out all the DeoptimizeIfSafepoints emitted during translation
    // (for deoptimization).
    for (DeoptimizeIfSafepointInst* di :
         translate_ctx.deoptimize_if_safepoints()) {
      *di->mutable_bytecode_offset() = inst.program_offset();
      for (const TryHandler& h : info.try_handlers()) {
        di->mutable_try_handlers()->push_back(h);
      }
      for (Value* v : entry_value_stack) {
        di->mutable_value_stack().push_back(v);
      }
      for (Value* v : entry_fastlocals) {
        di->mutable_fastlocals().push_back(v);
      }
    }

    // If this was the entry block, move the `zero` constant right to the start.
    if (info.block() == &entry) {
      zero->RemoveFromParent();
      info.block()->insert(info.block()->begin(), zero);
    }

    // Advance the profile counter. Note that we insert this at the start of the
    // bytecode's block so this is theoretically off-by-one, but that isn't
    // important for profiling. We insert it here so as to be before any control
    // flow inside the bytecode.
    ++profile_counter_advance;
    if (!info.only_falls_through()) {
      inserter.Create<AdvanceProfileCounterInst>(profile_counter_advance);
      profile_counter_advance = 0;
    }

    // Iterate over all generated terminators inserting branch arguments.
    // This must happen after the translate function so that terminators
    // added by the translate function are covered and so that the addresses
    // for Bind are not invalidated.
    for (Block& b : MakeRange(blocks_begin, blocks_end)) {
      S6_CHECK(b.GetTerminator()) << FormatOrDie(b) << " has no terminator";
      if (auto* ji = dyn_cast<UnconditionalTerminatorInst>(b.GetTerminator())) {
        ssa.InsertBranchArguments(ji);
      } else if (auto* bi = dyn_cast<BrInst>(b.GetTerminator())) {
        ssa.InsertBranchArguments(bi);
      }
    }

    // Iterate over all instructions adding precise bytecode offsets.
    for (Block& b : MakeRange(blocks_begin, blocks_end)) {
      for (Instruction& i : b) {
        if (auto* pi = PreciseLocationInst::Get(&i))
          *pi->mutable_bytecode_offset() = inst.program_offset();
      }
    }
  }

  S6_CHECK_OK(VerifyFunction(f)) << FormatOrDie(f);
  // Complete! \o/
  return absl::OkStatus();
}

absl::StatusOr<Function> IngestProgram(
    absl::Span<const BytecodeInstruction> program, absl::string_view name,
    int64_t num_fastlocals, int64_t num_arguments, bool except_observed) {
  Function f(name);
  S6_RETURN_IF_ERROR(IngestProgram(program, f, num_fastlocals, num_arguments,
                                   except_observed));
  return f;
}

std::string SourceLocationAnnotator::FormatAnnotation(
    const Value& v, FormatContext* ctx) const {
  const PreciseLocationInst* pi = PreciseLocationInst::Get(&v);
  if (!pi) return {};
  std::string filename = PyObjectToString(code_->co_filename);
  auto inst =
      bytecode_insts_[PcValue::FromOffset(pi->bytecode_offset()).AsIndex()];
  int line_number = PyCode_Addr2Line(code_, pi->bytecode_offset());
  return absl::StrCat(BytecodeOpcodeToString(inst.opcode()), " ",
                      file::Basename(filename), ":", line_number);
}

}  // namespace deepmind::s6
