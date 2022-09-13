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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/statusor.h"
#include "code_object.h"
#include "core_util.h"
#include "interpreter.h"
#include "strongjit/base.h"
#include "strongjit/builder.h"
#include "strongjit/formatter.h"
#include "strongjit/ssa.h"
#include "strongjit/util.h"
#include "type_feedback.h"

namespace deepmind::s6 {

// The ingestion of CPython bytecodes is a multi-stage process. We generate
// strongjit IR directly in SSA form with no value stack.
//
//   1) Every bytecode has an "Analyze" function. This takes an AnalysisContext
//      and describes the bytecode's interactions with the value stack (pushes
//      / pops) and generates control flow instructions (branches / jumps).
//   2) With the stack information, defined / live-in sets can be created and
//      PHI insert points computed. Block arguments are inserted.
//   3) Every bytecode has a "Translate" function. This takes a TranslateContext
//      and generates the actual code. This interacts with an abstract value
//      stack inside TranslateContext that rewrites pushes and pops on the
//      fly using the pruned SSA from 2).

// Forward declare implementation details.
class BytecodeInfo;
class IngestionContext;

// Provides the interface for Analyze functions.
class AnalysisContext {
 public:
  AnalysisContext(IngestionContext* ctx, BytecodeInfo* info,
                  SsaBuilder::BitVector* defines);

  // Records that this op pushes an item to the value stack.
  AnalysisContext& Push();

  // Records that this op pops an item from the value stack.
  AnalysisContext& Pop();

  // Records that this op drops n items from the value stack. n can be zero.
  AnalysisContext& Drop(int64_t n);

  // Records that this op does not fall through to the next bytecode
  // instruction.
  AnalysisContext& DoesNotFallthrough();

  // Records that this op may (or will) jump to the given target. The
  // target offset inherits the current stack height and handler stack.
  AnalysisContext& MayJumpTo(PcValue target);

  // Records that this op may (or will) jump to a finally handler at the
  // given target. Also specifies the reason for going to the finally handler
  AnalysisContext& MayJumpToFinally(PcValue target, WhyFlags why);

  // Binds a Value that must be updated by the Translate function. The
  // bound values are pushed to an ordered list.
  //
  // Care must be taken to not change the operand list sizes of the instruction
  // between the analysis and the translate function so that the
  // pointer stays valid.
  AnalysisContext& Bind(Value** value_ptr);

  // Records that this op defines the given fast local.
  AnalysisContext& DefFastLocal(int64_t index);

  // Returns the entry block for a bytecode instruction by PcValue.
  Block* BlockAt(PcValue target);

  // Pushes a new TryHandler::kExcept block to the handler stack.
  void PushExcept(PcValue target);

  // Pushes a new TryHandler::kFinally block to the handler stack.
  void PushFinally(PcValue target);

  // Pushes a new TryHandler::kLoop block to the handler stack.
  void PushLoop(PcValue exit);

  // Declares a current loop continue target for a parent finally handler.
  // handler must be of kind `kFinally`.
  void SetContinueTarget(const TryHandler& handler, PcValue target);

  // Get the continue Target when inside a finally handler. That
  // value must have been set with SetContinueTarget while inside the
  // original kFinally block.
  // We must be inside the corresponding kFinallyHandler block.
  absl::optional<PcValue> GetContinueTarget();

  // Gives the possible reasons we entered the current finally handler.
  // The top of the block stack must be a kFinallyHandler.
  WhyFlags GetWhyFinally();

  // Pops the last TryHandler block from the handler stack. The stack height
  // is set to the handler's height, but no stack adjustment takes place.
  TryHandler PopHandler();

  // Unwinds the handler stack for the given unwind reason. The handler
  // containing the next block to jump to, if any, is returned. Otherwise,
  // nullopt is returned.
  //
  // kWhyReturn can only return finally handlers.
  // kWhyBreak and kWhyContinue can return loop or finally handlers.
  absl::optional<TryHandler> UnwindHandlerStack(Why why);

  // Returns the top handler to jump to in the case of an exception. This could
  // either be a kExcept or kFinally handler (or no handler).
  absl::optional<TryHandler> GetTopExceptionHandler();

  // Accessors only to be used by IngestProgram.
  bool falls_through() const { return falls_through_; }
  bool only_falls_through() const { return only_falls_through_; }
  int64_t stack_height() const { return stack_height_; }
  const absl::InlinedVector<TryHandler, 4>& try_handlers() const {
    return try_handlers_;
  }

  // Returns the top (most recently pushed) handler in the block stack.
  const TryHandler& GetTopHandler() const {
    S6_CHECK(!try_handlers().empty());
    return try_handlers().back();
  }

 private:
  IngestionContext& ctx_;
  BytecodeInfo& info_;
  SsaBuilder::BitVector& defines_;
  int64_t stack_height_;
  bool falls_through_ = true;
  bool only_falls_through_ = true;
  absl::InlinedVector<TryHandler, 4> try_handlers_;
};

// Provides the interface for a Translate function.
class TranslateContext {
 public:
  TranslateContext(IngestionContext* ctx, BytecodeInfo* info, SsaBuilder* ssa);

  // Pops from the value stack.
  Value* Pop();

  // Drops n items from the value stack. n can be zero.
  void Drop(int64_t n);

  // Reads a value from the value stack. The top of stack (TOS) pointer points
  // one past the stored top value. So the TOS element exists at Peak(1). Thus
  // Peak(0) is invalid.
  Value* Peek(int64_t i);

  // Pushes to the value stack.
  void Push(Value* v);

  // Defines the given fast local to the given value. The value must not be
  // nullptr.
  void DefFastLocal(int64_t index, Value* v);

  // Reads the value of the given fast local.
  Value* UseFastLocal(int64_t index);

  // Returns the entry block for a bytecode instruction by PcValue.
  Block* BlockAt(PcValue target);

  // Binds `v` to the next binding given to AnalysisContext::Bind.
  void Bind(Value* v);

  // Emits code to pop an exception handler. This code decrefs the stack down
  // to handler.stack_height() except for the last three items which are used to
  // restore the exception state.
  void PopExceptHandler(Builder& b, const TryHandler& handler);

  // Emits code to pop a finally handler. This code decrefs the stack down
  // to handler.stack_height() except for the last six items.
  // The least significant bit of the top of these 6 items is a
  // discriminator that will determine if we are in exception or not.
  // if we are in an exception then it will behave as a PopExceptHandler,
  // if not it will just pop the last 6 items on the stack.
  //
  // If the returned boolean is true, the block below the finally handler
  // is a kExceptHandler which should be silently poped/discarded without
  // decrefing its value or any other usual operations for except handlers.
  bool PopFinallyHandler(Builder& b, const TryHandler& handler);

  // Pops a normal block.
  void PopBlock(Builder& b, const TryHandler& handler) {
    PopAndDecrefStackTo(b, handler.stack_height());
  }

  // Pops all stack values between the current stack height and `stack_height`.
  // Decref are always generated as possibly being null.
  void PopAndDecrefStackTo(Builder& b, int64_t stack_height);

  // Unwinds the handler stack for the given unwind reason. The handler
  // containing the next block to jump to, if any, is returned. Otherwise,
  // nullopt is returned.
  //
  // All reasons can return a finally handler. Additionally,
  // kWhyExcept can return except handlers.
  // kWhyBreak and kWhyContinue can return loop handlers.
  //
  // Starts from the current stack height so blocks above
  // the current stack height are considered as already popped.
  absl::optional<TryHandler> UnwindHandlerStack(Builder& b, Why why);

  // Emits code to unwind the stack and take an exception.
  //
  // Returns a boolean telling if the exception goes out of the function
  // and thus should be deoptimized. This is a performance hint that can be
  // ignored.
  //
  // It is better to use `ExceptIf` or `ExceptConditional` when possible
  // as they manage the deoptimize hint properly and silently.
  bool Except(Builder& b);

  // Conditionally unwinds the stack and takes an exception.
  void ExceptIf(Builder& b, Value* condition);

  // Conditionally unwinds the stack and takes an exception.
  // Allows to add some extra clean-up code if the condition is true,
  // before taking the exception.
  //
  // All stack manipulations done in `cleanup` are reverted.
  // The stack height is not changed by this function even
  // if `cleanup` pops some values.
  void ExceptConditional(
      Builder& b, Value* condition,
      absl::FunctionRef<Builder::DoesNotReturn(Builder& b)> cleanup);

  // Releases a reference to all fast locals. This should be emitted prior to
  // return or except.
  void DecrefFastLocals(Builder& b);

  // If the given value is nullptr, deoptimizes. This inserts a
  // DeoptimizeIfSafepointInst and ingestion will fill it in with the *current*
  // instruction's safepoint data.

  void DeoptimizeIfNull(Builder& b, Value* v, absl::string_view reason) {
    deoptimize_if_safepoints_.push_back(
        b.DeoptimizeIfSafepoint(v, /*negated=*/true, reason));
  }

  void DeoptimizeIf(Builder& b, Value* condition, absl::string_view reason) {
    deoptimize_if_safepoints_.push_back(
        b.DeoptimizeIfSafepoint(condition, /*negated=*/false, reason));
  }

  // Unboxes the boxed value, and deoptimizes on incorrect type.
  UnboxInst* UnboxOrDeoptimize(Builder& b, UnboxableType type, Value* boxed) {
    UnboxInst* op = b.inserter().Create<UnboxInst>(type, boxed);
    // If the boxed operand was not of the correct type, this is signalled
    // as an overflow condition.
    deoptimize_if_safepoints_.push_back(b.DeoptimizeIfOverflow(op, nullptr));
    return op;
  }

  // Returns the possible reasons we entered the current finally handler.
  // The top of the block stack must be a kFinallyHandler.
  WhyFlags GetWhyFinally();

  // Returns the top (most recently pushed) handler in the block stack.
  const TryHandler& GetTopHandler() const;

  // Returns the list of DeoptimizeIfSafepoint instructions inserted but not
  // filled in.
  const std::vector<DeoptimizeIfSafepointInst*>& deoptimize_if_safepoints()
      const {
    return deoptimize_if_safepoints_;
  }

  // Obtains the type feedback for this bytecode, if available.
  ClassDistributionSummary GetTypeFeedback() const;

  int64_t stack_height() { return stack_height_; }

 private:
  IngestionContext& ctx_;
  BytecodeInfo& info_;
  SsaBuilder& ssa_;
  int64_t stack_height_;
  int64_t next_binding_ = 0;
  std::vector<DeoptimizeIfSafepointInst*> deoptimize_if_safepoints_;
};

// The type of an opcode handler function for the Analysis phase.
using AnalyzeFunction = absl::FunctionRef<void(const BytecodeInstruction&,
                                               Builder, AnalysisContext*)>;

// The type of an opcode handler function for the Translate phase.
using TranslateFunction = absl::FunctionRef<void(const BytecodeInstruction&,
                                                 Builder, TranslateContext*)>;

// Creates a strongjit Function from a list of bytecode instructions.
// If except_observed is false, the program will be compiled assuming exceptions
// are rare; exception paths will be deoptimized.
absl::StatusOr<Function> IngestProgram(
    absl::Span<const BytecodeInstruction> program, absl::string_view name,
    int64_t num_fastlocals, int64_t num_arguments,
    bool except_observed = false);

// Ingests into the given the Function.
// If except_observed is false, the program will be compiled assuming exceptions
// are rare; exception paths will be deoptimized.
absl::Status IngestProgram(absl::Span<const BytecodeInstruction> program,
                           Function& f, int64_t num_fastlocals,
                           int64_t num_arguments, bool except_observed = false);

// An AnnotationFormatter that annotates PreciseLocationInsts with their
// bytecode instruction and source location.
class SourceLocationAnnotator final : public AnnotationFormatter {
 public:
  ~SourceLocationAnnotator() override {}
  explicit SourceLocationAnnotator(PyCodeObject* code)
      : bytecode_insts_(ExtractInstructions(code)), code_(code) {}

  std::string FormatAnnotation(const Value& v,
                               FormatContext* ctx) const override;

 private:
  std::vector<BytecodeInstruction> bytecode_insts_;
  PyCodeObject* code_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_H_
