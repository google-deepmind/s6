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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BUILDER_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BUILDER_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "event_counters.h"
#include "global_intern_table.h"
#include "strongjit/base.h"
#include "strongjit/block.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {

// Makes it easy to build Strongjit IR. Builders can either be attached to
// existing blocks with an insert point or directly onto a function.
//
// The latter is more useful in tests than in practice - it will create an
// initial entry block ending in `unreachable` and insert just before it.
//
// The Conditional functions create control flow. Each arm of the generated
// flow is generated using a new builder, and blocks are split and inserted
// correctly at the insertion point.
class Builder {
 public:
  using ValueList = absl::InlinedVector<Value*, 8>;

  // Creates the initial scaffold of an entry block ending in unreachable for
  // an empty function and inserts before the unreachable.
  explicit Builder(Function* f) : b_(f->CreateBlock(), Block::iterator()) {
    S6_CHECK_EQ(f->num_blocks(), 1) << "Function must be empty!";
    Block* block = b_.block();
    block->Create<UnreachableInst>();
    b_ = BlockInserter(block, block->begin());
  }

  // Starts building at a particular insertion point in `block`.
  Builder(Block* block, Block::iterator insert_pt) : b_(block, insert_pt) {}

  // Starts building at the point designated by a `BlockInserter`.
  explicit Builder(BlockInserter b) : b_(b) {}

  // Starts building at the end of `b`.
  // TODO: Replace with Builder::FromEnd()
  explicit Builder(Block* b) : b_(b, b->end()) {}

  static Builder FromStart(Block* b) { return Builder(b, b->begin()); }

  // Typesafe way to indicate that a conditional arm does not return. Ideally
  // we would use `void` but implicit conversions make this hard.
  struct DoesNotReturn {};

  // Creates a one-armed conditional. `if_true` creates the true arm
  // of the conditional, otherwise control falls through. The true
  // arm will not merge with the fallthrough block. `if_true`
  // must end the block by inserting its own terminator instruction.
  BrInst* Conditional(Value* condition,
                      absl::FunctionRef<DoesNotReturn(Builder)> if_true) {
    Function* f = b_.block()->parent();
    Block* this_block = b_.block();
    Block* fallthrough = this_block->Split(b_.insert_point());
    Block* if_block = f->CreateBlock(fallthrough->GetIterator());
    this_block->GetTerminator()->erase();
    BrInst* br = this_block->Create<BrInst>(condition, if_block, fallthrough);

    // Note that if_block deliberately does not have a terminator. This overload
    // expects the user to fill in their own terminator.
    if_block->AddPredecessor(this_block);
    fallthrough->AddPredecessor(this_block);
    b_ = BlockInserter(fallthrough, fallthrough->begin());

    if_true(Builder(if_block, if_block->begin()));

    S6_CHECK(if_block->GetTerminator()) << "`if_true` must insert a Terminator";

    return br;
  }

  // Creates a one-armed conditional. `if_true` creates the true arm
  // of the conditional, otherwise control falls through. The true
  // arm will jump back to the fallthrough. The true arm returns a
  // list of Values that are merged with `false_values` in the fallthrough.
  //
  // The returned value list may be empty.
  ValueList Conditional(Value* condition,
                        absl::FunctionRef<ValueList(Builder)> if_true,
                        ValueList false_values = {},
                        BrInst** out_branch = nullptr) {
    Function* f = b_.block()->parent();
    Block* this_block = b_.block();
    Block* fallthrough = this_block->Split(b_.insert_point());
    Block* if_block = f->CreateBlock(fallthrough->GetIterator());
    JmpInst* jmp = if_block->Create<JmpInst>(fallthrough);
    this_block->GetTerminator()->erase();
    BrInst* br = this_block->Create<BrInst>(
        condition, if_block, absl::Span<Value* const>{}, fallthrough,
        absl::Span<Value* const>(false_values));
    if_block->AddPredecessor(this_block);

    fallthrough->AddPredecessor(if_block);
    fallthrough->AddPredecessor(this_block);
    b_ = BlockInserter(fallthrough, fallthrough->begin());

    ValueList true_values = if_true(Builder(if_block, if_block->begin()));
    ValueList parameters;
    S6_CHECK_EQ(true_values.size(), false_values.size());
    for (Value* v : true_values) {
      jmp->AddArgument(v);
      parameters.push_back(fallthrough->CreateBlockArgument());
    }
    if (out_branch) *out_branch = br;
    return parameters;
  }

  // Creates a two-armed conditional. `if_true` creates the true arm
  // of the conditional, `if_false` creates the false arm. Both must
  // return equally sized ValueLists which may be empty. These will
  // be merged in the fallthrough block. The created branch instruction
  // is optionally returned in `out_branch` if not null.
  ValueList Conditional(Value* condition,
                        absl::FunctionRef<ValueList(Builder)> if_true,
                        absl::FunctionRef<ValueList(Builder)> if_false,
                        BrInst** out_branch = nullptr) {
    Function* f = b_.block()->parent();
    Block* this_block = b_.block();
    Block* fallthrough = this_block->Split(b_.insert_point());
    Block* if_block = f->CreateBlock(fallthrough->GetIterator());
    Block* else_block = f->CreateBlock(fallthrough->GetIterator());

    JmpInst* if_jmp = if_block->Create<JmpInst>(fallthrough);
    JmpInst* else_jmp = else_block->Create<JmpInst>(fallthrough);

    this_block->GetTerminator()->erase();
    BrInst* br = this_block->Create<BrInst>(condition, if_block, else_block);
    if_block->AddPredecessor(this_block);
    else_block->AddPredecessor(this_block);

    fallthrough->AddPredecessor(if_block);
    fallthrough->AddPredecessor(else_block);
    fallthrough->RemovePredecessor(this_block);
    b_ = BlockInserter(fallthrough, fallthrough->begin());

    ValueList true_values = if_true(Builder(if_block, if_block->begin()));
    ValueList false_values = if_false(Builder(else_block, else_block->begin()));
    S6_CHECK_EQ(true_values.size(), false_values.size());

    ValueList parameters;
    for (Value* v : true_values) {
      if_jmp->AddArgument(v);
      parameters.push_back(fallthrough->CreateBlockArgument());
    }
    for (Value* v : false_values) {
      else_jmp->AddArgument(v);
    }
    if (out_branch) *out_branch = br;
    return parameters;
  }

  // Returns a value that is `if_true` if `condition` is nonzero,
  // otherwise `if_false`.
  Value* Select(Value* condition, Value* if_true, Value* if_false) {
    return Conditional(
               condition, [&](Builder b) { return ValueList{if_true}; },
               [&](Builder b) { return ValueList{if_false}; })
        .front();
  }

  // Jumps to the given exception handler.
  ExceptInst* Except(Block* target = nullptr) {
    ExceptInst* inst = b_.Create<ExceptInst>(target);
    if (target) target->AddPredecessor(b_.block());
    return inst;
  }

  void IncrementEventCounter(const std::string& name) {
    // Ensure that the trace counter exists, so that counters that are never
    // eventually used are still output.
    (void)EventCounters::Instance().GetEventCounter(name);
    b_.Create<IncrementEventCounterInst>(
        GlobalInternTable::Instance().Intern(name));
  }

  void TraceBegin(GlobalInternTable::InternedString name) {
    b_.Create<TraceBeginInst>(name);
  }

  void TraceBegin(absl::string_view name) {
    b_.Create<TraceBeginInst>(GlobalInternTable::Instance().Intern(name));
  }

  void TraceEnd(GlobalInternTable::InternedString name) {
    b_.Create<TraceEndInst>(name);
  }

  void TraceEnd(absl::string_view name) {
    b_.Create<TraceEndInst>(GlobalInternTable::Instance().Intern(name));
  }

  DecrefInst* DecrefNotNull(Value* v, int64_t bytecode_offset = 0) {
    return b_.Create<DecrefInst>(Nullness::kNotNull, v, bytecode_offset);
  }
  void DecrefNotNull(absl::Span<Value* const> values,
                     int64_t bytecode_offset = 0) {
    for (auto v : values) DecrefNotNull(v, bytecode_offset);
  }
  DecrefInst* DecrefOrNull(Value* v, int64_t bytecode_offset = 0) {
    return b_.Create<DecrefInst>(Nullness::kMaybeNull, v, bytecode_offset);
  }
  void DecrefOrNull(absl::Span<Value* const> values,
                    int64_t bytecode_offset = 0) {
    for (auto v : values) DecrefOrNull(v, bytecode_offset);
  }

  IncrefInst* IncrefNotNull(Value* v) {
    return b_.Create<IncrefInst>(Nullness::kNotNull, v);
  }
  IncrefInst* IncrefOrNull(Value* v) {
    return b_.Create<IncrefInst>(Nullness::kMaybeNull, v);
  }

  // Creates a JmpInst to `b`, optionally with arguments. Updates the
  // predecessor list of `b`.
  JmpInst* Jmp(Block* b, absl::Span<Value* const> args = {}) {
    S6_CHECK_EQ(b_.block()->GetTerminator(), nullptr);
    b->AddPredecessor(b_.block());
    return b_.Create<JmpInst>(b, args);
  }

  // Creates a BrInst to `if_true` or `if_false` based on `condition`. Updates
  // the predecessor list of both target blocks.
  BrInst* Br(Value* condition, Block* if_true, Block* if_false) {
    S6_CHECK_EQ(b_.block()->GetTerminator(), nullptr);
    if_true->AddPredecessor(b_.block());
    if_false->AddPredecessor(b_.block());
    return b_.Create<BrInst>(condition, if_true, if_false);
  }

  // Creates a BrInst to `if_true` or `if_false` based on `condition`. Updates
  // the predecessor list of both target blocks.
  BrInst* Br(Value* condition, Block* if_true,
             absl::Span<Value* const> if_true_args, Block* if_false,
             absl::Span<Value* const> if_false_args) {
    S6_CHECK_EQ(b_.block()->GetTerminator(), nullptr);
    if_true->AddPredecessor(b_.block());
    if_false->AddPredecessor(b_.block());
    return b_.Create<BrInst>(condition, if_true, if_true_args, if_false,
                             if_false_args);
  }

  // Creates a BrInst that jumps to `if_true` or stays on the main path
  // depending on `condition`. After this function returns, the Builder's
  // insertion point is at the start of the fallthrough (else) block.
  BrInst* BrFallthrough(Value* condition, Block* if_true) {
    Block* this_block = b_.block();
    Block* fallthrough = this_block->Split(b_.insert_point());
    this_block->GetTerminator()->erase();
    b_ = BlockInserter(this_block, this_block->end());
    BrInst* br = Br(condition, if_true, fallthrough);
    b_ = BlockInserter(fallthrough, fallthrough->begin());
    return br;
  }

  // Creates a BrInst that jumps to `if_true` or stays on the main path
  // depending on `condition`. After this function returns, the Builder's
  // insertion point is at the start of the fallthrough (else) block.
  BrInst* BrFallthrough(Value* condition, Block* if_true,
                        absl::Span<Value* const> if_true_args) {
    Block* this_block = b_.block();
    Block* fallthrough = this_block->Split(b_.insert_point());
    this_block->GetTerminator()->erase();
    b_ = BlockInserter(this_block, this_block->end());
    BrInst* br = Br(condition, if_true, if_true_args, fallthrough, {});
    b_ = BlockInserter(fallthrough, fallthrough->begin());
    return br;
  }

  // Performs a short-circuiting `and`/`&&` operation.
  // Expects the two conditionals to return Int64(1)/Int64(0).
  // Returns an Int64(1)/Int64(0).
  Value* ShortcircuitAnd(absl::FunctionRef<Value*(Builder)> condition1,
                         absl::FunctionRef<Value*(Builder)> condition2) {
    ValueList and_results = Conditional(
        condition1(*this), [&](Builder b) { return ValueList{condition2(b)}; },
        [&](Builder b) { return ValueList{b.Int64(0)}; });
    return and_results.front();
  }

  // Performs a short-circuiting `or`/`||` operation.
  // Expects the two conditionals to return Int64(1)/Int64(0).
  // Returns an Int64(1)/Int64(0).
  Value* ShortcircuitOr(absl::FunctionRef<Value*(Builder)> condition1,
                        absl::FunctionRef<Value*(Builder)> condition2) {
    ValueList or_results = Conditional(
        condition1(*this), [&](Builder b) { return ValueList{b.Int64(1)}; },
        [&](Builder b) { return ValueList{condition2(b)}; });
    return or_results.front();
  }

  // Allows to insert multiple parallel branches at the current point:
  // It returns a Block pointer that may be used as a target
  // for jumping instructions. It is particalarly useful
  // in combination with BrFallthrough or Conditional and Jmp.
  // Once all the parallel branches are written, EndSplit must be called to
  // jump back to the block after the split.
  // The number of arguments used in the various jumps to the end block
  // must match the number of arguments given to EndSplit.
  //
  // For example the following code:
  //
  // Block* end = b.Split()
  // body1...
  // b.BrFallthrough(cond1, end, {res1})
  // body2...
  // b.Conditional(cond2, [&](Builder b){
  //   body3...
  //   b.Jmp(end, {res2});
  //   return Builder::DoNotReturn;
  //   });
  // body4...
  // Value* res = b.EndSplit(end, res3);
  // using res...
  //
  // will generate code equivalent to:
  //
  // body1...
  // if(cond1) goto end[res1];
  // body2...
  // if(cond2){
  //   body3...
  //   goto end [res2];
  // }
  // body4...
  // goto end [res3];
  // end: [res]
  // using res...
  //
  // For another example, look at the code of Switch.
  Block* Split() {
    Block* end = block()->Split(insert_point());
    b_ = BlockInserter(block(), Block::iterator(block()->GetTerminator()));
    return end;
  }

  // Ends a split opened with Split(), cf the documentation of Split.
  //
  // The value in `args` are passed to the `end` block in the fallthrough case.
  // All other jumps to the `end` block must have used the same number of
  // arguments.
  // The returned list is of the same size as `args` and is the set of
  // block arguments inside the `end` block.
  // The builder is now at the start of the `end` block and thus construction
  // can continue as normal after the split.
  //
  // The Block `end` parameter must have been obtained by Split().
  // The Split/EndSplit pairs must be matched to each other in a
  // well-parenthesised manner.
  // For example, this code is forbidden:
  // Value* a = Split();
  // Value* b = Split();
  // EndSplit(a);
  // EndSplit(b);
  ValueList EndSplit(Block* end, absl::Span<Value* const> args = {}) {
    S6_CHECK(block()->GetTerminator()) << "Origin block must have a terminator";
    S6_CHECK(isa<JmpInst>(block()->GetTerminator()))
        << "Origin block must end with a JmpInst";
    JmpInst* jmp = cast<JmpInst>(block()->GetTerminator());

    S6_CHECK(jmp->arguments().empty())
        << "Current block terminator instruction must not have arguments";
    S6_CHECK(end->block_arguments_empty())
        << "target block must not have arguments";

    b_ = BlockInserter(end, end->begin());
    ValueList parameters;
    for (Value* v : args) {
      jmp->AddArgument(v);
      parameters.push_back(end->CreateBlockArgument());
    }
    return parameters;
  }

  // Emits a chain of if/elses all branching to the same destination block. For
  // each T in `ts`, `cond(t)` and `body(t)` are invoked. If the final `cond(t)`
  // fails, the function is deoptimized.
  //
  // if (cond1()) {
  //   v = body1();
  //   goto end [v];
  // }
  // if (cond2()) {
  //   v = body2();
  //   goto end [v];
  // }
  //
  // deoptimize_if (!cond3()) "reason";
  // v = body3();
  // goto end [v];
  //
  // end: [retval]
  //
  // Returns `retval`.
  template <typename T>
  Value* Switch(absl::Span<T const> ts, SafepointInst* safepoint,
                absl::FunctionRef<Value*(Builder, T)> cond,
                absl::FunctionRef<Value*(Builder, T)> body,
                absl::string_view reason) {
    Block* end = Split();
    for (const T& t : ts.first(ts.size() - 1)) {
      Conditional(cond(*this, t), [&](Builder builder) {
        Value* v = body(builder, t);
        if (v) {
          builder.Jmp(end, {v});
        } else {
          builder.Jmp(end);
        }
        return Builder::DoesNotReturn();
      });
    }

    const T& t = ts.back();
    DeoptimizeIfSafepoint(cond(*this, t), /*negated=*/true, reason, safepoint);
    Value* v = body(*this, t);

    if (v) {
      return EndSplit(end, {v}).front();
    } else {
      EndSplit(end);
      return nullptr;
    }
  }

  // Integer addition; no overflow check.
  AddInst* Add(Value* lhs, Value* rhs) {
    return b_.Create<AddInst>(NumericInst::kInt64, lhs, rhs);
  }

  // Integer or float addition, with optional overflow check in the integer
  // case.
  AddInst* Add(NumericInst::NumericType type, Value* lhs, Value* rhs,
               SafepointInst* safepoint = nullptr) {
    AddInst* op = b_.Create<AddInst>(type, lhs, rhs);
    if (type == NumericInst::kInt64 && safepoint)
      DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer or float subtraction, with optional overflow check in the integer
  // case.
  SubtractInst* Subtract(NumericInst::NumericType type, Value* lhs, Value* rhs,
                         SafepointInst* safepoint = nullptr) {
    SubtractInst* op = b_.Create<SubtractInst>(type, lhs, rhs);
    if (type == NumericInst::kInt64 && safepoint)
      DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer or float negation, with overflow check in the integer case.
  NegateInst* Negate(NumericInst::NumericType type, Value* operand,
                     SafepointInst* safepoint) {
    NegateInst* op = b_.Create<NegateInst>(type, operand);
    if (type == NumericInst::kInt64) DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer or float multiplication, with overflow check in the integer case.
  MultiplyInst* Multiply(NumericInst::NumericType type, Value* lhs, Value* rhs,
                         SafepointInst* safepoint) {
    MultiplyInst* op = b_.Create<MultiplyInst>(type, lhs, rhs);
    if (type == NumericInst::kInt64) DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer or float division, with overflow check in the integer case.
  DivideInst* Divide(NumericInst::NumericType type, Value* lhs, Value* rhs,
                     SafepointInst* safepoint) {
    DeoptimizeIfDivisionByZero(type, rhs, safepoint);
    DivideInst* op = b_.Create<DivideInst>(type, lhs, rhs);
    if (type == NumericInst::kInt64) DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer remainder with overflow check.
  RemainderInst* Remainder(NumericInst::NumericType type, Value* lhs,
                           Value* rhs, SafepointInst* safepoint) {
    DeoptimizeIfDivisionByZero(type, rhs, safepoint);
    RemainderInst* op = b_.Create<RemainderInst>(type, lhs, rhs);
    if (type == NumericInst::kInt64) DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Integer left shift with optional overflow check.
  ShiftLeftInst* ShiftLeft(Value* lhs, Value* rhs,
                           SafepointInst* safepoint = nullptr) {
    DeoptimizeIfNegativeShift(rhs, safepoint);
    ShiftLeftInst* op = b_.Create<ShiftLeftInst>(NumericInst::kInt64, lhs, rhs);
    if (safepoint) {
      DeoptimizeIfOverflow(op, safepoint);
    }
    return op;
  }

  // Signed integer right shift; cannot overflow.
  ShiftRightSignedInst* ShiftRightSigned(Value* lhs, Value* rhs,
                                         SafepointInst* safepoint) {
    DeoptimizeIfNegativeShift(rhs, safepoint);
    return b_.Create<ShiftRightSignedInst>(NumericInst::kInt64, lhs, rhs);
  }

  AndInst* And(Value* lhs, Value* rhs) {
    return b_.Create<AndInst>(NumericInst::kInt64, lhs, rhs);
  }

  OrInst* Or(Value* lhs, Value* rhs) {
    return b_.Create<OrInst>(NumericInst::kInt64, lhs, rhs);
  }

  XorInst* Xor(Value* lhs, Value* rhs) {
    return b_.Create<XorInst>(NumericInst::kInt64, lhs, rhs);
  }

  NotInst* Not(Value* operand) {
    return b_.Create<NotInst>(NumericInst::kInt64, operand);
  }

  SextInst* Sext(Value* v) {
    return b_.Create<SextInst>(NumericInst::kInt64, v);
  }

  CompareInst* IsNegative(Value* v) {
    return IsLessThan(NumericInst::kInt64, v, b_.Create<ConstantInst>(0));
  }

  CompareInst* IsEqual(Value* v1, Value* v2) {
    return IsEqual(NumericInst::kInt64, v1, v2);
  }

  CompareInst* IsNotEqual(Value* v1, Value* v2) {
    return IsNotEqual(NumericInst::kInt64, v1, v2);
  }

  CompareInst* IsEqual(NumericInst::NumericType type, Value* v1, Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kEqual, type, v1, v2);
  }

  CompareInst* IsNotEqual(NumericInst::NumericType type, Value* v1, Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kNotEqual, type, v1, v2);
  }

  CompareInst* IsLessThan(NumericInst::NumericType type, Value* v1, Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kLessThan, type, v1, v2);
  }

  CompareInst* IsLessEqual(NumericInst::NumericType type, Value* v1,
                           Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kLessEqual, type, v1, v2);
  }

  CompareInst* IsGreaterThan(NumericInst::NumericType type, Value* v1,
                             Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kGreaterThan, type, v1, v2);
  }

  CompareInst* IsGreaterEqual(NumericInst::NumericType type, Value* v1,
                              Value* v2) {
    return b_.Create<CompareInst>(CompareInst::kGreaterEqual, type, v1, v2);
  }

  CompareInst* IsZero(Value* v) {
    return IsEqual(v, b_.Create<ConstantInst>(0));
  }

  CompareInst* IsNotZero(Value* v) {
    return IsNotEqual(v, b_.Create<ConstantInst>(0));
  }

  ConstantInst* Int64(int64_t c) { return b_.Create<ConstantInst>(c); }

  ConstantInst* Bool(bool b) { return b_.Create<ConstantInst>(b ? 1 : 0); }

  ConstantInst* Zero() { return b_.Create<ConstantInst>(0); }

  ConstantInst* Constant(const void* c) {
    return b_.Create<ConstantInst>(reinterpret_cast<int64_t>(c));
  }

  BoxInst* Box(UnboxableType type, Value* content) {
    return b_.Create<BoxInst>(type, content);
  }

  // Unboxes the boxed value, and deoptimises (by branching to a clone of
  // `safepoint`) on integer overflow or incorrect type.
  UnboxInst* Unbox(UnboxableType type, Value* boxed, SafepointInst* safepoint) {
    UnboxInst* op = b_.Create<UnboxInst>(type, boxed);
    DeoptimizeIfOverflow(op, safepoint);
    return op;
  }

  // Calls a CallNativeInst::Callee.
  CallNativeInst* Call(Callee callee, absl::Span<Value* const> arguments,
                       int32_t bytecode_offset = 0) {
    return b_.Create<CallNativeInst>(callee, arguments, bytecode_offset);
  }

  // Calls an value as an indirect call.
  CallNativeIndirectInst* CallIndirect(Value* callee,
                                       absl::Span<Value* const> arguments,
                                       int32_t bytecode_offset = 0) {
    return b_.Create<CallNativeIndirectInst>(callee, arguments,
                                             bytecode_offset);
  }

  // Calls an value as an indirect call with static information about the
  // callee.
  CallNativeIndirectInst* CallIndirect(Value* callee,
                                       absl::Span<Value* const> arguments,
                                       const CalleeInfo& info,
                                       int32_t bytecode_offset = 0) {
    return b_.Create<CallNativeIndirectInst>(callee, arguments, info,
                                             bytecode_offset);
  }

  // Calls a Python callable object.
  CallPythonInst* CallPython(Value* callee, absl::Span<Value* const> arguments,
                             Value* keywords = nullptr,
                             int32_t bytecode_offset = 0) {
    return b_.Create<CallPythonInst>(callee, arguments, keywords,
                                     bytecode_offset);
  }
  CallPythonInst* CallConstantAttribute(const Class* cls,
                                        absl::string_view attribute_name,
                                        absl::Span<Value* const> arguments,
                                        Value* keywords = nullptr,
                                        int32_t bytecode_offset = 0) {
    // CallPython steals the callee so this cannot be written as
    // CallPython(ConstantAttribute). The constant attribute must be increffed.
    ConstantAttributeInst* attribute = ConstantAttribute(cls, attribute_name);
    IncrefNotNull(attribute);
    return CallPython(attribute, arguments, keywords, bytecode_offset);
  }

  CallAttributeInst* CallAttribute(Value* receiver, absl::string_view attr_str,
                                   absl::Span<Value* const> arguments,
                                   Value* names, int32_t bytecode_offset,
                                   int32_t call_python_bytecode_offset) {
    return b_.Create<CallAttributeInst>(
        receiver,
        b_.block()->parent()->GetStringTable()->InternString(attr_str),
        arguments, names, bytecode_offset, call_python_bytecode_offset);
  }

  // Calls a C function with Python's METH_FASTCALL calling convention.
  CallVectorcallInst* CallVectorcall(Value* callee, Value* self, Value* names,
                                     absl::Span<Value* const> arguments,
                                     int32_t bytecode_offset = 0) {
    return b_.Create<CallVectorcallInst>(callee, self, names, arguments,
                                         bytecode_offset);
  }

  RematerializeInst* Rematerialize(Callee callee,
                                   absl::Span<Value* const> arguments) {
    return b_.Create<RematerializeInst>(callee, arguments);
  }

  // Creates a FrameVariableInst to access a frame variable. `index` is used by
  // some frame_variable kinds (for example `fastlocals`, `names`) to index
  // into a PyTupleObject.
  FrameVariableInst* FrameVariable(FrameVariableInst::FrameVariableKind kind,
                                   int64_t index = 0) {
    return b_.Create<FrameVariableInst>(kind, index);
  }

  // Loads a value.
  LoadInst* Load(const MemoryInst::Operand& op, bool steal = false,
                 LoadInst::Extension extension = LoadInst::kNoExtension) {
    return b_.Create<LoadInst>(op, steal, extension);
  }

  // Loads a 64 bit value from [%base + offset + index << shift].
  // This a plain load, that does not take the reference.
  LoadInst* Load64(Value* base, int64_t offset = 0, Value* index = nullptr,
                   MemoryInst::Shift shift = MemoryInst::Shift::k1) {
    return Load(MemoryInst::Operand(base, offset, index, shift));
  }
  // Loads a 64 bit value from [%base + index << shift].
  // This a plain load, that does not take the reference.
  LoadInst* Load64(Value* base, Value* index, MemoryInst::Shift shift) {
    return Load(MemoryInst::Operand(base, index, shift));
  }

  // Loads a 64 bit value from [%base + offset] without extension.
  // Steals the reference from the object loaded from. The reference in the
  // memory object in now invalid and must be overwritten before any call to the
  // Python runtime.
  LoadInst* LoadSteal(Value* base, int64_t offset = 0, Value* index = nullptr,
                      MemoryInst::Shift shift = MemoryInst::Shift::k1) {
    return Load(MemoryInst::Operand(base, offset, index, shift),
                /*steal=*/true);
  }

  // Loads a global given `index`. The symbol name is determined by the
  // `co_names` tuple of the function's code object, and is looked up from
  // globals and builtins per Python's lookup semantics.
  LoadGlobalInst* LoadGlobal(int64_t index) {
    return b_.Create<LoadGlobalInst>(index);
  }

  // Stores %value to the memory operand
  StoreInst* Store(
      Value* value, const MemoryInst::Operand& op, bool donate = false,
      StoreInst::Truncation truncation = StoreInst::kNoTruncation) {
    return b_.Create<StoreInst>(value, op, donate, truncation);
  }

  // Stores %value to [%base + offset + index << shift] without truncation.
  // This is a plain store, that does not store ownership.
  StoreInst* Store64(Value* value, Value* base, int64_t offset = 0,
                     Value* index = nullptr,
                     MemoryInst::Shift shift = MemoryInst::Shift::k1) {
    return Store(value, MemoryInst::Operand(base, offset, index, shift));
  }

  // Stores %value to [%base + offset + index << shift] without truncation.
  // Also stores the ownership of the value that is now managed by the object
  // in which we store. Care must be taken that the previous value that is
  // overwritten was not a valid reference either because it is nullptr or
  // because it has just been stolen by LoadSteal.
  StoreInst* StoreDonate(Value* value, Value* base, int64_t offset = 0,
                         Value* index = nullptr,
                         MemoryInst::Shift shift = MemoryInst::Shift::k1) {
    return Store(value, MemoryInst::Operand(base, offset, index, shift),
                 /*donate=*/true);
  }

  UnreachableInst* Unreachable() { return b_.Create<UnreachableInst>(); }

  ReturnInst* Return(Value* returned_value) {
    return b_.Create<ReturnInst>(returned_value);
  }

  // Checks whether the given divisor will lead to division-by-zero.
  // If so, branch to the safepoint for deoptimisation.
  void DeoptimizeIfDivisionByZero(NumericInst::NumericType type, Value* divisor,
                                  SafepointInst* safepoint) {
    switch (type) {
      case NumericInst::kInt64:
        DeoptimizeIfSafepoint(divisor, /*negated=*/true,
                              "Integer division by zero", safepoint);
        break;
      case NumericInst::kDouble:
        DeoptimizeIfSafepoint(b_.Create<FloatZeroInst>(divisor),
                              /*negated=*/false, "Float division by zero",
                              safepoint);
        break;
    }
  }

  // Checks whether the given shift amount is negative.
  // If so, branch to the safepoint for deoptimisation.
  void DeoptimizeIfNegativeShift(Value* shift, SafepointInst* safepoint) {
    DeoptimizeIfSafepoint(IsNegative(shift), /*negated=*/false,
                          "Shifting by a negative amount", safepoint);
  }

  // Checks whether the immediately preceding unbox or arithmetic op
  // resulted in overflow or incorrect type.
  // If so, branch to the safepoint for deoptimisation.
  DeoptimizeIfSafepointInst* DeoptimizeIfOverflow(Value* value,
                                                  SafepointInst* safepoint) {
    Value* of_op = b_.Create<OverflowedInst>(value);
    return DeoptimizeIfSafepoint(of_op, /*negated=*/false,
                                 "Value was not unboxable", safepoint);
  }

  YieldValueInst* YieldValue(Value* value) {
    return b_.Create<YieldValueInst>(value);
  }

  DeoptimizeIfSafepointInst* DeoptimizeIfSafepoint(
      Value* condition, bool negated, absl::string_view reason,
      int32_t bytecode_offset = 0, absl::Span<Value* const> value_stack = {},
      absl::Span<Value* const> fastlocals = {},
      absl::Span<TryHandler const> try_stack = {}) {
    auto interned = block()->parent()->GetStringTable()->InternString(reason);
    return b_.Create<DeoptimizeIfSafepointInst>(condition, negated, interned,
                                                bytecode_offset, value_stack,
                                                fastlocals, try_stack);
  }

  // Creates a DeoptimizeIfSafepointInst that copies safepoint information from
  // `safepoint`.
  DeoptimizeIfSafepointInst* DeoptimizeIfSafepoint(Value* condition,
                                                   bool negated,
                                                   absl::string_view reason,
                                                   SafepointInst* safepoint) {
    if (safepoint == nullptr) {
      // Caller's responsibility to fill in the safepoint data afterwards.
      return DeoptimizeIfSafepoint(condition, negated, reason);
    } else {
      auto interned = block()->parent()->GetStringTable()->InternString(reason);
      return b_.Create<DeoptimizeIfSafepointInst>(
          condition, negated, interned, safepoint->bytecode_offset(),
          safepoint->value_stack(), safepoint->fastlocals(),
          safepoint->try_handlers(), safepoint->decrefs(),
          safepoint->increfs());
    }
  }

  GetClassIdInst* GetClassId(Value* object) {
    return b_.Create<GetClassIdInst>(object);
  }

  GetObjectDictInst* GetObjectDict(Value* object, int64_t dictoffset = 0,
                                   PyTypeObject* type = nullptr) {
    return b_.Create<GetObjectDictInst>(object, dictoffset,
                                        reinterpret_cast<int64_t>(type));
  }

  GetInstanceClassIdInst* GetInstanceClassId(Value* dict) {
    return b_.Create<GetInstanceClassIdInst>(dict);
  }

  CheckClassIdInst* CheckClassId(Value* object, const Class* cls) {
    return b_.Create<CheckClassIdInst>(object, cls->id());
  }

  LoadFromDictInst* LoadFromDict(Value* dict, int64_t index,
                                 DictKind dict_kind) {
    return b_.Create<LoadFromDictInst>(dict, index, dict_kind);
  }

  StoreToDictInst* StoreToDict(Value* value, Value* dict, int64_t index,
                               DictKind dict_kind) {
    return b_.Create<StoreToDictInst>(value, dict, index, dict_kind);
  }

  ConstantAttributeInst* ConstantAttribute(const Class* cls,
                                           absl::string_view attribute_name) {
    return b_.Create<ConstantAttributeInst>(
        cls->id(),
        block()->parent()->GetStringTable()->InternString(attribute_name));
  }

  DeoptimizedAsynchronouslyInst* DeoptimizedAsynchronously() {
    return b_.Create<DeoptimizedAsynchronouslyInst>();
  }

  SetObjectClassInst* SetObjectClass(Value* object, Value* dict,
                                     int64_t class_id) {
    return b_.Create<SetObjectClassInst>(object, dict, class_id);
  }

  Value* GetType(Value* val) {
    return Load64(val, offsetof(PyObject, ob_type));
  }

  Value* GetSize(Value* val) {
    return Load64(val, offsetof(PyVarObject, ob_size));
  }

  static constexpr int kTupleStartByteOffset = offsetof(PyTupleObject, ob_item);

  // Gets an item in a tuple. Index known at compile time.
  // If the steal flag is steel, this performs a steal, see LoadSteal for more
  // info.
  LoadInst* TupleGetItem(Value* tuple, int64_t index, bool steal = false) {
    return Load(MemoryInst::Operand(
                    tuple, kTupleStartByteOffset + index * sizeof(PyObject*)),
                steal);
  }

  // Gets an item in a tuple. Index known at runtime.
  // If the steal flag is steel, this performs a steal, see LoadSteal for more
  // info.
  LoadInst* TupleGetItem(Value* tuple, Value* index, bool steal = false) {
    return Load(MemoryInst::Operand(tuple, kTupleStartByteOffset, index,
                                    MemoryInst::Shift::k8),
                steal);
  }

  // Sets an item in a tuple. Index known at compile time.
  // This also donates the ownership of the stored_value to the tuple. See
  // `StoreDonate`
  Value* TupleSetItem(Value* stored_value, Value* tuple, int64_t index) {
    return StoreDonate(stored_value, tuple,
                       kTupleStartByteOffset + index * sizeof(PyObject*));
  }

  // Sets an item in a tuple. Index known at runtime.
  // This also donates the ownership of the stored_value to the tuple. See
  // `StoreDonate`
  Value* TupleSetItem(Value* stored_value, Value* tuple, Value* index) {
    return StoreDonate(stored_value, tuple, kTupleStartByteOffset, index,
                       MemoryInst::Shift::k8);
  }

  void DeoptimizedAsynchronously(SafepointInst* safepoint) {
    constexpr absl::string_view kDeoptimizedAsynchronouslyReason =
        "Assumptions were made about class behavior "
        "that were invalidated asynchronously";
    DeoptimizeIfSafepoint(DeoptimizedAsynchronously(),
                          /*negated=*/false, kDeoptimizedAsynchronouslyReason,
                          safepoint);
  }

  BlockInserter inserter() { return b_; }
  Block* block() { return b_.block(); }
  Block::iterator insert_point() { return b_.insert_point(); }

 private:
  BlockInserter b_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BUILDER_H_
