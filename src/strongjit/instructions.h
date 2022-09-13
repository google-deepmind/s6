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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTIONS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTIONS_H_

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "arithmetic.h"
#include "classes/class.h"
#include "classes/class_manager.h"
#include "core_util.h"
#include "global_intern_table.h"
#include "strongjit/base.h"
#include "strongjit/callees.h"
#include "strongjit/instruction.h"
#include "strongjit/string_table.h"
#include "strongjit/util.h"
#include "utils/logging.h"

namespace deepmind::s6 {

////////////////////////////////////////////////////////////////////////////////
// Abstract instruction types

// A TerminatorInst transfers control flow. All successor blocks are explicit,
// and control flow cannot proceed beyond the instruction.
//
// All blocks must end with one and only one TerminatorInst.
//
// A TerminatorInst starts with the list of its sucessors, whose number
// is given by `successor_size()`.
class TerminatorInst : public Instruction {
 public:
  TerminatorInst() {}

  // Returns the list of successors. Is it guaranteed that none of them are
  // nullptr, and this should be maintained when using `mutable_successors`.
  absl::Span<Block* const> successors();
  absl::Span<const Block* const> successors() const;
  absl::Span<Block*> mutable_successors();

  // Number of arguments that are implicitly added by the instruction before
  // the explicit arguments.
  virtual int64_t num_implicit_arguments() const { return 0; }

  // Returns the argument list for the given successor index.
  virtual MutableOperandList mutable_successor_arguments(int64_t index) = 0;
  virtual absl::Span<const Value* const> successor_arguments(
      int64_t index) const = 0;
  absl::Span<Value* const> successor_arguments(int64_t index) {
    auto const_span =
        const_cast<const TerminatorInst*>(this)->successor_arguments(index);
    return absl::MakeSpan(const_cast<Value**>(const_span.data()),
                          const_span.size());
  }

  absl::Span<const Value* const> successor_arguments(const Block* block) const {
    for (int64_t index = 0; index < successor_size(); ++index) {
      if (block == successors()[index]) return successor_arguments(index);
    }
    S6_LOG(FATAL) << "Successor not found!";
  }

  absl::Span<Value* const> successor_arguments(const Block* block) {
    for (int64_t index = 0; index < successor_size(); ++index) {
      if (block == successors()[index]) return successor_arguments(index);
    }
    S6_LOG(FATAL) << "Successor not found!";
  }

  // For all block arguments and parameters on the edge to `succ`, invokes
  // the given callback.
  void ForEachArgumentOnEdge(
      const Block* succ,
      absl::FunctionRef<void(const BlockArgument*, const Value*)> fn) const {
    auto params = successor_arguments(succ);
    auto args = succ->block_arguments();
    int64_t implicit = num_implicit_arguments();
    S6_CHECK_EQ(args.size(), params.size() + implicit);
    for (int64_t i = 0; i < args.size(); ++i) {
      fn(args[i], i < implicit ? nullptr : params[i - implicit]);
    }
  }
  void ForEachArgumentOnEdge(
      Block* succ, absl::FunctionRef<void(BlockArgument*, Value*)> fn) {
    auto params = successor_arguments(succ);
    auto args = succ->block_arguments();
    int64_t implicit = num_implicit_arguments();
    S6_CHECK_EQ(args.size(), params.size() + implicit);
    for (int64_t i = 0; i < args.size(); ++i) {
      fn(args[i], i < implicit ? nullptr : params[i - implicit]);
    }
  }

  // Removes the argument from the specified successor at the specified index.
  // Note that later indices will refer to different arguments after removal.
  void RemoveSuccessorArgumentAt(int64_t successor_index,
                                 int64_t argument_index) {
    S6_CHECK_LT(successor_index, successor_size());
    S6_CHECK_LT(argument_index, successor_arguments(successor_index).size());
    mutable_successor_arguments(successor_index).erase_at(argument_index);
  }

  // Returns the number of successors.
  virtual int64_t successor_size() const = 0;

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;
};

class UnconditionalTerminatorInst : public TerminatorInst {
 public:
  UnconditionalTerminatorInst() { raw_operands().resize(1); }
  explicit UnconditionalTerminatorInst(
      Block* successor, absl::Span<Value* const> arguments = {}) {
    auto& operands = raw_operands();
    operands.push_back(successor);
    absl::c_copy(arguments, std::back_inserter(operands));
  }

  // Returns 0 or 1 depending on whether there is a successor present.
  int64_t successor_size() const final { return unique_successor() ? 1 : 0; }

  absl::Span<const Value* const> successor_arguments(
      int64_t index) const final {
    S6_CHECK_EQ(index, 0);
    S6_CHECK(unique_successor());
    return arguments();
  }
  MutableOperandList mutable_successor_arguments(int64_t index) final {
    S6_CHECK_EQ(index, 0);
    S6_CHECK(unique_successor());
    return mutable_arguments();
  }

  // If is possible for a unique successor to be nullptr, in which case,
  // there is in fact 0 successors (as given by successor_size());
  const Block* unique_successor() const;
  Block* unique_successor();
  Block** mutable_unique_successor() {
    S6_DCHECK_GE(operands().size(), 1);
    return reinterpret_cast<Block**>(&raw_operands().front());
  }

  absl::Span<Value* const> arguments() { return operands().subspan(1); }
  absl::Span<const Value* const> arguments() const {
    return operands().subspan(1);
  }
  MutableOperandList mutable_arguments() {
    S6_DCHECK_GE(operands().size(), 1);
    return raw_operands().line(0, 1);
  }

  void RemoveArgumentAt(int64_t index) { mutable_arguments().erase_at(index); }

  void AddArgument(Value* v) { mutable_arguments().push_back(v); }

  template <typename P>
  static typename P::value_type Format(UnconditionalTerminatorInst* bi, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->OptionalValue(bi->mutable_unique_successor()),
                     p->OptionalValueList(bi->mutable_arguments(), "[ ", " ]"));
  }
};

// A conditional branch to two successors.
class ConditionalTerminatorInst : public TerminatorInst {
 public:
  ConditionalTerminatorInst() {
    raw_operands().resize(3);
    raw_operands().push_line();
  }
  ConditionalTerminatorInst(Value* condition, Block* true_successor,
                            absl::Span<Value* const> true_arguments,
                            Block* false_successor,
                            absl::Span<Value* const> false_arguments) {
    auto& operands = raw_operands();
    operands.push_back(true_successor);
    operands.push_back(false_successor);
    operands.push_back(condition);
    absl::c_copy(true_arguments, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(false_arguments, std::back_inserter(operands));
  }
  ConditionalTerminatorInst(Value* condition, Block* true_successor,
                            Block* false_successor)
      : ConditionalTerminatorInst(condition, true_successor, {},
                                  false_successor, {}) {}

  int64_t successor_size() const final { return 2; }

  absl::Span<const Value* const> successor_arguments(
      int64_t index) const final {
    if (index == 0) {
      return true_arguments();
    } else {
      S6_CHECK_EQ(index, 1);
      return false_arguments();
    }
  }
  MutableOperandList mutable_successor_arguments(int64_t index) final {
    if (index == 0) {
      return mutable_true_arguments();
    } else {
      S6_CHECK_EQ(index, 1);
      return mutable_false_arguments();
    }
  }

  Block* true_successor();
  const Block* true_successor() const;
  Block** mutable_true_successor() {
    S6_DCHECK_GE(operands().size(), 3);
    return reinterpret_cast<Block**>(&raw_operands().front());
  }

  Block* false_successor();
  const Block* false_successor() const;
  Block** mutable_false_successor() {
    S6_DCHECK_GE(operands().size(), 3);
    return reinterpret_cast<Block**>(&raw_operands()[1]);
  }

  static constexpr int64_t kConditionOperandIndex = 2;
  Value* condition() { return operands().at(2); }
  const Value* condition() const { return operands().at(2); }
  Value** mutable_condition() {
    S6_DCHECK_GE(operands().size(), 3);
    return &raw_operands()[2];
  }

  absl::Span<Value* const> true_arguments() {
    S6_DCHECK_GE(operands().size(), 3);
    return const_raw_operands().line_span(0, 3);
  }
  absl::Span<const Value* const> true_arguments() const {
    S6_DCHECK_GE(operands().size(), 3);
    return const_raw_operands().line_span(0, 3);
  }
  MutableOperandList mutable_true_arguments() {
    S6_DCHECK_GE(operands().size(), 3);
    return raw_operands().line(0, 3);
  }

  // Removes the argument at the specified index from the 'true' branch.
  // argument_index must be in the interval [0, true_arguments.size())
  // Note: removal of an argument at an index will change the meaning of later
  // indices.
  void RemoveTrueArgumentAt(int64_t argument_index) {
    mutable_true_arguments().erase_at(argument_index);
  }

  absl::Span<Value* const> false_arguments() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 1);
    return const_raw_operands().line_span(1);
  }
  absl::Span<const Value* const> false_arguments() const {
    S6_DCHECK_GE(const_raw_operands().line_num(), 1);
    return const_raw_operands().line_span(1);
  }
  MutableOperandList mutable_false_arguments() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 1);
    return raw_operands().line(1);
  }
  // Remove the argument at the specified index from the 'false' branch.
  // argument_index must be in the interval [0, false_arguments.size())
  // Note: removal of an argument at an index will change the meaning of later
  // indices.
  void RemoveFalseArgumentAt(int64_t argument_index) {
    mutable_false_arguments().erase_at(argument_index);
  }
};

// An incref or decref instruction.
class RefcountInst : public Instruction {
 public:
  RefcountInst() { raw_operands().resize(1); }
  RefcountInst(Nullness nullness, Value* operand) : nullness_(nullness) {
    raw_operands().push_back(operand);
  }

  Nullness nullness() const { return nullness_; }
  Nullness* mutable_nullness() { return &nullness_; }

  Value* operand() { return operands()[0]; }
  const Value* operand() const { return operands()[0]; }
  Value** mutable_operand() {
    S6_DCHECK_GE(const_raw_operands().size(), 1);
    return &raw_operands().front();
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(RefcountInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Enum(i->mutable_nullness(), {"null?", "notnull"}),
                     p->Value(i->mutable_operand()));
  }

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;

 private:
  Nullness nullness_;
};

class NumericInst : public Instruction {
 public:
  enum NumericType {
    // Operates on 64-bit signed integers.
    kInt64,
    // Operates on 64-bit floating point numbers.
    kDouble,
  };

  // Declares if the instruction also support floating points or only intergers.
  static constexpr bool kSupportsDouble = true;

  NumericInst() = default;
  explicit NumericInst(NumericType type) : type_(type) {}

  NumericType type() const { return type_; }
  NumericType* mutable_type() { return &type_; }

  bool IsIntType() const { return type() == kInt64; }
  bool IsDoubleType() const { return type() == kDouble; }

  // Helper for formatting the type.
  template <typename P>
  static typename P::value_type FormatType(NumericInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Enum(i->mutable_type(), {"i64", "f64"});
  }

 private:
  NumericType type_;
};

class UnaryInst : public NumericInst {
 public:
  UnaryInst() { raw_operands().resize(1); }
  explicit UnaryInst(NumericType type, Value* operand) : NumericInst(type) {
    raw_operands().push_back(operand);
  }

  Value* operand() { return operands()[0]; }
  const Value* operand() const { return operands()[0]; }
  Value** mutable_operand() {
    S6_DCHECK_GE(const_raw_operands().size(), 1);
    return &raw_operands().front();
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(UnaryInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(FormatType(i, p), p->Value(i->mutable_operand()));
  }
};

class BinaryInst : public NumericInst {
 public:
  BinaryInst() { raw_operands().resize(2); }
  explicit BinaryInst(NumericType type, Value* lhs, Value* rhs)
      : NumericInst(type) {
    raw_operands().push_back(lhs);
    raw_operands().push_back(rhs);
  }

  Value* lhs() { return operands()[0]; }
  const Value* lhs() const { return operands()[0]; }
  Value** mutable_lhs() {
    S6_DCHECK_GE(const_raw_operands().size(), 2);
    return &raw_operands().front();
  }

  Value* rhs() { return operands()[1]; }
  const Value* rhs() const { return operands()[1]; }
  Value** mutable_rhs() {
    S6_DCHECK_GE(const_raw_operands().size(), 2);
    return &raw_operands()[1];
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(BinaryInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(
        FormatType(i, p),
        p->CommaConcat(p->Value(i->mutable_lhs()), p->Value(i->mutable_rhs())));
  }
};

// MemoryInst an operation in memory on a pointer + offset address.
class MemoryInst : public Instruction {
 public:
  enum class Shift { k1 = 0, k2 = 1, k4 = 2, k8 = 3 };
  static int64_t ShiftToInt(Shift s) { return static_cast<int64_t>(s); }

  // A helper struct to build a MemoryInst.
  class Operand {
   public:
    explicit Operand(Value* pointer, int64_t offset = 0, Value* index = nullptr,
                     Shift shift = Shift::k1)
        : pointer_(pointer), offset_(offset), index_(index), shift_(shift) {
      S6_CHECK(pointer);
      if (shift != Shift::k1) S6_CHECK(index);
    }
    explicit Operand(Value* pointer, Value* index, Shift shift = Shift::k1)
        : Operand(pointer, 0, index, shift) {
      S6_CHECK(index);
    }

   private:
    Value* pointer_;
    int64_t offset_ = 0;
    Value* index_ = nullptr;
    Shift shift_ = Shift::k1;
    friend MemoryInst;
  };

  MemoryInst() { raw_operands().resize(2); }
  explicit MemoryInst(const Operand& op)
      : offset_(op.offset_), shift_(op.shift_) {
    raw_operands().push_back(op.pointer_);
    raw_operands().push_back(op.index_);
  }

  const Value* pointer() const { return operands()[0]; }
  Value* pointer() { return operands()[0]; }
  Value** mutable_pointer() {
    S6_DCHECK_GE(const_raw_operands().size(), 2);
    return &raw_operands().front();
  }

  const Value* index() const { return operands()[1]; }
  bool HasIndex() const { return index(); }
  Value* index() { return operands()[1]; }
  Value** mutable_index() {
    S6_DCHECK_GE(const_raw_operands().size(), 2);
    return &raw_operands()[1];
  }

  int64_t offset() const { return offset_; }
  int64_t* mutable_offset() { return &offset_; }

  Shift shift() const { return shift_; }
  Shift* mutable_shift() { return &shift_; }

  template <typename P>
  static typename P::value_type FormatMemoryOp(MemoryInst* i, P* p) {
    return p->Concat(
        p->Str("["), p->Value(i->mutable_pointer()), p->Str("+"),
        p->Imm(i->mutable_offset()), p->OptionalValue(i->mutable_index(), "+"),
        p->Enum(i->mutable_shift(), {"", "* 2", "* 4", "* 8"}), p->Str("]"));
  }

 private:
  int64_t offset_;
  Shift shift_;
};

// Base class for all instruction that calls something.
// The call arguments are named are stored on line 1 of the operands
// MultiLineVector
class CallInst : public Instruction {
 public:
  CallInst() : CallInst(0) {}
  explicit CallInst(size_t num_subclass_operands) {
    auto& operands = raw_operands();
    operands.resize(num_subclass_operands);
    operands.push_line();
  }
  explicit CallInst(absl::Span<Value* const> subclass_operands,
                    absl::Span<Value* const> call_arguments) {
    auto& operands = raw_operands();
    absl::c_copy(subclass_operands, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(call_arguments, std::back_inserter(operands));
  }

  absl::Span<Value* const> call_arguments() {
    return const_raw_operands().line_span(1);
  }
  absl::Span<const Value* const> call_arguments() const {
    return const_raw_operands().line_span(1);
  }
  MutableOperandList mutable_call_arguments() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 1);
    return raw_operands().line(1);
  }

  // Given an index into the operand list, returns the index into the
  // call_arguments() list, or -1 if the index does not correspond to a call
  // argument.
  int64_t GetCallArgumentIndex(int64_t operand_index) const {
    size_t size_line0 = const_raw_operands().line_size(0);
    return operand_index >= size_line0 ? operand_index - size_line0 : -1;
  }

 protected:
  template <typename P>
  static typename P::value_type FormatArguments(CallInst* i, P* p) {
    return p->OptionalValueList(i->mutable_call_arguments(), "(", ")");
  }
};

// Base class to call a native C function
class CallNativeBaseInst : public CallInst {
 public:
  CallNativeBaseInst() = default;
  explicit CallNativeBaseInst(Callee callee,
                              absl::Span<Value* const> call_arguments = {})
      : CallInst({}, call_arguments), callee_(callee) {}

  Callee callee() const { return callee_; }
  Callee* mutable_callee() { return &callee_; }
  absl::string_view CalleeName() const { return ToString(callee()); }

  bool CalleeIs(Callee c) const { return callee() == c; }

  bool CalleeIsAnyOf(absl::Span<const Callee> callees) const {
    return absl::c_find(callees, callee()) != callees.end();
  }

  template <typename P>
  static typename P::value_type Format(CallNativeBaseInst* i, P* p) {
    return p->Concat(p->Enum(i->mutable_callee(), kCalleeNames),
                     FormatArguments(i, p));
  }

 private:
  Callee callee_;
};

// An instruction with precise bytecode location information. These instructions
// may either call out to other Python code or otherwise cause the bytecode
// offset to be required to be materialized.
//
// This is a helper class to access the bytecode_offset member of precise
// instruction as these instructions sit in different class hierarchies -
// DecrefInst is a RefcountInst, ExceptInst is a Terminator.
class PreciseLocationInst {
 public:
  PreciseLocationInst() : bytecode_offset_(0) {}
  explicit PreciseLocationInst(int bytecode_offset)
      : bytecode_offset_(bytecode_offset) {}
  ~PreciseLocationInst() = default;

  int32_t bytecode_offset() const { return bytecode_offset_; }
  int32_t* mutable_bytecode_offset() { return &bytecode_offset_; }

  static PreciseLocationInst* Get(Value* inst);
  static const PreciseLocationInst* Get(const Value* inst);

 private:
  int32_t bytecode_offset_;
};

static_assert(!std::is_polymorphic_v<PreciseLocationInst>,
              "PreciseLocationInst is not a safe mix-in class");

////////////////////////////////////////////////////////////////////////////////
// Concrete instructions

// Materializes an int64_t immediate as a Value.
class ConstantInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "constant";
  static constexpr Value::Kind kKind = Value::kConstant;
  Kind kind() const final { return kKind; }

  ConstantInst() = default;
  explicit ConstantInst(int64_t value) : value_(value) {}

  int64_t value() const { return value_; }
  void set_value(int64_t v) { value_ = v; }
  int64_t* mutable_value() { return &value_; }

  template <typename P>
  static typename P::value_type Format(ConstantInst* ci, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Imm(ci->mutable_value());
  }

 private:
  int64_t value_;
};

// An unconditional jump to a single successor.
class JmpInst final : public UnconditionalTerminatorInst {
 public:
  static constexpr absl::string_view kMnemonic = "jmp";
  static constexpr Value::Kind kKind = Value::kJmp;
  Kind kind() const final { return kKind; }

  using UnconditionalTerminatorInst::UnconditionalTerminatorInst;
};

// A conditional branch to two successors.
class BrInst final : public ConditionalTerminatorInst {
 public:
  static constexpr absl::string_view kMnemonic = "br";
  static constexpr Value::Kind kKind = Value::kBr;
  Kind kind() const final { return kKind; }

  using ConditionalTerminatorInst::ConditionalTerminatorInst;

  bool true_deoptimized() const { return true_deoptimized_; }
  void set_true_deoptimized(bool deoptimized) {
    true_deoptimized_ = deoptimized;
  }
  bool* mutable_true_deoptimized() { return &true_deoptimized_; }

  bool false_deoptimized() const { return false_deoptimized_; }
  void set_false_deoptimized(bool deoptimized) {
    false_deoptimized_ = deoptimized;
  }
  bool* mutable_false_deoptimized() { return &false_deoptimized_; }

  template <typename P>
  static typename P::value_type Format(BrInst* bi, P* p) {
    AssertIsInstructionFormatter<P>();
    // Note that this relies upon the parser filling in true_arguments before
    // any of false_arguments as both MutableOperandLists append to the same
    // container.
    return p->CommaConcat(
        p->Value(bi->mutable_condition()),
        p->Concat(
            p->Flag(bi->mutable_true_deoptimized(), "deoptimized"),
            p->Value(bi->mutable_true_successor()),
            p->OptionalValueList(bi->mutable_true_arguments(), "[ ", " ]")),
        p->Concat(
            // Because the true arguments may cause a realloc of the operands
            // vector, we must compute the address of the false successor
            // lazily.
            p->Flag(bi->mutable_false_deoptimized(), "deoptimized"),
            p->Value([=]() { return bi->mutable_false_successor(); }),
            p->OptionalValueList(bi->mutable_false_arguments(), "[ ", " ]")));
  }

 private:
  bool true_deoptimized_ = false;
  bool false_deoptimized_ = false;
};

// Compares two operands.
class CompareInst final : public NumericInst {
 public:
  static constexpr absl::string_view kMnemonic = "cmp";
  static constexpr Value::Kind kKind = Value::kCompare;
  Kind kind() const final { return kKind; }

  enum Comparison {
    kEqual,
    kNotEqual,
    kGreaterThan,
    kGreaterEqual,
    kLessThan,
    kLessEqual
  };

  // Evaluates the compare on two concrete arguments
  template <typename T>
  bool Evaluate(T lhs, T rhs) const {
    switch (comparison()) {
      case kEqual:
        return lhs == rhs;
      case kNotEqual:
        return lhs != rhs;
      case kGreaterEqual:
        return lhs >= rhs;
      case kGreaterThan:
        return lhs > rhs;
      case kLessEqual:
        return lhs <= rhs;
      case kLessThan:
        return lhs < rhs;
    }
    S6_UNREACHABLE();
  }

  CompareInst() { raw_operands().resize(2); }
  CompareInst(Comparison comparison, NumericType type, Value* lhs, Value* rhs)
      : NumericInst(type), comparison_(comparison) {
    raw_operands().push_back(lhs);
    raw_operands().push_back(rhs);
  }

  Comparison comparison() const { return comparison_; }
  Comparison* mutable_comparison() { return &comparison_; }

  // Checks if this is an equality comparison, kEqual or kNotEqual.
  bool IsEquality() const {
    return comparison_ == kEqual || comparison_ == kNotEqual;
  }

  Value* lhs() { return operands()[0]; }
  const Value* lhs() const { return operands()[0]; }
  Value** mutable_lhs() {
    S6_DCHECK_EQ(const_raw_operands().size(), 2);
    return &raw_operands().front();
  }

  Value* rhs() { return operands()[1]; }
  const Value* rhs() const { return operands()[1]; }
  Value** mutable_rhs() {
    S6_DCHECK_EQ(const_raw_operands().size(), 2);
    return &raw_operands()[1];
  }

  template <typename P>
  static typename P::value_type Format(CompareInst* ci, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(
        p->Enum(ci->mutable_comparison(), {"eq", "ne", "gt", "ge", "lt", "le"}),
        FormatType(ci, p),
        p->CommaConcat(p->Value(ci->mutable_lhs()),
                       p->Value(ci->mutable_rhs())));
  }

 private:
  Comparison comparison_;
};

// Unconditional jump to the exception handler.
class ExceptInst final : public UnconditionalTerminatorInst,
                         public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "except";
  static constexpr Value::Kind kKind = Value::kExcept;
  Kind kind() const final { return kKind; }

  int64_t num_implicit_arguments() const final { return 6; }

  static constexpr bool kClobbersAllRegisters = true;

  ExceptInst() = default;
  explicit ExceptInst(Block* successor, absl::Span<Value* const> arguments = {},
                      int32_t bytecode_offset = 0)
      : UnconditionalTerminatorInst(successor, arguments),
        PreciseLocationInst(bytecode_offset) {}

  template <typename P>
  static typename P::value_type Format(ExceptInst* bi, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->OptionalValue(bi->mutable_unique_successor()),
                     p->OptionalValueList(bi->mutable_arguments(), "[ ", " ]"),
                     p->BytecodeOffset(bi->mutable_bytecode_offset()));
  }
};

// A conditional deoptimize. the true_successor is a deoptimized block.
// Maintains a set of values that are required in order to run the deoptimized
// code.
class DeoptimizeIfInst final : public ConditionalTerminatorInst {
 public:
  static constexpr absl::string_view kMnemonic = "deoptimize_if";
  static constexpr Value::Kind kKind = Value::kDeoptimizeIf;
  Kind kind() const final { return kKind; }

  DeoptimizeIfInst() { raw_operands().push_line(); }
  DeoptimizeIfInst(Value* condition, bool negated, Block* true_successor,
                   absl::Span<Value* const> true_arguments,
                   Block* false_successor,
                   absl::Span<Value* const> false_arguments,
                   absl::Span<Value* const> required_values)
      : ConditionalTerminatorInst(condition, true_successor, true_arguments,
                                  false_successor, false_arguments) {
    negated_ = negated;
    auto& operands = raw_operands();
    operands.push_line();
    absl::c_copy(required_values, std::back_inserter(operands));
  }

  // Returns the list of required values. These values are required by the
  // deoptimization handler.
  absl::Span<Value* const> required_values() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 3);
    return const_raw_operands().line_span(2);
  }
  absl::Span<const Value* const> required_values() const {
    S6_DCHECK_GE(const_raw_operands().line_num(), 3);
    return const_raw_operands().line_span(2);
  }
  MutableOperandList mutable_required_values() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 3);
    return raw_operands().line(2);
  }

  // If true, the condition is negated and if *false*, the deoptimization branch
  // is taken.
  bool negated() const { return negated_; }
  void set_negated(bool b) { negated_ = b; }
  bool* mutable_negated() { return &negated_; }

  template <typename P>
  static typename P::value_type Format(DeoptimizeIfInst* bi, P* p) {
    AssertIsInstructionFormatter<P>();
    // Note that this relies upon the parser filling in true_arguments before
    // any of required_values or false_arguments as both MutableOperandLists
    // append to the same container.
    return p->CommaConcat(
        p->Concat(p->Flag(bi->mutable_negated(), "not"),
                  p->Value(bi->mutable_condition())),
        p->Concat(
            p->Value(bi->mutable_true_successor()),
            p->OptionalValueList(bi->mutable_true_arguments(), "[ ", " ]")),
        p->Concat(
            // Because the true arguments may cause a realloc of the operands
            // vector, we must compute the address of the false successor
            // lazily.
            p->Value([=]() { return bi->mutable_false_successor(); }),
            p->OptionalValueList(bi->mutable_false_arguments(), "[ ", " ]")),
        p->Concat(
            p->Str("materializing"), p->Str("values"),
            p->OptionalValueList(bi->mutable_required_values(), "[", "]")));
  }

 private:
  bool negated_ = false;
};

// Unconditional return.
class ReturnInst final : public TerminatorInst {
 public:
  static constexpr absl::string_view kMnemonic = "return";
  static constexpr Value::Kind kKind = Value::kReturn;
  Kind kind() const final { return kKind; }

  ReturnInst() { raw_operands().resize(1); }
  explicit ReturnInst(Value* returned_value) {
    auto& operands = raw_operands();
    operands.push_back(returned_value);
  }

  Value* returned_value() { return operands().front(); }
  const Value* returned_value() const { return operands().front(); }
  Value** mutable_returned_value() {
    S6_DCHECK_GE(const_raw_operands().size(), 1);
    return &raw_operands().front();
  }

  int64_t successor_size() const final { return 0; }

  absl::Span<const Value* const> successor_arguments(
      int64_t index) const final {
    S6_LOG(FATAL) << "ReturnInst has no successor";
  }
  MutableOperandList mutable_successor_arguments(int64_t index) final {
    S6_LOG(FATAL) << "ReturnInst has no successor";
  }

  template <typename P>
  static typename P::value_type Format(ReturnInst* ri, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(ri->mutable_returned_value());
  }
};

// Terminate the current control flow arc with an error.
class UnreachableInst final : public TerminatorInst {
 public:
  static constexpr absl::string_view kMnemonic = "unreachable";
  static constexpr Value::Kind kKind = Value::kUnreachable;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = false;

  UnreachableInst() = default;

  int64_t successor_size() const final { return 0; }

  absl::Span<const Value* const> successor_arguments(
      int64_t index) const final {
    S6_LOG(FATAL) << "UnreachableInst has no successor";
  }
  MutableOperandList mutable_successor_arguments(int64_t index) final {
    S6_LOG(FATAL) << "UnreachableInst has no successor";
  }
};

////////////////////////////////////////////////////////////////////////////////
// ALU operations

class NegateInst final : public UnaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "negate";
  static constexpr Value::Kind kKind = Value::kNegate;
  Kind kind() const final { return kKind; }

  template <typename T>
  T Evaluate(T t) const {
    return -t;
  }

  using UnaryInst::UnaryInst;
};

class NotInst final : public UnaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "not";
  static constexpr Value::Kind kKind = Value::kNot;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t t) const { return ~t; }

  using UnaryInst::UnaryInst;
};

class AddInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "add";
  static constexpr Value::Kind kKind = Value::kAdd;
  Kind kind() const final { return kKind; }

  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::Add(lhs, rhs).result;
  }

  double Evaluate(double lhs, double rhs) const { return lhs + rhs; }

  using BinaryInst::BinaryInst;
};

class SubtractInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "subtract";
  static constexpr Value::Kind kKind = Value::kSubtract;
  Kind kind() const final { return kKind; }

  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::Subtract(lhs, rhs).result;
  }

  double Evaluate(double lhs, double rhs) const { return lhs - rhs; }

  using BinaryInst::BinaryInst;
};

class MultiplyInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "multiply";
  static constexpr Value::Kind kKind = Value::kMultiply;
  Kind kind() const final { return kKind; }

  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::Multiply(lhs, rhs).result;
  }

  double Evaluate(double lhs, double rhs) const { return lhs * rhs; }

  using BinaryInst::BinaryInst;
};

class DivideInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "divide";
  static constexpr Value::Kind kKind = Value::kDivide;
  Kind kind() const final { return kKind; }

  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::Divide(lhs, rhs).result;
  }

  double Evaluate(double lhs, double rhs) const { return lhs / rhs; }

  using BinaryInst::BinaryInst;
};

class RemainderInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "remainder";
  static constexpr Value::Kind kKind = Value::kRemainder;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::Remainder(lhs, rhs).result;
  }

  using BinaryInst::BinaryInst;
};

class AndInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "and";
  static constexpr Value::Kind kKind = Value::kAnd;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const { return lhs & rhs; }

  using BinaryInst::BinaryInst;
};

class OrInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "or";
  static constexpr Value::Kind kKind = Value::kOr;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const { return lhs | rhs; }

  using BinaryInst::BinaryInst;
};

class XorInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "xor";
  static constexpr Value::Kind kKind = Value::kXor;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const { return lhs ^ rhs; }

  using BinaryInst::BinaryInst;
};

class ShiftLeftInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "shift_left";
  static constexpr Value::Kind kKind = Value::kShiftLeft;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::ShiftLeft(lhs, rhs).result;
  }

  using BinaryInst::BinaryInst;
};

class ShiftRightSignedInst final : public BinaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "shift_right_signed";
  static constexpr Value::Kind kKind = Value::kShiftRightSigned;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t lhs, int64_t rhs) const {
    return arithmetic::ShiftRight(lhs, rhs).result;
  }

  using BinaryInst::BinaryInst;
};

class IntToFloatInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "int_to_float";
  static constexpr Value::Kind kKind = Value::kIntToFloat;
  Kind kind() const final { return kKind; }

  Value* operand() { return operands()[0]; }
  const Value* operand() const { return operands()[0]; }
  Value** mutable_operand() {
    S6_DCHECK_GE(const_raw_operands().size(), 1);
    return &raw_operands().front();
  }

  double Evaluate(int64_t t) const { return static_cast<double>(t); }

  IntToFloatInst() { raw_operands().resize(1); }
  explicit IntToFloatInst(Value* v) { raw_operands().push_back(v); }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(IntToFloatInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(i->mutable_operand());
  }
};

// Sign extend from 32 bits to 64 bits, discard the highest 32 bits.
class SextInst final : public UnaryInst {
 public:
  static constexpr absl::string_view kMnemonic = "sext";
  static constexpr Value::Kind kKind = Value::kSext;
  Kind kind() const final { return kKind; }

  static constexpr bool kSupportsDouble = false;
  int64_t Evaluate(int64_t x) const { return static_cast<int32_t>(x); }

  using UnaryInst::UnaryInst;
};

////////////////////////////////////////////////////////////////////////////////
// Reference counting

class IncrefInst final : public RefcountInst {
 public:
  static constexpr absl::string_view kMnemonic = "incref";
  static constexpr Value::Kind kKind = Value::kIncref;
  Kind kind() const final { return kKind; }

  using RefcountInst::RefcountInst;
};

class DecrefInst final : public RefcountInst, public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "decref";
  static constexpr Value::Kind kKind = Value::kDecref;
  Kind kind() const final { return kKind; }

  static constexpr bool kClobbersAllRegisters = true;

  DecrefInst() = default;
  DecrefInst(Nullness nullness, Value* value, int32_t bytecode_offset = 0)
      : RefcountInst(nullness, value), PreciseLocationInst(bytecode_offset) {}

  template <typename P>
  static typename P::value_type Format(DecrefInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Enum(i->mutable_nullness(), {"null?", "notnull"}),
                     p->Value(i->mutable_operand()),
                     p->BytecodeOffset(i->mutable_bytecode_offset()));
  }

 private:
};

////////////////////////////////////////////////////////////////////////////////
// Loads and stores

// Loads from a pointer location with optional extension.
class LoadInst final : public MemoryInst {
 public:
  static constexpr absl::string_view kMnemonic = "load";
  static constexpr Value::Kind kKind = Value::kLoad;
  Kind kind() const final { return kKind; }

  enum Extension : uint16_t {
    kUnsigned8,
    kSigned8,
    kUnsigned16,
    kSigned16,
    kUnsigned32,
    kSigned32,
    kNoExtension
  };

  LoadInst() = default;
  explicit LoadInst(const Operand& op, bool steal = false,
                    Extension extension = kNoExtension)
      : MemoryInst(op), extension_(extension), steal_(steal) {
    if (steal_) S6_CHECK_EQ(extension, Extension::kNoExtension);
  }

  Extension extension() const { return extension_; }
  Extension* mutable_extension() { return &extension_; }

  bool steal() const { return steal_; }
  bool* mutable_steal() { return &steal_; }

  template <typename P>
  static typename P::value_type Format(LoadInst* i, P* p) {
    return p->Concat(p->Flag(i->mutable_steal(), "steal"),
                     p->Enum(i->mutable_extension(),
                             {"u8", "s8", "u16", "s16", "u32", "s32", "i64"}),
                     FormatMemoryOp(i, p));
  }

 private:
  Extension extension_;

  // When this flag is enabled the load also steals the reference ownership from
  // the memory object. The memory location is now invalid and must be
  // overwritten before any potential calls to the Python runtime (in particular
  // care should be taken about decrefs).
  // This flag does not change at all the actual semantics of the instruction
  // and is ignored by all backends. It is only used by the optimizer.
  bool steal_;
};

// Loads a global by name, given an index into the kNames tuple. Lookup follows
// the python convention, looking in the frame globals dict and then in
// builtins if not found in globals. The result is a borrowed reference to a
// PyObject* or 0 (with an appropriate exception set) if lookup failed.
class LoadGlobalInst final : public Instruction, public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "load_global";
  static constexpr Value::Kind kKind = Value::kLoadGlobal;
  Kind kind() const final { return kKind; }

  LoadGlobalInst() = default;
  explicit LoadGlobalInst(int64_t index, int32_t bytecode_offset = 0)
      : PreciseLocationInst(bytecode_offset), index_(index) {}

  int64_t index() const { return index_; }
  int64_t* mutable_index() { return &index_; }

  template <typename P>
  static typename P::value_type Format(LoadGlobalInst* i, P* p) {
    return p->Imm(i->mutable_index());
  }

 private:
  int64_t index_;
};

// Stores to a pointer location with optional truncation.
class StoreInst final : public MemoryInst {
 public:
  static constexpr Value::Kind kKind = Value::kStore;
  static constexpr absl::string_view kMnemonic = "store";
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;

  enum Truncation : uint16_t { kInt8, kInt16, kInt32, kNoTruncation };

  StoreInst() { raw_operands().resize(3); }
  explicit StoreInst(Value* stored_value, const Operand op, bool donate = false,
                     Truncation truncation = kNoTruncation)
      : MemoryInst(op), truncation_(truncation), donate_(donate) {
    if (donate) S6_CHECK_EQ(truncation, Truncation::kNoTruncation);
    raw_operands().push_back(stored_value);
  }

  const Value* stored_value() const { return operands()[2]; }
  Value* stored_value() { return operands()[2]; }
  Value** mutable_stored_value() {
    S6_DCHECK_GE(const_raw_operands().size(), 3);
    return &raw_operands()[2];
  }

  Truncation truncation() const { return truncation_; }
  Truncation* mutable_truncation() { return &truncation_; }

  bool donate() const { return donate_; }
  bool* mutable_donate() { return &donate_; }

  template <typename P>
  static typename P::value_type Format(StoreInst* i, P* p) {
    return p->Concat(
        p->Flag(i->mutable_donate(), "ref"),
        p->Enum(i->mutable_truncation(), {"i8", "i16", "i32", "i64"}),
        p->CommaConcat(p->Value(i->mutable_stored_value()),
                       FormatMemoryOp(i, p)));
  }

 private:
  Truncation truncation_;

  // When this flag is enabled the store also donates the reference ownership to
  // the memory object. Care should be taken to not overwrite a valid reference.
  // The current value of the memory location must be either 0 or must have just
  // been invalidated by a load steal.
  // This flag does not change at all the actual semantics of the instruction
  // and is ignored by all backends. It is only used by the optimizer.
  bool donate_;
};

// Obtains the address of a local variable. This instruction also takes an index
// `i`, used with the kConsts, kNames, kFastLocals and kFreeVars kinds.
class FrameVariableInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "frame_variable";
  static constexpr Value::Kind kKind = Value::kFrameVariable;
  Kind kind() const final { return kKind; }

  enum class FrameVariableKind : uint16_t {
    // Returns the PyFrameObject*.
    kFrame,
    // Returns the PyThreadState*.
    kThreadState,
    // Returns a PyObject*: code->co_consts[i].
    kConsts,
    // Returns a PyObject*: code->co_names[i].
    kNames,
    // Returns a PyCodeObject*.
    kCodeObject,
    // Returns a PyObject*.
    kBuiltins,
    // Returns a PyObject**: &fastlocals[i].
    kFastLocals,
    // Returns a PyObject*.
    kGlobals,
    // Returns a PyObject*.
    kLocals,
    // Returns a PyObject**: &fastlocals[nlocals + i].
    kFreeVars,
  };

  static constexpr absl::string_view kKinds[] = {
      "frame",    "thread_state", "consts",  "names",  "code_object",
      "builtins", "fastlocals",   "globals", "locals", "freevars"};

  FrameVariableInst() = default;
  explicit FrameVariableInst(FrameVariableKind kind, int64_t index)
      : index_(index), frame_variable_kind_(kind) {}

  FrameVariableKind frame_variable_kind() const { return frame_variable_kind_; }
  FrameVariableKind* mutable_frame_variable_kind() {
    return &frame_variable_kind_;
  }

  int64_t index() const { return index_; }
  int64_t* mutable_index() { return &index_; }

  template <typename P>
  static typename P::value_type Format(FrameVariableInst* i, P* p) {
    return p->CommaConcat(p->Enum(i->mutable_frame_variable_kind(), kKinds),
                          p->Imm(i->mutable_index()));
  }

 private:
  int64_t index_;
  FrameVariableKind frame_variable_kind_;
};

////////////////////////////////////////////////////////////////////////////////
// Calling

// Calls a C function by symbol. The callee must be in a known allowlilst.
class CallNativeInst final : public CallNativeBaseInst,
                             public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "call_native";
  static constexpr Value::Kind kKind = Value::kCallNative;
  Kind kind() const final { return kKind; }

  // TODO handle pure functions.
  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  CallNativeInst() = default;
  explicit CallNativeInst(Callee callee,
                          absl::Span<Value* const> arguments = {},
                          int32_t bytecode_offset = 0)
      : CallNativeBaseInst(callee, arguments),
        PreciseLocationInst(bytecode_offset) {}

  template <typename P>
  static typename P::value_type Format(CallNativeInst* i, P* p) {
    return p->Concat(CallNativeBaseInst::Format(i, p),
                     p->BytecodeOffset(i->mutable_bytecode_offset()));
  }
};

// A RematerializeInst is a lazily called CallNativeInst. It exists to
// allow eliding work on the fast path that needs to be done if we deoptimize.
//
// An example here is eliding an attribute load for a method call. If we are
// forced to deoptimize before the call, for example while computing arguments,
// we are forced to materialize the attribute load.
//
// A RematerializeInst is only run during deoptimization, if the deoptimizer
// finds a RematerializeInst in the value stack it will call the runtime
// function to materialize the true value.
class RematerializeInst final : public CallNativeBaseInst {
 public:
  static constexpr absl::string_view kMnemonic = "rematerialize";
  static constexpr Value::Kind kKind = Value::kRematerialize;
  Kind kind() const final { return kKind; }

  static constexpr bool kHasSideEffects = false;

  RematerializeInst() = default;
  explicit RematerializeInst(Callee callee,
                             absl::Span<Value* const> call_arguments = {})
      : CallNativeBaseInst(callee, call_arguments) {}
};

// Calls a Python value.
class CallPythonInst : public CallInst, public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "call_python";
  static constexpr Value::Kind kKind = Value::kCallPython;
  Kind kind() const override { return kKind; }

  // TODO: handle pure functions.
  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  // The index into the operands array of the callee and names, respectively.
  static constexpr int64_t kCalleeOperandIndex = 0;
  static constexpr int64_t kNamesOperandIndex = 1;

  // The calling convention to use; the fast convention is clearly the fastest,
  // but requires knowledge of the target function's signature.
  enum CallingConvention { kNormal, kFast };

  CallPythonInst() : CallInst(2) {}
  explicit CallPythonInst(Value* callee,
                          absl::Span<Value* const> call_arguments = {},
                          Value* names = nullptr, int32_t bytecode_offset = 0)
      : CallInst({callee, names}, call_arguments),
        PreciseLocationInst(bytecode_offset) {}

  Value* callee() { return operands()[0]; }
  const Value* callee() const { return operands()[0]; }
  Value** mutable_callee() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[0];
  }

  Value* names() { return operands()[1]; }
  const Value* names() const { return operands()[1]; }
  Value** mutable_names() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[1];
  }

  // Returns true if this call should use the fast calling convention. This is
  // more performant, but does not support keyword arguments or defaults.
  bool fastcall() const { return fastcall_; }
  void set_fastcall(bool fastcall) { fastcall_ = fastcall; }
  bool* mutable_fastcall() { return &fastcall_; }

  template <typename P>
  static typename P::value_type Format(CallPythonInst* i, P* p) {
    return p->Concat(p->Flag(i->mutable_fastcall(), "fast"),
                     p->Value(i->mutable_callee()),
                     p->OptionalValue(i->mutable_names(), "names"),
                     CallInst::FormatArguments(i, p),
                     p->BytecodeOffset(i->mutable_bytecode_offset()));
  }

 private:
  bool fastcall_ = false;
};

// Calls an attribute on a value. This is equivalent to
// (call_python (call_native s6::GetAttr $x, "attr"), ...).
//
// The attribute is stored as a string inside the Function's string table.
class CallAttributeInst final : public CallPythonInst {
 public:
  static constexpr absl::string_view kMnemonic = "call_attribute";
  static constexpr Value::Kind kKind = Value::kCallAttribute;
  Kind kind() const final { return kKind; }

  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  CallAttributeInst() = default;
  explicit CallAttributeInst(Value* object, StringTable::key_type attribute,
                             absl::Span<Value* const> call_arguments = {},
                             Value* names = nullptr,
                             int32_t getattr_bytecode_offset = 0,
                             int32_t call_python_bytecode_offset = 0)
      : CallPythonInst(object, call_arguments, names, getattr_bytecode_offset),
        call_python_bytecode_offset_(call_python_bytecode_offset),
        attribute_(attribute) {}

  StringTable::key_type attribute() const { return attribute_; }
  StringTable::key_type* mutable_attribute() { return &attribute_; }
  absl::string_view attribute_str() const {
    return parent()->GetStringTable().GetInternedString(attribute());
  }

  // CallAttributeInst doesn't have callee.
  Value* callee() = delete;
  const Value* callee() const = delete;
  Value** mutable_callee() = delete;

  // But we do have object, which uses the same storage.
  Value* object() { return operands()[0]; }
  const Value* object() const { return operands()[0]; }
  Value** mutable_object() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[0];
  }

  // A call_attribute has TWO bytecode offsets, because it is a combination of
  // a `%x = call_native s6::GetAttr` and `call_python %x`. The former is
  // returned by in `bytecode_offset()`, the latter is returned by
  // `call_python_bytecode_offset()`.
  int32_t call_python_bytecode_offset() const {
    return call_python_bytecode_offset_;
  }
  int32_t* mutable_call_python_bytecode_offset() {
    return &call_python_bytecode_offset_;
  }

  template <typename P>
  static typename P::value_type Format(CallAttributeInst* i, P* p) {
    return p->Concat(
        p->Value(i->mutable_object()), p->Str("::"),
        p->InternedString(i->mutable_attribute()),
        p->OptionalValue(i->mutable_names(), "names"),
        CallInst::FormatArguments(i, p),
        p->CommaConcat(
            p->BytecodeOffset(i->mutable_bytecode_offset()),
            p->BytecodeOffset(i->mutable_call_python_bytecode_offset())));
  }

 private:
  int32_t call_python_bytecode_offset_;
  StringTable::key_type attribute_;
};

// Calls a native function through a function pointer.
class CallNativeIndirectInst final : public CallInst,
                                     public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "call_native_indirect";
  static constexpr Value::Kind kKind = Value::kCallNativeIndirect;
  Kind kind() const final { return kKind; }

  // TODO handle pure native functions.
  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  CallNativeIndirectInst() : CallInst(1) {}
  explicit CallNativeIndirectInst(Value* callee,
                                  absl::Span<Value* const> call_arguments = {},
                                  int32_t bytecode_offset = 0)
      : CallInst({callee}, call_arguments),
        PreciseLocationInst(bytecode_offset) {}
  // `info` must outlive the CallNativeIndirect.
  explicit CallNativeIndirectInst(Value* callee,
                                  absl::Span<Value* const> call_arguments,
                                  const CalleeInfo& info,
                                  int32_t bytecode_offset = 0)
      : CallInst({callee}, call_arguments),
        PreciseLocationInst(bytecode_offset),
        info_(&info) {}

  Value* callee() { return operands()[0]; }
  const Value* callee() const { return operands()[0]; }
  Value** mutable_callee() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  bool HasInfo() const { return info_; }
  const CalleeInfo& info() const {
    S6_DCHECK(HasInfo());
    return *info_;
  }

  template <typename P>
  static typename P::value_type Format(CallNativeIndirectInst* i, P* p) {
    return p->Concat(p->Value(i->mutable_callee()),
                     CallInst::FormatArguments(i, p),
                     p->BytecodeOffset(i->mutable_bytecode_offset()));
  }

 private:
  // This pointer is non-owning, but it can be nullptr. If not null, it should
  // point to static data.
  const CalleeInfo* info_ = nullptr;
};

// Calls a native function with CPython's vectorcall calling convention.
//
// This implements PyCFunctionObjects with the METH_FASTCALL flag. Arguments
// are expected in a single array, rather than a tuple. We can achieve this by
// putting all arguments on the stack, like CallPythonInst, and passing a
// pointer to the stack to the callee.
class CallVectorcallInst final : public CallInst, public PreciseLocationInst {
 public:
  static constexpr absl::string_view kMnemonic = "call_vectorcall";
  static constexpr Value::Kind kKind = Value::kCallVectorcall;
  Kind kind() const final { return kKind; }

  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  // The index into the operands array of the callee, self and names,
  // respectively.
  static constexpr int64_t kCalleeOperandIndex = 0;
  static constexpr int64_t kSelfOperandIndex = 1;
  static constexpr int64_t kNamesOperandIndex = 2;

  CallVectorcallInst() : CallInst(3) {}
  explicit CallVectorcallInst(Value* callee, Value* self,
                              Value* names = nullptr,
                              absl::Span<Value* const> call_arguments = {},
                              int32_t bytecode_offset = 0)
      : CallInst({callee, self, names}, call_arguments),
        PreciseLocationInst(bytecode_offset) {}

  Value* callee() { return operands()[0]; }
  const Value* callee() const { return operands()[0]; }
  Value** mutable_callee() {
    S6_DCHECK_GE(operands().size(), 3);
    return &raw_operands()[0];
  }

  Value* self() { return operands()[1]; }
  const Value* self() const { return operands()[1]; }
  Value** mutable_self() {
    S6_DCHECK_GE(operands().size(), 3);
    return &raw_operands()[1];
  }

  Value* names() { return operands()[2]; }
  const Value* names() const { return operands()[2]; }
  Value** mutable_names() {
    S6_DCHECK_GE(operands().size(), 3);
    return &raw_operands()[2];
  }

  template <typename P>
  static typename P::value_type Format(CallVectorcallInst* i, P* p) {
    return p->Concat(p->Value(i->mutable_callee()),
                     p->OptionalValue(i->mutable_self(), "self"),
                     p->OptionalValue(i->mutable_names(), "names"),
                     CallInst::FormatArguments(i, p),
                     p->BytecodeOffset(i->mutable_bytecode_offset()));
  }

 private:
};

////////////////////////////////////////////////////////////////////////////////
// Safepoints

// A SafepointInst is a BytecodeBeginInst, YieldValueInst or a
// DeoptimizeIfSafepointInst. It contains a bytecode offset, value stack and try
// handler stack; enough information to construct an accurate interpreter frame.
class SafepointInst : public Instruction, public PreciseLocationInst {
 public:
  explicit SafepointInst() : SafepointInst(0) {}
  explicit SafepointInst(int32_t num_subclasses_operands) {
    auto& operands = raw_operands();
    operands.resize(num_subclasses_operands);
    operands.push_line();
    operands.push_line();
    operands.push_line();
    operands.push_line();
    operands.push_line();
  }
  explicit SafepointInst(absl::Span<Value* const> subclass_operands,
                         int32_t bytecode_offset,
                         absl::Span<Value* const> value_stack,
                         absl::Span<Value* const> fastlocals,
                         absl::Span<TryHandler const> try_stack,
                         absl::Span<Value* const> decrefs = {},
                         absl::Span<Value* const> increfs = {})
      : PreciseLocationInst(bytecode_offset) {
    auto& operands = raw_operands();
    absl::c_copy(subclass_operands, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(value_stack, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(fastlocals, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(decrefs, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(increfs, std::back_inserter(operands));
    operands.push_line();
    absl::c_copy(try_stack, std::back_inserter(try_stack_));
  }

  int64_t GetNumSubclassOperands() const {
    return const_raw_operands().line_size(0);
  }

  // Returns the value stack, with bottom-of-stack the first item and
  // top-of-stack the last item.
  absl::Span<Value* const> value_stack() {
    return const_raw_operands().line_span(1);
  }
  absl::Span<const Value* const> value_stack() const {
    return const_raw_operands().line_span(1);
  }
  MutableOperandList mutable_value_stack() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 6);
    return raw_operands().line(1);
  }

  // Returns the fastlocals.
  absl::Span<Value* const> fastlocals() {
    return const_raw_operands().line_span(2);
  }
  absl::Span<const Value* const> fastlocals() const {
    return const_raw_operands().line_span(2);
  }
  MutableOperandList mutable_fastlocals() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 6);
    return raw_operands().line(2);
  }

  // Checks if an index is in the stack or fastlocals part of the safepoint.
  bool IsInStackOrFastlocals(int64_t index) {
    return index >= const_raw_operands().line_start(1) &&
           index < const_raw_operands().line_end(2);
  }

  // Returns any values that must be decreffed when taking the safepoint.
  absl::Span<Value* const> decrefs() {
    return const_raw_operands().line_span(3);
  }
  absl::Span<const Value* const> decrefs() const {
    return const_raw_operands().line_span(3);
  }
  MutableOperandList mutable_decrefs() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 6);
    return raw_operands().line(3);
  }

  // Returns any values that must be increffed when taking the safepoint.
  absl::Span<Value* const> increfs() {
    return const_raw_operands().line_span(4);
  }
  absl::Span<const Value* const> increfs() const {
    return const_raw_operands().line_span(4);
  }
  MutableOperandList mutable_increfs() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 6);
    return raw_operands().line(4);
  }

  // Checks if an index is in the increfs or decrefs part of the safepoint.
  bool IsInIncrefsOrDecrefs(int64_t index) {
    return index >= const_raw_operands().line_start(3) &&
           index < const_raw_operands().line_end(4);
  }

  // Helper to add a value to the decref list smartly: If the value is in the
  // incref list it is removed instead of adding it to the decref list.
  void decref_value(Value* v) {
    auto incs = mutable_increfs();
    auto it = absl::c_find(incs, v);
    if (it == incs.end()) {
      mutable_decrefs().push_back(v);
    } else {
      incs.erase(it);
    }
  }

  // Helper to add a value to the incref list smartly: If the value is in the
  // decref list it is removed instead of adding it to the incref list.
  void incref_value(Value* v) {
    auto decs = mutable_decrefs();
    auto it = absl::c_find(decs, v);
    if (it == decs.end()) {
      mutable_increfs().push_back(v);
    } else {
      decs.erase(it);
    }
  }

  // Returns any extra values that are implicitly used by this instruction.
  absl::Span<Value* const> extras() {
    return const_raw_operands().line_span(5);
  }
  absl::Span<const Value* const> extras() const {
    return const_raw_operands().line_span(5);
  }
  MutableOperandList mutable_extras() {
    S6_DCHECK_GE(const_raw_operands().line_num(), 6);
    return raw_operands().line(5);
  }

  absl::Span<TryHandler const> try_handlers() const { return try_stack_; }
  std::vector<TryHandler>* mutable_try_handlers() { return &try_stack_; }

 protected:
  // A formatter for the "common tail" operands of a SafepointInst: the bytecode
  // offset, value stack and try handler.
  template <typename P>
  static typename P::value_type PartialFormat(SafepointInst* i, P* p) {
    return p->Concat(
        p->BytecodeOffset(i->mutable_bytecode_offset()),
        p->OptionalValueList(i->mutable_value_stack(), "stack [", "]"),
        p->OptionalValueList(i->mutable_fastlocals(), "fastlocals [", "]"),
        p->OptionalValueList(i->mutable_decrefs(), "decrefs [", "]"),
        p->OptionalValueList(i->mutable_increfs(), "increfs [", "]"),
        p->OptionalValueList(i->mutable_extras(), "extras [", "]"),
        p->TryHandlerList(i->mutable_try_handlers()));
  }

 private:
  // The stack of TryBlocks.
  std::vector<TryHandler> try_stack_;
};

// Marks the beginning of a bytecode instruction. This contains all the
// information required to reconstruct a CPython interpreter state: bytecode
// offset, value stack contents and try-block stack contents.
class BytecodeBeginInst final : public SafepointInst {
 public:
  static constexpr absl::string_view kMnemonic = "bytecode_begin";
  static constexpr Value::Kind kKind = Value::kBytecodeBegin;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;

  explicit BytecodeBeginInst() = default;
  explicit BytecodeBeginInst(int32_t bytecode_offset,
                             absl::Span<Value* const> value_stack,
                             absl::Span<Value* const> fastlocals,
                             absl::Span<TryHandler const> try_stack,
                             absl::Span<Value* const> decrefs = {},
                             absl::Span<Value* const> increfs = {})
      : SafepointInst({}, bytecode_offset, value_stack, fastlocals, try_stack,
                      decrefs, increfs) {}

  template <typename P>
  static typename P::value_type Format(BytecodeBeginInst* i, P* p) {
    return SafepointInst::PartialFormat(i, p);
  }
};

// Yields within a generator function. The function's state is saved, and can
// be resumed after the yield. YieldValueInst takes a single operand that is
// yielded to its caller, and its result is the value the caller sends back.
//
// Because generators may be deoptimized while paused, YieldValueInst also
// holds enough information to reconstruct an interpreter frame (it is a
// SafepointInst).
class YieldValueInst final : public SafepointInst {
 public:
  static constexpr absl::string_view kMnemonic = "yield_value";
  static constexpr Value::Kind kKind = Value::kYieldValue;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = true;
  static constexpr bool kHasSideEffects = true;
  static constexpr bool kClobbersAllRegisters = true;

  explicit YieldValueInst() : SafepointInst(1) {}
  explicit YieldValueInst(Value* yielded_value, int32_t bytecode_offset = 0,
                          absl::Span<Value* const> value_stack = {},
                          absl::Span<Value* const> fastlocals = {},
                          absl::Span<TryHandler const> try_stack = {},
                          absl::Span<Value* const> decrefs = {},
                          absl::Span<Value* const> increfs = {})
      : SafepointInst({yielded_value}, bytecode_offset, value_stack, fastlocals,
                      try_stack, decrefs, increfs) {}

  Value* yielded_value() { return operands()[0]; }
  const Value* yielded_value() const { return operands()[0]; }
  Value** mutable_yielded_value() {
    S6_DCHECK_GE(const_raw_operands().line_span(0).size(), 1);
    return &raw_operands()[0];
  }
  template <typename P>
  static typename P::value_type Format(YieldValueInst* i, P* p) {
    return p->Concat(p->Value(i->mutable_yielded_value()),
                     SafepointInst::PartialFormat(i, p));
  }
};

// Deoptimizes if a condition is true or optionally false. This instruction
// has enough information to materialize an interpreter frame (it is a
// SafepointInst) so does not need to act as control flow - if deoptimization
// occurs then no more Strongjit IR code is run.
//
// In this way it is distiguished from DeoptimizeIfInst, which deoptimizes but
// is not at a Safepoint boundary and needs to run more strongjit code (in the
// evaluator, usually) to get to a boundary.
class DeoptimizeIfSafepointInst final : public SafepointInst {
 public:
  static constexpr absl::string_view kMnemonic = "deoptimize_if_safepoint";
  static constexpr Value::Kind kKind = Value::kDeoptimizeIfSafepoint;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;

  explicit DeoptimizeIfSafepointInst() : SafepointInst(1) {}
  explicit DeoptimizeIfSafepointInst(Value* condition, bool negated,
                                     StringTable::key_type description,
                                     int32_t bytecode_offset,
                                     absl::Span<Value* const> value_stack,
                                     absl::Span<Value* const> fastlocals,
                                     absl::Span<TryHandler const> try_stack,
                                     absl::Span<Value* const> decrefs = {},
                                     absl::Span<Value* const> increfs = {})
      : SafepointInst({condition}, bytecode_offset, value_stack, fastlocals,
                      try_stack, decrefs, increfs),
        negated_(negated),
        description_(description) {}

  static constexpr int64_t kConditionOperandIndex = 0;
  Value* condition() { return operands()[0]; }
  const Value* condition() const { return operands()[0]; }
  Value** mutable_condition() {
    S6_DCHECK_GE(const_raw_operands().line_span(0).size(), 1);
    return &raw_operands()[0];
  }

  bool negated() const { return negated_; }
  void set_negated(bool negated) { negated_ = negated; }
  bool* mutable_negated() { return &negated_; }

  // A free-form text description of rationale of what the guard was testing.
  StringTable::key_type description() const { return description_; }
  StringTable::key_type* mutable_description() { return &description_; }
  absl::string_view description_str() const {
    return parent()->GetStringTable().GetInternedString(description());
  }

  template <typename P>
  static typename P::value_type Format(DeoptimizeIfSafepointInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->CommaConcat(p->Concat(p->Flag(i->mutable_negated(), "not"),
                                    p->Value(i->mutable_condition())),
                          SafepointInst::PartialFormat(i, p),
                          p->InternedString(i->mutable_description()));
  }

 private:
  bool negated_ = false;
  StringTable::key_type description_;
};

////////////////////////////////////////////////////////////////////////////////
// Profiling

// Adds `amount` to the profile counter. This indicates that approximately
// `amount` CPython bytecodes of work has been performed.
class AdvanceProfileCounterInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "advance_profile_counter";
  static constexpr Value::Kind kKind = Value::kAdvanceProfileCounter;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;

  AdvanceProfileCounterInst() = default;
  explicit AdvanceProfileCounterInst(int64_t amount) : amount_(amount) {}

  int64_t amount() const { return amount_; }
  int64_t* mutable_amount() { return &amount_; }

  template <typename P>
  static typename P::value_type Format(AdvanceProfileCounterInst* i, P* p) {
    return p->Imm(i->mutable_amount());
  }

 private:
  int64_t amount_;
};

// Abstract base class for profile instructions that all have a single interned
// string argument. Derived classes determine the mnemonic, and behavior of the
// instruction.
class ProfileInst : public Instruction {
 public:
  static constexpr bool kProducesValue = false;
  // Profile instructions are treated as having side effects so they are not
  // optimized away.
  static constexpr bool kHasSideEffects = true;

  ProfileInst() = default;
  explicit ProfileInst(GlobalInternTable::InternedString name) : name_(name) {}

  GlobalInternTable::InternedString name() const { return name_; }
  absl::string_view name_str() const { return name_.get(); }
  GlobalInternTable::InternedString* mutable_name() { return &name_; }

  template <typename P>
  static typename P::value_type Format(ProfileInst* i, P* p) {
    return p->InternedGlobalString(i->mutable_name());
  }

 private:
  GlobalInternTable::InternedString name_;
};

// Increments a named event counter managed by the `EventCounters` singleton.
// Each event counter counts the number of times that it has been incremented
// by reaching a trace point during the execution of compiled code.
class IncrementEventCounterInst final : public ProfileInst {
 public:
  static constexpr absl::string_view kMnemonic = "increment_event_counter";
  static constexpr Value::Kind kKind = Value::kIncrementEventCounter;
  Kind kind() const final { return kKind; }

  IncrementEventCounterInst() = default;
  using ProfileInst::ProfileInst;
};

// Begins a named trace event. These events are typically used to trace function
// execution. In this case a TraceBeginInst is inserted at the function entry
// point, and a matching TraceEndInst is inserted at all function exit points.
class TraceBeginInst final : public ProfileInst {
 public:
  static constexpr absl::string_view kMnemonic = "trace_begin";
  static constexpr Value::Kind kKind = Value::kTraceBegin;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = true;
  static constexpr bool kClobbersAllRegisters = true;

  TraceBeginInst() = default;
  using ProfileInst::ProfileInst;
};

// Ends a named trace event. The event should have been begun by a matching
// TraceBeginInst with the same name argument.
class TraceEndInst final : public ProfileInst {
 public:
  static constexpr absl::string_view kMnemonic = "trace_end";
  static constexpr Value::Kind kKind = Value::kTraceEnd;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = true;
  static constexpr bool kClobbersAllRegisters = true;

  TraceEndInst() = default;
  using ProfileInst::ProfileInst;
};

////////////////////////////////////////////////////////////////////////////////
// Boxing and unboxing

// Python types that can be unboxed.
enum class UnboxableType : int32_t {
  // Python Long.
  kPyLong,
  // Python Boolean.
  kPyBool,
  // Python Float.
  kPyFloat,
};

// Constructs a Python Long/Float/Boolean from the operand.
// It is assumed that this will never fail. Out of memory errors are not
// supported by S6.
class BoxInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "box";
  static constexpr Value::Kind kKind = Value::kBox;
  Kind kind() const final { return kKind; }

  static constexpr bool kProducesValue = true;
  static constexpr bool kClobbersAllRegisters = true;

  BoxInst() { raw_operands().resize(1); }
  BoxInst(UnboxableType type, Value* content) : type_(type) {
    raw_operands().push_back(content);
  }

  UnboxableType type() const { return type_; }
  UnboxableType* mutable_type() { return &type_; }

  Value* content() { return operands()[0]; }
  const Value* content() const { return operands()[0]; }
  Value** mutable_content() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  // Returns a callee that would be equivalent if called with CallNativeInst.
  Callee EquivalentCallee() {
    switch (type_) {
      case UnboxableType::kPyBool:
        return Callee::kPyBool_FromLong;
      case UnboxableType::kPyLong:
        return Callee::kPyLong_FromLong;
      case UnboxableType::kPyFloat:
        return Callee::kPyFloat_FromDouble;
    }
    S6_UNREACHABLE();
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(BoxInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Enum(i->mutable_type(), {"long", "bool", "float"}),
                     p->Value(i->mutable_content()));
  }

 private:
  UnboxableType type_;
};

// Extracts a native Long/Float/Boolean from a Python object.
// If the object is not of the correct type (e.g. PyFloat where PyLong expected)
// or, in the case of PyLong, its value exceeds 64 bits, then the "overflow"
// flag is set. This can be detected using "overflowed?" as the very next
// instruction.
class UnboxInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "unbox";
  static constexpr Value::Kind kKind = Value::kUnbox;
  Kind kind() const final { return kKind; }

  static constexpr bool kClobbersAllRegisters = true;

  UnboxInst() { raw_operands().resize(1); }
  UnboxInst(UnboxableType type, Value* boxed) : type_(type) {
    raw_operands().push_back(boxed);
  }

  UnboxableType type() const { return type_; }
  UnboxableType* mutable_type() { return &type_; }

  Value* boxed() { return operands()[0]; }
  const Value* boxed() const { return operands()[0]; }
  Value** mutable_boxed() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  // Unboxes the operand. Returns absl::nullopt on overflow or type error.
  absl::optional<int64_t> Evaluate(PyObject* boxed) const {
    switch (type()) {
      case UnboxableType::kPyLong:
        if (PyLong_Check(boxed)) {
          int overflow = 0;
          int64_t result = PyLong_AsLongAndOverflow(boxed, &overflow);
          if (overflow != 0) return absl::nullopt;
          return result;
        }
        return absl::nullopt;
      case UnboxableType::kPyBool:
        if (boxed == Py_True) return 1;
        if (boxed == Py_False) return 0;
        return absl::nullopt;
      case UnboxableType::kPyFloat:
        if (PyFloat_Check(boxed)) {
          return absl::bit_cast<int64_t>(PyFloat_AsDouble(boxed));
        }
        if (PyLong_Check(boxed)) {
          return absl::bit_cast<int64_t>(PyLong_AsDouble(boxed));
        }
        // Not a PyFloat.
        return absl::nullopt;
    }
    S6_UNREACHABLE();
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(UnboxInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Enum(i->mutable_type(), {"long", "bool", "float"}),
                     p->Value(i->mutable_boxed()));
  }

 private:
  UnboxableType type_;
};

// Determines whether the operand overflowed.
// The operand must be an arithmetic or unbox instruction, and must occur
// immediately prior to this instruction in the same block.
class OverflowedInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "overflowed?";
  static constexpr Value::Kind kKind = Value::kOverflowed;
  Kind kind() const final { return kKind; }

  OverflowedInst() { raw_operands().resize(1); }
  explicit OverflowedInst(Value* arithmetic_value) {
    raw_operands().push_back(arithmetic_value);
  }

  Value* arithmetic_value() { return operands()[0]; }
  const Value* arithmetic_value() const { return operands()[0]; }
  Value** mutable_arithmetic_value() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  bool Evaluate(const UnboxInst& inst, PyObject* boxed) const {
    return inst.Evaluate(boxed) == absl::nullopt;
  }

  bool Evaluate(const UnaryInst& inst, int64_t value) const {
    if (inst.IsDoubleType()) return false;
    if (inst.kind() == Value::kNegate) {
      return arithmetic::Negate(value).overflowed;
    }
    return false;
  }

  bool Evaluate(const BinaryInst& inst, int64_t lhs, int64_t rhs) const {
    if (inst.IsDoubleType()) return false;

    switch (inst.kind()) {
      case Value::kAdd:
        return arithmetic::Add(lhs, rhs).overflowed;
      case Value::kSubtract:
        return arithmetic::Subtract(lhs, rhs).overflowed;
      case Value::kMultiply:
        return arithmetic::Multiply(lhs, rhs).overflowed;
      case Value::kDivide:
        return arithmetic::Divide(lhs, rhs).overflowed;
      case Value::kRemainder:
        return arithmetic::Remainder(lhs, rhs).overflowed;
      case Value::kShiftLeft:
        return arithmetic::ShiftLeft(lhs, rhs).overflowed;
      case Value::kShiftRightSigned:
        return arithmetic::ShiftRight(lhs, rhs).overflowed;
      default:
        return false;
    }
  }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(OverflowedInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(i->mutable_arithmetic_value());
  }
};

// Determines whether the operand is floating point zero, either positive or
// negative.
class FloatZeroInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "float_zero?";
  static constexpr Value::Kind kKind = Value::kFloatZero;
  Kind kind() const final { return kKind; }

  FloatZeroInst() { raw_operands().resize(1); }
  explicit FloatZeroInst(Value* float_value) {
    raw_operands().push_back(float_value);
  }

  Value* float_value() { return operands()[0]; }
  const Value* float_value() const { return operands()[0]; }
  Value** mutable_float_value() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  bool Evaluate(double arg) const { return arg == 0.; }

  // Formatter, for parsing and printing.
  template <typename P>
  static typename P::value_type Format(FloatZeroInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(i->mutable_float_value());
  }
};

////////////////////////////////////////////////////////////////////////////////
// Python class manipulation

// Given an object, returns its class ID. The object operand cannot be null.
class GetClassIdInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "get_class_id";
  static constexpr Value::Kind kKind = Value::kGetClassId;
  Kind kind() const final { return kKind; }

  GetClassIdInst() { raw_operands().resize(1); }
  explicit GetClassIdInst(Value* object) { raw_operands().push_back(object); }

  Value* object() { return operands()[0]; }
  const Value* object() const { return operands()[0]; }
  Value** mutable_object() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  template <typename P>
  static typename P::value_type Format(GetClassIdInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(i->mutable_object());
  }
};

// Given an object, returns its __dict__ or zero on failure.
//
// If the dictoffset or type are known, this instruction can operate more
// efficiently.
//
// The object operand cannot be null.
class GetObjectDictInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "get_object_dict";
  static constexpr Value::Kind kKind = Value::kGetObjectDict;
  Kind kind() const final { return kKind; }

  GetObjectDictInst() { raw_operands().resize(1); }
  explicit GetObjectDictInst(Value* object, int64_t dictoffset = 0,
                             int64_t type = 0)
      : dictoffset_(dictoffset), type_(type) {
    raw_operands().push_back(object);
  }

  Value* object() { return operands()[0]; }
  const Value* object() const { return operands()[0]; }
  Value** mutable_object() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  // Returns the expected dict offset. This is zero if there is no expected
  // dict offset.
  int64_t dictoffset() const { return dictoffset_; }
  int64_t* mutable_dictoffset() { return &dictoffset_; }

  // Returns the expected type. This is zero if there is no expected type.
  int64_t type() const { return type_; }
  int64_t* mutable_type() { return &type_; }

  template <typename P>
  static typename P::value_type Format(GetObjectDictInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Value(i->mutable_object()), p->Str("dictoffset"),
                     p->Imm(i->mutable_dictoffset()), p->Str("type"),
                     p->Imm(i->mutable_type()));
  }

 private:
  int64_t dictoffset_ = 0;
  int64_t type_ = 0;
};

// Given an object's __dict__, returns its class ID.
//
// The dict operand cannot be null.
class GetInstanceClassIdInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "get_instance_class_id";
  static constexpr Value::Kind kKind = Value::kGetInstanceClassId;
  Kind kind() const final { return kKind; }

  GetInstanceClassIdInst() { raw_operands().resize(1); }
  explicit GetInstanceClassIdInst(Value* dict) {
    raw_operands().push_back(dict);
  }

  Value* dict() { return operands()[0]; }
  const Value* dict() const { return operands()[0]; }
  Value** mutable_dict() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  template <typename P>
  static typename P::value_type Format(GetInstanceClassIdInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Value(i->mutable_dict());
  }
};

// Checks if an object has a particular class ID. Returns nonzero on success,
// zero on failure.
//
// The object operand cannot be null.
class CheckClassIdInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "check_class_id";
  static constexpr Value::Kind kKind = Value::kCheckClassId;
  Kind kind() const final { return kKind; }

  CheckClassIdInst() { raw_operands().resize(1); }
  explicit CheckClassIdInst(Value* object, int64_t class_id)
      : class_id_(class_id) {
    raw_operands().push_back(object);
  }

  // The object to check the class of.
  Value* object() { return operands()[0]; }
  const Value* object() const { return operands()[0]; }
  Value** mutable_object() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  // The expected class ID.
  int64_t class_id() const { return class_id_; }
  int64_t* mutable_class_id() { return &class_id_; }

  // The expected Class.
  Class* class_(ClassManager& mgr = ClassManager::Instance()) const {
    return mgr.GetClassById(class_id_);
  }

  template <typename P>
  static typename P::value_type Format(CheckClassIdInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Value(i->mutable_object()), p->Str("class_id"),
                     p->Imm(i->mutable_class_id()));
  }

 private:
  int64_t class_id_;
};

// Given an object's __dict__, loads __dict__.ma_values[index].
class LoadFromDictInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "load_from_dict";
  static constexpr Value::Kind kKind = Value::kLoadFromDict;
  Kind kind() const final { return kKind; }

  LoadFromDictInst() { raw_operands().resize(1); }
  LoadFromDictInst(Value* dict, int64_t index, DictKind dict_kind)
      : index_(index), dict_kind_(dict_kind) {
    raw_operands().push_back(dict);
  }

  DictKind dict_kind() const { return dict_kind_; }
  DictKind* mutable_dict_kind() { return &dict_kind_; }

  Value* dict() { return operands()[0]; }
  const Value* dict() const { return operands()[0]; }
  Value** mutable_dict() {
    S6_DCHECK_GE(operands().size(), 1);
    return &raw_operands()[0];
  }

  int64_t index() const { return index_; }
  int64_t* mutable_index() { return &index_; }

  template <typename P>
  static typename P::value_type Format(LoadFromDictInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->CommaConcat(
        p->Value(i->mutable_dict()), p->Imm(i->mutable_index()),
        p->Enum(i->mutable_dict_kind(),
                {"empty", "split", "combined", "not_contiguous"}));
  }

 private:
  int64_t index_;
  DictKind dict_kind_;
};

// Given an object's __dict__, stores to __dict__.ma_values[index] and returns
// the value replaced.
class StoreToDictInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "store_to_dict";
  static constexpr Value::Kind kKind = Value::kStoreToDict;
  static constexpr bool kHasSideEffects = true;
  Kind kind() const final { return kKind; }

  StoreToDictInst() { raw_operands().resize(2); }
  StoreToDictInst(Value* value, Value* dict, int64_t index, DictKind dict_kind)
      : index_(index), dict_kind_(dict_kind) {
    raw_operands().push_back(value);
    raw_operands().push_back(dict);
  }

  DictKind dict_kind() const { return dict_kind_; }
  DictKind* mutable_dict_kind() { return &dict_kind_; }

  Value* value() { return operands()[0]; }
  const Value* value() const { return operands()[0]; }
  Value** mutable_value() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[0];
  }

  Value* dict() { return operands()[1]; }
  const Value* dict() const { return operands()[1]; }
  Value** mutable_dict() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[1];
  }

  int64_t index() const { return index_; }
  int64_t* mutable_index() { return &index_; }

  template <typename P>
  static typename P::value_type Format(StoreToDictInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(
        p->Value(i->mutable_value()), p->Str("into"),
        p->CommaConcat(p->Value(i->mutable_dict()), p->Imm(i->mutable_index())),
        p->Enum(i->mutable_dict_kind(),
                {"empty", "split", "combined", "not_contiguous"}));
  }

 private:
  int64_t index_;
  DictKind dict_kind_;
};

// Represents a constant that happens to be the result of an attribute lookup
// executed at compile time.
class ConstantAttributeInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "constant_attribute";
  static constexpr Value::Kind kKind = Value::kConstantAttribute;
  Kind kind() const final { return kKind; }

  ConstantAttributeInst() {}
  ConstantAttributeInst(int64_t class_id, StringTable::key_type attribute)
      : class_id_(class_id), attribute_(attribute) {}

  int64_t class_id() const { return class_id_; }
  int64_t* mutable_class_id() { return &class_id_; }

  StringTable::key_type attribute() const { return attribute_; }
  StringTable::key_type* mutable_attribute() { return &attribute_; }
  absl::string_view attribute_str() const {
    return parent()->GetStringTable().GetInternedString(attribute());
  }

  const Attribute& LookupAttribute(ClassManager& mgr) const {
    return *mgr.GetClassById(class_id_)
                ->attributes()
                .find(mgr.InternString(attribute_str()))
                ->second;
  }

  template <typename P>
  static typename P::value_type Format(ConstantAttributeInst* i, P* p) {
    return p->Concat(p->InternedString(i->mutable_attribute()), p->Str("of"),
                     p->Class(i->mutable_class_id()));
  }

 private:
  int64_t class_id_ = 0;
  StringTable::key_type attribute_;
};

// Sets the class ID of an object. The object and its instance dictionary are
// passed, and the ma_version_tag of the instance dictionary is written.
class SetObjectClassInst final : public Instruction {
 public:
  static constexpr absl::string_view kMnemonic = "set_object_class";
  static constexpr Value::Kind kKind = Value::kSetObjectClass;
  static constexpr bool kProducesValue = false;
  static constexpr bool kHasSideEffects = true;
  Kind kind() const final { return kKind; }

  SetObjectClassInst() { raw_operands().resize(2); }
  SetObjectClassInst(Value* object, Value* dict, int64_t class_id)
      : class_id_(class_id) {
    raw_operands().push_back(object);
    raw_operands().push_back(dict);
  }

  Value* object() { return operands()[0]; }
  const Value* object() const { return operands()[0]; }
  Value** mutable_object() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[0];
  }

  Value* dict() { return operands()[1]; }
  const Value* dict() const { return operands()[1]; }
  Value** mutable_dict() {
    S6_DCHECK_GE(operands().size(), 2);
    return &raw_operands()[1];
  }

  int64_t class_id() const { return class_id_; }
  int64_t* mutable_class_id() { return &class_id_; }

  template <typename P>
  static typename P::value_type Format(SetObjectClassInst* i, P* p) {
    AssertIsInstructionFormatter<P>();
    return p->Concat(p->Value(i->mutable_object()), p->Value(i->mutable_dict()),
                     p->Imm(i->mutable_class_id()));
  }

 private:
  int64_t class_id_;
};

////////////////////////////////////////////////////////////////////////////////
// Deoptimize asynchronously

// Returns nonzero if the s6::CodeObject's deoptimized() flag is set. Emit this
// and a DeoptimizeSafepointInst as assumption guards.
class DeoptimizedAsynchronouslyInst final : public Instruction {
 public:
  static constexpr Value::Kind kKind = Value::kDeoptimizedAsynchronously;
  static constexpr absl::string_view kMnemonic = "deoptimized_asynchronously?";
  Kind kind() const final { return kKind; }

  DeoptimizedAsynchronouslyInst() = default;
};
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTIONS_H_
