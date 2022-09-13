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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "strongjit/block.h"
#include "strongjit/string_table.h"
#include "strongjit/util.h"
#include "strongjit/value.h"
#include "utils/intrusive_list.h"

namespace deepmind::s6 {

class TryHandler;
class Instruction;

using MutableOperandList = MultiLineVector<Value*, 2>::Line;

////////////////////////////////////////////////////////////////////////////////
// Instruction Modification Listener

// A client or listener class that will be informed whenever a Value may
// potentially be mutated.
class InstructionModificationListener {
 public:
  virtual ~InstructionModificationListener();

  // A new Value has been created.
  virtual void InstructionAdded(Instruction* inst) = 0;

  // A value has been erased.
  virtual void InstructionErased(Instruction* inst) = 0;

  // Called whenever Instruction::mutable_operands() is accessed.
  // The instructions's operand list may be modified in the future.
  virtual void OperandsMayBeModified(Instruction* inst) = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Instruction Formatter

// An InstructionFormatter is an object called to parse or print an Instruction.
// An Instruction's static Format method uses an instance of
// InstructionFormatter to implement either parsing or printing. The calls to
// InstructionFormatter can be considered like a Builder that builds up the
// lexical syntax of the instruction and binds operands, immediates, enums and
// other class state. The InstructionFormatter takes mutable objects (value
// lists, operands etc) because the parser will use this interface to build up
// an Instruction in-place. For this reason all instructions are required to
// have a default empty constructor.
//
// Note that this is an abstract interface; Format() will be invoked with an
// implementation defined subclass of this interface.
class InstructionFormatter {
 public:
  // Instruction::Format uses value_type to determine its return type.
  using value_type = void;

  // Defines a sequence, separated by spaces. For example:
  //  p.Concat(p.Value(), p.Value()) -> "%0 %1"
  template <typename... T>
  std::string Concat(T... t);

  // Defines a sequence, separated by a comma. For example:
  //  p.CommaConcat(p.Value(), p.Value()) -> "%0, %1"
  template <typename... T>
  std::string CommaConcat(T... t);

  // Defines an integer immediate. For example:
  //  p.Imm(&i) -> "$42"
  std::string Imm(int64_t* imm);

  // Defines an integer bytecode offset. For example:
  //  p.BytecodeOffset(&i) -> "@42"
  std::string BytecodeOffset(int32_t* imm);

  // Defines a single value (which must not be a block). For example:
  //   p.Value(&v) -> "%0"
  std::string Value(Value** v);

  // Defines a single value (which must not be a block) that may be omitted.
  // For example:
  //   p.OptionalValue(&v) -> "%0" OR ""
  //   p.OptionalValue(&v, "prefix") -> "prefix %0" OR ""
  std::string OptionalValue(class Value** v, absl::string_view prefix = "");

  // Defines a single block. For example:
  //   p.Value(&block) -> "&0"
  std::string Value(Block** v);

  // Defines a single block. For example:
  //   p.Value(&block) -> "&0"
  // This overload computes the block lazily.
  std::string Value(std::function<Block**()> fn);

  // Defines a single block that may be omitted. For example:
  //   p.OptionalValue(&block) -> "&0" OR "".
  std::string OptionalValue(Block** v);

  // Defines a comma-separated list of values with custom open and close tokens.
  // For example:
  //   p.ValueList(list, "{", "}") -> "{%0, %1, %2}"
  std::string ValueList(MutableOperandList list,
                        absl::string_view open_bracket = "",
                        absl::string_view close_bracket = "");

  // Defines a comma-separated list of values with custom open and close tokens.
  // Unlike ValueList the list can be empty, in which case the open and close
  // tokens are not printed. For example:
  //   p.OptionalValueList(list, "{", "}") -> "{%0, %1, %2}"
  //   p.OptionalValueList({}, "{", "}") -> ""
  std::string OptionalValueList(MutableOperandList list,
                                absl::string_view open_bracket = "",
                                absl::string_view close_bracket = "");

  // Defines an enumeration, with string values for each integer value in
  // the enumeration. For example:
  //   enum E { kA, kB, kC } e;
  //   p.Enum(&e, {"a", "b", "c"}) -> "a" (or "b", or "c").
  template <typename T>
  std::string Enum(T* t, absl::Span<const absl::string_view> values);

  // Defines an enumeration that is also a bitfield. The `values` arrays gives
  // strings for every bit that could be set in T, least-significant-bit first.
  //
  // For example:
  //   enum E { kA, kB, kC } e;
  //   p.Enum(&e, {"a", "b", "c"}) -> "a|b".
  template <typename T>
  std::string EnumBitfield(T* t, absl::Span<const absl::string_view> values);

  // Defines a string token with no semantics.
  std::string Str(absl::string_view s) { return std::string(s); }

  // Defines an InternedString - a string with storage inside a Function.
  std::string InternedString(StringTable::key_type* id);

  // Defines a string token that, if present, sets the given boolean to true.
  std::string Flag(bool* b, absl::string_view flag);

  // Defines a list of BytecodeBeginInst::TryHandler objects.
  std::string TryHandlerList(std::vector<TryHandler>* try_handlers);

  // Defines a list of immediates.
  std::string ImmList(std::vector<int64_t>* list);

  // Defines a Class.
  std::string Class(int64_t* class_id);
};

template <typename P>
static void AssertIsInstructionFormatter() {
  static_assert(std::is_base_of<InstructionFormatter, P>::value,
                "Must be subclass of InstructionFormatter!");
}

////////////////////////////////////////////////////////////////////////////////
// Instruction

// An Instruction represents a single computation or action that uses a certain
// number of Values as operands.
// Each instruction subclass organizes the operands as it wants so the shape
// of the operands container is not mutable from the parent Instruction class.
//
// An Instruction is stored within a block (`parent`), but its memory belong to
// a function (`function`). Generally that function will be the grand-parent of
// the instruction in the sense that it is the parent of the parent block.
//
// The linked list represents the instructions in program order. The order of
// instructions in the containing function is semantically irrelevant.
//
// An Instruction also behaves as a container of its operands (const-only).
class Instruction : public Value, public IntrusiveLink<Instruction> {
 public:
  // Metaprogramming for parser and printer. Most Values produce an output.
  static constexpr bool kProducesValue = true;

  // Metadata to allow removal of instructions with no side-effects if output
  // value is not read. Most instructions have no side-effects.
  static constexpr bool kHasSideEffects = false;

  // Metadata indicating whether generated assembly for an instruction
  // clobbers all registers.
  static constexpr bool kClobbersAllRegisters = false;

  // An Instruction is a container of its operands, so define value_type.
  using value_type = Value*;

  // Instruction as a container only provides a non mutable interface of
  // operands.
  using const_iterator = const value_type*;

  Instruction() = default;
  Instruction(Instruction&&) = delete;
  Instruction& operator=(const Instruction&) = delete;
  Instruction& operator=(Instruction&&) = delete;

 protected:
  // Protected as derived classes need to be copied by `CloneWithNewOperands`.
  // The copy constructor of derived classes should not be called directly:
  // use CloneWithNewOperands instead.
  // The copied instruction has function_, IntrusiveLink next_ and prev_ set to
  // null.
  Instruction(const Instruction& instr)
      : Value(static_cast<const Value&>(instr)),
        IntrusiveLink<Instruction>(),
        function_(nullptr),
        operands_(instr.operands_) {}

 public:
  Block* parent() { return parent_; }
  const Block* parent() const { return parent_; }

  // Operands operations.
  absl::Span<const Value* const> operands() const { return operands_; }
  absl::Span<Value* const> operands() { return operands_; }
  absl::Span<Value*> mutable_operands() {
    CallListenerMutated();
    return operands_.span();
  }

  const_iterator begin() const { return operands_.begin(); }
  const_iterator end() const { return operands_.end(); }

  // Replaces all occurrences of `from` in the operands list to `to`.
  void ReplaceUsesOfWith(Value* from, Value* to) {
    CallListenerMutated();
    for (Value*& op : operands_) {
      if (op == from) op = to;
    }
  }

  // Removes all occurrences of `v` from the operand list.
  // TODO: This can easily break subclass invariants. Remove it?
  void EraseFromOperandList(const Value* v) {
    CallListenerMutated();
    auto it = operands_.begin();
    while (it != operands_.end()) {
      if (*it == v) {
        it = operands_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Returns this instruction as an iterator into its parent block.
  Block::iterator GetIterator() { return Block::iterator(this); }
  Block::const_iterator GetIterator() const {
    return Block::const_iterator(this);
  }

  // Unlinks this instruction from its parent, but does not delete it.
  void RemoveFromParent() { parent_->UnlinkInstruction(this); }

  // Unlinks this instruction from its parent and deletes it.
  void erase() {
    // Call the listener *before* unlinking in case the listener wants to know
    // our parent block.
    if (parent_) parent_->UnlinkInstruction(this);
    // Note that we never delete anything right now; we leave it for Function to
    // clean up.
    CallListenerErased();
  }

  // Define a default formatter for printing and parsing of instructions
  // without operands (only the mnemonic will be printed).
  template <typename P>
  static typename P::value_type Format(Instruction* v, P* p) {
    AssertIsInstructionFormatter<P>();
    S6_CHECK(v->operands().empty());
    return p->Str("");
  }

  Function* function() { return function_; }
  const Function* function() const { return function_; }

  // Creates a deep copy of this instruction in `function`, remapping any
  // operands using the map that is provided (leaving them as-is if no entry is
  // present).
  Instruction* CloneWithNewOperands(
      Function& function,
      const absl::flat_hash_map<const Value*, Value*>& mapping = {}) const;

  // Creates a deep copy of this instruction in the same function.
  // This is not const because we get the right to mutate the owning function
  // from this Instruction.
  Instruction* Clone() { return CloneWithNewOperands(*function()); }

 protected:
  MultiLineVector<Value*, 2>& raw_operands() {
    CallListenerMutated();
    return operands_;
  }
  const MultiLineVector<Value*, 2>& const_raw_operands() const {
    return operands_;
  }

  void CallListenerErased();
  void CallListenerMutated();

 private:
  friend class Block;     // Block can set parent_.
  friend class Function;  // Function can set function_.

  Block* parent_ = nullptr;
  Function* function_ = nullptr;

  MultiLineVector<Value*, 2> operands_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_H_
