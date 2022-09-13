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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FUNCTION_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FUNCTION_H_

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"
#include "classes/class_manager.h"
#include "core_util.h"
#include "strongjit/base.h"
#include "strongjit/cursor.h"
#include "strongjit/instructions.h"
#include "strongjit/string_table.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "strongjit/value_map.h"
#include "type_feedback.h"
#include "utils/intrusive_list.h"

namespace deepmind::s6 {

// A Function has a single entry point and owns all of its values. A function
// is moveable but not copyable.
//
// A Function does not have arguments; arguments are held in fast locals,
// which are accessible by load/store instructions.
//
// TODO: Add a mechanism to defragment Functions and clean up unused
// Instructions.
class Function {
 public:
  using iterator = IntrusiveList<Block>::iterator;
  using const_iterator = IntrusiveList<Block>::const_iterator;
  using reverse_iterator = IntrusiveList<Block>::reverse_iterator;
  using const_reverse_iterator = IntrusiveList<Block>::const_reverse_iterator;

  // TypeFeedback is keyed by <PcValue, operand_index>.
  using TypeFeedback = absl::flat_hash_map<std::pair<PcValue, int64_t>,
                                           ClassDistributionSummary>;

  explicit Function(absl::string_view name)
      : name_(name), string_table_(absl::make_unique<StringTable>()) {}
  Function(Function&& other)
      : name_(std::move(other.name_)),
        blocks_(std::move(other.blocks_)),
        block_storage_(std::move(other.block_storage_)),
        values_(std::move(other.values_)),
        string_table_(std::move(other.string_table_)),
        type_feedback_(std::move(other.type_feedback_)),
        classes_relied_upon_(std::move(other.classes_relied_upon_)),
        is_traced_(std::move(other.is_traced_)) {
    for (Block& block : blocks_) {
      block.parent_ = this;
    }
    for (FixedLengthValue& value : values_) {
      if (auto inst = dyn_cast<Instruction>(value.AsValue())) {
        inst->function_ = this;
      }
    }
  }

  // Functions can't be copied.
  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;

  virtual ~Function() = default;

  iterator begin() { return blocks_.begin(); }
  iterator end() { return blocks_.end(); }
  const_iterator begin() const { return blocks_.begin(); }
  const_iterator end() const { return blocks_.end(); }
  reverse_iterator rbegin() { return blocks_.rbegin(); }
  reverse_iterator rend() { return blocks_.rend(); }
  const_reverse_iterator rbegin() const { return blocks_.rbegin(); }
  const_reverse_iterator rend() const { return blocks_.rend(); }

  Block& entry() { return *begin(); }
  const Block& entry() const { return *begin(); }

  Cursor FirstInstruction() {
    if (blocks_.empty()) return Cursor();
    return Cursor(begin()->begin());
  }

  Cursor LastInstruction() {
    if (blocks_.empty()) return Cursor();
    return Cursor(--rbegin()->end());
  }

  // Creates a new Block and inserts it into the block list after `insert_pt`.
  Block* CreateBlock(iterator insert_pt) {
    Block* b = &block_storage_.emplace_back(this, GetStringTable());
    blocks_.insert(insert_pt, b);
    return b;
  }

  // Creates a new Block and inserts it at the end of the block list.
  Block* CreateBlock() {
    Block* b = &block_storage_.emplace_back(this, GetStringTable());
    blocks_.push_back(b);
    return b;
  }

  // Adds an argument to the given block at the end of its argument list.
  BlockArgument* CreateBlockArgument(Block* block) {
    BlockArgument* arg = values_.emplace_back().emplace<BlockArgument>(block);
    block->block_arguments_.push_back(arg);
    return arg;
  }

  // Creates a new Instruction. This is the only correct method for creating
  // an instruction.
  template <class T, typename... Args>
  std::enable_if_t<std::is_base_of<Instruction, T>::value, T*> Create(
      Args&&... args) {
    T* t = values_.emplace_back().emplace<T>(std::forward<Args>(args)...);
    t->function_ = this;
    if (listener_) {
      listener_->InstructionAdded(t);
    }
    return t;
  }

  // Unlinks a block from this function. The storage is still owned by this
  // function.
  void UnlinkBlock(Block* b) { blocks_.erase(b); }

  // Inserts a block into this function.
  void insert(Block* b, iterator insert_pt) {
    if (blocks_.is_linked(b)) {
      blocks_.erase(b);  // To be safe.
    }
    blocks_.insert(insert_pt, b);
  }

  // Returns the number of blocks in this function.
  int64_t num_blocks() const { return blocks_.size(); }

  // Returns an upper bound on the number of Instructions within this function.
  // This is only an upper bound, and not at all accurate.
  int64_t capacity() const { return values_.size(); }

  // Performs a stable sort of the blocks list, with the given comparator.
  void SortBlocks(
      const std::function<bool(const Block*, const Block*)>& compare) {
    // The blocks list is a linked list, so copy the current content to a
    // vector, sort that, then reinitialize the list in sorted order.
    std::vector<Block*> blocks;
    blocks.reserve(num_blocks());
    for (Block& b : *this) {
      blocks.push_back(&b);
    }
    absl::c_stable_sort(blocks, compare);
    blocks_.clear();
    for (Block* b : blocks) {
      blocks_.push_back(b);
    }
  }

  // Returns the type feedback, used for optimizations.
  TypeFeedback& type_feedback() { return type_feedback_; }
  const TypeFeedback& type_feedback() const { return type_feedback_; }

  // Returns the type feedback for a bytecode offset.
  absl::optional<ClassDistributionSummary> GetTypeFeedbackForBytecodeOffset(
      int64_t offset, int64_t operand_index = 0) const {
    auto it = type_feedback_.find({PcValue::FromOffset(offset), operand_index});
    if (it != type_feedback_.end()) return it->second;
    return {};
  }

  // Returns the name of this function.
  absl::string_view name() const { return name_; }

  // Only for use by the Parser, which needs to construct a Function before it
  // knows its name.
  void set_name(absl::string_view name) { name_ = name; }

  // Returns the string table for this function.
  StringTable* GetStringTable() { return string_table_.get(); }

  const StringTable* GetStringTable() const { return string_table_.get(); }

  // Notes that the implementation of this function relies upon the current
  // behavior of `cls`. If the behavior changes, this function is invalid.
  void AddReliedUponClass(const Class* cls) {
    classes_relied_upon_.push_back(const_cast<Class*>(cls));
  }

  // Returns the list of Classes this function relies upon, and clears the list.
  std::vector<Class*> ConsumeReliedUponClasses() {
    std::vector<Class*> l;
    classes_relied_upon_.swap(l);
    STLSortAndRemoveDuplicates(l);
    return l;
  }

  bool is_traced() const { return is_traced_; }
  void MarkTraced() { is_traced_ = true; }

  // Subscribes the given listener object to any modifications to operand lists.
  // The storage for the listener must outlive this Function (or live to the
  // next ClearInstructionModificationListener() call).
  //
  // REQUIRES: There is no current active subscriber.
  void SetInstructionModificationListener(
      InstructionModificationListener* listener) {
    S6_CHECK(!listener_)
        << "Cannot set multiple InstructionModificationListeners!";
    listener_ = listener;
  }

  // Returns the instruction modification listener.
  InstructionModificationListener* listener() const { return listener_; }

  // Removes an operand modification listener added by
  // SetInstructionModificationListener. Returns the existing listener.
  InstructionModificationListener* ClearInstructionModificationListener() {
    auto l = listener_;
    listener_ = nullptr;
    return l;
  }

 private:
  // The function name.
  std::string name_;

  // The list of Blocks in this function, in program order.
  IntrusiveList<Block> blocks_;

  // The storage for blocks_.
  std::deque<Block> block_storage_;

  // A FixedLengthValue contains a Value but is always 128 bytes in length.
  class FixedLengthValue {
   public:
    static constexpr int64_t kSize = 128;

    FixedLengthValue() {}
    ~FixedLengthValue();

    template <class T, typename... Args>
    T* emplace(Args&&... args) {
      static_assert(sizeof(T) <= sizeof(FixedLengthValue),
                    "Value type too large!");
      static_assert(std::is_base_of_v<Value, T>);
      return new (padding_.data()) T(std::forward<Args>(args)...);
    }

    Value* AsValue() { return reinterpret_cast<Value*>(padding_.data()); }

   private:
    std::array<uint8_t, kSize> padding_;
  };

  // Linear container that does not reallocate on insert. This holds all
  // allocated values in allocation order. The order is not related to program
  // order.
  std::deque<FixedLengthValue> values_;

  std::unique_ptr<StringTable> string_table_;

  // The type feedback information.
  TypeFeedback type_feedback_;

  // The list of Classes this function relies upon. If any of these Classes
  // change behavior, this function is invalid.
  std::vector<Class*> classes_relied_upon_;

  // True if trace_begin/end instructions have been inserted.
  bool is_traced_ = false;

  // An listener that is called whenever a Value's operands change.
  InstructionModificationListener* listener_ = nullptr;
};

template <class T, typename... Args>
T* Block::Create(Args&&... args) {
  T* inst = parent_->Create<T>(std::forward<Args>(args)...);
  push_back(inst);
  return inst;
}

template <class T, typename... Args>
T* BlockInserter::Create(Args&&... args) {
  T* inst = b_->parent()->Create<T>(std::forward<Args>(args)...);
  b_->insert(insert_pt_, inst);
  return inst;
}

ValueNumbering ComputeValueNumbering(const Function& f);

// Performs simple verification on the given Function.
//
// Ensures that all value uses and block predecessors are valid.
// Ensures successor lists agree with block predecessor lists.
absl::Status VerifyFunction(const Function& f);

// Snapshots the given type feedback into `f`.
absl::Status SnapshotTypeFeedback(
    absl::Span<absl::InlinedVector<ClassDistribution, 1> const> type_feedback,
    Function& f);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FUNCTION_H_
