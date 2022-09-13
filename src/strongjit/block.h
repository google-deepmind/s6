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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BLOCK_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BLOCK_H_

#include <cstdint>

#include "absl/types/span.h"
#include "core_util.h"
#include "strongjit/string_table.h"
#include "strongjit/value.h"
#include "utils/intrusive_list.h"

namespace deepmind::s6 {

class Block;
class Function;
class Instruction;
class TerminatorInst;

////////////////////////////////////////////////////////////////////////////////
// Block

// A BlockArgument is a Value that defines an argument to a Block.
class BlockArgument : public Value {
 public:
  static constexpr Value::Kind kKind = Value::kBlockArgument;
  Kind kind() const override { return kKind; }

  const Block* parent() const { return parent_; }
  Block* parent() { return parent_; }

 private:
  // Only creatable by Function.
  friend class Function;
  explicit BlockArgument(Block* parent) : parent_(parent) {}
  Block* parent_;
};

// A Block has a single entry and a single exit. A Block must end with a single
// TerminatorInst.
//
// A Block maintains a predecessor list. Its successors are implied by its
// Successor instructions. The predecessor list is manually managed; when adding
// a new Successor instruction the user code must call AddPredecessor on the
// successor block.
//
// Note that instructions with implicit control flow such as `deoptimize_if` are
// not accounted for in the successor list.
class Block : public Value, public IntrusiveLink<Block> {
 public:
  Block() = default;
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  static constexpr Value::Kind kKind = Value::kBlock;
  Kind kind() const override { return kKind; }

  using value_type = Instruction;
  using iterator = IntrusiveList<Instruction>::iterator;
  using const_iterator = IntrusiveList<Instruction>::const_iterator;
  using reverse_iterator = IntrusiveList<Instruction>::reverse_iterator;
  using const_reverse_iterator =
      IntrusiveList<Instruction>::const_reverse_iterator;

  Block(Function* parent, StringTable* string_table)
      : parent_(parent), string_table_(string_table) {}

  // Returns this block as an iterator into its parent function.
  IntrusiveList<Block>::iterator GetIterator() {
    return IntrusiveList<Block>::iterator(this);
  }
  IntrusiveList<Block>::const_iterator GetIterator() const {
    return IntrusiveList<Block>::const_iterator(this);
  }

  iterator begin() { return instructions_.begin(); }
  iterator end() { return instructions_.end(); }
  const_iterator begin() const { return instructions_.begin(); }
  const_iterator end() const { return instructions_.end(); }
  reverse_iterator rbegin() { return instructions_.rbegin(); }
  reverse_iterator rend() { return instructions_.rend(); }
  const_reverse_iterator rbegin() const { return instructions_.rbegin(); }
  const_reverse_iterator rend() const { return instructions_.rend(); }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  absl::Span<const Block* const> predecessors() const { return predecessors_; }
  absl::Span<Block* const> predecessors() { return predecessors_; }

  StringTable& GetStringTable() const { return *string_table_; }

  // Replaces `old_pred` with `new_pred` in `predecessors_`.
  void ReplacePredecessor(Block* old_pred, Block* new_pred) {
    // `old_pred` may not be in the current predecessor list, and `new_pred`
    // may already be present.
    RemovePredecessor(old_pred);
    AddPredecessor(new_pred);
  }

  // Adds `pred` as a direct predecessor. `pred` will only be recorded as a
  // predecessor once.
  void AddPredecessor(Block* pred) {
    if (!absl::c_linear_search(predecessors_, pred))
      predecessors_.push_back(pred);
  }

  // Removes `pred` as a direct predecessor. After this function `pred` will no
  // longer be in the predecessor list.
  void RemovePredecessor(Block* pred) { STLEraseAll(predecessors_, pred); }

  IteratorRange<iterator> instructions() {
    return {instructions_.begin(), instructions_.end()};
  }

  IteratorRange<const_iterator> instructions() const {
    return {instructions_.begin(), instructions_.end()};
  }

  // Returns the terminator instruction. All well-formed blocks must end with
  // exacty one terminator.
  const TerminatorInst* GetTerminator() const;
  TerminatorInst* GetTerminator();

  // Inserts `inst` at `insert_pt`. `inst` must not currently have a parent.
  void insert(iterator insert_pt, Instruction* inst);

  // Inserts `inst` at the end of the instruction list. `inst` must not
  // currently have a parent.
  void push_back(Instruction* inst) { return insert(end(), inst); }

  // Unlinks this block from its parent but does not destroy it.
  void RemoveFromParent();

  // Unlinks this block from its parent and destroys it.
  void erase();

  // Returns this block's parent.
  Function* parent() { return parent_; }
  const Function* parent() const { return parent_; }

  // Helper to create an instruction at the end of this block.
  // Use `CreateBlockArgument` to create a block argument.
  template <class T, class... Args>
  T* Create(Args&&... args);

  // Creates a new argument to this block and returns it.
  BlockArgument* CreateBlockArgument();

  // Returns the list of block arguments.
  absl::Span<BlockArgument*> block_arguments() {
    return absl::Span<BlockArgument*>(block_arguments_);
  }
  absl::Span<const BlockArgument* const> block_arguments() const {
    return absl::Span<const BlockArgument* const>(block_arguments_);
  }
  int64_t block_arguments_size() const { return block_arguments_.size(); }
  bool block_arguments_empty() const { return block_arguments_.empty(); }

  void RemoveBlockArgumentAt(int64_t index) {
    S6_CHECK(0 <= index && index < block_arguments_.size());
    block_arguments_.erase(std::next(block_arguments_.begin(), index));
  }

  // Returns true if this block is marked as deoptimized. Deoptimized blocks
  // will not have code generated for them, and exist only to recover to a
  // safe point during deoptimization.
  bool deoptimized() const { return deoptimized_; }

  // Marks this block as being deoptimized.
  void set_deoptimized(bool b) { deoptimized_ = b; }

  // Splits this block at the given iterator. All instructions prior to the
  // iterator stay in this block, all others move to the new block.
  //
  // A JmpInst is inserted to link the blocks together and predecessors are
  // updated.
  // Returns the newly created Block.
  Block* Split(iterator split_point);

  // Splices instructions from another block into this one.
  void splice(iterator insert_pt, Block* other);
  void splice(iterator insert_pt, Block* other, iterator other_begin,
              iterator other_end);

  // Is this block a special handler, and which kind.
  // A normal jump can only go to a kNot or a kFinallyHandler.
  // An except can only go to a kExceptHandler or a kFinallyHandler.
  enum class HandlerKind { kNot, kExcept, kFinally };

  HandlerKind handler_kind() const { return handler_kind_; }
  void SetExceptHandler() { handler_kind_ = HandlerKind::kExcept; }
  void SetFinallyHandler() { handler_kind_ = HandlerKind::kFinally; }
  bool IsHandler() const { return handler_kind_ != HandlerKind::kNot; }
  bool IsExceptHandler() const { return handler_kind_ == HandlerKind::kExcept; }
  bool IsFinallyHandler() const {
    return handler_kind_ == HandlerKind::kFinally;
  }

 private:
  friend class Instruction;  // To access UnlinkInstruction
  friend class Function;     // To update parent_ and block_arguments_.

  // Unlinks this instruction from this block but does not destroy it.
  void UnlinkInstruction(Instruction* inst);

  // We maintain a manual size as IntrusiveList's size() is O(N).
  int32_t size_ = 0;

  // True if this block is marked as deoptimized.
  bool deoptimized_ = false;

  Function* parent_;
  StringTable* string_table_;
  IntrusiveList<Instruction> instructions_;
  absl::InlinedVector<Block*, 2> predecessors_;
  absl::InlinedVector<BlockArgument*, 2> block_arguments_;
  HandlerKind handler_kind_ = HandlerKind::kNot;
};

// Helper class to insert instructions at an insertion point rather than at the
// end of a block.
class BlockInserter {
 public:
  BlockInserter(Block* b, Block::iterator insert_pt)
      : b_(b), insert_pt_(insert_pt) {}

  template <class T, typename... Args>
  T* Create(Args&&... args);

  Block* block() const { return b_; }
  Block::iterator insert_point() const { return insert_pt_; }

 private:
  Block* b_;
  Block::iterator insert_pt_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BLOCK_H_
