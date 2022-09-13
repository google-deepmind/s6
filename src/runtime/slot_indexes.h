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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_SLOT_INDEXES_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_SLOT_INDEXES_H_

#include <cstdint>

#include "strongjit/function.h"
#include "strongjit/value.h"

namespace deepmind::s6 {

// Maps Values to numeric `slots`. These slot indices are used to
// calculate live intervals. Every Instruction and BlockArgument gets a unique
// slot.
//
// Register allocation is built around the concept of a "live-interval". This is
// a range for which a single SSA value is live in a particular location -
// register or stack. A single SSA value may have multiple live-intervals as it
// is moved between locations but always starts with one.
//
// A live-interval is defined over "slot indices", which is a numbering over
// all instructions in the function. The numbering is arbitrary as long as it
// is a total order in which defs precede uses. Different numberings can affect
// the efficacy of register allocation but not its correctness.
//
// For example:
//   Slot  Instruction                  Live interval
//   0     %0 = frame_variable consts   %0 = [0, 2)
//   1     %1 = add %0, %0              %1 = [1, 2)
//   2     jmp &3 [%1, %0]
//   3     &3: [%4,                     %4 = [3, 5)  Note each block argument
//   4          %5]                     %5 = [4, 5)  gets its own slot.
//   5    %6 = add %4, %5               %6 = [5, 6)
//   6    return %6
//
// While primarily built for the register allocator, slots are useful for
// representing live intervals used after register allocation, for example
// mapping program locations to live values.
//
// Note that slots are not contiguous; adjacent values differ in slot index by
// two (all slot indexes are even). This allows the register allocator room to
// represent inserted copies between instructions.
class SlotIndexes {
 public:
  using Slot = int64_t;

  // Constructs SlotIndexes from `f`.
  explicit SlotIndexes(const Function& f);

  // Returns the slot that defines `v`.
  int64_t SlotForValue(const Value* v) const {
    auto it = slots_.find(v);
    return it == slots_.end() ? -1 : it->second;
  }

  // Returns the value at slot `slot`, or nullptr if there is no value at this
  // slot.
  const Value* ValueAtSlot(Slot slot) const { return values_[slot]; }

  // Returns all instructions and block arguments in slot order. This order
  // guarantees only that defs occur before uses.
  absl::Span<const Value* const> ordered_values() const { return values_; }

  int64_t num_values() const { return values_.size(); }

  // Returns the number of slots (values * 2).
  int64_t num_slots() const { return values_.size() * 2; }

  // Information about a block, computed as a sideeffect of computing slot
  // indexes.
  //
  // This is used by the register allocator to more efficiently iterate over
  // slots.
  struct BlockInfo {
    const Block* block;

    // The zero-based index of the block in the slot index traversal order.
    int64_t index;

    // The slot at which the block begins. This is equal to Slot(block).
    Slot start_slot;

    // The slot at which the block ends. This is equal to
    // Slot(block->GetTerminator()) + 1.
    Slot end_slot;
  };

  // Returns information per-block, computed as a sideeffect of computing
  // slot indexes. The list is ordered by `index`.
  absl::Span<BlockInfo const> block_info() const { return block_info_; }
  const BlockInfo& GetBlockInfo(const Block* b) const {
    return block_info_map_.at(b);
  }

 private:
  absl::flat_hash_map<const Value*, Slot> slots_;
  std::vector<const Value*> values_;
  std::vector<BlockInfo> block_info_;
  absl::flat_hash_map<const Block*, BlockInfo> block_info_map_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_SLOT_INDEXES_H_
