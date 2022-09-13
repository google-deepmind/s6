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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CURSOR_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CURSOR_H_

#include "strongjit/block.h"
#include "strongjit/instruction.h"

namespace deepmind::s6 {

// A cursor pointing to a particular instruction within a Function. The cursor
// guarantees that mutating the currently pointed to instruction does not
// affect the next or previous instructions, even if the current instruction
// is removed, or additional instructions are inserted immediately after or
// before the current instruction.
//
// Iteration order is instructions within a block, and then blocks in program
// order within a function.
//
// Next and previous positions are computed eagerly, so it is not safe to
// erase any instructions other than the current instruction.
class Cursor {
 public:
  // Creates an invalid cursor.
  Cursor() : curr_valid_(false) {}

  // Creates a cursor pointing at instruction.
  explicit Cursor(Instruction* instruction)
      : Cursor(instruction->GetIterator(), true) {}

  // Creates a cursor from an instruction iterator.
  explicit Cursor(Block::iterator curr) : Cursor(curr, true) {}

  // Returns the pointed to instruction.
  Instruction* GetInstruction() const {
    S6_CHECK(!Finished());
    return &*curr_;
  }

  // Returns the block of the pointed to instruction.
  Block* GetBlock() const {
    S6_CHECK(!Finished());
    return GetInstruction()->parent();
  }

  // Returns the function being iterated over.
  Function* GetFunction() const {
    S6_CHECK(!Finished());
    return GetBlock()->parent();
  }

  // Returns true if the cursor has iterated past the first or last instruction
  // of the function. From this point on the cursor is invalid.
  bool Finished() const { return !curr_valid_; }

  // Moves the cursor back one instruction.
  void StepBackward() { *this = Cursor(prev_, prev_valid_); }

  // Moves the cursor forward one instruction.
  void StepForward() { *this = Cursor(next_, next_valid_); }

 private:
  Cursor(Block::iterator curr, bool curr_valid)
      : curr_(curr), curr_valid_(curr_valid) {
    if (curr_valid_) Update();
  }

  // Update next and previous iterators, given current.
  void Update();

  Block::iterator prev_;
  Block::iterator curr_;
  Block::iterator next_;
  bool prev_valid_ = false;
  bool curr_valid_ = false;
  bool next_valid_ = false;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CURSOR_H_
