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

#include "strongjit/block.h"

#include "strongjit/function.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {

void Block::insert(iterator insert_pt, Instruction* inst) {
  instructions_.insert(insert_pt, inst);
  inst->parent_ = this;
  ++size_;
}

void Block::UnlinkInstruction(Instruction* inst) {
  instructions_.erase(inst);
  inst->parent_ = nullptr;
  --size_;
}

void Block::RemoveFromParent() { parent_->UnlinkBlock(this); }

void Block::erase() {
  S6_CHECK(parent_);
  parent_->UnlinkBlock(this);
}

Block* Block::Split(iterator split_point) {
  Block* b = parent_->CreateBlock(std::next(GetIterator()));
  b->splice(b->begin(), this, split_point, end());

  Create<JmpInst>(b);
  b->AddPredecessor(this);

  if (b->GetTerminator()) {
    for (Block* succ : b->GetTerminator()->successors()) {
      succ->ReplacePredecessor(this, b);
    }
  }
  return b;
}

void Block::splice(iterator insert_pt, Block* other) {
  splice(insert_pt, other, other->instructions_.begin(),
         other->instructions_.end());
}

void Block::splice(iterator insert_pt, Block* other, iterator other_begin,
                   iterator other_end) {
  for (auto it = other_begin; it != other_end; ++it) {
    it->parent_ = this;
    ++size_;
    --other->size_;
  }
  instructions_.splice(insert_pt, other->instructions_, other_begin, other_end);
}

const TerminatorInst* Block::GetTerminator() const {
  if (empty() || !isa<TerminatorInst>(*rbegin())) return nullptr;
  return cast<TerminatorInst>(&*rbegin());
}

TerminatorInst* Block::GetTerminator() {
  if (empty() || !isa<TerminatorInst>(*rbegin())) return nullptr;
  return cast<TerminatorInst>(&*rbegin());
}

BlockArgument* Block::CreateBlockArgument() {
  return parent_->CreateBlockArgument(this);
}
}  // namespace deepmind::s6
