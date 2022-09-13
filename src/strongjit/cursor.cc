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

#include "strongjit/cursor.h"

#include "strongjit/function.h"

namespace deepmind::s6 {

void Cursor::Update() {
  S6_CHECK(!Finished());
  Block* block = GetBlock();
  Function* function = GetFunction();

  next_ = curr_;
  next_valid_ = true;
  ++next_;
  if (next_ == block->end()) {
    auto next_block = ++block->GetIterator();
    if (next_block != function->end()) {
      next_ = next_block->begin();
    } else {
      next_valid_ = false;
    }
  }

  prev_ = curr_;
  prev_valid_ = true;
  if (prev_ != block->begin()) {
    --prev_;
  } else {
    auto block_iter = block->GetIterator();
    if (block_iter != function->begin()) {
      prev_ = --std::end(*--block_iter);
    } else {
      prev_valid_ = false;
    }
  }
}

}  // namespace deepmind::s6
