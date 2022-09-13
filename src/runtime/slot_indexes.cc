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

#include "runtime/slot_indexes.h"

#include <cstdint>

#include "strongjit/base.h"

namespace deepmind::s6 {

SlotIndexes::SlotIndexes(const Function& f) {
  // TODO: Should we use a depth-first ordering here?

  // Note that we assign values every other slot, so all values have an even
  // slot. This leaves room for copies to be inserted between value slots.
  int64_t index = 0;
  auto append = [&](const Value* v) {
    slots_[v] = index;
    values_.push_back(v);
    values_.push_back(nullptr);
    index += 2;
  };

  int64_t block_index = 0;
  // Try to ensure we don't re-reserve during allocation; this was observed to
  // be costly in profiling.
  values_.reserve(f.capacity() * 2);
  slots_.reserve(f.capacity() * 2);

  for (const Block& b : f) {
    int64_t start = index;
    append(&b);
    for (const BlockArgument* arg : b.block_arguments()) {
      append(arg);
    }
    for (const Instruction& inst : b) {
      append(&inst);
    }
    block_info_.push_back({&b, block_index++, start, index});
    block_info_map_[&b] = block_info_.back();
  }
}

}  // namespace deepmind::s6
