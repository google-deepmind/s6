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

#include "runtime/deoptimization_map.h"

#include <cstdint>

#include "runtime/util.h"

namespace deepmind::s6 {

DeoptimizationMap::DeoptimizationMap(SlotIndexes slot_indexes)
    : slot_indexes_(std::move(slot_indexes)) {
  SlotIndexes::Slot end_slot = slot_indexes_.num_slots() + 1;
  for (const Value* v : slot_indexes_.ordered_values()) {
    const BytecodeBeginInst* bytecode_begin = dyn_cast<BytecodeBeginInst>(v);
    if (!bytecode_begin) continue;
    SlotIndexes::Slot slot = slot_indexes_.SlotForValue(bytecode_begin);
    fastlocals_.resize(std::max<int64_t>(fastlocals_.size(),
                                         bytecode_begin->fastlocals().size()));
    int64_t i = 0;
    for (const Value* fastlocal : bytecode_begin->fastlocals()) {
      IntervalMap<SlotIndexes::Slot, const Value*>& map = fastlocals_[i++];
      map.SetAndCoalesce(slot, end_slot, fastlocal);
    }
    while (i != fastlocals_.size()) {
      IntervalMap<SlotIndexes::Slot, const Value*>& map = fastlocals_[i++];
      map.Erase(slot, end_slot);
    }
  }
}

const Instruction* DeoptimizationMap::GetInstructionAtAddress(
    ProgramAddress program_point) const {
  const Instruction* inst;
  if (!instruction_addresses_.Lookup(program_point, &inst)) return nullptr;
  return inst;
}

IteratorRange<DeoptimizationMap::LiveValuesIterator>
DeoptimizationMap::live_values(ProgramAddress program_point) const {
  const Instruction* inst = GetInstructionAtAddress(program_point);
  return live_values(inst);
}

IteratorRange<DeoptimizationMap::LiveValuesIterator>
DeoptimizationMap::live_values(const Instruction* inst) const {
  auto it = live_values_.find(inst);
  if (it == live_values_.end()) {
    static std::vector<ValueAndLocation> empty;
    return {empty.begin(), empty.end()};
  }
  return {it->second.begin(), it->second.end()};
}

void DeoptimizationMap::AddInstructionAddress(const Instruction* inst,
                                              ProgramAddress begin,
                                              ProgramAddress end) {
  instruction_addresses_.Set(begin, end, inst);
}

void DeoptimizationMap::AddLiveValue(const Value* value, Location location,
                                     const Instruction* inst) {
  live_values_[inst].push_back(ValueAndLocation{value, location});
}

std::vector<const Value*> DeoptimizationMap::GetLiveFastLocals(
    SlotIndexes::Slot slot) const {
  std::vector<const Value*> live;
  for (const IntervalMap<SlotIndexes::Slot, const Value*>& map : fastlocals_) {
    const Value* v = nullptr;
    if (!map.Lookup(slot, &v)) {
      // Live fastlocals are contiguous, so as soon as we find a map that does
      // not have a live fastlocal we have finished.
      return live;
    }
    live.push_back(v);
  }
  return live;
}

}  // namespace deepmind::s6
