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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_MAP_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_MAP_H_

#include <iterator>

#include "core_util.h"
#include "runtime/slot_indexes.h"
#include "runtime/util.h"
#include "strongjit/value.h"
#include "utils/interval_map.h"
#include "utils/range.h"

namespace deepmind::s6 {

// Holds a mapping of strongjit Values to Locations during a function's
// execution, and a mapping of program point to strongjit Instruction.
//
// This supports two types of query:
//   1) Which strongjit Instruction is being executed at a given program point?
//   2) Which Values are live at a given Instruction and where are they
//        located?
//
// A "program point" here means a program counter location within generated
// code.
class DeoptimizationMap {
 public:
  struct ValueAndLocation {
    using first_type = const Value*;
    using second_type = Location;

    const Value* value;
    Location location;

    friend bool operator==(const ValueAndLocation& lhs,
                           const ValueAndLocation& rhs) {
      return lhs.value == rhs.value && lhs.location == rhs.location;
    }
  };

  using LiveValuesIterator = std::vector<ValueAndLocation>::const_iterator;

  // Attempts to find the strongjit Instruction being executed at
  // `program_point`. If no Instruction was found, returns nullptr.
  const Instruction* GetInstructionAtAddress(
      ProgramAddress program_point) const;

  // Returns an iterator range over all values live on entry to the Instruction
  // at `program_point`. The order in which the values are returned is
  // undefined.
  IteratorRange<LiveValuesIterator> live_values(
      ProgramAddress program_point) const;
  // Returns an iterator range over all values live on entry to `inst`. The
  // order in which the values are returned is undefined.
  IteratorRange<LiveValuesIterator> live_values(const Instruction* inst) const;

  // Returns the fastlocals live at a program slot.
  std::vector<const Value*> GetLiveFastLocals(SlotIndexes::Slot slot) const;

  std::vector<const Value*> GetLiveFastLocals(const Instruction* inst) const {
    return GetLiveFastLocals(slot_indexes_.SlotForValue(inst));
  }

  // Builder methods
  // Adds an instruction in the half-open range of program addresses [begin,
  // end).
  void AddInstructionAddress(const Instruction* inst, ProgramAddress begin,
                             ProgramAddress end);

  // Adds a live value at the given instruction.
  void AddLiveValue(const Value* value, Location location,
                    const Instruction* inst);

  explicit DeoptimizationMap(SlotIndexes slot_indexes);

  const SlotIndexes& slot_indexes() const { return slot_indexes_; }

 private:
  SlotIndexes slot_indexes_;
  absl::flat_hash_map<const Instruction*, std::vector<ValueAndLocation>>
      live_values_;
  IntervalMap<ProgramAddress, const Instruction*> instruction_addresses_;
  std::vector<IntervalMap<SlotIndexes::Slot, const Value*>> fastlocals_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_MAP_H_
