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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_TRACE_REGISTER_ALLOCATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_TRACE_REGISTER_ALLOCATOR_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "code_generation/live_interval.h"
#include "code_generation/register_allocator.h"
#include "core_util.h"
#include "event_counters.h"
#include "strongjit/block.h"
#include "strongjit/function.h"
#include "strongjit/instruction.h"
#include "strongjit/ssa.h"
#include "strongjit/value.h"

namespace deepmind::s6 {

// This module implements Trace Register Allocation, modified from the thesis
// "Trace Register Allocation", Josef Eisl, 2018.
//   https://epub.jku.at/obvulihs/download/pdf/3261350
//
// Similar to trace scheduling, the idea is to split the program into disjoint
// `traces`. These are sequences of blocks joined by control-flow edges. That
// is, a single Trace is a possible execution path.
//
// When allocating registers, we allocate for a Trace at a time in priority
// order. We allocate for a single Trace in isolation, so there is no control
// flow to worry about and we can use a simpler allocator.
//
// After allocating all Traces, a global pass inserts copies on the inter-trace
// edges to fix up values allocated differently between Traces.
//
// The flow is:
//  1. Split critical edges. Trace allocation is only correct if there are no
//       critical edges in the CFG.
//  2. Calculate live-in/live-out sets for all blocks.
//  3. Calculate block frequencies, to prioritize trace creation.
//  4. Calculate the Traces.
//  5. Allocate registers per-trace, insert inter-block copies.
//
// Our intra-trace register allocator is a custom top-down allocator that works
// on SSA form. This is made possible by the very simple control flow within a
// Trace.

// A set of Values that uses insertion ordering.
class LiveValueSet {
 public:
  using const_iterator = std::vector<const Value*>::const_iterator;
  using value_type = const Value*;

  // Inserts `value` into the set. This is a constant time operation.
  void insert(const Value* value) {
    if (set_.insert(value).second) {
      ordered_.push_back(value);
    }
  }

  // Erases `value` from the set. If `value` was in the set, this is a linear
  // time operation.
  void erase(const Value* value) {
    if (set_.erase(value)) {
      STLEraseIf(ordered_, [&](const Value* v) { return !set_.contains(v); });
    }
  }

  // Queries if `value` is in the set. This is a constant time operation.
  bool contains(const Value* value) const { return set_.contains(value); }

  // Iterates over the live value set in insertion order.
  std::vector<const Value*>::const_iterator begin() const {
    return ordered_.begin();
  }
  std::vector<const Value*>::const_iterator end() const {
    return ordered_.end();
  }

 private:
  std::vector<const Value*> ordered_;
  absl::flat_hash_set<const Value*> set_;
};

// Holds live-in and live-out sets for all blocks.
class Liveness {
 public:
  Liveness(absl::flat_hash_map<const Block*, LiveValueSet> live_ins,
           absl::flat_hash_map<const Block*, LiveValueSet> live_outs)
      : live_ins_(std::move(live_ins)), live_outs_(std::move(live_outs)) {}

  // The live-ins of a block `b` are Values that dominate `b` and are used
  // by `b` or any successor block.
  LiveValueSet live_ins(const Block* b) const { return live_ins_.at(b); }

  // The live-outs of a block `b` are Values that are required by any successor
  // block but does NOT include arguments to successor blocks.
  LiveValueSet live_outs(const Block* b) const { return live_outs_.at(b); }

 private:
  absl::flat_hash_map<const Block*, LiveValueSet> live_ins_;
  absl::flat_hash_map<const Block*, LiveValueSet> live_outs_;
};

// Performs global liveness analysis. The output is a set of live-in and
// live-out values for each Block `b`.
Liveness AnalyzeLiveness(const Function& f, const DominatorTree& domtree);

using BlockFrequencyMap = absl::flat_hash_map<const Block*, int64_t>;

// Heuristically defines block execution frequencies.
//   Blocks that are postdominated by an ExceptInst are considered improbable.
//   Blocks that are part of a loop are considered more probable based on their
//   loop depth.
//
// Returns a map of opaque integer where a larger integer represents a higher
// frequency, indexed by Block.
BlockFrequencyMap HeuristicallyDetermineBlockFrequencies(
    const Function& f, const DominatorTree& domtree);

// A Trace is a sequence of basic blocks connected by edges. Therefore, the
// blocks in a trace *might* be executed sequentially.
//
// The zeroth block in a Trace is referred to as the Trace Head. Blocks are
// unique within a Trace, and a Trace does not follow loop backbranches.
//
// Every block is in exactly one Trace. Traces are non-empty.
using Trace = std::vector<const Block*>;

// Returns `trace` in an implementation-defined but human readable form.
std::string TraceToString(const deepmind::s6::Trace& trace);

// Allocates every Block in `f` to a Trace. The returned Traces are ordered
// by decreasing priority - blocks in the first Trace have the highest execution
// frequency and should be prioritized during register allocation.
//
// REQUIRES: There must be no critical edges in `f`.
std::vector<Trace> AllocateTraces(const Function& f,
                                  const BlockFrequencyMap& frequencies);

// The state of live variables at the start or end of a block.
struct LiveValueState {
  // The live variables and their locations.
  absl::flat_hash_map<const Value*, Location> locations;
  // The set of values that are live that have been spilled; their value is
  // expected to exist in memory.
  absl::flat_hash_set<const Value*> spilled;
};

// Contains information global across all Traces. Resolves live-in and live-out
// locations at the end of register allocation.
class GlobalResolver {
 public:
  // Constructs a GlobalResolver and pre-allocates all arguments to their
  // ABI-defined locations.
  explicit GlobalResolver(const Block& entry_block);

  // Returns a globally unique Location to store the spilled contents of
  // `value`. Because of SSA form, every value only requires a single spill
  // location.
  Location GetOrCreateSpillLocation(const Value& value);

  // Returns a Location to assign the given live-in value to `block`. This is
  // either hinted by prior traces, or returns the spill slot.
  Location GetLiveInLocation(const Block& block, const Value& value);

  // Computes the set of values that are pre-spilled on entry to `block`. This
  // uses the live-out data from predecessors.
  absl::flat_hash_set<const Value*> ComputeSpilledLiveIns(const Block& block);

  void SetLiveIns(const Block& block, LiveValueState state);
  void SetLiveOuts(const Block& block, int64_t successor_index,
                   LiveValueState state);

  // Resolves conflicting live-in and live-out values on edges.
  void Resolve();

  // Consumes the resulting block copies.
  absl::flat_hash_map<std::pair<const Block*, int64_t>,
                      std::vector<RegisterAllocation::Copy>>
  ConsumeBlockCopies() {
    return std::move(block_copies_);
  }

 private:
  absl::flat_hash_map<const Block*, LiveValueState> live_ins_;
  absl::flat_hash_map<std::pair<const Block*, int64_t>, LiveValueState>
      live_outs_;
  absl::flat_hash_map<const Value*, int64_t> spill_slots_;
  absl::flat_hash_map<std::pair<const Block*, int64_t>,
                      std::vector<RegisterAllocation::Copy>>
      block_copies_;
  int64_t extra_copy_slots_ = 0;
};

// Contains the final use points for Values within a trace (the end of their
// live range), and hints for where Values would like to be placed.
//
// This analysis induces an ordering over the instructions in a Trace.
class BottomUpAnalysis {
 public:
  using iterator = int64_t;

  explicit BottomUpAnalysis(const Trace& trace, const Liveness& liveness);

  // Obtains an iterator for an instruction.
  iterator lookup(const Instruction& value) const {
    return slot_indexes_.at(&value);
  }

  // Invokes `fn` for all Values whose last use points are on entry to, or
  // after, `it` in program order (top-down order). Does not return any Value
  // that is never used at all.
  absl::Status ExpireValuesUnusedAfter(
      iterator it, absl::FunctionRef<absl::Status(const Value*)> fn);

  // Returns the last use point in program order of a Value. The result is an
  // opaque but comparable iterator where larger iterators represent use points
  // farther towards the end of the program.
  iterator GetLastUsePoint(const Value& value) const;

  // Returns true if this Value is never used.
  bool IsNotUsed(const Value& value) const {
    return !last_use_slot_by_value_.contains(&value);
  }

  // Returns a location hint, if one exists, for `value` being allocated at
  // `it`.
  LocationRequirement GetLocationHint(iterator it, const Value& value);

  // Returns true if `value` is used at `it`.
  bool IsUsedAt(iterator it, const Value& value) {
    return used_at_.count({&value, it}) > 0;
  }

 private:
  // Records the slot index for `value`.
  absl::flat_hash_map<const Value*, iterator> slot_indexes_;

  // Records the last use slot for each Value.
  std::multimap<iterator, const Value*> last_use_slot_;

  // Records the slot index at which this Value is last used within this Trace.
  // Is the reverse of last_use_slot_.
  absl::flat_hash_map<const Value*, iterator> last_use_slot_by_value_;

  // For each {operand, slot_index} records a Location requirement.
  std::map<std::pair<const Value*, iterator>, LocationRequirement>
      use_location_;

  // Records every {operand, slot_index}.
  std::set<std::pair<const Value*, iterator>> used_at_;

  // Records the slots that clobber all registers.
  std::set<int64_t> clobber_location_;
};

// Tracks the Location assigned to Values within a program. This can be used
// to enumerate all of the live values at any point in the program.
class ValueLocationMap {
 public:
  // Sets the location of `value` at the point at which it is defined.
  void AddDefinition(const Value& value, const Location& loc) {
    data_[{nullptr, &value}] = loc;
  }

  // Sets the location of `value` at the entry to `program_point`.
  void Add(const Instruction& program_point, const Value& value,
           const Location& loc) {
    data_[{&program_point, &value}] = loc;
  }

  // Obtains the location of `value` at the point at which it is defined.
  const Location& GetDefinition(const Value& value) const {
    return data_.at({nullptr, &value});
  }

  Location GetDefinitionOrUndef(const Value& value) const {
    auto it = data_.find({nullptr, &value});
    if (it == data_.end()) return Location::Undefined();
    return it->second;
  }

  // Obtains the location of `value` on entry to `program_point`.
  const Location& Get(const Instruction& program_point,
                      const Value& value) const {
    return data_.at({&program_point, &value});
  }

  Location GetOrUndef(const Instruction& program_point,
                      const Value& value) const {
    auto it = data_.find({&program_point, &value});
    if (it == data_.end()) return Location::Undefined();
    return it->second;
  }

  // Calls `fn` for all <program_point, value> pairs.
  // If the Instruction* is nullptr, then the Location is the definition
  // location of the value.
  void ForAllLocations(
      absl::FunctionRef<void(const Instruction*, const Value&, const Location&)>
          fn) const {
    for (const auto& [key, location] : data_) {
      fn(key.first, *key.second, location);
    }
  }

  // Merges data from another ValueLocationMap into this.
  void Merge(ValueLocationMap&& other) { data_.merge(std::move(other.data_)); }

 private:
  absl::flat_hash_map<std::pair<const Instruction*, const Value*>, Location>
      data_;
};

// Maintains the state of Values assigned to Locations and the set of free
// registers.
class LocationTracker {
 public:
  struct Info {
    Location location;
    const Value* value;
    // Only used for immediates.
    std::vector<const Value*> immediate_values;
  };

  // Iterators are opaque. They represent Locations.
  using iterator = std::map<Location, Info>::iterator;

  using MoveFn =
      absl::FunctionRef<void(const Location& from, const Location& to)>;

  explicit LocationTracker(const RegisterAllocationOptions& options);

  iterator begin() { return info_by_location_.begin(); }
  iterator end() { return info_by_location_.end(); }

  // Looks up the given Value. May return end() if `value` is not assigned a
  // location.
  iterator find(const Value& value);

  // Looks up the given Location. Always returns a valid iterator.
  iterator find(const Location& location) {
    return info_by_location_.emplace(location, Info{location, nullptr}).first;
  }

  // Looks up the assigned location for `value`. Returns FAILED_PRECONDITION if
  // the value is not assigned.
  absl::StatusOr<Location> Lookup(const Value& value);

  // Moves the value assigned to `it` into `to_location`. `it` is no longer
  // assigned. Returns an iterator for `to_location`.
  // REQUIRES: `it` is assigned.
  // REQUIRES: `to_location` is not assigned.
  // ENSURES: `to_location` is assigned. `it` is not assigned.
  iterator Move(iterator it, const Value& incumbent,
                const Location& to_location, MoveFn insert_move);

  // Assigns `it` to `value`.
  // REQUIRES: `it` is not assigned.
  // REQUIRES: `value` is not assigned.
  // ENSURES: `it` is assigned. `value` is assigned.
  void Assign(iterator it, const Value& value);

  // Frees `it`.
  // REQUIRES: `it` is assigned to `value`.
  // ENSURES: `it` is not assigned.
  void Free(iterator it, const Value& value);

  // Returns true if `it` is assigned to a Value, false otherwise.
  bool IsAssigned(const iterator& it) {
    return it->second.value || !it->second.immediate_values.empty();
  }

  // Returns the Location for an iterator.
  const Location& GetLocation(const iterator& it) { return it->first; }

  // Returns the value assigned to this iterator, or nullptr.
  const Value* GetAssignedValue(const iterator& it) {
    S6_CHECK(!it->first.IsImmediate())
        << "Cannot use GetAssignedValue() on an immediate; there may be "
           "multiple values assigned.";
    return it->second.value;
  }

  // Returns an iterator to a free register. If `hint` is defined, it takes
  // preference if that hinted location is free. May return end() if no
  // registers are available.
  // REQUIRES: If `hint` is defined, it must be a register.
  // ENSURES: The resulting iterator is either end() or is unassigned.
  iterator FindFreeRegister(
      const LocationRequirement& hint = LocationRequirement::Anywhere());

  // Snapshots the current state of the tracker into a ValueLocationMap.
  void Snapshot(ValueLocationMap& map, const Instruction& program_point);

 private:
  std::map<Location, Info> info_by_location_;
  absl::flat_hash_map<const Value*, Info*> info_by_value_;
  std::set<Location> free_;
};

// Allocates registers within a single Trace. This is a top-down greedy
// allocator. Such a naive algorithm produces good results because of the lack
// of control flow within a Trace.
//
// Live ranges and use location hints are inferred by a BottomUpAnalysis.
class TopDownAllocator {
 public:
  TopDownAllocator(BottomUpAnalysis analysis, GlobalResolver& resolver,
                   const Liveness& liveness,
                   const RegisterAllocationOptions& options)
      : tracker_(options),
        analysis_(std::move(analysis)),
        resolver_(resolver),
        liveness_(liveness),
        options_(options) {}

  // Allocates registers for this Trace.
  absl::Status Allocate(const Trace& trace);

  // Consumes the result of register allocation.
  ValueLocationMap ConsumeResult() { return std::move(value_locations_); }

  // Consumes the copy lists.
  absl::flat_hash_map<const Value*, std::vector<RegisterAllocation::Copy>>
  ConsumeCopies() {
    return std::move(inst_copies_);
  }

 private:
  // Informs resolver_ about the expected live-ins, given the current state.
  absl::Status SnapshotLiveIns(const Block& block);
  // Informs resolver_ about the expected live-outs, given the current state.
  absl::Status SnapshotLiveOuts(const Block& block);
  // `requirement` is the location requirement for `value`. If needed, picks a
  // new Location that adheres to the requirement and moves `value`.
  // REQUIRES: `value` must already be assigned.
  absl::StatusOr<Location> FixupOperandLocation(
      LocationTracker::MoveFn move,
      BottomUpAnalysis::iterator analysis_iterator, const Value& value,
      const LocationRequirement& requirement);
  // Allocates a new Location for `value`.
  // REQUIRES: `value` must not be assigned.
  absl::StatusOr<Location> AllocateNewLocation(
      LocationTracker::MoveFn move,
      BottomUpAnalysis::iterator analysis_iterator, const Value& value,
      const LocationRequirement& requirement);
  // Selects a spill candidate. The result could be end().
  LocationTracker::iterator SelectSpillCandidate(
      BottomUpAnalysis::iterator analysis_iterator);
  // Spills the given value. Unassigns `it`.
  void Spill(LocationTracker::iterator it, LocationTracker::MoveFn move);
  // Inserts a copy before `inst`.
  void Copy(const Instruction* before, const Location& from,
            const Location& to) {
    if (to.IsOnStack()) {
      EventCounters::Instance().Add("ra_trace.spills", 1);
    } else if (from.IsOnStack()) {
      EventCounters::Instance().Add("ra_trace.reloads", 1);
    } else {
      EventCounters::Instance().Add("ra_trace.moves", 1);
    }

    inst_copies_[before].emplace_back(from, to);
  }

  ValueLocationMap value_locations_;
  absl::flat_hash_map<const Value*, std::vector<RegisterAllocation::Copy>>
      inst_copies_;
  absl::flat_hash_set<const Value*> spilled_values_;

  LocationTracker tracker_;
  BottomUpAnalysis analysis_;
  GlobalResolver& resolver_;
  const Liveness& liveness_;
  const RegisterAllocationOptions& options_;
};

// Allocates all registers within a function.
absl::StatusOr<std::unique_ptr<RegisterAllocation>> AllocateRegistersWithTrace(
    const Function& f, RegisterAllocationOptions options = {});

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_TRACE_REGISTER_ALLOCATOR_H_
