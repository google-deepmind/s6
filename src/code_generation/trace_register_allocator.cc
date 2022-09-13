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

#include "code_generation/trace_register_allocator.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <queue>
#include <string>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "code_generation/asmjit_util.h"
#include "code_generation/live_interval.h"
#include "code_generation/register_allocator.h"
#include "core_util.h"
#include "cppitertools/reversed.hpp"
#include "runtime/slot_indexes.h"
#include "strongjit/block.h"
#include "strongjit/builder.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "utils/logging.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

std::string TraceToString(const deepmind::s6::Trace& trace) {
  return absl::StrJoin(trace, " -> ",
                       [](std::string* s, const deepmind::s6::Block* b) {
                         absl::StrAppend(s, FormatOrDie(*b));
                       });
}

// Returns FAILED_PRECONDITION if there are critical edges in `f`.
absl::Status EnsureNoCriticalEdges(const Function& f) {
  for (const Block& pred : f) {
    // If this block has only one successor, the successor edges cannot be
    // critical.
    const TerminatorInst* ti = pred.GetTerminator();
    if (ti->successor_size() <= 1) continue;

    for (int64_t i = 0; i < ti->successor_size(); ++i) {
      const Block* succ = ti->successors()[i];
      S6_CHECK(succ);
      if (succ->predecessors().size() <= 1) continue;

      return absl::FailedPreconditionError(
          absl::StrCat("Edge ", FormatOrDie(pred), " -> ", FormatOrDie(*succ),
                       " is a critical edge."));
    }

    // Note that splitting critical edges never introduces more critical edges,
    // so we do not need to revisit any edges.
  }
  return absl::OkStatus();
}

Liveness AnalyzeLiveness(const Function& f, const DominatorTree& domtree) {
  // This one-pass algorithm for liveness analysis on SSA-form programs
  // is taken from "Linear scan register allocation on SSA form" by
  // Wimmer & Franz. https://dl.acm.org/doi/10.1145/1772954.1772979
  //
  // Because our top-down allocator does not use live intervals, we simply
  // use this algorithm to calculate the live-in and live-out sets.
  absl::flat_hash_map<const Block*, LiveValueSet> live_in;

  // Iterate across all blocks in reverse RPO (i.e. postorder).
  absl::Span<const DominatorTree::Node> nodes = domtree.Postorder();
  absl::flat_hash_map<const Block*, int64_t> block_postorder_index;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    const Block* block = it->block();
    block_postorder_index[block] = block_postorder_index.size();
    LiveValueSet live;

    // Live = union of successor.liveIn for each successor of block
    for (const Block* successor : block->GetTerminator()->successors()) {
      if (!successor) continue;
      const LiveValueSet& succ_live_in = live_in[successor];
      for (const Value* v : succ_live_in) {
        live.insert(v);
      }
    }

    // for each operation op of b in reverse order do:
    for (const Instruction& inst : iter::reversed(*block)) {
      // for each output operand opd of op do:
      //   live.remove(opd)
      live.erase(&inst);

      // for each input operand opd of op do:
      //   live.add(opd)
      for (const Value* operand : inst.operands()) {
        if (!isa<Instruction>(operand) && !isa<BlockArgument>(operand))
          continue;
        live.insert(operand);
      }
    }

    // For each phi function phi of b do
    //   live.remove(phi.output)
    for (const BlockArgument* arg : block->block_arguments()) {
      live.erase(arg);
    }

    // If `block` is a loop header, the backedge may not currently be valid as
    // it was processed before the header. Fix up by performing the union of
    // all values live-in to the header for all blocks within the loop.
    if (const DominatorTree::Node* backbranch =
            domtree.GetLongestBackbranch(*it)) {
      for (auto it2 = backbranch; it2 != it; ++it2) {
        if (!domtree.Dominates(&*it, &*it2)) continue;
        for (const Value* v : live) {
          live_in[it2->block()].insert(v);
        }
      }
    }

    live_in[block] = std::move(live);
  }

  // Live-outs are the union of live-ins for all successors.
  absl::flat_hash_map<const Block*, LiveValueSet> live_outs;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    const Block* block = it->block();
    LiveValueSet& live_out = live_outs[block];
    for (const Block* succ : block->GetTerminator()->successors()) {
      S6_CHECK(succ);
      for (const Value* v : live_in.at(succ)) {
        live_out.insert(v);
      }
    }
  }

  return Liveness(std::move(live_in), std::move(live_outs));
}

bool AllSuccessorsHaveZeroFrequency(
    const TerminatorInst* terminator,
    const absl::flat_hash_map<const Block*, int64_t>& frequencies) {
  for (const Block* succ : terminator->successors()) {
    S6_CHECK(succ);
    if (frequencies.at(succ) != 0) return false;
  }
  return true;
}

BlockFrequencyMap HeuristicallyDetermineBlockFrequencies(
    const Function& f, const DominatorTree& domtree) {
  // All blocks initially have a frequency of one.
  const int64_t kInitialFrequency = 1;
  const int64_t kLoopMultiplier = 2;

  absl::Span<DominatorTree::Node const> nodes = domtree.Postorder();
  absl::flat_hash_map<const Block*, int64_t> frequencies;
  for (const DominatorTree::Node& node : nodes) {
    frequencies[node.block()] = kInitialFrequency;
  }

  // Start by giving all blocks that are postdominated by ExceptInsts zero
  // frequency. We don't have a postdominator tree, so crawl from ExceptInsts to
  // all predecessors while the predecessors successors have zero frequency.
  for (const DominatorTree::Node& node : nodes) {
    const Block* block = node.block();
    if (isa<ExceptInst>(block->GetTerminator()) || block->deoptimized()) {
      frequencies[block] = 0;
      continue;
    }

    // If this block has no successors, don't set its frequency to zero... it's
    // the ReturnInst!
    if (block->GetTerminator()->successor_size() == 0) continue;

    // If all successors have zero frequency, this block should have zero
    // frequency. Postorder traversal guarantees that all successors have been
    // traversed already.
    if (AllSuccessorsHaveZeroFrequency(block->GetTerminator(), frequencies)) {
      frequencies[block] = 0;
    }
  }

  // Identify loops. A loop has a header, which is the target of the backbranch.
  // Loop headers can be identified by them dominating one of their
  // predecessors.
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    if (const DominatorTree::Node* backbranch =
            domtree.GetLongestBackbranch(*it)) {
      for (auto it2 = backbranch; it2 != it; ++it2) {
        frequencies[it2->block()] *= kLoopMultiplier;
      }
      frequencies[it->block()] *= kLoopMultiplier;
    }
  }

  return frequencies;
}

std::vector<Trace> AllocateTraces(const Function& f,
                                  const BlockFrequencyMap& frequencies) {
  // This uses the UnidirectionalTraceBuilder algorithm from Eisl et al.
  std::vector<Trace> traces;

  // Blocks not yet in a trace.
  absl::flat_hash_set<const Block*> blocks;
  for (const auto& [b, f] : frequencies) {
    blocks.emplace(b);
  }

  auto is_in_trace = [&blocks](const Block* b) { return !blocks.contains(b); };

  // Blocks eligible to begin a trace as a max-heap.
  std::priority_queue<std::pair<int64_t, const Block*>> eligible_heads;
  eligible_heads.push({1, &f.entry()});

  // Blocks that may have become eligible to begin a trace, exposed while
  // constructing another trace.
  std::vector<const Block*> maybe_eligible_heads;

  while (!blocks.empty()) {
    S6_CHECK(!eligible_heads.empty());
    const Block* b = eligible_heads.top().second;
    eligible_heads.pop();
    if (!blocks.contains(b)) {
      continue;
    }

    Trace t(1, b);
    blocks.erase(b);

    // Iteratively add the most eligible successor.
    while (true) {
      const Block* chosen_succ = nullptr;
      int64_t chosen_succ_frequency = -1;
      for (const Block* succ : b->GetTerminator()->successors()) {
        S6_CHECK(succ);
        int64_t frequency = frequencies.at(succ);
        if (frequency >= chosen_succ_frequency && blocks.count(succ) == 1 &&
            succ->deoptimized() == b->deoptimized()) {
          chosen_succ = succ;
          chosen_succ_frequency = frequency;
        }
      }

      // Add all non-chosen successors to the potentially-eligible-heads list.
      for (const Block* succ : b->GetTerminator()->successors()) {
        S6_CHECK(succ);
        if (succ != chosen_succ) {
          maybe_eligible_heads.push_back(succ);
        }
      }
      b = chosen_succ;
      if (!b) {
        break;
      }
      blocks.erase(b);
      t.push_back(b);
    }

    traces.push_back(std::move(t));

    // Do we have any more candidates to add to the eligible heads list?
    for (const Block* b : maybe_eligible_heads) {
      if (absl::c_all_of(b->predecessors(), is_in_trace)) {
        eligible_heads.push({frequencies.at(b), b});
      }
    }
    maybe_eligible_heads.clear();
  }

  return traces;
}

// The container that the rest of the system will use to query our register
// allocation decisions.
class TraceRegisterAllocation : public RegisterAllocation {
 public:
  // Returns the output location of `v`.
  Location DestinationLocation(const Value& v) const override {
    return value_locations_.GetDefinitionOrUndef(v);
  }

  // Returns the input location of operand `value` to instruction `inst`.
  Location OperandLocation(const Instruction& inst,
                           const Value& value) const override {
    return value_locations_.GetOrUndef(inst, value);
  }

  std::string ToString(const Function& f) const override {
    return FormatOrDie(f, RegisterAllocationAnnotator(*this));
  }

  // Returns the number of frame slots used.
  int64_t GetNumFrameSlots() const override { return num_frame_slots_; }

  // Returns the number of used call stack slots.
  int64_t GetNumCallStackSlots() const override {
    return num_call_stack_slots_;
  }

  // Populates the given deoptimization mapping with all live ranges.
  // Note that this implementation only populates values explicitly used by
  // instructions; it does not populate all live-ranges in the deoptimization
  // map.
  //
  // It is therefore not possible to deoptimize at arbitrary locations, only
  // SafepointInsts or DeoptimizeIfInsts (that contain their transitive uses
  // in their operand list).
  void PopulateDeoptimizationMap(DeoptimizationMap& deopt_map) const override {
    value_locations_.ForAllLocations([&](const Instruction* program_point,
                                         const Value& value,
                                         const Location& loc) {
      if (!program_point) return;
      deopt_map.AddLiveValue(&value, loc, program_point);
    });
  }

  // Returns the set of all registers used.
  std::vector<asmjit::x86::Gp> ComputeUsedRegisters() const override {
    return used_registers_;
  }

  // Used when constructing a TraceRegisterAllocation from multiple
  // TopDownAllocators.
  void Merge(
      ValueLocationMap value_locations,
      absl::flat_hash_map<const Value*, std::vector<RegisterAllocation::Copy>>
          inst_copies) {
    value_locations_.Merge(std::move(value_locations));
    inst_copies_.merge(std::move(inst_copies));
  }
  void MergeBlockCopies(
      absl::flat_hash_map<std::pair<const Block*, int64_t>, std::vector<Copy>>
          block_copies) {
    block_copies_.merge(std::move(block_copies));
  }

  // Calculates the number of frame and call stack slots, and the set of used
  // registers.
  void Finalize() {
    std::set<asmjit::x86::Gp> used;
    auto record = [&](const Location& location) {
      if (location.IsInRegister()) {
        used.insert(location.Register().as<asmjit::x86::Gp>());
      } else if (location.IsCallStackSlot()) {
        num_call_stack_slots_ = std::max<int64_t>(num_call_stack_slots_,
                                                  location.CallStackSlot() + 1);
      } else if (location.IsFrameSlot()) {
        num_frame_slots_ =
            std::max<int64_t>(num_frame_slots_, location.FrameSlot() + 1);
      }
    };
    value_locations_.ForAllLocations(
        [&](const Instruction* program_point, const Value& value,
            const Location& location) { record(location); });
    for (const auto& [inst, copies] : inst_copies_) {
      for (const Copy& copy : copies) {
        record(copy.first);
        record(copy.second);
      }
    }
    for (const auto& [key, copies] : block_copies_) {
      for (const Copy& copy : copies) {
        record(copy.first);
        record(copy.second);
      }
    }

    std::copy(used.begin(), used.end(), std::back_inserter(used_registers_));
    absl::c_sort(used_registers_);
  }

 private:
  ValueLocationMap value_locations_;
  std::vector<asmjit::x86::Gp> used_registers_;
  int64_t num_call_stack_slots_ = 0;
  int64_t num_frame_slots_ = 0;
};

// Returns true if this kind of Value requires a Location.
bool IsAllocatableValue(const Value* v) {
  return v && (isa<Instruction>(v) || isa<BlockArgument>(v));
}

GlobalResolver::GlobalResolver(const Block& entry_block) {
  LiveValueState state;
  // In the fast calling convention, the first argument is always the
  // PyFunctionObject, so argument numbering starts at 1.
  int64_t i = 1;
  for (const BlockArgument* arg : entry_block.block_arguments()) {
    if (!GetAbiLocation(i).IsInRegister()) {
      state.locations[arg] = GetOrCreateSpillLocation(*arg);
    } else {
      state.locations[arg] = GetAbiLocation(i);
    }
    ++i;
  }
  live_ins_[&entry_block] = std::move(state);
}

Location GlobalResolver::GetOrCreateSpillLocation(const Value& value) {
  return Location::FrameSlot(
      spill_slots_.emplace(&value, spill_slots_.size()).first->second);
}

void GlobalResolver::SetLiveIns(const Block& block, LiveValueState state) {
  live_ins_[&block] = std::move(state);
}

void GlobalResolver::SetLiveOuts(const Block& block, int64_t successor_index,
                                 LiveValueState state) {
  live_outs_[{&block, successor_index}] = std::move(state);
  auto& live_outs = live_outs_[{&block, successor_index}];

  const Block* succ = block.GetTerminator()->successors()[successor_index];
  S6_CHECK(succ);
  // The number of arguments in an ExceptInst does not match its successor.
  if (isa<ExceptInst>(block.GetTerminator())) return;

  block.GetTerminator()->ForEachArgumentOnEdge(
      succ, [&](const BlockArgument* arg, const Value* param) {
        Location l = live_outs.locations[param];
        live_outs.locations.emplace(arg, l);
        if (live_outs.spilled.contains(param)) live_outs.spilled.insert(arg);
      });
}

Location GlobalResolver::GetLiveInLocation(const Block& block,
                                           const Value& value) {
  // The live-ins may already be available (only really happens for the entry
  // block).
  if (auto it = live_ins_.find(&block); it != live_ins_.end()) {
    return it->second.locations.at(&value);
  }

  // Try and find a predecessor with live-outs.
  for (const Block* pred : block.predecessors()) {
    const TerminatorInst* ti = pred->GetTerminator();
    for (int64_t i = 0; i < ti->successor_size(); ++i) {
      if (ti->successors()[i] != &block) continue;

      auto it = live_outs_.find({pred, i});
      if (it == live_outs_.end()) continue;

      auto it2 = it->second.locations.find(&value);
      if (it2 != it->second.locations.end()) return it2->second;
      return GetOrCreateSpillLocation(value);
    }
  }
  return GetOrCreateSpillLocation(value);
}

void GlobalResolver::Resolve() {
  auto order_copies = [this](std::vector<RegisterAllocation::Copy>& copies) {
    // Give the copy list a consistent starting order.
    absl::c_sort(copies);
    int64_t extra_copy_slots = 0;
    for (int64_t i = 0; i < copies.size(); ++i) {
      // If the destination is not read from in following copies, this copy is
      // safe.
      Location to = copies[i].second;
      if (std::find_if(copies.begin() + i + 1, copies.end(),
                       [&to](const auto x) { return x.first == to; }) ==
          copies.end()) {
        continue;
      }

      // This copy will clobber others; change to a copy to memory, and add a
      // new copy back at the end.
      Location spill_loc =
          Location::FrameSlot(spill_slots_.size() + extra_copy_slots++);
      copies[i].second = spill_loc;
      copies.emplace_back(spill_loc, to);
    }
    extra_copy_slots_ = std::max(extra_copy_slots_, extra_copy_slots);
  };

  int64_t n = 0;
  for (const auto& [pred_and_successor_index, live_outs_ref] : live_outs_) {
    const Block* pred = pred_and_successor_index.first;
    int64_t successor_index = pred_and_successor_index.second;
    const TerminatorInst* ti = pred->GetTerminator();
    const Block* succ = ti->successors()[successor_index];
    S6_CHECK(succ);

    const LiveValueState& live_ins = live_ins_[succ];
    // Introduce a new, named reference to live_outs_ so that it can be
    // referenced in the lambda below.
    const LiveValueState& live_outs = live_outs_ref;
    auto& copies = block_copies_[{pred, successor_index}];
    std::set<std::pair<Location, Location>> inserted;

    for (const auto& [value, in_location] : live_ins.locations) {
      if (!live_outs.locations.contains(value)) continue;
      // If the live-out is actually a block argument to the successor, this
      // will be handled below.
      if (auto* ba = dyn_cast<BlockArgument>(value); ba && ba->parent() == succ)
        continue;
      const Location& out_location = live_outs.locations.at(value);
      if (in_location == out_location) continue;
      if (inserted.emplace(out_location, in_location).second) {
        copies.emplace_back(out_location, in_location);
        ++n;
      }
    }
    for (const auto* value : live_ins.spilled) {
      if (live_outs.spilled.contains(value) || isa<BlockArgument>(value))
        continue;
      const Location& from = live_outs.locations.at(value);
      const Location& to = Location::FrameSlot(spill_slots_.at(value));

      if (inserted.emplace(from, to).second) {
        copies.emplace_back(from, to);
        ++n;
      }
    }
    pred->GetTerminator()->ForEachArgumentOnEdge(
        succ, [&](const BlockArgument* arg, const Value* param) {
          if (!param) return;
          const Location& arg_location = GetLiveInLocation(*succ, *arg);
          const Location& param_location = live_outs.locations.at(param);
          if (arg_location != param_location) {
            if (inserted.emplace(param_location, arg_location).second) {
              copies.emplace_back(param_location, arg_location);
              ++n;
            }
          }
        });

    order_copies(copies);
  }

  EventCounters::Instance().Add("ra_trace.resolver_copies", n);
}

absl::flat_hash_set<const Value*> GlobalResolver::ComputeSpilledLiveIns(
    const Block& block) {
  absl::flat_hash_set<const Value*> v;
  for (const auto& [key, state] : live_outs_) {
    if (key.first->GetTerminator()->successors()[key.second] != &block) {
      continue;
    }
    v.merge(absl::flat_hash_set<const Value*>(state.spilled));
  }
  return v;
}

BottomUpAnalysis::BottomUpAnalysis(const Trace& trace,
                                   const Liveness& liveness) {
  // All uses are in terms of this `slot_index` which starts at zero at the
  // end of the Trace and decrements as we go backwards such that if `a`
  // occurs-before `b` in program order, `slot(a) < slot(b)`. This gives us
  // negative slots but makes the slot queries easier to reason about.
  iterator slot_index = 0;

  for (const Block* block : iter::reversed(trace)) {
    // Reserve a slot index for the "end of block"; this is beyond the successor
    // instruction.
    --slot_index;

    // For all live-outs, if we haven't already recorded a use point for this
    // value then it must be live-out into a different trace.
    for (const Value* v : liveness.live_outs(block)) {
      last_use_slot_by_value_.emplace(v, slot_index + 1);
    }

    // Similarly for arguments to successors.
    for (const Block* succ : block->GetTerminator()->successors()) {
      for (const Value* v : block->GetTerminator()->successor_arguments(succ)) {
        last_use_slot_by_value_.emplace(v, slot_index + 1);
      }
    }

    for (const Instruction& inst : iter::reversed(*block)) {
      for (int64_t operand_index = 0; operand_index < inst.operands().size();
           ++operand_index) {
        const Value* operand = inst.operands()[operand_index];
        if (!IsAllocatableValue(operand)) continue;

        last_use_slot_by_value_.emplace(operand, slot_index);
        LocationRequirement requirement =
            GetRegisterRequirement(&inst, operand_index);
        if (requirement.IsLocation()) {
          S6_VLOG(4) << "Inserting use at slot " << slot_index << " for "
                     << FormatOrDie(*operand) << " (" << requirement.ToString()
                     << ")";
          use_location_[{operand, slot_index}] = requirement;
        }
        used_at_.insert({operand, slot_index});
      }

      if (InstructionTraits::ClobbersAllRegisters(inst)) {
        S6_VLOG(4) << "Inserting clobber at slot " << slot_index << " for "
                   << FormatOrDie(inst);
        clobber_location_.insert(slot_index);
      }
      slot_indexes_[&inst] = slot_index;
      --slot_index;
    }
  }

  // Populate last_use_slot_ by reversing last_use_slot_by_value_.
  for (const auto& [v, it] : last_use_slot_by_value_) {
    last_use_slot_.emplace(it, v);
  }
}

absl::Status BottomUpAnalysis::ExpireValuesUnusedAfter(
    iterator it, absl::FunctionRef<absl::Status(const Value*)> fn) {
  for (auto del_it = last_use_slot_.begin();
       del_it != last_use_slot_.end() && del_it->first <= it;
       del_it = last_use_slot_.erase(del_it)) {
    S6_RETURN_IF_ERROR(fn(del_it->second));
  }
  return absl::OkStatus();
}

BottomUpAnalysis::iterator BottomUpAnalysis::GetLastUsePoint(
    const Value& value) const {
  auto it = last_use_slot_by_value_.find(&value);
  if (it == last_use_slot_by_value_.end()) return -1;
  return it->second;
}

LocationRequirement BottomUpAnalysis::GetLocationHint(iterator it,
                                                      const Value& value) {
  auto use_it = use_location_.lower_bound({&value, it});
  if (use_it == use_location_.end()) return LocationRequirement::Anywhere();

  // lower_bound() could be returning a false positive.
  if (use_it->first.first != &value) return LocationRequirement::Anywhere();

  // Does the value get clobbered before it gets used? Use upper_bound as we
  // assume `it` doesn't clobber itself.
  auto clobber_it = clobber_location_.upper_bound(it);
  if (clobber_it != clobber_location_.end() &&
      *clobber_it < use_it->first.second) {
    return LocationRequirement::Anywhere();
  }
  return use_it->second;
}

LocationTracker::LocationTracker(const RegisterAllocationOptions& options) {
  for (const asmjit::x86::Reg& reg : options.allocatable_registers) {
    free_.insert(Location::Register(reg));
  }
}

LocationTracker::iterator LocationTracker::find(const Value& value) {
  auto it = info_by_value_.find(&value);
  if (it == info_by_value_.end()) return end();
  return find(it->second->location);
}

absl::StatusOr<Location> LocationTracker::Lookup(const Value& value) {
  auto it = find(value);
  if (it != end()) return it->first;
  return absl::FailedPreconditionError(
      absl::StrCat("Value not assigned a Location: ", FormatOrDie(value)));
}

LocationTracker::iterator LocationTracker::Move(iterator it,
                                                const Value& incumbent,
                                                const Location& to_location,
                                                MoveFn insert_move) {
  auto to_it = find(to_location);
  S6_CHECK(IsAssigned(it));
  S6_CHECK(!IsAssigned(to_it));

  insert_move(it->first, to_location);

  Free(it, incumbent);
  Assign(to_it, incumbent);

  S6_CHECK(it->first.IsImmediate() || !IsAssigned(it));
  S6_CHECK(IsAssigned(to_it));
  return to_it;
}

void LocationTracker::Assign(iterator it, const Value& value) {
  S6_CHECK(it->first.IsImmediate() || !IsAssigned(it));
  if (it->first.IsImmediate()) {
    it->second.immediate_values.push_back(&value);
  } else {
    it->second.value = &value;
  }
  info_by_value_[&value] = &it->second;
  if (it->first.IsInRegister()) {
    free_.erase(it->first);
  }
  S6_CHECK(IsAssigned(it));
}

void LocationTracker::Free(iterator it, const Value& value) {
  S6_CHECK(IsAssigned(it));
  info_by_value_.erase(it->second.value);
  it->second.value = nullptr;
  auto& imm_values = it->second.immediate_values;
  if (!imm_values.empty()) {
    STLEraseAll(imm_values, &value);
  }
  if (it->first.IsInRegister()) {
    free_.insert(it->first);
  }
  S6_CHECK(it->first.IsImmediate() || !IsAssigned(it));
}

LocationTracker::iterator LocationTracker::FindFreeRegister(
    const LocationRequirement& hint) {
  if (hint.IsLocation()) {
    if (auto it = free_.find(hint.location()); it != free_.end()) {
      return find(*it);
    }
  }
  auto it = free_.begin();
  if (it == free_.end()) return end();
  return find(*it);
}

void LocationTracker::Snapshot(ValueLocationMap& map,
                               const Instruction& program_point) {
  for (auto& [location, info] : *this) {
    if (location.IsImmediate()) {
      for (const Value* v : info.immediate_values) {
        map.Add(program_point, *v, location);
      }
    } else if (info.value) {
      map.Add(program_point, *info.value, location);
    }
  }
}

void TopDownAllocator::Spill(LocationTracker::iterator it,
                             LocationTracker::MoveFn move) {
  const Value* candidate = tracker_.GetAssignedValue(it);
  S6_CHECK(candidate);
  if (spilled_values_.insert(candidate).second) {
    tracker_.Move(it, *candidate,
                  resolver_.GetOrCreateSpillLocation(*candidate), move);
  } else {
    EventCounters::Instance().Add("ra_trace.elided_spills", 1);
    tracker_.Free(it, *candidate);
    tracker_.Assign(
        tracker_.find(resolver_.GetOrCreateSpillLocation(*candidate)),
        *candidate);
  }
}

absl::Status TopDownAllocator::SnapshotLiveIns(const Block& block) {
  LiveValueState live;
  for (const Value* v : liveness_.live_ins(&block)) {
    if (spilled_values_.contains(v)) live.spilled.insert(v);
    S6_ASSIGN_OR_RETURN(live.locations[v], tracker_.Lookup(*v));
  }
  for (const Value* v : block.block_arguments()) {
    if (spilled_values_.contains(v)) live.spilled.insert(v);
    S6_ASSIGN_OR_RETURN(live.locations[v], tracker_.Lookup(*v));
  }
  resolver_.SetLiveIns(block, std::move(live));
  return absl::OkStatus();
}

absl::Status TopDownAllocator::SnapshotLiveOuts(const Block& block) {
  LiveValueState live;
  for (const Value* v : liveness_.live_outs(&block)) {
    if (spilled_values_.contains(v)) live.spilled.insert(v);
    S6_ASSIGN_OR_RETURN(live.locations[v], tracker_.Lookup(*v));
  }
  const TerminatorInst* ti = block.GetTerminator();
  for (int64_t i = 0; i < ti->successor_size(); ++i) {
    if (ti->successors()[i]->deoptimized()) continue;
    LiveValueState clone = live;
    for (const Value* v : ti->successor_arguments(i)) {
      if (spilled_values_.contains(v)) clone.spilled.insert(v);
      S6_ASSIGN_OR_RETURN(clone.locations[v], tracker_.Lookup(*v));
    }
    resolver_.SetLiveOuts(block, i, std::move(clone));
  }
  return absl::OkStatus();
}

LocationTracker::iterator TopDownAllocator::SelectSpillCandidate(
    BottomUpAnalysis::iterator analysis_iterator) {
  auto can_be_spilled = [&](const LocationTracker::Info& info) {
    bool is_used_here =
        info.value ? analysis_.IsUsedAt(analysis_iterator, *info.value) : false;
    return info.location.IsInRegister() && !is_used_here;
  };

  auto it = absl::c_max_element(tracker_, [&](const auto& a, const auto& b) {
    // If a value is used at `analysis_iterator`, it cannot be evicted to make
    // room for another value at the same program point!
    bool a_can_be_spilled = can_be_spilled(a.second);
    bool b_can_be_spilled = can_be_spilled(b.second);
    int64_t a_live_until =
        a.second.value ? analysis_.GetLastUsePoint(*a.second.value) : 0;
    int64_t b_live_until =
        b.second.value ? analysis_.GetLastUsePoint(*b.second.value) : 0;
    return std::make_tuple(a_can_be_spilled, a_live_until) <
           std::make_tuple(b_can_be_spilled, b_live_until);
  });
  if (it == tracker_.end()) return it;
  if (!tracker_.GetAssignedValue(it) ||
      !tracker_.GetLocation(it).IsInRegister() || !can_be_spilled(it->second)) {
    return tracker_.end();
  }
  return it;
}

absl::StatusOr<Location> TopDownAllocator::FixupOperandLocation(
    LocationTracker::MoveFn move, BottomUpAnalysis::iterator analysis_iterator,
    const Value& value, const LocationRequirement& requirement) {
  auto it = tracker_.find(value);
  S6_RET_CHECK(it != tracker_.end()) << "Operand must be allocated already";
  if (requirement.AdheresToRequirement(tracker_.GetLocation(it))) {
    return tracker_.GetLocation(it);
  }
  S6_CHECK(!requirement.IsAnywhere());

  if (requirement.IsInFrameSlot()) {
    Spill(it, move);
    return tracker_.find(value)->first;
  }

  if (requirement.IsInRegister()) {
    // Anything will do, as long as it's a register.
    LocationRequirement hint =
        analysis_.GetLocationHint(analysis_iterator, value);
    auto new_it = tracker_.FindFreeRegister(hint);
    if (new_it != tracker_.end()) {
      S6_VLOG(2) << "Allocated location: "
                 << tracker_.GetLocation(new_it).ToString();
      tracker_.Move(it, value, tracker_.GetLocation(new_it), move);
      return tracker_.GetLocation(new_it);
    }

    new_it = SelectSpillCandidate(analysis_iterator);
    if (new_it == tracker_.end()) {
      return absl::ResourceExhaustedError(
          "No spill candidates and no registers left");
    }
    const Value* candidate = tracker_.GetAssignedValue(new_it);
    S6_CHECK(candidate);
    Spill(new_it, move);
    tracker_.Move(it, value, tracker_.GetLocation(new_it), move);
    S6_VLOG(2) << "Spilled " << FormatOrDie(*candidate) << " from "
               << tracker_.GetLocation(it).ToString();
    return tracker_.GetLocation(it);
  }

  // If we get here the value requires a specific Location.
  S6_CHECK(requirement.IsLocation());
  Location requirement_loc = requirement.location();
  S6_CHECK(requirement_loc.IsCallStackSlot() || requirement_loc.IsInRegister());
  auto required_it = tracker_.find(requirement_loc);
  if (!tracker_.IsAssigned(required_it)) {
    tracker_.Move(it, value, requirement_loc, move);
    S6_VLOG(2) << "Allocated required location from free set (move inserted): "
               << requirement.ToString();
    return requirement_loc;
  }

  // Eject the incumbent to its spill slot...
  Spill(required_it, move);
  // ... And move `value` into its spot.
  tracker_.Move(it, value, requirement_loc, move);
  S6_VLOG(2) << "Evicted incumbent from required location "
             << requirement_loc.ToString();
  return requirement_loc;
}

absl::StatusOr<Location> TopDownAllocator::AllocateNewLocation(
    LocationTracker::MoveFn move, BottomUpAnalysis::iterator analysis_iterator,
    const Value& value, const LocationRequirement& requirement) {
  if (requirement.IsAnywhere() || requirement.IsInFrameSlot()) {
    Location loc = resolver_.GetOrCreateSpillLocation(value);
    tracker_.Assign(tracker_.find(loc), value);
    spilled_values_.insert(&value);
    return loc;
  }

  if (const ConstantInst* c = dyn_cast<ConstantInst>(&value);
      c && c->value() < options_.immediate_threshold) {
    Location loc = Location::Immediate(c->value());
    tracker_.Assign(tracker_.find(loc), value);
    return loc;
  }
  if (isa<RematerializeInst>(value)) {
    Location loc = Location::Immediate(-1);
    tracker_.Assign(tracker_.find(loc), value);
    return loc;
  }

  if (requirement.IsInRegister()) {
    // Anything will do, as long as it's a register.
    LocationRequirement hint =
        analysis_.GetLocationHint(analysis_iterator, value);
    auto it = tracker_.FindFreeRegister(hint);
    if (it != tracker_.end()) {
      tracker_.Assign(it, value);
      S6_VLOG(2) << "Allocated location: "
                 << tracker_.GetLocation(it).ToString() << " (hinted at "
                 << hint.ToString() << ")";
      return tracker_.GetLocation(it);
    }

    it = SelectSpillCandidate(analysis_iterator);
    if (it == tracker_.end()) {
      return absl::ResourceExhaustedError(
          "No spill candidates and no registers left");
    }
    const Value* candidate = tracker_.GetAssignedValue(it);
    S6_CHECK(candidate);
    Spill(it, move);
    tracker_.Assign(it, value);
    S6_VLOG(2) << "Spilled " << FormatOrDie(*candidate) << " from "
               << tracker_.GetLocation(it).ToString();
    return tracker_.GetLocation(it);
  }

  // If we get here the value requires a specific Location.
  S6_CHECK(requirement.IsLocation());
  Location requirement_loc = requirement.location();
  S6_CHECK(requirement_loc.IsCallStackSlot() ||
           requirement_loc.IsInRegister() || requirement_loc.IsImmediate());
  auto it = tracker_.find(requirement_loc);
  if (!tracker_.IsAssigned(it)) {
    tracker_.Assign(it, value);
    S6_VLOG(2) << "Allocated required location from free set: "
               << requirement_loc.ToString();
    return requirement_loc;
  }

  Spill(it, move);
  tracker_.Assign(it, value);
  S6_VLOG(2) << "Evicted incumbent from required location "
             << requirement_loc.ToString();
  return requirement_loc;
}

absl::Status TopDownAllocator::Allocate(const Trace& trace) {
  S6_VLOG(1) << "Allocating registers for trace: " << TraceToString(trace);
  if (trace.front()->deoptimized()) {
    S6_RET_CHECK(absl::c_all_of(trace, [](const Block* b) {
      return b->deoptimized();
    })) << "A trace must be either all deoptimized or all optimized!";
    return absl::OkStatus();
  }
  S6_RET_CHECK(absl::c_all_of(trace, [](const Block* b) {
    return !b->deoptimized();
  })) << "A trace must be either all deoptimized or all optimized!";

  // Allocate locations for live-ins to the Trace.
  const Block* front = trace.front();
  for (const Value* v : liveness_.live_ins(front)) {
    Location location = resolver_.GetLiveInLocation(*front, *v);
    tracker_.Assign(tracker_.find(location), *v);
  }
  for (const BlockArgument* v : front->block_arguments()) {
    Location location = resolver_.GetLiveInLocation(*front, *v);
    S6_VLOG(5) << "Assigning live-in " << FormatOrDie(*v) << " to "
               << location.ToString();
    // If we use a block argument that is also used as a dominating value, it
    // has two names for the same value and this isn't representable. In this
    // case, force the argument into its spill slot.
    //
    // Similarly we cannot assign an immediate to a block argument!
    auto it = tracker_.find(location);
    if (tracker_.IsAssigned(it) || location.IsImmediate()) {
      it = tracker_.find(resolver_.GetOrCreateSpillLocation(*v));
    }
    tracker_.Assign(it, *v);
  }
  spilled_values_ = resolver_.ComputeSpilledLiveIns(*trace.front());

  for (auto block_it = trace.begin(); block_it != trace.end(); ++block_it) {
    const Block* block = *block_it;

    // Double check all arguments have an assigned location. This in particular
    // is for exception handler blocks that have six arguments implicitly pushed
    // by the ExceptInst.
    for (const BlockArgument* v : block->block_arguments()) {
      auto it = tracker_.find(*v);
      if (it == tracker_.end()) {
        it = tracker_.find(resolver_.GetOrCreateSpillLocation(*v));
        tracker_.Assign(it, *v);
      }
    }

    S6_RETURN_IF_ERROR(SnapshotLiveIns(*block));
    for (const BlockArgument* arg : block->block_arguments()) {
      S6_ASSIGN_OR_RETURN(Location l, tracker_.Lookup(*arg));
      value_locations_.AddDefinition(*arg, l);
    }

    for (const Instruction& inst : *block) {
      S6_VLOG(2) << "  Visiting instruction " << FormatOrDie(inst);
      BottomUpAnalysis::iterator analysis_iterator = analysis_.lookup(inst);
      auto move = absl::bind_front(&TopDownAllocator::Copy, this, &inst);

      for (int64_t operand_index = 0; operand_index < inst.operands().size();
           ++operand_index) {
        const Value* operand = inst.operands()[operand_index];
        if (!IsAllocatableValue(operand)) continue;

        LocationRequirement requirement =
            GetRegisterRequirement(&inst, operand_index);
        S6_VLOG(3) << "    Visiting operand " << FormatOrDie(*operand)
                   << " (requirement: " << requirement.ToString() << ")";
        S6_RETURN_IF_ERROR(
            FixupOperandLocation(move, analysis_iterator, *operand, requirement)
                .status());
      }

      tracker_.Snapshot(value_locations_, inst);

      // Free up any unused operands.
      S6_RETURN_IF_ERROR(analysis_.ExpireValuesUnusedAfter(
          analysis_iterator, [&](const Value* v) -> absl::Status {
            S6_VLOG(3) << "    Expiring " << FormatOrDie(*v)
                       << " after final use";
            auto it = tracker_.find(*v);
            if (it != tracker_.end()) {
              tracker_.Free(it, *v);
            }
            return absl::OkStatus();
          }));

      if (InstructionTraits::ClobbersAllRegisters(inst)) {
        S6_VLOG(2) << "  Evicting all registers";
        for (auto it = tracker_.begin(); it != tracker_.end(); ++it) {
          if (tracker_.GetLocation(it).IsImmediate()) continue;
          // TODO: we assume all register-clobbering instructions also
          // clobber stack slots. This is true for CallPythonInst and
          // ExceptInst; we should make a trait for this.
          if (const Value* v = tracker_.GetAssignedValue(it);
              v && (tracker_.GetLocation(it).IsInRegister() ||
                    tracker_.GetLocation(it).IsCallStackSlot())) {
            Spill(it, move);
          }
        }
      }
      if (InstructionTraits::ProducesValue(inst)) {
        LocationRequirement requirement = GetRegisterRequirement(&inst);
        S6_ASSIGN_OR_RETURN(
            Location defined_loc,
            AllocateNewLocation(move, analysis_iterator, inst, requirement));
        value_locations_.AddDefinition(inst, defined_loc);
        S6_CHECK(tracker_.find(inst) != tracker_.end());
        if (analysis_.IsNotUsed(inst)) {
          S6_VLOG(2) << "Instantly freeing because value is never used: "
                     << defined_loc.ToString();
          tracker_.Free(tracker_.find(defined_loc), inst);
        }
      }
    }

    S6_RETURN_IF_ERROR(SnapshotLiveOuts(*block));

    if (std::next(block_it) == trace.end()) {
      continue;
    }

    // Modify all arguments passed to the successor to instead refer to the
    // successor's block argument.
    const Block* succ_block = *std::next(block_it);
    if (!isa<ExceptInst>(block->GetTerminator())) {
      block->GetTerminator()->ForEachArgumentOnEdge(
          succ_block, [&](const BlockArgument* arg, const Value* param) {
            auto it = tracker_.find(*param);
            if (it == tracker_.end() ||
                liveness_.live_outs(block).contains(param)) {
              // This value is passed as a parameter and also referred to by
              // liveness (or is passed as a parameter twice). It is therefore
              // referred to by two different names. This is a case that should
              // be caught by the CFG simplifier, but is valid input. We
              // therefore assign the BlockArgument a spilled location and let
              // the global move assignment pass sort it out later.
              tracker_.Assign(
                  tracker_.find(resolver_.GetOrCreateSpillLocation(*arg)),
                  *arg);
            } else if (tracker_.GetLocation(it).IsFrameSlot()) {
              // The location is a frame slot, which is immutable (frame slots
              // store SSA values, which are immutable).
              tracker_.Assign(
                  tracker_.find(resolver_.GetOrCreateSpillLocation(*arg)),
                  *arg);
            } else if (it->first.IsImmediate()) {
              // A block argument can never be an immediate; it would be
              // impossible for it to take multiple values depending on control
              // flow!
              tracker_.Assign(
                  tracker_.find(resolver_.GetOrCreateSpillLocation(*arg)),
                  *arg);
            } else {
              tracker_.Free(it, *param);
              tracker_.Assign(it, *arg);
            }
          });
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<RegisterAllocation>> AllocateRegistersWithTrace(
    const Function& f, RegisterAllocationOptions options) {
  S6_RETURN_IF_ERROR(EnsureNoCriticalEdges(f));
  S6_VLOG_LINES(8, FormatOrDie(f));

  DominatorTree domtree = ConstructDominatorTree(f);
  Liveness liveness = AnalyzeLiveness(f, domtree);
  absl::flat_hash_map<const Block*, int64_t> frequencies =
      HeuristicallyDetermineBlockFrequencies(f, domtree);
  std::vector<Trace> traces = AllocateTraces(f, frequencies);
  GlobalResolver resolver(f.entry());

  auto ra = absl::make_unique<TraceRegisterAllocation>();
  for (const Trace& trace : traces) {
    BottomUpAnalysis analysis(trace, liveness);
    TopDownAllocator a(std::move(analysis), resolver, liveness, options);
    S6_RETURN_IF_ERROR(a.Allocate(trace));
    ra->Merge(a.ConsumeResult(), a.ConsumeCopies());
  }
  resolver.Resolve();
  ra->MergeBlockCopies(resolver.ConsumeBlockCopies());
  ra->Finalize();
  S6_VLOG_LINES(2, ra->ToString(f));
  return ra;
}

}  // namespace deepmind::s6
