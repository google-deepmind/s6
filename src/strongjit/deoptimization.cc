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

#include "strongjit/deoptimization.h"

#include <iterator>
#include <random>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/ssa.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {
bool EdgeIsDeoptimized(const Block& pred, const Block& succ) {
  if (pred.deoptimized()) return true;
  const BrInst* br = dyn_cast<BrInst>(pred.GetTerminator());
  if (!br) return false;
  bool true_deopt = br->true_successor() == &succ && br->true_deoptimized();
  bool false_deopt = br->false_successor() == &succ && br->false_deoptimized();

  // Special case: if both successors are `succ`, ensure both true_deopt and
  // false_deopt!
  if (br->true_successor() == br->false_successor()) {
    return true_deopt && false_deopt;
  }
  return true_deopt || false_deopt;
}

// Traverses the function down all paths starting from `block` until:
//  1. A block with no successors (ReturnInst or ExceptInst)
//  2. A BytecodeBeginInst
//
// Throughout the traversal, track all used values and all defined values,
// including the required_values set of the found BytecodeBeginInsts. The
// returned sets can be differenced (used - defined) to determine the exact
// set of values that must be available (materialized) for the deoptimization
// handler to traverse to a bytecode_begin boundary.
void DiscoverRequiredValuesForDeoptimization(
    Block* block, absl::flat_hash_set<Value*>& defined_values,
    absl::flat_hash_set<Value*>& used_values) {
  // Crawl the frontier of successors until a BytecodeBeginInst is found.
  absl::flat_hash_set<Block*> seen;
  seen.insert(block);
  std::vector<Block*> worklist(1, block);
  std::vector<BytecodeBeginInst*> ret;

  while (!worklist.empty()) {
    Block* b = worklist.back();
    worklist.pop_back();

    for (BlockArgument* ba : b->block_arguments()) {
      defined_values.insert(ba);
    }

    BytecodeBeginInst* begin_inst = nullptr;
    for (Instruction& inst : *b) {
      begin_inst = dyn_cast<BytecodeBeginInst>(&inst);
      if (begin_inst) break;

      if (InstructionTraits::ProducesValue(inst)) {
        defined_values.insert(&inst);
      }
      for (Value* v : inst.operands()) {
        if (v && !isa<Block>(v)) used_values.insert(v);
      }
    }

    if (begin_inst) {
      for (Value* v : begin_inst->value_stack()) {
        used_values.insert(v);
      }
      for (Value* v : begin_inst->fastlocals()) {
        used_values.insert(v);
      }
      continue;
    }

    for (Block* succ : b->GetTerminator()->successors()) {
      S6_CHECK(succ);
      if (seen.insert(succ).second) {
        worklist.push_back(succ);
      }
    }
  }
}
}  // namespace

absl::Status MarkDeoptimizedBlocks(Function& f) {
  // If a block's predecessor edges are always deoptimize edges or the block's
  // predecessors are all deoptimized, mark this block as deoptimized. Repeat
  // to fixpoint.

  auto domtree = ConstructDominatorTree(f);
  bool changed = true;
  while (changed) {
    changed = false;
    for (Block& b : f) {
      if (b.deoptimized() || b.predecessors().empty()) continue;

      if (!absl::c_all_of(b.predecessors(), [&b, &domtree](const Block* pred) {
            return domtree.Dominates(&b, pred) || EdgeIsDeoptimized(*pred, b);
          })) {
        continue;
      }

      b.set_deoptimized(true);
      changed = true;
    }
  }

  return absl::OkStatus();
}

absl::Status RewriteFunctionForDeoptimization(Function& f) {
  S6_CHECK_OK(VerifyFunction(f));
  ValueNumbering vn = ComputeValueNumbering(f);
  for (Block& b : f) {
    if (b.deoptimized()) continue;
    BrInst* br = dyn_cast<BrInst>(b.GetTerminator());
    if (!br) continue;

    Block* succ;
    // If both branch targets are deoptimized, just pick the true successor
    // arbitrarily.
    if (br->true_deoptimized() && br->true_successor()->deoptimized()) {
      S6_RET_CHECK(br->true_deoptimized());
      succ = br->true_successor();
    } else if (br->false_deoptimized() &&
               br->false_successor()->deoptimized()) {
      S6_RET_CHECK(br->false_deoptimized());
      succ = br->false_successor();
    } else {
      continue;
    }

    absl::flat_hash_set<Value*> defined, used;
    DiscoverRequiredValuesForDeoptimization(succ, defined, used);

    // Compute value_set = used - defined. This is the set of values we must
    // materialize.
    std::vector<Value*> value_set;
    for (const auto v : used) {
      if (!defined.contains(v)) value_set.push_back(v);
    }

    // Give the value set a deterministic order.
    absl::c_sort(value_set, [&vn](const Value* a, const Value* b) {
      return vn.at(a) < vn.at(b);
    });
    if (succ == br->true_successor()) {
      b.Create<DeoptimizeIfInst>(br->condition(),
                                 /*negated=*/false, br->true_successor(),
                                 br->true_arguments(), br->false_successor(),
                                 br->false_arguments(), value_set);
    } else {
      b.Create<DeoptimizeIfInst>(br->condition(),
                                 /*negated=*/true, br->false_successor(),
                                 br->false_arguments(), br->true_successor(),
                                 br->true_arguments(), value_set);
    }
    br->erase();
  }

  // Any SafepointInsts that reference RematerializeInsts must have the
  // RematerializeInsts' operands added to the "extras" list, so that the
  // lifetime of the RematerializeInsts' operands are tied to the SafepointInst.
  for (Block& b : f) {
    if (b.deoptimized()) continue;
    for (Instruction& inst : b) {
      SafepointInst* safepoint = dyn_cast<SafepointInst>(&inst);
      if (!safepoint) continue;
      std::vector<Value*> extras;
      for (Value* v : safepoint->operands()) {
        RematerializeInst* remat = dyn_cast<RematerializeInst>(v);
        if (!remat) continue;
        absl::c_copy(remat->operands(), std::back_inserter(extras));
      }
      if (!extras.empty()) {
        safepoint->mutable_extras().resize(extras.size());
        absl::c_copy(extras, safepoint->mutable_extras().begin());
      }
    }
  }

  // Sort the blocks such that deoptimized blocks come after optimized blocks,
  // but otherwise their relative order is not changed.
  f.SortBlocks([](const Block* a, const Block* b) {
    // Same deoptimization kind? Unordered.
    if (a->deoptimized() == b->deoptimized()) return false;
    if (a->deoptimized()) return false;
    // Otherwise b->deoptimized() && !a->deoptimized();
    return true;
  });

  return absl::OkStatus();
}

absl::Status StressTestByDeoptimizingRandomly(
    Function& f, std::default_random_engine& rng,
    float probability_to_deoptimize_edge) {
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

  for (Block& b : f) {
    BrInst* br = dyn_cast<BrInst>(b.GetTerminator());
    if (!br || distribution(rng) > probability_to_deoptimize_edge) continue;
    bool deoptimize_true_branch = distribution(rng) >= 0.5f;
    if (deoptimize_true_branch) {
      br->set_true_deoptimized(true);
    } else {
      br->set_false_deoptimized(true);
    }
  }

  return absl::OkStatus();
}

}  // namespace deepmind::s6
