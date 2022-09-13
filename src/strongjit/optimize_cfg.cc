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

#include "strongjit/optimize_cfg.h"

#include <cstdint>

#include "absl/status/status.h"
#include "cppitertools/enumerate.hpp"
#include "cppitertools/reversed.hpp"
#include "strongjit/formatter.h"
#include "strongjit/optimizer_util.h"

namespace deepmind::s6 {

absl::Status EliminateTrivialJumpsPattern::Apply(JmpInst* jmp,
                                                 Rewriter& rewriter) {
  Block* b = jmp->parent();

  Block* succ = jmp->unique_successor();
  if (jmp->parent()->size() == 1 && b->block_arguments_empty() &&
      jmp->arguments().empty()) {
    // This is an empty block, so we can simply reroute all predecessors and
    // remove it.
    succ->RemovePredecessor(b);
    for (Block* pred : b->predecessors()) {
      for (Block*& old_succ : pred->GetTerminator()->mutable_successors()) {
        if (old_succ == b) old_succ = succ;
      }
      succ->AddPredecessor(pred);
    }
    b->erase();
    return absl::OkStatus();
  }

  if (succ->predecessors().size() != 1) {
    return absl::FailedPreconditionError("Multiple incoming edges");
  }
  jmp->ForEachArgumentOnEdge(succ, [&](BlockArgument* arg, Value* param) {
    rewriter.ReplaceAllUsesWith(*arg, *param);
  });

  // Merge `succ` into `b`.
  for (Block* succ_succ : succ->GetTerminator()->successors()) {
    S6_CHECK(succ_succ);
    succ_succ->ReplacePredecessor(succ, b);
  }

  // Splicing the block here is fine because the rewriter only maintains a
  // cursor and use lists; the use lists are not modified and the cursor is
  // already pre-advanced to succ->start().
  b->splice(b->end(), succ);
  succ->RemovePredecessor(b);
  succ->erase();
  rewriter.erase(*jmp);
  return absl::OkStatus();
}

absl::Status EliminateUnusedBlockArgumentsPattern::Apply(Block* b,
                                                         Rewriter& rewriter) {
  struct BlockArgAndIndex {
    int64_t index;
    const BlockArgument* block_argument;
  };
  std::vector<BlockArgAndIndex> unused;

  for (auto [index, arg] : iter::enumerate(b->block_arguments())) {
    if (rewriter.GetUsesOf(*arg).empty()) {
      unused.push_back({static_cast<int64_t>(index), arg});
    }
  }
  if (unused.empty())
    return absl::FailedPreconditionError("No unused arguments");

  int64_t max_shift = 0;
  for (Block* predecessor : b->predecessors()) {
    TerminatorInst* terminator = predecessor->GetTerminator();
    S6_CHECK(terminator);
    max_shift = std::max(max_shift, terminator->num_implicit_arguments());
  }
  STLEraseIf(unused,
             [max_shift](BlockArgAndIndex b) { return b.index < max_shift; });

  // Remove unused args from uses of the block.
  for (Block* predecessor : b->predecessors()) {
    TerminatorInst* terminator = predecessor->GetTerminator();

    int64_t index_shift = terminator->num_implicit_arguments();

    for (auto [successor_index, successor] :
         iter::enumerate(terminator->successors())) {
      if (successor != b) continue;
      // Reverse unused order to avoid invalidating later indices.
      for (auto [index, arg] : iter::reversed(unused)) {
        S6_CHECK_GE(index, index_shift);
        terminator->RemoveSuccessorArgumentAt(successor_index,
                                              index - index_shift);
      }
    }
  }
  // Remove unused args from the block.
  // Reverse unused order to avoid invalidating later indices.
  for (auto [index, arg] : iter::reversed(unused)) {
    b->RemoveBlockArgumentAt(index);
  }

  return absl::OkStatus();
}

}  // namespace deepmind::s6
