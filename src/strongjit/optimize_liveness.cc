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

#include "strongjit/optimize_liveness.h"

namespace deepmind::s6 {

// The LiveInfo constructor actually performs the analysis.
LiveInfo::LiveInfo(const Function& f) {
  // This function implement a reverse/backward data-flow analysis.
  // See https://en.wikipedia.org/wiki/Data-flow_analysis.

  numbering_ = ComputeValueNumbering(f);

  // First compute the gen and kill set of each block.
  // Since this is a backward live value analysis:
  //  - The generated live values are values used by the block.
  //  - The killed live values are values produced by the block that are thus
  //    not live before the block entry.
  absl::flat_hash_map<const Block*, BlockInfoBits> gens;
  absl::flat_hash_map<const Block*, BlockInfoBits> kills;
  for (const Block& b : f) {
    BlockInfoBits& gen = gens[&b];
    BlockInfoBits& kill = kills[&b];
    gen.resize(numbering_.size());
    kill.resize(numbering_.size());

    for (const Instruction& inst : b) {
      for (const Value* operand : inst.operands()) {
        if (operand && (InstructionTraits::ProducesValue(*operand) ||
                        isa<BlockArgument>(operand))) {
          gen.set_bit(numbering_[operand]);
        }
      }
      // If that instruction produce a value, that value must be killed.
      if (InstructionTraits::ProducesValue(inst)) {
        kill.set_bit(numbering_[&inst]);
      }
    }

    // Block argument are also value produced by a block.
    for (const Value* v : b.block_arguments()) kill.set_bit(numbering_[v]);
  }

  // Now do the proper reverse data-flow algorithm using a worklist.
  Worklist<const Block*> worklist;
  // Initialize each block live variable to be their gen minus their kill
  // It is certain that at least all values that are used by a block but not
  // produced by it are live at the block entry. Thus this is a sound
  // lower-bound for the set of live values of a block. The remaining algorithm
  // can only increase that set.
  //
  // The invariant that the live info of a block does not contains its kill set
  // is maintained by the algorithm.
  for (const Block& b : f) {
    worklist.Push(&b);
    infos_[&b].bits = gens[&b];
    infos_[&b].bits.Difference(kills[&b]);
  }

  while (!worklist.empty()) {
    const Block* b = worklist.Pop();
    BlockInfoBits info = infos_.at(b).bits;
    // Process a new block b by adding the live variable of b to all it's
    // predecessor live variable.
    for (const Block* pred : b->predecessors()) {
      BlockInfoBits& pred_info = infos_.at(pred).bits;
      BlockInfoBits newinfo = pred_info;
      newinfo.Union(info);
      // Maintain the invariant that a block live info does not contain its kill
      // set.
      newinfo.Difference(kills[pred]);
      if (newinfo != pred_info) {
        // If the live info of the predecessor has changed, then the predecessor
        // needs to be processed again to propagate the change.
        pred_info = newinfo;
        worklist.PushIfNew(pred);
      }
    }
  }
  FillListFromBits();
}

}  // namespace deepmind::s6
