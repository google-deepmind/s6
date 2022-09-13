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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_SSA_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_SSA_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "utils/inlined_bit_vector.h"

// Utilities for SSA construction and dominator tree construction.

namespace deepmind::s6 {

// The dominator tree allows traversal of immediate dominator relationships.
// All blocks correspond to a node in the dominator tree, and that node's
// parent is the block's immediate dominator.
//
// The root node is the function entry, and is the only Node that has a nullptr
// parent.
//
// The dominator tree creates as a byproduct a block ordering, which is a
// reverse postorder ordering of blocks.
class DominatorTree {
 public:
  // A node in the dominator tree.
  class Node {
   public:
    const Node* parent() const { return parent_; }
    absl::Span<Node* const> children() const {
      return absl::MakeSpan(children_.begin(), children_.end());
    }
    const Block* block() const { return block_; }

   private:
    // Nodes can only be constructed by ConstructDominatorTree.
    friend DominatorTree ConstructDominatorTree(const Function&);
    explicit Node(const Block* block) : block_(block) {}

    Node* parent_ = nullptr;
    const Block* block_;
    absl::InlinedVector<Node*, 4> children_;
  };

  // Returns the Node for the given Block. The block must exist in the dominator
  // tree.
  const Node* at(const Block* b) const { return block_to_node_.at(b); }

  // Returns the Node for the given Block, or nullptr if it does not exist in
  // the dominator tree.
  const Node* GetOrNull(const Block* b) const {
    return block_to_node_.contains(b) ? block_to_node_.at(b) : nullptr;
  }

  // Returns true if the given block exists in the dominator tree. All blocks
  // that existed at dominator tree construction time exist in the tree.
  bool contains(const Block* b) const { return block_to_node_.contains(b); }

  // Convenience function to return the immediate dominator of a block.
  // Note that the root block's immediate dominator is nullptr.
  const Block* ImmediateDominator(const Block* b) const {
    const Node* n = GetOrNull(b);
    return n ? n->parent()->block() : nullptr;
  }

  // Returns the index in a postorder traversal of the given Node.
  int64_t PostorderIndex(const Node& node) const {
    return &node - nodes_.data();
  }

  // Returns true if `a` dominates `b`.
  bool Dominates(const Block* a, const Block* b) const {
    const Node* x = GetOrNull(a);
    const Node* y = GetOrNull(b);
    return Dominates(x, y);
  }

  // Returns true if `x` dominates `y`.
  bool Dominates(const Node* x, const Node* y) const {
    if (!x || !y) return false;
    while (y && x != y) {
      y = y->parent();
    }
    return x == y;
  }

  // If this Node is a loop header, returns the longest backbranch.
  //
  // A loop header can be identified by it dominating one of its predecessors,
  // and while a loop can have several backbranches the one with the lowest
  // PostorderIndex will be the longest.
  const Node* GetLongestBackbranch(const Node& node) const {
    int64_t backbranch_po_index = INT64_MAX;
    for (const Block* pred : node.block()->predecessors()) {
      const Node* pred_node = at(pred);
      if (!Dominates(&node, pred_node)) continue;
      backbranch_po_index =
          std::min(backbranch_po_index, PostorderIndex(*pred_node));
    }
    return backbranch_po_index == INT64_MAX ? nullptr
                                            : &nodes_[backbranch_po_index];
  }

  // Returns the reachable nodes in postorder.
  absl::Span<const Node> Postorder() const { return nodes_; }

  // Returns a string representation of the dominator tree.
  std::string ToString() const;

 private:
  // DominatorTrees can only be constructed by ConstructDominatorTree.
  friend DominatorTree ConstructDominatorTree(const Function&);
  DominatorTree(std::vector<Node> nodes,
                absl::flat_hash_map<const Block*, Node*> block_to_node,
                const Function& f)
      : nodes_(std::move(nodes)),
        block_to_node_(std::move(block_to_node)),
        f_(&f) {}

  // Node storage. This is postordered.
  std::vector<Node> nodes_;

  // Lookup map from block to node.
  absl::flat_hash_map<const Block*, Node*> block_to_node_;

  // The function we were created from.
  const Function* f_;
};

// Constructs the dominator tree for `f`. This never fails.
DominatorTree ConstructDominatorTree(const Function& f);

// The representation of a dominance frontier, DF(x). For every block in the
// dominator tree, this holds the set of blocks in its dominance frontier.
using DominanceFrontier =
    absl::flat_hash_map<const Block*, absl::flat_hash_set<const Block*>>;

// Constructs the dominance frontier for `f`, given its dominator tree.
DominanceFrontier ConstructDominanceFrontier(const Function& f,
                                             const DominatorTree& domtree);

// Helps constructs SSA form. Given a finite sequence of values addressed by
// number, the user:
//   1. Fills in the live-in / live-through values for a block and the values
//      defined within a block using SetLiveInValue/SetDefinedValues.
//   2. Calls InsertBlockArguments(). This inserts all block arguments and
//      calculates reaching definitions.
//   3. Calls Use()/Def() when inserting uses or defs. Use() returns the correct
//      reaching definition, Def(x) updates the live definition in a block.
//      Subsequent calls to Use() from that block or a block dominated by it
//      will receive x as the reaching definition. For this reason Use/Def must
//      be called in dominator order within a block.
//   4. Calls InsertBranchArguments() on BrInsts or JmpInsts. This populates
//      the correct arguments for target blocks.
class SsaBuilder {
 public:
  using BitVector = InlinedBitVector<128>;
  using BlockResolverFunction =
      std::function<const Block*(const DominatorTree&, const Block*)>;

  // Sets the live in set for `b`. This must include all live-through
  // values.
  void SetLiveInValues(const Block* b, BitVector values) {
    num_values_ = std::max<int64_t>(num_values_, values.size());
    live_in_.emplace(b, std::move(values));
  }

  void SetDefinedValues(const Block* b, BitVector values) {
    num_values_ = std::max<int64_t>(num_values_, values.size());
    defined_.emplace(b, std::move(values));
  }

  // Inserts BlockArguments for all values and calculates reaching definitions.
  void InsertBlockArguments(Function* f);

  // Returns the reaching definition of `value_number` in `b`. If there is an
  // existing def inside `b` that will be returned, otherwise the reaching
  // definition at the entry to `b`.
  Value* Use(int64_t value_number, const Block* b);

  // Defines a definition of `v` within `b`.
  void Def(int64_t value_number, Value* v, const Block* b);

  // Inserts arguments for the given branch, if needed.
  void InsertBranchArguments(UnconditionalTerminatorInst* ui);
  void InsertBranchArguments(BrInst* bi);

  // Sets the block resolver function, which is called when
  // InsertBranchArguments does not find a block in the dominator tree. Given a
  // block `b` it must return the most immediate dominator block of `b` that is
  // in the dominator tree, `b'`. It is up to the user to ensure that the
  // returned block b':
  //   1. Dominates b
  //   2. Dominates exactly the same set of blocks that exist in the dominator
  //      tree as b.
  void SetBlockResolverFunction(const BlockResolverFunction& fn) {
    block_resolver_function_ = fn;
  }

 private:
  // Convenience functions to return a BitVector that is extended to num_values_
  // in size.
  const BitVector& GetLiveInValues(const Block* b);
  const BitVector& GetDefinedValues(const Block* b);

  // Helper to insert branch arguments.
  void InsertBranchArguments(const Block* predecessor, const Block* successor,
                             MutableOperandList arguments);

  // The dominator tree, once constructed by InsertBlockArguments.
  absl::optional<DominatorTree> domtree_;

  // The set of live-in and live-through variables per block.
  absl::flat_hash_map<const Block*, BitVector> live_in_;

  // The set of defined variables per block.
  absl::flat_hash_map<const Block*, BitVector> defined_;

  // The variables live-in to a block for which BlockArguments have been
  // inserted by InsertBlockArguments. The block arguments were inserted in
  // BitVector iteration order.
  absl::flat_hash_map<const Block*, BitVector> block_arguments_;

  // The currently live Def within a block for a given {block, variable}. This
  // is initially populated with block arguments (the implicit reaching def set
  // at block entry). Calls to Def() update this.
  absl::flat_hash_map<std::pair<const Block*, int64_t>, Value*> live_defs_;

  // The number of unique values seen by calls to SetLiveIn/SetDefinedValues.
  int64_t num_values_ = 0;

  // The block resolver function, used to find blocks within the dominator tree
  // for InsertBranchArguments when the function has been modified since
  BlockResolverFunction block_resolver_function_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_SSA_H_
