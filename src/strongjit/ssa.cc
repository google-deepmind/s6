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

#include "strongjit/ssa.h"

#include <cstdint>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {
namespace {

using Node = DominatorTree::Node;

// The two-finger "intersect" function from Cooper et al. This works on Nodes
// rather than Blocks as Nodes are stored in contiguous storage in post-order.
Node* Intersect(Node* b1, Node* b2) {
  Node* finger1 = b1;
  Node* finger2 = b2;

  // Node storage is in postorder, so use a less-than comparison on node
  // addresses.
  auto post_order_before = [](Node* n1, Node* n2) {
    return reinterpret_cast<uint64_t>(n1) < reinterpret_cast<uint64_t>(n2);
  };

  while (finger1 != finger2) {
    // Note the ordering - we use less-than because of post ordering. RPO would
    // require greater-than.
    while (post_order_before(finger1, finger2)) {
      finger1 = const_cast<Node*>(finger1->parent());
    }
    while (post_order_before(finger2, finger1)) {
      finger2 = const_cast<Node*>(finger2->parent());
    }
  }
  return finger1;
}

// Helper to traverse `b` and its children in postorder and invoke `fn`.
void TraverseInPostorder(const Block& b,
                         absl::flat_hash_set<const Block*>* visited,
                         absl::FunctionRef<void(const Block*)> fn) {
  if (!visited->emplace(&b).second) return;
  for (const Block* succ : b.GetTerminator()->successors()) {
    S6_CHECK(succ);
    TraverseInPostorder(*succ, visited, fn);
  }
  fn(&b);
}

// Helper to traverse `f` and its children in postorder and invoke `fn`.
void TraverseInPostorder(const Function& f,
                         absl::FunctionRef<void(const Block*)> fn) {
  absl::flat_hash_set<const Block*> visited;
  TraverseInPostorder(f.entry(), &visited, fn);
}

void ToStringHelper(const ValueNumbering& vn, int64_t indent,
                    const DominatorTree::Node* n, std::string* s) {
  absl::StrAppend(s, std::string(indent, ' '), "&", vn.at(n->block()));
  if (!n->children().empty())
    absl::StrAppend(s, " [", n->children().size(), " children]");
  *s += "\n";
  for (const DominatorTree::Node* child : n->children())
    ToStringHelper(vn, indent + 2, child, s);
}

}  // namespace

std::string DominatorTree::ToString() const {
  ValueNumbering value_numbering = ComputeValueNumbering(*f_);
  const Node* n = &nodes_.back();
  std::string s =
      absl::StrCat("Dominator tree for function \"", f_->name(), "\":\n");
  ToStringHelper(value_numbering, 0, n, &s);
  return s;
}

DominatorTree ConstructDominatorTree(const Function& f) {
  // Use the "engineered algorithm" from A Simple, Fast Dominance Algorithm
  // Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
  //  https://www.cs.rice.edu/~keith/EMBED/dom.pdf

  // Construct the dominator tree inside an array of Nodes in post-order. The
  // Node::parent_ pointer is used during construction as the immediate
  // dominator. This replaces the `doms` map in the above algorithm.
  std::vector<DominatorTree::Node> nodes_in_po;
  absl::flat_hash_map<const Block*, DominatorTree::Node*> block_to_node;
  nodes_in_po.reserve(f.num_blocks());

  TraverseInPostorder(f, [&](const Block* b) {
    nodes_in_po.push_back(DominatorTree::Node(b));
    block_to_node[b] = &nodes_in_po.back();
  });

  // The root is its own parent during domtree construction.
  nodes_in_po.back().parent_ = &nodes_in_po.back();

  bool changed = true;
  while (changed) {
    changed = false;
    // Iterate in RPO, skipping the root node.
    for (auto node = std::next(nodes_in_po.rbegin());
         node != nodes_in_po.rend(); ++node) {
      // Pick any predecessor that is already processed.
      auto predecessors = node->block()->predecessors();
      auto it =
          absl::c_find_if(predecessors, [&](const Block* b) -> const Node* {
            if (!block_to_node.contains(b)) return nullptr;
            return block_to_node.at(b)->parent();
          });
      S6_CHECK(it != predecessors.end());
      DominatorTree::Node* new_idom_node = block_to_node.at(*it);

      // For all other predecessors.
      for (const Block* pred : predecessors) {
        if (!block_to_node.contains(pred)) continue;
        Node* pred_node = block_to_node.at(pred);
        if (pred != new_idom_node->block() && pred_node->parent())
          new_idom_node = Intersect(pred_node, new_idom_node);
      }

      if (node->parent_ != new_idom_node) {
        changed = true;
        node->parent_ = new_idom_node;
      }
    }
  }

  // Now we can construct the actual dominator tree.
  for (DominatorTree::Node& node : nodes_in_po) {
    S6_CHECK(node.parent_);
    if (node.parent_ != &node) node.parent_->children_.push_back(&node);
  }

  // For simplicity's sake for users, set parent(root) = nullptr rather than
  // parent(root) = parent. This makes iteration over the dominator tree easier.
  nodes_in_po.back().parent_ = nullptr;

  return DominatorTree(std::move(nodes_in_po), std::move(block_to_node), f);
}

DominanceFrontier ConstructDominanceFrontier(const Function& f,
                                             const DominatorTree& domtree) {
  // Figure 5 from A Simple, Fast Dominance Algorithm
  // Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
  //  https://www.cs.rice.edu/~keith/EMBED/dom.pdf
  DominanceFrontier frontier;
  for (const Block& b : f) {
    if (b.predecessors().size() == 1) continue;
    for (const Block* pred : b.predecessors()) {
      const Block* runner = pred;
      const Block* idom = domtree.ImmediateDominator(&b);
      while (idom && runner && runner != idom) {
        frontier[runner].insert(&b);
        runner = domtree.ImmediateDominator(runner);
      }
    }
  }

  return frontier;
}

const SsaBuilder::BitVector& SsaBuilder::GetLiveInValues(const Block* b) {
  BitVector& v = live_in_[b];
  v.resize(num_values_);
  return v;
}

const SsaBuilder::BitVector& SsaBuilder::GetDefinedValues(const Block* b) {
  BitVector& v = defined_[b];
  v.resize(num_values_);
  return v;
}

void SsaBuilder::InsertBlockArguments(Function* f) {
  domtree_ = ConstructDominatorTree(*f);
  DominanceFrontier frontier = ConstructDominanceFrontier(*f, *domtree_);

  for (const Block& b : *f) {
    block_arguments_[&b].resize(num_values_);
  }

  // This is the standard worklist algorithm that implicitly traverses the
  // iterated dominance frontier.
  absl::InlinedVector<const Block*, 8> worklist;
  absl::flat_hash_set<const Block*> ever_on_worklist;
  absl::flat_hash_set<const Block*> already_has_phi;
  for (int64_t v = 0; v < num_values_; ++v) {
    worklist.clear();
    ever_on_worklist.clear();
    already_has_phi.clear();

    // Populate the worklist with all blocks that define `v`.
    for (const Block& b : *f) {
      if (GetDefinedValues(&b).get_bit(v)) {
        worklist.push_back(&b);
        ever_on_worklist.insert(&b);
      }
    }

    while (!worklist.empty()) {
      const Block* n = worklist.back();
      worklist.pop_back();

      if (!frontier.contains(n)) continue;
      for (const Block* d : frontier.at(n)) {
        if (!already_has_phi.insert(d).second) continue;
        if (GetLiveInValues(d).get_bit(v)) {
          block_arguments_[d].set_bit(v);
          // Note that because the outermost loop is over `v` in ascending order
          // we can create the block argument here while keeping the argument
          // list ordered by ascending `v`.
          //
          // Note also that the const_cast is OK because we're given the mutable
          // function as input, we've just fished this block out of a const
          // analysis.
          live_defs_[{d, v}] = const_cast<Block*>(d)->CreateBlockArgument();
        }
        if (ever_on_worklist.insert(d).second) {
          worklist.push_back(d);
        }
      }
    }
  }
}

void SsaBuilder::Def(int64_t value_number, Value* v, const Block* b) {
  live_defs_[{b, value_number}] = v;
}

Value* SsaBuilder::Use(int64_t value_number, const Block* b) {
  if (auto it = live_defs_.find({b, value_number}); it != live_defs_.end())
    return it->second;

  if (!domtree_->contains(b) && block_resolver_function_) {
    b = block_resolver_function_(*domtree_, b);
  }
  const DominatorTree::Node* n = domtree_->at(b);
  while (true) {
    auto it = live_defs_.find({n->block(), value_number});
    if (it != live_defs_.end()) {
      return it->second;
    }

    S6_CHECK(n->parent()) << "Walked off the end of the dominator tree without "
                             "seeing a def for value "
                          << value_number << ": " << FormatOrDie(*b->parent());
    n = n->parent();
  }
}

void SsaBuilder::InsertBranchArguments(const Block* predecessor,
                                       const Block* successor,
                                       MutableOperandList arguments) {
  if (!block_arguments_.contains(successor)) return;
  const BitVector& arg_variables = block_arguments_[successor];
  for (int64_t v = 0; v < arg_variables.size(); ++v) {
    if (!arg_variables.get_bit(v)) continue;
    // Find the reaching def at the end of `predecessor` for `v`.
    arguments.push_back(Use(v, predecessor));
  }
}

void SsaBuilder::InsertBranchArguments(UnconditionalTerminatorInst* ui) {
  InsertBranchArguments(ui->parent(), ui->unique_successor(),
                        ui->mutable_arguments());
}

void SsaBuilder::InsertBranchArguments(BrInst* bi) {
  // Note that due to the way the argument inserter works, it is important to
  // populate the true successor's arguments before the false sucessor's (they
  // both append at the end of the operand list).
  InsertBranchArguments(bi->parent(), bi->true_successor(),
                        bi->mutable_true_arguments());
  InsertBranchArguments(bi->parent(), bi->false_successor(),
                        bi->mutable_false_arguments());
}

}  // namespace deepmind::s6
