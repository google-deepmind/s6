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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/base.h"
#include "strongjit/builder.h"
#include "strongjit/formatter.h"
#include "strongjit/instructions.h"
#include "strongjit/test_util.h"

namespace deepmind::s6 {
namespace {

using testing::Eq;

std::string ToString(const DominanceFrontier& frontier, const Function& f) {
  ValueNumbering vn = ComputeValueNumbering(f);
  // Ensure we emit a sorted order.
  std::vector<std::string> v;
  for (const auto& [block, df] : frontier) {
    v.push_back(absl::StrCat(
        "DF[&", vn[block],
        "] = ", absl::StrJoin(df, ", ", [&](std::string* s, const Block* b) {
          absl::StrAppend(s, "&", vn[b]);
        })));
  }
  absl::c_sort(v);
  return absl::StrJoin(v, "\n");
}

TEST(DominatorTreeTest, Fork) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &1
  // &0:
  //   unreachable
  // &1:
  //   unreachable
  b.Conditional(nullptr, [&](Builder b) {
    b.Unreachable();
    return Builder::DoesNotReturn();
  });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();

  const DominatorTree::Node& entry_node = domtree.Postorder().back();
  // Entry node must have null parent.
  EXPECT_THAT(entry_node.parent(), Eq(nullptr));
  // The other two blocks have the entry as their parent.
  EXPECT_THAT(domtree.Postorder()[0].parent(), Eq(&entry_node));
  EXPECT_THAT(domtree.Postorder()[1].parent(), Eq(&entry_node));
}

TEST(DominatorTreeTest, Triangle) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &1
  // &0:
  //   jmp &1
  // &1:
  //  unreachable
  b.Conditional(nullptr, [&](Builder b) { return Builder::ValueList(); });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  EXPECT_THAT(RawString(domtree.ToString()),
              Eq(RawString(R"(Dominator tree for function "test":
&0 [2 children]
  &4
  &2
)")));
}

TEST(DominatorTreeTest, Diamond) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &1
  // &0:
  //   jmp &1
  // &1:
  //   jmp &2
  // &2:
  //  unreachable
  b.Conditional(
      nullptr, [&](Builder b) { return Builder::ValueList(); },
      [&](Builder b) { return Builder::ValueList(); });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  EXPECT_THAT(RawString(domtree.ToString()),
              Eq(RawString(R"(Dominator tree for function "test":
&0 [3 children]
  &6
  &2
  &4
)")));
}

TEST(DominatorTreeTest, ElongatedDiamond) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &2
  // &0:
  //   jmp &1
  // &1:
  //   jmp &3
  // &2:
  //   jmp &3
  // &3:
  //  unreachable
  b.Conditional(
      nullptr,
      [&](Builder b) {
        b.block()->Split(b.insert_point());
        return Builder::ValueList();
      },
      [&](Builder b) { return Builder::ValueList(); });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  EXPECT_THAT(RawString(domtree.ToString()),
              Eq(RawString(R"(Dominator tree for function "test":
&0 [3 children]
  &8
  &2 [1 children]
    &4
  &6
)")));
}

TEST(DominatorTreeTest, Loop) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &2
  // &0:
  //   jmp &1
  // &1:
  //   jmp &entry
  // &2:
  //  unreachable
  b.Conditional(nullptr, [&](Builder b) {
    JmpInst* jmp = b.Jmp(&f.entry());
    b.block()->Split(jmp->GetIterator());
    return Builder::DoesNotReturn();
  });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  EXPECT_THAT(RawString(domtree.ToString()),
              Eq(RawString(R"(Dominator tree for function "test":
&0 [2 children]
  &2 [1 children]
    &4
  &6
)")));
}

TEST(DominatorTreeTest, Unreachable) {
  Function f("test");

  // Creates:
  // &0:
  //   jmp &3
  // &1: // unreachable
  //   br &2, &3
  // &2: // unreachable
  //   jmp &3
  // &3:
  //   unreachable
  Block* one = f.CreateBlock();
  Block* two = f.CreateBlock();
  Block* three = f.CreateBlock();
  Block* four = f.CreateBlock();

  one->Create<JmpInst>(four);
  four->AddPredecessor(one);
  two->Create<BrInst>(nullptr, three, four);
  three->AddPredecessor(two);
  four->AddPredecessor(two);
  three->Create<JmpInst>(four);
  four->AddPredecessor(three);
  four->Create<UnreachableInst>();
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  EXPECT_THAT(RawString(domtree.ToString()),
              Eq(RawString(R"(Dominator tree for function "test":
&0 [1 children]
  &6
)")));
}

TEST(DominanceFrontierTest, Join) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &2
  // &0:
  //   jmp &1
  // &1:
  //   jmp &3
  // &2:
  //   jmp &3
  // &3:
  //  unreachable
  b.Conditional(
      nullptr,
      [&](Builder b) {
        b.block()->Split(b.insert_point());
        return Builder::ValueList();
      },
      [&](Builder b) { return Builder::ValueList(); });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  DominanceFrontier frontier = ConstructDominanceFrontier(f, domtree);
  EXPECT_THAT(RawString(ToString(frontier, f)), Eq(RawString(R"(DF[&2] = &8
DF[&4] = &8
DF[&6] = &8)")));
}

TEST(DominanceFrontierTest, NestedJoin) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:
  //   br &0, &4
  // &0:
  //   br &1, &2
  // &1:
  //   jmp &3
  // &2:
  //   jmp &3
  // &3:
  //   jmp &5
  // &4:
  //   jmp &5
  // &5:
  //  unreachable
  b.Conditional(
      nullptr,
      [&](Builder b) {
        b.Conditional(nullptr, [&](Builder b) { return Builder::ValueList(); });
        return Builder::ValueList();
      },
      [&](Builder b) { return Builder::ValueList(); });
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  DominanceFrontier frontier = ConstructDominanceFrontier(f, domtree);
  EXPECT_THAT(RawString(ToString(frontier, f)), Eq(RawString(R"(DF[&2] = &10
DF[&4] = &6
DF[&6] = &10
DF[&8] = &10)")));
}

TEST(DominanceFrontierTest, Unreachable) {
  Function f("test");

  // Creates:
  // &0:
  //   jmp &3
  // &1: // unreachable
  //   br &2, &3
  // &2: // unreachable
  //   jmp &3
  // &3:
  //   unreachable
  Block* one = f.CreateBlock();
  Block* two = f.CreateBlock();
  Block* three = f.CreateBlock();
  Block* four = f.CreateBlock();

  one->Create<JmpInst>(four);
  four->AddPredecessor(one);
  two->Create<BrInst>(nullptr, three, four);
  three->AddPredecessor(two);
  four->AddPredecessor(two);
  three->Create<JmpInst>(four);
  four->AddPredecessor(three);
  four->Create<UnreachableInst>();
  S6_LOG(INFO) << FormatOrDie(f);

  DominatorTree domtree = ConstructDominatorTree(f);
  S6_LOG(INFO) << domtree.ToString();
  DominanceFrontier frontier = ConstructDominanceFrontier(f, domtree);
  EXPECT_THAT(RawString(ToString(frontier, f)), Eq(RawString(R"(DF[&2] = &6
DF[&4] = &6)")));
}

TEST(SsaBuilderTest, NestedJoin) {
  Function f("test");
  Builder b(&f);

  // Creates:
  // &entry:        def(0, 1)
  //   br &0, &4
  // &0:
  //   br &1, &2
  // &1:
  //   jmp &3       def(0)
  // &2:
  //   jmp &3
  // &3:
  //   jmp &5
  // &4:
  //   jmp &5
  // &5:
  //  unreachable   use(0, 1)
  SsaBuilder ssa;

  SsaBuilder::BitVector zero_and_one(2);
  zero_and_one.SetBits(0, 2);
  SsaBuilder::BitVector zero(2);
  zero.set_bit(0);

  ssa.SetDefinedValues(b.block(), zero_and_one);
  Value* v0 = b.Int64(1);
  Value* v1 = b.Int64(2);
  Value* v2 = b.Int64(3);
  Block* entry = b.block();

  Block* midpoint;
  b.Conditional(
      nullptr,
      [&](Builder b) {
        ssa.SetLiveInValues(b.block(), zero_and_one);
        b.Conditional(nullptr, [&](Builder b) {
          ssa.SetLiveInValues(b.block(), zero_and_one);
          ssa.SetDefinedValues(b.block(), zero);
          midpoint = b.block();
          return Builder::ValueList();
        });
        ssa.SetLiveInValues(b.block(), zero_and_one);
        return Builder::ValueList();
      },
      [&](Builder b) {
        ssa.SetLiveInValues(b.block(), zero_and_one);
        return Builder::ValueList();
      });
  ssa.SetLiveInValues(b.block(), zero_and_one);
  AddInst* add = b.Add(nullptr, nullptr);
  Block* final_block = b.block();

  S6_LOG(INFO) << FormatOrDie(f);

  ssa.InsertBlockArguments(&f);

  // Check the result after block argument insertion.
  EXPECT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function test {
&0:                                                         // entry point
  %1 = constant $1
  %2 = constant $2
  %3 = constant $3
  br %-1, &5, &12

&5:                                                         // preds: &0
  br %-1, &7, &9

&7:                                                         // preds: &5
  jmp &9

&9: [ %10 ]                                                 // preds: &5, &7
  jmp &14

&12:                                                        // preds: &0
  jmp &14

&14: [ %15 ]                                                // preds: &12, &9
  %16 = add i64 %-1, %-1
  unreachable
})")));

  ssa.Def(0, v0, entry);
  ssa.Def(1, v1, entry);
  ssa.Def(0, v2, midpoint);
  *add->mutable_lhs() = ssa.Use(0, final_block);
  *add->mutable_rhs() = ssa.Use(1, final_block);
  for (Block& b : f) {
    if (BrInst* br = dyn_cast<BrInst>(b.GetTerminator())) {
      ssa.InsertBranchArguments(br);
    } else if (JmpInst* ji = dyn_cast<JmpInst>(b.GetTerminator())) {
      ssa.InsertBranchArguments(ji);
    }
  }

  // And after full rewriting.
  EXPECT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function test {
&0:                                                         // entry point
  %1 = constant $1
  %2 = constant $2
  %3 = constant $3
  br %-1, &5, &12

&5:                                                         // preds: &0
  br %-1, &7, &9 [ %1 ]

&7:                                                         // preds: &5
  jmp &9 [ %3 ]

&9: [ %10 ]                                                 // preds: &5, &7
  jmp &14 [ %10 ]

&12:                                                        // preds: &0
  jmp &14 [ %1 ]

&14: [ %15 ]                                                // preds: &12, &9
  %16 = add i64 %15, %2
  unreachable
})")));
}

}  // namespace
}  // namespace deepmind::s6
