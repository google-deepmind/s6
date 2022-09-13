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

#include "strongjit/block.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {
namespace {
using testing::ElementsAre;
using testing::Eq;

// Given an iterator range returning references, return pointers instead.
template <typename T, typename R>
std::vector<T*> RefToPtr(R&& range) {
  std::vector<T*> v;
  for (auto& i : range) {
    v.push_back(&i);
  }
  return v;
}

TEST(BaseTest, CanIterateInstructions) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst1 = f.Create<ConstantInst>(42);
  Instruction* inst2 = f.Create<ConstantInst>(42);
  Instruction* inst3 = f.Create<ConstantInst>(42);
  block->push_back(inst1);
  block->push_back(inst2);
  block->push_back(inst3);

  ASSERT_THAT(RefToPtr<Instruction>(block->instructions()),
              ElementsAre(inst1, inst2, inst3));

  inst2->RemoveFromParent();
  ASSERT_THAT(RefToPtr<Instruction>(block->instructions()),
              ElementsAre(inst1, inst3));

  block->insert(Block::iterator(inst1), inst2);
  ASSERT_THAT(RefToPtr<Instruction>(block->instructions()),
              ElementsAre(inst2, inst1, inst3));
}

TEST(BaseTest, CanIterateBlocks) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Block* block2 = f.CreateBlock();
  Block* block3 = f.CreateBlock();

  ASSERT_THAT(RefToPtr<Block>(f), ElementsAre(block1, block2, block3));

  Block* block4 = f.CreateBlock(Function::iterator(block1));
  ASSERT_THAT(RefToPtr<Block>(f), ElementsAre(block4, block1, block2, block3));
}

TEST(BaseTest, BlockPredecessors) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Block* block2 = f.CreateBlock();

  block1->push_back(f.Create<JmpInst>(block2));
  block2->AddPredecessor(block1);
  ASSERT_THAT(block2->predecessors().size(), Eq(1));

  block2->AddPredecessor(block1);
  ASSERT_THAT(block2->predecessors().size(), Eq(1));
}

TEST(BaseTest, Split) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Instruction* inst1 = block1->Create<ConstantInst>(42);
  Instruction* inst2 = block1->Create<ConstantInst>(42);
  Instruction* inst3 = block1->Create<UnreachableInst>();

  Block* block2 = block1->Split(inst2->GetIterator());
  EXPECT_EQ(inst1->parent(), block1);
  EXPECT_EQ(inst2->parent(), block2);
  EXPECT_EQ(inst3->parent(), block2);

  EXPECT_THAT(RefToPtr<Instruction>(block1->instructions()),
              ElementsAre(inst1, block1->GetTerminator()));
  EXPECT_THAT(RefToPtr<Instruction>(block2->instructions()),
              ElementsAre(inst2, inst3));
  EXPECT_THAT(block1->size(), Eq(2));
  EXPECT_THAT(block2->size(), Eq(2));
}

TEST(BaseTest, Splice) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Block* block2 = f.CreateBlock();

  Instruction* inst1 = block1->Create<ConstantInst>(42);
  Instruction* inst2 = block2->Create<ConstantInst>(42);
  Instruction* inst3 = block2->Create<ConstantInst>(42);

  block1->splice(block1->begin(), block2);
  EXPECT_THAT(RefToPtr<Instruction>(block1->instructions()),
              ElementsAre(inst2, inst3, inst1));
  EXPECT_THAT(block1->size(), Eq(3));
  EXPECT_THAT(RefToPtr<Instruction>(block2->instructions()), ElementsAre());
  EXPECT_THAT(block2->size(), Eq(0));
}

TEST(BaseTest, ReplaceUsesOfWith) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst1 = block->Create<ConstantInst>(42);
  Instruction* inst2 = block->Create<ConstantInst>(42);
  Instruction* inst3 =
      block->Create<AddInst>(NumericInst::kInt64, inst1, inst2);

  Instruction* inst4 = block->Create<ConstantInst>(42);
  inst3->ReplaceUsesOfWith(inst2, inst4);
  EXPECT_EQ(inst3->operands()[0], inst1);
  EXPECT_EQ(inst3->operands()[1], inst4);
}

TEST(BaseTest, RemoveBlockArgAtZeroIndex) {
  // Given a block with 3 args.
  Function f("test");
  Block* block = f.CreateBlock();
  block->CreateBlockArgument();
  auto arg1 = block->CreateBlockArgument();
  auto arg2 = block->CreateBlockArgument();

  // When the argument at position zero is removed
  block->RemoveBlockArgumentAt(0);

  // Then the remaining arguments have been shifted down by one index.
  // Note we require pointer stability so check pointers not pointees.
  auto block_args = block->block_arguments();
  auto cblock_args = std::as_const(*block).block_arguments();

  EXPECT_EQ(block->block_arguments_size(), 2);
  EXPECT_EQ(*block_args.begin(), arg1);
  EXPECT_EQ(*cblock_args.begin(), arg1);
  EXPECT_EQ(*std::next(block_args.begin()), arg2);
  EXPECT_EQ(*std::next(cblock_args.begin()), arg2);
}

TEST(BaseTest, RemoveBlockArgAtLaterIndex) {
  // Given a block with 3 args.
  Function f("test");
  Block* block = f.CreateBlock();
  auto arg0 = block->CreateBlockArgument();
  block->CreateBlockArgument();
  auto arg2 = block->CreateBlockArgument();

  // When the argument at position one is removed
  block->RemoveBlockArgumentAt(1);

  // Then the later arguments have been shifted down by one index.
  // Note we require pointer stability so check pointers not pointees.
  auto block_args = block->block_arguments();
  auto cblock_args = std::as_const(*block).block_arguments();

  EXPECT_EQ(block->block_arguments_size(), 2);
  EXPECT_EQ(*block_args.begin(), arg0);
  EXPECT_EQ(*cblock_args.begin(), arg0);
  EXPECT_EQ(*std::next(block_args.begin()), arg2);
  EXPECT_EQ(*std::next(cblock_args.begin()), arg2);
}

TEST(BaseTest, RemoveAllBlockArgs) {
  // Given a block with 3 args.
  Function f("test");
  Block* block = f.CreateBlock();
  block->CreateBlockArgument();
  block->CreateBlockArgument();
  block->CreateBlockArgument();

  // When all arguments are removed
  block->RemoveBlockArgumentAt(2);
  block->RemoveBlockArgumentAt(0);
  block->RemoveBlockArgumentAt(0);

  // The block considers itself to be empty.
  EXPECT_EQ(block->block_arguments_size(), 0);
  EXPECT_TRUE(block->block_arguments_empty());
}
}  // namespace
}  // namespace deepmind::s6
