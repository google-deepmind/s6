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

#include "strongjit/instructions.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/function.h"

namespace deepmind::s6 {
namespace {
using testing::Eq;
using testing::IsNull;
using testing::IsTrue;

TEST(InstructionsTest, CanAllocateInstructions) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst = f.Create<ConstantInst>(42);
  ASSERT_THAT(inst->parent(), IsNull());

  block->push_back(inst);
  ASSERT_THAT(inst->parent(), Eq(block));
  ASSERT_THAT(block->size(), Eq(1));

  inst->RemoveFromParent();
  ASSERT_THAT(inst->parent(), IsNull());
  ASSERT_THAT(block->size(), Eq(0));
  ASSERT_THAT(block->empty(), IsTrue());
}

TEST(InstructionsTest, CanAllocateRealInstructions) {
  Function f("test");
  Instruction* inst = f.Create<ConstantInst>(42);
  ASSERT_EQ(inst->kind(), Value::kConstant);
}

TEST(InstructionsTest, LeftIsLeftAndRightIsRight) {
  Function f("test");
  Block* block = f.CreateBlock();

  Instruction* inst1 = block->Create<ConstantInst>(42);
  Instruction* inst2 = block->Create<ConstantInst>(42);
  BinaryInst* inst = block->Create<AddInst>(NumericInst::kInt64, inst1, inst2);
  EXPECT_EQ(inst->lhs(), inst1);
  EXPECT_EQ(inst->rhs(), inst2);
}

TEST(InstructionsTest, BlockArgumentIsNotAnInstruction) {
  Function f("test");
  Block* block = f.CreateBlock();
  BlockArgument* a = block->CreateBlockArgument();

  EXPECT_TRUE(isa<BlockArgument>(a));
  EXPECT_FALSE(isa<Instruction>(a));
}

TEST(InstructionsTest, RemoveSuccessorArgumentAtIndexForJmp) {
  // Given a block with a jump teminator.
  Function f("test");
  Block* pred = f.CreateBlock();
  Block* successor = f.CreateBlock();

  std::vector<Value*> args;
  args.push_back(pred->Create<ConstantInst>(0));
  args.push_back(pred->Create<ConstantInst>(1));
  args.push_back(pred->Create<ConstantInst>(2));
  JmpInst* jmp = pred->Create<JmpInst>(successor, args);

  // When an argument is removed from the terminator.
  jmp->RemoveSuccessorArgumentAt(0, 1);

  // Then the argument count and arguments are as expected.
  EXPECT_EQ(jmp->successor_arguments(0).size(), 2);
  EXPECT_EQ(cast<ConstantInst>(jmp->successor_arguments(0)[0])->value(), 0);
  EXPECT_EQ(cast<ConstantInst>(jmp->successor_arguments(0)[1])->value(), 2);
}

TEST(InstructionsTest, RemoveSuccessorArgumentAtIndexForBranch) {
  // Given a block with a jump teminator.
  Function f("test");
  Block* pred = f.CreateBlock();
  Block* true_successor = f.CreateBlock();
  Block* false_successor = f.CreateBlock();

  auto condition = pred->Create<ConstantInst>(0);

  std::vector<Value*> true_args;
  true_args.push_back(pred->Create<ConstantInst>(0));
  true_args.push_back(pred->Create<ConstantInst>(1));
  true_args.push_back(pred->Create<ConstantInst>(2));

  std::vector<Value*> false_args;
  false_args.push_back(pred->Create<ConstantInst>(0));
  false_args.push_back(pred->Create<ConstantInst>(-1));
  false_args.push_back(pred->Create<ConstantInst>(-2));
  BrInst* br = pred->Create<BrInst>(condition, true_successor, true_args,
                                    false_successor, false_args);

  // When an argument is removed from the terminator.
  br->RemoveSuccessorArgumentAt(0, 1);
  br->RemoveSuccessorArgumentAt(1, 0);

  // Then the argument count and arguments are as expected.
  EXPECT_EQ(br->successor_arguments(0).size(), 2);
  EXPECT_EQ(cast<ConstantInst>(br->successor_arguments(0)[0])->value(), 0);
  EXPECT_EQ(cast<ConstantInst>(br->successor_arguments(0)[1])->value(), 2);

  EXPECT_EQ(br->successor_arguments(1).size(), 2);
  EXPECT_EQ(cast<ConstantInst>(br->successor_arguments(1)[0])->value(), -1);
  EXPECT_EQ(cast<ConstantInst>(br->successor_arguments(1)[1])->value(), -2);
}

TEST(InlinerTest, CanCloneConstantInst) {
  // Given an instruction.
  Function f("test");
  Instruction* inst = f.Create<ConstantInst>(42);

  // When it is cloned.
  Instruction* clone = inst->CloneWithNewOperands(f, {});

  // Then member data is correctly copied.
  ASSERT_EQ(cast<ConstantInst>(inst)->value(),
            cast<ConstantInst>(clone)->value());
}

TEST(InlinerTest, ClonedInstructionOperandsAreUpdatedIfMapped) {
  // Given an instruction that uses values.
  Function f("test");
  Instruction* one = f.Create<ConstantInst>(1);
  Instruction* two = f.Create<ConstantInst>(2);
  Instruction* inst = f.Create<AddInst>(AddInst::kInt64, one, two);

  // When it is cloned with a non-empty mapping.
  Instruction* zero = f.Create<ConstantInst>(0);
  absl::flat_hash_map<const Value*, Value*> mapping;
  mapping[one] = zero;

  Instruction* clone = inst->CloneWithNewOperands(f, mapping);

  // Then the mapped instructions are replaced and unmapped ones are left alone.
  ASSERT_EQ(clone->operands()[0], zero);
  ASSERT_EQ(clone->operands()[1], two);
}
}  // namespace
}  // namespace deepmind::s6
