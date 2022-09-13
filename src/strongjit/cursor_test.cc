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

#include "strongjit/cursor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {
namespace {

TEST(CursorTest, CursorForwardWithinBlock) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst[] = {
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
  };

  Cursor c = f.FirstInstruction();
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetInstruction(), inst[0]);
  c.StepForward();
  ASSERT_FALSE(c.Finished());
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetInstruction(), inst[1]);
  c.StepForward();
  ASSERT_TRUE(c.Finished());
}

TEST(CursorTest, CursorBackwardWithinBlock) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst[] = {
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
  };

  Cursor c = f.LastInstruction();
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetInstruction(), inst[1]);
  c.StepBackward();
  ASSERT_FALSE(c.Finished());
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetInstruction(), inst[0]);
  c.StepBackward();
  ASSERT_TRUE(c.Finished());
}

TEST(CursorTest, CursorAcrossBlocks) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Block* block2 = f.CreateBlock();
  Instruction* inst[] = {
      block1->Create<ConstantInst>(42), block1->Create<ConstantInst>(42),
      block1->Create<JmpInst>(block2),  block2->Create<ConstantInst>(42),
      block2->Create<ConstantInst>(42),
  };
  block2->AddPredecessor(block1);

  Cursor c = f.FirstInstruction();
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetInstruction(), inst[0]);
  for (int i = 0; i < 4; ++i) {
    c.StepForward();
    ASSERT_FALSE(c.Finished());
    ASSERT_EQ(c.GetFunction(), &f);
    ASSERT_EQ(c.GetInstruction(), inst[i + 1]);
  }

  for (int i = 0; i < 4; ++i) {
    c.StepBackward();
    ASSERT_FALSE(c.Finished());
    ASSERT_EQ(c.GetFunction(), &f);
    ASSERT_EQ(c.GetInstruction(), inst[3 - i]);
  }
}

TEST(CursorTest, CursorForwardAcrossDeletion) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst[] = {
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
  };

  Cursor c = f.FirstInstruction();
  c.StepForward();
  inst[1]->erase();
  c.StepForward();
  ASSERT_EQ(c.GetInstruction(), inst[2]);
}

TEST(CursorTest, CursorBackwardAcrossDeletion) {
  Function f("test");
  Block* block = f.CreateBlock();
  Instruction* inst[] = {
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
      block->Create<ConstantInst>(42),
  };

  Cursor c = f.LastInstruction();
  c.StepBackward();
  inst[1]->erase();
  c.StepBackward();
  ASSERT_EQ(c.GetInstruction(), inst[0]);
}

TEST(CursorTest, CursorAcrossSplit) {
  Function f("test");
  Block* block1 = f.CreateBlock();
  Instruction* inst[] = {
      block1->Create<ConstantInst>(42),
      block1->Create<ConstantInst>(42),
  };
  Cursor c = f.FirstInstruction();
  Block* block2 = block1->Split(inst[1]->GetIterator());
  c.StepForward();
  ASSERT_FALSE(c.Finished());
  ASSERT_EQ(c.GetFunction(), &f);
  ASSERT_EQ(c.GetBlock(), block2);
  ASSERT_EQ(c.GetInstruction(), inst[1]);
}

}  // namespace
}  // namespace deepmind::s6
