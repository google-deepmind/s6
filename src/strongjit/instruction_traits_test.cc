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

#include "strongjit/instruction_traits.h"

#include <cstddef>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/function.h"

namespace deepmind::s6 {
namespace {

TEST(InstructionTraitsTest, InstructionList) {
  Function f("test");
  ConstantInst* constant_inst = f.Create<ConstantInst>(42);
  CallNativeInst* call_native_inst = f.Create<CallNativeInst>(
      Callee::kPyObject_GetAttr, absl::Span<Value* const>{});
  ASSERT_FALSE(InstructionTraits::ClobbersAllRegisters(*constant_inst));
  ASSERT_TRUE(InstructionTraits::ClobbersAllRegisters(*call_native_inst));
}

TEST(InstructionTraitsTest, InstructionAttr) {
  Function f("test");
  IncrefInst* incref_inst = f.Create<IncrefInst>();
  ConstantInst* constant_inst = f.Create<ConstantInst>(42);
  ASSERT_FALSE(InstructionTraits::ProducesValue(*incref_inst));
  ASSERT_TRUE(InstructionTraits::ProducesValue(*constant_inst));
}

TEST(InstructionTraitsTest, InstructionInheritance) {
  Function f("test");
  IncrefInst* incref_inst = f.Create<IncrefInst>();
  ASSERT_FALSE(ValueTraits::IsA<BrInst>(*incref_inst));
  ASSERT_TRUE(ValueTraits::IsA<RefcountInst>(*incref_inst));
}

TEST(InstructionTraitsTest, Block) {
  Function f("test");
  Block* block = f.CreateBlock();
  ASSERT_TRUE(ValueTraits::IsA<Block>(*block));
}

}  // namespace
}  // namespace deepmind::s6
