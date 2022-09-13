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

#include "strongjit/value_traits.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"

namespace deepmind::s6 {
namespace {

template <typename BaseClass>
struct IsATest {
  template <typename DerivedClass>
  static std::optional<bool> Visit() {
    // Hide ASSERT_EQ return
    []() {
      constexpr bool is_base_class =
          std::is_base_of<BaseClass, DerivedClass>::value;
      DerivedClass fake;
      ASSERT_EQ(is_base_class, isa<BaseClass>(fake))
          << absl::StrCat("Failed at ", DerivedClass::kMnemonic);
    }();
    return {};
  }
  static bool Default() { return false; }
};

template <typename BaseClass>
void TestAllInstructions() {
  ForAllInstructionKinds<IsATest<BaseClass>>();
}

#define TEST_CLASS(CLASS) \
  TEST(ValueTraits, CLASS) { TestAllInstructions<CLASS>(); }

TEST_CLASS(Instruction)
TEST_CLASS(CallPythonInst)
TEST_CLASS(NumericInst)
TEST_CLASS(BinaryInst)
TEST_CLASS(UnaryInst)
TEST_CLASS(RefcountInst)
TEST_CLASS(SafepointInst)
TEST_CLASS(TerminatorInst)
TEST_CLASS(ConditionalTerminatorInst)
TEST_CLASS(UnconditionalTerminatorInst)

}  // namespace
}  // namespace deepmind::s6
