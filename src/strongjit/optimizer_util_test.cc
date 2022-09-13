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

#include "strongjit/optimizer_util.h"

#include <Python.h>

#include <cstdint>

#include "api.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/formatter.h"
#include "strongjit/parser.h"
#include "strongjit/test_util.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

void InitializePython() {
  static std::once_flag once;
  std::call_once(once, []() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });
}

TEST(OptimizeReplaceAllUsesWithTest, UseListIsUpdated) {
  // Given a strongJIT function
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = constant $3
  %4 = frame_variable names, $0
  %5 = call_python %4 (%1, %2) @4
  %6 = call_python %4 (%5, %2) @4
  return %6
})";

  PyCodeObject code;
  InitializePython();
  code.co_names = PyTuple_New(4);
  PyTuple_SET_ITEM(code.co_names, 0, PyUnicode_InternFromString("add"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));

  auto cursor = f.FirstInstruction();
  auto* constant_1 = cursor.GetInstruction();

  cursor.StepForward();
  auto* constant_2 = cursor.GetInstruction();

  cursor.StepForward();
  auto* constant_3 = cursor.GetInstruction();

  cursor.StepForward();
  auto* py_add = cursor.GetInstruction();

  cursor.StepForward();
  auto* py_call_add_1 = cursor.GetInstruction();

  cursor.StepForward();
  auto* py_call_add_2 = cursor.GetInstruction();

  cursor.StepForward();
  [[maybe_unused]] auto* py_return = cursor.GetInstruction();

  // When we replace all uses of one instruction with another.
  UseLists uses = ComputeUses(f);
  ReplaceAllUsesWith(uses, constant_2, constant_3);

  // Use counts in the rewriter are as expected.
  EXPECT_EQ(uses[constant_1].size(), 1);
  EXPECT_EQ(uses[constant_2].size(), 0);
  EXPECT_EQ(uses[constant_3].size(), 2);
  EXPECT_EQ(uses[py_call_add_1].size(), 1);
  EXPECT_EQ(uses[py_call_add_2].size(), 1);
  EXPECT_EQ(uses[py_add].size(), 2);

  // Updated use counts mirror those in a newly-constructed UseLists.
  UseLists newUses = ComputeUses(f);
  EXPECT_EQ(newUses[constant_1].size(), uses[constant_1].size());
  EXPECT_EQ(newUses[constant_2].size(), uses[constant_2].size());
  EXPECT_EQ(newUses[constant_3].size(), uses[constant_3].size());
  EXPECT_EQ(newUses[py_call_add_1].size(), uses[py_call_add_1].size());
  EXPECT_EQ(newUses[py_call_add_2].size(), uses[py_call_add_2].size());
  EXPECT_EQ(newUses[py_add].size(), uses[py_add].size());
}

TEST(DelayDecrefTest, DelayDecrefSuccess) {
  // Given this strongJIT function, I will try to move the decref in front of
  // the constant $0. To avoid a leak, I expect DelayDecref to also add a decref
  // at the start of &11 and in all intermediate safepoints.

  absl::string_view input = R"(function Simple {
&0: [%1]
  deoptimize_if_safepoint %1, @0, ""
  decref notnull %1 @2
  deoptimize_if_safepoint %1, @4, ""
  br %1 &6, &11

&6:
  deoptimize_if_safepoint %1, @6, ""
  %8 = constant $0
  deoptimize_if_safepoint %1, @8, ""
  return %8

&11:
  deoptimize_if_safepoint %1, @10, ""
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));

  auto cursor = f.FirstInstruction();
  cursor.StepForward();
  DecrefInst* decref = cast<DecrefInst>(cursor.GetInstruction());

  cursor.StepForward();
  cursor.StepForward();
  cursor.StepForward();
  cursor.StepForward();
  ConstantInst* constant = cast<ConstantInst>(cursor.GetInstruction());

  S6_ASSERT_OK(DelayDecref(cast<DecrefInst>(*decref), {constant}));

  absl::string_view output = R"(function Simple {
&0: [ %1 ]                                                  // entry point
  deoptimize_if_safepoint %1, @0, ""
  deoptimize_if_safepoint %1, @4 decrefs [%1], ""
  br %1, &5, &11

&5:                                                         // preds: &0
  deoptimize_if_safepoint %1, @6 decrefs [%1], ""
  decref notnull %1 @2
  %8 = constant $0
  deoptimize_if_safepoint %1, @8, ""
  return %8

&11:                                                        // preds: &0
  decref notnull %1 @2
  deoptimize_if_safepoint %1, @10, ""
  return %1
})";

  ASSERT_EQ(FormatOrDie(f), output);
}

TEST(SplitCriticalEdgesTest, SplitCriticalBlock) {
  absl::string_view input = R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5

&3:
  jmp &5

&5:
  br %1, &7, &9

&7:
  return %1

&9:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  ASSERT_THAT(RawString(FormatOrDie(f)), testing::Eq(RawString(R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5

&3:                                                         // preds: &0
  jmp &7

&5:                                                         // preds: &0
  jmp &7

&7:                                                         // preds: &3, &5
  br %1, &9, &11

&9:                                                         // preds: &7
  return %1

&11:                                                        // preds: &7
  return %1
})")));
}

TEST(SplitCriticalEdgesTest, SplitCriticalEdge) {
  // &7 is not a critical block, but the edge &3->&7 is a critical edge.
  absl::string_view input = R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &7

&3:
  br %1, &7, &5

&5:
  return %1

&7:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  ASSERT_THAT(RawString(FormatOrDie(f)), testing::Eq(RawString(R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &7

&3:                                                         // preds: &0
  br %1, &9, &5

&5:                                                         // preds: &3
  return %1

&7:                                                         // preds: &0
  jmp &11

&9:                                                         // preds: &3
  jmp &11

&11:                                                        // preds: &7, &9
  return %1
})")));
}

TEST(WorklistTest, Worklist) {
  Worklist<int64_t> worklist;

  worklist.Push(4);
  worklist.Push(6);
  worklist.Push(10);
  ASSERT_TRUE(worklist.contains(10));
  ASSERT_EQ(worklist.Pop(), 4);
  worklist.PushIfNew(6);
  worklist.PushIfNew(8);
  worklist.PushIfNew(10);
  ASSERT_TRUE(worklist.contains(8));
  ASSERT_EQ(worklist.Pop(), 6);
  ASSERT_EQ(worklist.Pop(), 10);
  ASSERT_EQ(worklist.Pop(), 8);
  ASSERT_TRUE(worklist.empty());
}

}  // namespace
}  // namespace deepmind::s6
