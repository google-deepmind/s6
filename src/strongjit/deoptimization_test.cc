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

#include "strongjit/deoptimization.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/parser.h"
#include "strongjit/test_util.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

TEST(MarkDeoptimizedBlocksTest, SimpleTrueBranch) {
  absl::string_view input = R"(function SimpleTrueBranch {
&0:
  %1 = constant $5
  br %1, deoptimized &3, &5

&3:
  return %1

&5:
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(MarkDeoptimizedBlocks(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("&3: deoptimized"));
  S6_ASSERT_OK(lm.OnAnyLine("&5:"));
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("deoptimized"), "preds"));
}

TEST(MarkDeoptimizedBlocksTest, SimpleFalseBranch) {
  absl::string_view input = R"(function SimpleFalseBranch {
&0:
  %1 = constant $5
  br %1, &3, deoptimized &5

&3:
  return %1

&5:
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(MarkDeoptimizedBlocks(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("&3:"));
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("deoptimized"), "preds"));
  S6_ASSERT_OK(lm.OnAnyLine("&5: deoptimized"));
}

TEST(MarkDeoptimizedBlocksTest, SameTarget) {
  absl::string_view input = R"(function SameTarget {
&0:
  %1 = constant $5
  br %1, deoptimized &3, &3

&3:
  return %1

&5:
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(MarkDeoptimizedBlocks(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("&3:"));
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("deoptimized"), "preds"));
}

TEST(RewriteFunctionForDeoptimizationTest, Simple) {
  absl::string_view input = R"(function SimpleFalseBranch {
&0:
  %1 = constant $5
  %2 = constant $5
  %3 = constant $5
  %4 = constant $5
  br %1, &6 [ %1 ], deoptimized &10

&6: [ %7 ]
  bytecode_begin @5 stack [%1, %3, %7]
  return %1

&10: deoptimized
  bytecode_begin @10 stack [%2]
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(RewriteFunctionForDeoptimization(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(
      "deoptimize_if not %1, &10, &6 [ %1 ], materializing values [%2]"));
}

TEST(RewriteFunctionForDeoptimizationTest, OnlyDominatingValues) {
  absl::string_view input = R"(function SimpleFalseBranch {
&0:
  %1 = constant $5
  %2 = constant $5
  %3 = constant $5
  %4 = constant $5
  br %1, deoptimized &6 [ %1 ], &11

&6: deoptimized [ %7 ]
  %8 = add i64 %7, %1
  bytecode_begin @5 stack [%3, %7]
  return %8

&11:
  bytecode_begin @10 stack [%2]
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(RewriteFunctionForDeoptimization(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(
      "deoptimize_if %1, &9 [ %1 ], &6, materializing values [%1, %3]"));
}

TEST(RewriteFunctionForDeoptimizationTest, MultipleNextBytecodes) {
  absl::string_view input = R"(function MultipleNextBytecodes {
&0:
  %1 = constant $5
  %2 = constant $5
  %3 = constant $5
  %4 = constant $5
  br %1, deoptimized &6 [ %1 ], &12

&6: deoptimized [ %7 ]
  br %1, &9, &12

&9:
  bytecode_begin @5 stack [%1, %3, %7]
  return %1

&12:
  bytecode_begin @10 stack [%2]
  return %1
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(RewriteFunctionForDeoptimization(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(
      "deoptimize_if %1, &12 [ %1 ], &9, materializing values [%1, %2, %3]"));
}

TEST(RewriteFunctionForDeoptimizationTest, SafepointExtras) {
  absl::string_view input = R"(function SafepointExtras {
&0:
  %1 = constant $5
  %2 = rematerialize s6::RematerializeGetAttr (%1)
  %3 = constant $5
  %4 = constant $5
  deoptimize_if_safepoint %3, @5 stack [%2, %3], "reason"
  return %4
})";

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(RewriteFunctionForDeoptimization(f));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(
      "deoptimize_if_safepoint %3, @5 stack [%2, %3] extras [%1]"));
}
}  // namespace

}  // namespace deepmind::s6
