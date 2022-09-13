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

#include "code_generation/trace_register_allocator.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/parser.h"
#include "strongjit/test_util.h"
#include "utils/matchers.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {
using ::deepmind::s6::matchers::StatusIs;
using testing::ElementsAre;
using testing::Eq;
using testing::UnorderedElementsAre;

MATCHER_P(ValueIs, value, "") { return FormatOrDie(*arg) == value; }

TEST(LivenessAnalysisTest, LivenessAnalysis) {
  // Constructs a loop with an if-else.
  absl::string_view input = R"(function f {
&0:                                                         // entry point
  %1 = constant $42 // live throughout, unused in the loop.
  %2 = constant $32
  jmp &4 [ %2 ]

&4: [ %5 ]
  %6 = constant $42
  br %1, &8, &11

&8:
  %9 = constant $23
  jmp &14 [ %9 ]

&11:
  %12 = add i64 %6, %5
  jmp &14 [ %12 ]

&14: [ %15 ]
  br %15, &19, &17 // maybe-loop-exit

&17:
  jmp &4 [ %15 ]

&19:
  %20 = add i64 %1, %15
  return %20
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  DominatorTree domtree = ConstructDominatorTree(f);
  Liveness liveness = AnalyzeLiveness(f, domtree);

  for (const Block& b : f) {
    S6_LOG(INFO) << FormatOrDie(b) << ": "
                 << absl::StrJoin(liveness.live_ins(&b), ", ",
                                  [&](std::string* s, const Value* v) {
                                    absl::StrAppend(s, FormatOrDie(*v));
                                  });
  }

  auto block_it = f.begin();
  EXPECT_THAT(liveness.live_ins(&*block_it++), UnorderedElementsAre());
  EXPECT_THAT(liveness.live_ins(&*block_it++),
              UnorderedElementsAre(ValueIs("%1")));
  EXPECT_THAT(liveness.live_ins(&*block_it++),
              UnorderedElementsAre(ValueIs("%1")));
  EXPECT_THAT(
      liveness.live_ins(&*block_it++),
      UnorderedElementsAre(ValueIs("%1"), ValueIs("%5"), ValueIs("%6")));
  EXPECT_THAT(liveness.live_ins(&*block_it++),
              UnorderedElementsAre(ValueIs("%1")));
  EXPECT_THAT(liveness.live_ins(&*block_it++),
              UnorderedElementsAre(ValueIs("%1"), ValueIs("%15")));
  EXPECT_THAT(liveness.live_ins(&*block_it++),
              UnorderedElementsAre(ValueIs("%1"), ValueIs("%15")));
}

TEST(BlockFrequencyTest, HeuristicallyDetermineBlockFrequencies) {
  absl::string_view input = R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $32
  jmp &4 [ %2 ]

&4: [ %5 ]
  %6 = constant $42
  br %1, &8, &11

&8:
  %9 = constant $23
  except @0

&11:
  %12 = add i64 %6, %5
  jmp &14 [ %12 ]

&14: [ %15 ]
  br %15, &19, &17

&17:
  jmp &4 [ %15 ]

&19:
  %20 = add i64 %1, %15
  return %20
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  DominatorTree domtree = ConstructDominatorTree(f);
  absl::flat_hash_map<const Block*, int64_t> frequencies =
      HeuristicallyDetermineBlockFrequencies(f, domtree);

  for (const Block& b : f) {
    S6_LOG(INFO) << FormatOrDie(b) << ": " << frequencies[&b];
  }

  auto block_it = f.begin();
  EXPECT_THAT(frequencies[&*block_it++], Eq(1));
  EXPECT_THAT(frequencies[&*block_it++], Eq(2));
  EXPECT_THAT(frequencies[&*block_it++], Eq(0));
  EXPECT_THAT(frequencies[&*block_it++], Eq(2));
  EXPECT_THAT(frequencies[&*block_it++], Eq(2));
  EXPECT_THAT(frequencies[&*block_it++], Eq(2));
  EXPECT_THAT(frequencies[&*block_it++], Eq(1));
}

TEST(AllocateTracesTest, AllocateTraces) {
  absl::string_view input = R"(function f {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $32
  jmp &4 [ %2 ]

&4: [ %5 ]
  %6 = constant $42
  br %1, &8, &11

&8:
  %9 = constant $23
  except @0

&11:
  %12 = add i64 %6, %5
  jmp &14 [ %12 ]

&14: [ %15 ]
  br %15, &19, &17

&17:
  jmp &4 [ %15 ]

&19:
  %20 = add i64 %1, %15
  return %20
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  DominatorTree domtree = ConstructDominatorTree(f);
  absl::flat_hash_map<const Block*, int64_t> frequencies =
      HeuristicallyDetermineBlockFrequencies(f, domtree);
  std::vector<Trace> traces = AllocateTraces(f, frequencies);
  ASSERT_EQ(traces.size(), 3);
  EXPECT_THAT(traces[0],
              ElementsAre(ValueIs("&0"), ValueIs("&4"), ValueIs("&11"),
                          ValueIs("&14"), ValueIs("&17")));
  EXPECT_THAT(traces[1], ElementsAre(ValueIs("&19")));
  EXPECT_THAT(traces[2], ElementsAre(ValueIs("&8")));
}

using testing::Eq;

// Command the register allocator not to convert constants into immediates.
const RegisterAllocationOptions kNoImmediates = {kAllocatableRegisters, -1};

TEST(RegisterAllocationTest, Simple) {
  absl::string_view input = R"(function Simple {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Simple {
&0:
  %1 = constant $42                                         // def(rax)
  return %1                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, TwoLiveIntervals) {
  absl::string_view input = R"(function TwoLiveIntervals {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $10
  %3 = add i64 %2, %1
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function TwoLiveIntervals {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $10                                         // def(rcx)
  %3 = add i64 %2, %1                                       // def(rax) use(rcx, rax)
  return %3                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, BlockArgument) {
  absl::string_view input = R"(function BlockArgument {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3 [ %1 ]

&3: [ %4 ]
  %5 = constant $10
  %6 = add i64 %4, %1
  return %6
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function BlockArgument {
&0:
  %1 = constant $42                                         // def(rax)
  jmp &3 [ %1 ]                                             // use(rax)
                                                            // copies(rax -> stack[0])

&3: [ %4 ]                                                  // def(stack[0])
  %5 = constant $10                                         // def(rcx)
  %6 = add i64 %4, %1                                       // def(rax) use(rcx, rax) copies(stack[0] -> rcx)
  return %6                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, ReuseRegister) {
  absl::string_view input = R"(function ReuseRegister {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $10
  %3 = add i64 %2, %1
  %4 = constant $50
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function ReuseRegister {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $10                                         // def(rcx)
  %3 = add i64 %2, %1                                       // def(rax) use(rcx, rax)
  %4 = constant $50                                         // def(rcx)
  return %3                                                 // use(rax)
})")));
}

// Tests that register allocation fails when not enough registers are
// available.
// TODO: This is a bad test. It originally forced operands to be in
// memory,
TEST(RegisterAllocationTest, Spill) {
  absl::string_view input = R"(function Spill {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $10
  %3 = add i64 %2, %1
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  ASSERT_THAT(AllocateRegistersWithTrace(f, {{asmjit::x86::rbx}, -1}),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

// Tests that block arguments are allocated sanely.
TEST(RegisterAllocationTest, BlockArguments) {
  absl::string_view input = R"(function BlockArguments {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3 [ %1 ]

&3: [ %4 ]
  %5 = constant $43
  %6 = add i64 %5, %4
  jmp &8 [ %6 ]

&8: [ %9 ]
  %10 = add i64 %9, %9
  return %10
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function BlockArguments {
&0:
  %1 = constant $42                                         // def(rax)
  jmp &3 [ %1 ]                                             // use(rax)

&3: [ %4 ]                                                  // def(rax)
  %5 = constant $43                                         // def(rcx)
  %6 = add i64 %5, %4                                       // def(rax) use(rcx, rax)
  jmp &8 [ %6 ]                                             // use(rax)

&8: [ %9 ]                                                  // def(rax)
  %10 = add i64 %9, %9                                      // def(rax) use(rax, rax)
  return %10                                                // use(rax)
})")));
}

// Tests that block arguments are allocated correctly by deliberately passing
// conflicting values to block arguments.
//
// The second jump has to give the exact inverse arguments to the first branch,
// which also means naive copies(rax -> rcx, rcx -> rax) would clobber each
// other. The copies need to use a temporary.
TEST(RegisterAllocationTest, BlockArguments2) {
  absl::string_view input = R"(function BlockArguments2 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  br %1, &4, &6 [ %1, %2 ]

&4:
  jmp &6 [ %2, %1 ]

&6: [ %7, %8 ]
  %9 = add i64 %7, %8
  return %9
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function BlockArguments2 {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $42                                         // def(rcx)
  br %1, &4, &6                                             // use(rax)

&4:
  jmp &8 [ %2, %1 ]                                         // use(rcx, rax)
                                                            // copies(rax -> stack[0], rcx -> rax, stack[0] -> rcx)

&6:
  jmp &8 [ %1, %2 ]                                         // use(rax, rcx)

&8: [ %9, %10 ]                                             // def(rax) def(rcx)
  %11 = add i64 %9, %10                                     // def(rax) use(rax, rcx)
  return %11                                                // use(rax)
})")));
}

TEST(RegisterAllocationTest, BlockArguments3) {
  absl::string_view input = R"(function BlockArguments3 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  br %1, &4 [ %2, %1 ], &8 [ %1, %2 ]

&4: [ %5, %6 ]
  jmp &8 [ %5, %6 ]

&8: [ %9, %10 ]
  %11 = add i64 %9, %10
  return %11
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function BlockArguments3 {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $42                                         // def(rcx)
  br %1, &4 [ %2, %1 ], &8                                  // use(rax, rcx, rax)

&4: [ %5, %6 ]                                              // def(rcx) def(rax)
  jmp &10 [ %5, %6 ]                                        // use(rcx, rax)
                                                            // copies(rax -> stack[0], rcx -> rax, stack[0] -> rcx)

&8:
  jmp &10 [ %1, %2 ]                                        // use(rax, rcx)

&10: [ %11, %12 ]                                           // def(rax) def(rcx)
  %13 = add i64 %11, %12                                    // def(rax) use(rax, rcx)
  return %13                                                // use(rax)
})")));
}

TEST(RegisterAllocationTest, Call) {
  absl::string_view input = R"(function Call {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%1) @2
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Call {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  %3 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi)
  return %3                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, Call2) {
  absl::string_view input = R"(function Call2 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%1) @2
  %4 = call_native PyObject_GetAttr (%2) @2
  return %4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Call2 {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  %3 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(rax -> stack[0])
  %4 = call_native PyObject_GetAttr (%2) @2                 // def(rax) use(rdi) copies(stack[0] -> rdi)
  return %4                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, Call3) {
  absl::string_view input = R"(function Call3 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%2) @2
  %4 = call_native PyObject_GetAttr (%1) @2
  return %4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Call3 {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $43                                         // def(rdi)
  %3 = call_native PyObject_GetAttr (%2) @2                 // def(rax) use(rdi) copies(rax -> stack[0])
  %4 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(stack[0] -> rdi)
  return %4                                                 // use(rax)
})")));
}

// Small constants can be allocated to immediates. These don't need to be
// spilled.
TEST(RegisterAllocationTest, Immediates) {
  absl::string_view input = R"(function Immediates {
&0:                                                         // entry point
  %1 = constant $42
  %2 = call_native PyObject_GetAttr @0
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Immediates {
&0:
  %1 = constant $42                                         // def(imm)
  %2 = call_native PyObject_GetAttr @0                      // def(rax)
  return %1                                                 // use(imm)
})")));
}

TEST(RegisterAllocationTest, LoopWithDominatingValue) {
  absl::string_view input = R"(function LoopWithDominatingValue {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &10

&3:
  %4 = add i64 %1, %1
  %5 = constant $54
  jmp &7

&7:
  %8 = constant $23
  jmp &3

&10:
  %11 = constant $77
  return %11
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function LoopWithDominatingValue {
&0:
  %1 = constant $42                                         // def(rax)
  br %1, &3, &12                                            // use(rax)

&3:
  jmp &5

&5:
  %6 = add i64 %1, %1                                       // def(rcx) use(rax, rax)
  %7 = constant $54                                         // def(rcx)
  jmp &9

&9:
  %10 = constant $23                                        // def(rcx)
  jmp &5

&12:
  %13 = constant $77                                        // def(rax)
  return %13                                                // use(rax)
})")));
}

// %1 is used only in the first and last blocks. It is dormant inside the loop
// so rax should be reallocated.
TEST(RegisterAllocationTest, LiveRangeHole) {
  absl::string_view input = R"(function LiveRangeHole {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &10

&3:
  %4 = constant $54
  %5 = constant $53
  jmp &7

&7:
  %8 = constant $23
  jmp &3

&10:
  %11 = add i64 %1, %1
  return %11
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function LiveRangeHole {
&0:
  %1 = constant $42                                         // def(rax)
  br %1, &3, &12                                            // use(rax)

&3:
  jmp &5

&5:
  %6 = constant $54                                         // def(rax)
  %7 = constant $53                                         // def(rax)
  jmp &9

&9:
  %10 = constant $23                                        // def(rax)
  jmp &5

&12:
  %13 = add i64 %1, %1                                      // def(rax) use(rax, rax)
  return %13                                                // use(rax)
})")));
}

TEST(RegisterAllocationTest, SpillAroundCall) {
  absl::string_view input = R"(function SpillAroundCall {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%1) @2
  %4 = add i64 %1, %2
  return %4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function SpillAroundCall {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  %3 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(rax -> stack[0], rdi -> stack[1])
  %4 = add i64 %1, %2                                       // def(rax) use(rax, rcx) copies(stack[1] -> rax, stack[0] -> rcx)
  return %4                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, SpillAroundCall2) {
  absl::string_view input = R"(function SpillAroundCall2 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  br %1, &7, &4

&4:
  %5 = call_native PyObject_GetAttr (%1) @2
  jmp &7

&7:
  %8 = add i64 %1, %2
  return %8
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function SpillAroundCall2 {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  br %1, &7, &4                                             // use(rdi)

&4:
  %5 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(rax -> stack[0], rdi -> stack[1])
  jmp &9

&7:
  jmp &9                                                    // copies(rax -> stack[0], rdi -> stack[1])

&9:
  %10 = add i64 %1, %2                                      // def(rax) use(rax, rcx) copies(stack[1] -> rax, stack[0] -> rcx)
  return %10                                                // use(rax)
})")));
}

TEST(RegisterAllocationTest, SpillAroundCallWithStackSlots) {
  absl::string_view input = R"(function SpillAroundCallWithStackSlots {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%2) @2
  %4 = add i64 %3, %1
  return %4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RegisterAllocation> ra,
      AllocateRegistersWithTrace(f, {{asmjit::x86::rax}, -1}));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function SpillAroundCallWithStackSlots {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $43                                         // def(rax) copies(rax -> stack[0])
  %3 = call_native PyObject_GetAttr (%2) @2                 // def(rax) use(rdi) copies(rax -> rdi)
  %4 = add i64 %3, %1                                       // def(rax) use(rax, rdi) copies(stack[0] -> rdi)
  return %4                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, AllocateAllRanges) {
  // This function is large but is taken from a real test that failed.
  // Distilling it has proved unsuccessful :) The behaviour is that, at the end
  // of the function, the unhandled set is empty but there are still zombie
  // ranges waiting to become active. We previously stopped allocating when the
  // unhandled set became empty, so we ended up with undef ranges.
  absl::string_view input = R"(function has_location {
&0:
  %1 = frame_variable fastlocals, $0                        // def(rax)
  %2 = load i64 [ %1 + $0 ]                                 // def(rax) use(rax)
  %3 = constant $0                                          // def(imm)
  %4 = cmp eq i64 %2, %3                                    // def(rcx) use(rax, imm)
  br %4, &6, &27                                            // use(rcx)

&6:
  %7 = frame_variable code_object, $0                       // def(rax)
  %8 = load i64 [ %7 + $80 ]                                // def(rax) use(rax)
  %9 = constant $94804114869240                             // def(rcx)
  %10 = load i64 [ %8 + $40 ]                               // def(rax) use(rax)
  %11 = constant $0                                         // def(imm)
  %12 = cmp ne i64 %10, %11                                 // def(rdx) use(rax, imm)
  br %12, &14, &25                                          // use(rdx)

&14:
  %15 = call_native PyUnicode_AsUTF8 (%10) @0               // def(rax) use(rax) copies(rcx -> stack[0])
  %16 = constant $0                                         // def(imm) copies(stack[0] -> rcx)
  %17 = cmp ne i64 %15, %16                                 // def(rdx) use(rax, imm)
  br %17, &19, &23                                          // use(rdx)

&19:
  %20 = constant $94804030587977                            // def(rdx)
  %21 = call_native PyErr_Format (%9, %20, %15) @0          // def(rax) use(rcx, rdx, rax)
  jmp &23

&23:
  jmp &25

&25:
  except @0

&27:
  incref notnull %2                                         // use(rax)
  jmp &30

&30:
  %31 = frame_variable names, $4                            // def(rcx)
  %32 = load i64 [ %31 + $40 ]                              // def(rcx) use(rcx)
  %33 = constant $0                                         // def(imm)
  %34 = call_native PyObject_GetAttr (%2, %32, %33) @2      // def(rcx) use(stack[0], rcx, imm) copies(rax -> stack[0])
  decref notnull %2 @2                                      // use(rdx) copies(stack[0] -> rdx)
  %36 = constant $0                                         // def(imm)
  %37 = cmp eq i64 %34, %36                                 // def(rdx) use(rcx, imm)
  br %37, &39, &41                                          // use(rdx)

&39:
  except @2

&41:
  jmp &43

&43:
  return %34                                                // use(undef)

&45:
  unreachable
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f));
  LineMatcher lm(ra->ToString(f));
  // No undefs pls.
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("undef"), "unreachable"));
}

TEST(RegisterAllocationTest, ForceAbiCompliance) {
  absl::string_view input = R"(function ForceAbiCompliance {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  br %1, &4, &7

&4:
  %5 = call_native PyObject_GetAttr (%1, %2) @2
  jmp &10

&7:
  %8 = call_native PyObject_GetAttr (%2, %1) @2
  jmp &10

&10:
  return %2
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function ForceAbiCompliance {
&0:
  %1 = constant $42                                         // def(rsi)
  %2 = constant $43                                         // def(rdi)
  br %1, &4, &7                                             // use(rsi)

&4:
  %5 = call_native PyObject_GetAttr (%1, %2) @2             // def(rax) use(rdi, rsi) copies(rdi -> stack[0], rsi -> rdi, stack[0] -> rsi)
  jmp &10

&7:
  %8 = call_native PyObject_GetAttr (%2, %1) @2             // def(rax) use(rdi, rsi) copies(rdi -> stack[0])
  jmp &10

&10:
  return %2                                                 // use(rax) copies(stack[0] -> rax)
})")));
}

TEST(RegisterAllocationTest, Deoptimize) {
  absl::string_view input = R"(function Deoptimize {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  deoptimize_if %1, &7 [ %1 ], &4 materializing values [ %2 ]

&4:
  %5 = call_native PyObject_GetAttr (%1) @2
  return %5

&7: deoptimized [ %8 ]
  %9 = add i64 %1, %8
  jmp &4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Deoptimize {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  deoptimize_if %1, &9 [ %1 ], &4, materializing values [%2] // use(rdi, rdi, rax)

&4:
  jmp &6

&6:
  %7 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi)
  return %7                                                 // use(rax)

&9: deoptimized [ %10 ]
  %11 = add i64 %1, %10
  jmp &6
})")));
  const DeoptimizeIfInst* di = nullptr;
  for (const Block& b : f) {
    for (const Instruction& i : b) {
      if (!di) di = dyn_cast<DeoptimizeIfInst>(&i);
    }
  }
}

TEST(RegisterAllocationTest, MaterializeValuesNotInRegister) {
  absl::string_view input = R"(function Deoptimize {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $32
  %3 = call_native PyObject_GetAttr (%1) @2
  deoptimize_if %1, &8 [ %1 ], &5 materializing values [ %2 ]

&5:
  %6 = call_native PyObject_GetAttr (%1) @2
  return %6

&8: deoptimized [ %9 ]
  %10 = add i64 %1, %9
  jmp &5
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Deoptimize {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $32                                         // def(rax)
  %3 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(rax -> stack[0], rdi -> stack[1])
  deoptimize_if %1, &10 [ %1 ], &5, materializing values [%2] // use(rdi, rdi, stack[0]) copies(stack[1] -> rdi)

&5:
  jmp &7

&7:
  %8 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi)
  return %8                                                 // use(rax)

&10: deoptimized [ %11 ]
  %12 = add i64 %1, %11
  jmp &7
})")));
}

TEST(RegisterAllocationTest, MaterializeValuesUsedLater) {
  absl::string_view input = R"(function Deoptimize {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $32
  %3 = call_native PyObject_GetAttr (%1) @2
  deoptimize_if %1, &8 [ %1 ], &5 materializing values [ %2 ]

&5:
  %6 = call_native PyObject_GetAttr (%1, %2) @2
  return %6

&8: deoptimized [ %9 ]
  %10 = add i64 %1, %9
  jmp &5
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Deoptimize {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $32                                         // def(rax)
  %3 = call_native PyObject_GetAttr (%1) @2                 // def(rax) use(rdi) copies(rax -> stack[0], rdi -> stack[1])
  deoptimize_if %1, &10 [ %1 ], &5, materializing values [%2] // use(rdi, rdi, stack[0]) copies(stack[1] -> rdi)

&5:
  jmp &7

&7:
  %8 = call_native PyObject_GetAttr (%1, %2) @2             // def(rax) use(rdi, rsi) copies(stack[0] -> rsi)
  return %8                                                 // use(rax)

&10: deoptimized [ %11 ]
  %12 = add i64 %1, %11
  jmp &7
})")));
}

TEST(RegisterAllocationTest, ChainedBlockArguments) {
  // The block argument %9 is used as rdi; chaining means %1 should be assigned
  // rdi.
  absl::string_view input = R"(function ChainedBlockArguments {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  br %1, &4 [ %2, %1 ], &8 [ %1, %2 ]

&4: [ %5, %6 ]
  jmp &8 [ %5, %6 ]

&8: [ %9, %10 ]
  decref notnull %9 @0
  %12 = add i64 %9, %10
  return %12
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function ChainedBlockArguments {
&0:
  %1 = constant $42                                         // def(rax)
  %2 = constant $42                                         // def(rcx)
  br %1, &4 [ %2, %1 ], &8                                  // use(rax, rcx, rax)

&4: [ %5, %6 ]                                              // def(rcx) def(rax)
  jmp &10 [ %5, %6 ]                                        // use(rcx, rax)
                                                            // copies(rax -> stack[2], rcx -> rax, stack[2] -> rcx)

&8:
  jmp &10 [ %1, %2 ]                                        // use(rax, rcx)

&10: [ %11, %12 ]                                           // def(rax) def(rcx)
  decref notnull %11 @0                                     // use(rdi) copies(rax -> rdi, rcx -> stack[0], rdi -> stack[1])
  %14 = add i64 %11, %12                                    // def(rax) use(rax, rcx) copies(stack[1] -> rax, stack[0] -> rcx)
  return %14                                                // use(rax)
})")));
}

TEST(RegisterAllocationTest, DontBotherMaterializingValueStackToRegisters) {
  absl::string_view input = R"(function Deoptimize {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = constant $44
  %4 = call_native PyObject_GetAttr(%1) @6
  %5 = yield_value %2 @42 stack [ %1, %3 ]
  return %5
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  const YieldValueInst* yi = nullptr;
  for (const Block& b : f) {
    for (const Instruction& i : b) {
      if (!yi) yi = dyn_cast<YieldValueInst>(&i);
    }
  }
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function Deoptimize {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rax)
  %3 = constant $44                                         // def(rcx)
  %4 = call_native PyObject_GetAttr (%1) @6                 // def(rax) use(rdi) copies(rax -> stack[0], rcx -> stack[1], rdi -> stack[2])
  %5 = yield_value %2 @42 stack [%1, %3]                    // def(rax) use(rax, stack[2], stack[1]) copies(stack[0] -> rax)
  return %5                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, AbiCallStackSlots1) {
  absl::string_view input = R"(
function AbiCallStackSlots {
&0:                                                         // entry point
  %1 = constant $1
  %2 = constant $1
  %3 = constant $1
  %4 = constant $1
  %5 = call_python %1 (%2, %3, %4) @0
  return %5
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  SplitCriticalEdges(f);
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function AbiCallStackSlots {
&0:
  %1 = constant $1                                          // def(rax)
  %2 = constant $1                                          // def(rcx)
  %3 = constant $1                                          // def(rdx)
  %4 = constant $1                                          // def(rsi)
  %5 = call_python %1 (%2, %3, %4) @0                       // def(rax) use(callstack[0], callstack[1], callstack[2], callstack[3]) copies(rax -> callstack[0], rcx -> callstack[1], rdx -> callstack[2], rsi -> callstack[3])
  return %5                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, AbiCallStackSlots2) {
  absl::string_view input = R"(
function AbiCallStackSlots {
&0:                                                         // entry point
  %1 = constant $1
  %2 = constant $1
  %3 = constant $1
  %4 = constant $1
  %5 = constant $1
  %6 = constant $1
  %7 = constant $1
  %8 = constant $1
  %9 = constant $1
  %10 = call_native PyObject_GetAttr (%2, %3, %4, %5, %6, %7, %8, %9) @0
  return %10
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));

  LineMatcher lm(ra->ToString(f));
  S6_ASSERT_OK(lm.OnAnyLine(
      "def(rax) use(rdi, rsi, rdx, rcx, r8, r9, callstack[0], callstack[1]) "
      "copies(rax -> callstack[0], r10 -> callstack[1])"));
}

TEST(RegisterAllocationTest, RematerializeInst) {
  absl::string_view input = R"(function RematerializeInst {
&0:                                                         // entry point
  %1 = constant $2
  %2 = constant $32
  %3 = rematerialize PyObject_GetAttr (%1)
  %4 = constant $5
  deoptimize_if_safepoint %2, @6 stack [%3], ""
  %6 = add i64 %2, %4
  return %6
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)),
              Eq(RawString(R"(function RematerializeInst {
&0:
  %1 = constant $2                                          // def(rax)
  %2 = constant $32                                         // def(rcx)
  %3 = rematerialize PyObject_GetAttr (%1)                  // def(imm) use(rax)
  %4 = constant $5                                          // def(rax)
  deoptimize_if_safepoint %2, @6 stack [%3], ""             // use(rcx, imm)
  %6 = add i64 %2, %4                                       // def(rax) use(rcx, rax)
  return %6                                                 // use(rax)
})")));
}

TEST(RegisterAllocationTest, PreSpilled) {
  absl::string_view input = R"(function PreSpilled {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  %3 = call_native PyObject_GetAttr (%1, %2) @2
  br %3, &5, &8

&5:
  %6 = call_native PyObject_GetAttr (%1, %2) @2
  jmp &11

&8:
  %9 = call_native PyObject_GetAttr (%2, %1) @2
  jmp &11

&11:
  %12 = add i64 %1, %2
  return %12
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                          AllocateRegistersWithTrace(f, kNoImmediates));
  ASSERT_THAT(RawString(ra->ToString(f)), Eq(RawString(R"(function PreSpilled {
&0:
  %1 = constant $42                                         // def(rdi)
  %2 = constant $43                                         // def(rsi)
  %3 = call_native PyObject_GetAttr (%1, %2) @2             // def(rax) use(rdi, rsi) copies(rsi -> stack[0], rdi -> stack[1])
  br %3, &5, &8                                             // use(rax)

&5:
  %6 = call_native PyObject_GetAttr (%1, %2) @2             // def(rax) use(rdi, rsi) copies(stack[1] -> rdi, stack[0] -> rsi)
  jmp &11

&8:
  %9 = call_native PyObject_GetAttr (%2, %1) @2             // def(rax) use(rdi, rsi) copies(stack[0] -> rdi, stack[1] -> rsi)
  jmp &11

&11:
  %12 = add i64 %1, %2                                      // def(rax) use(rax, rcx) copies(stack[1] -> rax, stack[0] -> rcx)
  return %12                                                // use(rax)
})")));
}
}  // namespace

}  // namespace deepmind::s6
