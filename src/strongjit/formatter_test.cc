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

#include "strongjit/formatter.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/base.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"
#include "strongjit/test_util.h"
#include "type_feedback.h"
#include "utils/matchers.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using deepmind::s6::matchers::IsOkAndHolds;
using testing::StrEq;

absl::StatusOr<RawString> RawStringFormat(
    const Function& f, const ClassManager& mgr = ClassManager::Instance()) {
  S6_ASSIGN_OR_RETURN(std::string s, Format(f, PredecessorAnnotator(), mgr));
  return RawString(s);
}

void ExpectFormatted(Instruction* inst, absl::string_view expected, Block* b) {
  b->push_back(inst);
  EXPECT_THAT(Format(*inst), IsOkAndHolds(StrEq(expected)));
  inst->erase();
}

TEST(FormatterTest, CanFormatInstructions) {
  Function f("foobar");
  Block* b = f.CreateBlock();
  Instruction* c = f.Create<ConstantInst>(42);
  b->push_back(c);
  Instruction* c2 = f.Create<ConstantInst>(42);
  b->push_back(c2);

  ExpectFormatted(f.Create<ConstantInst>(42), "%3 = constant $42", b);
  ExpectFormatted(f.Create<ConstantInst>(-64), "%3 = constant $-64", b);
  ExpectFormatted(f.Create<JmpInst>(b), "jmp &0", b);
  ExpectFormatted(f.Create<BrInst>(c, b, absl::Span<Value* const>(), b,
                                   absl::Span<Value* const>()),
                  "br %1, &0, &0", b);
  ExpectFormatted(
      f.Create<CompareInst>(CompareInst::kEqual, NumericInst::kInt64, c, c2),
      "%3 = cmp eq i64 %1, %2", b);
  ExpectFormatted(f.Create<CompareInst>(CompareInst::kGreaterEqual,
                                        NumericInst::kInt64, c, c2),
                  "%3 = cmp ge i64 %1, %2", b);
  ExpectFormatted(f.Create<CompareInst>(CompareInst::kGreaterThan,
                                        NumericInst::kInt64, c, c2),
                  "%3 = cmp gt i64 %1, %2", b);
  ExpectFormatted(
      f.Create<CompareInst>(CompareInst::kLessThan, NumericInst::kInt64, c, c2),
      "%3 = cmp lt i64 %1, %2", b);
  ExpectFormatted(f.Create<CompareInst>(CompareInst::kLessEqual,
                                        NumericInst::kInt64, c, c2),
                  "%3 = cmp le i64 %1, %2", b);
  ExpectFormatted(f.Create<UnreachableInst>(), "unreachable", b);
  ExpectFormatted(f.Create<ExceptInst>(), "except @0", b);
  ExpectFormatted(f.Create<ExceptInst>(b, absl::Span<Value* const>{}, 42),
                  "except &0 @42", b);
  ExpectFormatted(f.Create<ReturnInst>(c), "return %1", b);
  ExpectFormatted(f.Create<NotInst>(NumericInst::kInt64, c), "%3 = not i64 %1",
                  b);
  ExpectFormatted(f.Create<NegateInst>(NumericInst::kInt64, c),
                  "%3 = negate i64 %1", b);
  ExpectFormatted(f.Create<AddInst>(NumericInst::kInt64, c, c2),
                  "%3 = add i64 %1, %2", b);
  ExpectFormatted(f.Create<SubtractInst>(NumericInst::kInt64, c, c2),
                  "%3 = subtract i64 %1, %2", b);
  ExpectFormatted(f.Create<MultiplyInst>(NumericInst::kInt64, c, c2),
                  "%3 = multiply i64 %1, %2", b);
  ExpectFormatted(f.Create<DivideInst>(NumericInst::kInt64, c, c2),
                  "%3 = divide i64 %1, %2", b);
  ExpectFormatted(f.Create<RemainderInst>(NumericInst::kInt64, c, c2),
                  "%3 = remainder i64 %1, %2", b);
  ExpectFormatted(f.Create<AndInst>(NumericInst::kInt64, c, c2),
                  "%3 = and i64 %1, %2", b);
  ExpectFormatted(f.Create<OrInst>(NumericInst::kInt64, c, c2),
                  "%3 = or i64 %1, %2", b);
  ExpectFormatted(f.Create<XorInst>(NumericInst::kInt64, c, c2),
                  "%3 = xor i64 %1, %2", b);
  ExpectFormatted(f.Create<ShiftLeftInst>(NumericInst::kInt64, c, c2),
                  "%3 = shift_left i64 %1, %2", b);
  ExpectFormatted(f.Create<ShiftRightSignedInst>(NumericInst::kInt64, c, c2),
                  "%3 = shift_right_signed i64 %1, %2", b);
  ExpectFormatted(f.Create<IntToFloatInst>(c), "%3 = int_to_float %1", b);
  ExpectFormatted(f.Create<IncrefInst>(Nullness::kMaybeNull, c),
                  "incref null? %1", b);
  ExpectFormatted(f.Create<DecrefInst>(Nullness::kNotNull, c, 42),
                  "decref notnull %1 @42", b);
  ExpectFormatted(f.Create<LoadInst>(LoadInst::Operand(c, 42)),
                  "%3 = load i64 [ %1 + $42 ]", b);
  ExpectFormatted(
      f.Create<LoadInst>(LoadInst::Operand(c), false, LoadInst::kUnsigned16),
      "%3 = load u16 [ %1 + $0 ]", b);
  ExpectFormatted(f.Create<LoadInst>(LoadInst::Operand(c, c2), true),
                  "%3 = load steal i64 [ %1 + $0 + %2 ]", b);
  ExpectFormatted(f.Create<StoreInst>(c, StoreInst::Operand(c2)),
                  "store i64 %1, [ %2 + $0 ]", b);
  ExpectFormatted(f.Create<StoreInst>(c, StoreInst::Operand(c2, 23), false,
                                      StoreInst::kInt8),
                  "store i8 %1, [ %2 + $23 ]", b);
  ExpectFormatted(
      f.Create<StoreInst>(
          c, StoreInst::Operand(c2, 23, c, StoreInst::Shift::k4), true),
      "store ref i64 %1, [ %2 + $23 + %1 * 4 ]", b);
  ExpectFormatted(f.Create<FrameVariableInst>(
                      FrameVariableInst::FrameVariableKind::kNames, 2),
                  "%3 = frame_variable names, $2", b);
  ExpectFormatted(f.Create<CallNativeInst>(Callee::kPyObject_GetAttr,
                                           absl::Span<Value* const>{c}),
                  "%3 = call_native PyObject_GetAttr (%1) @0", b);
  ExpectFormatted(f.Create<CallNativeInst>(Callee::kPyObject_SetAttr,
                                           absl::Span<Value* const>{c, c2}),
                  "%3 = call_native PyObject_SetAttr (%1, %2) @0", b);
  ExpectFormatted(f.Create<CallPythonInst>(c), "%3 = call_python %1 @0", b);
  ExpectFormatted(
      f.Create<CallPythonInst>(c, absl::Span<Value* const>{c2}, c2, 56),
      "%3 = call_python %1 names %2 (%2) @56", b);

  ExpectFormatted(
      f.Create<CallAttributeInst>(c, f.GetStringTable()->InternString("hello"),
                                  absl::Span<Value* const>{c2}, c2, 56, 42),
      "%3 = call_attribute %1 :: \"hello\" names %2 (%2) @56, @42", b);
}

TEST(FormatterTest, CanFormatFunctions) {
  Function f("foobar");
  Block* b = f.CreateBlock();
  Block* b2 = f.CreateBlock();
  b2->CreateBlockArgument();
  b2->CreateBlockArgument();
  f.CreateBlock();
  Instruction* c = f.Create<ConstantInst>(42);
  b->push_back(c);
  Instruction* c2 = f.Create<ConstantInst>(42);
  b->push_back(c2);
  Instruction* c3 = f.Create<JmpInst>(b2);
  b->push_back(c3);
  Instruction* c4 = f.Create<ConstantInst>(65);
  b2->push_back(c4);
  Instruction* c5 = f.Create<BrInst>(c4, b, b);
  b2->push_back(c5);

  b->AddPredecessor(b2);
  b2->AddPredecessor(b);

  EXPECT_THAT(RawStringFormat(f), IsOkAndHolds(RawString(R"(function foobar {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  jmp &4

&4: [ %5, %6 ]                                              // preds: &0
  %7 = constant $65
  br %7, &0, &0

&9:                                                         // no predecessors
})")));
}

TEST(FormatterTest, CanFormatTypeFeedback) {
  Function f("foobar");
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls1, Class::Create(mgr, "test-class"));
  S6_ASSERT_OK_AND_ASSIGN(Class * cls2, Class::Create(mgr, "another_test"));
  f.type_feedback()[{PcValue::FromOffset(0), 0}] = ClassDistributionSummary(
      ClassDistributionSummary::kMonomorphic,
      {static_cast<int32_t>(cls1->id()), 0, 0, 0}, true);
  f.type_feedback()[{PcValue::FromOffset(6), 1}] =
      ClassDistributionSummary(ClassDistributionSummary::kPolymorphic,
                               {static_cast<int32_t>(cls2->id()),
                                static_cast<int32_t>(cls1->id()), 0, 0},
                               true);

  EXPECT_THAT(
      RawStringFormat(f, mgr),
      IsOkAndHolds(RawString(R"(type_feedback @0 monomorphic, test-class#1
type_feedback @6.1 polymorphic, either another_test#2 or test-class#1

function foobar {
})")));
}

}  // namespace
}  // namespace deepmind::s6
