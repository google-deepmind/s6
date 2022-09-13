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

#include "strongjit/parser.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "classes/class_manager.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"
#include "utils/matchers.h"

namespace deepmind::s6 {
namespace {

using ::deepmind::s6::matchers::IsOkAndHolds;
using ::deepmind::s6::matchers::StatusIs;
using testing::StartsWith;

// A trivial subclass of std::string. Strings wrapped in RawString can be
// printed by UniversalPrint, below. We don't escape newlines to make test
// output easier to read.
class RawString : public std::string {
 public:
  explicit RawString(std::string s) : std::string(s) {}
  explicit RawString(absl::string_view s) : std::string(s) {}
};

void UniversalPrint(const RawString& value, ::std::ostream* os) {
  *os << "R\"(" << value << ")\"";
}

absl::StatusOr<std::string> FormatParsedInstruction(absl::string_view str,
                                                    Function* f,
                                                    const ClassManager& mgr) {
  S6_ASSIGN_OR_RETURN(Instruction * inst, ParseInstruction(str, f, mgr));
  S6_ASSIGN_OR_RETURN(std::string s, Format(*inst, f, mgr));
  inst->erase();
  return s;
}

absl::StatusOr<RawString> FormatParsedFunction(
    absl::string_view str, const ClassManager& mgr = ClassManager::Instance()) {
  S6_ASSIGN_OR_RETURN(Function f, ParseFunction(str, mgr));
  S6_ASSIGN_OR_RETURN(std::string s, Format(f, PredecessorAnnotator(), mgr));
  return RawString(s);
}

void Roundtrip(absl::string_view str, Function* f,
               const ClassManager& mgr = ClassManager::Instance()) {
  EXPECT_THAT(FormatParsedInstruction(str, f, mgr), IsOkAndHolds(str));
}

void RoundtripFunction(absl::string_view str,
                       const ClassManager& mgr = ClassManager::Instance()) {
  EXPECT_THAT(FormatParsedFunction(str, mgr), IsOkAndHolds(RawString(str)));
}

TEST(Parser, RoundtripInsts) {
  Function f("test");
  Block* b = f.CreateBlock();                // &0
  b->push_back(f.Create<ConstantInst>(42));  // %1 = constant $42
  b->push_back(f.Create<ConstantInst>(43));  // %2 = constant $43
  b->push_back(f.Create<ConstantInst>(44));  // %3 = constant $44
  b = f.CreateBlock();                       // &4

  Roundtrip("%5 = constant $42", &f);
  Roundtrip("%5 = constant $-65", &f);
  Roundtrip("jmp &0 [ %1, %3 ]", &f);
  Roundtrip("br %1, &0, &4", &f);
  Roundtrip("br %1, &0 [ %1, %3 ], &4", &f);
  Roundtrip("br %1, &0, &4 [ %1 ]", &f);
  Roundtrip("br %1, &0 [ %1, %3 ], &4 [ %2, %1 ]", &f);
  Roundtrip("br %1, deoptimized &0 [ %1, %3 ], &4 [ %2, %1 ]", &f);
  Roundtrip("br %1, &0, deoptimized &4", &f);
  Roundtrip("%5 = cmp eq i64 %1, %2", &f);
  Roundtrip("%5 = cmp ge i64 %1, %2", &f);
  Roundtrip("%5 = cmp gt f64 %1, %2", &f);
  Roundtrip("%5 = cmp lt f64 %1, %2", &f);
  Roundtrip("%5 = cmp le i64 %1, %2", &f);
  Roundtrip("unreachable", &f);
  Roundtrip("except @0", &f);
  Roundtrip("except &4 @2", &f);
  Roundtrip("except [ %1, %3 ] @2", &f);
  Roundtrip("except &4 [ %3, %2 ] @2", &f);
  Roundtrip("deoptimize_if %1, &0 [ %2 ], &4, materializing values [%1]", &f);
  Roundtrip(
      "deoptimize_if not %2, &4, &0 [ %2, %1 ], materializing values [%3, %1]",
      &f);
  Roundtrip("return %2", &f);
  Roundtrip("%5 = negate f64 %1", &f);
  Roundtrip("%5 = not i64 %1", &f);
  Roundtrip("%5 = add i64 %1, %2", &f);
  Roundtrip("%5 = subtract f64 %1, %2", &f);
  Roundtrip("%5 = multiply i64 %1, %2", &f);
  Roundtrip("%5 = divide f64 %1, %2", &f);
  Roundtrip("%5 = remainder i64 %1, %2", &f);
  Roundtrip("%5 = and i64 %1, %2", &f);
  Roundtrip("%5 = or i64 %1, %2", &f);
  Roundtrip("%5 = xor i64 %1, %2", &f);
  Roundtrip("%5 = shift_left i64 %1, %2", &f);
  Roundtrip("%5 = shift_right_signed i64 %1, %2", &f);
  Roundtrip("%5 = int_to_float %1", &f);
  Roundtrip("%5 = sext i64 %1", &f);
  Roundtrip("incref null? %1", &f);
  Roundtrip("decref notnull %1 @52", &f);
  Roundtrip("%5 = load s32 [ %1 + $42 ]", &f);
  Roundtrip("%5 = load steal i64 [ %1 + $42 + %2 * 4 ]", &f);
  Roundtrip("store i16 %1, [ %2 + $0 ]", &f);
  Roundtrip("store ref i64 %1, [ %2 + $0 + %3 ]", &f);
  Roundtrip("%5 = frame_variable consts, $0", &f);
  Roundtrip("%5 = call_native PyObject_SetAttr (%1) @0", &f);
  Roundtrip("%5 = call_native PyObject_SetAttr @0", &f);
  Roundtrip("%5 = call_python %1 @0", &f);
  Roundtrip("%5 = call_python %1 (%2, %3) @0", &f);
  Roundtrip("%5 = call_native_indirect %1 (%2, %3) @0", &f);
  Roundtrip("%5 = call_python %1 names %3 (%2) @4", &f);
  Roundtrip("%5 = call_attribute %1 :: \"hello\" names %2 (%2) @56, @58", &f);
  Roundtrip("bytecode_begin @56 stack [%2, %3]", &f);
  Roundtrip("bytecode_begin @56 stack [%2, %3] {kExcept @22 $4}", &f);
  Roundtrip(
      "bytecode_begin @56 stack [%2, %3] {kExcept @22 $4}, {kLoop @11 $0}", &f);
  Roundtrip(
      "bytecode_begin @56 stack [%2, %3] fastlocals [%3, %2] "
      "{kExceptHandler @22 $4}, {kLoop @11 $0}",
      &f);
  Roundtrip("bytecode_begin @56 {kFinally @11 $0}, {kExceptHandler @22 $4}",
            &f);
  Roundtrip("bytecode_begin @56 {kFinallyHandler @22 $4}, {kLoop @33 $0}", &f);
  Roundtrip("bytecode_begin @56 {kFinallyHandler @22 @42 $4}", &f);
  Roundtrip(
      "bytecode_begin @56 {kExceptHandler @22 $4}, {kFinallyHandler @22 "
      "fallthrough_popped $4}",
      &f);
  Roundtrip(
      "bytecode_begin @56 {kExceptHandler @22 $4}, {kFinallyHandler @22 @42 "
      "fallthrough_popped $4}",
      &f);
  Roundtrip("bytecode_begin @56 fastlocals [%2]", &f);
  Roundtrip("%5 = load_from_dict %2, $3, split", &f);
  Roundtrip("%5 = store_to_dict %3 into %2, $1 combined", &f);
  Roundtrip(
      "%5 = yield_value %3 @56 stack [%2, %3] {kExcept @22 $4}, {kLoop @11 $0}",
      &f);
  Roundtrip(
      "%5 = yield_value %3 @56 stack [%2, %3] fastlocals [%3] {kExcept @22 "
      "$4}, {kLoop @11 $0}",
      &f);
  Roundtrip("%5 = unbox long %2", &f);
  Roundtrip("%5 = box bool %3", &f);
  Roundtrip("%5 = box float %4", &f);
  Roundtrip("%5 = overflowed? %4", &f);
  Roundtrip("%5 = float_zero? %4", &f);
  Roundtrip(
      "deoptimize_if_safepoint %3, @56 stack [%2, %3] {kExcept @22 $4}, {kLoop "
      "@11 $0}, \"reason\"",
      &f);
  Roundtrip(
      "deoptimize_if_safepoint not %3, @56 stack [%2, %3] fastlocals [%2] "
      "{kExcept @22 $4}, {kLoop @11 $0}, \"\"",
      &f);
  Roundtrip("%5 = get_class_id %2", &f);
  Roundtrip("%5 = get_instance_class_id %2", &f);
  Roundtrip("%5 = get_object_dict %2 dictoffset $0 type $0", &f);
  Roundtrip("%5 = check_class_id %2 class_id $3", &f);
  Roundtrip("%5 = deoptimized_asynchronously?", &f);

  ClassManager mgr;
  S6_EXPECT_OK(Class::Create(mgr, "myclassname").status());
  Roundtrip("%5 = constant_attribute \"foobar\" of myclassname#1", &f, mgr);
  Roundtrip("%5 = call_vectorcall %2 self %3 (%4) @2", &f);
  Roundtrip("%5 = call_vectorcall %2 self %3 names %4 (%4, %3) @4", &f);
  Roundtrip("bytecode_begin @56 extras [%1]", &f);
}

TEST(Parser, NegativeInsts) {
  Function f("test");
  Block* b = f.CreateBlock();
  b->push_back(f.Create<ConstantInst>(42));  // %1 = constant $42

  EXPECT_THAT(ParseInstruction("%2 = pumpernickel", &f),
              StatusIs(absl::StatusCode::kInternal,
                       "unknown instruction mnemonic `pumpernickel'"));
  EXPECT_THAT(
      ParseInstruction("br &0, &0", &f),
      StatusIs(absl::StatusCode::kInvalidArgument, "expected `%' at:&0, &0"));
  EXPECT_THAT(ParseInstruction("br %1, &0 [  %1, ", &f),
              StatusIs(absl::StatusCode::kInvalidArgument, "expected `%' at:"));
  EXPECT_THAT(ParseInstruction("br %1, &0 [  %1 ", &f),
              StatusIs(absl::StatusCode::kInvalidArgument, "expected `]' at:"));
  EXPECT_THAT(ParseInstruction(
                  R"(%5 = call_attribute %1 :: "foo\"bar" (%1) @2, @4)", &f),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("invalid string literal")));
}

TEST(Parser, CanParseFunctions) {
  RoundtripFunction(R"(function foobar {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  jmp &4 [ %1, %2 ]

&4: [ %5, %6 ]                                              // preds: &0
  %7 = constant $65

&8: deoptimized [ %9, %10 ]                                 // no predecessors
  %11 = constant $65

&12:                                                        // no predecessors
})");

  RoundtripFunction(R"(function foobar {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5 [ %1, %1, %1, %1, %1, %1 ]

&3:                                                         // preds: &0
  except &5 @0

&5: finally [ %6, %7, %8, %9, %10, %11 ]                    // preds: &0, &3
  except &13 @2

&13: except [ %14, %15, %16, %17, %18, %19 ]                // preds: &5
})");

  ClassManager mgr;
  S6_EXPECT_OK(Class::Create(mgr, "myclassname").status());
  RoundtripFunction(R"(type_feedback @2 monomorphic, myclassname#1
type_feedback @4.1 UNSTABLE megamorphic
type_feedback @6 polymorphic, either myclassname#1 or myclassname#1

function foobar {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
})",
                    mgr);
}

}  // namespace
}  // namespace deepmind::s6
