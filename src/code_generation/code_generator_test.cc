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

#include "code_generation/code_generator.h"

#include <Python.h>
#include <frameobject.h>
#include <opcode.h>

#include <cstdint>
#include <iterator>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "allocator.h"
#include "api.h"
#include "code_generation/register_allocator.h"
#include "code_generation/trace_register_allocator.h"
#include "code_object.h"
#include "event_counters.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "metadata.h"
#include "runtime/runtime.h"
#include "runtime/stack_frame.h"
#include "strongjit/function.h"
#include "strongjit/parser.h"
#include "strongjit/test_util.h"
#include "utils/matchers.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using ::deepmind::s6::matchers::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Key;

void PyAndS6Init() {
  static std::once_flag once;
  std::call_once(once, []() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });
}

PyCodeObject* CreateCodeObject() {
  PyObject* empty_tuple = PyTuple_New(0);
  PyObject* names_tuple = PyTuple_New(1);
  // CallAttributeWithManyArguments needs there to be a string called "hello".
  PyTuple_SET_ITEM(names_tuple, 0, PyUnicode_FromString("hello"));
  PyObject* empty_unicode = PyUnicode_FromString("");
  // Ensure bytes has some content so that the bytecode count is nonzero.
  PyObject* dummy_bytes = PyBytes_FromString("abcdefghijklm");
  return PyCode_New(
      /*argcount=*/0, /*kwonlyargcount=*/0, /*nlocals=*/0,
      /*stacksize=*/0, /*flags=*/0, /*code=*/dummy_bytes,
      /*consts=*/empty_tuple, /*names=*/names_tuple,
      /*varnames=*/empty_tuple, /*freevars=*/empty_tuple,
      /*cellvars=*/empty_tuple, /*filename=*/empty_unicode,
      /*name=*/empty_unicode, /*firstlineno=*/0, /*lnotab=*/dummy_bytes);
}

static JitAllocator allocator(4096);  // NOLINT

// Runs the function defined by `b` and returns the result and the CodeObject.
// TODO: The generated code is not yet ASAN-compatible.
__attribute__((no_sanitize("address")))
absl::StatusOr<std::pair<int64_t, std::unique_ptr<CodeObject>>>
RunGeneratedCode(Function f, const CodeGeneratorOptions& options = {}) {
  PyAndS6Init();

  S6_ASSIGN_OR_RETURN(std::unique_ptr<RegisterAllocation> ra,
                      AllocateRegistersWithTrace(f));
  S6_LOG(INFO) << ra->ToString(f);
  S6_RET_CHECK_EQ(PyThreadState_GET()->frame, nullptr);

  PyCodeObject* co = CreateCodeObject();
  S6_CHECK(co);
  S6_ASSIGN_OR_RETURN(std::unique_ptr<CodeObject> object,
                      GenerateCode(std::move(f), *ra, /*program=*/{}, allocator,
                                   co, nullptr, options));

  S6_LOG(INFO) << object->Disassemble();
  auto body = object->GetPyFrameBody();

  auto frame = PyFrame_New(PyThreadState_Get(), co, PyDict_New(), nullptr);
  int64_t ret = reinterpret_cast<int64_t>(
      body(frame, /*profile_counter=*/nullptr, object.get()));
  PyThreadState_GET()->frame = nullptr;
  return std::make_pair(ret, std::move(object));
}

__attribute__((no_sanitize("address")))
absl::StatusOr<std::pair<int64_t, std::unique_ptr<CodeObject>>>
RunGeneratedCode(absl::string_view input,
                 const CodeGeneratorOptions& options = {}) {
  PyAndS6Init();

  S6_ASSIGN_OR_RETURN(Function f, ParseFunction(input));
  return RunGeneratedCode(std::move(f), options);
}

__attribute__((no_sanitize("address"))) absl::StatusOr<std::string>
DisassembleGeneratedCode(absl::string_view input) {
  JitAllocator allocator(4096);

  PyCodeObject* co = CreateCodeObject();

  S6_ASSIGN_OR_RETURN(Function f, ParseFunction(input));
  S6_ASSIGN_OR_RETURN(std::unique_ptr<RegisterAllocation> ra,
                      AllocateRegistersWithTrace(f));
  S6_ASSIGN_OR_RETURN(std::unique_ptr<CodeObject> object,
                      GenerateCode(std::move(f), *ra, {}, allocator, co));

  return object->Disassemble();
}

TEST(CodeGeneratorTest, StackFrameCanRunAndReturn) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $55
  return %1
})";

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(55)));
}

TEST(CodeGeneratorTest, LargeImmediateIsMaterialized) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $120000
  return %1
})";

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(120000)));
}

TEST(CodeGeneratorTest, CompareEqualAndBranch) {
  absl::string_view input = R"(function CompareEqualAndBranch {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = cmp eq i64 %1, %2
  br %3, &5, &7

&5:
  return %2

&7:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// br %3, &5, &7"));
  S6_ASSERT_OK(lm.OnNextLine("mov r11, 0x1"));
  S6_ASSERT_OK(lm.OnNextLine("cmp r11, 0x2"));
  S6_ASSERT_OK(lm.OnNextLine("jnz"));
  S6_ASSERT_OK(lm.OnNextLine("// return"));

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, CompareLessThanAndBranch) {
  absl::string_view input = R"(function CompareLessThanAndBranch {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = cmp lt i64 %1, %2
  br %3, &5, &7

&5:
  return %2

&7:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// br %3, &5, &7"));
  S6_ASSERT_OK(lm.OnNextLine("mov r11, 0x1"));
  S6_ASSERT_OK(lm.OnNextLine("cmp r11, 0x2"));
  S6_ASSERT_OK(lm.OnNextLine("jge"));
  S6_ASSERT_OK(lm.OnNextLine("// return"));

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(2)));
}

TEST(CodeGeneratorTest, ForceAbiCompliance) {
  absl::string_view input = R"(function ForceAbiCompliance {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $43
  br %1, &4, &7

&4:
  %5 = call_native PyTuple_New (%1) @2
  jmp &10 [ %5 ]

&7:
  %8 = call_native PyTuple_New (%2) @2
  jmp &10 [ %8 ]

&10: [ %11 ]
  %12 = call_native PyTuple_Size (%11) @4
  return %12
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(42)));
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  // Check the disassembly to ensure the expected copies exist.
  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// %5 = call_native PyTuple_New (%1)"));
  S6_ASSERT_OK(lm.OnAnyLine("mov rdi, 0x2a"));
  S6_ASSERT_OK(lm.OnAnyLine("call qword"));
  S6_ASSERT_OK(lm.OnAnyLine("// jmp &10 [ %5 ]"));
  S6_ASSERT_OK(lm.OnNextLine("jmp"));
}

// An asmjit Builder pass that rewrites a specific call target to another.
class InterposeSymbol : public asmjit::Pass {
 public:
  InterposeSymbol(asmjit::Imm from, asmjit::Imm to)
      : Pass("interpose-symbol"), from_(from), to_(to) {}

  asmjit::Error run(asmjit::Zone* zone,
                    asmjit::Logger* logger) noexcept override {
    for (auto node = cb()->firstNode(); node != nullptr; node = node->next()) {
      if (!node->isInst()) continue;
      asmjit::InstNode* inst = node->as<asmjit::InstNode>();
      if (inst->id() != asmjit::x86::Inst::kIdCall) continue;
      asmjit::Imm& imm = inst->operands()[0].as<asmjit::Imm>();
      if (imm == from_) imm = to_;
    }
    return asmjit::ErrorCode::kErrorOk;
  }

 private:
  asmjit::Imm from_, to_;
};

TEST(CodeGeneratorTest, Decref1) {
  absl::string_view input = R"(function Decref1 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = call_native PyTuple_New (%1) @2
  decref notnull %2 @2
  return %1
})";
  // As `cb` has to be a pure lambda, we communicate its result through this
  // global.
  static int64_t refcount = -1;

  auto cb = +[](PyObject* obj) {
    // Record the refcount of the object.
    refcount = obj->ob_refcnt;
    Dealloc(obj);
  };

  auto add_passes = [&](asmjit::x86::Builder& builder) {
    builder.addPassT<InterposeSymbol>(asmjit::imm(Dealloc), asmjit::imm(cb));
  };

  ASSERT_THAT(RunGeneratedCode(input, {.add_passes = add_passes}),
              IsOkAndHolds(Key(42)));
  ASSERT_EQ(refcount, 0);
}

TEST(CodeGeneratorTest, Decref1UnmatchedIncref) {
  absl::string_view input = R"(function Decref1 {
&0:                                                         // entry point
  %1 = constant $42
  %2 = call_native PyTuple_New (%1) @2
  incref notnull %2
  decref notnull %2 @2
  return %1
})";
  static int64_t refcount = -1;

  auto cb = +[](PyObject* obj) {
    // Record the refcount of the object.
    refcount = obj->ob_refcnt;
    Dealloc(obj);
  };

  auto add_passes = [&](asmjit::x86::Builder& builder) {
    builder.addPassT<InterposeSymbol>(asmjit::imm(Dealloc), asmjit::imm(cb));
  };

  ASSERT_THAT(RunGeneratedCode(input, {.add_passes = add_passes}),
              IsOkAndHolds(Key(42)));
  // The object should have leaked.
  ASSERT_EQ(refcount, -1);
}

TEST(CodeGeneratorTest, DecrefNull) {
  absl::string_view input = R"(function DecrefNull {
&0:                                                         // entry point
  %1 = constant $0
  decref null? %1 @2
  return %1
})";
  static int64_t refcount = -1;

  auto cb = +[](PyObject* obj) {
    // Record the refcount of the object.
    refcount = obj->ob_refcnt;
    Dealloc(obj);
  };

  auto add_passes = [&](asmjit::x86::Builder& builder) {
    builder.addPassT<InterposeSymbol>(asmjit::imm(Dealloc), asmjit::imm(cb));
  };

  ASSERT_THAT(RunGeneratedCode(input, {.add_passes = add_passes}),
              IsOkAndHolds(Key(0)));
  // The destructor should not have been called on nullptr.
  ASSERT_EQ(refcount, -1);
}

TEST(CodeGeneratorTest, AddInt) {
  absl::string_view input = R"(function AddInt {
&0:
  %1 = constant $37
  %2 = constant $14
  %3 = add i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(51)));
}

TEST(CodeGeneratorTest, SubInt) {
  absl::string_view input = R"(function SubtractInt {
&0:
  %1 = constant $49
  %2 = constant $13
  %3 = subtract i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(36)));
}

TEST(CodeGeneratorTest, MultiplyInt) {
  absl::string_view input = R"(function MultiplyInt {
&0:
  %1 = constant $-17
  %2 = constant $15
  %3 = multiply i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(-255)));
}

struct DivisionTestCase {
  int64_t numerator;
  int64_t denominator;
  int64_t quotient;
  int64_t remainder;
};

using CodeGeneratorDivisionTest = testing::TestWithParam<DivisionTestCase>;

TEST_P(CodeGeneratorDivisionTest, DivideIntImmediate) {
  const DivisionTestCase& test_case = GetParam();
  std::string input = R"(function DivideIntImmediate {
&0:
  %1 = constant $)" + std::to_string(test_case.numerator) +
                      R"(
  %2 = constant $)" + std::to_string(test_case.denominator) +
                      R"(
  %3 = divide i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(test_case.quotient)));
}

TEST_P(CodeGeneratorDivisionTest, DivideIntRegister) {
  const DivisionTestCase& test_case = GetParam();
  std::string input = R"(function DivideIntRegister {
&0:
  %1 = constant $)" + std::to_string(-test_case.numerator) +
                      R"(
  %2 = constant $)" + std::to_string(-test_case.denominator) +
                      R"(
  %3 = negate i64 %1
  %4 = negate i64 %2
  %5 = divide i64 %3, %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(test_case.quotient)));
}

TEST_P(CodeGeneratorDivisionTest, RemainderIntImmediate) {
  const DivisionTestCase& test_case = GetParam();
  std::string input = R"(function RemainderIntImmediate {
&0:
  %1 = constant $)" + std::to_string(test_case.numerator) +
                      R"(
  %2 = constant $)" + std::to_string(test_case.denominator) +
                      R"(
  %3 = remainder i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(test_case.remainder)));
}

TEST_P(CodeGeneratorDivisionTest, RemainderIntRegister) {
  const DivisionTestCase& test_case = GetParam();
  std::string input = R"(function RemainderIntRegister {
&0:
  %1 = constant $)" + std::to_string(-test_case.numerator) +
                      R"(
  %2 = constant $)" + std::to_string(-test_case.denominator) +
                      R"(
  %3 = negate i64 %1
  %4 = negate i64 %2
  %5 = remainder i64 %3, %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(test_case.remainder)));
}

// Python always rounds the quotient down, even when negative.
INSTANTIATE_TEST_SUITE_P(
    CodeGeneratorDivisionTests, CodeGeneratorDivisionTest,
    testing::ValuesIn<DivisionTestCase>({{0, 7, 0, 0},
                                         {0, -7, 0, 0},
                                         {5, 5, 1, 0},
                                         {-5, 5, -1, 0},
                                         {5, -5, -1, 0},
                                         {-5, -5, 1, 0},
                                         {65, 13, 5, 0},
                                         {-65, 13, -5, 0},
                                         {65, -13, -5, 0},
                                         {-65, -13, 5, 0},
                                         {113, 12, 9, 5},
                                         {-113, 12, -10, 7},
                                         {113, -12, -10, -7},
                                         {-113, -12, 9, -5}}));

TEST(CodeGeneratorTest, NegateInt) {
  absl::string_view input = R"(function NegateInt {
&0:
  %1 = constant $38
  %2 = negate i64 %1
  return %2
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(-38)));
}

TEST(CodeGeneratorTest, AndInt) {
  absl::string_view input = R"(function AndInt {
&0:
  %1 = constant $31
  %2 = constant $48
  %3 = and i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(16)));
}

TEST(CodeGeneratorTest, OrInt) {
  absl::string_view input = R"(function OrInt {
&0:
  %1 = constant $31
  %2 = constant $48
  %3 = or i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(63)));
}

TEST(CodeGeneratorTest, XorInt) {
  absl::string_view input = R"(function XorInt {
&0:
  %1 = constant $31
  %2 = constant $48
  %3 = xor i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(47)));
}

TEST(CodeGeneratorTest, ShiftLeftInt) {
  absl::string_view input = R"(function ShiftLeftInt {
&0:
  %1 = constant $5
  %2 = constant $9
  %3 = shift_left i64 %1, %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(2560)));
}

TEST(CodeGeneratorTest, ShiftRightSignedInt) {
  absl::string_view input = R"(function ShiftRightSignedInt {
&0:
  %1 = constant $-191
  %2 = constant $5
  %3 = shift_right_signed i64 %1, %2
  return %3
})";
  // Expect this to perform signed right-shift.
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(-6)));
}

TEST(CodeGeneratorTest, NotInt) {
  absl::string_view input = R"(function NotInt {
&0:
  %1 = constant $63
  %2 = not i64 %1
  return %2
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(-64)));
}

TEST(CodeGeneratorTest, AddFloat) {
  absl::string_view input = R"(function AddFloat {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = constant $1
  %8 = constant $4
  %9 = box long %7
  %10 = box long %8
  %11 = call_native PyNumber_TrueDivide (%9, %10) @4
  %12 = unbox float %11
  %13 = add f64 %6, %12
  return %13
})";
  double expected = absl::bit_cast<int64_t>(0.75);  // 1/2 + 1/4 = 3/4
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, SubtractFloat) {
  absl::string_view input = R"(function SubtractFloat {
&0:
  %1 = constant $3
  %2 = constant $8
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = constant $1
  %8 = constant $2
  %9 = box long %7
  %10 = box long %8
  %11 = call_native PyNumber_TrueDivide (%9, %10) @4
  %12 = unbox float %11
  %13 = subtract f64 %6, %12
  return %13
})";
  double expected = absl::bit_cast<int64_t>(-0.125);  // 3/8 - 1/2 = -1/8
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, MultiplyFloat) {
  absl::string_view input = R"(function MultiplyFloat {
&0:
  %1 = constant $3
  %2 = constant $2
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = constant $-5
  %8 = constant $4
  %9 = box long %7
  %10 = box long %8
  %11 = call_native PyNumber_TrueDivide (%9, %10) @4
  %12 = unbox float %11
  %13 = multiply f64 %6, %12
  return %13
})";
  int64_t expected = absl::bit_cast<int64_t>(-1.875);  // 3/2 * -5/4 = -15/8
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, DivideFloat) {
  absl::string_view input = R"(function DivideFloat {
&0:
  %1 = constant $-3
  %2 = constant $2
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = constant $8
  %8 = constant $1
  %9 = box long %7
  %10 = box long %8
  %11 = call_native PyNumber_TrueDivide (%9, %10) @4
  %12 = unbox float %11
  %13 = divide f64 %6, %12
  return %13
})";
  int64_t expected = absl::bit_cast<int64_t>(-0.1875);  // -3/2 / 8 = -3/16
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, NegFloat) {
  absl::string_view input = R"(function NegateFloat {
&0:
  %1 = constant $37
  %2 = constant $16
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = negate f64 %6
  return %7
})";
  int64_t expected = absl::bit_cast<int64_t>(-2.3125);  // -37/16
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, CmpFloat) {
  absl::string_view input = R"(function CmpFloat {
&0:
  %1 = constant $-9
  %2 = constant $8
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox float %5
  %7 = constant $-1
  %8 = constant $2
  %9 = box long %7
  %10 = box long %8
  %11 = call_native PyNumber_TrueDivide (%9, %10) @4
  %12 = unbox float %11
  %13 = cmp lt f64 %6, %12
  return %13
})";
  // Expect 'true', as -9/8 < -1/2.
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, IntToFloat) {
  absl::string_view input = R"(function IntToFloat {
&0:
  %1 = constant $-13
  %2 = int_to_float %1
  return %2
})";
  int64_t expected = absl::bit_cast<int64_t>(-13.);
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, LargeIntToFloat) {
  // Convert (3<<56) to floating point.
  // This should occur without loss of precision.
  absl::string_view input = R"(function LargeIntToFloat {
&0:
  %1 = constant $3
  %2 = constant $56
  %3 = shift_left i64 %1, %2
  %4 = int_to_float %3
  return %4
})";
  int64_t expected = absl::bit_cast<int64_t>(3. * 256. * 256. * 256. * 256. *
                                             256. * 256. * 256.);
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(expected)));
}

TEST(CodeGeneratorTest, NonOverflowingMultiply) {
  // Perform (1<<30) * (1<<30), which should not overflow.
  absl::string_view input = R"(function NonOverflowingMultiply {
&0:
  %1 = constant $1
  %2 = constant $30
  %3 = shift_left i64 %1, %2
  %4 = multiply i64 %3, %3
  %5 = overflowed? %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, OverflowingMultiply) {
  // Perform (1<<40) * (1<<40), which should overflow.
  absl::string_view input = R"(function OverflowingMultiply {
&0:
  %1 = constant $1
  %2 = constant $40
  %3 = shift_left i64 %1, %2
  %4 = multiply i64 %3, %3
  %5 = overflowed? %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, NonOverflowingMultiplyAndBranch) {
  // Perform (1<<30) * (1<<30), which should not overflow.
  absl::string_view input = R"(function NonOverflowingMultiplyAndBranch {
&0:
  %1 = constant $1
  %2 = constant $30
  %3 = shift_left i64 %1, %2
  %4 = multiply i64 %3, %3
  %5 = overflowed? %4
  br %5, &7, &9

&7:
  return %1

&9:
  return %2
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// br %5, &7, &9"));
  S6_ASSERT_OK(lm.OnNextLine("jno"));
  S6_ASSERT_OK(lm.OnNextLine("// return"));

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(30)));
}

TEST(CodeGeneratorTest, OverflowingMultiplyAndBranch) {
  // Perform (1<<40) * (1<<40), which should overflow.
  absl::string_view input = R"(function OverflowingMultiplyAndBranch {
&0:
  %1 = constant $1
  %2 = constant $40
  %3 = shift_left i64 %1, %2
  %4 = multiply i64 %3, %3
  %5 = overflowed? %4
  br %5, &7, &9

&7:
  return %1

&9:
  return %2
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// br %5, &7, &9"));
  S6_ASSERT_OK(lm.OnNextLine("jno"));
  S6_ASSERT_OK(lm.OnNextLine("// return"));

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, NonOverflowingLeftShift) {
  // Perform (1<<30) << 30, which should not overflow.
  absl::string_view input = R"(function NonOverflowingLeftShift {
&0:
  %1 = constant $1
  %2 = constant $30
  %3 = shift_left i64 %1, %2
  %4 = shift_left i64 %3, %2
  %5 = overflowed? %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, OverflowingLeftShift) {
  // Perform (1<<40) << 40, which should overflow.
  absl::string_view input = R"(function OverflowingLeftShift {
&0:
  %1 = constant $1
  %2 = constant $40
  %3 = shift_left i64 %1, %2
  %4 = shift_left i64 %3, %2
  %5 = overflowed? %4
  return %5
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, NonzeroDivisorValid) {
  absl::string_view input = R"(function DivisorCheck {
&0:
  %1 = constant $7
  %2 = int_to_float %1
  %3 = float_zero? %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, ZeroDivisorDetected) {
  absl::string_view input = R"(function DivisorCheck {
&0:
  %1 = constant $0
  %2 = int_to_float %1
  %3 = float_zero? %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, NonzeroDivisorValidWithBranch) {
  absl::string_view input = R"(function DivisorCheck {
&0:
  %1 = constant $3
  %2 = constant $7
  %3 = int_to_float %2
  %4 = float_zero? %3
  br %4, &6, &8

&6:
  return %1

&8:
  return %2
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(7)));
}

TEST(CodeGeneratorTest, ZeroDivisorDetectedWithBranch) {
  absl::string_view input = R"(function DivisorCheck {
&0:
  %1 = constant $3
  %2 = constant $0
  %3 = int_to_float %2
  %4 = float_zero? %3
  br %4, &6, &8

&6:
  return %1

&8:
  return %2
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(3)));
}

TEST(CodeGeneratorTest, BoxAndUnboxLong) {
  absl::string_view input = R"(function BoxAndUnboxLong {
&0:
  %1 = constant $37
  %2 = box long %1
  %3 = call_native PyNumber_Negative (%2) @2
  %4 = unbox long %3
  return %4
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(-37)));
}

TEST(CodeGeneratorTest, BoxAndUnboxBoolFalse) {
  absl::string_view input = R"(function BoxAndUnboxBoolFalse {
&0:
  %1 = constant $0
  %2 = box bool %1
  %3 = unbox bool %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, BoxAndUnboxBoolTrue) {
  absl::string_view input = R"(function BoxAndUnboxBoolTrue {
&0:
  %1 = constant $1
  %2 = box bool %1
  %3 = unbox bool %2
  return %3
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, BoxAndUnboxFloat) {
  absl::string_view input = R"(function BoxAndUnboxFloat {
&0:
  %1 = constant $41
  %2 = box float %1
  %3 = unbox float %2
  return %3
})";
  // This simply round-trips the bit pattern '41', which will of course have
  // an entirely different interpretation as a float.
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(41)));
}

TEST(CodeGeneratorTest, UnboxSmallIntDoesNotOverflow) {
  absl::string_view input = R"(function UnboxOverflowingInt {
&0:
  %1 = constant $23
  %2 = box long %1
  %3 = unbox long %2
  %4 = overflowed? %3
  return %4
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, UnboxOverflowingPositiveInt) {
  // Construct (1<<48) * (1<<48) as a Python long, and try and unbox it.
  absl::string_view input = R"(function UnboxOverflowingInt {
&0:
  %1 = constant $1
  %2 = constant $48
  %3 = shift_left i64 %1 %2
  %4 = box long %3
  %5 = call_native PyNumber_Multiply (%4, %4) @2
  %6 = unbox long %5
  %7 = overflowed? %6
  return %7
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, UnboxOverflowingNegativeInt) {
  // Construct -(1<<48) * (1<<48) as a Python long, and try and unbox it.
  absl::string_view input = R"(function UnboxOverflowingInt {
&0:
  %1 = constant $1
  %2 = constant $48
  %3 = shift_left i64 %1 %2
  %4 = box long %3
  %5 = call_native PyNumber_Multiply (%4, %4) @2
  %6 = call_native PyNumber_Negative (%5) @4
  %7 = unbox long %6
  %8 = overflowed? %7
  return %8
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, UnboxIntRejectsFloat) {
  absl::string_view input = R"(function UnboxFloatAsInt {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = box long %1
  %4 = box long %2
  %5 = call_native PyNumber_TrueDivide (%3, %4) @2
  %6 = unbox long %5
  %7 = overflowed? %6
  return %7
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, UnboxBoolRejectsInt) {
  absl::string_view input = R"(function UnboxIntAsBool {
&0:
  %1 = constant $1
  %2 = box long %1
  %3 = unbox bool %2
  %4 = overflowed? %3
  return %4
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

TEST(CodeGeneratorTest, UnboxFloatAcceptsInt) {
  absl::string_view input = R"(function UnboxIntAsFloat {
&0:
  %1 = constant $1
  %2 = box long %1
  %3 = unbox float %2
  %4 = overflowed? %3
  return %4
})";
  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(0)));
}

TEST(CodeGeneratorTest, CallPython) {
  absl::string_view input = R"(function CallPython {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $3
  %3 = constant $66666
  %4 = constant $77777
  %5 = call_python %1 (%4, %2, %3) @23
  return %5
})";
  static int64_t callee = -1;
  static NoDestructor<std::vector<int64_t>> args;
  static int64_t bytecode_offset = -1;
  static int64_t names = -1;
  constexpr int64_t kReturnValue = 0xdeadbeef;

  auto cb = +[](int64_t arg_count_, int64_t* args_, int64_t names_,
                StackFrame* stack_frame) {
    callee = args_[-1];
    std::copy(args_, args_ + arg_count_, std::back_inserter(*args));
    bytecode_offset = stack_frame->bytecode_offset();
    names = names_;
    return kReturnValue;
  };

  auto add_passes = [&](asmjit::x86::Builder& builder) {
    builder.addPassT<InterposeSymbol>(asmjit::imm(CallPython), asmjit::imm(cb));
  };

  ASSERT_THAT(RunGeneratedCode(input, {.add_passes = add_passes}),
              IsOkAndHolds(Key(kReturnValue)));
  ASSERT_EQ(callee, 42);
  ASSERT_THAT(*args, ElementsAre(77777, 3, 66666));
  ASSERT_EQ(bytecode_offset, 23);
  ASSERT_EQ(names, 0);
}

TEST(CodeGeneratorTest, CallAttributeWithManyArguments) {
  absl::string_view input = R"(function CallAttributeWithManyArguments {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $3
  %3 = constant $1
  %4 = constant $2
  %5 = constant $3
  %6 = constant $4
  %7 = constant $5
  %8 = constant $6
  %9 = constant $7
  %10 = constant $8
  %11 = constant $9
  %12 = call_attribute %1 :: "hello" (%3, %4, %5, %6, %7, %8, %9, %10, %11) @56, @58
  return %12
})";
  static int64_t callee = -1;
  static NoDestructor<std::vector<int64_t>> args;
  static int64_t names = -1;
  constexpr int64_t kReturnValue = 0xdeadbeef;

  auto cb = +[](int64_t arg_count_, int64_t* args_, int64_t names_,
                StackFrame* stack_frame) {
    callee = args_[-1];
    std::copy(args_, args_ + arg_count_, std::back_inserter(*args));
    names = names_;
    return kReturnValue;
  };

  auto add_passes = [&](asmjit::x86::Builder& builder) {
    builder.addPassT<InterposeSymbol>(asmjit::imm(CallAttribute),
                                      asmjit::imm(cb));
  };

  ASSERT_THAT(RunGeneratedCode(input, {.add_passes = add_passes}),
              IsOkAndHolds(Key(kReturnValue)));
  ASSERT_EQ(callee, 42);
  ASSERT_THAT(*args, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));
  ASSERT_EQ(names, 0);
}

TEST(CodeGeneratorTest, TrivialJmpsRemoved) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $1
  %2 = load i64 [ %1 + $0 ]
  jmp &4

&4:
  unreachable
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("load"));
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("jmp"), "int3"));
}

TEST(CodeGeneratorTest, EventCounterIsIncremented) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $1
  increment_event_counter "test"
  return %1
})";

  *EventCounters::Instance().GetEventCounter("test") = 0;
  S6_ASSERT_OK(RunGeneratedCode(input));
  ASSERT_THAT(*EventCounters::Instance().GetEventCounter("test"),
              testing::Eq(1));
}

TEST(CodeGeneratorTest, DeoptimizeIfSafepointInst) {
  absl::string_view input = R"(function DeoptimizeIfSafepointInst {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = cmp gt i64 %1, %2
  deoptimize_if_safepoint %3, @0, ""
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(std::string s, DisassembleGeneratedCode(input));

  LineMatcher lm(s);
  S6_ASSERT_OK(lm.OnAnyLine("// deoptimize_if_safepoint %3, @0"));
  S6_ASSERT_OK(lm.OnNextLine("mov r11, 0x1"));
  S6_ASSERT_OK(lm.OnNextLine("cmp r11, 0x2"));
  S6_ASSERT_OK(lm.OnNextLine("jg"));
  S6_ASSERT_OK(lm.OnNextLine("// return"));

  ASSERT_THAT(RunGeneratedCode(input), IsOkAndHolds(Key(1)));
}

}  // namespace
}  // namespace deepmind::s6
