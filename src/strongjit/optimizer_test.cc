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

#include "strongjit/optimizer.h"

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "api.h"
#include "classes/class.h"
#include "classes/class_manager.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "interpreter.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/optimize_calls.h"
#include "strongjit/optimize_cfg.h"
#include "strongjit/optimize_constants.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimize_nullconst.h"
#include "strongjit/optimizer_analysis.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/parser.h"
#include "strongjit/test_util.h"
#include "utils/matchers.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using ::testing::Eq;

void InitializePython() {
  static std::once_flag once;
  std::call_once(once, []() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });
}

PyCodeObject* CreateCodeObject(int64_t num_args = 0) {
  PyObject* empty_tuple = PyTuple_New(0);
  PyObject* varname_tuple = PyTuple_New(num_args);
  PyObject* empty_unicode = PyUnicode_FromString("");
  for (int64_t i = 0; i < num_args; ++i) {
    Py_INCREF(empty_unicode);
    PyTuple_SET_ITEM(varname_tuple, i, empty_unicode);
  }
  // Ensure bytes has some content so that the bytecode count is nonzero.
  PyObject* dummy_bytes = PyBytes_FromString("abcdefghijklm");
  return PyCode_New(
      /*argcount=*/num_args, /*kwonlyargcount=*/0, /*nlocals=*/num_args,
      /*stacksize=*/0, /*flags=*/0, /*code=*/dummy_bytes,
      /*consts=*/empty_tuple, /*names=*/empty_tuple,
      /*varnames=*/varname_tuple, /*freevars=*/empty_tuple,
      /*cellvars=*/empty_tuple, /*filename=*/empty_unicode,
      /*name=*/empty_unicode, /*firstlineno=*/0, /*lnotab=*/dummy_bytes);
}

TEST(OptimizeCallAttributeTest, Simple) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = frame_variable names, $1
  %2 = constant $5
  %3 = constant $42
  %4 = call_native PyObject_GetAttr (%3, %1) @2
  %5 = decref null? %3 @2
  %6 = bytecode_begin @4 stack [%4]
  %7 = call_python %4 (%2, %2) @4
  %8 = decref notnull %4 @6
  return %7
})";
  PyCodeObject code;
  InitializePython();
  code.co_names = PyTuple_New(2);
  PyTuple_SET_ITEM(code.co_names, 0, PyUnicode_InternFromString("foo"));
  PyTuple_SET_ITEM(code.co_names, 1, PyUnicode_InternFromString("bar"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  CreateCallAttributePattern create_call_attribute;
  S6_ASSERT_OK(RewritePatterns(f, &code, {&create_call_attribute}));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("arg"), "constant $5"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("obj"), "constant $42"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("res"), "call_attribute", lm.Ref("obj"),
                            ":: \"bar\" (", lm.Ref("arg"), lm.Ref("arg"),
                            "@2"));
  S6_ASSERT_OK(lm.OnNextLine("return", lm.Ref("res")));
}

TEST(OptimizeCallAttributeTest, NotUnicode) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = frame_variable names, $1
  %2 = constant $5
  %3 = constant $42
  %4 = call_native PyObject_GetAttr (%3, %1) @2
  %5 = constant $43
  %6 = call_python %4 (%5, %5) @4
  return %6
})";
  PyCodeObject code;
  InitializePython();
  code.co_names = PyTuple_New(4);
  PyTuple_SET_ITEM(code.co_names, 0, PyUnicode_InternFromString("foo"));
  PyTuple_SET_ITEM(code.co_names, 1, PyLong_FromLong(42));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  CreateCallAttributePattern create_call_attribute;
  S6_ASSERT_OK(RewritePatterns(f, &code, {&create_call_attribute}));
  LineMatcher lm(FormatOrDie(f));
  // The optimization cannot fire because the attribute was an integer type.
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("call_attribute"), "return"));
}

TEST(OptimizeCallAttributeTest, MultipleUses) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = frame_variable names, $1
  %2 = constant $5
  %3 = constant $42
  %4 = call_native PyObject_GetAttr (%3, %1) @2
  %5 = constant $43
  %6 = call_python %4 (%5, %4) @4
  %7 = decref notnull %4 @6
  return %6
})";
  PyCodeObject code;
  InitializePython();
  code.co_names = PyTuple_New(4);
  PyTuple_SET_ITEM(code.co_names, 0, PyUnicode_InternFromString("foo"));
  PyTuple_SET_ITEM(code.co_names, 1, PyUnicode_InternFromString("bar"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  CreateCallAttributePattern create_call_attribute;
  S6_ASSERT_OK(RewritePatterns(f, &code, {&create_call_attribute}));
  LineMatcher lm(FormatOrDie(f));
  // The optimization cannot fire because call_native had multiple uses.
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("call_attribute"), "return"));
}

TEST(OptimizeGetAttrTest, GetAttrFast) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "test_class"));
  AttributeDescription attr =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(Class * new_cls,
                          cls->Transition(attr, DictKind::kSplit));
  ASSERT_EQ(new_cls->id(), 2);

  absl::string_view input = R"(
type_feedback @0 monomorphic, test_class+x#2

function Simple {
&0:
  %1 = bytecode_begin @0
  %2 = frame_variable names, $0
  %3 = constant $5
  %4 = constant $42
  %5 = call_native PyObject_GetAttr (%4, %2, %2) @0
  %6 = decref null? %4 @2
  return %5
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();
  code->co_names = PyTuple_New(1);
  PyTuple_SET_ITEM(code->co_names, 0, PyUnicode_InternFromString("x"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(OptimizeGetSetAttr(f, code, {.mgr = mgr}));
  S6_EXPECT_OK(VerifyFunction(f));

  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("dict"), "get_object_dict %4"));
  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("id"), "get_instance_class_id", lm.Ref("dict")));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("check"), "cmp ne", lm.Ref("id")));
  S6_ASSERT_OK(lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("check")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("async"), "deoptimized_asynchronously?"));
  S6_ASSERT_OK(lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("async")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("load"), "load_from_dict", lm.Ref("dict"),
                             "$0", "split"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("load")));
}

TEST(OptimizeGetAttrTest, GetAttrMultiple) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "test_class"));
  AttributeDescription attr =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(Class * new_cls,
                          cls->Transition(attr, DictKind::kSplit));
  ASSERT_EQ(new_cls->id(), 2);
  absl::string_view input = R"(
type_feedback @0 monomorphic, test_class+x#2
type_feedback @2 monomorphic, test_class+x#2

function Simple {
&0:
  %1 = bytecode_begin @0
  %2 = constant $1
  %3 = frame_variable names, $0
  %4 = constant $5
  %5 = constant $42
  %6 = call_native PyObject_GetAttr (%5, %3, %3) @0
  %7 = bytecode_begin @2
  %8 = call_native PyObject_GetAttr (%4, %3, %3) @2
  %9 = decref null? %5 @2
  return %8
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();
  code->co_names = PyTuple_New(1);
  PyTuple_SET_ITEM(code->co_names, 0, PyUnicode_InternFromString("x"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(OptimizeGetSetAttr(f, code, {.mgr = mgr}));
  S6_EXPECT_OK(VerifyFunction(f));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine("load_from_dict"));
  S6_ASSERT_OK(lm.OnAnyLine("load_from_dict"));
}

TEST(OptimizeGetAttrTest, SetAttrFastPolymorphic) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "test_class"));
  AttributeDescription attr =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(Class * new_cls,
                          cls->Transition(attr, DictKind::kSplit));
  ASSERT_EQ(new_cls->id(), 2);

  S6_ASSERT_OK_AND_ASSIGN(Class * cls2, Class::Create(mgr, "unrelated_class"));
  S6_ASSERT_OK_AND_ASSIGN(Class * new_cls2,
                          cls2->Transition(attr, DictKind::kSplit));
  ASSERT_EQ(new_cls2->id(), 4);

  absl::string_view input = R"(
type_feedback @0 polymorphic, either test_class+x#2 or unrelated_class+x#4

function Simple {
&0:
  %1 = bytecode_begin @0
  %2 = constant $1
  %3 = frame_variable names, $0
  %4 = constant $5
  %5 = constant $42
  %6 = call_native PyObject_SetAttr (%5, %3, %4, %3) @0
  %7 = decref null? %5 @2
  return %6
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();
  code->co_names = PyTuple_New(1);
  PyTuple_SET_ITEM(code->co_names, 0, PyUnicode_InternFromString("x"));

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(OptimizeGetSetAttr(f, code, {.mgr = mgr}));
  S6_EXPECT_OK(VerifyFunction(f));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("dict"), "get_object_dict %5"));
  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("id"), "get_instance_class_id", lm.Ref("dict")));
  S6_ASSERT_OK(lm.OnAnyLine("br", DefBlock("yes"), DefBlock("no")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("yes")));
  S6_ASSERT_OK(lm.OnNextLine("deoptimized_asynchronously?"));
  S6_ASSERT_OK(lm.OnNextLine("deoptimize_if_safepoint"));
  S6_ASSERT_OK(lm.OnAnyLine("incref", "%4"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("yesv"), "store_to_dict", "%4 into",
                             lm.Ref("dict"), "$0", "split"));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("yesv")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", DefBlock("out")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("no")));
  S6_ASSERT_OK(lm.OnAnyLine("incref", "%4"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("yes2v"), "store_to_dict", "%4 into",
                             lm.Ref("dict"), "$0", "split"));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("yes2v")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", lm.Ref("out")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("out")));
}

void SetUpBoxClasses(ClassManager& mgr) {
  S6_ASSERT_OK_AND_ASSIGN(Class * int_cls,
                          Class::Create(mgr, "int", &PyLong_Type));
  ASSERT_EQ(int_cls->id(), 1);
  S6_ASSERT_OK_AND_ASSIGN(Class * float_cls,
                          Class::Create(mgr, "float", &PyFloat_Type));
  ASSERT_EQ(float_cls->id(), 2);
  S6_ASSERT_OK_AND_ASSIGN(Class * bool_cls,
                          Class::Create(mgr, "bool", &PyBool_Type));
  ASSERT_EQ(bool_cls->id(), 3);
}

absl::Status ApplyUnboxingOptimization(ClassManager& mgr, Function& f,
                                       PyCodeObject* code) {
  return RewritePatterns<UnboxPyNumberOpsPattern, BypassBoxUnboxPattern,
                         RemoveUnusedBoxOpPattern>(
      f, code, {.mgr = mgr, .enable_unboxing_optimization = true});
}

TEST(OptimizePyNumberOpUnboxingTest, UnaryOpIsUnboxed) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, bool#3

function Not {
&0:
  %1 = constant $89
  bytecode_begin @0
  %3 = call_native PyNumber_Invert (%1) @0
  decref notnull %1 @0
  %5 = constant $0
  %6 = cmp eq i64 %3, %5
  br %6 &8 &10
&8:
  unreachable
&10:
  return %3
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("arg"), "constant $89"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Input should be unboxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_arg"), "unbox bool", lm.Ref("arg")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_arg"), "overflowed?", lm.Ref("unbox_arg")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_arg"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_res"), "not i64", lm.Ref("unbox_arg")));
  // Output should be boxed.
  // Unlike other Boolean ops, PyNumber_Invert evaluates to a long when
  // applied to Boolean inputs.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box long", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("arg"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, BinaryOpIsUnboxed) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, int#1

function Subtract {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_Subtract (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Inputs should be unboxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox long", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox long", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));

  // Perform the subtraction, with overflow checking.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "subtract",
                             lm.Ref("unbox_lhs"), lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_res"), "overflowed?", lm.Ref("unbox_res")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_res"), "@0"));
  // Output should be boxed.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box long", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, DivisionByZeroDeoptimizes) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, int#1

function Divide {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_Remainder (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Inputs should be unboxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox long", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox long", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));

  // Perform the remainder operation, checking for division by zero.
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint not", lm.Ref("unbox_rhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "remainder",
                             lm.Ref("unbox_lhs"), lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_res"), "overflowed?", lm.Ref("unbox_res")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_res"), "@0"));
  // Output should be boxed.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box long", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, TrueDivideConvertsToFloat) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, int#1

function Divide {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_TrueDivide (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Inputs should be unboxed and converted from long to float.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox long", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox long", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("float_lhs"), "int_to_float",
                             lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("float_rhs"), "int_to_float",
                             lm.Ref("unbox_rhs")));

  // Perform the floating point division, checking for division by zero.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("is_zero"), "float_zero?", lm.Ref("float_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("is_zero"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "divide f64",
                             lm.Ref("float_lhs"), lm.Ref("float_rhs")));
  // Output should be boxed as a float.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("res"), "box float", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, NegativeShiftDeoptimizes) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, int#1

function Divide {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_Rshift (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Inputs should be unboxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox long", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox long", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));

  // Perform the right-shift, checking for a negative shift.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("zero"), "constant $0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("is_neg"), "cmp lt i64",
                             lm.Ref("unbox_rhs"), lm.Ref("zero")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("is_neg"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "shift_right_signed",
                             lm.Ref("unbox_lhs"), lm.Ref("unbox_rhs")));
  // Output should be boxed.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box long", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, ComparisonOpIsUnboxed) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, float#2

function LessOrEqual {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = constant $1
  %5 = call_native PyObject_RichCompare (%1, %2, %4) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %8 = constant $0
  %9 = cmp eq i64 %5, %8
  br %9, &11 &13
&11:
  unreachable
&13:
  return %5
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Operands should be unboxed.
  // However, the comparison op (specified as the third argument) should not.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox float", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox float", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "cmp le f64",
                             lm.Ref("unbox_lhs"), ",", lm.Ref("unbox_rhs")));
  // Output should be boxed as a Boolean.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box bool", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, ArithmeticOpOnBooleansIsUnboxed) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, bool#3

function Subtract {
&0:
  %1 = constant $0
  %2 = constant $1
  bytecode_begin @0
  %4 = call_native PyNumber_Subtract (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $1"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  // Operands should be unboxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox bool", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox bool", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));
  // Perform the subtraction, with overflow checking.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "subtract",
                             lm.Ref("unbox_lhs"), lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_res"), "overflowed?", lm.Ref("unbox_res")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_res"), "@0"));
  // Output should be boxed as a Long, not a Boolean.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("res"), "box long", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, PowerIsSelfMultiply) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, float#2

function Square {
&0:
  %1 = constant $33
  %2 = constant $2
  bytecode_begin @0
  %4 = constant $0
  %5 = call_native PyNumber_Power (%1, %2, %4) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %8 = constant $0
  %9 = cmp eq i64 %5, %8
  br %9 &11 &13
&11:
  unreachable
&13:
  return %5
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $33"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $2"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox float", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox long", lm.Ref("rhs")));
  // Check that the second operand is 2. We currently only support squaring.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("two"), "constant $2"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("is_not_two"), "cmp ne i64",
                             lm.Ref("unbox_rhs"), ",", lm.Ref("two")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("is_not_two"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "multiply f64",
                             lm.Ref("unbox_lhs"), ",", lm.Ref("unbox_lhs")));
  // Final output should be boxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("res"), "box float", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", "notnull", lm.Ref("lhs"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, MegamorphicOpIsNotUnboxed) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 megamorphic

function Subtract {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_Subtract (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  // No unboxing/boxing expected.
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("box"), "return"));
}

TEST(OptimizePyNumberOpUnboxingTest, CompositionHasNoIntermediateBoxing) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, float#2
type_feedback @2 monomorphic, float#2

function NegativeAdd {
&0:
  %1 = constant $55
  bytecode_begin @0
  %3 = call_native PyNumber_Negative (%1) @0
  decref notnull %1 @0
  %5 = constant $0
  %6 = cmp eq i64 %3, %5
  br %6 &8 &10
&8:
  unreachable
&10:
  bytecode_begin @1
  %12 = constant $34
  bytecode_begin @2 stack [%3]
  %14 = call_native PyNumber_Add (%3, %12) @2
  decref notnull %3 @2
  decref notnull %12 @2
  %17 = constant $0
  %18 = cmp eq i64 %14, %17
  br %18 &20 &22
&20:
  unreachable
&22:
  return %14
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox float", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_neg"), "negate f64", lm.Ref("unbox_lhs")));
  // Instead of a "box" op, we call the Python version on the slow path.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("remat_neg"),
                             "rematerialize PyFloat_FromDouble",
                             lm.Ref("unbox_neg")));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("lhs"), "@0"));

  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(
      lm.OnAnyLine("bytecode_begin", "@2", "stack", lm.Ref("remat_neg")));
  // Instead of unboxing the negated lhs, use its float value directly.
  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("unbox_rhs"), "unbox float", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@2"));
  // First argument to Add should be the output of Negate,
  // without any intermediate boxing.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_res"), "add f64",
                             lm.Ref("unbox_neg"), ",", lm.Ref("unbox_rhs")));
  // Final output should be boxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("res"), "box float", lm.Ref("unbox_res")));
  // Only decref rhs. The first input of 'add' was never boxed.
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@2"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, UnaryPositiveIsNoOp) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, float#2
type_feedback @2 monomorphic, float#2

function PositiveAdd {
&0:
  %1 = constant $55
  %2 = constant $34
  bytecode_begin @0
  %4 = call_native PyNumber_Positive (%2) @0
  decref notnull %2 @0
  %6 = constant $0
  %7 = cmp eq i64 %4, %6
  br %7 &9 &11
&9:
  unreachable
&11:
  bytecode_begin @2 stack [%4]
  %13 = call_native PyNumber_Add (%1, %4) @2
  decref notnull %1 @2
  decref notnull %4 @2
  %16 = constant $0
  %17 = cmp eq i64 %13, %16
  br %17 &19 &21
&19:
  unreachable
&21:
  return %13
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_rhs"), "unbox float", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@0"));
  // Instead of a "box" op, we call the Python version on the slow path.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("remat_pos"),
                             "rematerialize PyFloat_FromDouble",
                             lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@0"));
  S6_ASSERT_OK(
      lm.OnAnyLine("bytecode_begin", "@2", "stack", lm.Ref("remat_pos")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs"), "unbox float", lm.Ref("lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs"), "overflowed?", lm.Ref("unbox_lhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs"), "@2"));
  // Second argument to Add should be the rhs input,
  // without any opcode generated for the "unary positive".
  // It's already unboxed; use its float value directly.
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("unbox_res"), "add f64",
                            lm.Ref("unbox_lhs"), ",", lm.Ref("unbox_rhs")));
  // Final output should be boxed.
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("res"), "box float", lm.Ref("unbox_res")));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("lhs"), "@2"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("res")));
}

TEST(OptimizePyNumberOpUnboxingTest, ReusedIntermediateRetainsBoxing) {
  ClassManager mgr;
  SetUpBoxClasses(mgr);

  absl::string_view input = R"(
type_feedback @0 monomorphic, bool#3
type_feedback @2 monomorphic, bool#3

function XorWithIgnoredOr {
&0:
  %1 = constant $55
  %2 = constant $89
  bytecode_begin @0
  %4 = call_native PyNumber_Xor (%1, %2) @0
  decref notnull %1 @0
  decref notnull %2 @0
  %7 = constant $0
  %8 = cmp eq i64 %4, %7
  br %8 &10 &12
&10:
  unreachable
&12:
  incref null? %4
  %14 = constant $34
  bytecode_begin @2 stack [%4]
  %16 = call_native PyNumber_Or (%4, %14) @2
  decref notnull %4 @2
  decref notnull %14 @2
  %19 = constant $0
  %20 = cmp eq i64 %16, %19
  br %20 &22 &24
&22:
  unreachable
&24:
  %25 = constant $1
  br %25 &27 &29
&27:
  return %4
&29:
  return %16
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input, mgr));
  S6_ASSERT_OK(ApplyUnboxingOptimization(mgr, f, code));
  LineMatcher lm(FormatOrDie(f, PredecessorAnnotator(), mgr));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs1"), "constant $55"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("lhs2"), "constant $89"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs1"), "unbox bool", lm.Ref("lhs1")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs1"), "overflowed?", lm.Ref("unbox_lhs1")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs1"), "@0"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("unbox_lhs2"), "unbox bool", lm.Ref("lhs2")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_lhs2"), "overflowed?", lm.Ref("unbox_lhs2")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_lhs2"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_xor"), "xor i64",
                             lm.Ref("unbox_lhs1"), lm.Ref("unbox_lhs2")));
  // This output will be returned, so it should be boxed.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("xor"), "box bool", lm.Ref("unbox_xor")));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("lhs1"), "@0"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("lhs2"), "@0"));

  S6_ASSERT_OK(lm.OnAnyLine("incref", "null?", lm.Ref("xor")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("rhs"), "constant $34"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@2", "stack", lm.Ref("xor")));
  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("unbox_rhs"), "unbox bool", lm.Ref("rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("of_rhs"), "overflowed?", lm.Ref("unbox_rhs")));
  S6_ASSERT_OK(
      lm.OnNextLine("deoptimize_if_safepoint", lm.Ref("of_rhs"), "@2"));
  // First argument to Or should be the output of Not,
  // bypassing the intermediate boxing.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("unbox_or"), "or i64",
                             lm.Ref("unbox_xor"), ",", lm.Ref("unbox_rhs")));
  // Final output should be boxed.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("or"), "box bool", lm.Ref("unbox_or")));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("xor"), "@2"));
  S6_ASSERT_OK(lm.OnNextLine("decref", "notnull", lm.Ref("rhs"), "@2"));

  // Return the intermediate result extracted earlier.
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("xor")));
  // On the other branch, return the final result.
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("or")));
}

TEST(EliminateDeadCodeTest, Simple) {
  // Given a strongJIT function
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $1
  %2 = constant $2
  %3 = constant $3
  %4 = add i64 %1, %2
  %5 = subtract i64 %2, %3
  return %4
})";

  InitializePython();
  PyCodeObject* code = CreateCodeObject();
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));

  // When unused instructions are removed.
  S6_ASSERT_OK(RewritePatterns(f, code, {}));

  // Then unused instructions and their unused inputs are removed.
  /* Optimized function:
    %1 = constant $1
    %2 = constant $2
    %3 = add i64 %1, %2
    return %3
  */
  std::string optimized_jit = FormatOrDie(f);

  {
    LineMatcher lm(optimized_jit);
    S6_ASSERT_OK(
        lm.OnAnyLine(lm.WithoutSeeing("subtract i64 %2, %3"), "return %3"));
  }

  {
    LineMatcher lm(optimized_jit);
    S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("constant $3"), "return %3"));
  }

  // And used instructions are preserved.
  {
    LineMatcher lm(optimized_jit);
    S6_ASSERT_OK(lm.OnAnyLine("constant $1"));
    S6_ASSERT_OK(lm.OnNextLine("constant $2"));
    S6_ASSERT_OK(lm.OnNextLine("add i64 %1, %2"));
    S6_ASSERT_OK(lm.OnNextLine("return %3"));
  }
}

TEST(RemoveUnusedBlockArgumentsTest, Simple) {
  // Test repeated use of the same argument to robustly test argument removal.
  absl::string_view input = R"(function UBA {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $101
  jmp &4 [ %1, %2, %2 ]

&4: [ %5, %6, %7 ]                                          // preds: &0
  return %7
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  S6_ASSERT_OK(RewritePatterns<EliminateUnusedBlockArgumentsPattern>(f, code));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function UBA {
&0:                                                         // entry point
  %1 = constant $101
  jmp &3 [ %1 ]

&3: [ %4 ]                                                  // preds: &0
  return %4
})")));
}

TEST(SimplifyOrphanedBlocksTest, Simple) {
  absl::string_view input = R"(function Simple {
&0:                                                         // entry point
  %1 = constant $42
  return %1

&3:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  S6_ASSERT_OK(RewritePatterns(f, code, {}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(SimplifyOrphanedBlocksTest, Recursive) {
  absl::string_view input = R"(function Recursive {
&0:                                                         // entry point
  %1 = constant $42
  jmp &7

&3:
  jmp &5

&5:
  return %1

&7:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  S6_ASSERT_OK(RewritePatterns(f, code, {}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Recursive {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3

&3:                                                         // preds: &0
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, Simple) {
  absl::string_view input = R"(function Simple {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3

&3:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  // Exercise the variant of RewritePatterns that can flatten a tuple.
  using TupleType = std::tuple<EliminateTrivialJumpsPattern>;
  S6_ASSERT_OK(RewritePatterns<TupleType>(f, code));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, Recursive) {
  absl::string_view input = R"(function Recursive {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3

&3:
  jmp &5

&5:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Recursive {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, ExceptIsntRemoved) {
  absl::string_view input = R"(function ExceptIsntRemoved {
&0:                                                         // entry point
  %1 = constant $42
  except &3 @0

&3: except [ %4, %5, %6, %7, %8, %9 ]
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)),
              Eq(RawString(R"(function ExceptIsntRemoved {
&0:                                                         // entry point
  %1 = constant $42
  except &3 @0

&3: except [ %4, %5, %6, %7, %8, %9 ]                       // preds: &0
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, EmptyBlockThreadsThroughExcept) {
  absl::string_view input = R"(function EBTTE {
&0:                                                         // entry point
  %1 = constant $42
  except &3 @0

&3: except [ %4, %5, %6, %7, %8, %9 ]
  jmp &11

&11:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function EBTTE {
&0:                                                         // entry point
  %1 = constant $42
  except &3 @0

&3: except [ %4, %5, %6, %7, %8, %9 ]                       // preds: &0
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, EmptyBlockThreadsThroughBr) {
  absl::string_view input = R"(function EBTTB {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5

&3:
  jmp &5

&5:
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function EBTTB {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &3

&3:                                                         // preds: &0
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, BlockArgumentsInhibits1) {
  absl::string_view input = R"(function BAI1 {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5 [ %1 ]

&3:
  jmp &5 [ %1 ]

&5: [ %6 ]
  return %6
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function BAI1 {
&0:                                                         // entry point
  %1 = constant $42
  br %1, &3, &5 [ %1 ]

&3:                                                         // preds: &0
  jmp &5 [ %1 ]

&5: [ %6 ]                                                  // preds: &0, &3
  return %6
})")));
}

TEST(SimplifyControlFlowEdgesTest, BlockArgument2) {
  absl::string_view input = R"(function BAI2 {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3

&3:
  jmp &5 [ %1 ]

&5: [ %6 ]
  return %6
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function BAI2 {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(SimplifyControlFlowEdgesTest, BlockArguments3) {
  absl::string_view input = R"(function BAI3 {
&0:                                                         // entry point
  %1 = constant $42
  jmp &3 [ %1 ]

&3: [ %4 ]
  jmp &6

&6:
  return %4
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  EliminateTrivialJumpsPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function BAI3 {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(ConstantFolding, ComparisonEq) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  %3 = cmp eq i64 %1 %2
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldCompareInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $1
  return %1
})")));
}

TEST(ConstantFolding, ComparisonLt) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $42
  %2 = constant $42
  %3 = cmp lt i64 %1 %2
  return %3
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldCompareInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $0
  return %1
})")));
}

TEST(ConstantFolding, BrInst) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $1
  br %1, &3 [% 1], &6 [%1, %1]

&3: [ %4]
  return %4

&6: [%7, %8]
  return %7
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldBrInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $1
  jmp &3 [ %1 ]

&3: [ %4 ]                                                  // preds: &0
  return %4
})")));
}

TEST(ConstantFolding, UboxBool) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $0
  %2 = unbox bool %1
  return %2
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  cast<ConstantInst>(*f.entry().begin())
      .set_value(reinterpret_cast<int64_t>(Py_True));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldUnboxPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $1
  return %1
})")));
}

TEST(ConstantFolding, UboxLong) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $0
  %2 = unbox long %1
  return %2
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  cast<ConstantInst>(*f.entry().begin())
      .set_value(reinterpret_cast<int64_t>(PyLong_FromLong(42)));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldUnboxPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $42
  return %1
})")));
}

TEST(ConstantFolding, RemoveDeoptimizeIfSafepoint) {
  absl::string_view input = R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $0
  deoptimize_if_safepoint %1 @0 ""
  return %1
})";
  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  PyCodeObject* code = CreateCodeObject();
  ConstantFoldDeoptimizeIfSafepointInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Cmp {
&0:                                                         // entry point
  %1 = constant $0
  return %1
})")));
}

static absl::Status OptimizeNullConst(Function& f, PyCodeObject* code,
                                      OptimizerOptions options) {
  LiveInfo live_info(f);

  analysis::Manager manager(f, code, live_info, nullconst::Analysis());
  S6_ASSIGN_OR_RETURN(auto result, manager.Analyze());

  Rewriter rewriter(f, code, options);
  analysis::RewriteAnalysis(rewriter, live_info, nullconst::Rewriter{}, result);

  return {};
}

TEST(NullConstTest, ConstantFolding) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $3
  %2 = constant $4
  %3 = add i64 %1 %2
  return %3
})";
  InitializePython();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(OptimizeNullConst(f, CreateCodeObject(), {}));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("seven"), "constant $7"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("seven")));
}

TEST(NullConstTest, RemoveDeoptimizeIf) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $0
  jmp &3 [ %1 ]

&3 : [ %4 ]
  deoptimize_if_safepoint %4 @0 ""
  return %4
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(OptimizeNullConst(f, code, {}));

  ConstantFoldDeoptimizeIfSafepointInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $0
  jmp &3 [ %1 ]

&3: [ %4 ]                                                  // preds: &0
  %5 = constant $0
  return %5
})")));
}

TEST(NullConstTest, RemoveBr) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $0
  jmp &3 [ %1 ]

&3: [ %4 ]
  br %4, &6 [ %4 ], &9

&6 : [ %7 ]
  return %4
&9:
  return %1
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(OptimizeNullConst(f, code, {}));

  ConstantFoldBrInstPattern p;
  S6_ASSERT_OK(RewritePatterns(f, code, {&p}));
  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $0
  jmp &3 [ %1 ]

&3: [ %4 ]                                                  // preds: &0
  jmp &6

&6:                                                         // preds: &3
  return %1
})")));
}

TEST(NullConstTest, DecrefNull) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $3
  decref null? %1 @0
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(OptimizeNullConst(f, code, {}));

  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $3
  decref notnull %1 @0
})")));
}

TEST(NullConstTest, BrWithCmpToZero) {
  absl::string_view input = R"(function Simple {
&0:
  %1 = constant $0
  %2 = constant $42
  %3 = load i64 [ %2 + $0 ]
  %4 = cmp eq i64 %1 %3
  br %4, &6, &9
&6:
  decref null? %3 @0
  return %1
&9:
  decref null? %3 @0
  return %1
})";
  InitializePython();
  PyCodeObject* code = CreateCodeObject();

  S6_ASSERT_OK_AND_ASSIGN(Function f, ParseFunction(input));
  S6_ASSERT_OK(OptimizeNullConst(f, code, {}));
  S6_ASSERT_OK(RewritePatterns(f, code, {}));

  ASSERT_THAT(RawString(FormatOrDie(f)), Eq(RawString(R"(function Simple {
&0:                                                         // entry point
  %1 = constant $0
  %2 = constant $42
  %3 = load i64 [ %2 + $0 ]
  %4 = cmp eq i64 %1, %3
  br %4, &6, &8

&6:                                                         // preds: &0
  return %1

&8:                                                         // preds: &0
  decref notnull %3 @0
  return %1
})")));
}

}  // namespace
}  // namespace deepmind::s6
