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

#include "strongjit/ingestion.h"

#include <Python.h>
#include <opcode.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "strongjit/formatter.h"
#include "strongjit/test_util.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

TEST(IngestionTest, ReturnConst) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 1},
      {2, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "ReturnConst", 0, 0));
  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("load"), "= frame_variable consts, $1"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull", lm.Ref("load")));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @2 stack [", lm.Ref("load"), "]"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("load")));
}

TEST(IngestionTest, CallPython) {
  // def f(x):
  //    return x()
  BytecodeInstruction program[] = {
      {0, LOAD_FAST, 0},  // (x)
      {2, CALL_FUNCTION, 0},
      {4, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "CallPython", 1, 1));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "fastlocals", DefValue("arg")));

  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("py_call"), "= call_python", lm.Ref("arg")));

  // If it was zero we should except. We should decref the argument on all
  // paths.
  S6_ASSERT_OK(lm.OnAnyLine("decref", "null?", lm.Ref("arg")));
  S6_ASSERT_OK(lm.OnNextLine("except"));

  S6_ASSERT_OK(lm.OnAnyLine("decref", "null?", lm.Ref("arg")));
  S6_ASSERT_OK(lm.OnNextLine("return", lm.Ref("py_call")));
}

TEST(IngestionTest, BinaryDivide) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 1},
      {2, DUP_TOP, 0},
      {4, BINARY_FLOOR_DIVIDE, 0},
      {6, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "BinaryDivide", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("v"), "= frame_variable consts"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull", lm.Ref("v")));
  // The DUP_TOP must also increase the refcount.
  S6_ASSERT_OK(lm.OnAnyLine("incref null?", lm.Ref("v")));
  S6_ASSERT_OK(lm.OnAnyLine("call_native PyNumber_FloorDivide", lm.Ref("v"),
                            lm.Ref("v")));
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("v")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("v")));
}

TEST(IngestionTest, PopJumpIfFalse) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},
      {2, POP_JUMP_IF_FALSE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "PopJumpIfFalse", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  // Find the target of the jump. This will be the entry block.
  S6_ASSERT_OK(lm.OnAnyLine(DefBlock("target"), "// entry point"));

  // Find the pushed item.
  S6_ASSERT_OK(lm.OnAnyLine("incref", DefValue("item")));

  // The stack item must be popped (decreffed) prior to every jump.
  // Py_True -> fallthrough
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", DefBlock("fallthrough")));

  // Py_False -> target
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", lm.Ref("target")));

  S6_ASSERT_OK(lm.OnAnyLine("call_native PyObject_IsTrue"));

  // IsTrue -> fallthrough
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", lm.Ref("fallthrough")));

  // !IsTrue -> target
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item")));
  S6_ASSERT_OK(lm.OnNextLine("jmp", lm.Ref("target")));

  // Otherwise except.
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item")));
  S6_ASSERT_OK(lm.OnNextLine("except"));
}

TEST(IngestionTest, JumpIfTrueOrPop) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},   {2, LOAD_CONST, 0}, {4, JUMP_IF_TRUE_OR_POP, 8},
      {6, RETURN_VALUE, 0}, {8, LOAD_CONST, 1},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "PopJumpIfFalse", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  // Find the pushed item. This will be the second incremented value (the second
  // LOAD_CONST).
  S6_ASSERT_OK(lm.OnAnyLine("incref", DefValue("return_value")));
  S6_ASSERT_OK(lm.OnAnyLine("incref", DefValue("item")));

  // Skip to the start of the JUMP_IF_TRUE_OR_POP, after the LOAD_CONST.
  S6_ASSERT_OK(lm.OnAnyLine(AnyBlock(), ":"));

  // The stack item must be popped (decreffed) only if the branch is NOT taken.
  // Py_True -> target
  S6_ASSERT_OK(
      lm.OnAnyLine(lm.WithoutSeeing("decref"), "jmp", DefBlock("target")));

  // Py_False -> fallthrough. Also check the bytecode annotations.
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item"), "@4"));
  S6_ASSERT_OK(lm.OnNextLine("jmp", DefBlock("fallthrough")));

  S6_ASSERT_OK(lm.OnAnyLine("call_native PyObject_IsTrue", "@4"));

  // IsTrue -> target
  S6_ASSERT_OK(
      lm.OnAnyLine(lm.WithoutSeeing("decref"), "jmp", lm.Ref("target")));

  // !IsTrue -> fallthrough
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item"), "@4"));
  S6_ASSERT_OK(lm.OnNextLine("jmp", lm.Ref("fallthrough")));

  // Otherwise except.
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("item"), "@4"));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("return_value"), "@4"));
  S6_ASSERT_OK(lm.OnNextLine("except"));
}

TEST(IngestionTest, LoadGlobal) {
  BytecodeInstruction program[] = {
      {0, LOAD_GLOBAL, 0},
      {2, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "LoadGlobal", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  // load_global returns a borrowed reference to the looked up global object,
  // or 0 if there was an exception.
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("result"), "load_global"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero"), "constant $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero_test"), "cmp eq", lm.Ref("result"),
                            lm.Ref("zero")));

  // If this was zero we go to `finish`. Otherwise we're on the painful road to
  // exceptions.
  S6_ASSERT_OK(lm.OnAnyLine("br", lm.Ref("zero_test"), DefBlock("exception"),
                            DefBlock("ok")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("exception"), ":"));
  S6_ASSERT_OK(lm.OnNextLine("except"));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("ok"), ":"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull", lm.Ref("result")));
}

TEST(IngestionTest, LoadName) {
  BytecodeInstruction program[] = {
      {0, LOAD_NAME, 0},
      {2, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f, IngestProgram(program, "LoadName", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("name"), "frame_variable names"));

  // Does locals exist?
  S6_ASSERT_OK(lm.OnAnyLine("call_native PyErr_Format"));
  S6_ASSERT_OK(lm.OnNextLine("except"));

  // It exists, is it a dict?
  S6_ASSERT_OK(lm.OnAnyLine("br", AnyValue(), DefBlock("is_dict"),
                            DefBlock("not_dict")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("is_dict"), ":"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("dictval"), "call_native PyDict_GetItem"));
  S6_ASSERT_OK(lm.OnNextLine("jmp", DefBlock("merge"), lm.Ref("dictval")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("not_dict"), ":"));
  S6_ASSERT_OK(
      lm.OnNextLine(DefValue("objval"), "call_native PyObject_GetItem"));
  // Skip checking the exception path, just check that we merge with objval.
  S6_ASSERT_OK(lm.OnAnyLine("jmp", DefBlock("merge"), lm.Ref("objval")));

  // Merge the two paths and check globals.
  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("merge"), ":", DefValue("localval")));
  // If not zero, branch to real_finish.
  S6_ASSERT_OK(lm.OnAnyLine("br", AnyBlock(), DefBlock("real_finish"),
                            lm.Ref("localval")));

  S6_ASSERT_OK(lm.OnAnyLine(DefValue("load_global"), "load_global"));

  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero"), "constant $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero_test"), "cmp eq",
                            lm.Ref("load_global"), lm.Ref("zero")));

  // If this was zero we go to `finish`. Otherwise we're on the painful road to
  // exceptions.
  S6_ASSERT_OK(lm.OnAnyLine("br", lm.Ref("zero_test"),
                            DefBlock("load_global_exception"),
                            DefBlock("load_global_finish")));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("load_global_exception"), ":"));
  S6_ASSERT_OK(lm.OnNextLine("except"));

  S6_ASSERT_OK(lm.OnAnyLine(lm.Ref("load_global_finish"), ":"));

  S6_ASSERT_OK(lm.OnAnyLine("jmp", lm.Ref("load_global")));

  S6_ASSERT_OK(
      lm.OnAnyLine(lm.Ref("real_finish"), ":", DefValue("merged_value")));
  S6_ASSERT_OK(lm.OnNextLine("incref", lm.Ref("merged_value")));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("merged_value")));
}

TEST(IngestionTest, MakeFunction) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},
      {2, LOAD_CONST, 1},
      {4, MAKE_FUNCTION, 0},
      {6, RETURN_VALUE, 0},
  };
  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "MakeFunction", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("code"), "frame_variable consts, $0"));
  S6_ASSERT_OK(
      lm.OnAnyLine(DefValue("qualified_name"), "frame_variable consts, $1"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("globals"), "frame_variable globals"));
  S6_ASSERT_OK(lm.OnAnyLine(
      DefValue("ret"), "call_native PyFunction_NewWithQualName", lm.Ref("code"),
      lm.Ref("globals"), lm.Ref("qualified_name")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("qualified_name")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("code")));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("ret")));
}

TEST(IngestionTest, RichCompare) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},
      {2, LOAD_CONST, 0},
      {4, COMPARE_OP, PyCmp_LT},
      {6, RETURN_VALUE, 0},
  };
  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "RichCompare", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @4"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero"), "constant $0"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("ret"),
                             "call_native PyObject_RichCompare",
                             DefValue("lhs"), DefValue("rhs"), lm.Ref("zero")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("lhs")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("rhs")));
  S6_ASSERT_OK(lm.OnAnyLine("except @4"));
  S6_ASSERT_OK(lm.OnAnyLine("return", lm.Ref("ret")));
}

TEST(IngestionTest, StoreFast) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},
      {2, STORE_FAST, 0},
      {4, LOAD_CONST, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "StoreFast", 1, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("zero"), "constant $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("v"), "frame_variable consts, $0"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull", lm.Ref("v")));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@2", "stack", lm.Ref("v"),
                            "fastlocals", lm.Ref("zero")));
  S6_ASSERT_OK(lm.OnAnyLine("decref null?", lm.Ref("zero")));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin", "@4", "fastlocals", lm.Ref("v")));
}

TEST(IngestionTest, DeleteFast) {
  BytecodeInstruction program[] = {
      {0, DELETE_FAST, 0},
      {2, LOAD_CONST, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "DeleteFast", 1, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @0 fastlocals", DefValue("prev")));
  S6_ASSERT_OK(lm.OnNextLine("deoptimize_if_safepoint not", lm.Ref("prev"),
                             "@0 fastlocals [", lm.Ref("prev"), "]"));

  // Find the nullptr that the fastlocal has been set to.
  S6_ASSERT_OK(lm.OnNextLine(DefValue("zero2"), "constant $0"));
  S6_ASSERT_OK(lm.OnAnyLine("decref notnull", lm.Ref("prev")));
  S6_ASSERT_OK(
      lm.OnAnyLine("bytecode_begin @2", "fastlocals", lm.Ref("zero2")));
}

TEST(IngestionTest, StoreSubscr) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 2},
      {2, LOAD_CONST, 0},
      {4, LOAD_CONST, 1},
      {6, STORE_SUBSCR, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "StoreSubscr", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("val"), "frame_variable consts, $2"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("obj"), "frame_variable consts, $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("key"), "frame_variable consts, $1"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @6"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("ret"), "call_native PyObject_SetItem",
                             lm.Ref("obj"), lm.Ref("key"), lm.Ref("val")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("key")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("obj")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("val")));
  S6_ASSERT_OK(lm.OnAnyLine("except @6"));
}

TEST(IngestionTest, DeleteSubscr) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 0},
      {2, LOAD_CONST, 1},
      {4, DELETE_SUBSCR, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "DeleteSubscr", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("obj"), "frame_variable consts, $0"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("key"), "frame_variable consts, $1"));
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @4"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("ret"), "call_native PyObject_DelItem",
                             lm.Ref("obj"), lm.Ref("key")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("key")));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("obj")));
  S6_ASSERT_OK(lm.OnAnyLine("except @4"));
}

TEST(IngestionTest, PopBlock) {
  // clang-format off
  BytecodeInstruction program[] = {
    {0, SETUP_EXCEPT, 10},                         // (to 12)
    {2, LOAD_CONST, 1},
    {4, DUP_TOP, 0},
    {6, POP_BLOCK, 0},
    {8, LOAD_CONST, 1},
    {10, RETURN_VALUE, 0},
    {12, LOAD_CONST, 1},
    {14, RETURN_VALUE, 0}};
  // clang-format on

  S6_ASSERT_OK_AND_ASSIGN(Function f, IngestProgram(program, "PopBlock", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("v"), "frame_variable consts"));
  // The POP_BLOCK at @6 should decref both values on the stack.
  S6_ASSERT_OK(lm.OnAnyLine("decref", lm.Ref("v"), "@6"));
  S6_ASSERT_OK(lm.OnNextLine("decref", lm.Ref("v"), "@6"));
}

TEST(IngestionTest, Loop) {
  // Generated from:
  //   def f(x, y):
  //     for i in x:
  //       y(i)
  // clang-format off
  BytecodeInstruction program[] = {
    {0, SETUP_LOOP, 20},                         // (to 22)
    {2, LOAD_FAST, 0},                           // (x)
    {4, GET_ITER, 0},
    {6, FOR_ITER, 12},                           // (to 20)
    {8, STORE_FAST, 2},                          // (i)
    {10, LOAD_FAST, 1},                          // (y)
    {12, LOAD_FAST, 2},                          // (i)
    {14, CALL_FUNCTION, 1},
    {16, STORE_FAST, 3},
    {18, JUMP_ABSOLUTE, 6},
    {20, POP_BLOCK, 0},
    {22, LOAD_FAST, 3},
    {24, RETURN_VALUE, 0}};
  // clang-format on

  S6_ASSERT_OK_AND_ASSIGN(Function f, IngestProgram(program, "Loop", 4, 2));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("iter"), "PyObject_GetIter", "@4"));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("next"), "call_native", "s6::IteratorNext",
                            "(", lm.Ref("iter"), ")"));

  // Ensure "next" is stored into fastlocals.
  S6_ASSERT_OK(lm.OnAnyLine("bytecode_begin @10", "fastlocals", AnyValue(),
                            AnyValue(), lm.Ref("next")));
}

TEST(IngestionTest, UnpackSequence) {
  // Generated from:
  //   def f(l):
  //     x, y, z = l
  // clang-format off
  BytecodeInstruction program[] = {
    {0, LOAD_FAST, 0},                           // (l)
    {2, UNPACK_SEQUENCE, 3},
    {4, STORE_FAST, 1},                          // (x)
    {6, STORE_FAST, 2},                          // (y)
    {8, STORE_FAST, 3},                          // (z)
    {10, LOAD_CONST, 0},                         // (None)
    {12, RETURN_VALUE, 0}};
  // clang-format on

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "UnpackSequence", 3, 1));
  LineMatcher lm(FormatOrDie(f));

  S6_ASSERT_OK(lm.OnAnyLine("&0:", DefValue("array")));

  // Jump to the first check where we determine the tuple size. Note it's
  // difficult to have a reliable check on determining whether the value is a
  // tuple.
  S6_ASSERT_OK(lm.OnAnyLine("constant $3"));

  // Now jump forward to where we save the 3 tuple items.
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("tuple1"), " = load i64 [ ",
                            lm.Ref("array"), " + $40 ]"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull ", lm.Ref("tuple1")));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("tuple2"), " = load i64 [ ",
                            lm.Ref("array"), " + $32 ]"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull ", lm.Ref("tuple2")));
  S6_ASSERT_OK(lm.OnAnyLine(DefValue("tuple3"), " = load i64 [ ",
                            lm.Ref("array"), " + $24 ]"));
  S6_ASSERT_OK(lm.OnAnyLine("incref notnull ", lm.Ref("tuple3")));

  // Jump to array check.
  S6_ASSERT_OK(lm.OnAnyLine("constant $3"));

  S6_ASSERT_OK(lm.OnAnyLine(DefValue("ob_item"), "= load i64 [ ",
                            lm.Ref("array"), " + $24 ]"));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("array1"), " = load i64 [ ",
                             lm.Ref("ob_item"), " + $16 ]"));
  S6_ASSERT_OK(lm.OnNextLine("incref notnull ", lm.Ref("array1")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("array2"), " = load i64 [ ",
                             lm.Ref("ob_item"), " + $8 ]"));
  S6_ASSERT_OK(lm.OnNextLine("incref notnull ", lm.Ref("array2")));
  S6_ASSERT_OK(lm.OnNextLine(DefValue("array3"), " = load i64 [ ",
                             lm.Ref("ob_item"), " + $0 ]"));
  S6_ASSERT_OK(lm.OnNextLine("incref notnull ", lm.Ref("array3")));

  // Check we have the generic case which uses the iterable version of the
  // value.
  S6_ASSERT_OK(lm.OnAnyLine("PyObject_GetIter"));
  S6_ASSERT_OK(lm.OnAnyLine("PyIter_Next"));
  S6_ASSERT_OK(lm.OnAnyLine("PyIter_Next"));
  S6_ASSERT_OK(lm.OnAnyLine("PyIter_Next"));
  // One last call to check if the iterator is exhausted.
  S6_ASSERT_OK(lm.OnAnyLine("PyIter_Next"));
  // You shouldn't have any more instructions to get the next element, as there
  // are only 3.
  ASSERT_FALSE(lm.OnAnyLine("PyIter_Next").ok());
}

TEST(IngestionTest, ListAppend) {
  // Generated from:
  //   def foo():
  //     l = [i+1 for i in range(10)]
  //     return l
  // Specifically the `<listcomp>` code object inside.
  // clang-format off
  BytecodeInstruction program[] = {
    {0, BUILD_LIST, 0},
    {2, LOAD_FAST, 0},                           // (.0)
    {4, FOR_ITER, 12},                           // (to 18)
    {6, STORE_FAST, 1},                          // (i)
    {8, LOAD_FAST, 1},                           // (i)
    {10, LOAD_CONST, 0},                         // (1)
    {12, BINARY_ADD, 0},
    {14, LIST_APPEND, 2},
    {16, JUMP_ABSOLUTE, 4},
    {18, RETURN_VALUE, 0}};
  // clang-format on

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "ListAppend", 2, 1));
  LineMatcher lm(FormatOrDie(f));

  S6_ASSERT_OK(lm.OnAnyLine(DefValue("array"), " = call_native PyList_New"));

  // Make sure we call append and decref the popped off element.
  S6_ASSERT_OK(lm.OnAnyLine("call_native PyList_Append (", lm.Ref("array"),
                            ", ", DefValue("element"), ")"));
  S6_ASSERT_OK(lm.OnNextLine("decref notnull ", lm.Ref("element")));
}

TEST(IngestionTest, ProfileCounterAdvance) {
  BytecodeInstruction program[] = {
      {0, LOAD_CONST, 1},
      {2, DUP_TOP, 0},
      {4, BINARY_FLOOR_DIVIDE, 0},
      {6, RETURN_VALUE, 0},
  };

  S6_ASSERT_OK_AND_ASSIGN(
      Function f, IngestProgram(program, "ProfileCounterAdvance", 0, 0));

  LineMatcher lm(FormatOrDie(f));
  S6_ASSERT_OK(lm.OnAnyLine("advance_profile_counter $4"));
}

TEST(IngestionTest, YieldValue) {
  // clang-format off
  BytecodeInstruction program[] = {
    {0, LOAD_CONST, 0},
    {2, LOAD_CONST, 2},
    {4, YIELD_VALUE, 0},
    {6, RETURN_VALUE, 1},
  };
  // clang-format on

  S6_ASSERT_OK_AND_ASSIGN(Function f,
                          IngestProgram(program, "YieldValue", 0, 0));
  LineMatcher lm(FormatOrDie(f));

  // The yield_value should have one element in its value stack.
  S6_ASSERT_OK(lm.OnAnyLine("yield_value", "[", DefValue("x"), "]"));
}

}  // namespace
}  // namespace deepmind::s6
