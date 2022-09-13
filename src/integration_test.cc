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

#include <Python.h>
#include <pythonrun.h>

#include <cstdint>
#include <functional>
#include <mutex>  // NOLINT
#include <random>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "api.h"
#include "code_generation/code_generator.h"
#include "code_generation/register_allocator.h"
#include "code_generation/trace_register_allocator.h"
#include "core_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "metadata.h"
#include "runtime/evaluator.h"
#include "strongjit/deoptimization.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion.h"
#include "strongjit/optimizer.h"
#include "strongjit/util.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6::strongjit {
namespace {

// Gets the top level function named "f".
absl::StatusOr<PyObject*> GetTopLevelFuncF(PyObject* const locals,
                                           PyObject* const globals) {
  PyObject* f = PyDict_GetItemString(locals, "f");
  if (f == nullptr) {
    return absl::NotFoundError("Couldn't find the definition of f.");
  }
  if (!PyFunction_Check(f)) {
    return absl::FailedPreconditionError("f is not a PyFunctionObject.");
  }
  return f;
}

using FunctionExtractor =
    std::function<absl::StatusOr<PyObject*>(PyObject*, PyObject*)>;

// `program` is Python code that creates a function `f()`. Evaluate it, call
// `func_extractor` to extract `f`, and run that once under CPython and once
// under strongjit.
//
// We do it this roundabout way because Py_CompileStringObject(Py_file_input)
// always returns None and Py_CompileStringObject(Py_eval_input) only accepts
// expressions so we can't test nontrivial programs.
void EvalProgramTwoWays(const std::string& program,
                        bool use_code_generator = false,
                        FunctionExtractor extract_function = GetTopLevelFuncF) {
  static std::once_flag once;
  std::call_once(once, []() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });

  PyCompilerFlags flags = {0};
  PyObject* m = PyImport_AddModule("__main__");
  ASSERT_THAT(m, testing::NotNull());
  PyObject* globals = PyModule_GetDict(m);
  PyObject* locals = PyDict_New();
  // Evaluate the test string which will populate locals with a function "f".
  PyRun_StringFlags(program.c_str(), Py_file_input, globals, locals, &flags);

  // Extract the function under test and its code object.
  S6_ASSERT_OK_AND_ASSIGN(PyObject * f, extract_function(locals, globals));
  PyCodeObject* co = reinterpret_cast<PyCodeObject*>(PyFunction_GetCode(f));
  ASSERT_THAT(co, testing::NotNull());

  // All defined functions end up in locals, so merge locals into globals so
  // later evaluations can see them.
  PyDict_Merge(PyFunction_GetGlobals(f), locals, /*override=*/1);

  // Now we can compile `co`.
  std::vector<BytecodeInstruction> bytecode_insts = ExtractInstructions(co);
  auto function_or =
      IngestProgram(bytecode_insts, PyObjectToString(co->co_name),
                    co->co_nlocals, co->co_argcount);
  if (function_or.status().code() == absl::StatusCode::kUnimplemented &&
      absl::StrContains(function_or.status().message(), "LOAD_METHOD")) {
    // TODO: Remove when LOAD_METHOD is implemented.
    GTEST_SKIP_("Test requires LOAD_METHOD, which is not yet implemented.");
  }
  S6_ASSERT_OK(function_or.status());
  Function function = std::move(function_or.value());
  S6_ASSERT_OK(OptimizeFunction(function, co));
  S6_VLOG(1) << FormatOrDie(function);

  PyObject* reference_retval = _PyFunction_FastCallKeywords(
      f, /*stack=*/nullptr, /*nargs=*/0, /*kwnames=*/nullptr);

  PyObject* reference_exc_type;
  PyObject* reference_exc_value;
  PyObject* reference_exc_traceback;
  PyErr_Fetch(&reference_exc_type, &reference_exc_value,
              &reference_exc_traceback);
  PyErr_Clear();

  // Now patch the interpreter function and go again.
  PyThreadState* tstate = PyThreadState_GET();
  static Function* global_function;
  global_function = &function;
  static PyCodeObject* global_pycode_object;
  global_pycode_object = co;
  static NoDestructor<absl::Status> global_status;
  *global_status = absl::OkStatus();

  if (use_code_generator) {
    std::default_random_engine rng(42);
    S6_ASSERT_OK(StressTestByDeoptimizingRandomly(*global_function, rng));
    S6_ASSERT_OK(MarkDeoptimizedBlocks(*global_function));
    S6_ASSERT_OK(RewriteFunctionForDeoptimization(*global_function));

    // This can conflict with the Oracle!
    static NoDestructor<JitAllocator> allocator(4096);
    Metadata* meta = Metadata::Get(global_pycode_object);
    SplitCriticalEdges(*global_function);
    S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RegisterAllocation> ra,
                            AllocateRegistersWithTrace(*global_function));
    S6_VLOG(2) << ra->ToString(*global_function);
    S6_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<CodeObject> co,
        GenerateCode(std::move(function), *ra, {}, *allocator,
                     global_pycode_object, meta));
    S6_VLOG(1) << co->Disassemble();
    static CodeObject* global_co;
    global_co = co.release();
    static int64_t global_profile_counter = -1;

    meta->GetOrCreateThunk(*allocator);

    tstate->interp->eval_frame =
        +[](PyFrameObject* f, int throwflag) -> PyObject* {
      if (f->f_code == global_pycode_object) {
        Py_INCREF(f);
        return global_co->GetPyFrameBody()(f, &global_profile_counter,
                                           global_co);
      }
      return _PyEval_EvalFrameDefault(f, throwflag);
    };
  } else {
    S6_ASSERT_OK(MarkDeoptimizedBlocks(*global_function));
    S6_ASSERT_OK(RewriteFunctionForDeoptimization(*global_function));

    tstate->interp->eval_frame =
        +[](PyFrameObject* f, int throwflag) -> PyObject* {
      if (f->f_code == global_pycode_object) {
        auto obj_or = EvaluateFunction(*global_function, f);
        if (obj_or.status().ok()) return *obj_or;
        *global_status = obj_or.status();
        return nullptr;
      }
      return _PyEval_EvalFrameDefault(f, throwflag);
    };
  }

  PyObject* strongjit_retval =
      _PyFunction_FastCallKeywords(f, nullptr, 0, /*kwnames=*/nullptr);
  tstate->interp->eval_frame = _PyEval_EvalFrameDefault;

  S6_ASSERT_OK(*global_status);

  PyObject* strongjit_exc_type;
  PyObject* strongjit_exc_value;
  PyObject* strongjit_exc_traceback;
  PyErr_Fetch(&strongjit_exc_type, &strongjit_exc_value,
              &strongjit_exc_traceback);

  S6_LOG(INFO) << "Reference returned: " << PyObjectToString(reference_retval);
  S6_LOG(INFO) << "strongjit returned: " << PyObjectToString(strongjit_retval);
  EXPECT_EQ(PyObjectToString(reference_retval),
            PyObjectToString(strongjit_retval));

  if (!reference_retval) {
    S6_LOG(INFO) << "Reference error: " << PyObjectToString(reference_exc_type)
                 << " " << PyObjectToString(reference_exc_value);
  }
  if (!strongjit_retval) {
    S6_LOG(INFO) << "strongjit error: " << PyObjectToString(strongjit_exc_type)
                 << " " << PyObjectToString(strongjit_exc_value);
  }

  EXPECT_EQ(PyObjectToString(reference_exc_type),
            PyObjectToString(strongjit_exc_type));
  EXPECT_EQ(PyObjectToString(reference_exc_value),
            PyObjectToString(strongjit_exc_value));

  _Py_IDENTIFIER(stderr);
  PyObject* fileobj = _PySys_GetObjectId(&PyId_stderr);
  PyTraceBack_Print(reference_exc_traceback, fileobj);
  PyTraceBack_Print(strongjit_exc_traceback, fileobj);
}

class CodegenEvaluatorTest : public testing::TestWithParam<bool> {
 public:
  bool UseCodeGen() const { return GetParam(); }
};

// Created program has {LOAD_CONST, RETURN_VALUE}.
TEST_P(CodegenEvaluatorTest, SimpleReturn) {
  EvalProgramTwoWays(R"(
def f():
  return 2 + 2
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, TracebackLocation) {
  EvalProgramTwoWays(R"(
import sys
def f():
  1/0
  return None
)",
                     GetParam());
}

TEST_P(CodegenEvaluatorTest, Arithmetic) {
  EvalProgramTwoWays(R"(
def f():
  return 43+42/43+2*6
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, LoadFastAndBinaryMultiply) {
  EvalProgramTwoWays(R"(
def f(a = 0.0001, b = 1000.0):
    return (a * b)
)");
}

TEST_P(CodegenEvaluatorTest, IfStatement) {
  EvalProgramTwoWays(R"(
def f(a = 0.0001):
  if a:
    return 2
  return a * 4
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, CallFunction) {
  EvalProgramTwoWays(R"(
def f():
  open(None)
)",
                     GetParam());
}

TEST_P(CodegenEvaluatorTest, CallFunctionSuccess) {
  EvalProgramTwoWays(R"(
def g():
  return 42
def f():
  g()
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, CallFunctionArguments) {
  EvalProgramTwoWays(R"(
def g(foo):
  return foo + 42
def f():
  return g(5)
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, LoadStoreAttr) {
  EvalProgramTwoWays(R"(
class C:
  pass
def f(c = C()):
    c.x = 4
    return c.x
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, StoreDeleteAttr) {
  EvalProgramTwoWays(R"(
class C:
  pass
def f(c = C()):
    c.x = 4
    del c.x
    return c.x
)",
                     GetParam());
}

TEST_P(CodegenEvaluatorTest, MakeFunction) {
  EvalProgramTwoWays(R"(
def f():
    return (lambda x: x)(42)
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, MakeFunctionWithDefaults) {
  EvalProgramTwoWays(R"(
def f():
    return (lambda x = 4: x)()
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, BuildTuple) {
  EvalProgramTwoWays(R"(
def f(x = 5):
    return 4, x, 6
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, BuildList) {
  EvalProgramTwoWays(R"(
def f(x = 6):
    return [4, x, 6]
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, UnaryNot) {
  EvalProgramTwoWays(R"(
def f(x = 6):
    return not x
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, CallFunctionKwargs) {
  EvalProgramTwoWays(R"(
def g(y = 2, x = 0):
  return y * x
def f():
  return g(x=4)
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, CallFunctionKwargs2) {
  EvalProgramTwoWays(R"(
def g(y, z = 2, x = 0):
  return y * x
def f():
  return g(2, x=4)
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, StoreFast) {
  EvalProgramTwoWays(R"(
def f(x = 42):
    x = 42
    return x
)",
                     UseCodeGen());
}
TEST(EvaluatorTest, DeleteFast) {
  EvalProgramTwoWays(R"(
def f(x = 42):
    del x
    return x
)");
}

TEST_P(CodegenEvaluatorTest, ReturnInTry) {
  EvalProgramTwoWays(R"(
def f():
  a = 42
  try:
    7
  except:
    return 2
  return a
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, RaiseReturnInTry) {
  EvalProgramTwoWays(R"(
def f():
  a = 42
  try:
    1/0
  except:
    return 2
  return a
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, NestedExcept) {
  EvalProgramTwoWays(R"(
def f():
  try:
    try:
      1/0
    except ValueError:
      pass
  except:
    return 6
  return 42
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, RaiseInExceptHandler) {
  EvalProgramTwoWays(R"(
def g():
  raise ValueError("value error")

def f():
  a = 42
  try:
    1/0
  except:
    g()
  return a
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, CatchTheWrongException) {
  EvalProgramTwoWays(R"(
def f():
  try:
    return 1/0
  except ValueError:
    return None
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, UnpackSequenceTuple) {
  EvalProgramTwoWays(R"(
def f():
  l = (1, 2, 3)
  x, y, z = l
  return x + z
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, UnpackSequenceList) {
  EvalProgramTwoWays(R"(
def f():
  l = [1, 2, 3]
  x, y, z = l
  return x + z
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, UnpackSequenceCustom) {
  EvalProgramTwoWays(R"(
class CustomIterator:
  def __iter__(self, max_size = 3):
    self.a = 1
    self.max_size = max_size
    return self

  def __next__(self):
    if self.a > self.max_size:
      raise StopIteration
    x = self.a
    self.a += 1
    return x

def f():
  l = CustomIterator()
  x, y, z = l
  return x + z
)",
                     false);
}

TEST_P(CodegenEvaluatorTest, UnpackValueErrorTupleTooFewValues) {
  EvalProgramTwoWays(R"(
def f():
  try:
    l = (1, 2, 3)
    w, x, y, z = l
    return x + z
  except ValueError:
    return None
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, UnpackValueErrorTupleTooManyValues) {
  EvalProgramTwoWays(R"(
def f():
  try:
    l = (1, 2, 3)
    w, x = l
    return x + z
  except ValueError:
    return None
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, ListAppend) {
  EvalProgramTwoWays(
      R"(
def f():
  l = [i for i in range(10)]
  return l[0]
)",
      UseCodeGen(),
      [](PyObject* const locals,
         PyObject* const globals) -> absl::StatusOr<PyObject*> {
        S6_ASSIGN_OR_RETURN(PyObject * f, GetTopLevelFuncF(locals, globals));
        PyCodeObject* co =
            reinterpret_cast<PyCodeObject*>(PyFunction_GetCode(f));
        if (co == nullptr) {
          return absl::InternalError("Couldn't get the code object from f.");
        }
        int64_t i = PyTuple_Size(co->co_consts);
        bool found_listcomp = false;
        while (--i >= 0) {
          PyObject* element = PyTuple_GetItem(co->co_consts, i);
          if (element && PyCode_Check(element)) {
            PyCodeObject* new_co = reinterpret_cast<PyCodeObject*>(element);
            if (PyObjectToString(new_co->co_name) == "<listcomp>") {
              f = PyFunction_New(element, globals);
              found_listcomp = true;
              break;
            }
          }
        }
        if (!found_listcomp) {
          return absl::NotFoundError("Couldn't find <listcomp> inside f.");
        }
        return f;
      });
}

TEST_P(CodegenEvaluatorTest, CallNativeIndirect) {
  EvalProgramTwoWays(R"(
def f():
  s = 0
  for i in range(10):
    s += i
  return s
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, BreakLoopWhile) {
  EvalProgramTwoWays(R"(
def f():
  i = -1
  while i < 2:
    try:
      i/i
    except:
      break
    i += 1
  return i
)",
                     // TODO: Re-enable for code generator after fixing
                     // unhandled copies after ExceptInsts.
                     false);
}

// TODO: uncomment after fixing unhandled copies in code_generator.cc
// TEST_P(CodegenEvaluatorTest, BreakLoopFor) {
//   EvalProgramTwoWays(R"(
// def f():
//   for i in [1, 0, -1]:
//     try:
//       i/i
//     except:
//       break
//   return i
// )",
//                      UseCodeGen());
// }

TEST_P(CodegenEvaluatorTest, ContinueLoopWhile) {
  EvalProgramTwoWays(R"(
def f():
  lst = []
  i = -1
  while i < 2:
    i += 1
    try:
      1/i
    except:
      continue
    lst.append(i)
  return lst
)",
                     UseCodeGen());
}

TEST_P(CodegenEvaluatorTest, ContinueLoopFor) {
  EvalProgramTwoWays(R"(
def f():
  lst = []
  for i in [1, 0, -1]:
    try:
      1/i
    except:
      continue
    lst.append(i)
  return lst
)",
                     // TODO: Re-enable for code generator after fixing
                     // unhandled copies after ExceptInsts.
                     false);
}

TEST_P(CodegenEvaluatorTest, DeoptimizeDuringArgs) {
  EvalProgramTwoWays(R"(
class C(object):
  def g(self, x): pass

def f():
  c = C()
  # x should raise UnboundLocalError
  c.g(x)
  x = 2
)",
                     true);
}

INSTANTIATE_TEST_SUITE_P(CGEvaluatorTest, CodegenEvaluatorTest, testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "UsingCodeGenerator"
                                             : "UsingEvaluator";
                         });
}  // namespace
}  // namespace deepmind::s6::strongjit
