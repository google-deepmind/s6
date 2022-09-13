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

#include "eval.h"

#include <Python.h>
#include <pthread.h>
#include <stdlib.h>

#include <cstdint>
#include <string>
#include <utility>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "code_object.h"
#include "interpreter.h"
#include "oracle.h"
#include "runtime/deoptimization_runtime.h"
#include "runtime/evaluator.h"
#include "runtime/generator.h"
#include "utils/logging.h"

enum class Mode {
  kCPythonInterpreter,
  kInterpreter,
  kStrongjitEvaluator,
  kStrongjitCodegen,
};

inline bool AbslParseFlag(absl::string_view text, Mode* out,
                          std::string* error) {
  if (text == "cpython-interpreter") {
    *out = Mode::kCPythonInterpreter;
    return true;
  }
  if (text == "interpreter") {
    *out = Mode::kInterpreter;
    return true;
  }
  if (text == "strongjit-evaluator") {
    *out = Mode::kStrongjitEvaluator;
    return true;
  }
  if (text == "strongjit-codegen") {
    *out = Mode::kStrongjitCodegen;
    return true;
  }
  *error = "Unrecognized mode name.";
  return false;
}

inline std::string AbslUnparseFlag(Mode in) {
  if (in == Mode::kCPythonInterpreter) {
    return "cpython-interpreter";
  }
  if (in == Mode::kInterpreter) {
    return "interpreter";
  }
  if (in == Mode::kStrongjitEvaluator) {
    return "strongjit-evaluator";
  }
  if (in == Mode::kStrongjitCodegen) {
    return "strongjit-codegen";
  }
  return "unknown";
}

ABSL_FLAG(Mode, s6_mode, Mode::kStrongjitCodegen, "Evaluation mode");

// Bisection support. Specify --s6_fuel_end to any value to start
// bisecting. S6 will keep a counter of the number of successful
// compilations `n`. If `0 <= n < s6_fuel_end`,
// JIT will be used otherwise it will bail out to the interpreter.
ABSL_FLAG(int64_t, s6_fuel_end, ~0U, "Bisection fuel - end");
ABSL_FLAG(bool, s6_fuel_verbose, false,
          "If bisection is enabled and"
          " fuel drops to zero, dump the last compiled program verbosely.");
ABSL_FLAG(bool, s6_compile_synchronous, false,
          "Compile synchronously. The default is asynchronous (background) "
          "compilation.");
ABSL_FLAG(bool, s6_compile_always, false,
          "Compile always. The default is to only compile functions when "
          "profiling events occur.");

ABSL_FLAG(
    int64_t, s6_profile_bytecode_instruction_interval, 0,
    "Testing only: the interval in CPython bytecodes between profiling events");

ABSL_FLAG(
    bool, s6_enable_event_counters, false,
    "Enables the emission of event counter instructions in generated code. "
    "Dumps the result to the log; use -vmodule=oracle=1.");
ABSL_FLAG(
    std::string, s6_event_counters_dir, "",
    "Enables the emission of event counter instructions in generated code, and "
    "dumps the counters to the given directory.");

ABSL_FLAG(bool, s6_enable_unboxing_optimization, true,
          "Enable insertion of unbox/box instructions to perform successive "
          "arithmetic operations using machine instructions.");

ABSL_FLAG(bool, s6_harden_refcount_analysis, false,
          "Enable hard checks in the refcount analysis. Any detected reference "
          "manipulation errors will result in a crash of S6 instead of "
          "skipping some related optimizations.");

// Defined in perfetto_trace.cc
ABSL_DECLARE_FLAG(bool, s6_enable_function_tracing);

namespace deepmind::s6 {

PyObject* S6EvalFrame(PyFrameObject* frame, int throwflag) {
  if (throwflag ||
      (frame->f_code->co_flags & (CO_COROUTINE | CO_ASYNC_GENERATOR)) ||
      PyThreadState_GET()->use_tracing || IsDeoptimizedGeneratorFrame(frame)) {
    // If this frame is a paused generator, we need to deoptimize.
    if (GeneratorState* gen_state = GeneratorState::Get(frame);
        gen_state && gen_state->yield_value_inst()) {
      DeoptimizePausedGenerator(gen_state, frame);
    }
    S6_DCHECK(!GeneratorState::Get(frame));
    return EvalFrame(frame, throwflag);
  }

  CodeObject* code_object = nullptr;
  if (GeneratorState* gen_state = GeneratorState::Get(frame)) {
    code_object = gen_state->code_object();
    if (code_object && code_object->deoptimized()) {
      S6_CHECK(gen_state->yield_value_inst());
      DeoptimizePausedGenerator(gen_state, frame);
      return EvalFrame(frame, throwflag);
    }
  }
  if (!code_object) {
    code_object = Oracle::Get()->GetCodeObject(frame->f_code, frame);
  }

  if (!code_object) {
    return EvalFrame(frame, throwflag);
  }

  frame->f_executing = 0;
  return code_object->GetPyFrameBody()(
      frame, Oracle::Get()->GetProfileCounter(), code_object);
}

PyObject* S6EvalFrameWithEvaluator(PyFrameObject* frame, int throwflag) {
  if (throwflag ||
      (frame->f_code->co_flags & (CO_COROUTINE | CO_ASYNC_GENERATOR)) ||
      PyThreadState_GET()->use_tracing || IsDeoptimizedGeneratorFrame(frame)) {
    // If this frame is a strongjit generator that has yielded, we need to
    // deoptimize before we can call EvalFrame.
    if (GeneratorState* gen_state = GeneratorState::Get(frame);
        gen_state && gen_state->yield_value_inst()) {
      DeoptimizePausedGenerator(gen_state, frame);
    }
    return EvalFrame(frame, throwflag);
  }
  CodeObject* code_object = Oracle::Get()->GetCodeObject(frame->f_code, frame);
  if (!code_object) {
    return EvalFrame(frame, throwflag);
  }

  return *EvaluateFunction(code_object->function(), frame);
}

_PyFrameEvalFunction GetFrameEvalFunction() {
  if (!Oracle::Get()) {
    OracleDebugOptions options;
    if (FLAGS_s6_fuel_end.IsSpecifiedOnCommandLine())
      options.bisection_fuel = absl::GetFlag(FLAGS_s6_fuel_end);
    options.bisection_verbose = absl::GetFlag(FLAGS_s6_fuel_verbose);
    options.compile_always = absl::GetFlag(FLAGS_s6_compile_always);
    options.synchronous = absl::GetFlag(FLAGS_s6_compile_synchronous);
    if (FLAGS_s6_profile_bytecode_instruction_interval
            .IsSpecifiedOnCommandLine()) {
      options.profile_bytecode_instruction_interval =
          absl::GetFlag(FLAGS_s6_profile_bytecode_instruction_interval);
    }
    options.enable_event_counters =
        absl::GetFlag(FLAGS_s6_enable_event_counters);
    options.enable_unboxing_optimization =
        absl::GetFlag(FLAGS_s6_enable_unboxing_optimization);
    options.harden_refcount_analysis =
        absl::GetFlag(FLAGS_s6_harden_refcount_analysis);
    options.event_counters_dir = absl::GetFlag(FLAGS_s6_event_counters_dir);
    if (!options.event_counters_dir.empty()) {
      options.enable_event_counters = true;
    }

    Oracle::SetSingleton(new Oracle(options));
    atexit(+[]() {
      // Oracle's destructor prints profiling info if verbose logging is
      // enabled.
      delete Oracle::Get();
    });
    // Ensure the oracle is reinitialized in the child process after a fork,
    // because its thread will have died.
    pthread_atfork(
        // In the parent before forking, drain the queue and ensure the oracle
        // is quiesced.
        +[]() { Oracle::Get()->DenyCompilationRequestsAndDrainQueue(); },
        // In the parent after forking, reenable compilation requests.
        +[]() { Oracle::Get()->AllowCompilationRequests(); },
        // In the child after forking, create a new Oracle.
        +[]() {
          Oracle::SetSingleton(new Oracle(std::move(*Oracle::Get())));
          Oracle::Get()->Initialize();
        });
    Oracle::Get()->Initialize();
  }

  switch (absl::GetFlag(FLAGS_s6_mode)) {
    case Mode::kCPythonInterpreter:
      return _PyEval_EvalFrameDefault;
    case Mode::kStrongjitEvaluator:
      return S6EvalFrameWithEvaluator;
    case Mode::kStrongjitCodegen:
      return S6EvalFrame;
    case Mode::kInterpreter:
      return s6::EvalFrame;
  }
  S6_UNREACHABLE();
}

}  // namespace deepmind::s6
