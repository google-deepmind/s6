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

#include "api.h"

#include <Python.h>
#include <sys/resource.h>

#include <fstream>
#include <mutex>  // NOLINT
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "classes/object.h"
#include "core_util.h"  // For IsEnabled().
#include "eval.h"
#include "interpreter.h"
#include "oracle.h"
#include "runtime/interposer.h"
#include "strongjit/optimizer_util.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"
#include "utils/path.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
// The frame evaluation function to use if JIT is enabled.
_PyFrameEvalFunction jit_eval_frame_function;

absl::Status Initialize(bool adopt_types) {
  static std::once_flag init_once;
  std::call_once(init_once, [&]() {
    // Parse flags passed in the environment variable "S6FLAGS".
    if (const char* flags_from_env = getenv("S6FLAGS")) {
      std::vector<std::string> split = absl::StrSplit(flags_from_env, ' ');
      // We set argv[0] to UNKNOWN. An argv[0] for the program name is
      // required by absl::ParseCommandLine, and the string "UNKNOWN" is
      // handled specially (it is the initial "program name not yet set"
      // sentinel).
      std::vector<const char*> argv(1, "UNKNOWN");
      for (const std::string& str : split) {
        argv.push_back(str.data());
      }
      absl::ParseCommandLine(argv.size(), const_cast<char**>(argv.data()));
    }

    // In debug mode we need a reasonable stack size to run all tests.
#ifndef NDEBUG
    const rlim_t kStackSize = 64L * 1024L * 1024L;  // min stack size = 16 Mb
    struct rlimit rl;
    S6_CHECK(getrlimit(RLIMIT_STACK, &rl) == 0);
    if (rl.rlim_cur < kStackSize) {
      rl.rlim_cur = kStackSize;
      S6_CHECK(setrlimit(RLIMIT_STACK, &rl) == 0);
    }
#endif

    BuiltinObjects::Instance().Initialize();

    SetUpInterposers();
    jit_eval_frame_function = GetFrameEvalFunction();
  });

  if (adopt_types) {
    static std::once_flag adopt_once;
    std::call_once(adopt_once, [&]() {
      AdoptExistingTypes();
      AdoptNewTypes();
    });
  }

  return absl::OkStatus();
}

EvalFrameScope::EvalFrameScope(_PyFrameEvalFunction new_eval_frame_function) {
  PyThreadState* tstate = PyThreadState_GET();
  previous_ = tstate->interp->eval_frame;

  tstate->interp->eval_frame = new_eval_frame_function;
}

void EvalFrameScope::RestoreNow() {
  if (!previous_) return;
  PyThreadState* tstate = _PyThreadState_UncheckedGet();
  // This could be called during shutdown, so be tolerant of there not being
  // a thread state.
  if (tstate) tstate->interp->eval_frame = previous_;
  previous_ = nullptr;
}

JitScope::JitScope() : EvalFrameScope(jit_eval_frame_function) {}

EvaluatorScope::EvaluatorScope() : EvalFrameScope(S6EvalFrameWithEvaluator) {}

InterpreterScope::InterpreterScope() : EvalFrameScope(s6::EvalFrame) {}

}  // namespace deepmind::s6
