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

#ifndef THIRD_PARTY_DEEPMIND_S6_API_H_
#define THIRD_PARTY_DEEPMIND_S6_API_H_

#include <Python.h>

#include "absl/status/statusor.h"

namespace deepmind::s6 {

// Initializes S6. This performs once-only initialization:
//   * Consumes flags from the environment variable "S6FLAGS", if it exists.
//   * Ensures the stack rlimit is large enough in debug mode.
//
// JIT is not enabled.
//   `adopt_types`: Initializes with type adoption. Type adoption is required
//     for performance just-in-time compilation, but not required for profiling.
//     Turning this on provides a small performance penalty for all attribute
//     stores.
absl::Status Initialize(bool adopt_types = false);

// RAII object that enables to change temporarily change the main frame
// evaluation function. The original frame evaluation function is restored when
// the object is destroyed.
//
// These objects nest; enabling profiling within a JIT scope will enable both
// profiling and JIT, for example.
class EvalFrameScope {
 public:
  explicit EvalFrameScope(_PyFrameEvalFunction new_eval_frame_function);
  ~EvalFrameScope() { RestoreNow(); }

  // Retores the original frame evaluation function before destruction.
  void RestoreNow();

  // Not copyable or moveable.
  EvalFrameScope(const EvalFrameScope&) = delete;
  EvalFrameScope& operator=(const EvalFrameScope&) const = delete;
  EvalFrameScope(EvalFrameScope&& other) = delete;
  EvalFrameScope& operator=(EvalFrameScope&&) = delete;

 private:
  // The eval function to restore when this scope exits.
  _PyFrameEvalFunction previous_;
};

// RAII object that enables JIT compilation using S6 within a scope. Behavior of
// this function is well defined within a threaded context.
//
// These objects nest; enabling profiling within a JIT scope will enable both
// profiling and JIT, for example.
class JitScope : public EvalFrameScope {
 public:
  JitScope();
};

// RAII object that enables JIT compilation using S6 within a scope, but then
// forces the use of the evaluator. Behavior of this function is well defined
// within a threaded context.
//
// These objects nest; enabling profiling within a JIT scope will enable both
// profiling and JIT, for example.
class EvaluatorScope : public EvalFrameScope {
 public:
  EvaluatorScope();
};

// RAII object that forces the use of S6 Interpreter. Behavior of this function
// is well defined within a threaded context.
//
// These objects nest; enabling profiling within a JIT scope will enable both
// profiling and JIT, for example.
class InterpreterScope : public EvalFrameScope {
 public:
  InterpreterScope();
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_API_H_
