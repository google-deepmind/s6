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

#ifndef THIRD_PARTY_DEEPMIND_S6_EVAL_H_
#define THIRD_PARTY_DEEPMIND_S6_EVAL_H_

#include <Python.h>
#include <frameobject.h>

namespace deepmind::s6 {

// A frame evaluation function that evaluates a frame by compiling it to
// strongJIT then evaluating it with the S6 evaluator. See `EvaluateFunction`.
PyObject* S6EvalFrameWithEvaluator(PyFrameObject* frame, int throwflag);

// Returns a frame evaluation function. The returned function depends on if
// fastjit or strongjit is enabled, and if any bisection commands are enabled.
_PyFrameEvalFunction GetFrameEvalFunction();

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_EVAL_H_
