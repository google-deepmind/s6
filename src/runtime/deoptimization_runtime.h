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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_RUNTIME_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_RUNTIME_H_

#include <Python.h>
#include <frameobject.h>

#include <cstdint>

#include "runtime/generator.h"
#include "runtime/stack_frame.h"
#include "strongjit/util.h"

namespace deepmind::s6 {

// Deoptimizes a function. Returns the result of the function.
PyObject* Deoptimize();

// Attempts to scan the stack and find a StackFrame corresponding to the given
// PyFrameObject. This is best-effort, and will only traverse `max_frames` in
// an attempt to find a matching stack frame.
StackFrame* FindStackFrameForPyFrameObject(PyFrameObject* pyframe,
                                           int64_t max_frames = 5);

// Converts a Location as recorded by the code generator to a real value.
// REQUIRES: !l.IsRegister();
int64_t LookupLocation(const Location& l, int64_t* spill_slots);

// Deoptimizes this generator state. `pyframe` is the frame for this generator.
// After this function, this GeneratorState object is no longer valid.
void DeoptimizePausedGenerator(GeneratorState* gen_state,
                               PyFrameObject* pyframe);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_DEOPTIMIZATION_RUNTIME_H_
