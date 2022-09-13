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

#include "runtime/interposer.h"

#include <Python.h>

#include <cstdint>
#include <utility>

#include "absl/base/call_once.h"
#include "code_object.h"
#include "core_util.h"
#include "interpreter.h"
#include "runtime/deoptimization_runtime.h"
#include "strongjit/formatter.h"

namespace deepmind::s6 {
namespace {

PyObject* LookupLocationForValueAt(const Value* value, const Instruction* inst,
                                   const DeoptimizationMap& deopt_map,
                                   int64_t* spill_slots) {
  for (const auto& [v, location] : deopt_map.live_values(inst)) {
    if (v != value) continue;
    return reinterpret_cast<PyObject*>(LookupLocation(location, spill_slots));
  }
  S6_LOG(FATAL) << "Value not found!";
}

// Populates the fastlocals with borrowed references from stack_frame. Returns
// the number of fastlocals populated.
int64_t PopulateFastLocals(StackFrame* stack_frame, PyFrameObject* pyframe,
                           int64_t num_fastlocals = INT64_MAX) {
  const DeoptimizationMap& deopt_map =
      stack_frame->s6_code_object()->deoptimization_map();
  StackFrameWithLayout layed_out_stack_frame(
      *stack_frame, stack_frame->s6_code_object()->stack_frame_layout());
  const Instruction* inst = deopt_map.GetInstructionAtAddress(
      *layed_out_stack_frame.GetReturnAddress());
  absl::Span<int64_t> spill_slots =
      layed_out_stack_frame.GetSpillSlots<int64_t>();

  int64_t fastlocal_index = 0;
  for (const Value* fastlocal_value : deopt_map.GetLiveFastLocals(inst)) {
    if (fastlocal_index >= num_fastlocals) break;
    pyframe->f_localsplus[fastlocal_index++] = LookupLocationForValueAt(
        fastlocal_value, inst, deopt_map, spill_slots.data());
  }
  return fastlocal_index;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Dynamic interposers
//
// These override some CPython runtime datastructures.
//
// Must be installed by calling SetUpInterposers.

extern "C" {

static initproc init_original;

static int SuperInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  if (PyTuple_GET_SIZE(args) != 0) {
    return init_original(self, args, kwargs);
  }

  PyFrameObject* pyframe = PyThreadState_GET()->frame;
  StackFrame* stack_frame = FindStackFrameForPyFrameObject(pyframe, 20);
  if (!stack_frame) {
    return init_original(self, args, kwargs);
  }

  // This is a call to super() with no arguments inside a Strongjit frame.
  // init_original_ is going to try to inspect the fastlocals, so materialize
  // them now.
  //
  // Note that super_init() only needs f_localsplus[0] (or a cellvar, but those
  // are not SSA-promoted so already exist).
  int64_t num_fastlocals = PopulateFastLocals(stack_frame, pyframe, 1);

  int result = init_original(self, args, kwargs);

  // Reset all fastlocals to nullptr again.
  for (int64_t i = 0; i < num_fastlocals; ++i) {
    pyframe->f_localsplus[i] = nullptr;
  }

  return result;
}

}  // extern "C"

void SetUpInterposers() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    init_original = std::exchange(PySuper_Type.tp_init, SuperInit);
  });
}

////////////////////////////////////////////////////////////////////////////////
// Static interposers
//
// The linkopts for this library include `-Wl,--wrap=X`, for each interposed
// symbol. This instructs the linker to reroute all calls to X to __wrap_X, and
// makes the original X available as __real_X.
//
// We use this to wrap functions that need state initializing if run under S6.

extern "C" int __real_PyFrame_FastToLocalsWithError(PyFrameObject* f);

// PyFrame_FastToLocalsWithError reads the fastlocals.
extern "C" int __wrap_PyFrame_FastToLocalsWithError(PyFrameObject* f) {
  StackFrame* stack_frame = FindStackFrameForPyFrameObject(f, 20);
  if (!stack_frame) {
    return __real_PyFrame_FastToLocalsWithError(f);
  }

  int64_t num_fastlocals = PopulateFastLocals(stack_frame, f);
  int result = __real_PyFrame_FastToLocalsWithError(f);

  // Reset all fastlocals to nullptr again.
  for (int64_t i = 0; i < num_fastlocals; ++i) {
    f->f_localsplus[i] = nullptr;
  }
  return result;
}

}  // namespace deepmind::s6
