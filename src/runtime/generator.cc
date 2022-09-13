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

#include "runtime/generator.h"

#include <frameobject.h>
#include <genobject.h>

#include <cstdint>

#include "strongjit/formatter.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {

void DeallocGeneratorState(GeneratorStateObject* obj) {
  obj->state.~GeneratorState();
}

PyTypeObject GeneratorState_Type = {
    // clang-format off
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    // clang-format on
    "s6_internal.GeneratorState",                        /* tp_name */
    sizeof(GeneratorStateObject),                        /* tp_basicsize */
    sizeof(uint64_t),                                    /* tp_itemsize */
    reinterpret_cast<destructor>(DeallocGeneratorState), /* tp_dealloc */
    nullptr,                                             /* tp_print */
    nullptr,                                             /* tp_getattr */
    nullptr,                                             /* tp_setattr */
    nullptr,                                             /* tp_as_async */
    nullptr,                                             /* tp_repr */
    nullptr,                                             /* tp_as_number */
    nullptr,                                             /* tp_as_sequence */
    nullptr,                                             /* tp_as_mapping */
    nullptr,                                             /* tp_hash */
    nullptr,                                             /* tp_call */
    nullptr,                                             /* tp_str */
    nullptr,                                             /* tp_getattro */
    nullptr,                                             /* tp_setattro */
    nullptr,                                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                  /* tp_flags */
    "S6 generator state"                                 /* tp_doc */
};

GeneratorState* GeneratorState::Create(PyFrameObject* frame,
                                       int64_t num_spill_slots) {
  GeneratorStateObject* state_obj = PyObject_NewVar(
      GeneratorStateObject, &GeneratorState_Type, num_spill_slots);
  new (&state_obj->state) GeneratorState(&state_obj->spill_slots[0]);

  // Store in frame->f_valuestack[0] and donate our reference to it.
  frame->f_valuestack[0] = reinterpret_cast<PyObject*>(state_obj);
  frame->f_stacktop = &frame->f_valuestack[1];
  return &state_obj->state;
}

void GeneratorState::EnsureValueMapCreated(const Function& f) {
  if (value_map_) return;
  value_map_ = new ValueMap(f);
}

void DeallocateGeneratorState(PyFrameObject* frame) {
  Py_CLEAR(frame->f_valuestack[0]);
  frame->f_stacktop = nullptr;
}
}  // namespace deepmind::s6
