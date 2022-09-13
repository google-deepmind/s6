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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_GENERATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_GENERATOR_H_
#include <Python.h>

#include <cstdint>
#include <memory>

#include "code_object.h"
#include "strongjit/instructions.h"
#include "strongjit/value_map.h"
#include "utils/logging.h"

namespace deepmind::s6 {
template <typename T>
class JitStub;

// Generator functions are a kind of coroutine. Note that Python has the concept
// of "coroutines" as distinct from generators. Coroutines use `async def` and
// while they share some implementation details with generators, they are not
// the same.
//
// A generator is a coroutine in the classical sense; it is a function that
// can `yield` a value during execution, be paused, then continue later from
// that yield statement. Using PyGen_Send(), a user may pass an object to
// the generator once resumed; this becomes the return value of the `yield`
// expression.
//
//   def f():
//     yield 1
//     x = yield 2 # `x` is the value sent by PyGen_Send(), or None, usually.
//
// A code object that contains a `yield` expression has the flag CO_GENERATOR
// set.
//
// When a function with the flag CO_GENERATOR is called, instead of actually
// running the function, Python creates a PyGenObject and returns that.
//
// The PyGenObject contains an initialized PyFrameObject. Generators are run
// by _PyGen_Send(x), where `x` is usually nullptr.
//
// _PyGen_Send(x) pushes x onto the value stack and calls PyEval_EvalFrame.
//
// How S6 runs generators
// ======================
//
// S6 requires extra state, not just the PyFrameObject. For generated code,
// the spilled stack slots must be stored permanently somewhere, and for the
// evaluator, the ValueMap must be stored somewhere. We store this data in a
// GeneratorState object.
//
// How S6 runs PyGenObjects
// ========================
//
// This assumes that S6 is presented with a PyGenObject, already created. There
// are optimizations we can do to create a subclass of PyGenObject that contains
// the state we need (or elide frame creation entirely), but we assume here that
// we failed to perform this optimization.
//
// We need three things:
//  1) to be able to find the value given to the yield expression by _PyGen_Send
//  2) to be able to find our local state (ValueMap or spill slots).
//  3) to be able to destroy our local state when the generator is destroyed.
//
// We (ab)use the PyFrameObject's value stack for this purpose. _PyGen_Send
// uses the frame's value stack already to pass the yielded value. We ensure
// that the value stack has either one or two entries:
//   valuestack[0] = GeneratorStateObject
//   valuestack[1] = yielded value (on entry to the function only)
//
// The GeneratorStateObject is a PyObject subclass that owns a GeneratorState.
// Because it is a PyObject, it can be stored in valuestack and left there
// for the PyFrameObject's destructor to find, and destroy. This allows us to
// correctly clean up when a generator is closed, or deoptimized.

// The state required for a strongjit generator function.
class GeneratorState {
 public:
  explicit GeneratorState(uint64_t* spill_slots) : spill_slots_(spill_slots) {}
  ~GeneratorState() { delete value_map_; }

  // Returns the value map for use by the evaluator.
  ValueMap* value_map() { return value_map_; }

  // Ensures that a ValueMap has been created. This is only intended for use
  // when executing code with the evaluator.
  void EnsureValueMapCreated(const Function& f);

  // If this generator is paused, returns the YieldValueInst that it is paused
  // at. Otherwise, returns nullptr.
  const YieldValueInst* yield_value_inst() const { return yield_value_inst_; }
  void set_yield_value_inst(const YieldValueInst* yi) {
    yield_value_inst_ = yi;
  }

  // Returns the spill slot storage, for use by the code generator.
  uint64_t* spill_slots() { return spill_slots_; }

  // Obtains the "resume PC"; an address within generated code to resume
  // execution.
  uint64_t resume_pc() const { return resume_pc_; }
  void set_resume_pc(uint64_t pc) { resume_pc_ = pc; }

  // The code object this generator is executing with.
  CodeObject* code_object() { return code_object_; }
  void set_code_object(CodeObject* code_object) { code_object_ = code_object; }

  // Returns the GeneratorState object for `frame`, if one exists. If frame
  // is not associated with a generator, nullptr is returned.
  inline static GeneratorState* Get(PyFrameObject* frame);

  // Returns the GeneratorState object for `frame`, assuming it exists. The
  // behavior is undefined if the GeneratorState does not exist.
  inline static GeneratorState* GetUnsafe(PyFrameObject* frame);

  // Creates a new GeneratorState object for `frame`, with room for
  // `num_spill_slots` spill slots.
  static GeneratorState* Create(PyFrameObject* frame, int64_t num_spill_slots);

 private:
  template <typename T>
  friend class ::deepmind::s6::JitStub;

  // Owned by this GeneratorState, but we don't use a std::unique_ptr so that
  // we can use memcpy in PyGeneratorFrameObjectCache.
  ValueMap* value_map_ = nullptr;
  const YieldValueInst* yield_value_inst_ = nullptr;
  uint64_t* spill_slots_;
  uint64_t resume_pc_ = 0;
  CodeObject* code_object_ = nullptr;
};

// Returns true if this frame object corresponds to a "fresh" generator.
// A fresh generator frame has not been run yet; its f_lasti is -1.
inline bool IsFreshGeneratorFrame(PyFrameObject* frame) {
  return frame->f_gen != nullptr && frame->f_lasti < 0;
}

// Returns true if this is a generator frame that has been deoptimized; if it
// isn't fresh and doesn't have a generator state.
inline bool IsDeoptimizedGeneratorFrame(PyFrameObject* frame) {
  return frame->f_gen != nullptr && !IsFreshGeneratorFrame(frame) &&
         GeneratorState::Get(frame) == nullptr;
}

// Deallocates the GeneratorStateObject held by `frame`, and sets `frame`'s
// f_stacktop to nullptr to signify there are no more items on its stack.
void DeallocateGeneratorState(PyFrameObject* frame);

// Wraps (owns) a GeneratorState. This is a PyObject, so is created with
// PyObject_Alloc, so it cannot have a constructor or destructor. We construct
// `state` in-place with placement new.
struct GeneratorStateObject {
  PyObject_VAR_HEAD;

  // The wrapped GeneratorState object.
  GeneratorState state;

  // Spill slots for the code generator. This is a trailing array; the
  // GeneratorStateObject is allocated with enough space for a known array size.
  //
  // These spill slots are the state that the code generator relies upon to be
  // held while a generator is paused.
  uint64_t spill_slots[1];

  GeneratorStateObject() : state(&spill_slots[0]) {}
};

extern PyTypeObject GeneratorState_Type;

GeneratorState* GeneratorState::Get(PyFrameObject* frame) {
  if (!frame || !frame->f_gen) {
    return nullptr;
  }

  // If we have a generator state object, it'll be the first item in the frame's
  // value stack.
  if (frame->f_stacktop == nullptr ||
      frame->f_stacktop == frame->f_valuestack) {
    // The stack is empty.
    return nullptr;
  }

  PyObject* state_obj = frame->f_valuestack[0];
  if (state_obj && Py_TYPE(state_obj) == &GeneratorState_Type) {
    return &(reinterpret_cast<GeneratorStateObject*>(state_obj)->state);
  }
  return nullptr;
}

GeneratorState* GeneratorState::GetUnsafe(PyFrameObject* frame) {
  PyObject* state_obj = frame->f_valuestack[0];
  S6_DCHECK(state_obj);
  S6_DCHECK(Py_TYPE(state_obj) == &GeneratorState_Type);
  return &(reinterpret_cast<GeneratorStateObject*>(state_obj)->state);
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_GENERATOR_H_
