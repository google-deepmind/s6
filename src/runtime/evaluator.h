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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_EVALUATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_EVALUATOR_H_

#include <Python.h>
#include <frameobject.h>

#include <cstdint>

#include "absl/status/statusor.h"
#include "runtime/stack_frame.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "strongjit/value_map.h"

namespace deepmind::s6 {

// The Evaluator is an interpreter over Strongjit IR. It is used for two
// purposes:
//   1) Debugging the code generator as a source of truth for Strongjit
//        instruction semantics.
//   2) During deoptimization, fast-forwarding execution from wherever the
//        deoptimization event occurred to the next boundary point at which
//        we can construct a CPython interpreter frame.
//
// As a result it is not optimized for speed, it is optimized for readability.
// If a program spends a significant amount of time in this module, that is
// a bug in S6. Even during deoptimization we should only execute < 10
// instructions before we find a boundary point.

// Context for the Evaluate() function. Describes the state of a frame before
// and after an Instruction is executed.
class EvaluatorContext {
 public:
  EvaluatorContext(const Function& f, PyFrameObject* pyframe);

  // Returns true if the evaluator should consider a BytecodeBeginInst as
  // function completion.
  bool finish_on_bytecode_begin() const { return finish_on_bytecode_begin_; }
  void set_finish_on_bytecode_begin(bool b) { finish_on_bytecode_begin_ = b; }

  // Returns true if the function has finished - returned, excepted or yielded.
  bool IsFinished() const { return function_return_value_.has_value(); }

  // Returns the return value of the function.
  // REQUIRES: IsFinished().
  PyObject* GetReturnValue() const {
    S6_CHECK(function_return_value_.has_value());
    return *function_return_value_;
  }

  // Returns the next instruction to execute.
  // REQUIRES: !IsFinished();
  const Instruction* GetNextInstruction() const {
    S6_CHECK(!IsFinished());
    return next_instruction_;
  }

  // Copies all values from the given ValueMap.
  void CopyValues(const ValueMap& others) { values_ = others; }

  // Returns the raw ValueMap.
  const ValueMap& value_map() const { return values_; }

  // Interface for Evaluate() functions.
  PyFrameObject* pyframe() { return pyframe_; }

  template <typename T = int64_t>
  void Set(const Value* v, T value) {
    values_.Set(v, value);
  }

  template <typename T = int64_t>
  T Get(const Value* v) const {
    return values_.Get<T>(v);
  }

  void SetNextInstruction(const Instruction* next) { next_instruction_ = next; }

  // Sets the function return value, and implicitly the function is now
  // Finished.
  void SetReturnValue(PyObject* ret) { function_return_value_ = ret; }
  void ClearReturnValue() { function_return_value_ = absl::nullopt; }

  // Queries the `names_table` for the given string.
  PyObject* GetUnicodeObjectForStringName(const std::string& str) {
    return names_table_.at(str);
  }

 private:
  ValueMap values_;
  PyFrameObject* pyframe_;
  // This is set to non-nullptr by an Evaluate function that changes control
  // flow.
  const Instruction* next_instruction_ = nullptr;
  // Set by an ExceptInst, ReturnInst or YieldInst when the function has
  // completed.
  absl::optional<PyObject*> function_return_value_ = absl::nullopt;
  absl::flat_hash_map<std::string, PyObject*> names_table_;
  bool finish_on_bytecode_begin_ = false;
};

// The result of EvaluateFunction.
struct FunctionResult {
  // The return or yielded value of the function. If the function has not
  // completed, this is undefined.
  PyObject* return_value;

  // The final instruction executed in the function.
  const Instruction* final_instruction;
};

// Evaluates `inst` inside `ctx`. After this returns:
//   `ctx.GetNextInstruction()` returns the next instruction to execute.
//   `ctx.IsFinished()` returns true if the function has completed or yielded.
//   `ctx.GetReturnValue()` returns the returned or yielded value.
//
// Note: This is implemented in evaluator_instructions.cc.
absl::Status Evaluate(const Instruction& inst, EvaluatorContext& ctx);

// Evaluates the given Function with the given frame object and returns the
// result.
absl::StatusOr<PyObject*> EvaluateFunction(const Function& f,
                                           PyFrameObject* pyframe);

// Evaluates the given EvaluatorContext repeatedly until the function is
// finished. As a special case, it treats BytecodeBeginInst as causing the
// function to Finish.
absl::StatusOr<FunctionResult> EvaluateFunction(const Instruction& begin_inst,
                                                EvaluatorContext& ctx);

// Sets up ctx.pyframe() to be ready to jump to the interpreter. `program` is
// used to back up over EXTENDED_ARG prefixes.
void PrepareForBytecodeInterpreter(
    const SafepointInst& safepoint, EvaluatorContext& ctx,
    absl::Span<BytecodeInstruction const> program);

// Invokes the CPython interpreter and returns the result of execution. This is
// equivalent to:
//   PrepareForBytecodeInterpreter(...);
//   return EvalFrame(ctx.pyframe(), 0);
PyObject* InvokeBytecodeInterpreter(
    const SafepointInst& safepoint, EvaluatorContext& ctx,
    absl::Span<BytecodeInstruction const> program);

// Calls `callee`, with arguments and result as int64_t.
int64_t CallNative(void* callee, absl::Span<int64_t const> arguments);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_EVALUATOR_H_
