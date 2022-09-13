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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_OBJECT_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_OBJECT_H_

#include <Python.h>
#include <frameobject.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "allocator.h"
#include "core_util.h"
#include "runtime/deoptimization_map.h"
#include "runtime/stack_frame.h"
#include "strongjit/function.h"
#include "utils/logging.h"

namespace deepmind::s6 {
// Forward declare.
class Function;
class Instruction;
class Metadata;
class ProfileCounterInfo;
class RegisterAllocation;
class Value;

// Represents a debug annotation that will be associated with an address.
class DebugAnnotation {
 public:
  DebugAnnotation() {}

  // Creates a debug annotation from the given instruction.
  explicit DebugAnnotation(const BytecodeInstruction& instruction)
      : instruction_(instruction) {}

  // Creates a debug annotation from a string. If `is_code` is false, data after
  // this annotation should be treated as data, not code.
  explicit DebugAnnotation(absl::string_view str, bool is_code = true)
      : str_(str), is_code_(is_code) {}

  // Obtains an instruction from this annotation, if one exists.
  absl::optional<BytecodeInstruction> GetInstruction() const {
    return instruction_;
  }

  // Obtains a string representation of this instruction.
  std::string ToString() const {
    if (instruction_.has_value()) {
      return instruction_->ToString();
    } else {
      return str_;
    }
  }

  // Returns true if the data following this annotation is code, false if it is
  // data.
  bool is_code() const { return is_code_; }

 private:
  absl::optional<BytecodeInstruction> instruction_;
  std::string str_;
  bool is_code_ = true;
};

// Contains a code object (function) created by fastjit. This holds the entry
// point and also compiled metadata.
class CodeObject {
 public:
  // An entry point for the ABI that consumes an existing PyFrameObject. The
  // frame object is passed as a stolen reference.
  //
  // `profile_counter`: address of the profile counter, decremented by the
  //   `profile_bytecode` instruction.
  // `code_object`: the called code's CodeObject.
  using PyFrameAbiFunctionType = PyObject* (*)(PyFrameObject* frame,
                                               int64_t* profile_counter,
                                               CodeObject* code_object);

  // An entry point for the `fast` ABI. This does not require a PyFrameObject;
  // takes arguments and a PyFunctionObject, from which the globals are derived.
  // The caller must be a Strongjit function, because the StackFrame object is
  // looked up on the caller's stack.
  //
  // The number of arguments is always exactly the number that the receiver
  // expects (co->co_argcount).
  // `r12` must already contain the address of the profile counter.
  using FastAbiFunctionType = PyObject* (*)(PyObject* pyfunction_object, ...);

  // Constructs a CodeObject with the given function body, which was allocated
  // by the given allocator. The optional bytecode program allows for better
  // disassembly.
  //
  // There are two provided entry points corresponding to the two possible ABIs
  // a function may be called with. The `fast` ABI entry point may be nullptr.
  CodeObject(PyFrameAbiFunctionType pyframe_entry,
             FastAbiFunctionType fast_entry, void* body_ptr,
             int64_t size_in_bytes, absl::string_view name,
             absl::Span<const BytecodeInstruction> program,
             absl::flat_hash_map<void*, DebugAnnotation> debug_annotations,
             JitAllocator* allocator, std::shared_ptr<Function> function,
             DeoptimizationMap deoptimization_map,
             StackFrameLayout stack_frame_layout, Metadata* metadata)
      : pyframe_entry_(pyframe_entry),
        fast_entry_(fast_entry),
        body_ptr_(body_ptr),
        body_size_(size_in_bytes),
        debug_annotations_(std::move(debug_annotations)),
        allocator_(allocator),
        name_(name),
        function_(std::move(function)),
        deoptimization_map_(absl::make_unique<DeoptimizationMap>(
            std::move(deoptimization_map))),
        stack_frame_layout_(std::move(stack_frame_layout)),
        metadata_(metadata) {
    program_.reserve(program.size());
    absl::c_copy(program, std::back_inserter(program_));
  }

  // Construct a CodeObject with the given strongjit Function and optional
  // RegisterAllocation, for use with the strongjit evaluator.
  CodeObject(std::shared_ptr<Function> function, absl::string_view name,
             absl::Span<const BytecodeInstruction> program)
      : pyframe_entry_(nullptr),
        fast_entry_(nullptr),
        body_ptr_(nullptr),
        body_size_(0),
        allocator_(nullptr),
        name_(name),
        function_(function) {
    program_.reserve(program.size());
    absl::c_copy(program, std::back_inserter(program_));
  }
  virtual ~CodeObject();

  // Returns the body of the jitted function.
  PyFrameAbiFunctionType GetPyFrameBody() const { return pyframe_entry_; }
  // Returns the "fast" ABI body for the jitted function. This may return
  // nullptr.
  FastAbiFunctionType GetFastBody() const { return fast_entry_; }

  // Returns the bytecode offset for the given program address. Only program
  // addresses that are pushed to the block stack during compilation are
  // guaranteed to have an entry in this map.
  int64_t GetBytecodeOffset(void* code_address) const {
    S6_CHECK(debug_annotations_.contains(code_address));
    absl::optional<BytecodeInstruction> inst =
        debug_annotations_.at(code_address).GetInstruction();
    S6_CHECK(inst.has_value());
    return inst->program_offset();
  }

  const absl::flat_hash_map<void*, DebugAnnotation>& debug_annotations() const {
    return debug_annotations_;
  }

  int64_t GetBodySize() const { return body_size_; }

  // Returns the disassembly of the body. If a bytecode program was provided
  // during construction, the bytecode will be interleaved in the assembly.
  std::string Disassemble() const;

  // Returns a stringified representation of the bytecode program.
  std::string ToString() const;

  // Returns the optimized Strongjit function.
  const Function& function() const { return *function_; }
  Function& function() { return *function_; }

  // Returns true if `address` is contained within the generated code.
  bool ContainsAddress(void* address) const {
    auto addressi = reinterpret_cast<uint64_t>(address);
    auto lower = reinterpret_cast<uint64_t>(body_ptr_);
    auto upper = lower + body_size_;
    return addressi >= lower && addressi < upper;
  }

  const DeoptimizationMap& deoptimization_map() const {
    S6_CHECK(deoptimization_map_);
    return *deoptimization_map_;
  }

  const StackFrameLayout& stack_frame_layout() const {
    return stack_frame_layout_;
  }

  // Is this code object deoptimized already?
  bool deoptimized() const { return deoptimized_; }
  void MarkDeoptimized() { deoptimized_ = true; }
  bool* mutable_deoptimized() { return &deoptimized_; }

  // Get the raw pointer of the deoptimized flag from the raw pointer of a
  // future code object. This is useful for code generation before the
  // construction of the code object.
  static bool* DeoptimizedPointer(void* c) {
    return reinterpret_cast<CodeObject*>(c)->mutable_deoptimized();
  }

  // Returns the bytecode program.
  absl::Span<BytecodeInstruction const> program() const { return program_; }

  // Returns the metadata this program was compiled from.
  Metadata* metadata() const { return metadata_; }

 private:
  // The function to call.
  PyFrameAbiFunctionType pyframe_entry_;
  FastAbiFunctionType fast_entry_;

  // The start of the allocated code region.
  void* body_ptr_;

  // The size in bytes of the body function.
  int64_t body_size_;

  // Debug info mapping from code address to annotation.
  absl::flat_hash_map<void*, DebugAnnotation> debug_annotations_;

  // The bytecode program.
  std::vector<BytecodeInstruction> program_;

  // The allocator used to allocate the body function.
  JitAllocator* allocator_;

  // The symbol name.
  std::string name_;

  // Stores the optimized function. The function is referred to by the
  // deoptimization handler so must outlive this CodeObject.
  // This is a shared_ptr so we don't take a dependency on Strongjit here.
  std::shared_ptr<Function> function_;

  // The deoptimization map.
  std::unique_ptr<DeoptimizationMap> deoptimization_map_;

  // The stack frame layout.
  StackFrameLayout stack_frame_layout_;

  // The metadata this was compiled for.
  Metadata* metadata_;

  bool deoptimized_ = false;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_OBJECT_H_
