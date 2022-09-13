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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_STACK_FRAME_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_STACK_FRAME_H_

#include <Python.h>
#include <frameobject.h>
#include <pystate.h>

#include <cstdint>

#include "absl/types/span.h"
#include "runtime/util.h"

namespace deepmind::s6 {
// TODO: Remove forward reference when we move CodeObject into runtime.
class CodeObject;

// A StackFrame contains data required by generated code during runtime. Some of
// this data can be read and manipulated by runtime code.
//
// Most data - PyCodeObject, globals, builtins, previous frame - are held in a
// PyFrameObject. This object has its own dedicated register in generated code
// for performance, but is also stored in the StackFrame so that runtime
// functions can find it.
//
// TODO: The fast locals also live in the PyFrameObject, but these
// will soon be removed; generated code will not update fastlocals (they will
// all be nullptr), so new code should not rely on this.
//
// The StackFrame then contains a PyFrameObject* and S6-specific information -
// the CodeObject.
//
// The CodeObject is important because it alone knows how to interpret the
// variable sized members of the StackFrame.
//
// Beyond this fixed-size header struct there lives:
//  * Spill slots for the register allocator (optional: generators hold their
//      spill slots on the heap).
//  * Saved values of callee-saved registers from the function prolog.
//  * The call stack, used to hold data for outgoing call instructions where the
//      callee expects items in a particular place in the caller's stack frame.
//
// All of these are variable-sized, and the CodeObject contains knowledge from
// the code generator of their sizes and offsets.
//
// The StackFrame is always placed by the code generator immediately following
// the saved base pointer (RBP), so that a stack activation looks like this:
//
//   high addresses     Saved RBP    <-- RBP points here
//                         ...
//                      StackFrame   fixed location RBP+8+sizeof(StackFrame)
//                         ...
//                      Spill slots            variable location
//                      Callee-saved registers variable location
//   low addresses      Call stack slots       variable location
//                                   <-- RSP points here
//
// Note in particular that the StackFrame always follows RBP and is accessible
// at RBP+8+sizeof(StackFrame) because the stack grows *down*. The stack frame
// has a fixed size throughout the function (RSP is never changed by generated
// code), so if we have the StackFrame and the various sizes from the CodeObject
// we can calculate the expected RSP location. This is useful because when
// generated code has called some other function, the return address will be
// here. This allows us to inspect a frame and understand which strongjit
// Instruction is being executed (and thus the state of the registers and spill
// slots).
class StackFrame {
 public:
  // Returns the StackFrame for a stack activation, given the base pointer
  // (RBP) value for that stack activation.
  static StackFrame* GetFromBasePointer(ProgramAddress rbp) {
    return reinterpret_cast<StackFrame*>(rbp - sizeof(StackFrame));
  }

  // Tests to see whether this stack frame object is live. Live stack frame
  // objects contain a constant magic value that is cleared on destruction.
  // This enables us to test a native stack frame to see if it contains a live
  // stack frame object.
  bool HasValidMagic() const { return magic_ == kMagic; }

  // Clears the magic cookie to mark this stack frame as inactive.
  void ClearMagic() { magic_ = 0; }

  // Accessors for objects contained within the StackFrame.
  PyFrameObject* pyframe() const { return pyframe_; }
  PyThreadState* thread_state() const { return thread_state_; }
  CodeObject* s6_code_object() const { return s6_code_object_; }

  // True if this function was called with the fast calling convention. If
  // this is true, `pyframe()` will have been allocated by the
  // PyFrameObjectCache and must be released via it too.
  bool called_with_fast_calling_convention() const {
    return called_with_fast_calling_convention_;
  }

  // Accessors for useful objects within pyframe().
  PyObject* globals() const { return pyframe_->f_globals; }
  PyObject* builtins() const { return pyframe_->f_builtins; }
  PyObject** fastlocals() const { return pyframe_->f_localsplus; }
  int32_t bytecode_offset() const { return pyframe_->f_lasti; }
  void set_bytecode_offset(int32_t offset) { pyframe_->f_lasti = offset; }
  PyCodeObject* py_code_object() const { return pyframe_->f_code; }

  // Returns the base pointer (RBP) value for the stack activation containing
  // this StackFrame.
  ProgramAddress GetBasePointer() const {
    return reinterpret_cast<ProgramAddress>(this + 1);
  }

  // Creates a StackFrame from an existing PyFrameObject.
  // Note that `pyframe`'s reference count is NOT increased, a reference is
  // stolen.
  StackFrame(PyFrameObject* pyframe, CodeObject* s6_code_object)
      : magic_(kMagic),
        pyframe_(pyframe),
        thread_state_(PyThreadState_GET()),
        s6_code_object_(s6_code_object),
        called_with_fast_calling_convention_(false) {}

  ~StackFrame() { ClearMagic(); }

  // Magic value constant used for detecting live stack frames. A live stack
  // frame must have magic_ == kMagic.
  static constexpr uint64_t kMagic = 0x517e'a55e75'57a7eful;

  // TODO: Kept public for direct access by CodeGenerator.
  uint64_t magic_;
  PyFrameObject* pyframe_;
  PyThreadState* thread_state_;
  CodeObject* s6_code_object_;
  bool called_with_fast_calling_convention_;
};

// The layout of a stack frame. This contains the size of all variable members;
// spill slots, callee-saved registers and call stack slots.
class StackFrameLayout {
 public:
  // Returns the offset of the start of the spill slots relative to the
  // frame base pointer in bytes.
  int64_t spill_slots_offset() const { return spill_slots_offset_; }

  // Returns the number of spill slots.
  int64_t spill_slots_count() const { return spill_slots_count_; }

  // Returns the offset of the start of the callee-saved registers relative to
  // the frame base pointer in bytes.
  int64_t callee_saved_registers_offset() const {
    return callee_saved_registers_offset_;
  }

  // Returns the number of callee-saved registers.
  int64_t callee_saved_registers_count() const {
    return callee_saved_registers_count_;
  }

  // Returns the offset of the function call return address relative to the
  // base pointer object in bytes. This is the location at which the return
  // address of a `call` instruction will be saved.
  int64_t return_address_offset() const { return return_address_offset_; }

  StackFrameLayout() {}
  explicit StackFrameLayout(int64_t spill_slots_offset,
                            int64_t spill_slots_count,
                            int64_t callee_saved_registers_offset,
                            int64_t callee_saved_registers_count,
                            int64_t return_address_offset)
      : spill_slots_offset_(spill_slots_offset),
        spill_slots_count_(spill_slots_count),
        callee_saved_registers_offset_(callee_saved_registers_offset),
        callee_saved_registers_count_(callee_saved_registers_count),
        return_address_offset_(return_address_offset) {}

 private:
  int64_t spill_slots_offset_;
  int64_t spill_slots_count_;
  int64_t callee_saved_registers_offset_;
  int64_t callee_saved_registers_count_;
  int64_t return_address_offset_;
};

// Combines a StackFrame and a StackFrameLayout to produce an object that knows
// its location at runtime (StackFrame) and its internal sizes
// (StackFrameLayout).
class StackFrameWithLayout {
 public:
  StackFrameWithLayout(StackFrame& stack_frame, const StackFrameLayout& layout)
      : stack_frame_(stack_frame), layout_(layout) {}

  // Returns the spill slots of the frame as a Span.
  template <typename T>
  absl::Span<T> GetSpillSlots() const {
    static_assert(sizeof(T) == sizeof(void*),
                  "Each frame slot is 8 bytes in size.");
    T* data = reinterpret_cast<T*>(stack_frame_.GetBasePointer() +
                                   layout_.spill_slots_offset());
    return absl::MakeSpan(data, layout_.spill_slots_count());
  }

  // Returns the address of the return address.
  ProgramAddress* GetReturnAddress() {
    return reinterpret_cast<ProgramAddress*>(
        stack_frame_.GetBasePointer() + layout_.return_address_offset() - 8);
  }

 private:
  StackFrame& stack_frame_;
  StackFrameLayout layout_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_STACK_FRAME_H_
