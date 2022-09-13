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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_JIT_STUB_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_JIT_STUB_H_

#include <Python.h>

#include <cstdint>

#include "asmjit/asmjit.h"
#include "asmjit/x86/x86emitter.h"
#include "runtime/generator.h"
#include "runtime/stack_frame.h"
#include "runtime/util.h"

namespace deepmind::s6 {

// JitStubs provide an interface for generating code that accesses C structs in
// a more type-safe manner.
//
// Generating assembly is necessarily type-unsafe; the emitted instructions deal
// with raw pointers and offsets. Frequently when accessing structs we need to
// use `offsetof()` to find the correct offset to give to a MOV instruction.
// This makes code generation hard to read and reason about.
//
// A JitStub<T> is an interface type mirroring T. It provides methods to access
// members of T, which in turn return a new JitStub<U>. A JitStub can be emitted
// by an asmjit Emitter.
//
// A JitStub only ever represents an operand that can be represented by a single
// LEA or MOV instruction. Attempts to double-dereference will result in
// S6_CHECK failures.
//
// To perform double dereference, a JitStub needs to be "latched" into a
// register:
//   JitStub<PyObject> object = ...;
//   JitStub<PyTypeObject> type = object.type();
//
//   JitStub<int32_t> version_tag = type.version_tag(); // WRONG!
//   // The above requires double-dereferencing, so will S6_CHECK-fail.
//
//   // Perform the single dereference and store the result in RAX. The returned
//   // object now is [rax].
//   JitStub<PyTypeObject> type2 = type.Load(x86::rax, emitter); // RIGHT!
//   JitStub<int32_t> version_tag = type2.version_tag(); // RIGHT!
//
// The raw asmjit Mem operand can always be obtained with JitStub::Mem().
//
// Note that "Load()" on a JitStub that is not dereferenced emits an `lea`
// instruction instead of a `mov` instruction. `Store()` on a JitStub that
// is not dereferenced is illegal.
template <typename T>
class JitStub;

enum class DereferenceKind {
  // The x86::Mem operand contains an effective address. Load() should emit
  // an LEA instruction.
  kEffectiveAddress,
  // The x86::Mem operand must be dereferenced to obtain the effective
  // address. Load() should emit a MOV instruction.
  kDereferenced
};

// The base class of JitStub<T>. This gives common data and methods for all
// specializations of JitStub<T>.
template <typename T>
class JitStubBase {
 public:
  // Loads this JitStub object into the given register, using the given emitter.
  // This emits a single `mov` instruction, and returns a new JitStub pointing
  // at `gp`.
  JitStub<T> Load(asmjit::x86::Gp gp, asmjit::x86::Emitter& emitter) const;

  // Stores an asmjit Operand into this JitStub object.
  void Store(asmjit::Operand operand, asmjit::x86::Emitter& emitter) const;

  // Returns a copy of this JitStub but with the base register of the memory
  // operand swapped out for `gp`.
  JitStub<T> CloneWithNewBase(asmjit::x86::Gp gp) const {
    asmjit::x86::Mem operand = operand_;
    operand.setBase(gp);
    return JitStub<T>(operand, kind_);
  }

  // Returns the x86::Mem operand backing this JitStub.
  asmjit::x86::Mem Mem() const { return operand_; }

  // Returns the operand backing this JitStub as a x86::Gp.
  // REQUIRES: dereference_kind() == kEffectiveAddress
  // REQUIRES: offset == 0
  asmjit::x86::Gp Reg() const;

  DereferenceKind dereference_kind() const { return kind_; }

 protected:
  explicit JitStubBase(
      asmjit::x86::Mem operand,
      DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : operand_(operand), kind_(kind) {
    // Default to qword_ptr unless otherwise sized.
    if (!operand.hasSize()) operand.setSize(8);
  }

  // Creates a new JitStub object by dereferencing the current operand with the
  // given offset in bytes. This is the equivalent of the memory operation
  //  U = [T + offset].
  //
  // If this is already a memory operand, a dereference is not representable in
  // one X86 operand, so we get a check-failure. `mov`s will NOT be
  // automatically inserted.
  template <typename U>
  JitStub<U> Dereference(int64_t offset, int64_t size = 8) const;

 private:
  asmjit::x86::Mem operand_;
  DereferenceKind kind_;
};

// The primary template does not add any functionality above and beyond
// JitStubBase.
template <typename T>
class JitStub : public JitStubBase<T> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<T>(operand, kind) {}
};

// Explicit specialization for PyVarObject.
template <>
class JitStub<PyVarObject> : public JitStubBase<PyVarObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyVarObject>(operand, kind) {}

  JitStub<int64_t> ob_size() const {
    return Dereference<int64_t>(offsetof(PyVarObject, ob_size));
  }
};

// Explicit specialization for PyTypeObject.
template <>
class JitStub<PyTypeObject> : public JitStubBase<PyTypeObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyTypeObject>(operand, kind) {}

  JitStub<int64_t> tp_flags() const {
    return Dereference<int64_t>(offsetof(PyTypeObject, tp_flags));
  }
  JitStub<int64_t> tp_dictoffset() const {
    return Dereference<int64_t>(offsetof(PyTypeObject, tp_dictoffset));
  }
  JitStub<int32_t> tp_version_tag() const {
    return Dereference<int32_t>(offsetof(PyTypeObject, tp_version_tag),
                                /*size=*/4);
  }
};

// Explicit specialization for PyObject.
template <>
class JitStub<PyObject> : public JitStubBase<PyObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyObject>(operand, kind) {}

  JitStub<PyTypeObject> ob_type() const {
    return Dereference<PyTypeObject>(offsetof(PyObject, ob_type));
  }
  JitStub<int64_t> ob_refcnt() const {
    return Dereference<int64_t>(offsetof(PyObject, ob_refcnt));
  }
};

// Explicit specialization for PyCodeObject.
template <>
class JitStub<PyCodeObject> : public JitStubBase<PyCodeObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyCodeObject>(operand, kind) {}

  JitStub<void*> co_extras() const {
    return Dereference<void*>(offsetof(PyCodeObject, co_extra));
  }
  JitStub<int32_t> co_nlocals() const {
    return Dereference<int32_t>(offsetof(PyCodeObject, co_nlocals), /*size=*/4);
  }
};

// Explicit specialization for PyDictKeysObject.
template <>
class JitStub<PyDictKeysObject> : public JitStubBase<PyDictKeysObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyDictKeysObject>(operand, kind) {}

  JitStub<int64_t> dk_size() const {
    return Dereference<int64_t>(offsetof(_PyDictKeysObject, dk_size));
  }
  JitStub<uint8_t*> dk_indices() const {
    return Dereference<uint8_t*>(offsetof(_PyDictKeysObject, dk_indices));
  }
};

// Explicit specialization for PyDictObject.
template <>
class JitStub<PyDictObject> : public JitStubBase<PyDictObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyDictObject>(operand, kind) {}

  JitStub<PyDictKeysObject> ma_keys() const {
    return Dereference<PyDictKeysObject>(offsetof(PyDictObject, ma_keys));
  }
  JitStub<PyObject> ma_values() const {
    return Dereference<PyObject>(offsetof(PyDictObject, ma_values));
  }
  JitStub<uint64_t> ma_version_tag() const {
    return Dereference<uint64_t>(offsetof(PyDictObject, ma_version_tag));
  }
};

// Explicit specialization for PyFunctionObject.
template <>
class JitStub<PyFunctionObject> : public JitStubBase<PyFunctionObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyFunctionObject>(operand, kind) {}

  JitStub<PyCodeObject> func_code() const {
    return Dereference<PyCodeObject>(offsetof(PyFunctionObject, func_code));
  }
  JitStub<PyObject> func_globals() const {
    return Dereference<PyObject>(offsetof(PyFunctionObject, func_globals));
  }
};

// Explicit specialization for PyFrameObject.
template <>
class JitStub<PyFrameObject> : public JitStubBase<PyFrameObject> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyFrameObject>(operand, kind) {}

  JitStub<PyFrameObject> f_back() const {
    return Dereference<PyFrameObject>(offsetof(PyFrameObject, f_back));
  }
  JitStub<PyObject> f_globals() const {
    return Dereference<PyObject>(offsetof(PyFrameObject, f_globals));
  }
  JitStub<PyObject> f_builtins() const {
    return Dereference<PyObject>(offsetof(PyFrameObject, f_builtins));
  }
  JitStub<PyObject> f_locals() const {
    return Dereference<PyObject>(offsetof(PyFrameObject, f_locals));
  }
  JitStub<PyCodeObject> f_code() const {
    return Dereference<PyCodeObject>(offsetof(PyFrameObject, f_code));
  }
  JitStub<int32_t> f_lasti() const {
    return Dereference<int32_t>(offsetof(PyFrameObject, f_lasti), /*size=*/4);
  }
  JitStub<PyObject> f_stacktop() const {
    return Dereference<PyObject>(offsetof(PyFrameObject, f_stacktop));
  }
  JitStub<PyObject> fastlocals(int64_t index) const {
    return Dereference<PyObject>(offsetof(PyFrameObject, f_localsplus[0]) +
                                 index * sizeof(PyObject*));
  }
  JitStub<int64_t> ob_refcnt() const {
    return Dereference<int64_t>(
        offsetof(PyFrameObject, ob_base.ob_base.ob_refcnt));
  }
  JitStub<PyTypeObject> ob_type() const {
    return Dereference<PyTypeObject>(
        offsetof(PyFrameObject, ob_base.ob_base.ob_type));
  }
};

// Explicit specialization for PyThreadState.
template <>
class JitStub<PyThreadState> : public JitStubBase<PyThreadState> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<PyThreadState>(operand, kind) {}

  JitStub<PyFrameObject> frame() const {
    return Dereference<PyFrameObject>(offsetof(PyThreadState, frame));
  }
  JitStub<int64_t> recursion_depth() const {
    return Dereference<int64_t>(offsetof(PyThreadState, recursion_depth));
  }
};

// Explicit specialization for StackFrame.
template <>
class JitStub<StackFrame> : public JitStubBase<StackFrame> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<StackFrame>(operand, kind) {}

  JitStub<PyFrameObject> pyframe() const {
    return Dereference<PyFrameObject>(offsetof(StackFrame, pyframe_));
  }
  JitStub<PyThreadState> thread_state() const {
    return Dereference<PyThreadState>(offsetof(StackFrame, thread_state_));
  }
  JitStub<CodeObject> s6_code_object() const {
    return Dereference<CodeObject>(offsetof(StackFrame, s6_code_object_));
  }
  JitStub<bool> called_with_fast_calling_convention() const {
    return Dereference<bool>(
        offsetof(StackFrame, called_with_fast_calling_convention_), 1);
  }
  JitStub<uint64_t> magic() const {
    return Dereference<uint64_t>(offsetof(StackFrame, magic_));
  }
};

// Explicit specialization for StackFrameWithLayout.
template <>
class JitStub<StackFrameWithLayout> : public JitStub<StackFrame> {
 public:
  JitStub(asmjit::x86::Mem operand, const StackFrameLayout& layout,
          DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStub<StackFrame>(operand, kind), layout_(layout) {}

  // Returns the given spill slot, where the register allocator keeps stack
  // values.
  JitStub<void*> spill_slot(int64_t slot_index) const {
    // The spill slots are layed out relative to the base pointer. The
    // StackFrame object is immediately prior to the base pointer, so add
    // sizeof(StackFrame) to get to the base pointer.
    int64_t adjust = sizeof(StackFrame);
    // Now add the spill_slots_offset. This is negative.
    adjust += layout_.spill_slots_offset();
    return Dereference<void*>(adjust + slot_index * sizeof(void*));
  }

  // Returns the slot where we save the given callee-saved register.
  JitStub<void*> callee_saved_register_slot(int64_t slot_index) const {
    // The CSR slots are layed out relative to the base pointer. The
    // StackFrame object is immediately prior to the base pointer, so add
    // sizeof(StackFrame) to get to the base pointer.
    int64_t adjust = sizeof(StackFrame);
    // Now add the offset. This is negative.
    adjust += layout_.callee_saved_registers_offset();
    return Dereference<void*>(adjust + slot_index * sizeof(void*));
  }

 private:
  const StackFrameLayout& layout_;
};

// Explicit specialization for GeneratorState.
template <>
class JitStub<GeneratorState> : public JitStubBase<GeneratorState> {
 public:
  explicit JitStub(asmjit::x86::Mem operand,
                   DereferenceKind kind = DereferenceKind::kEffectiveAddress)
      : JitStubBase<GeneratorState>(operand, kind) {}

  JitStub<uint64_t*> spill_slots() const {
    return Dereference<uint64_t*>(offsetof(GeneratorState, spill_slots_));
  }
  JitStub<int64_t> resume_pc() const {
    return Dereference<int64_t>(offsetof(GeneratorState, resume_pc_));
  }
};

template <typename T>
template <typename U>
JitStub<U> JitStubBase<T>::Dereference(int64_t offset, int64_t size) const {
  S6_CHECK(kind_ == DereferenceKind::kEffectiveAddress)
      << "Double dereferencing is not allowed. Use Load() to latch the value "
         "into a register first.";
  asmjit::x86::Mem operand = operand_.cloneAdjusted(offset);
  operand.setSize(size);
  return JitStub<U>(operand, DereferenceKind::kDereferenced);
}

template <typename T>
JitStub<T> JitStubBase<T>::Load(asmjit::x86::Gp gp,
                                asmjit::x86::Emitter& emitter) const {
  if (kind_ == DereferenceKind::kEffectiveAddress) {
    emitter.lea(gp, operand_);
  } else {
    emitter.mov(gp, operand_);
  }
  return JitStub<T>(asmjit::x86::qword_ptr(gp),
                    DereferenceKind::kEffectiveAddress);
}

template <typename T>
void JitStubBase<T>::Store(asmjit::Operand operand,
                           asmjit::x86::Emitter& emitter) const {
  S6_CHECK(!operand.isMem())
      << "Store() of a memory operand to a memory operand "
         "is not representable.";
  S6_CHECK(kind_ == DereferenceKind::kDereferenced)
      << "Store() to a non-dereferenced address is not defined.";
  if (operand.isImm()) {
    emitter.mov(operand_, operand.as<asmjit::Imm>());
  } else {
    S6_CHECK(operand.isReg());
    emitter.mov(operand_, operand.as<asmjit::x86::Gp>());
  }
}

template <typename T>
asmjit::x86::Gp JitStubBase<T>::Reg() const {
  S6_CHECK(kind_ == DereferenceKind::kEffectiveAddress);
  S6_CHECK_EQ(operand_.offset(), 0);
  return operand_.baseReg().as<asmjit::x86::Gp>();
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_JIT_STUB_H_
