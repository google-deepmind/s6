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

// This is an implementation of frame evaluation for Python bytecodes. This
// isn't as finely tuned as the CPython ceval.c version, but it is more
// accessible for modification.
//

#include "interpreter.h"

#include <Python.h>
#include <alloca.h>
#include <opcode.h>

#include <cstddef>
#include <cstdint>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "classes/getsetattr.h"
#include "classes/object.h"
#include "classes/util.h"
#include "core_util.h"
#include "global_intern_table.h"
#include "metadata.h"
#include "runtime/python_function_info_table.h"
#include "utils/logging.h"

// Private API for the LOAD_METHOD opcode.
extern "C" int _PyObject_GetMethod(PyObject*, PyObject*, PyObject**);

namespace deepmind::s6 {

// If we take a dependency on Oracle we end up with a rather nasty dependency
// cycle. Instead, extern the only symbol we need from oracle.cc as weak.
ABSL_ATTRIBUTE_WEAK extern void OracleProfileEvent(PyFrameObject* frame,
                                                   int64_t num_bytecodes);

namespace {

using Value = PyObject*;
using StackIndex = int;

// Encapsulates the Python value stack during frame evaluation.
class ValueStack {
 public:
  ValueStack(absl::Span<Value> storage, Value* stack_pointer)
      : storage_(storage),
        sp_(stack_pointer ? stack_pointer : storage_.data()) {}

  bool Empty() const { return sp_ == storage_.data(); }

  int64_t Size() const { return sp_ - storage_.data(); }

  // Accessors and mutators. Note that for performance we very deliberately do
  // not do bounds checking here.
  Value Top() const { return sp_[-1]; }
  Value& Top() { return sp_[-1]; }
  Value Second() const { return sp_[-2]; }
  Value& Second() { return sp_[-2]; }
  Value Third() const { return sp_[-3]; }
  Value& Third() { return sp_[-3]; }
  Value Fourth() const { return sp_[-4]; }
  Value& Fourth() { return sp_[-4]; }

  Value Peek(StackIndex n) const { return sp_[-n]; }
  void Poke(StackIndex n, Value v) { sp_[-n] = v; }

  void Drop(StackIndex n) { sp_ -= n; }
  void DropAndDecref(StackIndex n) {
    while (--n >= 0) {
      Py_XDECREF(Pop());
    }
  }

  void Push(Value v) { *sp_++ = v; }
  Value Pop() { return *--sp_; }

  bool VerifyInBounds() const {
    return storage_.begin() <= sp_ && sp_ <= storage_.end();
  }

  Value* StackPointer() const { return sp_; }

  void ClearAndDecref() {
    while (!Empty()) {
      Py_XDECREF(Pop());
    }
  }

 private:
  // The backing storage that the stack uses. The stack grows upwards from
  // storage_.data().
  absl::Span<Value> storage_;
  // The current stack pointer. The stack pointer points one past the
  // top of the stack, so TOS = sp_[-1].
  Value* sp_;
};

// RAII object that sets up CPython EvalFrame state and tears it down on
// destruction.
class FrameGuard {
 public:
  FrameGuard(PyThreadState* tstate, PyFrameObject* f) : tstate_(tstate), f_(f) {
    tstate_->frame = f;
    // TODO: CPython uses the empty string, but perhaps use a more
    // S6-descriptive string when we hit the recursion limit?
    value_ = Py_EnterRecursiveCall("");
    if (!value_) f_->f_executing = 1;
  }
  ~FrameGuard() {
    if (f_) release();
  }
  void release() {
    Py_LeaveRecursiveCall();
    f_->f_executing = 0;
    tstate_->frame = f_->f_back;
    f_ = nullptr;
  }
  bool IsValid() const { return value_ == 0; }

 private:
  PyThreadState* tstate_;
  PyFrameObject* f_;
  int value_;
};

// This class maintains two variables: PC and NextPC. PC contains the currently
// executing instruction index. NextPC contains the next instruction index to
// execute.
class InstructionStream {
 public:
  // Creates an instruction stream. `last_pc` gives the last successfully
  // executed instruction. This will be None if no instructions have previously
  // been executed.
  //   PC <- last_pc_index
  //   NextPC <- last_pc_index + 1
  InstructionStream(absl::optional<PcValue> last_pc, _Py_CODEUNIT* program,
                    PyFrameObject* frame) {
    int last_pc_index = last_pc.has_value() ? last_pc->AsIndex() : -1;
    pc_ = last_pc_index;
    next_pc_ = last_pc_index + 1;
    program_ = program;
    frame_ = frame;
  }

  // Applies an adjustment to NextPC:
  //   NextPC <- NextPC + adjust
  void IncrementNextPcBy(PcValue adjust) {
    next_pc_ += adjust.AsIndex();
    OracleProfileEvent(frame_, instructions_since_last_profile_event_);
    instructions_since_last_profile_event_ = 0;
  }

  // Sets NextPC:
  //   NextPC <- value
  void SetNextPc(PcValue value) {
    next_pc_ = value.AsIndex();
    OracleProfileEvent(frame_, instructions_since_last_profile_event_);
    instructions_since_last_profile_event_ = 0;
  }

  // Peeks back in the instruction stream without rewinding the iterator.
  _Py_CODEUNIT Previous() {
    S6_CHECK_GT(pc_, 0);
    return program_[pc_ - 1];
  }

  // Advances the iterator and returns the next instruction in the stream.
  //   PC <- NextPC
  //   NextPC <- PC + 1
  _Py_CODEUNIT AdvanceAndFetch() {
    pc_ = next_pc_;
    next_pc_ = pc_ + 1;
    // Don't call OracleProfileEvent on every instruction fetch; only when
    // we jump. This reduces the frequency of event calls while also ensuring
    // we don't defer events too long.
    ++instructions_since_last_profile_event_;
    return program_[pc_];
  }

  // Returns the current PC.
  PcValue GetPc() const { return PcValue::FromIndex(pc_); }

  // Returns the next PC.
  PcValue GetNextPc() const { return PcValue::FromIndex(next_pc_); }

  // Returns true if the instruction stream is valid; this returns true until
  // Terminate() is called.
  bool IsValid() const { return !terminated_; }

  // Terminates the instruction stream. It is no longer correct to call any
  // function other than IsValid() or GetTerminatingValue().
  void Terminate(PyObject* return_value = nullptr) {
    terminated_ = true;
    return_value_ = return_value;
    if (instructions_since_last_profile_event_ > 0) {
      OracleProfileEvent(frame_, instructions_since_last_profile_event_);
    }
  }

  // Returns the value given to Terminate().
  PyObject* GetTerminatingValue() const { return return_value_; }

 private:
  int64_t pc_;
  int64_t next_pc_;
  _Py_CODEUNIT* program_;
  bool terminated_ = false;
  PyObject* return_value_ = nullptr;
  PyFrameObject* frame_;
  int64_t instructions_since_last_profile_event_ = 0;
};

// TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION >= 7
#define EXC_INFO(x) exc_state.x
#else
#define EXC_INFO(x) x
#endif

// Encapsulates the CPython block stack during frame evaluation. Blocks are
// pushed and popped in response to lexical constructs; loops and try/catch/
// finally are examples.
//
// Blocks are associated with a sequence of values on the value stack.
class BlockStack {
 public:
  BlockStack(PyThreadState* tstate, PyFrameObject* f, ValueStack* value_stack)
      : tstate_(tstate), f_(f), value_stack_(value_stack) {}

  // Pushes a new block onto the stack.
  //   `opcode`: Type of the block e.g. EXCEPT_HANDLER. This is a Python opcode.
  //   `handler`: Program counter value for the handler.
  void Push(int opcode, PcValue handler) {
    S6_CHECK(f_->f_iblock < CO_MAXBLOCKS);
    PyTryBlock& b = f_->f_blockstack[f_->f_iblock++];
    b.b_type = opcode;
    b.b_level = value_stack_->Size();
    b.b_handler = handler.AsOffset();
  }

  // Pops a block from the block stack due to a Return instruction. The only
  // reason a return may jump elsewhere is due to a Finally handler; if such a
  // handler is found, this function returns the absolute PC location to jump
  // to. The return value and kReturn should be pushed to the stack so that
  // END_FINALLY can resume the return.
  absl::optional<PcValue> Return() {
    while (f_->f_iblock > 0) {
      PyTryBlock& b = f_->f_blockstack[--f_->f_iblock];
      if (b.b_type == EXCEPT_HANDLER) {
        // Exception handlers are unwound differently to all other blocks.
        UnwindExceptHandler(b);
        continue;
      }
      UnwindBlock(b);
      if (b.b_type == SETUP_FINALLY) {
        return PcValue::FromOffset(b.b_handler);
      }
    }
    return {};
  }

  // Unwinds the block stack due to a continue statement. If the unwind finds an
  // finally handler, it is returned and should be jumped to after pushing
  // oparg and kContinue to the stack so that END_FINALLY can resume the
  // continue.
  // If not, returns None and oparg should be jumped to.
  absl::optional<PcValue> Continue() {
    while (f_->f_iblock > 0) {
      PyTryBlock& b = f_->f_blockstack[f_->f_iblock - 1];
      if (b.b_type == SETUP_LOOP) {
        return {};
      }

      --f_->f_iblock;
      if (b.b_type == EXCEPT_HANDLER) {
        // Exception handlers are unwound differently to all other blocks.
        UnwindExceptHandler(b);
        continue;
      }
      UnwindBlock(b);
      if (b.b_type == SETUP_FINALLY) {
        return PcValue::FromOffset(b.b_handler);
      }
    }
    S6_LOG(FATAL) << "Continue outside loop!";
  }

  // Unwinds the block stack due to a break statement. Returns a pair
  // <next_pc, push_kBreak?>. If the unwind finds an finally handler, it is
  // returned and should be jumped to after pushing kBreak to the stack
  // (push_kBreak = true) so that END_FINALLY can resume the break.
  // If not, returns the head of the loop which should be
  // jumped to without pushing anything (push_kBreak = false).
  std::pair<PcValue, bool> Break() {
    while (f_->f_iblock > 0) {
      PyTryBlock& b = f_->f_blockstack[--f_->f_iblock];
      if (b.b_type == EXCEPT_HANDLER) {
        // Exception handlers are unwound differently to all other blocks.
        UnwindExceptHandler(b);
        continue;
      }
      UnwindBlock(b);
      if (b.b_type == SETUP_LOOP) {
        return {PcValue::FromOffset(b.b_handler), false};
      }
      if (b.b_type == SETUP_FINALLY) {
        return {PcValue::FromOffset(b.b_handler), true};
      }
    }
    S6_LOG(FATAL) << "Break outside loop!";
  }

  // True if there is an exception handler block on the block stack.
  bool InsideExcept() {
    for (int i = 0; i < f_->f_iblock; ++i) {
      if (f_->f_blockstack[i].b_type == EXCEPT_HANDLER) return true;
    }
    return false;
  }

  // Unwinds the block stack due to taking an Exception. The exception
  // information is in PyThreadState::exc_{type, value, traceback}.
  //
  // If the unwind finds an exception handler, it is returned and should be
  // jumped to. If not, returns nullopt.
  //
  // In the case an exception handler was found, we push a new block
  // EXCEPT_HANDLER on the block stack to signal we are handling an exception.
  // Then the old exception state is pushed to the stack (3 values),
  // and then the current exception state is pushed to the stack (3 values).
  // The old exception state belongs to the EXCEPT_HANDLER block and will be
  // popped when exiting the EXCEPT_HANDLER block via UnwindExceptHandler.
  // The current exception state on the stack is to be used by the exception
  // handling code as needed. It also has the correct format to be immediately
  // raised again by END_FINALLY.
  absl::optional<PcValue> Exception() {
    while (f_->f_iblock > 0) {
      PyTryBlock& b = f_->f_blockstack[--f_->f_iblock];
      if (b.b_type == EXCEPT_HANDLER) {
        // Exception handlers are unwound differently to all other blocks.
        UnwindExceptHandler(b);
        continue;
      }
      UnwindBlock(b);
      if (b.b_type != SETUP_EXCEPT && b.b_type != SETUP_FINALLY) {
        continue;
      }
      int handler = b.b_handler;
      // Beware, this invalidates all b.b_* fields.
      PyFrame_BlockSetup(f_, EXCEPT_HANDLER, -1, value_stack_->Size());
      value_stack_->Push(tstate_->EXC_INFO(exc_traceback));
      value_stack_->Push(tstate_->EXC_INFO(exc_value));
      if (tstate_->EXC_INFO(exc_type)) {
        value_stack_->Push(tstate_->EXC_INFO(exc_type));
      } else {
        Py_INCREF(Py_None);
        value_stack_->Push(Py_None);
      }
      PyObject *exc_type, *exc_value, *exc_traceback;
      PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
      // Make the raw exception data available to the handler, so a program
      // can emulate the Python main loop.
      PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);
      PyException_SetTraceback(exc_value,
                               exc_traceback ? exc_traceback : Py_None);
      Py_INCREF(exc_type);
      tstate_->EXC_INFO(exc_type) = exc_type;
      Py_INCREF(exc_value);
      tstate_->EXC_INFO(exc_value) = exc_value;
      tstate_->EXC_INFO(exc_traceback) = exc_traceback;
      Py_INCREF(exc_traceback ? exc_traceback : Py_None);
      value_stack_->Push(exc_traceback);
      value_stack_->Push(exc_value);
      value_stack_->Push(exc_type);
      return PcValue::FromOffset(handler);
    }
    return {};
  }

  // Pops the top block from the block stack and unwinds it.
  void Pop() {
    PyTryBlock& b = f_->f_blockstack[--f_->f_iblock];
    UnwindBlock(b);
  }

  // Pops the top block from the block stack and unwinds it. The block must be
  // of type EXCEPT_HANDLER.
  void PopExceptHandler() {
    PyTryBlock& b = f_->f_blockstack[--f_->f_iblock];
    S6_CHECK_EQ(b.b_type, EXCEPT_HANDLER);
    UnwindExceptHandler(b);
  }

  const PyTryBlock& Top() { return f_->f_blockstack[f_->f_iblock - 1]; }

 private:
  // Unwinds `b`, which is of type EXCEPT_HANDLER, from the block stack. This
  // pops all values down to b_level apart from the last three, which are
  // written into the thread state's exception state.
  void UnwindExceptHandler(const PyTryBlock& b) {
    S6_DCHECK_GE(value_stack_->Size(), b.b_level + 3);
    while (value_stack_->Size() > b.b_level + 3) {
      Py_XDECREF(value_stack_->Pop());
    }
    PyObject* type = tstate_->EXC_INFO(exc_type);
    PyObject* value = tstate_->EXC_INFO(exc_value);
    PyObject* traceback = tstate_->EXC_INFO(exc_traceback);
    tstate_->EXC_INFO(exc_type) = value_stack_->Pop();
    tstate_->EXC_INFO(exc_value) = value_stack_->Pop();
    tstate_->EXC_INFO(exc_traceback) = value_stack_->Pop();
    // Decrement reference counts for the previous exception state last, to
    // ensure that we have a valid exception state for their finalizers.
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
  }

  // Unwinds `b`, which is not of type EXCEPT_HANDLER, from the block stack.
  // This pops all values down to b_level.
  void UnwindBlock(const PyTryBlock& b) {
    S6_DCHECK_GE(value_stack_->Size(), b.b_level);
    while (value_stack_->Size() > b.b_level) {
      Py_XDECREF(value_stack_->Pop());
    }
  }

  PyThreadState* tstate_;
  PyFrameObject* f_;
  ValueStack* value_stack_;
};

// Calls _PyObject_LookupSpecial, and sets AttributeError if it returns nullptr
// without setting an exception.
PyObject* SpecialLookup(PyObject* object, _Py_Identifier* identifier) {
  PyObject* res = _PyObject_LookupSpecial(object, identifier);
  if (!res && !PyErr_Occurred()) {
    PyErr_SetObject(PyExc_AttributeError, identifier->object);
    return nullptr;
  }
  return res;
}

// "Normal" unary op handling. Pop the top of the stack, pass to handler. If
// handler returns nullptr, call `handle_exception()`.
template <typename T, typename F>
void HandleUnaryOp(T handler, ValueStack* stack,
                   ClassDistribution& type_feedback, F handle_exception) {
  type_feedback.Add(GetClassId(stack->Top()));

  PyObject* res = handler(stack->Top());
  Py_DECREF(stack->Pop());
  stack->Push(res);
  if (!res) handle_exception();
}

// "Normal" binary op handling. Pop RHS then LHS off the stack, pass to handler.
// If handler returns nullptr, call `handle_exception`.
template <typename T, typename F>
void HandleBinaryOp(T handler, ValueStack* stack,
                    ClassDistribution& type_feedback, F handle_exception) {
  type_feedback.Add(GetClassId(stack->Top()));
  type_feedback.Add(GetClassId(stack->Second()));

  PyObject* res = handler(stack->Second(), stack->Top());
  Py_DECREF(stack->Pop());
  Py_DECREF(stack->Pop());
  stack->Push(res);
  if (!res) handle_exception();
}

// "Power" binary op handling. Pop RHS then LHS off the stack, pass to handler.
// If handler returns nullptr, call `handle_exception`.
template <typename T, typename F>
void HandlePowerOp(T handler, ValueStack* stack,
                   ClassDistribution& type_feedback, F handle_exception) {
  PyObject* lhs = stack->Second();
  PyObject* rhs = stack->Top();
  // Only gather feedback for the LHS type.
  // We want to be able to optimise squaring floats (e.g. 3.4 ** 2).
  // So, the LHS drives the type analysis, and the instruction should not be
  // considered polymorphic if the RHS type differs.
  type_feedback.Add(GetClassId(lhs));

  PyObject* res = handler(lhs, rhs, Py_None);
  Py_DECREF(stack->Pop());
  Py_DECREF(stack->Pop());
  stack->Push(res);
  if (!res) handle_exception();
}

PyObject* ImportName(PyFrameObject* f, PyObject* name, PyObject* fromlist,
                     PyObject* level) {
  _Py_IDENTIFIER(__import__);
  PyObject* import_func = _PyDict_GetItemId(f->f_builtins, &PyId___import__);

  PyObject* result;
  if (!import_func) {
    PyErr_SetString(PyExc_ImportError, "__import__ not found");
    result = nullptr;
  } else {
    // TODO: Implement the missing fast path from the CPython
    // interpreter for non-overloaded __import__.
    Py_INCREF(import_func);
    PyObject* args[] = {name, f->f_globals, f->f_locals ? f->f_locals : Py_None,
                        fromlist, level};
    result = _PyObject_FastCall(import_func, args, /*nargs=*/5);
    Py_DECREF(import_func);
  }
  Py_DECREF(level);
  Py_DECREF(fromlist);
  return result;
}

// Imports all names from `object` into scope.
bool ImportAllFrom(PyFrameObject* frame, PyObject* object) {
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return false;
  }

  PyObject* locals = frame->f_locals;
  if (!locals) {
    PyErr_SetString(PyExc_SystemError, "no locals found during 'import *'");
    return false;
  }

  _Py_IDENTIFIER(__all__);
  PyObject* all = _PyObject_GetAttrId(object, &PyId___all__);
  bool skip_leading_underscores = false;
  if (!all) {
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
      // Unexpected error.
      return false;
    }
    PyErr_Clear();
    _Py_IDENTIFIER(__dict__);
    PyObject* dict = _PyObject_GetAttrId(object, &PyId___dict__);
    if (!dict) {
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_SetString(PyExc_ImportError,
                        "from-import-* object has no __dict__ and no __all__");
      }
      return false;
    }
    all = PyMapping_Keys(dict);
    Py_DECREF(dict);
    if (!all) return false;
    skip_leading_underscores = true;
  }

  int index = 0;
  bool ok = true;
  do {
    PyObject* name = PySequence_GetItem(all, index++);
    if (!name) {
      // CPython does not iterate over the size of the sequence, so neither do
      // we.
      if (PyErr_ExceptionMatches(PyExc_IndexError)) {
        PyErr_Clear();
      } else {
        ok = false;
      }
      break;
    }

    if (skip_leading_underscores && PyUnicode_Check(name)) {
      if (PyUnicode_READY(name) == -1) {
        Py_DECREF(name);
        ok = false;
        break;
      }
      if (PyUnicode_READ_CHAR(name, 0) == '_') {
        Py_DECREF(name);
        continue;
      }
    }

    PyObject* value = PyObject_GetAttr(object, name);
    if (!value) {
      Py_DECREF(name);
      ok = false;
      break;
    }

    int error = PyDict_CheckExact(locals)
                    ? PyDict_SetItem(locals, name, value)
                    : PyObject_SetItem(locals, name, value);
    Py_DECREF(name);
    Py_DECREF(value);
    ok = error == 0;
  } while (ok);

  Py_DECREF(all);
  PyFrame_LocalsToFast(frame, 0);
  Py_DECREF(object);
  return ok;
}

// Imports `name` from `object`.
PyObject* ImportFrom(PyObject* object, PyObject* name) {
  PyObject* value = PyObject_GetAttr(object, name);
  if (value) {
    return value;
  } else if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
    return nullptr;
  }

  // From CPython:
  //
  // Issue #17636: in case this failed because of a circular relative import,
  // try to fallback on reading the module directly from sys.modules.
  PyErr_Clear();
  _Py_IDENTIFIER(__name__);
  PyObject* package_name = _PyObject_GetAttrId(object, &PyId___name__);
  if (package_name) {
    if (PyUnicode_Check(package_name)) {
      PyObject* full_module_name =
          PyUnicode_FromFormat("%U.%U", package_name, name);
      Py_DECREF(package_name);
      if (!full_module_name) {
        return nullptr;
      }
      value = PyDict_GetItem(PyImport_GetModuleDict(), full_module_name);
      Py_DECREF(full_module_name);
      if (value) {
        Py_INCREF(value);
        return value;
      }
    } else {
      Py_CLEAR(package_name);
    }
  }

  // The generic error cases flow here.
  PyErr_Format(PyExc_ImportError, "cannot import name %R", name);
  return nullptr;
}

PyObject* Compare(int op, PyObject* left, PyObject* right) {
  switch (op) {
    case PyCmp_IS:
      if (left == right) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case PyCmp_IS_NOT:
      if (left != right) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case PyCmp_IN: {
      int contains = PySequence_Contains(right, left);
      if (contains < 0) {
        return nullptr;
      } else if (contains != 0) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    }
    case PyCmp_NOT_IN: {
      int contains = PySequence_Contains(right, left);
      if (contains < 0) {
        return nullptr;
      } else if (contains == 0) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    }
    case PyCmp_EXC_MATCH: {
      const char* kMessage =
          "catching classes that do not inherit from BaseException is not "
          "allowed";
      if (PyTuple_Check(right)) {
        int length = PyTuple_Size(right);
        for (int i = 0; i < length; ++i) {
          if (!PyExceptionClass_Check(PyTuple_GET_ITEM(right, i))) {
            PyErr_SetString(PyExc_TypeError, kMessage);
            return nullptr;
          }
        }
      } else if (!PyExceptionClass_Check(right)) {
        PyErr_SetString(PyExc_TypeError, kMessage);
        return nullptr;
      }

      int matches = PyErr_GivenExceptionMatches(left, right);
      if (matches < 0) {
        return nullptr;
      } else if (matches != 0) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    }
    default:
      return PyObject_RichCompare(left, right, op);
  }
}

PyObject* BuildListUnpack(int count, ValueStack* stack,
                          PyObject* func = nullptr) {
  auto* list = reinterpret_cast<PyListObject*>(PyList_New(0));
  if (!list) {
    return nullptr;
  }

  for (int i = count; i > 0; --i) {
    PyObject* iterable = stack->Peek(i);
    PyObject* none = _PyList_Extend(list, iterable);
    if (!none) {
      // func(*args) call where args is not iterable.
      if (func && PyErr_ExceptionMatches(PyExc_TypeError)) {
        if (!(Py_TYPE(iterable)->tp_iter || PySequence_Check(iterable))) {
          FormatIterableError(iterable, func);
        }
      }
      Py_DECREF(list);
      return nullptr;
    }
    Py_DECREF(none);
  }
  stack->DropAndDecref(count);
  return reinterpret_cast<PyObject*>(list);
}

PyObject* BuildSetUnpack(int count, ValueStack* stack) {
  PyObject* set = PySet_New(/*iterable=*/nullptr);
  if (!set) {
    return nullptr;
  }

  for (int i = count; i > 0; --i) {
    if (_PySet_Update(set, stack->Peek(i)) < 0) {
      Py_DECREF(set);
      return nullptr;
    }
  }
  stack->DropAndDecref(count);
  return reinterpret_cast<PyObject*>(set);
}

PyObject* BuildMapUnpack(int count, ValueStack* stack,
                         PyObject* func = nullptr) {
  PyObject* map = PyDict_New();
  if (!map) {
    return nullptr;
  }

  for (int i = count; i > 0; --i) {
    PyObject* other = stack->Peek(i);
    // Raise if a function is given, otherwise use the last value for each key.
    if (_PyDict_MergeEx(map, other, /*override=*/func ? 2 : 1) == 0) {
      continue;
    }

    if (func) {
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        FormatMappingError(other, func);
      } else if (PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyObject *exc_type, *exc_value, *exc_tb;
        PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
        if (exc_value && PyTuple_Check(exc_value) &&
            PyTuple_GET_SIZE(exc_value) == 1) {
          PyObject* key = PyTuple_GET_ITEM(exc_value, 0);
          if (!PyUnicode_Check(key)) {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%.200s keywords must be strings",
                         PyEval_GetFuncName(func), PyEval_GetFuncDesc(func));
          } else {
            PyErr_Format(PyExc_TypeError,
                         "%.200s%.200s got multiple "
                         "values for keyword argument '%U'",
                         PyEval_GetFuncName(func), PyEval_GetFuncDesc(func),
                         key);
          }
          Py_XDECREF(exc_type);
          Py_XDECREF(exc_value);
          Py_XDECREF(exc_tb);
        } else {
          PyErr_Restore(exc_type, exc_value, exc_tb);
        }
      }
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
      PyErr_Format(PyExc_TypeError, "'%.200s' object is not a mapping",
                   Py_TYPE(other)->tp_name);
    }
    Py_DECREF(map);
    return nullptr;
  }
  stack->DropAndDecref(count);
  return reinterpret_cast<PyObject*>(map);
}

// Constants indicating how calls, returns, and exceptions are instrumented.
enum class TraceMode {
  kTracing,
  kProfiling,
};

// Calls the indicated tracing function, unless we're already inside the tracing
// function.  The return value is non-zero if an exception was raised, and zero
// on success.
int CallTrace(TraceMode mode, PyThreadState* tstate, PyFrameObject* frame,
              int what, PyObject* arg) {
  if (tstate->tracing) return 0;
  tstate->tracing++;
  tstate->use_tracing = 0;
  int result =
      mode == TraceMode::kTracing
          ? tstate->c_tracefunc(tstate->c_traceobj, frame, what, arg)
          : tstate->c_profilefunc(tstate->c_profileobj, frame, what, arg);
  tstate->use_tracing =
      tstate->c_tracefunc != nullptr || tstate->c_profilefunc != nullptr;
  tstate->tracing--;
  return result;
}

// Performs profiling bookkeeping before entering a C function.
bool ProfileCEntry(PyThreadState* tstate, PyObject* function) {
  int status = CallTrace(TraceMode::kProfiling, tstate, tstate->frame,
                         PyTrace_C_CALL, function);
  return status == 0;
}

// Performs profiling bookkeeping after exiting a C function.
PyObject* ProfileCExit(PyThreadState* tstate, PyObject* function,
                       PyObject* result) {
  // This mirrors the logic in CPython: we check again if profiling is enabled
  // after the call.
  if (!tstate->c_profilefunc) return result;

  // If there was an exception, instrument it.
  if (!result) {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    int status = CallTrace(TraceMode::kProfiling, tstate, tstate->frame,
                           PyTrace_C_EXCEPTION, function);
    if (status != 0) {
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    } else {
      PyErr_Restore(type, value, traceback);
    }
    return nullptr;
  }

  // If there was no exception, instrument the return.
  int status = CallTrace(TraceMode::kProfiling, tstate, tstate->frame,
                         PyTrace_C_RETURN, function);
  if (status != 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}

// Calls the callable that is below the `arg_count` arguments on the value
// stack.
// Returns false exactly when the call raised an exception.
bool CallFunction(PyThreadState* tstate, ValueStack* stack, int arg_count,
                  PyObject* names) {
  // The callable is under the arguments.  After the call we will remove the
  // arguments and the callable.
  PyObject* callable = stack->Peek(arg_count + 1);
  int unwind_height = stack->Size() - arg_count - 1;

  int keyword_count = names ? PyTuple_Size(names) : 0;
  PyObject* result = nullptr;

  if (PyCFunction_Check(callable)) {
    int positional_count = arg_count - keyword_count;
    PyObject** arg_base = stack->StackPointer() - arg_count;
    bool use_profiling =
        tstate->use_tracing != 0 && tstate->c_profilefunc != nullptr;
    if (!use_profiling) {
      result = _PyCFunction_FastCallKeywords(callable, arg_base,
                                             positional_count, names);
    } else if (ProfileCEntry(tstate, callable)) {
      result = ProfileCExit(tstate, callable,
                            _PyCFunction_FastCallKeywords(
                                callable, arg_base, positional_count, names));
    }
  } else {
    if (PyMethod_Check(callable)) {
      PyObject* self = PyMethod_GET_SELF(callable);
      if (self) {
        // Optimize bound methods by directly calling the original method.
        // Overwrite the function on the value stack with the bound method's
        // self value.
        ++arg_count;
        stack->Poke(arg_count, self);
        Py_INCREF(self);

        PyObject* method = callable;
        callable = PyMethod_GET_FUNCTION(method);
        // It can be the case that the only reference to `function` is from
        // `method` so be careful about the order of INCREF and DECREF.
        Py_INCREF(callable);
        Py_DECREF(method);
      }
    }
    int positional_count = arg_count - keyword_count;
    PyObject** arg_base = stack->StackPointer() - arg_count;
    result = PyFunction_Check(callable)
                 ? _PyFunction_FastCallKeywords(callable, arg_base,
                                                positional_count, names)
                 : _PyObject_FastCallKeywords(callable, arg_base,
                                              positional_count, names);
  }

  Py_XDECREF(names);
  while (stack->Size() > unwind_height) {
    Py_XDECREF(stack->Pop());
  }
  stack->Push(result);

  return result != nullptr;
}

// Resumes a stack unwinding that was intercepted by a finally block.
// This is called at the end of the finally block by END_FINALLY.
// The reason cannot be kException, kYield or kNormal because they are
// not handled that way by END_FINALLY.
//
// When a finally block intercepts a control-flow operation like
// break, continue, return, etc. It will push the exit_status on the TOS,
// so that at the end of the finally block, this control-flow operation can
// resumed.
void ResumeUnwind(Why why, InstructionStream* pc, ValueStack* stack,
                  BlockStack* block_stack) {
  switch (why) {
    case Why::kBreak: {
      const auto [new_pc, push_kbreak] = block_stack->Break();
      if (push_kbreak) {
        stack->Push(PyLong_FromWhy(Why::kBreak));
      }
      pc->SetNextPc(new_pc);
      break;
    }
    case Why::kContinue: {
      PyObject* retval = stack->Pop();
      if (absl::optional<PcValue> new_pc = block_stack->Continue()) {
        stack->Push(retval);
        stack->Push(PyLong_FromWhy(Why::kContinue));
        pc->SetNextPc(*new_pc);
      } else {
        pc->SetNextPc(PcValue::FromOffset(PyLong_AS_LONG(retval)));
        Py_DECREF(retval);
      }
      break;
    }
    case Why::kReturn: {
      PyObject* retval = stack->Pop();
      if (absl::optional<PcValue> next_pc = block_stack->Return()) {
        stack->Push(retval);
        stack->Push(PyLong_FromWhy(Why::kReturn));
        pc->SetNextPc(*next_pc);
        break;
      }
      // No block to pop to; actually return.
      pc->Terminate(retval);
      break;
    }
    case Why::kSilenced:
      // An exception was silenced by 'with'. We must manually unwind
      // the EXCEPT_HANDLER block which was created when the exception
      // was caught, otherwise the stack will be in an inconsistent
      // state.
      block_stack->PopExceptHandler();
      break;

    default:
      S6_LOG(FATAL) << "Impossible case for stack unwinder";
  }
}

// Sets the error indicator and formats an error message if `object` is non-null
// and can be converted to a string.  `exception` should be a Python exception
// class, `format_string` is a C printf format string with one format specifier
// and `object` is the value for that specifier.
void FormatError(PyObject* exception, const char* format_string,
                 PyObject* object) {
  if (!object) return;
  const char* object_string = PyUnicode_AsUTF8(object);
  if (!object_string) return;
  PyErr_Format(exception, format_string, object_string);
}

constexpr const char* kNameErrorMessage = "name '%.200s' is not defined";
constexpr const char* kUnboundLocalErrorMessage =
    "local variable '%.200s' referenced before assignment";
constexpr const char* kUnboundFreeErrorMessage =
    "free variable '%.200s' referenced before assignment in enclosing scope";

// Sets the error indicator for an unbound variable error.  The kind of error
// depends on whether `index` is an index into the code object's closed-over
// variables or into its free variables.
void FormatUnboundError(PyCodeObject* code, int index) {
  // Do not overwrite an existing exception.
  if (PyErr_Occurred()) return;

  if (index < PyTuple_GET_SIZE(code->co_cellvars)) {
    FormatError(PyExc_UnboundLocalError, kUnboundLocalErrorMessage,
                PyTuple_GET_ITEM(code->co_cellvars, index));
    return;
  }

  index -= PyTuple_GET_SIZE(code->co_cellvars);
  FormatError(PyExc_NameError, kUnboundFreeErrorMessage,
              PyTuple_GET_ITEM(code->co_freevars, index));
}

// Unpacks exactly count values from an iterable into the stack. The stack
// is updated downwards starting from TOS.
//
// Returns true if unpacking was successful. Otherwise sets an exception
// and returns false.
bool UnpackIterable(PyObject* iterable, int count, ValueStack* stack) {
  S6_CHECK_GE(stack->Size(), count);
  PyObject* it = PyObject_GetIter(iterable);
  if (!it) {
#if PY_MINOR_VERSION >= 7
    PyErr_Format(PyExc_TypeError, "cannot unpack non-iterable %.200s object",
                 iterable->ob_type->tp_name);
#endif
    return false;
  }

  for (int i = 0; i < count; ++i) {
    if (PyObject* value = PyIter_Next(it)) {
      stack->Poke(i + 1, value);
    } else {
      // Iteration stopped either due to an error or because the iterator is
      // exhausted.
      if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError,
                     "not enough values to unpack (expected %d, got %d)", count,
                     i);
      }

      Py_DECREF(it);
      return false;
    }
  }

  // Ensure that the iterator is exhausted.
  if (PyObject* next = PyIter_Next(it)) {
    PyErr_Format(PyExc_ValueError, "too many values to unpack (expected %d)",
                 count);
    Py_DECREF(next);
    Py_DECREF(it);
    return false;
  } else {
    Py_DECREF(it);
    return PyErr_Occurred() == nullptr;
  }
}

// Similar to UnpackIterable() but supports "catch-all" name which is
// assigned a list of values not assigned to other names. See PEP-3123 for
// details.
//
// For example, here
//
//   a, b, *rest, c = iterable
//
// names a and b come before *rest ("catch-all" name), and name c -- after.
// Therefore before == 2 and after == 1.
//
// Returns true if unpacking was successful. Otherwise sets an exception
// and returns false.
bool ExtendedUnpackIterable(PyObject* iterable, int before, int after,
                            ValueStack* stack) {
  PyObject* it = PyObject_GetIter(iterable);
  if (!it) {
    return false;
  }

  int i = 0;
  // ++i because TOS is updated as ValueStack::Poke(-1, ...).
  while (++i <= before) {
    if (PyObject* value = PyIter_Next(it)) {
      stack->Poke(i, value);
    } else {
      // Iteration stopped either due to an error or because the iterator is
      // exhausted.
      if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError,
                     "not enough values to unpack "
                     "(expected at least %d, got %d)",
                     before + after, i - 1);
      }

      Py_DECREF(it);
      return false;
    }
  }

  PyObject* extra = PySequence_List(it);
  if (!extra) {
    Py_DECREF(it);
    return false;
  }

  stack->Poke(i++, extra);

  std::size_t remaining = PyList_GET_SIZE(extra);
  if (remaining < after) {
    PyErr_Format(PyExc_ValueError,
                 "not enough values to unpack (expected at least %d, got %zd)",
                 before + after, before + remaining);
    Py_DECREF(it);
    return false;
  }

  // Move the after values from the list into the stack.
  for (int j = after; j > 0; --j) {
    stack->Poke(i++, PyList_GET_ITEM(extra, remaining - j));
  }
  Py_SIZE(extra) = remaining - after;
  Py_DECREF(it);
  return true;
}

// Looks up `name` in globals and if that fails, in builtins.  Returns nullptr
// if there was an exception.
PyObject* LoadGlobal(PyObject* name, PyObject* globals, PyObject* builtins) {
  // CPython uses _PyDict_LoadGlobal if builtins is a PyDictObject.  If we do
  // not implement caching for global access we could consider
  // _PyDict_LoadGlobal here as well.

  // 1. Look in globals.  Globals is expected to be a PyDictObject.
  PyObject* result = PyDict_GetItem(globals, name);
  if (result) {
    return result;
  }

  // 2. Look in builtins.  It can be an arbitrary Python object.
  if (PyDict_CheckExact(builtins)) {
    result = PyDict_GetItem(builtins, name);
    if (!result) {
      FormatError(PyExc_NameError, kNameErrorMessage, name);
    }
  } else {
    result = PyObject_GetItem(builtins, name);
    if (!result && PyErr_ExceptionMatches(PyExc_KeyError)) {
      FormatError(PyExc_NameError, kNameErrorMessage, name);
    }
  }

  return result;
}
}  // namespace

// An exception has occurred and a trace function is registered with `tstate`.
// Call it.
void CallExceptionTrace(PyThreadState* tstate, PyFrameObject* f) {
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  if (!value) {
    value = Py_None;
    Py_INCREF(value);
  }
  PyErr_NormalizeException(&type, &value, &traceback);
  PyObject* arg = PyTuple_Pack(3, type, value, traceback ? traceback : Py_None);
  if (!arg) {
    PyErr_Restore(type, value, traceback);
    return;
  }
  int err = CallTrace(TraceMode::kTracing, tstate, f, PyTrace_EXCEPTION, arg);
  Py_DECREF(arg);
  if (err == 0) {
    PyErr_Restore(type, value, traceback);
  } else {
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
  }
}

// Sets an exception with a given __cause__ attribute.
void Raise(PyObject* exc, PyObject* cause) {
  PyObject* type = nullptr;
  PyObject* value = nullptr;
  if (PyExceptionClass_Check(exc)) {
    type = exc;
    value = PyObject_CallObject(exc, /*args=*/nullptr);
    if (!value) {
      Py_DECREF(type);
      return;
    }
    if (!PyExceptionInstance_Check(value)) {
      PyErr_Format(PyExc_TypeError,
                   "calling %R should have returned an instance of "
                   "BaseException, not %R",
                   type, Py_TYPE(value));
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(cause);
      return;
    }
  } else if (PyExceptionInstance_Check(exc)) {
    value = exc;
    type = PyExceptionInstance_Class(exc);
    Py_INCREF(type);
  } else {
    Py_DECREF(exc);
    Py_XDECREF(cause);
    PyErr_SetString(PyExc_TypeError,
                    "exceptions must derive from BaseException");
    return;
  }

  if (cause) {
    PyObject* fixed_cause = nullptr;
    if (PyExceptionClass_Check(cause)) {
      fixed_cause = PyObject_CallObject(cause, /*args=*/nullptr);
      if (!fixed_cause) {
        Py_DECREF(cause);
        return;
      }
    } else if (PyExceptionInstance_Check(cause)) {
      fixed_cause = cause;
      Py_INCREF(fixed_cause);
    } else if (cause != Py_None) {
      PyErr_SetString(PyExc_TypeError,
                      "exception causes must derive from BaseException");
      Py_DECREF(cause);
      return;
    }
    PyException_SetCause(value, fixed_cause);
  }

  PyErr_SetObject(type, value);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(cause);
}

PyObject* EvalFrame(PyFrameObject* f, int throwflag) {
  PyCodeObject* co = f->f_code;
  Metadata* meta = Metadata::Get(co);
  if (co->co_argcount > 0) {
    meta = meta->SelectSpecialization(f->f_localsplus[0]);
  }
  PyThreadState* tstate = PyThreadState_GET();
  if (tstate->use_tracing) return _PyEval_EvalFrameDefault(f, throwflag);

  if (meta->has_compilation_failed()) {
    // This function has failed compilation. We will never successfully compile
    // this function, so go directly to CPython's interpreter. The only thing
    // we gain from our interpreter is type feedback, everything else is less
    // performant than the hyper-optimized CPython interpreter loop.
    return _PyEval_EvalFrameDefault(f, throwflag);
  }
  tstate->frame = f;

  // Push frame, and pop when we leave.
  FrameGuard frame_guard(tstate, f);
  if (!frame_guard.IsValid()) return nullptr;

  S6_DVLOG(1) << "Evaluating frame with "
              << PyObjectToString(reinterpret_cast<PyObject*>(co));

  // Help the compiler out with LICM and alias analysis by caching important
  // values as locals.
  PyObject* builtins = f->f_builtins;
  PyObject* globals = f->f_globals;
  PyObject* locals = f->f_locals;
  PyObject** fastlocals = f->f_localsplus;
  PyObject** freevars = &fastlocals[co->co_nlocals];
  PyObject* consts = co->co_consts;
  PyObject* names = co->co_names;

  ValueStack stack(absl::MakeSpan(f->f_valuestack, co->co_stacksize),
                   f->f_stacktop);
  BlockStack block_stack(tstate, f, &stack);

  // Set the stacktop to nullptr to copy what CPython does.
  f->f_stacktop = nullptr;

  // Attempt to adopt the globals dict, if we haven't seen it before. We do this
  // once per function call because we really don't want to do this work on
  // every LOAD_GLOBAL.
  AdoptGlobalsDict(reinterpret_cast<PyDictObject*>(globals));

  absl::optional<PcValue> last_pc;
  if (f->f_lasti >= 0) last_pc = PcValue::FromOffset(f->f_lasti);
  InstructionStream pc(
      last_pc, reinterpret_cast<_Py_CODEUNIT*>(PyBytes_AS_STRING(co->co_code)),
      f);

  // Escape hatch - ask CPython to continue from our current state.
  // If `no_insts_executed` is true, the usual check for PyErr_Occurred() should
  // be elided as we're escaping while still validating the frame input
  // conditions, not as a result of executing an instruction.
  auto escape_to_cpython = [&](bool no_insts_executed = false) -> PyObject* {
    f->f_stacktop = stack.StackPointer();
    // f_lasti is expected to be up-to-date. This is what allows CPython to pick
    // up after we left off.
    // Release the frame guard; CPython will enter a new recursive call itself.
    frame_guard.release();
    // The op name is already printed under S6_DVLOG(1) at the start of the
    // interpreter switch.
    S6_DVLOG(2) << "Unhandleable op -> escape()";
    if (!PyErr_Occurred() && !no_insts_executed) {
      // We are escaping to CPython without an error having occurred, so back up
      // one instruction in the static instruction stream so CPython repeats the
      // instruction we bailed out on.
      f->f_lasti =
          std::max<int>(-1, pc.GetPc().AsOffset() - sizeof(_Py_CODEUNIT));
    }
    return _PyEval_EvalFrameDefault(
        f, throwflag != 0 || PyErr_Occurred() != nullptr);
  };

  // Handle an exception. On return from this function the user must immediately
  // break out of the interpreter switch and take another interpreter loop
  // iteration.
  auto handle_exception = [&]() -> void {
    meta->set_except_observed(true);
    S6_CHECK(PyErr_Occurred());
    PyTraceBack_Here(f);
    if (tstate->c_tracefunc) {
      CallExceptionTrace(tstate, f);
    }
    if (absl::optional<PcValue> new_pc = block_stack.Exception()) {
      pc.SetNextPc(*new_pc);
    } else {
      pc.Terminate();
    }
  };

  // TODO: Removed in 3.7.
#if PY_MINOR_VERSION < 7
  // On entry to a generator's or coroutine's frame we have to save the caller's
  // exception state and possibly restore the generator's exception state.
  const int kGeneratorOrCoroutine =
      CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR;
  if (co->co_flags & kGeneratorOrCoroutine) {
    if (throwflag || !f->f_exc_type || f->f_exc_type == Py_None) {
      SaveExceptionState(tstate, f);
    } else {
      SwapExceptionState(tstate, f);
    }
  }
#endif

  // Support for generator.throw.  Set the PC to a local exception handler or
  // return an exception value.
  if (throwflag) handle_exception();
  // Reset throwflag in case we bail out to CPython.
  throwflag = 0;

  // Carryover higher bits of oparg from EXTENDED_ARG instruction.
  absl::optional<int> oparg_carry;

  // Main interpreter loop
  while (pc.IsValid()) {
    S6_DCHECK(stack.VerifyInBounds());
    S6_DCHECK(!PyErr_Occurred());

    // TODO: Yield periodically.

    _Py_CODEUNIT instr = pc.AdvanceAndFetch();

    int oparg = _Py_OPARG(instr);
    if (oparg_carry.has_value()) {
      oparg |= *oparg_carry << 8;
      oparg_carry = absl::nullopt;
    }

    int opcode = _Py_OPCODE(instr);
    S6_DVLOG(2) << pc.GetPc().AsIndex() << ": "
                << BytecodeOpcodeToString(opcode);

    // Update the last successfully executed instruction to be this instruction.
    // While this is technically false - we haven't yet executed it successfully
    // - CPython eagerly advances this before executing anything also. If we
    // escape_to_cpython() without PyErr_Occurred(), escape_to_cpython() will
    // back this up by one instruction (in the static program order) to ask
    // CPython to retry execution.
    f->f_lasti = pc.GetPc().AsOffset();

    if (tstate->use_tracing) {
      S6_DVLOG(1) << "Tracing enabled -> escape()";
      return escape_to_cpython();
    }

    ClassDistribution& type_feedback = meta->GetTypeFeedback(pc.GetPc());

    switch (opcode) {
      case NOP: {
        // Do nothing code. Used as a placeholder by the bytecode optimizer.
        break;
      }

      case LOAD_FAST: {
        // Pushes a reference to the local co_varnames[var_num] onto the stack.
        PyObject* value = fastlocals[oparg];
        if (!value) {
          FormatError(PyExc_UnboundLocalError, kUnboundLocalErrorMessage,
                      PyTuple_GetItem(co->co_varnames, oparg));
          handle_exception();
          break;
        }
        Py_INCREF(value);
        stack.Push(value);
        break;
      }

      case LOAD_CONST: {
        // Pushes co_consts[consti] onto the stack.
        PyObject* value = PyTuple_GET_ITEM(consts, oparg);
        Py_INCREF(value);
        stack.Push(value);
        break;
      }

      case STORE_FAST: {
        // Stores TOS into the local co_varnames[var_num].
        PyObject* old_value = fastlocals[oparg];
        fastlocals[oparg] = stack.Pop();
        Py_XDECREF(old_value);
        break;
      }

      case POP_TOP: {
        // Removes the top-of-stack (TOS) item.
        Py_DECREF(stack.Pop());
        break;
      }

      case ROT_TWO: {
        // Swaps the two top-most stack items.
        std::swap(stack.Top(), stack.Second());
        break;
      }

      case ROT_THREE: {
        // Lifts second and third stack item one position up, moves top down
        // to position three.
        std::swap(stack.Top(), stack.Second());
        std::swap(stack.Second(), stack.Third());
        break;
      }

      case DUP_TOP: {
        // Duplicates the reference on top of the stack.
        Py_INCREF(stack.Top());
        stack.Push(stack.Top());
        break;
      }

      case DUP_TOP_TWO: {
        // Duplicates the two references on top of the stack, leaving them in
        // the same order.
        Py_INCREF(stack.Second());
        Py_INCREF(stack.Top());
        stack.Push(stack.Second());
        // Old Top() is now Second().
        stack.Push(stack.Second());
        break;
      }

      case UNARY_POSITIVE: {
        // Implements TOS = +TOS.
        HandleUnaryOp(&PyNumber_Positive, &stack, type_feedback,
                      handle_exception);
        break;
      }

      case UNARY_NEGATIVE: {
        // Implements TOS = -TOS.
        HandleUnaryOp(&PyNumber_Negative, &stack, type_feedback,
                      handle_exception);
        break;
      }

      case UNARY_NOT: {
        // Implements TOS = not TOS.
        int err = PyObject_IsTrue(stack.Top());
        Py_DECREF(stack.Pop());
        if (err < 0) {
          handle_exception();
          break;
        }
        PyObject* obj = err == 0 ? Py_True : Py_False;
        Py_INCREF(obj);
        stack.Push(obj);
        break;
      }

      case UNARY_INVERT: {
        // Implements TOS = ~TOS.
        HandleUnaryOp(&PyNumber_Invert, &stack, type_feedback,
                      handle_exception);
        break;
      }

      case BINARY_POWER: {
        // Implements TOS = TOS1 ** TOS.
        HandlePowerOp(&PyNumber_Power, &stack, type_feedback, handle_exception);
        break;
      }

      case BINARY_MULTIPLY: {
        // Implements TOS = TOS1 * TOS.
        HandleBinaryOp(&PyNumber_Multiply, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_MATRIX_MULTIPLY: {
        // Implements in-place TOS = TOS1 @ TOS.
        HandleBinaryOp(&PyNumber_MatrixMultiply, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_TRUE_DIVIDE: {
        // Implements TOS = TOS1 / TOS.
        HandleBinaryOp(&PyNumber_TrueDivide, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_FLOOR_DIVIDE: {
        // Implements TOS = TOS1 // TOS.
        HandleBinaryOp(&PyNumber_FloorDivide, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_MODULO: {
        // Implements TOS = TOS1 % TOS.
        auto handler = [](PyObject* dividend, PyObject* divisor) {
          if (PyUnicode_CheckExact(dividend) &&
              (!PyUnicode_Check(divisor) || PyUnicode_CheckExact(divisor))) {
            // Fast path, but not if the RHS is a str subclass.
            // See issue28598.
            return PyUnicode_Format(dividend, divisor);
          }
          return PyNumber_Remainder(dividend, divisor);
        };
        HandleBinaryOp(handler, &stack, type_feedback, handle_exception);
        break;
      }

      case BINARY_ADD: {
        // Implements TOS = TOS1 + TOS.
        if (PyUnicode_CheckExact(stack.Top()) &&
            PyUnicode_CheckExact(stack.Second())) {
          // TODO: CPython has an optimization here that looks for
          // a following STORE and optimizes a refcount away, so the
          // append can be done in-place.
          PyUnicode_Append(&stack.Second(), stack.Top());
          type_feedback.Add(GetClassId(stack.Second()));
          type_feedback.Add(GetClassId(stack.Top()));
          // Pop Top(), leaving Second() as Top.
          Py_DECREF(stack.Pop());
          break;
        }
        HandleBinaryOp(&PyNumber_Add, &stack, type_feedback, handle_exception);
        break;
      }

      case BINARY_SUBTRACT: {
        // Implements TOS = TOS1 - TOS.
        HandleBinaryOp(&PyNumber_Subtract, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_SUBSCR: {
        // Implements TOS = TOS1[TOS].
        HandleBinaryOp(&PyObject_GetItem, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_LSHIFT: {
        // Implements TOS = TOS1 << TOS.
        HandleBinaryOp(&PyNumber_Lshift, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_RSHIFT: {
        // Implements TOS = TOS1 >> TOS.
        HandleBinaryOp(&PyNumber_Rshift, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case BINARY_AND: {
        // Implements TOS = TOS1 & TOS.
        HandleBinaryOp(&PyNumber_And, &stack, type_feedback, handle_exception);
        break;
      }

      case BINARY_XOR: {
        // Implements TOS = TOS1 ^ TOS.
        HandleBinaryOp(&PyNumber_Xor, &stack, type_feedback, handle_exception);
        break;
      }

      case BINARY_OR: {
        // Implements TOS = TOS1 | TOS.
        HandleBinaryOp(&PyNumber_Or, &stack, type_feedback, handle_exception);
        break;
      }

      case LIST_APPEND: {
        // Calls list.append(TOS1[-i], TOS). Used to implement comprehensions.
        //
        // NOTE: While the added value or key/value pair is popped off, the
        // container object remains on the stack so that it is available for
        // further iterations of the loop.
        PyObject* v = stack.Pop();
        int err = PyList_Append(stack.Peek(oparg), v);
        Py_DECREF(v);
        if (err != 0) {
          handle_exception();
        }
        break;
      }

      case SET_ADD: {
        // Calls set.add(TOS1[-i], TOS). Used to implement set comprehensions.
        //
        // NOTE: While the added value or key/value pair is popped off, the
        // container object remains on the stack so that it is available for
        // further iterations of the loop.
        PyObject* v = stack.Pop();
        int err = PySet_Add(stack.Peek(oparg), v);
        Py_DECREF(v);
        if (err != 0) {
          handle_exception();
        }
        break;
      }

      case INPLACE_POWER: {
        // Implements in-place TOS = TOS1 ** TOS.
        HandlePowerOp(&PyNumber_InPlacePower, &stack, type_feedback,
                      handle_exception);
        break;
      }

      case INPLACE_MULTIPLY: {
        // Implements in-place TOS = TOS1 * TOS.
        HandleBinaryOp(&PyNumber_InPlaceMultiply, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_MATRIX_MULTIPLY: {
        // Implements in-place TOS = TOS1 @ TOS.
        HandleBinaryOp(&PyNumber_InPlaceMatrixMultiply, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_TRUE_DIVIDE: {
        // Implements in-place TOS = TOS1 / TOS.
        HandleBinaryOp(&PyNumber_InPlaceTrueDivide, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_FLOOR_DIVIDE: {
        // Implements in-place TOS = TOS1 // TOS.
        HandleBinaryOp(&PyNumber_InPlaceFloorDivide, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_MODULO: {
        // Implements in-place TOS = TOS1 % TOS.
        HandleBinaryOp(&PyNumber_InPlaceRemainder, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_ADD: {
        // Implements in-place TOS = TOS1 + TOS.
        if (PyUnicode_CheckExact(stack.Top()) &&
            PyUnicode_CheckExact(stack.Second())) {
          // TODO: CPython has an optimization here that looks for
          // a following STORE and optimizes a refcount away, so the
          // append can be done in-place.
          PyUnicode_Append(&stack.Second(), stack.Top());
          // Pop Top(), leaving Second() as Top.
          Py_DECREF(stack.Pop());
          break;
        }
        HandleBinaryOp(&PyNumber_InPlaceAdd, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_SUBTRACT: {
        // Implements in-place TOS = TOS1 - TOS.
        HandleBinaryOp(&PyNumber_InPlaceSubtract, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_LSHIFT: {
        // Implements in-place TOS = TOS1 << TOS.
        HandleBinaryOp(&PyNumber_InPlaceLshift, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_RSHIFT: {
        // Implements in-place TOS = TOS1 >> TOS.
        HandleBinaryOp(&PyNumber_InPlaceRshift, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_AND: {
        // Implements in-place TOS = TOS1 & TOS.
        HandleBinaryOp(&PyNumber_InPlaceAnd, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_XOR: {
        // Implements in-place TOS = TOS1 ^ TOS.
        HandleBinaryOp(&PyNumber_InPlaceXor, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case INPLACE_OR: {
        // Implements in-place TOS = TOS1 | TOS.
        HandleBinaryOp(&PyNumber_InPlaceOr, &stack, type_feedback,
                       handle_exception);
        break;
      }

      case STORE_SUBSCR: {
        PyObject* subscript = stack.Pop();
        PyObject* container = stack.Pop();
        PyObject* value = stack.Pop();
        int err = PyObject_SetItem(container, subscript, value);
        Py_DECREF(value);
        Py_DECREF(container);
        Py_DECREF(subscript);
        if (err != 0) handle_exception();
        break;
      }

// TODO: Removed in 3.7.
#if PY_MINOR_VERSION < 7
      case STORE_ANNOTATION: {
        if (!locals) {
          PyErr_Format(PyExc_SystemError,
                       "no locals found when storing annotation");
          handle_exception();
          break;
        }

        // 1. Get __annotations__ from locals.
        _Py_IDENTIFIER(__annotations__);
        PyObject* annotations;
        if (PyDict_CheckExact(locals)) {
          annotations = _PyDict_GetItemId(locals, &PyId___annotations__);
          Py_XINCREF(annotations);
        } else {
          PyObject* annotations_string =
              _PyUnicode_FromId(&PyId___annotations__);
          if (!annotations_string) {
            handle_exception();
            break;
          }
          annotations = PyObject_GetItem(locals, annotations_string);
        }
        if (!annotations) {
          PyErr_SetString(PyExc_NameError, "__annotations__ not found");
          handle_exception();
          break;
        }

        // 2. The annotation is on the stack and the name index is in oparg.
        PyObject* annotation = stack.Pop();
        PyObject* name = PyTuple_GET_ITEM(names, oparg);

        // 3. __annotations__[name] = annotation.
        int error = PyDict_CheckExact(annotations)
                        ? PyDict_SetItem(annotations, name, annotation)
                        : PyObject_SetItem(annotations, name, annotation);
        Py_DECREF(annotations);
        Py_DECREF(annotation);
        if (error != 0) handle_exception();
        break;
      }
#endif

      case DELETE_SUBSCR: {
        PyObject* subscript = stack.Pop();
        PyObject* container = stack.Pop();
        int err = PyObject_DelItem(container, subscript);
        Py_DECREF(container);
        Py_DECREF(subscript);
        if (err != 0) handle_exception();
        break;
      }

      case PRINT_EXPR: {
        _Py_IDENTIFIER(displayhook);
        PyObject* value = stack.Pop();
        PyObject* hook = _PySys_GetObjectId(&PyId_displayhook);
        if (!hook) {
          PyErr_SetString(PyExc_RuntimeError, "lost sys.displayhook");
          Py_DECREF(value);
          handle_exception();
          break;
        }
        PyObject* result = PyObject_CallFunctionObjArgs(hook, value, nullptr);
        Py_DECREF(value);
        if (!result) {
          handle_exception();
          break;
        }
        Py_DECREF(result);
        break;
      }

      case LOAD_BUILD_CLASS: {
        // Pushes builtins.__build_class__() onto the stack. It is later called
        // by CALL_FUNCTION to construct a class.
        _Py_IDENTIFIER(__build_class__);

        PyObject* res = nullptr;
        if (PyDict_CheckExact(builtins)) {
          res = _PyDict_GetItemId(builtins, &PyId___build_class__);
          Py_XINCREF(res);
        } else {
          PyObject* build_class_str = _PyUnicode_FromId(&PyId___build_class__);
          if (!build_class_str) {
            handle_exception();
            break;
          }

          res = PyObject_GetItem(f->f_builtins, build_class_str);
          if (!res && !PyErr_ExceptionMatches(PyExc_KeyError)) {
            handle_exception();
            break;
          }
        }

        if (!res) {
          PyErr_SetString(PyExc_NameError, "__build_class__ not found");
          handle_exception();
          break;
        }

        stack.Push(res);
        break;
      }

      case STORE_NAME: {
        // Implements name = TOS.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* value = stack.Pop();
        if (!locals) {
          PyErr_Format(PyExc_SystemError, "no locals found when storing %R",
                       name);
          Py_DECREF(value);
          handle_exception();
          break;
        }
        int err = PyDict_CheckExact(locals)
                      ? PyDict_SetItem(locals, name, value)
                      : PyObject_SetItem(locals, name, value);
        Py_DECREF(value);
        if (err != 0) {
          handle_exception();
        }
        break;
      }

      case DELETE_NAME: {
        // Implements del name, where oparg is the index into co_names
        // attribute of the code object.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        if (!locals) {
          PyErr_Format(PyExc_SystemError, "no locals when deleting %R", name);
          handle_exception();
        } else if (PyDict_DelItem(locals, name) < 0) {
          FormatError(PyExc_NameError, kNameErrorMessage, name);
          handle_exception();
        }
        break;
      }

      case UNPACK_SEQUENCE: {
        // Unpacks TOS into oparg individual values, which are put onto the
        // stack right-to-left.
        int count = oparg;
        PyObject* seq = stack.Pop();
        type_feedback.Add(GetClassId(seq));
        if (PyTuple_CheckExact(seq) && PyTuple_GET_SIZE(seq) == count) {
          while (--count >= 0) {
            PyObject* item = PyTuple_GET_ITEM(seq, count);
            Py_INCREF(item);
            stack.Push(item);
          }
        } else if (PyList_CheckExact(seq) && PyList_GET_SIZE(seq) == count) {
          while (--count >= 0) {
            PyObject* item = PyList_GET_ITEM(seq, count);
            Py_INCREF(item);
            stack.Push(item);
          }
        } else {
          for (int i = 0; i < count; ++i) {
            stack.Push(nullptr);
          }
          if (!UnpackIterable(seq, count, &stack)) {
            Py_DECREF(seq);
            handle_exception();
            break;
          }
        }
        Py_DECREF(seq);
        break;
      }

      case UNPACK_EX: {
        // Implements assignment with a starred target: Unpacks an iterable in
        // TOS into individual values, where the total number of values can be
        // smaller than the number of items in the iterable: one of the new
        // values will be a list of all leftover items.
        int before = oparg & 0xFF;
        int after = oparg >> 8;
        int total = before + 1 + after;
        PyObject* seq = stack.Pop();
        for (int i = 0; i < total; ++i) {
          stack.Push(nullptr);
        }
        if (!ExtendedUnpackIterable(seq, before, after, &stack)) {
          handle_exception();
          break;
        }
        Py_DECREF(seq);
        break;
      }

      case STORE_ATTR: {
        // Implements TOS.name = TOS1, where oparg is the index of name in
        // co_names.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* owner = stack.Pop();
        PyObject* value = stack.Pop();
        S6_DCHECK(value);
        // Obtain the class ID of owner *before* it possibly transitions.
        int64_t class_id = GetClassId(owner);
        int err = PyObject_SetAttr(owner, name, value);
        Py_DECREF(owner);
        Py_DECREF(value);
        if (err != 0) {
          handle_exception();
        } else {
          type_feedback.Add(class_id);
        }
        break;
      }

      case DELETE_ATTR: {
        // Implements del TOS.name, using oparg as index into co_names.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* owner = stack.Pop();
        int err = PyObject_SetAttr(owner, name, /*value=*/nullptr);
        Py_DECREF(owner);
        if (err != 0) {
          handle_exception();
        }
        break;
      }

      case STORE_GLOBAL: {
        // Works as STORE_NAME, but stores the name as a global.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* value = stack.Pop();
        int err = SetAttrForGlobalsDict(globals, name, value);
        Py_DECREF(value);
        if (err != 0) {
          handle_exception();
        } else {
          type_feedback.Add(
              GetGlobalsClassId(reinterpret_cast<PyDictObject*>(globals)));
        }
        break;
      }

      case DELETE_GLOBAL: {
        // Works as DELETE_NAME, but deletes a global name.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        if (PyDict_DelItem(globals, name) < 0) {
          FormatError(PyExc_NameError, kNameErrorMessage, name);
          handle_exception();
        }
        break;
      }

      case LOAD_NAME: {
        // Pushes the value associated with co_names[oparg] onto the stack.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        if (!locals) {
          PyErr_Format(PyExc_SystemError, "no locals when loading %R", name);
          handle_exception();
          break;
        }

        PyObject* value;
        // 1. Look in locals.
        if (PyDict_CheckExact(locals)) {
          value = PyDict_GetItem(locals, name);
        } else {
          value = PyObject_GetItem(locals, name);
          if (!value) {
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
              PyErr_Clear();
            } else {
              handle_exception();
              break;
            }
          }
        }

        if (!value) {
          // 2. Look in globals and builtins.
          value = LoadGlobal(name, globals, builtins);
          if (!value) {
            handle_exception();
            break;
          }
        }

        Py_INCREF(value);
        stack.Push(value);
        break;
      }

      case LOAD_GLOBAL: {
        // Loads the global named co_names[oparg] onto the stack.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        type_feedback.Add(
            GetGlobalsClassId(reinterpret_cast<PyDictObject*>(globals)));
        PyObject* value = LoadGlobal(name, globals, builtins);
        if (!value) {
          handle_exception();
          break;
        }
        Py_INCREF(value);
        stack.Push(value);
        break;
      }

      case DELETE_FAST: {
        // Deletes local co_varnames[var_num].
        PyObject* old_value = fastlocals[oparg];
        if (!old_value) {
          FormatError(PyExc_UnboundLocalError, kUnboundLocalErrorMessage,
                      PyTuple_GetItem(co->co_varnames, oparg));
          handle_exception();
          break;
        }
        fastlocals[oparg] = nullptr;
        Py_DECREF(old_value);
        break;
      }

      case DELETE_DEREF: {
        PyObject* cell = freevars[oparg];
        if (!PyCell_GET(cell)) {
          FormatUnboundError(co, oparg);
          handle_exception();
          break;
        }
        PyCell_Set(cell, nullptr);
        break;
      }

      case LOAD_CLOSURE: {
        PyObject* cell = freevars[oparg];
        Py_INCREF(cell);
        stack.Push(cell);
        break;
      }

      case LOAD_DEREF: {
        PyObject* cell = freevars[oparg];
        PyObject* value = PyCell_GET(cell);
        if (!value) {
          FormatUnboundError(co, oparg);
          handle_exception();
          break;
        }
        Py_INCREF(value);
        stack.Push(value);
        break;
      }

      case LOAD_CLASSDEREF: {
        // Much like LOAD_DEREF but first checks the locals dictionary before
        // consulting the cell. This is used for loading free variables in class
        // bodies.
        S6_DCHECK(locals);
        int idx = oparg - PyTuple_GET_SIZE(co->co_cellvars);
        S6_DCHECK_GE(idx, 0);
        S6_DCHECK_LT(idx, PyTuple_GET_SIZE(co->co_freevars));
        PyObject* name = PyTuple_GET_ITEM(co->co_freevars, idx);
        PyObject* value = nullptr;
        if (PyDict_CheckExact(locals)) {
          value = PyDict_GetItem(locals, name);
          Py_XINCREF(value);
        } else {
          value = PyObject_GetItem(locals, name);
          if (!value) {
            if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
              handle_exception();
              break;
            }

            PyErr_Clear();
          }
        }

        if (!value) {
          PyObject* cell = freevars[oparg];
          value = PyCell_GET(cell);
          if (!value) {
            FormatUnboundError(co, oparg);
            handle_exception();
            break;
          }
          Py_INCREF(value);
        }
        stack.Push(value);
        break;
      }

      case STORE_DEREF: {
        PyObject* value = stack.Pop();
        PyObject* cell = freevars[oparg];
        PyObject* previous = PyCell_GET(cell);
        PyCell_SET(cell, value);
        Py_XDECREF(previous);
        break;
      }

      case BUILD_STRING: {
        // Concatenates count strings from the stack and pushes the resulting
        // string onto the stack.
        PyObject* empty = PyUnicode_New(/*size=*/0, /*maxchar=*/0);
        if (!empty) {
          handle_exception();
          break;
        }

        PyObject* str = _PyUnicode_JoinArray(
            /*separator=*/empty, /*items=*/stack.StackPointer() - oparg,
            /*seqlen=*/oparg);
        Py_DECREF(empty);
        if (!str) {
          handle_exception();
          break;
        }

        stack.DropAndDecref(oparg);
        stack.Push(str);
        break;
      }

      case BUILD_TUPLE: {
        // Creates a tuple consuming count items from the stack, and pushes
        // the resulting tuple onto the stack.
        PyObject* tuple = PyTuple_New(oparg);
        if (!tuple) {
          handle_exception();
          break;
        }
        while (--oparg >= 0) {
          PyTuple_SET_ITEM(tuple, oparg, stack.Pop());
        }
        stack.Push(tuple);
        break;
      }

      case BUILD_TUPLE_UNPACK: {
        // Pops oparg iterables from the stack, joins them in a single tuple,
        // and pushes the result.
        PyObject* list = BuildListUnpack(oparg, &stack);
        if (!list) {
          handle_exception();
          break;
        }

        PyObject* tuple = PyList_AsTuple(list);
        Py_DECREF(list);
        if (!tuple) {
          handle_exception();
          break;
        }

        stack.Push(tuple);
        break;
      }

      case BUILD_TUPLE_UNPACK_WITH_CALL: {
        // This is similar to BUILD_TUPLE_UNPACK, but is used for
        // func(*x, *y, *z) call syntax.
        PyObject* list =
            BuildListUnpack(oparg, &stack, /*func=*/stack.Peek(oparg + 1));
        if (!list) {
          handle_exception();
          break;
        }

        PyObject* tuple = PyList_AsTuple(list);
        Py_DECREF(list);
        if (!tuple) {
          handle_exception();
          break;
        }

        stack.Push(tuple);
        break;
      }

      case BUILD_LIST: {
        // Works as BUILD_TUPLE, but creates a list.
        PyObject* list = PyList_New(oparg);
        if (!list) {
          handle_exception();
          break;
        }
        while (--oparg >= 0) {
          PyList_SET_ITEM(list, oparg, stack.Pop());
        }
        stack.Push(list);
        break;
      }

      case BUILD_LIST_UNPACK: {
        // This is similar to BUILD_TUPLE_UNPACK, but pushes a list instead of
        // tuple.
        if (PyObject* list = BuildListUnpack(oparg, &stack)) {
          stack.Push(list);
        } else {
          handle_exception();
        }
        break;
      }

      case BUILD_SET: {
        // Works as BUILD_TUPLE, but creates a set.
        PyObject* set = PySet_New(/*iterable=*/nullptr);
        if (!set) {
          handle_exception();
          break;
        }
        int err = 0;
        for (int i = oparg; i > 0; --i) {
          PyObject* item = stack.Peek(i);
          // Do not break out on error to DECREF remaining items.
          if (err == 0) {
            err = PySet_Add(set, item);
          }
          Py_DECREF(item);
        }
        stack.Drop(oparg);
        if (err != 0) {
          Py_DECREF(set);
          handle_exception();
        } else {
          stack.Push(set);
        }
        break;
      }

      case BUILD_SET_UNPACK: {
        // This is similar to BUILD_TUPLE_UNPACK, but pushes a set instead
        // of tuple.
        if (PyObject* set = BuildSetUnpack(oparg, &stack)) {
          stack.Push(set);
        } else {
          handle_exception();
        }
        break;
      }

      case BUILD_MAP: {
        PyObject* map = _PyDict_NewPresized(static_cast<Py_ssize_t>(oparg));
        if (!map) {
          handle_exception();
          break;
        }
        int err = 0;
        for (int i = oparg; i > 0; --i) {
          PyObject* key = stack.Peek(2 * i);
          PyObject* value = stack.Peek(2 * i - 1);
          // Do not break out on error to DECREF remaining items.
          if (err == 0) {
            err = PyDict_SetItem(map, key, value);
          }
          Py_DECREF(key);
          Py_DECREF(value);
        }
        stack.Drop(oparg * 2);
        if (err != 0) {
          Py_DECREF(map);
          handle_exception();
        } else {
          stack.Push(map);
        }
        break;
      }

      case SETUP_ANNOTATIONS: {
        if (!locals) {
          PyErr_Format(PyExc_SystemError,
                       "no locals found when setting up annotations");
          handle_exception();
          break;
        }

        _Py_IDENTIFIER(__annotations__);
        // This string will be non-null exactly when locals is not a PyDict.
        PyObject* annotations_string = nullptr;
        if (!PyDict_CheckExact(locals)) {
          annotations_string = _PyUnicode_FromId(&PyId___annotations__);
          if (!annotations_string) {
            handle_exception();
            break;
          }
        }

        // 1. Check if __annotations__ is in locals.
        if (annotations_string) {
          PyObject* annotations = PyObject_GetItem(locals, annotations_string);
          if (annotations) {
            Py_DECREF(annotations);
            break;
          } else if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            // __annotations__ was not found.
            PyErr_Clear();
          } else {
            // There was some unexpected error.
            handle_exception();
            break;
          }
        } else {
          if (_PyDict_GetItemId(locals, &PyId___annotations__)) break;
        }

        // 2. If not, create it.
        PyObject* annotations = PyDict_New();
        if (!annotations) {
          handle_exception();
          break;
        }

        // 3. And store it in locals.
        int error =
            annotations_string
                ? PyObject_SetItem(locals, annotations_string, annotations)
                : _PyDict_SetItemId(locals, &PyId___annotations__, annotations);
        Py_DECREF(annotations);
        if (error != 0) handle_exception();
        break;
      }

      case BUILD_CONST_KEY_MAP: {
        // The version of BUILD_MAP specialized for constant keys.
        PyObject* keys = stack.Pop();
        if (!PyTuple_CheckExact(keys) ||
            PyTuple_Size(keys) != static_cast<Py_ssize_t>(oparg)) {
          PyErr_SetString(PyExc_SystemError,
                          "bad BUILD_CONST_KEY_MAP keys argument");
          handle_exception();
          break;
        }
        PyObject* map = _PyDict_NewPresized(static_cast<Py_ssize_t>(oparg));
        if (!map) {
          handle_exception();
          break;
        }

        int err = 0;
        for (int i = oparg; i > 0; --i) {
          PyObject* value = stack.Peek(i);
          // Do not break out on error to DECREF remaining items.
          if (err == 0) {
            PyObject* key = PyTuple_GET_ITEM(keys, oparg - i);
            err = PyDict_SetItem(map, key, value);
          }
          Py_DECREF(value);
        }
        Py_DECREF(keys);
        stack.Drop(oparg);
        if (err != 0) {
          Py_DECREF(map);
          handle_exception();
        } else {
          stack.Push(map);
        }
        break;
      }

      case BUILD_MAP_UNPACK: {
        if (PyObject* map = BuildMapUnpack(oparg, &stack)) {
          stack.Push(map);
        } else {
          handle_exception();
        }
        break;
      }

      case BUILD_MAP_UNPACK_WITH_CALL: {
        // This is similar to BUILD_MAP_UNPACK, but is used for
        // func(**x, **y, **z) call syntax.
        if (PyObject* map =
                BuildMapUnpack(oparg, &stack, /*func=*/stack.Peek(oparg + 2))) {
          stack.Push(map);
        } else {
          handle_exception();
        }
        break;
      }

      case MAP_ADD: {
        // Calls dict.__setitem__(TOS1[-i], TOS1, TOS). Used to implement dict
        // comprehensions.
        //
        // NOTE: While the added value or key/value pair is popped off, the
        // container object remains on the stack so that it is available for
        // further iterations of the loop.
        PyObject* key = stack.Pop();
        PyObject* value = stack.Pop();
        PyObject* map = stack.Peek(oparg);
        S6_DCHECK(PyDict_CheckExact(map));
        int err = PyDict_SetItem(map, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
        if (err != 0) {
          handle_exception();
        }
        break;
      }

      case LOAD_ATTR: {
        // Replaces TOS with getattr(TOS, co_names[oparg]).
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* owner = stack.Pop();
        PyObject* value = PyObject_GetAttr(owner, name);
        if (value) {
          type_feedback.Add(GetClassId(owner));
        }
        Py_DECREF(owner);
        if (!value) {
          handle_exception();
        } else {
          stack.Push(value);
        }
        break;
      }

      case COMPARE_OP: {
        // Performs a Boolean operation.
        PyObject* right = stack.Pop();
        PyObject* left = stack.Pop();
        type_feedback.Add(GetClassId(left));
        type_feedback.Add(GetClassId(right));
        PyObject* result = Compare(oparg, left, right);
        Py_DECREF(left);
        Py_DECREF(right);
        if (!result) {
          handle_exception();
        } else {
          stack.Push(result);
        }
        break;
      }

      case IMPORT_NAME: {
        // Imports the module co_names[namei]. TOS and TOS1 are popped and
        // provide the fromlist and level arguments of __import__(). The module
        // object is pushed onto the stack.
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* fromlist = stack.Pop();
        PyObject* level = stack.Pop();
        PyObject* result = ImportName(f, name, fromlist, level);
        if (!result) {
          handle_exception();
        } else {
          stack.Push(result);
        }
        break;
      }

      case IMPORT_STAR: {
        PyObject* from = stack.Pop();
        bool ok = ImportAllFrom(f, from);
        // The frame's locals may have been changed as part of the import
        // operation.
        locals = f->f_locals;
        if (!ok) handle_exception();
        break;
      }

      case IMPORT_FROM: {
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* from = stack.Top();
        PyObject* result = ImportFrom(from, name);
        if (!result) {
          handle_exception();
        } else {
          stack.Push(result);
        }
        break;
      }

      case SETUP_LOOP:
      case SETUP_EXCEPT:
      case SETUP_FINALLY: {
        block_stack.Push(opcode, pc.GetNextPc().AddOffset(oparg));
        break;
      }

      case BEFORE_ASYNC_WITH: {
        // Resolves __aenter__ and __aexit__ from the object on top of the
        // stack.  Pushes __aexit__ and result of __aenter__() to the stack.
        PyObject* context_manager = stack.Top();
        // WITH looks up __enter__ and then __exit__.  ASYNC_WITH looks up
        // __aexit__ and then __aenter__.  We match CPython's behavior because
        // it's observable.
        _Py_IDENTIFIER(__aexit__);
        PyObject* aexit = SpecialLookup(context_manager, &PyId___aexit__);
        if (!aexit) {
          handle_exception();
          break;
        }
        stack.Top() = aexit;
        _Py_IDENTIFIER(__aenter__);
        PyObject* aenter = SpecialLookup(context_manager, &PyId___aenter__);
        Py_DECREF(context_manager);
        if (!aenter) {
          handle_exception();
          break;
        }
        PyObject* result = PyObject_CallFunctionObjArgs(aenter, nullptr);
        Py_DECREF(aenter);
        if (!result) {
          handle_exception();
          break;
        }
        stack.Push(result);
        break;
      }

      case SETUP_ASYNC_WITH: {
        // The block stack's value stack height should not include the result of
        // __aenter__.
        PyObject* result = stack.Pop();
        block_stack.Push(SETUP_FINALLY, pc.GetNextPc().AddOffset(oparg));
        stack.Push(result);
        break;
      }

#if PY_MINOR_VERSION >= 7
      case LOAD_METHOD: {
        PyObject* name = PyTuple_GET_ITEM(names, oparg);
        PyObject* obj = stack.Pop();

        PyObject* method = nullptr;
        int is_method = _PyObject_GetMethod(obj, name, &method);

        if (!method) {
          handle_exception();
          break;
        }
        type_feedback.Add(GetClassId(obj));
        if (is_method) {
          stack.Push(method);
          stack.Push(obj);
        } else {
          stack.Push(nullptr);
          Py_DECREF(obj);
          stack.Push(method);
        }
        break;
      }

      case CALL_METHOD: {
        bool ok = false;
        if (PyObject* method = stack.Peek(oparg + 2)) {
          // Method call.
          ok = CallFunction(tstate, &stack, oparg + 1, nullptr);
        } else {
          // LOAD_METHOD determined this to not be a method call.
          ok = CallFunction(tstate, &stack, oparg, nullptr);
          PyObject* result = stack.Pop();
          // Remove the nullptr that was pushed by LOAD_METHOD.
          [[maybe_unused]] PyObject* top = stack.Pop();
          S6_DCHECK_EQ(top, nullptr);
          stack.Push(result);
        }
        if (!ok) handle_exception();
        break;
      }
#endif

      case CALL_FUNCTION: {
        type_feedback.Add(GetClassId(stack.Peek(oparg + 1)));
        bool ok = CallFunction(tstate, &stack, oparg, nullptr);
        if (!ok) handle_exception();
        break;
      }

      case CALL_FUNCTION_KW: {
        PyObject* names = stack.Pop();
        S6_CHECK(PyTuple_CheckExact(names));
        S6_CHECK(PyTuple_GET_SIZE(names) <= oparg);
        bool ok = CallFunction(tstate, &stack, oparg, names);
        if (!ok) handle_exception();
        break;
      }

      case CALL_FUNCTION_EX: {
        // Calls a callable object with variable set of positional and keyword
        // arguments.  If the lowest bit of flags is set, the top of the stack
        // contains a mapping object containing additional keyword arguments.
        // Below that is an iterable object containing positional arguments and
        // a callable object to call.  BUILD_MAP_UNPACK_WITH_CALL and
        // BUILD_TUPLE_UNPACK_WITH_CALL can be used for merging multiple mapping
        // objects and iterables containing arguments.  Before the callable is
        // called, the mapping object and iterable object are each “unpacked”
        // and their contents passed in as keyword and positional arguments
        // respectively.  CALL_FUNCTION_EX pops all arguments and the callable
        // object off the stack, calls the callable object with those arguments,
        // and pushes the return value returned by the callable object.
        PyObject* keyword_args = nullptr;
        if (oparg & 0x1) {
          PyObject* mapping = stack.Pop();
          if (PyDict_CheckExact(mapping)) {
            keyword_args = mapping;
          } else {
            // Allocate a fresh dictionary and copy in the mapping.
            keyword_args = PyDict_New();
            if (keyword_args && PyDict_Update(keyword_args, mapping) != 0 &&
                PyErr_ExceptionMatches(PyExc_AttributeError)) {
              // PyDict_Update will raise an AttributeError if the second
              // argument is not a mapping.  We want a TypeError instead.
              FormatMappingError(mapping, /*function=*/stack.Second());
            }

            Py_DECREF(mapping);
            if (PyErr_Occurred()) {
              Py_XDECREF(keyword_args);
              handle_exception();
              break;
            }
          }
        }

        PyObject* iterable = stack.Pop();
        if (!Py_TYPE(iterable)->tp_iter && !PySequence_Check(iterable)) {
          FormatIterableError(iterable, /*function=*/stack.Top());
          handle_exception();
          break;
        }
        PyObject* positional_args = PySequence_Tuple(iterable);
        Py_DECREF(iterable);
        if (!positional_args) {
          handle_exception();
          break;
        }

        PyObject* callable = stack.Pop();
        PyObject* result = nullptr;
        if (PyCFunction_Check(callable)) {
          bool use_profiling =
              tstate->use_tracing != 0 && tstate->c_profilefunc != nullptr;
          if (!use_profiling) {
            result = PyCFunction_Call(callable, positional_args, keyword_args);
          } else if (ProfileCEntry(tstate, callable)) {
            result = ProfileCExit(
                tstate, callable,
                PyCFunction_Call(callable, positional_args, keyword_args));
          }
        } else {
          result = PyObject_Call(callable, positional_args, keyword_args);
        }
        Py_DECREF(callable);
        Py_DECREF(positional_args);
        Py_XDECREF(keyword_args);
        if (!result) {
          handle_exception();
        } else {
          stack.Push(result);
        }
        break;
      }

      case MAKE_FUNCTION: {
        PyObject* qualified_name = stack.Pop();
        PyObject* code = stack.Pop();
        PyFunctionObject* function = reinterpret_cast<PyFunctionObject*>(
            PythonFunctionInfoTable::NewFunctionWithQualName(code, globals,
                                                             qualified_name));
        Py_DECREF(code);
        Py_DECREF(qualified_name);
        if (!function) {
          handle_exception();
          break;
        }

        // Depending on oparg, there will be extra values on the stack that need
        // to be installed in the function object.
        if (oparg & 0x08) {
          PyObject* closure = stack.Pop();
          S6_CHECK(PyTuple_CheckExact(closure));
          function->func_closure = closure;
        }
        if (oparg & 0x04) {
          PyObject* annotations = stack.Pop();
          S6_CHECK(PyDict_CheckExact(annotations));
          function->func_annotations = annotations;
        }
        if (oparg & 0x02) {
          PyObject* keyword_defaults = stack.Pop();
          S6_CHECK(PyDict_CheckExact(keyword_defaults));
          function->func_kwdefaults = keyword_defaults;
        }
        if (oparg & 0x01) {
          PyObject* defaults = stack.Pop();
          S6_CHECK(PyTuple_CheckExact(defaults));
          function->func_defaults = defaults;
        }

        stack.Push(reinterpret_cast<PyObject*>(function));
        break;
      }

      case JUMP_FORWARD: {
        // Increments bytecode counter by delta.
        pc.IncrementNextPcBy(PcValue::FromOffset(oparg));
        break;
      }

      case JUMP_ABSOLUTE: {
        // Set bytecode counter to target.
        pc.SetNextPc(PcValue::FromOffset(oparg));
        break;
      }

      case POP_JUMP_IF_TRUE: {
        // If TOS is true, sets the bytecode counter to target. TOS is popped.
        PyObject* cond = stack.Pop();
        type_feedback.Add(GetClassId(cond));
        if (cond == Py_False) {
          Py_DECREF(cond);
        } else if (cond == Py_True) {
          Py_DECREF(cond);
          pc.SetNextPc(PcValue::FromOffset(oparg));
        } else {
          int err = PyObject_IsTrue(cond);
          Py_DECREF(cond);
          if (err < 0) {
            handle_exception();
            break;
          }
          if (err > 0) {
            // cond was True.
            pc.SetNextPc(PcValue::FromOffset(oparg));
          }
        }
        break;
      }

      case POP_JUMP_IF_FALSE: {
        // If TOS is false, sets the bytecode counter to target. TOS is popped.
        PyObject* cond = stack.Pop();
        type_feedback.Add(GetClassId(cond));
        if (cond == Py_True) {
          Py_DECREF(cond);
        } else if (cond == Py_False) {
          Py_DECREF(cond);
          pc.SetNextPc(PcValue::FromOffset(oparg));
        } else {
          int err = PyObject_IsTrue(cond);
          Py_DECREF(cond);
          if (err < 0) {
            handle_exception();
            break;
          }
          if (err == 0) {
            // cond was False.
            pc.SetNextPc(PcValue::FromOffset(oparg));
          }
        }
        break;
      }

      case JUMP_IF_TRUE_OR_POP: {
        // If TOS is true, sets the bytecode counter to target and leaves TOS on
        // the stack. Otherwise (TOS is false), TOS is popped.
        PyObject* cond = stack.Top();
        if (cond == Py_False) {
          Py_DECREF(stack.Pop());
        } else if (cond == Py_True) {
          pc.SetNextPc(PcValue::FromOffset(oparg));
        } else {
          int err = PyObject_IsTrue(cond);
          if (err < 0) {
            handle_exception();
            break;
          }
          if (err > 0) {
            // cond was True.
            pc.SetNextPc(PcValue::FromOffset(oparg));
          } else {
            Py_DECREF(stack.Pop());
          }
        }
        break;
      }

      case JUMP_IF_FALSE_OR_POP: {
        // If TOS is false, sets the bytecode counter to target and leaves TOS
        // on the stack. Otherwise (TOS is true), TOS is popped.
        PyObject* cond = stack.Top();
        if (cond == Py_True) {
          Py_DECREF(stack.Pop());
        } else if (cond == Py_False) {
          pc.SetNextPc(PcValue::FromOffset(oparg));
        } else {
          int err = PyObject_IsTrue(cond);
          if (err < 0) {
            handle_exception();
            break;
          }
          if (err == 0) {
            // cond was False.
            pc.SetNextPc(PcValue::FromOffset(oparg));
          } else {
            Py_DECREF(stack.Pop());
          }
        }
        break;
      }

      case RAISE_VARARGS: {
        // Raises an exception using one of the 3 forms of the raise statement.
        if (oparg == 0) {
          // raise (re-raise previous exception).
          // TODO: Clean up once 3.7 has landed.
#if PY_MINOR_VERSION >= 7
          auto* exc_info = _PyErr_GetTopmostException(tstate);
#else
          auto* exc_info = tstate;
#endif
          PyObject* type = exc_info->exc_type;
          PyObject* value = exc_info->exc_value;
          PyObject* tb = exc_info->exc_traceback;
          if (type && type != Py_None) {
            Py_INCREF(type);
            Py_XINCREF(value);
            Py_XINCREF(tb);
            PyErr_Restore(type, value, tb);
          } else {
            PyErr_SetString(PyExc_RuntimeError,
                            "No active exception to reraise");
          }
        } else if (oparg == 1) {
          // raise TOS (raise exception instance or type at TOS).
          PyObject* exc = stack.Pop();
          Raise(exc, /*cause=*/nullptr);
        } else if (oparg == 2) {
          // raise TOS1 from TOS (raise exception instance or type at TOS1
          // with __cause__ set to TOS).
          PyObject* cause = stack.Pop();
          PyObject* exc = stack.Pop();
          Raise(exc, cause);
        } else {
          PyErr_SetString(PyExc_SystemError, "bad RAISE_VARARGS oparg");
        }

        handle_exception();
        break;
      }

      case RETURN_VALUE: {
        // Returns with TOS to the caller of the function.
        // Eagerly pop retval as block_stack.Return() modifies the value stack.
        PyObject* retval = stack.Pop();
        if (absl::optional<PcValue> next_pc = block_stack.Return()) {
          stack.Push(retval);
          stack.Push(PyLong_FromWhy(Why::kReturn));
          pc.SetNextPc(*next_pc);
          break;
        }
        // No block to pop to; actually return.
        pc.Terminate(retval);
        break;
      }

      case GET_AITER: {
        // Implements TOS = get_awaitable(TOS.__aiter__()).  See GET_AWAITABLE
        // for details about get_awaitable.
        PyObject* object = stack.Top();
        PyTypeObject* type = Py_TYPE(object);
        if (!type->tp_as_async || !type->tp_as_async->am_aiter) {
          PyErr_Format(PyExc_TypeError,
                       "'async for' requires an object with __aiter__ method, "
                       "got %.100s",
                       type->tp_name);
          handle_exception();
          break;
        }

        PyObject* iterable = type->tp_as_async->am_aiter(object);
        if (!iterable) {
          handle_exception();
          break;
        }

        PyTypeObject* iterable_type = Py_TYPE(iterable);
        // TODO: Semantics changed between 3.6 and 3.7.
#if PY_MINOR_VERSION >= 7
        if (!iterable_type->tp_as_async ||
            !iterable_type->tp_as_async->am_anext) {
          PyErr_Format(PyExc_TypeError,
                       "'async for' received an object from __aiter__ "
                       "that does not implement __anext__: %.100s",
                       iterable_type->tp_name);
          Py_DECREF(iterable);
          handle_exception();
          break;
        }
        stack.Top() = iterable;
        Py_DECREF(object);
        break;
#else
        if (iterable_type->tp_as_async &&
            iterable_type->tp_as_async->am_anext) {
          // From CPython:
          //
          // Starting with CPython 3.5.2 __aiter__ should return asynchronous
          // iterators directly (not awaitables that resolve to asynchronous
          // iterators.)
          //
          // Therefore, we check if the object that was returned from __aiter__
          // has an __anext__ method.  If it does, we wrap it in an awaitable
          // that resolves to `iter`.
          //
          // See http://bugs.python.org/issue27243 for more details.
          stack.Top() = _PyAIterWrapper_New(iterable);
          Py_DECREF(object);
          Py_DECREF(iterable);
          break;
        }

        PyObject* awaitable = _PyCoro_GetAwaitableIter(iterable);
        if (!awaitable) {
          _PyErr_FormatFromCause(
              PyExc_TypeError,
              "'async for' received an invalid object from __aiter__: %.100s",
              iterable_type->tp_name);
          Py_DECREF(iterable);
          handle_exception();
          break;
        }

        Py_DECREF(iterable);
        if (PyErr_WarnFormat(
                PyExc_DeprecationWarning, 1,
                "'%.100s' implements legacy __aiter__ protocol; __aiter__ "
                "should return an asynchronous iterator, not awaitable",
                type->tp_name)) {
          // Warning was converted to an error.
          Py_DECREF(awaitable);
          handle_exception();
          break;
        }

        stack.Top() = awaitable;
        Py_DECREF(object);
        break;
#endif
      }

      case GET_ANEXT: {
        // Implements PUSH(get_awaitable(TOS.__anext__())).  See GET_AWAITABLE
        // for details about get_awaitable.
        PyObject* object = stack.Top();
        PyTypeObject* type = Py_TYPE(object);
        if (PyAsyncGen_CheckExact(object)) {
          PyObject* awaitable = type->tp_as_async->am_anext(object);
          if (!awaitable) {
            handle_exception();
            break;
          }
          stack.Push(awaitable);
          break;
        }

        if (!type->tp_as_async || !type->tp_as_async->am_anext) {
          PyErr_Format(PyExc_TypeError,
                       "'async for' requires an iterator with __anext__ "
                       "method, got %.100s",
                       type->tp_name);
          handle_exception();
          break;
        }

        PyObject* next_iterable = type->tp_as_async->am_anext(object);
        if (!next_iterable) {
          handle_exception();
          break;
        }

        PyObject* awaitable = _PyCoro_GetAwaitableIter(next_iterable);
        if (!awaitable) {
          _PyErr_FormatFromCause(
              PyExc_TypeError,
              "'async for' received an invalid object from __anext__: %.100s",
              Py_TYPE(next_iterable)->tp_name);
          Py_DECREF(next_iterable);
          handle_exception();
          break;
        }

        Py_DECREF(next_iterable);
        stack.Push(awaitable);
        break;
      }

      case GET_AWAITABLE: {
        // Implements TOS = get_awaitable(TOS), where get_awaitable(o) returns o
        // if o is a coroutine object or a generator object with the
        // CO_ITERABLE_COROUTINE flag, or resolves o.__await__.
        PyObject* iterable = stack.Top();
        PyObject* awaitable = _PyCoro_GetAwaitableIter(iterable);
        if (!awaitable) {
          // Peek back in the instruction stream to format helpful error
          // messages identifying the cause of some type errors.
          PyTypeObject* type = Py_TYPE(iterable);
          if (!type->tp_as_async || !type->tp_as_async->am_await) {
            int previous_opcode = _Py_OPCODE(pc.Previous());
            if (previous_opcode == BEFORE_ASYNC_WITH) {
              PyErr_Format(PyExc_TypeError,
                           "'async with' received an object from __aenter__ "
                           "that does not implement __await__: %.100s",
                           type->tp_name);
            } else if (previous_opcode == WITH_CLEANUP_START) {
              PyErr_Format(PyExc_TypeError,
                           "'async with' received an object from __aexit__ "
                           "that does not implement __await__: %.100s",
                           type->tp_name);
            }
          }
          handle_exception();
          break;
        }

        if (PyCoro_CheckExact(awaitable)) {
          // If awaitable is a coroutine, make sure we are not already awaiting
          // on it.
          PyObject* yf = _PyGen_yf(reinterpret_cast<PyGenObject*>(awaitable));
          if (yf) {
            Py_DECREF(yf);
            Py_DECREF(awaitable);
            PyErr_SetString(PyExc_RuntimeError,
                            "coroutine is being awaited already");
            handle_exception();
            break;
          }
        }

        Py_DECREF(iterable);
        stack.Top() = awaitable;
        break;
      }

      case YIELD_FROM: {
        // Pops TOS and delegates to it as a subiterator from a generator.
        PyObject* value = stack.Pop();
        PyObject* receiver = stack.Top();
        PyObject* result;
        if (PyGen_CheckExact(receiver) || PyCoro_CheckExact(receiver)) {
          result = _PyGen_Send(reinterpret_cast<PyGenObject*>(receiver), value);
        } else if (value == Py_None) {
          result = Py_TYPE(receiver)->tp_iternext(receiver);
        } else {
          _Py_IDENTIFIER(send);
          result = _PyObject_CallMethodIdObjArgs(receiver, &PyId_send, value,
                                                 nullptr);
        }
        Py_DECREF(value);

        if (!result) {
          // No result is either StopIteration or some other error.
          int error = _PyGen_FetchStopIterationValue(&result);
          if (error < 0) {
            handle_exception();
            break;
          }
          Py_DECREF(receiver);
          stack.Top() = result;
        } else {
          // The receiver stays on the stack and the result is yielded.  Rewind
          // the frame's instruction pointer by one instruction so that this one
          // will be executed again when the frame is re-entered.
          f->f_stacktop = stack.StackPointer();
          f->f_lasti -= sizeof(_Py_CODEUNIT);
          pc.Terminate(result);
        }
        break;
      }

      case YIELD_VALUE: {
        // Pops TOS and yields it from a generator.
        PyObject* value = stack.Pop();

        if (co->co_flags & CO_ASYNC_GENERATOR) {
          PyObject* wrapper = _PyAsyncGenValueWrapperNew(value);
          Py_DECREF(value);
          if (!wrapper) {
            handle_exception();
            break;
          }
          value = wrapper;
        }

        f->f_stacktop = stack.StackPointer();
        pc.Terminate(value);
        break;
      }

      case POP_BLOCK: {
        // Removes one block from the block stack. Per frame, there is a stack
        // of blocks, denoting try statements, and such.
        block_stack.Pop();
        break;
      }

      case POP_EXCEPT: {
        // Removes one block from the block stack. The popped block must be an
        // exception handler block, as implicitly created when entering an
        // except handler. In addition to popping extraneous values from the
        // frame stack, the last three popped values are used to restore the
        // exception state.

        // CPython's implementation exposes this assertion to the user as a
        // SystemError.
        if (block_stack.Top().b_type != EXCEPT_HANDLER) {
          PyErr_SetString(PyExc_SystemError,
                          "popped block is not an except handler");
          handle_exception();
          break;
        }
        block_stack.PopExceptHandler();
        break;
      }

      case GET_ITER: {
        // Implements TOS = iter(TOS).
        PyObject* iterable = stack.Pop();
        type_feedback.Add(GetClassId(iterable));
        PyObject* iter = PyObject_GetIter(iterable);
        Py_DECREF(iterable);
        stack.Push(iter);
        if (!iter) handle_exception();
        break;
      }

      case GET_YIELD_FROM_ITER: {
        // If TOS is a generator iterator or coroutine object it is left as
        // is. Otherwise, implements TOS = iter(TOS).
        PyObject* iterable = stack.Top();
        if (PyCoro_CheckExact(iterable)) {
          if (!(co->co_flags & (CO_COROUTINE | CO_ITERABLE_COROUTINE))) {
            // TOS is a coroutine and we are trying to yield from a generator.
            // This is an error.
            PyErr_SetString(PyExc_TypeError,
                            "cannot 'yield from' a coroutine object in a "
                            "non-coroutine generator");
            handle_exception();
            break;
          }
        } else if (!PyGen_CheckExact(iterable)) {
          // TOS is not a generator or a coroutine.
          PyObject* result = PyObject_GetIter(iterable);
          Py_DECREF(iterable);
          if (!result) {
            handle_exception();
            break;
          }
          stack.Top() = result;
        }
        break;
      }

      case FOR_ITER: {
        // TOS is an iterator. Call its __next__() method. If this yields a new
        // value, push it on the stack (leaving the iterator below it). If the
        // iterator indicates it is exhausted TOS is popped, and the byte code
        // counter is incremented by delta
        PyObject* iter = stack.Top();
        type_feedback.Add(GetClassId(iter));
        auto& next_fn = *iter->ob_type->tp_iternext;
        PyObject* next = next_fn(iter);
        if (next) {
          stack.Push(next);
          break;
        }

        if (PyErr_Occurred()) {
          if (!PyErr_ExceptionMatches(PyExc_StopIteration)) {
            handle_exception();
            break;
          }
          // StopIteration is ignored and treated as a normal loop termination.
          if (tstate->c_tracefunc) {
            CallExceptionTrace(tstate, f);
          }
          PyErr_Clear();
        }
        // Remove iter from the stack.
        stack.Pop();
        Py_DECREF(iter);
        pc.IncrementNextPcBy(PcValue::FromOffset(oparg));
        break;
      }

      case BREAK_LOOP: {
        const auto [new_pc, push_kbreak] = block_stack.Break();
        if (push_kbreak) stack.Push(PyLong_FromWhy(Why::kBreak));
        pc.SetNextPc(new_pc);
        break;
      }

      case CONTINUE_LOOP: {
        // TODO: This is CPython's exact semantics. Is it reasonable
        // that this could ever fail? oparg is between -128 and 127.
        PyObject* target = PyLong_FromLong(oparg);
        if (!target) {
          handle_exception();
          break;
        }

        if (absl::optional<PcValue> new_pc = block_stack.Continue()) {
          stack.Push(target);
          stack.Push(PyLong_FromWhy(Why::kContinue));
          pc.SetNextPc(*new_pc);
        } else {
          Py_DECREF(target);
          pc.SetNextPc(PcValue::FromOffset(oparg));
        }
        break;
      }

      case SETUP_WITH: {
        // TODO: These create static variables, which puts extra if()s
        // on the hot path. It seems crazy to have these here.
        _Py_IDENTIFIER(__exit__);
        _Py_IDENTIFIER(__enter__);
        PyObject* context_manager = stack.Top();
        PyObject* enter = SpecialLookup(context_manager, &PyId___enter__);
        if (!enter) {
          handle_exception();
          break;
        }
        PyObject* exit = SpecialLookup(context_manager, &PyId___exit__);
        if (!exit) {
          Py_DECREF(enter);
          handle_exception();
          break;
        }
        stack.Top() = exit;
        Py_DECREF(context_manager);
        PyObject* res = PyObject_CallFunctionObjArgs(enter, nullptr);
        Py_DECREF(enter);
        if (!res) {
          handle_exception();
          break;
        }
        // Setup the finally block before pushing the result of __enter__ on the
        // stack.
        block_stack.Push(SETUP_FINALLY, pc.GetNextPc().AddOffset(oparg));
        stack.Push(res);
        break;
      }

      case WITH_CLEANUP_START: {
        // Starts cleaning up the stack when a with statement block exits.
        //
        // At the top of the stack are either NULL (pushed by BEGIN_FINALLY) or
        // 6 values pushed if an exception has been raised in the with block.
        // Below is the context manager’s __exit__() or __aexit__() bound
        // method.
        //
        // If TOS is NULL, calls SECOND(None, None, None), removes the function
        // from the stack, leaving TOS, and pushes None to the stack. Otherwise
        // calls SEVENTH(TOP, SECOND, THIRD), shifts the bottom 3 values of the
        // stack down, replaces the empty spot with NULL and pushes TOS. Finally
        // pushes the result of the call.
        PyObject* exc = stack.Top();
        PyObject* val = Py_None;
        PyObject* tb = Py_None;
        PyObject* exit_func = nullptr;
        if (exc == Py_None) {
          stack.Drop(1);
          exit_func = stack.Pop();
          stack.Push(exc);
        } else if (PyLong_Check(exc)) {
          stack.Drop(1);
          Why why = static_cast<Why>(PyLong_AsLong(exc));
          if (why == Why::kReturn || why == Why::kContinue) {
            // Retval is at TOS. Bury retval beneath exc.
            exit_func = stack.Second();
            stack.Second() = stack.Top();
            stack.Top() = exc;
          } else {
            // No retval.
            exit_func = stack.Top();
            stack.Top() = exc;
          }
          exc = Py_None;
        } else {
          val = stack.Second();
          tb = stack.Third();
          exit_func = stack.Peek(7);
          // Remove stack[7], move stack[6..4] to stack[7..5] and null-out
          // stack[4].
          stack.Poke(7, stack.Peek(6));
          stack.Poke(6, stack.Peek(5));
          stack.Poke(5, stack.Peek(4));
          // UNWIND_EXCEPT_HANDLER will pop this off.
          stack.Fourth() = nullptr;

          // We just shifted the stack down, so we have to tell the except
          // handler block that the values are lower than it expects.
          PyTryBlock& block = f->f_blockstack[f->f_iblock - 1];
          S6_CHECK_EQ(block.b_type, EXCEPT_HANDLER);
          --block.b_level;
        }

        S6_CHECK(exit_func);
        PyObject* res =
            PyObject_CallFunctionObjArgs(exit_func, exc, val, tb, nullptr);
        Py_DECREF(exit_func);
        if (!res) {
          handle_exception();
          break;
        }

        Py_INCREF(exc); /* Duplicating the exception on the stack */
        stack.Push(exc);
        stack.Push(res);
        break;
      }

      case WITH_CLEANUP_FINISH: {
        // Finishes cleaning up the stack when a with statement block exits.
        //
        // TOS is result of __exit__() or __aexit__() function call pushed by
        // WITH_CLEANUP_START. SECOND is None or an exception type (pushed when
        // an exception has been raised).
        //
        // Pops two values from the stack. If SECOND is not None and TOS is true
        // unwinds the EXCEPT_HANDLER block which was created when the exception
        // was caught and pushes NULL to the stack.
        PyObject* res = stack.Pop();
        PyObject* exc = stack.Pop();

        int err = 0;
        if (exc != Py_None) {
          err = PyObject_IsTrue(res);
        }

        Py_DECREF(res);
        Py_DECREF(exc);

        if (err < 0) {
          handle_exception();
          break;
        } else if (err > 0) {
          err = 0;
          // There was an exception and a True return.
          stack.Push(PyLong_FromWhy(Why::kSilenced));
        }
        break;
      }

      case END_FINALLY: {
        // Official (outdated) CPython Doc:
        // Terminates a finally clause. The interpreter recalls whether the
        // exception has to be re-raised or execution has to be continued
        // depending on the value of TOS.
        //
        // If TOS is NULL (pushed by BEGIN_FINALLY) continue from the next
        // instruction. TOS is popped.
        //
        // If TOS is an integer (pushed by CALL_FINALLY), sets the bytecode
        // counter to TOS. TOS is popped.
        //
        // If TOS is an exception type (pushed when an exception has been
        // raised) 6 values are popped from the stack, the first three popped
        // values are used to re-raise the exception and the last three popped
        // values are used to restore the exception state. An exception handler
        // block is removed from the block stack.
        //----------------------------------------------------------------------
        // Up to date comment:
        // END_FINALLY is both used to a end `finally` block of a
        // try-with-finally construct and for re-raising an exception at
        // other moments.
        // When ending a finally block, it can called with 3 possible values
        // on TOS which depends on how the finally block was entered:
        // - If the TOS is Py_None, that means that the finally block was
        //   entered on a normal control flow path, by falling trough from
        //   the try block. In which case at the end of the finally, we just
        //   fallthrough to normal code path.
        // - If the TOS is a long, that mean that the finally block was
        //   entered when handling a control-flow instruction like break,
        //   continue, return, etc. The long value indicate which operation as
        //   defined by the BlockExitStatus enumeration. It means that
        //   the finally block intercepted that control-flow operation and
        //   now needs to resume it
        // - If the TOS is an exception type, that mean that the finally block
        //   was entered when taking an exception. Thus to end the finally, we
        //   need to raise again the exception. Since exception states are
        //   always constitued of three values, there are also two other values
        //   below the exception type on the stack that are part of the new
        //   exception state that need to be restored.
        //
        // Additionally since END_FINALLY has the nice behavior of raising an
        // exception from the TOS, it is also sometime used to simply reraise
        // an exception in some circumstances other than ending a finally.
        PyObject* status = stack.Pop();
        if (PyLong_Check(status)) {
          Why why = static_cast<Why>(PyLong_AS_LONG(status));
          Py_DECREF(status);
          ResumeUnwind(why, &pc, &stack, &block_stack);
          break;
        }

        if (PyExceptionClass_Check(status)) {
          PyObject* exc = stack.Pop();
          PyObject* tb = stack.Pop();
          PyErr_Restore(status, exc, tb);
          handle_exception();
          break;
        }

        if (status != Py_None) {
          PyErr_SetString(PyExc_SystemError, "'finally' pops bad exception");
          Py_DECREF(status);
          handle_exception();
          break;
        }

        Py_DECREF(status);
        break;
      }

      case BUILD_SLICE: {
        // Pushes a slice object on the stack.
        S6_DCHECK(oparg == 2 || oparg == 3);
        PyObject* step = oparg == 3 ? stack.Pop() : nullptr;
        PyObject* stop = stack.Pop();
        PyObject* start = stack.Pop();
        PyObject* slice = PySlice_New(start, stop, step);
        Py_DECREF(start);
        Py_DECREF(stop);
        Py_XDECREF(step);
        stack.Push(slice);
        if (!slice) handle_exception();
        break;
      }

      case FORMAT_VALUE: {
        // Used for implementing formatted literal strings (f-strings).
        PyObject* fmt_spec =
            (oparg & FVS_MASK) == FVS_HAVE_SPEC ? stack.Pop() : nullptr;
        PyObject* (*conv_func)(PyObject*);
        switch (oparg & FVC_MASK) {
          case FVC_STR:
            conv_func = PyObject_Str;
            break;
          case FVC_REPR:
            conv_func = PyObject_Repr;
            break;
          case FVC_ASCII:
            conv_func = PyObject_ASCII;
            break;
          default:
            conv_func = nullptr;  // No conversion.
        }

        PyObject* value = stack.Pop();
        if (conv_func) {
          PyObject* result = conv_func(value);
          Py_DECREF(value);
          if (!result) {
            Py_XDECREF(fmt_spec);
            handle_exception();
            break;
          }
          value = result;
        }

        PyObject* result = nullptr;
        // If value is a unicode object and there is no format spec then
        // PyObject_Format(value, fmt_spec) == value, so avoid calling
        // PyObject_Format() and use value directly.
        if (PyUnicode_CheckExact(value) && !fmt_spec) {
          result = value;  // Transfer ownership to result.
        } else {
          result = PyObject_Format(value, fmt_spec);
          Py_DECREF(value);
          Py_XDECREF(fmt_spec);
          if (!result) {
            handle_exception();
            break;
          }
        }

        stack.Push(result);
        break;
      }

      case EXTENDED_ARG: {
        // Prefixes any opcode which has an argument too big to fit into the
        // default one byte.
        oparg_carry = oparg;
        break;
      }

      default: {
        S6_DVLOG(1) << "Unhandled opcode: " << opcode << " '"
                    << BytecodeOpcodeToString(opcode) << "' -> escape()";
        // Unhandled opcode; shim out to CPython.
        return escape_to_cpython();
      }
    }
  }

  // TODO: The special casing for coroutines is removed in 3.7.
#if PY_MINOR_VERSION < 7
  if (!pc.GetTerminatingValue()) {
    // Exiting without RETURN_VALUE or YIELD_VALUE - empty the stack and restore
    // the caller's exception state if we are in a generator or coroutine frame.
    stack.ClearAndDecref();
    if (co->co_flags & kGeneratorOrCoroutine) {
      RestoreAndClearExceptionState(tstate, f);
    }
  } else if (co->co_flags & kGeneratorOrCoroutine) {
    // Yielding or returning from a generator or coroutine frame.  Restore the
    // caller's exception state and preserve this frame's state if we are in a
    // local exception handler.
    if (block_stack.InsideExcept()) {
      // This must be a yield, because return would unwind the block stack.
      // Preserve the generator's exception state in case it is re-entered.
      SwapExceptionState(tstate, f);
    } else {
      RestoreAndClearExceptionState(tstate, f);
    }
  }
#else
  if (!pc.GetTerminatingValue()) {
    // Exiting without RETURN_VALUE or YIELD_VALUE - empty the stack.
    stack.ClearAndDecref();
  }
#endif

  meta->set_completion_observed(true);
  return _Py_CheckFunctionResult(nullptr, pc.GetTerminatingValue(),
                                 "s6::EvalFrame");
}  // NOLINT(readability/fn_size)

}  // namespace deepmind::s6
