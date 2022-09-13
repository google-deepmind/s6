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

#include "runtime/runtime.h"

#include <Python.h>
#include <frameobject.h>
#include <opcode.h>

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "classes/object.h"
#include "code_object.h"
#include "core_util.h"
#include "interpreter.h"
#include "runtime/generator.h"
#include "runtime/pyframe_object_cache.h"
#include "runtime/python_function_info_table.h"
#include "runtime/stack_frame.h"
#include "strongjit/function.h"
#include "strongjit/instructions.h"
#include "utils/logging.h"
#include "utils/path.h"

// TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION >= 7
#define EXC_INFO(x) exc_state.x
#else
#define EXC_INFO(x) x
#endif

namespace deepmind::s6 {
extern void OracleProfileEvent(PyFrameObject*, int64_t);

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

// Sets the error indicator for an unbound variable error.
// If `is_local` is true, the error occurred when accessing a fastlocal.
// Otherwise the error occurred when accessing a cell or free variable; The kind
// of error depends on whether `index` is an index into the code object's
// closed-over variables or into its free variables.
void FormatUnboundError(PyCodeObject* code, int index, bool is_local) {
  // Do not overwrite an existing exception.
  if (PyErr_Occurred()) return;

  if (is_local) {
    FormatError(PyExc_UnboundLocalError, kUnboundLocalErrorMessage,
                PyTuple_GetItem(code->co_varnames, index));
    return;
  }

  if (index < PyTuple_GET_SIZE(code->co_cellvars)) {
    FormatError(PyExc_UnboundLocalError, kUnboundLocalErrorMessage,
                PyTuple_GET_ITEM(code->co_cellvars, index));
    return;
  }

  index -= PyTuple_GET_SIZE(code->co_cellvars);
  FormatError(PyExc_NameError, kUnboundFreeErrorMessage,
              PyTuple_GET_ITEM(code->co_freevars, index));
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

PyObject* IteratorNext(PyObject* iter) {
  PyObject* result = Py_TYPE(iter)->tp_iternext(iter);
  if (result) return result;

  if (PyErr_Occurred()) {
    if (!PyErr_ExceptionMatches(PyExc_StopIteration)) {
      return nullptr;
    }
    // StopIteration is ignored and treated as a normal loop termination.
    S6_CHECK(!PyThreadState_Get()->c_tracefunc)
        << "Strongjit doesn't support tracing!";
    PyErr_Clear();
  }
  return reinterpret_cast<PyObject*>(1);
}

void HandleRaiseVarargs(int64_t argument, PyObject* exc, PyObject* cause) {
  if (argument != 0) {
    Raise(exc, cause);
    return;
  }

  // raise (re-raise previous exception).
  // TODO: Clean up once 3.7 has landed.
#if PY_MINOR_VERSION >= 7
  PyThreadState* tstate = PyThreadState_GET();
  auto* exc_info = _PyErr_GetTopmostException(tstate);
#else
  auto* exc_info = PyThreadState_GET();
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
    PyErr_SetString(PyExc_RuntimeError, "No active exception to reraise");
  }
}

int64_t CheckedGivenExceptionMatches(PyObject* left, PyObject* right) {
  // Implements COMPARE_OP:PyCmp_EXC_MATCH.
  const char* kMessage =
      "catching classes that do not inherit from BaseException is not "
      "allowed";
  if (PyTuple_Check(right)) {
    int length = PyTuple_Size(right);
    for (int i = 0; i < length; ++i) {
      if (!PyExceptionClass_Check(PyTuple_GET_ITEM(right, i))) {
        PyErr_SetString(PyExc_TypeError, kMessage);
        return -1;
      }
    }
  } else if (!PyExceptionClass_Check(right)) {
    PyErr_SetString(PyExc_TypeError, kMessage);
    return -1;
  }
  return PyErr_GivenExceptionMatches(left, right);
}

// Sets up the free variables for a function call.
void SetupFreeVars(PyObject* func, PyFrameObject* pyframe) {
  if (PyMethod_Check(func) && !PyMethod_GET_SELF(func))
    func = PyMethod_GET_FUNCTION(func);
  PyCodeObject* co = reinterpret_cast<PyCodeObject*>(PyFunction_GET_CODE(func));
  PyObject** fastlocals = &pyframe->f_localsplus[0];

  int64_t cell_vars_count = PyTuple_GET_SIZE(co->co_cellvars);
  for (int i = 0; i < cell_vars_count; i++) {
    PyObject* cell = nullptr;
    if (co->co_cell2arg != nullptr) {
      auto arg_i = co->co_cell2arg[i];
      if (arg_i != CO_CELL_NOT_AN_ARG) {
        cell = PyCell_New(fastlocals[arg_i]);
        // Before we had a local copy in `fast_locals`. Remove in preference to
        // the cell free variable.
        PyObject* old = fastlocals[arg_i];
        fastlocals[arg_i] = nullptr;
        Py_XDECREF(old);
      }
    }
    // Not an argument, create an empty cell.
    if (cell == nullptr) cell = PyCell_New(nullptr);

    if (cell == nullptr) {
      S6_LOG(FATAL) << "Fatal error in SetupFreeVars";
    }

    // Set the new local in the frame.
    int64_t offset = co->co_nlocals + i;
    PyObject* old = fastlocals[offset];
    fastlocals[offset] = cell;
    Py_XDECREF(old);
  }

  // Copy over all closure variables to the frame's free variables.
  PyObject** free_vars = fastlocals + co->co_nlocals;
  int64_t free_vars_count = PyTuple_GET_SIZE(co->co_freevars);
  PyObject* closure = PyFunction_GET_CLOSURE(func);
  for (int i = 0; i < free_vars_count; i++) {
    PyObject* closure_var = PyTuple_GET_ITEM(closure, i);
    Py_INCREF(closure_var);
    free_vars[cell_vars_count + i] = closure_var;
  }
}

void Dealloc(PyObject* obj) { _Py_Dealloc(obj); }

PyObject* CallPython(int64_t arg_count, PyObject** args, PyObject* names,
                     StackFrame* stack_frame) {
  PyObject* callee = args[-1];
  S6_VLOG(2) << absl::StrCat(
      "RuntimeCallFunction(callee=", absl::Hex(callee),
      PyObjectToString(callee), ", args=", absl::Hex(args),
      ", arg_count=", arg_count, ", names=", absl::Hex(names), ")");

  // Note that strongjit assumes that a Python callee will steal references to
  // all arguments, so it doesn't decref them in the caller after we return.
  //
  // Because _PyFunction_FastCallKeywords *doesn't* steal a reference, we need
  // to compensate by decreffing all arguments before we return.
  //
  // We make a note of the current value of `args` and `arg_count` so we decref
  // exactly what was given (there is a special case for bound methods that
  // alters the arguments array).
  absl::Span<PyObject* const> to_decref(args, arg_count);

  // These type checks are all pointer equality on ob_type checks, so are almost
  // free.
  PyObject* ret;
  if (PyCFunction_Check(callee)) {
    ret = _PyCFunction_FastCallKeywords(callee, args, arg_count, names);
    Py_DECREF(callee);
  } else if (PyMethod_Check(callee) && PyMethod_GET_SELF(callee) &&
             PyFunction_Check(PyMethod_GET_FUNCTION(callee))) {
    // Optimize bound methods by directly calling the original method.
    // Prepend the `self` value before all existing args.
    // NOTE that code_generator ensures that args[-1] is accessible and
    // available for this reason.
    PyObject* self = PyMethod_GET_SELF(callee);
    ++arg_count;
    --args;
    args[0] = self;
    Py_INCREF(self);

    PyObject* method = callee;
    callee = PyMethod_GET_FUNCTION(method);
    // It can be the case that the only reference to `function` is from
    // `method` so be careful about the order of INCREF and DECREF.
    Py_INCREF(callee);
    Py_DECREF(method);
    ret = _PyFunction_FastCallKeywords(callee, args, arg_count, names);
    Py_DECREF(callee);
    Py_DECREF(self);
  } else if (PyFunction_Check(callee)) {
    ret = _PyFunction_FastCallKeywords(callee, args, arg_count, names);
    Py_DECREF(callee);
  } else {
    ret = _PyObject_FastCallKeywords(callee, args, arg_count, names);
    Py_DECREF(callee);
  }

  for (int64_t i = 0; i < to_decref.size(); ++i) {
    Py_DECREF(to_decref[i]);
  }

  return ret;
}

PyObject* CallAttribute(int64_t arg_count, PyObject** args, PyObject* names,
                        StackFrame* stack_frame, PyUnicodeObject* attr_str,
                        int64_t call_python_bytecode_offset) {
  PyObject* receiver = args[-1];
  PyObject* attr =
      PyObject_GetAttr(receiver, reinterpret_cast<PyObject*>(attr_str));
  if (!attr) {
    return nullptr;
  }
  stack_frame->set_bytecode_offset(call_python_bytecode_offset);
  // Allow CallPython to decref attr and args.
  args[-1] = attr;
  PyObject* ret = CallPython(arg_count, args, names, stack_frame);
  Py_DECREF(receiver);
  return ret;
}

void ExceptWithoutHandler(int64_t bytecode_offset) {
  S6_VLOG(2) << absl::StrCat(
      "ExceptWithoutHandler(bytecode_offset=", bytecode_offset, ")");
  PyThreadState* tstate = PyThreadState_GET();
  tstate->frame->f_lasti = bytecode_offset;

  S6_CHECK(PyErr_Occurred());
  PyTraceBack_Here(tstate->frame);
  if (tstate->c_tracefunc) {
    CallExceptionTrace(tstate, tstate->frame);
  }
}

void Except(int64_t bytecode_offset, PyObject** output_values) {
  S6_VLOG(2) << absl::StrCat("Except(bytecode_offset=", bytecode_offset,
                             ", output_values=", absl::Hex(output_values), ")");
  PyThreadState* tstate = PyThreadState_GET();
  tstate->frame->f_lasti = bytecode_offset;

  S6_CHECK(PyErr_Occurred());
  PyTraceBack_Here(tstate->frame);
  if (tstate->c_tracefunc) {
    CallExceptionTrace(tstate, tstate->frame);
  }
  output_values[0] = tstate->EXC_INFO(exc_traceback);
  output_values[1] = tstate->EXC_INFO(exc_value);
  output_values[2] = tstate->EXC_INFO(exc_type);
  if (!tstate->EXC_INFO(exc_type)) {
    Py_INCREF(Py_None);
    output_values[2] = Py_None;
  }
  PyObject *exc_type, *exc_value, *exc_traceback;
  PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
  // Make the raw exception data available to the handler, so a program
  // can emulate the Python main loop.
  PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);
  PyException_SetTraceback(exc_value, exc_traceback ? exc_traceback : Py_None);
  Py_INCREF(exc_type);
  tstate->EXC_INFO(exc_type) = exc_type;
  Py_INCREF(exc_value);
  tstate->EXC_INFO(exc_value) = exc_value;
  tstate->EXC_INFO(exc_traceback) = exc_traceback;
  if (!exc_traceback) {
    exc_traceback = Py_None;
  }
  Py_INCREF(exc_traceback);
  output_values[3] = exc_traceback;
  output_values[4] = exc_value;
  output_values[5] = exc_type;
}

// Cleans up a StackFrame after function execution has completed.
void CleanupStackFrame(StackFrame* stack_frame) {
  PyThreadState* tstate = stack_frame->thread_state();
  PyFrameObject* pyframe = stack_frame->pyframe();
  pyframe->f_executing = 0;
  tstate->frame = pyframe->f_back;

  stack_frame->ClearMagic();

  // If the frame wasn't called with the fast calling convention, a caller
  // function owns a reference to the PyFrameObject, so just do nothing here.
  // Everything will be deallocated when the frame destructor fires and we
  // hold a borrowed reference.
  if (!stack_frame->called_with_fast_calling_convention()) {
    return;
  }

  // This frame was created by the fast calling convention, so it used the
  // PyFrameObjectCache to borrow a reference to a PyFrameObject. We must
  // deallocate all the members we used and release it back to the frame object
  // cache.

  // This was a cached frame object. We can't rely on the destructor to fire,
  // so deallocate all fastlocals now.
  PyCodeObject* code = stack_frame->py_code_object();
  int64_t fastlocals_count = PyTuple_GET_SIZE(code->co_cellvars) +
                             PyTuple_GET_SIZE(code->co_freevars) +
                             code->co_nlocals;
  for (int64_t i = 0; i < fastlocals_count; ++i) {
    Py_CLEAR(pyframe->f_localsplus[i]);
  }
  pyframe->f_valuestack = &pyframe->f_localsplus[fastlocals_count];
  Py_CLEAR(pyframe->f_locals);
  // TODO: What about other members like f_exc_type?
  GetPyFrameObjectCache()->Finished(pyframe);
}

// Cleans up a StackFrame after function execution has completed.
void CleanupStackFrameForGenerator(StackFrame* stack_frame) {
  PyThreadState* tstate = stack_frame->thread_state();
  PyFrameObject* pyframe = stack_frame->pyframe();

  stack_frame->ClearMagic();

#if PY_MINOR_VERSION < 7
  RestoreAndClearExceptionState(tstate, pyframe);
#endif
  tstate->frame = pyframe->f_back;
  if (pyframe->f_executing == 0) {
    // If the frame is no longer executing at this point, it has been completed
    // by the interpreter so do nothing.
    return;
  }

  // If the frame is a true PyFrameObject, a caller function owns a reference to
  // the PyFrameObject, so just do nothing here. Everything will be deallocated
  // when the frame destructor fires and we hold a borrowed reference.
  if (Py_TYPE(pyframe) == &PyFrame_Type) {
    pyframe->f_executing = 0;
    // We may not have a generator state at this point if we've deoptimized back
    // to the interpreter.
    if (GeneratorState::Get(pyframe)) {
      DeallocateGeneratorState(pyframe);
    }
    return;
  }

  // Otherwise, give this back to the frame cache.
  if (pyframe->f_executing) {
    // This is important, as it signals to gen_send_ex that the frame is
    // finished.
    pyframe->f_stacktop = nullptr;
  }
  pyframe->f_executing = 0;

  // This was a cached frame object. We can't rely on the destructor to fire,
  // so deallocate all fastlocals now.
  PyCodeObject* code = stack_frame->py_code_object();
  int64_t fastlocals_count = PyTuple_GET_SIZE(code->co_cellvars) +
                             PyTuple_GET_SIZE(code->co_freevars) +
                             code->co_nlocals;
  for (int64_t i = 0; i < fastlocals_count; ++i) {
    Py_CLEAR(pyframe->f_localsplus[i]);
  }
  Py_CLEAR(pyframe->f_locals);
  // Erase the generator state.
  pyframe->f_valuestack[0] = nullptr;
}

S6_ATTRIBUTE_PRESERVE_CALLER_SAVED_REGISTERS void ProfileEventReachedZero() {
  OracleProfileEvent(nullptr, 0);
}

S6_ATTRIBUTE_PRESERVE_CALLER_SAVED_REGISTERS
void AllocateFrameOnHeap() {
  PyFrameObject* frame = GetPyFrameObjectCache()->AllocateOnHeap();
  // Per https://clang.llvm.org/docs/AttributeReference.html#preserve-most ,
  // all registers including rax are preserved across this call, except r11.
  // We therefore return in r11.
  asm volatile("mov %0, %%r11" : : "r"(frame));
}

std::pair<PyObject*, int64_t> SetUpForYieldValue(PyObject* result,
                                                 StackFrame* stack_frame,
                                                 const YieldValueInst* yi) {
  PyFrameObject* frame = stack_frame->pyframe();

  GeneratorState* generator_state = GeneratorState::Get(frame);
  S6_CHECK(generator_state);
  generator_state->set_yield_value_inst(yi);
  generator_state->set_resume_pc(
      reinterpret_cast<uint64_t>(__builtin_return_address(0)));
  frame->f_lasti = yi->bytecode_offset();

  PyThreadState* tstate = stack_frame->thread_state();
#if PY_MINOR_VERSION < 7
  if (!generator_state->yield_value_inst()->try_handlers().empty()) {
    SwapExceptionState(tstate, frame);
  } else {
    RestoreAndClearExceptionState(tstate, frame);
  }
#endif
  Py_LeaveRecursiveCall();

  frame->f_executing = 0;
  tstate->frame = frame->f_back;
  return {result, 0};
}

std::pair<PyObject*, GeneratorState*> SetUpStackFrameForGenerator(
    PyFrameObject* pyframe, StackFrame* stack_frame, CodeObject* code_object,
    int64_t num_spill_slots) {
  *stack_frame = StackFrame(pyframe, code_object);
  S6_DCHECK(pyframe->f_gen);

  // Fast path to get the generator state if it exists. We know the stack is
  // valid, and we know that either:
  //   a) We haven't run before (f_lasti == -1) and therefore there is no
  //     gen_state or
  //   b) We have run before (f_lasti != -1) and therefore there
  //     MUST be a gen_state.
  GeneratorState* gen_state = nullptr;
  if (pyframe->f_lasti != -1) {
    gen_state = GeneratorState::GetUnsafe(pyframe);
  } else {
    gen_state = GetPyGeneratorFrameObjectCache()->GetOrCreateGeneratorState(
        pyframe, num_spill_slots);
    gen_state->set_code_object(code_object);
  }
  S6_DCHECK(GeneratorState::Get(pyframe));

  PyThreadState* tstate = stack_frame->thread_state();
#if PY_MINOR_VERSION < 7
  // On entry to a generator's or coroutine's frame we have to save the
  // caller's exception state and possibly restore the generator's exception
  // state.
  if (!pyframe->f_exc_type || pyframe->f_exc_type == Py_None) {
    SaveExceptionState(tstate, pyframe);
  } else {
    SwapExceptionState(tstate, pyframe);
  }
#endif
  tstate->frame = pyframe;
  pyframe->f_executing = 1;

  PyObject* yield_result = nullptr;
  if (const YieldValueInst* yi = gen_state->yield_value_inst()) {
    S6_DCHECK_GE(pyframe->f_lasti, 0);
    S6_DCHECK(pyframe->f_stacktop != nullptr);
    yield_result = *--pyframe->f_stacktop;
    *pyframe->f_stacktop = nullptr;
    // The generator is no longer paused.
    gen_state->set_yield_value_inst(nullptr);
  }

  return std::make_pair(yield_result, gen_state);
}

// TODO: Please inline me.
void SetUpStackFrame(PyFrameObject* pyframe, StackFrame* stack_frame,
                     CodeObject* code_object) {
  *stack_frame = StackFrame(pyframe, code_object);
  stack_frame->thread_state()->frame = pyframe;
  pyframe->f_executing = 1;
}

PyObject* RematerializeGetAttr(PyObject* receiver, int64_t names_index) {
  PyObject* names = PyThreadState_GET()->frame->f_code->co_names;
  return PyObject_GetAttr(receiver, PyTuple_GET_ITEM(names, names_index));
}

int InitializeObjectDict(PyObject* object, PyObject* name,
                         PyObject* stored_value, const Class* cls) {
  PyObject** dict_ptr = _PyObject_GetDictPtr(object);
  _PyObjectDict_SetItem(Py_TYPE(object), dict_ptr, name, stored_value);
  PyDictObject* dict = reinterpret_cast<PyDictObject*>(*dict_ptr);
  if (cls->dict_kind() == DictKind::kSplit && !_PyDict_HasSplitTable(dict)) {
    return 0;
  }
  SetClassId(dict, cls->id());
  return 1;
}

PyObject* NewFunctionWithQualName(PyObject* code, PyObject* globals,
                                  PyObject* qualified_name) {
  return PythonFunctionInfoTable::NewFunctionWithQualName(code, globals,
                                                          qualified_name);
}

PyGenObject* CreateGenerator(PyFunctionObject* function, PyObject* builtins,
                             ...) {
  PyCodeObject* code = reinterpret_cast<PyCodeObject*>(function->func_code);
  int64_t n = code->co_argcount;
  S6_CHECK_EQ((code->co_flags & (CO_VARARGS | CO_VARKEYWORDS)), 0);
  va_list vl;
  va_start(vl, builtins);

  auto cache = GetPyGeneratorFrameObjectCache();
  PyGeneratorFrameObjectCache::SizedFrameObject* sf = cache->AllocateOrNull();
  if (!sf) {
    // No cached objects available.
    EventCounters::Instance().Add("runtime.generator_cache_exhausted", 1);
    PyObject** args =
        reinterpret_cast<PyObject**>(alloca(n * sizeof(PyObject*)));
    for (int64_t i = 0; i < n; ++i) {
      args[i] = va_arg(vl, PyObject*);
    }
    va_end(vl);
    PyGenObject* gen =
        reinterpret_cast<PyGenObject*>(_PyFunction_FastCallKeywords(
            reinterpret_cast<PyObject*>(function), args, n, nullptr));
    for (int64_t i = 0; i < n; ++i) Py_XDECREF(args[i]);
    return gen;
  }

  PyFrameObject* frame = &sf->frame;
  PyGenObject* gen = &sf->gen_object;

  for (int64_t i = 0; i < n; ++i) {
    frame->f_localsplus[i] = va_arg(vl, PyObject*);
  }
  va_end(vl);

  frame->f_code = reinterpret_cast<PyCodeObject*>(function->func_code);
  frame->f_lasti = -1;
  frame->f_globals = function->func_globals;
  frame->f_builtins = builtins;
  frame->f_valuestack = frame->f_localsplus + code->co_nlocals +
                        PyTuple_GET_SIZE(code->co_cellvars) +
                        PyTuple_GET_SIZE(code->co_freevars);
  frame->f_stacktop = frame->f_valuestack;
  frame->f_iblock = 0;
  Py_XINCREF(frame->f_back);
  Py_XINCREF(frame->f_code);
  Py_XINCREF(frame->f_builtins);
  Py_XINCREF(frame->f_globals);

  SetupFreeVars(reinterpret_cast<PyObject*>(function), frame);

  gen->gi_code = function->func_code;
  gen->gi_name = function->func_name;
  gen->gi_qualname = function->func_qualname;
  Py_INCREF(gen->gi_code);
  Py_INCREF(gen->gi_name);
  Py_INCREF(gen->gi_qualname);
  S6_CHECK_EQ(Py_REFCNT(frame), 1);
  S6_CHECK_EQ(Py_REFCNT(gen), 1);
  return gen;
}

// Similar to BuildListUnpack in interpreter.cc, but takes the stack as variadic
// arguments.
PyObject* BuildListUnpackVararg(int count, PyObject* func, ...) {
  auto* list = reinterpret_cast<PyListObject*>(PyList_New(0));
  if (!list) {
    return nullptr;
  }

  va_list vl;
  va_start(vl, func);

  for (int i = count; i > 0; --i) {
    PyObject* iterable = va_arg(vl, PyObject*);
    PyObject* none = _PyList_Extend(list, iterable);
    if (!none) {
      // func(*args) call where args is not iterable.
      if (func && PyErr_ExceptionMatches(PyExc_TypeError)) {
        if (!(Py_TYPE(iterable)->tp_iter || PySequence_Check(iterable))) {
          PyErr_Format(
              PyExc_TypeError,
              "%.200s%.200s argument after * must be an iterable, not %.200s",
              PyEval_GetFuncName(func), PyEval_GetFuncDesc(func),
              Py_TYPE(iterable)->tp_name);
        }
      }
      Py_DECREF(list);
      return nullptr;
    }
    Py_DECREF(none);
  }
  va_end(vl);
  return reinterpret_cast<PyObject*>(list);
}

PyObject* CallFunctionEx(PyObject* callable, PyObject* positional,
                         PyObject* mapping) {
  PyObject* keyword_args = nullptr;
  if (mapping) {
    if (PyDict_CheckExact(mapping)) {
      keyword_args = mapping;
    } else {
      // Allocate a fresh dictionary and copy in the mapping.
      keyword_args = PyDict_New();
      if (keyword_args && PyDict_Update(keyword_args, mapping) != 0 &&
          PyErr_ExceptionMatches(PyExc_AttributeError)) {
        // PyDict_Update will raise an AttributeError if the second
        // argument is not a mapping.  We want a TypeError instead.
        FormatMappingError(mapping, callable);
      }

      if (PyErr_Occurred()) {
        Py_XDECREF(keyword_args);
        return nullptr;
      }
    }
  }

  if (!Py_TYPE(positional)->tp_iter && !PySequence_Check(positional)) {
    if (keyword_args != mapping) {
      Py_DECREF(keyword_args);
    }
    FormatIterableError(positional, callable);
    return nullptr;
  }
  PyObject* positional_args = PySequence_Tuple(positional);
  if (!positional_args) {
    if (keyword_args != mapping) {
      Py_DECREF(keyword_args);
    }
    return nullptr;
  }

  PyObject* result = PyObject_Call(callable, positional_args, keyword_args);
  if (keyword_args != mapping) {
    Py_DECREF(keyword_args);
  }
  Py_DECREF(positional_args);
  return result;
}

int64_t SinDouble(int64_t x) {
  return absl::bit_cast<int64_t>(::sin(absl::bit_cast<double>(x)));
}
int64_t CosDouble(int64_t x) {
  return absl::bit_cast<int64_t>(::cos(absl::bit_cast<double>(x)));
}

}  // namespace deepmind::s6
