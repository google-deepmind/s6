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

#ifndef THIRD_PARTY_DEEPMIND_S6_INTERPRETER_H_
#define THIRD_PARTY_DEEPMIND_S6_INTERPRETER_H_

#include <Python.h>
#include <frameobject.h>

namespace deepmind::s6 {

// Evaluate a PyFrameObject. This function is drop-in compatible with CPython's
// PyEval_EvalFrameDefault.
//
// Function parameters and semantics are as PyEval_EvalFrameDefault.
PyObject* EvalFrame(PyFrameObject* f, int throwflag);

// An exception has occurred and a trace function is registered with `tstate`.
// Call it.
void CallExceptionTrace(PyThreadState* tstate, PyFrameObject* f);

// Sets an exception with a given __cause__ attribute.
void Raise(PyObject* exc, PyObject* cause);

// TODO: Removed in 3.7
#if PY_MINOR_VERSION < 7

// Copies the exception state from `tstate` to `frame`.
//
// We are entering or re-entering a coroutine's or generator's frame and the
// frame does not have a valid exception state (or else we have entered via
// generator.throw). Duplicate the thread's exception state in the frame so we
// can restore it when we exit the frame.
inline void SaveExceptionState(PyThreadState* tstate, PyFrameObject* frame) {
  Py_XINCREF(tstate->exc_type);
  Py_XINCREF(tstate->exc_value);
  Py_XINCREF(tstate->exc_traceback);
  PyObject* type = frame->f_exc_type;
  PyObject* value = frame->f_exc_value;
  PyObject* traceback = frame->f_exc_traceback;
  frame->f_exc_type = tstate->exc_type;
  frame->f_exc_value = tstate->exc_value;
  frame->f_exc_traceback = tstate->exc_traceback;
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
}

// Swaps the exception state of `tstate` and `frame`.
//
// We are either entering or leaving a coroutine's or generator's frame and the
// frame has a valid exception state. If we are entering then that exception
// state is the frame's state which was saved on last exit. If we are leaving
// then the exception state was the thread's state which was saved on last
// entry. In either case, the exception state in the frame should be used as
// the thread's exception state and the thread's exception state should be saved
// in the frame to restore later.
inline void SwapExceptionState(PyThreadState* tstate, PyFrameObject* frame) {
  std::swap(tstate->exc_type, frame->f_exc_type);
  std::swap(tstate->exc_value, frame->f_exc_value);
  std::swap(tstate->exc_traceback, frame->f_exc_traceback);
}

// Restores the exception state from `frame` to `tstate`.
//
// We are leaving a couroutine's or generator's frame and we do not need to
// preserve the thread's exception state. Restore the exception state in the
// frame which was saved on entry.
inline void RestoreAndClearExceptionState(PyThreadState* tstate,
                                          PyFrameObject* frame) {
  PyObject* type = tstate->exc_type;
  PyObject* value = tstate->exc_value;
  PyObject* traceback = tstate->exc_traceback;
  tstate->exc_type = frame->f_exc_type;
  tstate->exc_value = frame->f_exc_value;
  tstate->exc_traceback = frame->f_exc_traceback;
  frame->f_exc_type = nullptr;
  frame->f_exc_value = nullptr;
  frame->f_exc_traceback = nullptr;
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
}
#endif

inline void FormatIterableError(PyObject* object, PyObject* function) {
  PyErr_Format(PyExc_TypeError,
               "%.200s%.200s argument after * must be an iterable, not %.200s",
               PyEval_GetFuncName(function), PyEval_GetFuncDesc(function),
               Py_TYPE(object)->tp_name);
}

inline void FormatMappingError(PyObject* object, PyObject* function) {
  PyErr_Format(PyExc_TypeError,
               "%.200s%.200s argument after ** must be a mapping, not %.200s",
               PyEval_GetFuncName(function), PyEval_GetFuncDesc(function),
               Py_TYPE(object)->tp_name);
}
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_INTERPRETER_H_
