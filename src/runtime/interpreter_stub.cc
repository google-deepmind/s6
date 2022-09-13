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

#include "runtime/interpreter_stub.h"

#include <cstdint>

#include "utils/logging.h"

namespace deepmind::s6 {

PyObject* StrongjitInterpreterStub(PyObject* pyfunc_object, ...) {
  S6_DCHECK(pyfunc_object);
  PyCodeObject* co =
      reinterpret_cast<PyCodeObject*>(PyFunction_GET_CODE(pyfunc_object));
  bool is_varargs = co->co_flags & CO_VARARGS;
  bool is_varkeywords = co->co_flags & CO_VARKEYWORDS;

  int64_t n = co->co_argcount + is_varargs + is_varkeywords;
  PyObject** args = reinterpret_cast<PyObject**>(alloca(n * sizeof(PyObject*)));
  va_list vl;
  va_start(vl, pyfunc_object);

  for (int64_t i = 0; i < n; ++i) {
    args[i] = va_arg(vl, PyObject*);
  }
  va_end(vl);
  return StrongjitInterpreterStubArrayArgs(pyfunc_object, n, args);
}

PyObject* StrongjitInterpreterStubArrayArgs(PyObject* pyfunc_object, int64_t n,
                                            PyObject** args) {
  PyCodeObject* co =
      reinterpret_cast<PyCodeObject*>(PyFunction_GET_CODE(pyfunc_object));
  bool is_varargs = co->co_flags & CO_VARARGS;
  bool is_varkeywords = co->co_flags & CO_VARKEYWORDS;
  if (!is_varargs && !is_varkeywords) {
    PyObject* ret = _PyFunction_FastCallKeywords(pyfunc_object, args, n,
                                                 /*kwnames=*/nullptr);
    // Strongjit expects us to steal a reference to args.
    // _PyFunction_FastCallKeywords does not, so simulate by decreffing all
    // arguments now.
    for (int64_t i = 0; i < n; ++i) {
      Py_DECREF(args[i]);
    }
    return ret;
  }

  // This is the variadic case.
  PyObject* star_args = nullptr;
  int64_t star_args_size = 0;
  if (is_varargs) {
    star_args = args[co->co_argcount];
    S6_DCHECK(PyTuple_Check(star_args));
    star_args_size = PyTuple_GET_SIZE(star_args);
  }
  PyObject* positional_args = PyTuple_New(star_args_size + co->co_argcount);

  for (int64_t i = 0; i < co->co_argcount; ++i) {
    PyTuple_SET_ITEM(positional_args, i, args[i]);
  }
  for (int64_t i = 0; i < star_args_size; ++i) {
    PyObject* v = PyTuple_GET_ITEM(star_args, i);
    Py_INCREF(v);
    PyTuple_SET_ITEM(positional_args, co->co_argcount + i, v);
  }

  PyObject* star_kwargs = nullptr;
  if (is_varkeywords) {
    star_kwargs = args[co->co_argcount + is_varargs];
  }

  PyObject* result = PyObject_Call(pyfunc_object, positional_args, star_kwargs);
  Py_DECREF(positional_args);
  Py_XDECREF(star_args);
  Py_XDECREF(star_kwargs);
  return result;
}

}  // namespace deepmind::s6
