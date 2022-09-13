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

#include "api.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "code_object.h"
#include "interpreter.h"
#include "metadata.h"
#include "oracle.h"
#include "pybind11/attr.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/status_casters.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion.h"

namespace deepmind::s6 {
namespace py = ::pybind11;

struct S6JitCallable {
  py::function fn;
  py::object self;
};

struct S6CodeDetail {
  py::object code;
};

struct NotCompiledError {
  const char* what() const {
    return "Function has not been compiled or has been deoptimized.";
  }
};

struct CompilationFailedError {
  const char* what() const { return message.c_str(); }

  std::string message;
};

PYBIND11_MODULE(api, py_module) {
  pybind11::google::ImportStatusModule();
  py_module.doc() = "Defines the internal S6 interface.";

  py::register_exception<NotCompiledError>(py_module, "NotCompiledError");
  py::register_exception<CompilationFailedError>(py_module,
                                                 "CompilationFailedError");

  //////////////////////////////////////////////////////////////////////////////
  // S6JitCallable

  py::class_<S6JitCallable>(py_module, "S6JitCallable", py::dynamic_attr())
      .def_readonly("fn", &S6JitCallable::fn)
      .def("__call__",
           [](S6JitCallable self, py::args args,
              py::kwargs kwargs) -> py::handle {
             JitScope jit_scope;
             if (self.self.is_none()) {
               PyObject* obj =
                   PyObject_Call(self.fn.ptr(), args.ptr(), kwargs.ptr());
               if (!obj) throw py::error_already_set();
               return obj;
             } else {
               return self.fn(self.self, *args, **kwargs).release();
             }
           })
      .def(
          "_interpret",
          [](S6JitCallable self, py::args args,
             py::kwargs kwargs) -> py::handle {
            InterpreterScope interpreter_scope;
            if (self.self.is_none()) {
              PyObject* obj =
                  PyObject_Call(self.fn.ptr(), args.ptr(), kwargs.ptr());
              if (!obj) throw py::error_already_set();
              return obj;
            } else {
              return self.fn(self.self, *args, **kwargs).release();
            }
          },
          py::doc(R"(Interpret `fn` with the S6 interpreter.

The S6 interpreter is always used, even if `fn` was compiled or deoptimized.
This means that the type feedback will be updated based on this call.

This will apply to all functions called by `fn` recursively except if another
explicit call to the S6 API changes it.)"))
      .def(
          "_evaluate",
          [](S6JitCallable self, py::args args,
             py::kwargs kwargs) -> py::handle {
            EvaluatorScope evaluator_scope;
            if (self.self.is_none()) {
              PyObject* obj =
                  PyObject_Call(self.fn.ptr(), args.ptr(), kwargs.ptr());
              if (!obj) throw py::error_already_set();
              return obj;
            } else {
              return self.fn(self.self, *args, **kwargs).release();
            }
          },
          py::doc(R"(Evaluates `fn` with the S6 evaluator.

"Evaluating" means interpreting the S6 strongjit (Intermediate representation)
of `fn`.

The function `fn` needs to be compiled, otherwise execution will fall back to
using the CPython interpreter and compilation will never be triggered. This
function has no purpose other than debugging. If calling a compiled function
with `_evaluate` doesn't result in the same behavior as when calling it will the
normal call (of a jitted function), then S6 code generation has a bug.

This will apply to all functions called by `fn` recursively except if another
explicit call to the S6 API changes it. If a called function wasn't compiled,
execution will revert to using the plain CPython interpreter.)"))
      .def("__get__",
           [](S6JitCallable self, py::object obj,
              py::object type) -> S6JitCallable {
             // Implements the descriptor protocol, so a @jitted object can be
             // used as a Python method.
             return S6JitCallable{self.fn, obj};
           });

  //////////////////////////////////////////////////////////////////////////////
  // S6CodeDetail

  py::class_<S6CodeDetail>(py_module, "S6CodeDetail")
      .def_property_readonly(
          "x86asm",
          [](S6CodeDetail self) -> absl::StatusOr<std::string> {
            PyCodeObject* co = reinterpret_cast<PyCodeObject*>(self.code.ptr());
            Metadata* metadata = Metadata::Get(co);
            if (metadata->has_compilation_failed()) {
              throw CompilationFailedError{
                  std::string(metadata->compilation_status().message())};
            }
            CodeObject* code_object = metadata->current_code_object().load();
            if (!code_object) {
              throw NotCompiledError();
            }
            return code_object->Disassemble();
          },
          py::doc(
              R"(Returns a string version of the x86 assembly of `fn`.

The X86 assembly returned is obtained from disassembled machine code.)"))
      .def_property_readonly(
          "strongjit",
          [](S6CodeDetail self) -> absl::StatusOr<std::string> {
            PyCodeObject* co = reinterpret_cast<PyCodeObject*>(self.code.ptr());
            Metadata* metadata = Metadata::Get(co);
            if (metadata->has_compilation_failed()) {
              throw CompilationFailedError{
                  std::string(metadata->compilation_status().message())};
            }
            CodeObject* code_object = metadata->current_code_object().load();
            if (!code_object) {
              throw NotCompiledError();
            }
            return Format(code_object->function(),
                          ChainAnnotators(PredecessorAnnotator(),
                                          SourceLocationAnnotator(co)));
          },
          py::doc(R"(Returns a string version of the strongjit of `fn`.

Strongjit is the name of the intermediate representation used by S6 between
Python and machine code. It is the representation on which compiler
optimizations are done. The version returned here is the version after
optimizations as it was compiled to machine code. It can be interpreted by using
the "evaluate" method of the s6.jit object.)"))
      .def(
          "force_compile",
          [](S6CodeDetail self) -> absl::Status {
            return Oracle::Get()->ForceCompilation(
                reinterpret_cast<PyCodeObject*>(self.code.ptr()));
          },
          py::doc("Compiles `fn`, if it has not already been compiled. Throws "
                  "an exception if the compilation failed."))
      .def(
          "deoptimize",
          [](S6CodeDetail self) {
            Metadata* metadata =
                Metadata::Get(reinterpret_cast<PyCodeObject*>(self.code.ptr()));
            CodeObject* code_object = metadata->current_code_object().load();
            if (code_object != nullptr) {
              metadata->Deoptimize();
              code_object->MarkDeoptimized();
            } else {
              throw NotCompiledError();
            }
          },
          py::doc("Deoptimize `fn`. This only deoptimizes the main "
                  "specialization. Throws NotCompiledError if not compiled."))
      .def_property_readonly(
          "is_compiled",
          [](S6CodeDetail self) -> bool {
            PyCodeObject* co = reinterpret_cast<PyCodeObject*>(self.code.ptr());
            return Metadata::Get(co)->current_code_object().load() != nullptr;
          },
          py::doc("Returns True if `fn` is currently compiled, and False "
                  "otherwise."));

  //////////////////////////////////////////////////////////////////////////////
  // Globals

  py::object functools_wraps = py::module::import("functools").attr("wraps");
  py_module.def(
      "jit",
      [functools_wraps](py::function fn) -> absl::StatusOr<py::object> {
        S6_RETURN_IF_ERROR(Initialize(/*adopt_types=*/true));
        // functools_wraps(a)(b) gets function attributes from `a` and applies
        // them to `b`.
        return functools_wraps(fn)(S6JitCallable{fn, py::none()});
      },
      py::arg("fn"), py::doc(R"(Just-in-time compiles `fn`.

When the returned callable is called, just-in-time compilation mode is enabled
if it was not enabled already. All functions called transitively by `fn` become
considerable for just-in-time compilation.

Returns:
  A Callable wrapping `fn`, such that `jit` is suitable for use as a decorator.
)"));

  py_module.def(
      "inspect",
      [](py::object fn) -> S6CodeDetail {
        if (PyFunction_Check(fn.ptr())) {
          return S6CodeDetail{py::reinterpret_borrow<py::object>(
              PyFunction_GET_CODE(fn.ptr()))};
        } else if (PyCode_Check(fn.ptr())) {
          return S6CodeDetail{fn};
        }

        try {
          S6JitCallable c = fn.cast<S6JitCallable>();
          return S6CodeDetail{py::reinterpret_borrow<py::object>(
              PyFunction_GET_CODE(c.fn.ptr()))};
        } catch (py::cast_error) {
          throw py::type_error();
        }
      },
      py::arg("fn"), py::doc(R"(Inspects a function or code object.

Returns an S6CodeDetail or None if the function or code object has not been
compiled.)"));
}

}  // namespace deepmind::s6
