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

#include <Python.h>
#include <classobject.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "code_object.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

namespace py = ::pybind11;

// Wrapper around `FormatOrDie` to act as a disassembler API of a Python
// function to its StrongJIT IR.
absl::StatusOr<std::string> dis6(const py::function& py_func) {
  PyCodeObject* co;
  if (PyFunction_Check(py_func.ptr())) {
    co = reinterpret_cast<PyCodeObject*>(PyFunction_GET_CODE(py_func.ptr()));
  } else if (PyMethod_Check(py_func.ptr())) {
    co = reinterpret_cast<PyCodeObject*>(
        PyFunction_GET_CODE(PyMethod_GET_FUNCTION(py_func.ptr())));
  } else {
    return absl::InvalidArgumentError("Argument must be a function or method.");
  }

  std::vector<BytecodeInstruction> bytecode_insts = ExtractInstructions(co);
  S6_ASSIGN_OR_RETURN(
      Function function,
      IngestProgram(bytecode_insts, PyObjectToString(co->co_name),
                    co->co_nlocals, co->co_argcount));

  return Format(function);
}

PYBIND11_MODULE(dis6, m) {
  pybind11::google::ImportStatusModule();

  m.def("dis6", &dis6, py::arg("func"),
        "Disassembles a function to its Strongjit IR.");
}

}  // namespace
}  // namespace deepmind::s6
