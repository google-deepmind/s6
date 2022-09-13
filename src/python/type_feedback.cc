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

#include "type_feedback.h"

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "classes/class.h"
#include "classes/class_manager.h"
#include "classes/object.h"
#include "metadata.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/status_casters.h"

namespace pybind11 {
using ::deepmind::s6::ClassManager;
using ::deepmind::s6::Metadata;

PYBIND11_MODULE(type_feedback, py_module) {
  pybind11::google::ImportStatusModule();
  py_module.doc() = "Defines internal S6 functions related to type feedback";

  py_module.def(
      "extract_from_code_object",
      [](object obj) -> absl::StatusOr<list> {
        if (!PyCode_Check(obj.ptr()))
          return absl::FailedPreconditionError("Expected a code object");
        const auto& type_feedback =
            Metadata::Get(reinterpret_cast<PyCodeObject*>(obj.ptr()))
                ->type_feedback();
        list l;
        for (const absl::InlinedVector<deepmind::s6::ClassDistribution, 1>&
                 distributions : type_feedback) {
          for (const deepmind::s6::ClassDistribution& distribution :
               distributions) {
            l.append(
                distribution.Summarize().ToString(&ClassManager::Instance()));
          }
        }
        return l;
      },
      "Returns the class ID of an object.");
}

}  // namespace pybind11
