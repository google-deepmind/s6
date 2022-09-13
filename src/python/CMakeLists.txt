# Copyright 2021 The s6 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


s6_pybind_extension(
  NAME s6_python_api
  OUTPUT_NAME api
  SRCS api.cc
  DEPS absl::status
       s6_api
       s6_code_object
       s6_interpreter
       s6_metadata
       s6_oracle
       s6_strongjit_formatter
       s6_strongjit_ingestion
  INCLUDE_DIRS ${pybind11_abseil_INCLUDE_DIRS})

s6_py_test(NAME s6_python_api_test SRCS api_test.py)

s6_pybind_extension(
  NAME s6_python_type_feedback
  OUTPUT_NAME type_feedback
  SRCS type_feedback.cc
  DEPS absl::inlined_vector
       absl::status
       s6_classes_class
       s6_classes_object
       s6_metadata
       s6_type_feedback
  INCLUDE_DIRS ${pybind11_abseil_INCLUDE_DIRS})

s6_py_test(NAME s6_python_type_feedback_test SRCS type_feedback_test.py)

s6_py_test(NAME s6_python_jit_test SRCS jit_test.py)
