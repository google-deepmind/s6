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
  NAME s6_classes_python_classes
  OUTPUT_NAME classes
  SRCS classes.cc
  DEPS s6_classes_class s6_classes_object s6_utils_status_macros
  INCLUDE_DIRS ${pybind11_abseil_INCLUDE_DIRS})

s6_py_test(NAME s6_classes_python_classes_test SRCS classes_test.py)

s6_py_test(NAME s6_classes_python_adopt_existing_types_test
           SRCS adopt_existing_types_test.py)
