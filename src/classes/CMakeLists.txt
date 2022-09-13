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


s6_cc_library(
  NAME s6_classes_class
  SRCS attribute.cc class.cc class_manager.cc
  HDRS attribute.h class.h class_manager.h
  DEPS Python::Python
       absl::base
       absl::flat_hash_map
       absl::status
       absl::statusor
       absl::strings
       absl::synchronization
       s6_classes_util
       s6_core_util
       s6_event_counters
       s6_utils_intrusive_list
       s6_utils_keyed_intern_table
       s6_utils_no_destructor
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_classes_class_test
  SRCS class_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::hash
       s6_classes_class
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_classes_object
  SRCS getsetattr.cc object.cc
  HDRS getsetattr.h object.h
  DEPS Python::Python
       absl::status
       s6_classes_class
       s6_classes_util
       s6_event_counters
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_classes_util
  SRCS util.cc
  HDRS util.h
  DEPS Python::Python absl::base)

add_subdirectory(python)
