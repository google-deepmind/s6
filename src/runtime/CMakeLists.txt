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
  NAME s6_runtime_pyframe_object_cache
  HDRS pyframe_object_cache.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::status
       absl::strings
       s6_core_util
       s6_runtime_generator
       s6_utils_logging
       s6_utils_no_destructor)

s6_cc_test(
  NAME s6_runtime_pyframe_object_cache_test
  SRCS pyframe_object_cache_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       s6_runtime_pyframe_object_cache)

s6_cc_library(
  NAME s6_runtime_stack_frame
  HDRS stack_frame.h
  DEPS Python::Python absl::base s6_runtime_util)

s6_cc_library(
  NAME s6_runtime_callee_address
  SRCS callee_address.cc
  HDRS callee_address.h
  DEPS Python::Python
       absl::status
       absl::statusor
       s6_classes_object
       s6_runtime_runtime
       s6_strongjit_base)

s6_cc_library(
  NAME s6_runtime_slot_indexes
  SRCS slot_indexes.cc
  HDRS slot_indexes.h
  DEPS s6_strongjit_base)

s6_cc_library(
  NAME s6_runtime_runtime
  SRCS runtime.cc
  HDRS runtime.h
  DEPS Python::Python
       absl::strings
       s6_classes_object
       s6_code_object
       s6_core_util
       s6_interpreter
       s6_runtime_generator
       s6_runtime_pyframe_object_cache
       s6_runtime_python_function_info_table
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_utils_logging
       s6_utils_path)

s6_cc_library(
  NAME s6_runtime_interpreter_stub
  SRCS interpreter_stub.cc
  HDRS interpreter_stub.h
  DEPS Python::Python s6_utils_logging)

s6_cc_library(
  NAME s6_runtime_deoptimization_runtime
  SRCS deoptimization_runtime.cc
  HDRS deoptimization_runtime.h
  DEPS Python::Python
       absl::strings
       s6_code_object
       s6_core_util
       s6_event_counters
       s6_global_intern_table
       s6_interpreter
       s6_metadata
       s6_runtime_callee_address
       s6_runtime_evaluator
       s6_runtime_generator
       s6_runtime_pyframe_object_cache
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_util
       s6_utils_logging
       s6_utils_path)

s6_cc_library(
  NAME s6_runtime_evaluator
  SRCS evaluator.cc evaluator_instructions.cc
  HDRS evaluator.h
  DEPS Python::Python
       absl::base
       absl::status
       absl::statusor
       absl::strings
       s6_core_util
       s6_event_counters
       s6_interpreter
       s6_runtime_callee_address
       s6_runtime_generator
       s6_runtime_interpreter_stub
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_util
       s6_utils_logging
       s6_utils_path
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_runtime_util
  HDRS util.h
  DEPS Python::Python)

s6_cc_library(
  NAME s6_runtime_deoptimization_map
  SRCS deoptimization_map.cc
  HDRS deoptimization_map.h
  DEPS s6_core_util
       s6_runtime_slot_indexes
       s6_runtime_util
       s6_strongjit_base
       s6_utils_interval_map
       s6_utils_range)

s6_cc_test(
  NAME s6_runtime_deoptimization_map_test
  SRCS deoptimization_map_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::algorithm
       s6_runtime_deoptimization_map
       s6_runtime_slot_indexes
       s6_strongjit_base
       s6_strongjit_builder)

s6_cc_library(
  NAME s6_runtime_generator
  SRCS generator.cc
  HDRS generator.h
  DEPS Python::Python
       s6_code_object
       s6_strongjit_base
       s6_strongjit_formatter
       s6_utils_logging)

s6_cc_library(
  NAME s6_runtime_interposer
  SRCS interposer.cc
  HDRS interposer.h
  DEPS Python::Python
       absl::base
       s6_code_object
       s6_core_util
       s6_interpreter
       s6_runtime_deoptimization_runtime
       s6_strongjit_formatter
       LINKOPTS
       -Wl,--wrap=PyFrame_FastToLocalsWithError)

s6_cc_library(
  NAME s6_runtime_python_function_info_table
  SRCS python_function_info_table.cc
  HDRS python_function_info_table.h
  DEPS Python::Python
       absl::node_hash_map
       s6_core_util
       s6_event_counters
       s6_global_intern_table
       s6_utils_logging
       s6_utils_no_destructor)
