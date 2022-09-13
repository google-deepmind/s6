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


# ==============================================================================
# Top-level setup
# ==============================================================================
cmake_minimum_required(VERSION 3.16.3)

project(s6)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/CMake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(deps)
include(s6)

enable_testing()

s6_cc_library(
  NAME s6_global_intern_table
  SRCS global_intern_table.cc
  HDRS global_intern_table.h
  DEPS absl::base
       absl::strings
       absl::synchronization
       s6_utils_keyed_intern_table
       s6_utils_no_destructor)

s6_cc_proto_library(NAME s6_event_counters_cc_proto SRCS event_counters.proto)

s6_cc_library(
  NAME s6_event_counters
  SRCS event_counters.cc
  HDRS event_counters.h
  DEPS ${Protobuf_LIBRARY}
       absl::flat_hash_map
       absl::status
       absl::strings
       absl::synchronization
       s6_event_counters_cc_proto
       s6_global_intern_table
       s6_utils_no_destructor
       s6_utils_path
       s6_utils_status_macros
  INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_core_util
  SRCS core_util.cc
  HDRS core_util.h
  DEPS Python::Python
       absl::base
       absl::status
       absl::strings
       asmjit
       s6_utils_logging
       s6_utils_status_macros
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_api
  SRCS api.cc
  HDRS api.h
  DEPS Python::Python
       absl::base
       absl::flags_parse
       absl::status
       absl::statusor
       absl::strings
       s6_classes_object
       s6_core_util
       s6_eval
       s6_interpreter
       s6_oracle
       s6_runtime_interposer
       s6_strongjit_optimizer_util
       s6_utils_logging
       s6_utils_no_destructor
       s6_utils_path
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_eval
  SRCS eval.cc
  HDRS eval.h
  DEPS Python::Python
       absl::flags
       absl::statusor
       absl::strings
       s6_code_object
       s6_interpreter
       s6_oracle
       s6_runtime_deoptimization_runtime
       s6_runtime_evaluator
       s6_runtime_generator
       s6_utils_logging)

s6_cc_library(
  NAME s6_tuple_util
  HDRS tuple_util.h
  DEPS absl::status absl::statusor s6_core_util)

s6_cc_test(
  NAME s6_tuple_util_test
  SRCS tuple_util_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::status
       absl::statusor
       s6_tuple_util
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_arithmetic
  SRCS arithmetic.cc
  HDRS arithmetic.h
  DEPS absl::base s6_utils_logging)

s6_cc_library(
  NAME s6_interpreter
  SRCS interpreter.cc
  HDRS interpreter.h
  DEPS Python::Python
       absl::base
       absl::flat_hash_map
       absl::strings
       s6_classes_object
       s6_classes_util
       s6_core_util
       s6_global_intern_table
       s6_metadata
       s6_runtime_python_function_info_table
       s6_utils_logging)

s6_cc_library(
  NAME s6_metadata
  SRCS metadata.cc
  HDRS metadata.h
  DEPS Python::Python
       absl::base
       absl::fixed_array
       absl::flags
       absl::inlined_vector
       absl::status
       absl::strings
       absl::synchronization
       s6_allocator
       s6_classes_util
       s6_code_generation_thunk_generator
       s6_code_object
       s6_core_util
       s6_runtime_interpreter_stub
       s6_type_feedback
       s6_utils_logging)

s6_cc_library(
  NAME s6_allocator
  SRCS allocator.cc
  HDRS allocator.h
  DEPS absl::base
       absl::flat_hash_map
       absl::flat_hash_set
       absl::strings
       absl::synchronization
       s6_utils_interval_map
       s6_utils_logging
       s6_utils_mathutil)

s6_cc_library(
  NAME s6_code_object
  SRCS code_object.cc
  HDRS code_object.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::flat_hash_map
       absl::memory
       absl::status
       absl::strings
       s6_allocator
       s6_core_util
       s6_runtime_deoptimization_map
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_utils_logging
       udis86
  INCLUDE_DIRS ${udis86_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_oracle
  SRCS oracle.cc
  HDRS oracle.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::fixed_array
       absl::flags
       absl::flat_hash_map
       absl::flat_hash_set
       absl::inlined_vector
       absl::memory
       absl::status
       absl::statusor
       absl::strings
       absl::synchronization
       absl::time
       asmjit
       s6_allocator
       s6_classes_class
       s6_code_generation_code_generator
       s6_code_generation_register_allocator
       s6_code_generation_thunk_generator
       s6_code_generation_trace_register_allocator
       s6_code_object
       s6_core_util
       s6_event_counters
       s6_global_intern_table
       s6_metadata
       s6_strongjit_base
       s6_strongjit_deoptimization
       s6_strongjit_ingestion
       s6_strongjit_optimizer
       s6_strongjit_util
       s6_type_feedback
       s6_utils_channel
       s6_utils_inlined_bit_vector
       s6_utils_logging
       s6_utils_no_destructor
       s6_utils_status_macros
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_oracle_test
  SRCS oracle_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::statusor
       s6_api
       s6_oracle)

s6_cc_test(
  NAME s6_integration_test
  SRCS integration_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::status
       absl::statusor
       s6_api
       s6_code_generation_code_generator
       s6_code_generation_register_allocator
       s6_code_generation_trace_register_allocator
       s6_core_util
       s6_metadata
       s6_runtime_evaluator
       s6_strongjit_deoptimization
       s6_strongjit_formatter
       s6_strongjit_ingestion
       s6_strongjit_optimizer
       s6_strongjit_util
       s6_utils_no_destructor
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_type_feedback
  SRCS type_feedback.cc
  HDRS type_feedback.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::inlined_vector
       absl::strings
       s6_classes_class
       s6_utils_logging)

s6_cc_test(
  NAME s6_type_feedback_test
  SRCS type_feedback_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       s6_type_feedback)

add_subdirectory(classes)
add_subdirectory(code_generation)
add_subdirectory(runtime)
add_subdirectory(strongjit)
add_subdirectory(utils)
add_subdirectory(python)
