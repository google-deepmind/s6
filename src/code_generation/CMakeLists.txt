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
  NAME s6_code_generation_live_interval
  HDRS live_interval.h
  DEPS absl::algorithm
       absl::base
       absl::inlined_vector
       absl::strings
       s6_core_util
       s6_strongjit_base
       s6_strongjit_formatter)

s6_cc_library(
  NAME s6_code_generation_register_allocator
  SRCS register_allocator.cc
  HDRS register_allocator.h
  DEPS absl::algorithm
       absl::base
       absl::flat_hash_map
       absl::flat_hash_set
       absl::statusor
       absl::strings
       asmjit
       s6_code_generation_live_interval
       s6_core_util
       s6_runtime_deoptimization_map
       s6_runtime_slot_indexes
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_ssa
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_code_generation_code_generator
  SRCS code_generator.cc
  HDRS code_generator.h
  DEPS Python::Python
       absl::base
       absl::flags
       absl::flat_hash_map
       absl::status
       absl::statusor
       absl::strings
       asmjit
       s6_allocator
       s6_classes_class
       s6_code_generation_call_stub
       s6_code_generation_code_generation_context
       s6_code_generation_jit_stub
       s6_code_generation_prolog_epilog_insertion
       s6_code_generation_register_allocator
       s6_code_object
       s6_core_util
       s6_event_counters
       s6_metadata
       s6_runtime_callee_address
       s6_runtime_deoptimization_runtime
       s6_runtime_generator
       s6_runtime_pyframe_object_cache
       s6_runtime_runtime
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_ingestion
       s6_strongjit_optimizer
       s6_utils_logging
       s6_utils_path
       s6_utils_status_macros
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_code_generation_code_generator_test
  SRCS code_generator_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::status
       absl::statusor
       absl::strings
       s6_allocator
       s6_api
       s6_code_generation_code_generator
       s6_code_generation_register_allocator
       s6_code_generation_trace_register_allocator
       s6_code_object
       s6_event_counters
       s6_metadata
       s6_runtime_runtime
       s6_runtime_stack_frame
       s6_strongjit_base
       s6_strongjit_parser
       s6_strongjit_test_util
       s6_utils_matchers
       s6_utils_no_destructor
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_code_generation_jit_stub
  HDRS jit_stub.h
  DEPS Python::Python
       asmjit
       s6_runtime_generator
       s6_runtime_stack_frame
       s6_runtime_util
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_code_generation_code_generation_context
  SRCS code_generation_context.cc
  HDRS code_generation_context.h
  DEPS asmjit
       s6_code_generation_jit_stub
       s6_code_generation_register_allocator
       s6_runtime_stack_frame
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_code_generation_prolog_epilog_insertion
  SRCS prolog_epilog_insertion.cc
  HDRS prolog_epilog_insertion.h
  DEPS Python::Python
       absl::status
       s6_code_generation_asmjit_util
       s6_code_generation_call_stub
       s6_code_generation_code_generation_context
       s6_code_generation_jit_stub
       s6_code_generation_register_allocator
       s6_metadata
       s6_runtime_generator
       s6_runtime_pyframe_object_cache
       s6_runtime_runtime
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_code_generation_call_stub
  HDRS call_stub.h
  DEPS Python::Python asmjit s6_runtime_runtime
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_code_generation_asmjit_util
  HDRS asmjit_util.h
  DEPS absl::status
       absl::strings
       asmjit
       s6_utils_status_macros
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_code_generation_jit_stub_test
  SRCS jit_stub_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       asmjit
       s6_code_generation_asmjit_util
       s6_code_generation_jit_stub
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})

s6_cc_library(
  NAME s6_code_generation_trace_register_allocator
  SRCS trace_register_allocator.cc
  HDRS trace_register_allocator.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::flat_hash_map
       absl::flat_hash_set
       absl::status
       absl::statusor
       cppitertools::cppitertools
       s6_code_generation_asmjit_util
       s6_code_generation_live_interval
       s6_code_generation_register_allocator
       s6_core_util
       s6_event_counters
       s6_runtime_slot_indexes
       s6_strongjit_base
       s6_strongjit_builder
       s6_strongjit_formatter
       s6_strongjit_ssa
       s6_utils_logging
       s6_utils_status_macros
  INCLUDE_DIRS ${cppitertools_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_code_generation_trace_register_allocator_test
  SRCS trace_register_allocator_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::status
       absl::strings
       s6_code_generation_trace_register_allocator
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_optimizer_util
       s6_strongjit_parser
       s6_strongjit_test_util
       s6_utils_matchers
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_code_generation_thunk_generator
  SRCS thunk_generator.cc
  HDRS thunk_generator.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::status
       absl::statusor
       absl::strings
       asmjit
       s6_allocator
       s6_code_generation_asmjit_util
       s6_utils_logging
       s6_utils_status_macros
  INCLUDE_DIRS ${asmjit_INCLUDE_DIRS})
