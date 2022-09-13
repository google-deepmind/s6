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
  NAME s6_strongjit_util
  SRCS util.cc
  HDRS util.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::inlined_vector
       absl::strings
       s6_core_util
       s6_utils_logging
       s6_utils_range)

s6_cc_test(
  NAME s6_strongjit_util_test
  SRCS util_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_util)

s6_cc_library(
  NAME s6_strongjit_base
  SRCS block.cc
       callees.cc
       cursor.cc
       function.cc
       instruction.cc
       instructions.cc
       value_map.cc
  HDRS base.h
       block.h
       callees.h
       callees.inc
       cursor.h
       function.h
       instruction.h
       instruction_traits.h
       instructions.h
       string_table.h
       value.h
       value_casts.h
       value_map.h
       value_traits.h
  DEPS absl::algorithm
       absl::base
       absl::flat_hash_map
       absl::inlined_vector
       absl::status
       absl::statusor
       absl::strings
       s6_arithmetic
       s6_classes_class
       s6_core_util
       s6_global_intern_table
       s6_strongjit_util
       s6_type_feedback
       s6_utils_intrusive_list
       s6_utils_keyed_intern_table
       s6_utils_logging)

s6_cc_test(
  NAME s6_strongjit_block_test
  SRCS block_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_base)

s6_cc_test(
  NAME s6_strongjit_cursor_test
  SRCS cursor_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_base)

s6_cc_test(
  NAME s6_strongjit_instructions_test
  SRCS instructions_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_base)

s6_cc_test(
  NAME s6_strongjit_value_traits_test
  SRCS value_traits_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_base)

s6_cc_library(
  NAME s6_strongjit_formatter
  SRCS formatter.cc
  HDRS formatter.h
  DEPS absl::base
       absl::flat_hash_map
       absl::status
       absl::statusor
       absl::strings
       s6_classes_class
       s6_strongjit_base
       s6_strongjit_util
       s6_tuple_util
       s6_utils_no_destructor
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_strongjit_test_util
  HDRS test_util.h
  DEPS GTest::gmock
       GTest::gtest
       absl::algorithm
       absl::status
       absl::strings
       re2
       s6_utils_logging
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_strongjit_test_util_test
  SRCS test_util_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::status
       s6_strongjit_test_util
       s6_utils_matchers)

s6_cc_test(
  NAME s6_strongjit_formatter_test
  SRCS formatter_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::statusor
       absl::strings
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_test_util
       s6_type_feedback
       s6_utils_matchers
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_strongjit_instruction_traits_test
  SRCS instruction_traits_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_strongjit_base)

s6_cc_library(
  NAME s6_strongjit_parser
  SRCS parser.cc
  HDRS parser.h
  DEPS absl::algorithm
       absl::status
       absl::statusor
       absl::strings
       s6_classes_class
       s6_global_intern_table
       s6_strongjit_base
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_strongjit_parser_test
  SRCS parser_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::status
       absl::statusor
       s6_classes_class
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_parser
       s6_utils_matchers)

s6_cc_library(
  NAME s6_strongjit_builder
  HDRS builder.h
  DEPS absl::inlined_vector
       absl::strings
       s6_event_counters
       s6_global_intern_table
       s6_strongjit_base)

s6_cc_library(
  NAME s6_strongjit_ssa
  SRCS ssa.cc
  HDRS ssa.h
  DEPS absl::algorithm
       absl::base
       absl::flat_hash_map
       absl::flat_hash_set
       s6_strongjit_base
       s6_strongjit_formatter
       s6_utils_inlined_bit_vector)

s6_cc_test(
  NAME s6_strongjit_ssa_test
  SRCS ssa_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       s6_strongjit_base
       s6_strongjit_builder
       s6_strongjit_formatter
       s6_strongjit_ssa
       s6_strongjit_test_util)

s6_cc_library(
  NAME s6_strongjit_ingestion
  SRCS ingestion.cc ingestion_handlers.cc
  HDRS ingestion.h ingestion_handlers.h
  DEPS Python::Python
       absl::algorithm
       absl::inlined_vector
       absl::status
       absl::statusor
       absl::strings
       cppitertools::cppitertools
       s6_code_object
       s6_core_util
       s6_interpreter
       s6_strongjit_base
       s6_strongjit_builder
       s6_strongjit_formatter
       s6_strongjit_ssa
       s6_strongjit_util
       s6_type_feedback
       s6_utils_logging
       s6_utils_no_destructor
       s6_utils_path
       s6_utils_range
       s6_utils_status_macros
  INCLUDE_DIRS ${cppitertools_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_strongjit_ingestion_test
  SRCS ingestion_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       s6_strongjit_formatter
       s6_strongjit_ingestion
       s6_strongjit_test_util
       s6_utils_status_macros)

s6_cc_library(
  NAME s6_strongjit_optimizer_util
  SRCS optimizer_util.cc
  HDRS optimizer_util.h
  DEPS Python::Python
       absl::base
       absl::flat_hash_map
       absl::inlined_vector
       absl::status
       absl::strings
       s6_classes_class
       s6_core_util
       s6_event_counters
       s6_strongjit_base
       s6_strongjit_builder
       s6_strongjit_formatter
       s6_strongjit_util
       s6_tuple_util
       s6_utils_logging
       s6_utils_no_destructor)

s6_cc_library(
  NAME s6_strongjit_optimizer
  SRCS optimize_calls.cc
       optimize_cfg.cc
       optimize_constants.cc
       optimize_liveness.cc
       optimize_nullconst.cc
       optimize_refcount.cc
       optimize_type_construction.cc
       optimizer.cc
  HDRS optimize_calls.h
       optimize_cfg.h
       optimize_constants.h
       optimize_liveness.h
       optimize_nullconst.h
       optimize_refcount.h
       optimize_type_construction.h
       optimizer.h
       optimizer_analysis.h
  DEPS Python::Python
       absl::algorithm
       absl::base
       absl::flags
       absl::flat_hash_map
       absl::flat_hash_set
       absl::inlined_vector
       absl::status
       absl::statusor
       absl::strings
       cppitertools::cppitertools
       s6_classes_class
       s6_core_util
       s6_event_counters
       s6_runtime_pyframe_object_cache
       s6_runtime_python_function_info_table
       s6_strongjit_base
       s6_strongjit_builder
       s6_strongjit_formatter
       s6_strongjit_optimizer_util
       s6_tuple_util
       s6_type_feedback
       s6_utils_diffs
       s6_utils_inlined_bit_vector
       s6_utils_logging
       s6_utils_no_destructor
       s6_utils_path
       s6_utils_range
       s6_utils_status_macros
  INCLUDE_DIRS ${cppitertools_INCLUDE_DIRS})

s6_cc_test(
  NAME s6_strongjit_optimizer_test
  SRCS optimizer_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       absl::status
       absl::strings
       s6_api
       s6_classes_class
       s6_interpreter
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_optimizer
       s6_strongjit_optimizer_util
       s6_strongjit_parser
       s6_strongjit_test_util
       s6_utils_matchers
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_strongjit_optimizer_util_test
  SRCS optimizer_util_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       Python::Python
       s6_api
       s6_strongjit_formatter
       s6_strongjit_optimizer_util
       s6_strongjit_parser
       s6_strongjit_test_util
       s6_utils_status_macros)

s6_pybind_extension(
  NAME s6_strongjit_dis6
  OUTPUT_NAME dis6
  SRCS dis6.cc
  DEPS absl::status
       absl::statusor
       s6_code_object
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_ingestion
       s6_utils_status_macros
  INCLUDE_DIRS ${pybind11_abseil_INCLUDE_DIRS})

s6_py_test(NAME s6_strongjit_dis6_test SRCS dis6_test.py)

s6_cc_library(
  NAME s6_strongjit_deoptimization
  SRCS deoptimization.cc
  HDRS deoptimization.h
  DEPS absl::algorithm
       absl::flat_hash_set
       absl::status
       s6_strongjit_base
       s6_strongjit_formatter
       s6_strongjit_ssa
       s6_strongjit_util
       s6_utils_status_macros)

s6_cc_test(
  NAME s6_strongjit_deoptimization_test
  SRCS deoptimization_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       s6_strongjit_base
       s6_strongjit_deoptimization
       s6_strongjit_formatter
       s6_strongjit_parser
       s6_strongjit_test_util
       s6_utils_status_macros)
