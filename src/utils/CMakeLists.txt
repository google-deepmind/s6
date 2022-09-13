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


s6_cc_library(NAME s6_utils_intrusive_list HDRS intrusive_list.h)

s6_cc_test(
  NAME s6_utils_intrusive_list_test
  SRCS intrusive_list_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_intrusive_list)

s6_cc_library(NAME s6_utils_no_destructor HDRS no_destructor.h)

s6_cc_test(
  NAME s6_utils_no_destructor_test
  SRCS no_destructor_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::flat_hash_map
       s6_utils_no_destructor)

s6_cc_library(
  NAME s6_utils_path
  SRCS path.cc
  HDRS path.h
  DEPS absl::strings)

s6_cc_test(
  NAME s6_utils_path_test
  SRCS path_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_path)

s6_cc_library(NAME s6_utils_range HDRS range.h)

s6_cc_test(
  NAME s6_utils_range_test
  SRCS range_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_range)

s6_cc_library(
  NAME s6_utils_demangle
  SRCS demangle.cc
  HDRS demangle.h)

s6_cc_test(
  NAME s6_utils_demangle_test
  SRCS demangle_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_demangle)

s6_cc_library(
  NAME s6_utils_diffs
  HDRS diffs.h
  DEPS absl::flat_hash_map s6_utils_logging)

s6_cc_test(
  NAME s6_utils_diffs_test
  SRCS diffs_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_diffs)

s6_cc_library(
  NAME s6_utils_logging
  SRCS logging.cc
  HDRS logging.h
  DEPS absl::base
       absl::flags
       absl::status
       absl::statusor
       absl::strings
       absl::synchronization
       s6_utils_no_destructor)

s6_cc_library(
  NAME s6_utils_mathutil
  HDRS mathutil.h
  DEPS s6_utils_logging)

s6_cc_test(
  NAME s6_utils_mathutil_test
  SRCS mathutil_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_mathutil)

s6_cc_library(
  NAME s6_utils_channel
  HDRS channel.h
  DEPS absl::base absl::synchronization s6_utils_logging)

s6_cc_test(
  NAME s6_utils_channel_test
  SRCS channel_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::synchronization
       s6_utils_channel)

s6_cc_library(
  NAME s6_utils_status_macros
  HDRS status_macros.h
  DEPS s6_utils_logging s6_utils_status_builder)

s6_cc_library(
  NAME s6_utils_status_builder
  HDRS status_builder.h
  DEPS absl::status)

s6_cc_test(
  NAME s6_utils_status_builder_test
  SRCS status_builder_test.cc
  DEPS GTest::gmock
       GTest::gtest_main
       absl::status
       s6_utils_status_builder)

s6_cc_library(
  NAME s6_utils_interval_map
  HDRS interval_map.h
  DEPS s6_utils_logging)

s6_cc_test(
  NAME s6_utils_interval_map_test
  SRCS interval_map_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_interval_map)

s6_cc_library(
  NAME s6_utils_keyed_intern_table
  HDRS keyed_intern_table.h
  DEPS absl::flat_hash_set s6_utils_logging)

s6_cc_test(
  NAME s6_utils_keyed_intern_table_test
  SRCS keyed_intern_table_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_keyed_intern_table)

s6_cc_library(
  NAME s6_utils_inlined_bit_vector
  HDRS inlined_bit_vector.h
  DEPS absl::base
       absl::inlined_vector
       absl::numeric
       s6_utils_logging
       s6_utils_mathutil)

s6_cc_test(
  NAME s6_utils_inlined_bit_vector_test
  SRCS inlined_bit_vector_test.cc
  DEPS GTest::gmock GTest::gtest_main s6_utils_inlined_bit_vector)

s6_cc_library(
  NAME s6_utils_matchers
  SRCS matchers.cc
  HDRS matchers.h
  DEPS GTest::gmock
       GTest::gtest
       absl::status
       absl::statusor
       s6_utils_status_builder)
