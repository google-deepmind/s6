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

#include "utils/mathutil.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
TEST(MathUtil, RoundUpTo) {
  EXPECT_EQ(RoundUpTo<int64_t>(10, 1), 10);
  EXPECT_EQ(RoundUpTo<int64_t>(10, 2), 10);
  EXPECT_EQ(RoundUpTo<int64_t>(10, 5), 10);
  EXPECT_EQ(RoundUpTo<int64_t>(10, 10), 10);
  EXPECT_EQ(RoundUpTo<int64_t>(10, 3), 12);
  EXPECT_EQ(RoundUpTo<int32_t>(9, 10), 10);
  EXPECT_EQ(RoundUpTo<int32_t>(11, 10), 20);
  EXPECT_EQ(RoundUpTo<int32_t>(0, 10), 0);
  EXPECT_EQ(RoundUpTo<int64_t>(10000000001, 10000000000), 20000000000);
  EXPECT_EQ(RoundUpTo<int64_t>(9999999999, 10000000000), 10000000000);
  EXPECT_EQ(RoundUpTo<int64_t>(1, 10000000000), 10000000000);
}

TEST(MathUtil, RoundDownTo) {
  EXPECT_EQ(RoundDownTo<int64_t>(10, 1), 10);
  EXPECT_EQ(RoundDownTo<int64_t>(10, 2), 10);
  EXPECT_EQ(RoundDownTo<int64_t>(10, 5), 10);
  EXPECT_EQ(RoundDownTo<int64_t>(10, 10), 10);
  EXPECT_EQ(RoundDownTo<int64_t>(10, 3), 9);
  EXPECT_EQ(RoundDownTo<int32_t>(9, 10), 0);
  EXPECT_EQ(RoundDownTo<int32_t>(11, 10), 10);
  EXPECT_EQ(RoundDownTo<int32_t>(0, 10), 0);
  EXPECT_EQ(RoundDownTo<int64_t>(10000000001, 10000000000), 10000000000);
  EXPECT_EQ(RoundDownTo<int64_t>(9999999999, 10000000000), 0);
  EXPECT_EQ(RoundDownTo<int64_t>(1, 10000000000), 0);
}
}  // namespace
}  // namespace deepmind::s6
