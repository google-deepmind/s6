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

#include "strongjit/util.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

using ::testing::ElementsAre;

TEST(UtilTest, MultiLineVector) {
  {
    MultiLineVector<int, 4> mlvec;
    mlvec.push_back(1);
    mlvec.push_back(2);
    mlvec.push_line();
    mlvec.push_back(3);
    mlvec.push_back(4);
    mlvec.push_line();
    mlvec.push_line();
    mlvec.push_back(5);
    ASSERT_THAT(mlvec, ElementsAre(1, 2, 3, 4, 5));
    ASSERT_EQ(mlvec.line_num(), 4);
    ASSERT_THAT(mlvec.line(0), ElementsAre(1, 2));
    ASSERT_THAT(mlvec.line(1), ElementsAre(3, 4));
    ASSERT_TRUE(mlvec.line(2).empty());
    ASSERT_THAT(mlvec.line(3), ElementsAre(5));

    auto line1 = mlvec.line(1);
    mlvec.line(0).push_back(6);
    mlvec.insert_at(3, 7);
    ASSERT_THAT(mlvec, ElementsAre(1, 2, 6, 7, 3, 4, 5));
    ASSERT_EQ(mlvec.line_num(), 4);
    ASSERT_THAT(mlvec.line(0), ElementsAre(1, 2, 6));
    ASSERT_THAT(mlvec.line(1), ElementsAre(7, 3, 4));
    ASSERT_TRUE(mlvec.line(2).empty());
    ASSERT_THAT(mlvec.line(3), ElementsAre(5));

    line1.insert_at(2, 8);
    mlvec.insert_at(7, 9);
    line1.push_back(10);
    ASSERT_THAT(mlvec, ElementsAre(1, 2, 6, 7, 3, 8, 4, 10, 9, 5));
    ASSERT_EQ(mlvec.line_num(), 4);
    ASSERT_THAT(mlvec.line_span(0), ElementsAre(1, 2, 6));
    ASSERT_THAT(line1, ElementsAre(7, 3, 8, 4, 10));
    ASSERT_THAT(line1.SubLine(2), ElementsAre(8, 4, 10));
    ASSERT_TRUE(mlvec.line(2).empty());
    ASSERT_THAT(mlvec.line(3), ElementsAre(9, 5));

    mlvec.line(2).push_back(11);
    mlvec.line(2).push_back(12);
    line1.pop_back();
    line1.erase_at(2);
    mlvec.erase_at(1);
    ASSERT_THAT(mlvec, ElementsAre(1, 6, 7, 3, 4, 11, 12, 9, 5));
    ASSERT_EQ(mlvec.line_num(), 4);
    ASSERT_THAT(mlvec.line(0), ElementsAre(1, 6));
    ASSERT_THAT(line1, ElementsAre(7, 3, 4));
    ASSERT_THAT(mlvec.line_span(2), ElementsAre(11, 12));
    ASSERT_THAT(mlvec.line(3), ElementsAre(9, 5));

    mlvec.pop_line();
    mlvec.erase_at(4);
    line1.insert_at(0, 13);
    mlvec.resize(8);
    mlvec.back() = 14;
    ASSERT_THAT(mlvec, ElementsAre(1, 6, 13, 7, 3, 11, 12, 14));
    ASSERT_EQ(mlvec.line_num(), 3);
    ASSERT_THAT(mlvec.line(0), ElementsAre(1, 6));
    ASSERT_THAT(line1, ElementsAre(13, 7, 3));
    ASSERT_THAT(mlvec.line_span(1, 1), ElementsAre(7, 3));
    ASSERT_THAT(mlvec.line(2), ElementsAre(11, 12, 14));
    ASSERT_THAT(mlvec.line(2, 1), ElementsAre(12, 14));

    mlvec.resize(4);
    mlvec.resize(5);
    mlvec.line(0).resize(4);
    mlvec.line(0)[3] = 20;
    line1.back() = 22;
    ASSERT_THAT(mlvec, ElementsAre(1, 6, 0, 20, 13, 7, 22));
    ASSERT_EQ(mlvec.line_num(), 2);
    ASSERT_THAT(mlvec.line(0), ElementsAre(1, 6, 0, 20));
    ASSERT_THAT(line1, ElementsAre(13, 7, 22));
  }
}

TEST(UtilTest, FixedLengthBitVector) {
  FixedLengthBitVector<10> bits{3, 5, 9};

  ASSERT_FALSE(bits.IsSet(0));
  ASSERT_TRUE(bits.IsSet(3));
  ASSERT_FALSE(bits.IsSet(4));
  ASSERT_TRUE(bits.IsSet(5));
  ASSERT_FALSE(bits.IsSet(6));
  ASSERT_TRUE(bits.IsSet(9));
}

}  // namespace
}  // namespace deepmind::s6
