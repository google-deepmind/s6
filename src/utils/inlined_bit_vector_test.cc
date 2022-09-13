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

#include "utils/inlined_bit_vector.h"

#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

template <size_t N>
std::string ToString(const InlinedBitVector<N>& bits) {
  std::stringstream ss;
  for (size_t i = 0; i < bits.size(); ++i) {
    ss << (bits[i] ? 1 : 0);
  }
  return ss.str();
}

TEST(InlinedBitVector, ConstructedWithSizeZero) {
  InlinedBitVector<5> b;
  EXPECT_EQ(b.size(), 0);
}

TEST(InlinedBitVector, ConstructedWithBitsSetToZero) {
  InlinedBitVector<5> b(127);

  EXPECT_EQ(b.size(), 127);

  for (size_t i = 0, size = b.size(); i < size; ++i) {
    EXPECT_EQ(b.get_bit(i), false);
  }
}

TEST(InlinedBitVector, GetSetOnMultipleWordVector) {
  InlinedBitVector<5> b(127);
  for (size_t i = 0, size = b.size(); i < size; ++i) {
    EXPECT_EQ(b.get_bit(i), false);
    b.set_bit(i);
    EXPECT_EQ(b.get_bit(i), true);
    b.clear_bit(i);
    EXPECT_EQ(b.get_bit(i), false);
  }
}

TEST(InlinedBitVector, SetBits) {
  InlinedBitVector<5> b(11);

  // In the middle.
  EXPECT_EQ(ToString(b), "00000000000");
  b.SetBits(4, 8);
  EXPECT_EQ(ToString(b), "00001111000");

  // At the beginning.
  b.ClearBits(0, b.size());
  EXPECT_EQ(ToString(b), "00000000000");
  b.SetBits(0, 5);
  EXPECT_EQ(ToString(b), "11111000000");

  // At the end.
  b.ClearBits(0, b.size());
  EXPECT_EQ(ToString(b), "00000000000");
  b.SetBits(7, 11);
  EXPECT_EQ(ToString(b), "00000001111");
}

TEST(InlinedBitVector, ClearBits) {
  InlinedBitVector<5> b(11);

  // In the middle.
  b.SetBits(0, b.size());
  EXPECT_EQ(ToString(b), "11111111111");
  b.ClearBits(4, 8);
  EXPECT_EQ(ToString(b), "11110000111");

  // At the beginning.
  b.SetBits(0, b.size());
  EXPECT_EQ(ToString(b), "11111111111");
  b.ClearBits(0, 5);
  EXPECT_EQ(ToString(b), "00000111111");

  // At the end.
  b.SetBits(0, b.size());
  EXPECT_EQ(ToString(b), "11111111111");
  b.ClearBits(7, 11);
  EXPECT_EQ(ToString(b), "11111110000");
}

TEST(InlinedBitVector, resize) {
  InlinedBitVector<5> b(11);
  for (size_t i = 0, size = b.size(); i < size; i += 2) {
    b.set_bit(i);
  }
  EXPECT_EQ(ToString(b), "10101010101");

  // resize to shrink.
  b.resize(5);
  EXPECT_EQ(ToString(b), "10101");

  // resize to enlargen - check formerly junk bits are zero.
  b.resize(13);
  EXPECT_EQ(ToString(b), "1010100000000");
}

TEST(InlinedBitVector, count) {
  InlinedBitVector<5> b(11);
  for (size_t i = 0, size = b.size(); i < size; i += 2) {
    b.set_bit(i);
  }
  EXPECT_EQ(ToString(b), "10101010101");
  EXPECT_EQ(b.count(), 6);
}

TEST(InlinedBitVector, clear) {
  InlinedBitVector<5> b(11);

  EXPECT_EQ(b.size(), 11);

  b.clear();
  EXPECT_EQ(b.size(), 0);
}

TEST(InlinedBitVector, Difference) {
  InlinedBitVector<5> b1(5);
  b1.set_bit(1);
  b1.set_bit(3);
  EXPECT_EQ(ToString(b1), "01010");

  InlinedBitVector<5> b2(4);
  b2.set_bit(1);
  b2.set_bit(2);
  EXPECT_EQ(ToString(b2), "0110");

  b1.Difference(b2);
  EXPECT_EQ(ToString(b1), "00010");
}

TEST(InlinedBitVector, Union) {
  InlinedBitVector<5> b1(5);
  b1.set_bit(1);
  b1.set_bit(3);
  EXPECT_EQ(ToString(b1), "01010");

  InlinedBitVector<5> b2(4);
  b2.set_bit(1);
  b2.set_bit(2);
  EXPECT_EQ(ToString(b2), "0110");

  b1.Union(b2);
  EXPECT_EQ(ToString(b1), "01110");
}

TEST(InlinedBitVector, Intersection) {
  InlinedBitVector<5> b1(5);
  b1.set_bit(1);
  b1.set_bit(3);
  EXPECT_EQ(ToString(b1), "01010");

  InlinedBitVector<5> b2(4);
  b2.set_bit(1);
  b2.set_bit(2);
  EXPECT_EQ(ToString(b2), "0110");

  b1.Intersection(b2);
  EXPECT_EQ(ToString(b1), "01000");
}

TEST(InlinedBitVector, EqualityAndInequality) {
  InlinedBitVector<5> b1(5);
  InlinedBitVector<5> b2(5);
  b2.set_bit(3);

  EXPECT_EQ(b1, b1);
  EXPECT_EQ(b2, b2);
  EXPECT_NE(b1, b2);
}

}  // namespace
}  // namespace deepmind::s6
