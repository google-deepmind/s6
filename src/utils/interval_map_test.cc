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

#include "utils/interval_map.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
using testing::ElementsAre;
using testing::Key;
using testing::Pair;

TEST(IntervalMap, NonOverlapping) {
  IntervalMap<int64_t, std::string> map;
  map.Set(4, 10, "hello");
  map.Set(12, 14, "world");
  map.Set(10, 12, "yes");

  EXPECT_THAT(map,
              ElementsAre(Pair(Pair(4, 10), "hello"), Pair(Pair(10, 12), "yes"),
                          Pair(Pair(12, 14), "world")));
}

TEST(IntervalMap, Overlapping1) {
  IntervalMap<int64_t, std::string> map;
  map.Set(4, 10, "hello");
  map.Set(2, 6, "world");

  EXPECT_THAT(
      map, ElementsAre(Pair(Pair(2, 6), "world"), Pair(Pair(6, 10), "hello")));
}

TEST(IntervalMap, Overlapping2) {
  IntervalMap<int64_t, std::string> map;
  map.Set(4, 10, "hello");
  map.Set(6, 13, "world");

  EXPECT_THAT(
      map, ElementsAre(Pair(Pair(4, 6), "hello"), Pair(Pair(6, 13), "world")));
}

TEST(IntervalMap, Subsuming) {
  IntervalMap<int64_t, std::string> map;
  map.Set(4, 10, "hello");
  map.Set(2, 14, "world");

  EXPECT_THAT(map, ElementsAre(Pair(Pair(2, 14), "world")));
}

TEST(IntervalMap, Erase) {
  IntervalMap<int64_t, std::string> map;
  map.Set(4, 10, "hello");
  map.Erase(6, 8);

  EXPECT_THAT(
      map, ElementsAre(Pair(Pair(4, 6), "hello"), Pair(Pair(8, 10), "hello")));
}

TEST(IntervalMap, Coalescing1) {
  IntervalMap<int64_t, std::string> map;
  map.Set(2, 6, "hello");
  map.Set(6, 8, "hello");

  EXPECT_THAT(map, ElementsAre(Pair(Pair(2, 8), "hello")));
}

TEST(IntervalMap, Coalescing2) {
  IntervalMap<int64_t, std::string> map;
  map.Set(6, 8, "hello");
  map.Set(2, 6, "hello");

  EXPECT_THAT(map, ElementsAre(Pair(Pair(2, 8), "hello")));
}

TEST(IntervalMap, NonEqualityComparableKeyDoesNotCoalesce) {
  class C {
   public:
    explicit C(std::string s) : s_(s) {}

   private:
    std::string s_;
  };

  // Ensure that IntervalMap can compile.
  IntervalMap<int64_t, C> map;
  map.Set(6, 8, C("hello"));
  map.Set(2, 6, C("hello"));

  EXPECT_THAT(map, ElementsAre(Key(Pair(2, 6)), Key(Pair(6, 8))));
}

}  // namespace
}  // namespace deepmind::s6
