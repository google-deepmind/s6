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

#include "utils/diffs.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

TEST(DiffVecTest, IntegerTest) {
  // A DiffVecTrait implementation for int.
  class DiffVecTraitsInt {
   public:
    using Diff = int;
    static void Apply(int& t, int d) { t += d; }
    /*static Diff MakeDiff(const T& origin, const T& dest) {
      return origin.MakeDiff(dest);
    }*/
  };
  DiffVec<int, DiffVecTraitsInt> dv;

  dv.set_front(4);
  dv.push_back(2);
  dv.push_back(-3);
  ASSERT_EQ(dv.size(), 3);
  ASSERT_EQ(dv.at(0), 4);
  ASSERT_EQ(dv.at(1), 6);
  ASSERT_EQ(dv.at(2), 3);

  dv.resize(2);
  ASSERT_EQ(dv.size(), 2);
  ASSERT_EQ(dv.at(0), 4);
  ASSERT_EQ(dv.at(1), 6);

  auto cur = dv.BeginCursor();
  ASSERT_EQ(*cur, 4);
  cur.StepForward();
  ASSERT_EQ(*cur, 6);
  dv.push_back_maintain(-5, cur);
  // it is still valid because it was maintained.
  ASSERT_EQ(*cur, 6);
  ASSERT_TRUE(cur.NextDiff());
  ASSERT_EQ(*cur.NextDiff(), -5);
  ASSERT_FALSE(cur.IsEnd());
  ASSERT_FALSE(cur.IsLast());
  cur.StepForward();
  ASSERT_EQ(*cur, 1);
  // There is no next diff, this is the last element.
  ASSERT_FALSE(cur.NextDiff());
  ASSERT_FALSE(cur.IsEnd());
  ASSERT_TRUE(cur.IsLast());
  cur.StepForward();
  ASSERT_EQ(cur, dv.EndCursor());
  ASSERT_TRUE(cur.IsEnd());
  ASSERT_FALSE(cur.IsLast());
  ASSERT_EQ(dv.size(), 3);
  ASSERT_EQ(dv.at(0), 4);
  ASSERT_EQ(dv.at(1), 6);
  ASSERT_EQ(dv.at(2), 1);

  dv.front() = 7;
  ASSERT_EQ(dv.size(), 3);
  ASSERT_EQ(dv.at(0), 7);
  ASSERT_EQ(dv.at(1), 9);
  ASSERT_EQ(dv.at(2), 4);
}

}  // namespace
}  // namespace deepmind::s6
