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

#include "utils/range.h"

#include <iterator>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

using ::testing::ElementsAre;

TEST(IteratorRange, ArrayMakeRange) {
  int v[] = {2, 3, 5, 7, 11, 13};
  EXPECT_THAT(MakeRange(&v[1], &v[4]), ElementsAre(3, 5, 7));
}

TEST(IteratorRange, VectorMakeRange) {
  std::vector v = {2, 3, 5, 7, 11, 13};
  EXPECT_THAT(MakeRange(std::next(v.begin()), std::prev(v.end())),
              ElementsAre(3, 5, 7, 11));
}

}  // namespace
}  // namespace deepmind::s6
