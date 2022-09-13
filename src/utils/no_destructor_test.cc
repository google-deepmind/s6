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

#include "utils/no_destructor.h"

#include <initializer_list>

#include "absl/container/flat_hash_map.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

struct StructWithDeletedDestructor {
  ~StructWithDeletedDestructor() = delete;
};

TEST(NoDestructorTest, DestructorNeverCalled) {
  [[maybe_unused]] NoDestructor<StructWithDeletedDestructor> t;
}

struct Data {
  int x;
  explicit Data(int x_) : x(x_) {}
  Data(std::initializer_list<int> xs) : x(0) {
    for (int value : xs) {
      x += value;
    }
  }
};

TEST(NoDestructorTest, ConstAccessors) {
  static const NoDestructor<Data> data(42);
  EXPECT_EQ((*data).x, 42);
  EXPECT_EQ(data.get()->x, 42);
  EXPECT_EQ(data->x, 42);
}

TEST(NoDestructorTest, NonConstAccessors) {
  static NoDestructor<Data> data(0);

  (*data).x = 1;
  EXPECT_EQ((*data).x, 1);

  // Split `get()` and `->` to avoid triggering
  // readability-redundant-smartptr-get
  // https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-smartptr-get.html#readability-redundant-smartptr-get
  Data* p = data.get();
  p->x = 2;
  EXPECT_EQ(data.get()->x, 2);

  data->x = 3;
  EXPECT_EQ(data->x, 3);
}

TEST(NoDestructorTest, InitialiserListConstruction) {
  static NoDestructor<absl::flat_hash_map<int, int>> m(
      {{0, 1}, {1, 2}, {2, 4}, {3, 8}});
  EXPECT_EQ(m->size(), 4);
}

TEST(NoDestructorTest, IsTriviallyDestructible) {
  EXPECT_TRUE(std::is_trivially_destructible_v<NoDestructor<std::string>>);
  EXPECT_TRUE(std::is_trivially_destructible_v<NoDestructor<int>>);
}

TEST(NoDestructorTest, ZeroInitialization) {
  static NoDestructor<int> x;
  EXPECT_EQ(*x, 0);
}

}  // namespace
}  // namespace deepmind::s6
