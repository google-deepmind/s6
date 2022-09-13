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

#include "utils/intrusive_list.h"

#include <initializer_list>
#include <iterator>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

struct Data : deepmind::s6::IntrusiveLink<Data> {
  explicit Data(int x) : value(x) {}
  Data() = default;
  Data(const Data& other) : value(other.value) {}

  friend void swap(Data& lhs, Data& rhs) { std::swap(lhs.value, rhs.value); }

  int value = 0;
};

TEST(NewIntrusiveListTest, EmptyListIsTwoPointersInSize) {
  EXPECT_EQ(sizeof(IntrusiveList<Data>), sizeof(void*) * 2);
}

std::vector<Data> CreateData(std::initializer_list<int> ints) {
  std::vector<Data> data;
  data.reserve(ints.size());
  for (int i : ints) {
    data.emplace_back(i);
  }
  return data;
}

IntrusiveList<Data> AsList(std::vector<Data>& data) {
  IntrusiveList<Data> list;
  for (Data& d : data) {
    list.push_back(&d);
  }
  return list;
}

TEST(NewIntrusiveListTest, ClearAndEmpty) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  EXPECT_FALSE(list.empty());
  EXPECT_EQ(list.size(), 4);

  list.clear();

  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0);
}

TEST(NewIntrusiveListTest, PushPopBack) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  EXPECT_EQ(list.size(), 4);
  EXPECT_EQ(list.back().value, 3);

  list.pop_back();
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.back().value, 2);
}

TEST(NewIntrusiveListTest, PushPopFront) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  EXPECT_EQ(list.size(), 4);
  EXPECT_EQ(list.front().value, 0);

  list.pop_front();
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list.front().value, 1);
}

TEST(NewIntrusiveListTest, Insert) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  Data d(7);
  auto it = list.insert(std::next(list.begin()), &d);

  EXPECT_EQ(it, std::next(list.begin()));
  EXPECT_EQ(list.front().value, 0);
  list.pop_front();
  EXPECT_EQ(list.front().value, 7);
}

TEST(NewIntrusiveListTest, EraseIterator) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  list.erase(std::next(list.begin()));

  EXPECT_EQ(list.front().value, 0);
  list.pop_front();
  EXPECT_EQ(list.front().value, 2);
}

TEST(NewIntrusiveListTest, EraseObject) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  list.erase(&data[1]);

  EXPECT_EQ(list.front().value, 0);
  list.pop_front();
  EXPECT_EQ(list.front().value, 2);
}

TEST(NewIntrusiveListTest, Iterate) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  int i = 0;
  for (Data& d : list) {
    EXPECT_EQ(d.value, i++);
  }
}

TEST(NewIntrusiveListTest, ConstIterate) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  const IntrusiveList<Data> list = AsList(data);

  int i = 0;
  for (const Data& d : list) {
    EXPECT_EQ(d.value, i++);
  }
}

TEST(NewIntrusiveListTest, ReverseIterate) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  int i = 3;
  for (auto it = list.rbegin(); it != list.rend(); ++it) {
    EXPECT_EQ(it->value, i--);
  }
}

TEST(NewIntrusiveListTest, ConstReverseIterate) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  const IntrusiveList<Data> list = AsList(data);

  int i = 3;
  for (auto it = list.rbegin(); it != list.rend(); ++it) {
    EXPECT_EQ(it->value, i--);
  }
}

TEST(NewIntrusiveListTest, Reverse) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  int i = 3;
  std::reverse(list.begin(), list.end());
  for (Data& d : list) {
    EXPECT_EQ(d.value, i--);
  }
}

TEST(NewIntrusiveListTest, Swap) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  IntrusiveList<Data> another_list;

  using std::swap;
  swap(list, another_list);

  EXPECT_TRUE(list.empty());
  int i = 0;
  for (const Data& d : another_list) {
    EXPECT_EQ(d.value, i++);
  }
}

TEST(NewIntrusiveListTest, SwapEmptyLists) {
  IntrusiveList<Data> list;
  IntrusiveList<Data> another_list(std::move(list));

  using std::swap;
  swap(list, another_list);

  EXPECT_TRUE(list.empty());
  EXPECT_TRUE(another_list.empty());
}

TEST(NewIntrusiveListTest, Move) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  IntrusiveList<Data> another_list(std::move(list));

  EXPECT_TRUE(list.empty());  // NOLINT bugprone-use-after-move
  int i = 0;
  for (const Data& d : another_list) {
    EXPECT_EQ(d.value, i++);
  }
}

TEST(NewIntrusiveListTest, MoveFromAnEmptyList) {
  IntrusiveList<Data> list;
  IntrusiveList<Data> another_list(std::move(list));
  EXPECT_TRUE(list.empty());  // NOLINT bugprone-use-after-move
  EXPECT_TRUE(another_list.empty());
}

TEST(NewIntrusiveListTest, SpliceEntireList) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  std::vector<Data> data2 = CreateData({4, 5, 6, 7});
  IntrusiveList<Data> list2 = AsList(data2);

  list.splice(std::next(list.begin()), list2);

  EXPECT_TRUE(list2.empty());

  std::vector<int> expected_values = {0, 4, 5, 6, 7, 1, 2, 3};
  for (int i = 0, length = expected_values.size(); i < length; ++i) {
    EXPECT_EQ(std::next(list.begin(), i)->value, expected_values[i]);
  }
}

TEST(NewIntrusiveListTest, SplicePartialList) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  std::vector<Data> data2 = CreateData({4, 5, 6, 7});
  IntrusiveList<Data> list2 = AsList(data2);

  list.splice(std::next(list.begin()), list2, std::next(list2.begin(), 1),
              std::next(list2.begin(), 3));

  EXPECT_EQ(list2.size(), 2);

  std::vector<int> expected_values = {0, 5, 6, 1, 2, 3};
  for (int i = 0, length = expected_values.size(); i < length; ++i) {
    EXPECT_EQ(std::next(list.begin(), i)->value, expected_values[i]);
  }
}

TEST(NewIntrusiveListTest, SpliceEmptyList) {
  std::vector<Data> data = CreateData({0, 1, 2, 3});
  IntrusiveList<Data> list = AsList(data);

  IntrusiveList<Data> list2;

  list.splice(std::next(list.begin()), list2);

  EXPECT_TRUE(list2.empty());

  for (int i = 0, length = data.size(); i < length; ++i) {
    EXPECT_EQ(std::next(list.begin(), i)->value, data[i].value);
  }
}

}  // namespace
}  // namespace deepmind::s6
