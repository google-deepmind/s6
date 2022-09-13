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

#include "tuple_util.h"

#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utils/status_macros.h"

namespace deepmind::s6::tuple {
namespace {

struct MoveOnly {
  MoveOnly() = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  explicit MoveOnly(int i) : i(i) {}
  int i;

  bool operator==(const MoveOnly& other) const { return i == other.i; }
  bool operator!=(const MoveOnly& other) const { return i != other.i; }
};

TEST(TupleTransformTest, IntegerTest) {
  ASSERT_EQ(transform([](int i) { return i + 3; }, std::tuple(1, 2, 3)),
            std::tuple(4, 5, 6));
}

TEST(TupleTransformTest, OneIntegerTest) {
  ASSERT_EQ(transform([](int i) { return i + 3; }, std::tuple(1)),
            std::tuple(4));
}

TEST(TupleTransformTest, MultiTypeTest) {
  ASSERT_EQ(transform([](auto i) { return i + 3; }, std::tuple(1, 2.0)),
            std::tuple(4, 5.0));
}

TEST(TupleTransformTest, PreserveReferenceTest) {
  auto t = std::tuple(4, 5);
  auto& [t0, t1] = t;
  auto tref = transform([](auto& i) -> auto& { return i; }, t);
  EXPECT_EQ(&t0, &std::get<0>(tref));
  EXPECT_EQ(&t1, &std::get<1>(tref));
}

TEST(TupleTransformTest, PreserveMoveTest) {
  // Compilation test.
  auto t = std::tuple(MoveOnly{}, MoveOnly{});
  transform(
      [](MoveOnly&& mo) { return 0; },
      transform([](MoveOnly&& mo) { return std::move(mo); }, std::move(t)));
  auto t2 = std::tuple(MoveOnly{}, MoveOnly{});
  transform([](MoveOnly&& mo) { return 0; },
            transform([](MoveOnly&& mo) -> MoveOnly&& { return std::move(mo); },
                      std::move(t2)));
}

TEST(TupleForEachTest, StringTest) {
  std::string s;
  for_each([&](auto t) { absl::StrAppend(&s, t, " "); },
           std::tuple(1, 3.2, "cat"));

  ASSERT_EQ(s, "1 3.2 cat ");
}

TEST(TupleForEachTest, PreserveMoveTest) {
  auto t = std::tuple(MoveOnly{1}, MoveOnly{2});
  std::vector<MoveOnly> v;
  for_each([&](MoveOnly&& m) { v.push_back(std::move(m)); }, std::move(t));
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0].i, 1);
  ASSERT_EQ(v[1].i, 2);
}

TEST(TupleForEachTest, MutationTest) {
  auto t = std::tuple(1, 2.0, std::string("3"));
  for_each([&](auto& x) { x += 42; }, t);
  ASSERT_EQ(t, std::tuple(43, 44.0, std::string("3*")));
}

TEST(TupleForwardTest, LValueTest) {
  int j = 2;
  auto t = std::tuple<int, int&, MoveOnly>(1, j, MoveOnly{});
  auto t2 = forward(t);
  std::get<1>(t2) = 7;
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(t2)>, int&>);
  EXPECT_EQ(j, 7);
  EXPECT_EQ(&std::get<0>(t), &std::get<0>(t2));
  EXPECT_EQ(&std::get<1>(t), &j);
  EXPECT_EQ(&std::get<2>(t), &std::get<2>(t2));
}

TEST(TupleForwardTest, ConstLValueTest) {
  int j = 2;
  auto t = std::tuple<int, int&, MoveOnly>(1, j, MoveOnly{});
  const auto& tref = t;
  auto t2 = forward(tref);
  std::get<1>(t2) = 7;
  static_assert(
      std::is_same_v<std::tuple_element_t<0, decltype(t2)>, const int&>);
  EXPECT_EQ(j, 7);
  EXPECT_EQ(&std::get<0>(t), &std::get<0>(t2));
  EXPECT_EQ(&std::get<1>(t), &j);
  EXPECT_EQ(&std::get<2>(t), &std::get<2>(t2));
}

TEST(TupleForwardTest, RValueTest) {
  int j = 2;
  auto t = std::tuple<int, int&, MoveOnly>(1, j, MoveOnly{});
  auto t2 = forward(std::move(t));
  std::get<1>(t2) = 7;
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(t2)>, int&&>);
  EXPECT_EQ(j, 7);
  EXPECT_EQ(&std::get<0>(t), &std::get<0>(t2));
  EXPECT_EQ(&std::get<1>(t), &j);
  EXPECT_EQ(&std::get<2>(t), &std::get<2>(t2));
}

TEST(TupleTransposeTest, IntegerTest) {
  std::tuple t = {std::tuple{1, 2, 3}, std::tuple{4, 5, 6}};
  std::tuple tt = {std::tuple{1, 4}, std::tuple{2, 5}, std::tuple{3, 6}};

  ASSERT_EQ(transpose(t), tt);
  ASSERT_EQ(transpose(tt), t);
}

TEST(TupleTransposeTest, OneIntegerTest) {
  std::tuple<std::tuple<int>, std::tuple<int>> t = {{1}, {4}};
  std::tuple<std::tuple<int, int>> tt = {{1, 4}};

  ASSERT_EQ(transpose(t), tt);
  ASSERT_EQ(transpose(tt), t);
}

TEST(TupleTransposeTest, IntegerTest2) {
  std::tuple t = {std::tuple{1, 2}, std::tuple{3, 4}};
  std::tuple tt = {std::tuple{1, 3}, std::tuple{2, 4}};

  ASSERT_EQ(transpose(t), tt);
  ASSERT_EQ(transpose(tt), t);
}

TEST(TupleTransposeTest, PreserveMoveTest) {
  std::tuple t = {std::tuple{MoveOnly{1}, MoveOnly{2}},
                  std::tuple{MoveOnly{3}, MoveOnly{4}}};
  std::tuple tt = {std::tuple{MoveOnly{1}, MoveOnly{3}},
                   std::tuple{MoveOnly{2}, MoveOnly{4}}};

  ASSERT_EQ(transpose(std::move(t)), tt);
}

TEST(TupleMultiTransformTest, IntegerTest) {
  std::tuple t1 = {1, 2, 3};
  std::tuple t2 = {4, 5, 6};

  ASSERT_EQ(transform(std::plus<int>{}, t1, t2), std::tuple(5, 7, 9));
}

TEST(TupleMultiTransformTest, OneIntegerTest) {
  std::tuple t1 = {1};
  std::tuple t2 = {4};

  ASSERT_EQ(transform(std::plus<int>{}, t1, t2), std::tuple(5));
}

TEST(TupleMultiTransformTest, MoveOnlyTest) {
  std::tuple t1 = {1, MoveOnly{2}, 3};
  std::tuple t2 = {4, MoveOnly{5}, 6};

  ASSERT_EQ(transform(
                [](auto x1, auto x2) {
                  if constexpr (std::is_same_v<decltype(x1), int>)
                    return MoveOnly{x1 + x2};
                  else
                    return x2.i - x1.i;
                },
                std::move(t1), std::move(t2)),
            std::tuple(MoveOnly{5}, 3, MoveOnly{9}));
}

TEST(TupleMultiForEachTest, MoveOnlyTest) {
  std::tuple t1 = {1, MoveOnly{2}, 3};
  std::tuple t2 = {4, MoveOnly{5}, 6};

  int int_count = 0;
  int mo_count = 0;

  for_each(
      [&](auto x1, auto x2) {
        if constexpr (std::is_same_v<decltype(x1), int>)
          return int_count += x2 - x1;
        else
          return mo_count += x1.i + x2.i;
      },
      std::move(t1), std::move(t2));

  ASSERT_EQ(int_count, 6);
  ASSERT_EQ(mo_count, 7);
}

TEST(TupleMultiForEachTest, MutationTest) {
  std::tuple t1 = {1, 2.0, std::string("3")};
  std::tuple t2 = {4, 5.0, std::string("6")};

  for_each(
      [&](auto& x1, auto&& x2) {
        static_assert(std::is_rvalue_reference_v<decltype(x2)>);
        x1 += std::forward<decltype(x2)>(x2);
      },
      t1, std::move(t2));
  ASSERT_EQ(t1, std::tuple(5, 7.0, std::string("36")));
}

TEST(TupleMultiForEachTest, MutationTest2) {
  std::tuple t1 = {1, 2.0, std::string("3")};
  const std::tuple t2 = {4, 5.0, std::string("6")};

  for_each(
      [&](auto& x1, auto& x2) {
        static_assert(std::is_const_v<std::remove_reference_t<decltype(x2)>>);
        x1 += x2;
      },
      t1, t2);
  ASSERT_EQ(t1, std::tuple(5, 7.0, std::string("36")));
}

TEST(TupleFlattenTest, IntegerTest) {
  std::tuple t1 = {1, 2, 3};
  std::tuple t2 = {5, 6, 7};

  ASSERT_EQ(flatten(t1, 4, t2, 8, 9), std::tuple(1, 2, 3, 4, 5, 6, 7, 8, 9));
}
TEST(TupleFlattenTest, MoveOnlyTest) {
  std::tuple t1 = {1, 2, MoveOnly{3}};
  std::tuple t2 = {5, MoveOnly{6}, 7};

  ASSERT_EQ(
      flatten(std::move(t1), 4, std::move(t2), 8, MoveOnly{9}),
      std::tuple(1, 2, MoveOnly{3}, 4, 5, MoveOnly{6}, 7, 8, MoveOnly{9}));
}

TEST(TupleUnwrapStatusOrTest, NoErrorTest) {
  auto t = std::tuple(absl::StatusOr<int>{3}, absl::StatusOr<double>{4.0});
  S6_ASSERT_OK_AND_ASSIGN(auto t2, unwrap_StatusOr(t));
  ASSERT_EQ(t2, std::tuple(3, 4.0));
}

TEST(TupleUnwrapStatusOrTest, NoErrorMoveOnlyTest) {
  auto t = std::tuple(absl::StatusOr<int>{3}, absl::StatusOr<MoveOnly>{4});
  S6_ASSERT_OK_AND_ASSIGN(auto t2, unwrap_StatusOr(std::move(t)));
  ASSERT_EQ(t2, std::tuple(3, MoveOnly(4)));
}

TEST(TupleUnwrapStatusOrTest, ErrorTest) {
  auto t = std::tuple(absl::StatusOr<double>(absl::UnknownError("")),
                      absl::StatusOr<int>{3});
  ASSERT_EQ(unwrap_StatusOr(t).status(), absl::UnknownError(""));
}

TEST(TupleUnwrapStatusOrTest, ErrorMoveOnlyTest) {
  auto t = std::tuple(absl::StatusOr<double>(absl::UnknownError("")),
                      absl::StatusOr<MoveOnly>{3});
  ASSERT_EQ(unwrap_StatusOr(std::move(t)).status(), absl::UnknownError(""));
}

TEST(TupleTransformStatusOrTest, BasicTest) {
  auto t = std::tuple(2, 3);
  S6_ASSERT_OK_AND_ASSIGN(
      auto t1, transform_StatusOr(
                   [](auto x) { return absl::StatusOr<decltype(x)>(x); }, t));
  ASSERT_EQ(t, t1);
}

}  // namespace
}  // namespace deepmind::s6::tuple
