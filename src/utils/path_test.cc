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

#include "utils/path.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6::file {
namespace {

TEST(PathTest, JoinEmptyListIsEmpty) {  // clang-format: force newline
  EXPECT_EQ(JoinPath(), "");
}

TEST(PathTest, JoinWithSinglePath) {  // clang-format: force newline
  EXPECT_EQ(JoinPath("path"), "path");
}

TEST(PathTest, JoinWithLeadingSlashes) {
  EXPECT_EQ(JoinPath("/a", "/b"), "/a/b");
}

TEST(PathTest, JoinWithTrailingSlashes) {
  EXPECT_EQ(JoinPath("a/", "b/"), "a/b/");
}

TEST(PathTest, JoinWithLeadingAndTrailingSlashes) {
  EXPECT_EQ(JoinPath("/a/", "/b/"), "/a/b/");
}

TEST(PathTest, JoinWithNoSlashes) {
  EXPECT_EQ(JoinPath("a", "b", "c"), "a/b/c");
}

TEST(PathTest, BasenameWithNoSlashes) {  // clang-format: force newline
  EXPECT_EQ(Basename("a"), "a");
}

TEST(PathTest, BasenameWithSingleSlash) {  // clang-format: force newline
  EXPECT_EQ(Basename("a/b"), "b");
}

TEST(PathTest, BasenameWithMultipleSlashes) {  // clang-format: force newline
  EXPECT_EQ(Basename("a/b/c"), "c");
}

}  // namespace
}  // namespace deepmind::s6::file
