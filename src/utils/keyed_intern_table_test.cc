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

#include "utils/keyed_intern_table.h"

#include <cstdint>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {

TEST(KeyedInternTable, DefaultConstructedTableHasZeroSize) {
  KeyedInternTable<uint32_t> table;
  EXPECT_EQ(table.size(), 0);
}

TEST(KeyedInternTable, InsertString) {
  KeyedInternTable<uint32_t> table;

  // Insert a new string.
  {
    auto [key, value] = table.Insert("Test");
    EXPECT_EQ(key, 0);
    EXPECT_STREQ(value, "Test");
    EXPECT_EQ(table.size(), 1);
  }

  // Insert another new string.
  {
    auto [key, value] = table.Insert("Hello");
    EXPECT_EQ(key, 1);
    EXPECT_STREQ(value, "Hello");
    EXPECT_EQ(table.size(), 2);
  }

  // Insert an existing string.
  {
    auto [key, value] = table.Insert("Test");
    EXPECT_EQ(key, 0);
    EXPECT_STREQ(value, "Test");
    EXPECT_EQ(table.size(), 2);
  }
}

TEST(KeyedInternTable, LookupKey) {
  // Create table with a single entry.
  KeyedInternTable<uint32_t> table;
  table.Insert("Test");

  // Lookup a non-inserted key.
  EXPECT_EQ(table.ToKey("Not in table"), std::nullopt);

  // Lookup an inserted key.
  EXPECT_EQ(table.ToKey("Test"), std::optional(0));
}

}  // namespace
}  // namespace deepmind::s6
