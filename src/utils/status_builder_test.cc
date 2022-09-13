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

#include "utils/status_builder.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
TEST(StatusBuilder, EmptyMessage) {
  absl::Status status = StatusBuilder(absl::StatusCode::kOk);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(status.message().empty());
}

TEST(StatusBuilder, NoEmptyMessage) {
  absl::Status status = StatusBuilder(absl::StatusCode::kUnknown)
                        << "These are not the droids you're looking for. "
                        << "Move along.";
  EXPECT_EQ(status.code(), absl::StatusCode::kUnknown);
  EXPECT_EQ(status.message(),
            "These are not the droids you're looking for. Move along.");
}
}  // namespace
}  // namespace deepmind::s6
