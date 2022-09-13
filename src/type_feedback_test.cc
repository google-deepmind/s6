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

#include "type_feedback.h"

#include <cstdint>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
using testing::ElementsAre;
using testing::Eq;

TEST(ClassDistributionTest, Empty) {
  ClassDistribution cd;
  EXPECT_EQ(cd.Summarize().kind(), ClassDistributionSummary::kEmpty);
}

TEST(ClassDistributionTest, Monomorphic) {
  ClassDistribution cd;
  // Add the class "2" several times.
  cd.Add(2);
  cd.Add(2);
  cd.Add(2);
  ClassDistributionSummary summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kMonomorphic);
  EXPECT_THAT(summary.common_class_ids(), ElementsAre(2));
}

TEST(ClassDistributionTest, Polymorphic) {
  ClassDistribution cd;
  cd.Add(2);
  cd.Add(3);
  cd.Add(2);
  cd.Add(3);
  cd.Add(3);
  ClassDistributionSummary summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kPolymorphic);
  // 3 occurs more frequently than 2, so it should occur first.
  EXPECT_THAT(summary.common_class_ids(), ElementsAre(3, 2));
}

TEST(ClassDistributionTest, SkewedMegamorphic) {
  ClassDistribution cd;
  std::default_random_engine generator;
  std::uniform_int_distribution<int32_t> uniform_distribution(1, 50);
  const int64_t kNumInsertions = 1000;
  for (int64_t i = 0; i < kNumInsertions; ++i) {
    int32_t n = uniform_distribution(generator);
    if (n <= 40) n = 1;  // Make n=1 modal.
    cd.Add(n);
  }

  ClassDistributionSummary summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kSkewedMegamorphic);
  EXPECT_THAT(summary.common_class_ids()[0], Eq(1));
}

TEST(ClassDistributionTest, Megamorphic) {
  ClassDistribution cd;
  std::default_random_engine generator;
  std::uniform_int_distribution<int32_t> uniform_distribution(1, 50);
  const int64_t kNumInsertions = 1000;
  for (int64_t i = 0; i < kNumInsertions; ++i) {
    int32_t n = uniform_distribution(generator);
    cd.Add(n);
  }

  ClassDistributionSummary summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kMegamorphic);
}

TEST(ClassDistributionTest, DistributionChangesOverTime) {
  ClassDistribution cd;
  const int64_t kNumInsertions = 1000;
  // Insert kNumInsertions entries from a uniform distribution, then switch to
  // a much more skewed distribution. After a lot of insertions, the
  // distribution should switch to skewed.
  std::default_random_engine generator;
  std::uniform_int_distribution<int32_t> uniform_distribution(1, 50);
  for (int64_t i = 0; i < kNumInsertions; ++i) {
    int32_t n = uniform_distribution(generator);
    cd.Add(n);
  }
  ClassDistributionSummary summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kMegamorphic);

  // This needs to be at least twice kResetThreshold.
  const int64_t kNumBiasedInsertions = 200000;
  for (int64_t i = 0; i < kNumBiasedInsertions; ++i) {
    int32_t n = uniform_distribution(generator);
    if (n <= 40) n = 1;  // Make n = 1 modal.
    cd.Add(n);
  }
  summary = cd.Summarize();
  EXPECT_EQ(summary.kind(), ClassDistributionSummary::kSkewedMegamorphic);
  EXPECT_THAT(summary.common_class_ids()[0], Eq(1));
}

}  // namespace
}  // namespace deepmind::s6
