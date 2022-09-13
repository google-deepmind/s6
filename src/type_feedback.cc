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
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "utils/logging.h"

namespace deepmind::s6 {

void ClassDistribution::Add(int64_t class_id) {
  for (int64_t i = 0; i < kNumClasses; ++i) {
    if (class_id == 0) {
      seen_zero_class_ = true;
    }
    if (class_id != 0 &&
        (bucket_class_ids_[i] == 0 || bucket_class_ids_[i] == class_id)) {
      bucket_class_ids_[i] = class_id;
      ++bucket_counts_[i];

      // Bubble the bucket counts such that the most frequently occurring bucket
      // is bucket 0. This make summarizing easier and also makes this outer
      // loop quicker.
      int64_t j = i;
      while (j > 0 && bucket_counts_[j] > bucket_counts_[j - 1]) {
        std::swap(bucket_counts_[j], bucket_counts_[j - 1]);
        std::swap(bucket_class_ids_[j], bucket_class_ids_[j - 1]);
        --j;
      }

      --count_disparity_;
      return;
    }
  }

  ++other_count_;
  ++count_disparity_;

  // Arbitrary choice; how do we best benchmark this?
  static constexpr int64_t kResetThreshold = 100000;
  if (count_disparity_ > kResetThreshold) {
    pre_reset_event_count_ = absl::c_accumulate(bucket_counts_, other_count_);
    absl::c_fill(bucket_class_ids_, 0UL);
    absl::c_fill(bucket_counts_, 0UL);
    count_disparity_ = 0;
    other_count_ = 0;
  }
}

std::string ClassDistribution::DebugString() const {
  return absl::StrFormat(
      "bucket_classes: %s\n, bucket_counts: %s\n, other_count: %d\n, "
      "pre_reset_event_count: %d",
      absl::StrJoin(bucket_class_ids_, ", "),
      absl::StrJoin(bucket_counts_, ", "), other_count_,
      pre_reset_event_count_);
}

ClassDistributionSummary ClassDistribution::Summarize() const {
  int64_t event_count = absl::c_accumulate(bucket_counts_, other_count_);
  // TODO: This may be a high bar for stability. What we're trying to
  // judge here is whether there are enough events to give some indication that
  // the result is trustworthy.
  bool stable = !seen_zero_class_ &&
                event_count >= (pre_reset_event_count_ / 2);  // seen_zero_class

  ClassDistributionSummary::Kind kind;
  if (other_count_ == 0) {
    if (bucket_class_ids_[0] == 0) {
      // We had no events at all.
      kind = ClassDistributionSummary::kEmpty;
    } else if (bucket_class_ids_[1] == 0) {
      // We only used bucket 0.
      kind = ClassDistributionSummary::kMonomorphic;
    } else {
      // We only used bucketed classes.
      kind = ClassDistributionSummary::kPolymorphic;
    }
  } else {
    // We choose between megamorphic and skewed megamorphic based on whether
    // buckets[0] holds at least three quarters of events. There are other ways
    // to approximate, but this is a heuristic.
    if ((static_cast<double>(bucket_counts_[0]) /
         static_cast<double>(event_count)) > 0.75) {
      kind = ClassDistributionSummary::kSkewedMegamorphic;
    } else {
      kind = ClassDistributionSummary::kMegamorphic;
    }
  }

  return ClassDistributionSummary(kind, bucket_class_ids_, stable);
}

std::string ClassDistributionSummary::ToString(const ClassManager* mgr) const {
  auto format_class_id = [mgr](std::string* s, int32_t class_id) {
    if (mgr) {
      absl::StrAppend(s, mgr->GetClassById(class_id)->name(), "#", class_id);
    } else {
      absl::StrAppend(s, "class#", class_id);
    }
  };
  std::string s = stable_ ? "" : "UNSTABLE ";
  switch (kind_) {
    case kEmpty:
      return "empty";
    case kMonomorphic:
      absl::StrAppend(&s, "monomorphic, ");
      format_class_id(&s, common_class_ids_[0]);
      return s;
    case kPolymorphic:
      absl::StrAppend(
          &s, "polymorphic, either ",
          absl::StrJoin(common_class_ids_, " or ", format_class_id));
      return s;
    case kSkewedMegamorphic:
      absl::StrAppend(
          &s, "skewed megamorphic, commonly ",
          absl::StrJoin(common_class_ids_, " or ", format_class_id));
      return s;
    case kMegamorphic:
      absl::StrAppend(&s, "megamorphic");
      return s;
  }
  S6_UNREACHABLE();
}

Class* ClassDistributionSummary::MonomorphicClass(ClassManager& mgr) const {
  if (IsMonomorphic()) {
    return mgr.GetClassById(common_class_ids_.front());
  } else {
    return nullptr;
  }
}

}  // namespace deepmind::s6
