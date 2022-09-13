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

#ifndef THIRD_PARTY_DEEPMIND_S6_TYPE_FEEDBACK_H_
#define THIRD_PARTY_DEEPMIND_S6_TYPE_FEEDBACK_H_

#include <Python.h>

#include <array>
#include <cstdint>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "classes/class.h"
#include "classes/class_manager.h"

namespace deepmind::s6 {

// Forward-declare.
class ClassDistributionSummary;

// A datastructure used to approximate the distribution of types seen at a
// particular program point. Interesting kinds of distribution are:
//   1) Monomorphic, where only one type is seen.
//   2) Polymorphic, where two or more types are seen but the cardinality is
//        small.
//   3) Skewed megamorphic, where many types are seen but one (or two?) types
//        dominate. It may still be worthwhile optimizing for these common
//        types.
//   4) Megamorphic, where there is no clear common type.
//
// TypeDistribution aims to approximate the seen types at a program point into
// one of these four categories. It aims to do this with a very small runtime
// cost and memory cost.
//
// The implementation has N buckets, where N is small (4 here). Each bucket
// holds a Class ID and a count of the number of times that class has been seen.
// A final bucket holds the count of the number of times that any class not
// previously accounted for has been seen.
//
// When new classes are seen, they are allocated the next available bucket or
// sent to the "other" bucket.
//
// We try to account for the case where a type distribution stabilizes over time
// and a common type ends up in the "other" bucket. In this case, the "other"
// bucket would dominate all other buckets. If we detect this, we reset the
// distribution to its initial state (but we remember that we did this, because
// if the distribution is sampled after this before many data points have been
// received, it may seem falsely monomorphic).
//
// TODO: The number of classes is set quite high, at 4, because the
// only mechanism we have of speeding up polymorphic calls is inline caching,
// and the Richards benchmark has callsites with polymorphism of 4. We should
// implement dispatch tables to allow inline caching to only kick in when
// the effective polymorphism is small - 1 or 2.
//
// This class is not thread-safe.
class ClassDistribution {
 public:
  ClassDistribution()
      : bucket_class_ids_(),
        bucket_counts_(),
        other_count_(0),
        count_disparity_(0),
        pre_reset_event_count_(0),
        seen_zero_class_(false) {}

  // The number of classes we have buckets for. All other seen classes are sent
  // to the "other" bucket.
  static constexpr int64_t kNumClasses = 4;

  // Adds a datapoint to the distribution.
  void Add(const Class* cls) {
    if (cls) Add(cls->id());
  }
  void Add(int64_t class_id);

  // Summarizes the distribution as a string.
  std::string DebugString() const;

  // Snapshots the current state of the distribution into a summary.
  ClassDistributionSummary Summarize() const;

  bool empty() const { return bucket_counts_[0] == 0; }

 private:
  // Stores the class ID for each bucket.
  // Class IDs are 20 bits in size, so 32 bits is more than enough.
  std::array<int32_t, kNumClasses> bucket_class_ids_;

  // The event counts for each bucket.
  std::array<int64_t, kNumClasses> bucket_counts_;

  // The event count for the "other" bucket.
  int64_t other_count_;

  // The disparity between `other_count_` and the sum of all `bucket_counts_`.
  // If this is positive, there are more events classed as "other" than those
  // with categories. When this reaches some critical value, it may be a good
  // idea to reset the distribution.
  int64_t count_disparity_;

  // The number of events in the distribution when it was last reset. We can
  // consider that a distribution is at least as stable as it was before
  // reset if our current event is at least this.
  int64_t pre_reset_event_count_;

  // Has a class with ID 0 been added. Do not consider a distribution to be
  // stable if this is set.
  bool seen_zero_class_;
};

// A summary of a ClassDistribution at a particular point in time. Exact counter
// details are not stored; the distribution is simply categorized
// (approximately) and class IDs for non-unknown buckets are available.
class ClassDistributionSummary {
 public:
  enum Kind {
    // The summary holds no data.
    kEmpty,

    // There is only one class.
    kMonomorphic,

    // There are multiple classes, but all of them are identifiable.
    // common_class_ids() holds all seen classes.
    kPolymorphic,

    // There are multiple classes. common_class_ids() only holds a subset of the
    // seen classes. At least one class in common_class_ids() dominates all
    // other seen classes.
    kSkewedMegamorphic,

    // There are multiple classes, and there is no dominating class.
    // common_class_ids() is undefined.
    kMegamorphic
  };

  ClassDistributionSummary() = default;
  ClassDistributionSummary(
      Kind kind,
      std::array<int32_t, ClassDistribution::kNumClasses> common_class_ids,
      bool stable)
      : kind_(kind), stable_(stable) {
    for (int32_t class_id : common_class_ids) {
      if (class_id != 0) common_class_ids_.push_back(class_id);
    }
  }

  // Returns the categorical kind of this summary. All categorizations are
  // approximate.
  Kind kind() const { return kind_; }

  // Returns the non-unknown class IDs, sorted by decreasing frequency.
  absl::Span<const int32_t> common_class_ids() const {
    return common_class_ids_;
  }

  // Returns true if this summary is considered "stable". Stability is
  // incredibly approximate; all summaries are stable unless it is derived from
  // a ClassDistribution that was recently reset.
  bool stable() const { return stable_; }

  // Returns a string representation of the distribution. The optional
  // ClassManager is used to give string names of the classes.
  std::string ToString(const ClassManager* mgr) const;

  // Convenience predicates.
  bool IsMonomorphic() const { return kind() == kMonomorphic; }
  bool IsPolymorphic() const { return kind() == kPolymorphic; }
  bool IsSkewedMegamorphic() const { return kind() == kSkewedMegamorphic; }
  bool IsMegamorphic() const { return kind() == kMegamorphic; }

  // Returns the monomorphic class, or nullptr if it's not monomorphic.
  Class* MonomorphicClass(ClassManager& mgr = ClassManager::Instance()) const;

  // Returns the monomorphic Type, or nullptr if it's not monomorphic.
  PyTypeObject* MonomorphicType(
      ClassManager& mgr = ClassManager::Instance()) const {
    const Class* cls = MonomorphicClass(mgr);
    if (cls && !cls->is_globals_class()) return cls->type();
    return nullptr;
  }

 private:
  absl::InlinedVector<int32_t, ClassDistribution::kNumClasses>
      common_class_ids_;
  Kind kind_;
  bool stable_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_TYPE_FEEDBACK_H_
