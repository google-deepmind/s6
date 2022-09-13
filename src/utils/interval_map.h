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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_INTERVAL_MAP_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_INTERVAL_MAP_H_

#include <map>
#include <type_traits>
#include <utility>

#include "utils/logging.h"

namespace deepmind::s6 {

namespace detail {
template <typename T,
          typename = decltype(std::declval<T>() == std::declval<T>())>
constexpr bool is_equality_comparable(int) {
  return true;
}

template <typename T>
constexpr bool is_equality_comparable(long) {  // NOLINT
  return false;
}
}  // namespace detail

// Type trait that returns true if T provides operator==(T, T).
template <typename T>
constexpr bool is_equality_comparable() {
  // The detail::is_equality_comparable method takes a single argument that
  // is unambiguous in specification (int or long) but is ambiguous in argument
  // (zero could be int or long) to allow SFINAE.
  return detail::is_equality_comparable<T>(0);
}

// Maps intervals to a value. This interval map only supports non-overlapping
// intervals.
//
// Intervals are always specified as half-open: [start, end).
//
// Template arguments:
//   K: The type of the interval key. Must be EqualityComparable.
//   V: The type of the mapped value.
template <typename K, typename V>
class IntervalMap {
 public:
  // Sets [start, limit) to `value`.
  void Set(const K& start, const K& limit, V value);

  // Synonym for Set(), defined for consistency with existing API.
  void SetAndCoalesce(const K& start, const K& limit, V value) {
    Set(start, limit, value);
  }

  // Erases all data for the range [start, limit).
  void Erase(const K& start, const K& limit);

  // Removes all entries from the map.
  void Clear();

  using value_type = typename std::map<std::pair<K, K>, V>::value_type;
  using iterator = typename std::map<std::pair<K, K>, V>::iterator;
  using const_iterator = typename std::map<std::pair<K, K>, V>::const_iterator;

  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  size_t size() const { return data_.size(); }

  // Returns true if `k` is within the map and sets `*value`.
  bool Lookup(const K& k, V* v) const {
    auto it = Lookup(k);
    if (it == end()) return false;
    *v = it->second;
    return true;
  }

  // Returns true if the map contains entries that cover the entire range.
  bool CoversRange(const K& start, const K& limit) const {
    auto it = Lookup(start);
    if (it == end()) return false;
    return it->first.second >= limit;
  }

 private:
  using KeyType = std::pair<K, K>;
  using MapType = std::map<KeyType, V>;

  // Returns the interval starting at or after `k`. If an interval spans `k`, it
  // is split.
  typename MapType::iterator IntervalAfter(const K& k);

  // Attempts to coalesce two intervals. Interval `prior` will be retained and
  // `next` erased if the intervals are coalesced.
  void MaybeCoalesce(typename MapType::iterator prior,
                     typename MapType::iterator next);

  // Returns the interval containing `k`.
  const_iterator Lookup(const K& k) const;

  // Validates the integrity of the map.
  void Validate();

  std::map<KeyType, V> data_;
};

template <typename K, typename V>
void IntervalMap<K, V>::Set(const K& start, const K& limit, V value) {
  // Erase all intervals between [start, limit).
  typename MapType::iterator before_interval = IntervalAfter(start);
  typename MapType::iterator after_interval = IntervalAfter(limit);

  data_.erase(before_interval, after_interval);
  auto it = data_.insert({{start, limit}, std::move(value)}).first;

  // Coalesce adjacent values. Coalesce {it, next(it)} first as {prev(it), it}
  // may erase `it`.
  if (it != data_.end()) {
    MaybeCoalesce(it, std::next(it));
  }
  if (it != data_.begin()) {
    MaybeCoalesce(std::prev(it), it);
  }

  Validate();
}

template <typename K, typename V>
void IntervalMap<K, V>::Erase(const K& start, const K& limit) {
  typename MapType::iterator before_interval = IntervalAfter(start);
  typename MapType::iterator after_interval = IntervalAfter(limit);

  data_.erase(before_interval, after_interval);
  Validate();
}

template <typename K, typename V>
void IntervalMap<K, V>::Clear() {
  data_.clear();
}

template <typename K, typename V>
void IntervalMap<K, V>::Validate() {
#ifndef NDEBUG
  return;
#endif
  if (data_.empty()) return;
  for (auto it = std::next(data_.begin()); it != data_.end(); ++it) {
    auto prior = std::prev(it);
    // [start1, end1), [start2, end2) => start2 >= end1
    S6_CHECK_GE(it->first.first, prior->first.second);
    // [start1, end1) => end1 > start1
    S6_CHECK_GT(it->first.second, it->first.first);
  }
}

template <typename K, typename V>
typename IntervalMap<K, V>::MapType::iterator IntervalMap<K, V>::IntervalAfter(
    const K& k) {
  if (data_.empty()) return data_.end();

  auto it = data_.lower_bound({k, k});

  // `it` starts at or after `k` as a postcondition from lower_bound. If it
  // starts exactly at `k`, we are done.
  if (it != data_.end() && it->first.first == k) return it;

  // `it` starts after `k`. Check the prior interval to see if it contains `k`.
  if (it != data_.begin()) {
    auto prior_it = std::prev(it);
    if (prior_it->first.second > k) {
      S6_CHECK_LT(prior_it->first.first, k);
      it = prior_it;
    }
  }

  if (it == data_.end()) {
    // Still doesn't contain `k`, bail out.
    return it;
  }
  if (it->first.first > k) {
    // `it` starts after `k` and no prior interval contained `k`.
    return it;
  }

  // Otherwise the interval starts before `k` and we must split it. Verify we
  // have a valid interval first.
  S6_CHECK_GE(k, it->first.first);
  S6_CHECK_LT(k, it->first.second);

  auto split_after = data_.insert({{k, it->first.second}, it->second}).first;
  auto node = data_.extract(it);
  node.key().second = k;
  data_.insert(std::move(node));
  return split_after;
}

template <typename K, typename V>
typename IntervalMap<K, V>::const_iterator IntervalMap<K, V>::Lookup(
    const K& k) const {
  if (data_.empty()) return data_.end();

  auto it = data_.lower_bound({k, k});

  // `it` starts at or after `k` as a postcondition from lower_bound. If it
  // starts exactly at `k`, we are done.
  if (it != data_.end() && it->first.first == k) return it;

  // `it` starts after `k`. Check the prior interval to see if it contains `k`.
  if (it != data_.begin()) {
    auto prior_it = std::prev(it);
    if (prior_it->first.second > k) {
      S6_CHECK_LT(prior_it->first.first, k);
      return prior_it;
    }
  }

  if (it == data_.end()) {
    // Still doesn't contain `k`, bail out.
    return it;
  }
  if (it->first.first <= k && it->first.second > k) {
    return it;
  }
  // `it` starts after `k` and no prior interval contained `k`.
  return data_.end();
}

template <typename K, typename V>
void IntervalMap<K, V>::MaybeCoalesce(typename MapType::iterator prior,
                                      typename MapType::iterator next) {
  // If V is not equality-comparable, we just can't coalesce.
  if constexpr (is_equality_comparable<V>()) {
    // Are the intervals valid?
    if (prior == data_.end() || next == data_.end()) {
      return;
    }
    // Are the intervals consecutive: [a, b), [b, c) ?
    if (prior->first.second != next->first.first) {
      return;
    }

    // Are the mapped values the same?
    if (prior->second != next->second) {
      return;
    }

    // Okay!
    auto node = data_.extract(prior);
    node.key().second = next->first.second;
    data_.insert(std::move(node));
    data_.erase(next);
  }
}
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_INTERVAL_MAP_H_
