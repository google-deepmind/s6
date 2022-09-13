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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_RANGE_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_RANGE_H_

#include <iterator>
#include <utility>

namespace deepmind::s6 {

// A range adapter for a pair of iterators.
//
// Wraps two iterators into a range-compatible view-like object.
//
// Does not propagate const - if constructed with non-const iterators `I`
// then members of the underlying collection can be mutated through an
// IteratorRange<I> accessed through a const-access path.
template <typename T>
class IteratorRange {
 public:
  using iterator = T;
  using const_iterator = T;
  using value_type = typename std::iterator_traits<T>::value_type;

  IteratorRange() : begin_iterator_(), end_iterator_() {}
  IteratorRange(T begin_iterator, T end_iterator)
      : begin_iterator_(std::move(begin_iterator)),
        end_iterator_(std::move(end_iterator)) {}

  T begin() const { return begin_iterator_; }
  T end() const { return end_iterator_; }

 private:
  T begin_iterator_, end_iterator_;
};

// Makes a range out of two iterators, similar to std::make_pair.
template <typename T>
IteratorRange<T> MakeRange(T begin, T end) {
  return IteratorRange<T>(std::move(begin), std::move(end));
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_RANGE_H_
