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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_DIFFS_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_DIFFS_H_

#include "absl/container/flat_hash_map.h"
#include "utils/logging.h"

// This file provides utilities to manipulate the concept of diffs as in the
// difference between too similar datastructures. The main component of this
// file is DiffVec a class to represent a list of element by storing the
// differences between one element and the next.

namespace deepmind::s6 {

// Applies a diff to a map.
// Returns the reverse Diff that does the opposite operation
//
// A diff of maps is stored in a linear container over pairs constituted of a
// key and on optional value. If a value is present, it means that the key must
// now point to this value whether it existed before or not. If a value is not
// present (nullopt), it means that this key must be deleted
template <typename Diff, typename Key, typename Value>
Diff ApplyMapDiff(absl::flat_hash_map<Key, Value>& map, const Diff& diff) {
  Diff reverse;
  for (auto [key, value_opt] : diff) {
    if (auto it = map.find(key); it != map.end()) {
      reverse.push_back(*it);
    } else {
      reverse.push_back({key, absl::nullopt});
    }
    if (value_opt) {
      map.insert_or_assign(key, *value_opt);
    } else {
      map.erase(key);
    }
  }
  return reverse;
}

// This struct trait provides the definition of the DiffVec trait.
// A DiffVec trait defines how an object should be diffable.
// A trait over a type T provide a second type Diff and a function
// Apply(T&, const Diff&) which applies the Diff by mutating the T&.
// The Diff type must provide an `empty()` method such that an empty Diff has no
// effect.
//
// This definition is the default implementation of the trait. To use it,
// T must define a dependent type T::Diff and a method Apply(const Diff&).
template <typename T>
struct DiffVecTraits {
  using Diff = typename T::Diff;
  static void Apply(T& t, const Diff& d) { t.Apply(d); }
};

// This class stores a vector of values by storing the original value and diffs
// between successive pairs of values.
//
// This vector behaves externaly as a container of T (which is value_type).
// However internally, it uses an initial element and a series of diffs. The
// value of element n is the initial element on which diff 0 to n-1 excluded
// have been applied. See the `at` method for more information. The DiffVec can
// contain 0 elements in which case an initial element is not present.
//
// The behavior of diffing elements of type T is provided by the `Traits`
// argument which must be a valid DiffVecTrait. See the documentation of
// DiffVecTrait. If T satisfies the requirements for the default implementation
// of DiffVecTrait, then DiffVec can be instantiated with just a T.
//
// Is is not possible to provide a light-weight iterator on such a
// data-structure therefore this class provides a Cursor which behaves exactly
// like an iterator except that copying it is as expensive as copying a T. The
// cursor only provides an immutable view on the container.
//
// DiffVec may be used to store a revision history of an object. There is an
// initial element, and then a list of modification to that element that
// represent an history of modifications on that element.
template <typename T, typename Traits = DiffVecTraits<T>>
class DiffVec {
 public:
  using value_type = T;
  using Diff = typename Traits::Diff;

  // The cursor class behaves like an iterator except that it is heavyweight as
  // it contains an object of type T.
  //
  // Incrementing the Cursor will mutably apply the next diff on the contained
  // element and do the bookkeeping to logically point the next element.
  //
  // The Cursor can also point to the past-the-end slot in which case
  // dereferencing it is invalid simlarly to iterators.
  class Cursor {
   public:
    // Returns the contained element.
    const T& operator*() const& {
      S6_DCHECK(current_.has_value());
      return *current_;
    }

    // Moves out the contained element.
    T&& operator*() && {
      S6_DCHECK(current_.has_value());
      return std::move(*current_);
    }

    // Accesses a method or attribute of the contained element.
    const T* operator->() const {
      S6_DCHECK(current_.has_value());
      return &*current_;
    }

    // Increments the Cursor to point to the next element.
    void StepForward() {
      if (next_diff_ == end_) {
        current_ = absl::nullopt;
        return;
      }
      Traits::Apply(*current_, *next_diff_);
      ++next_diff_;
    }

    // Checks that two cursor point to the same element
    bool operator==(const Cursor& oth) const {
      return next_diff_ == oth.next_diff_ &&
             current_.has_value() == oth.current_.has_value();
    }
    bool operator!=(const Cursor& oth) const { return !(*this == oth); }

    // Returns the diff that is going to be applied on the next incrementation.
    // Returns nullptr if the Cursor is pointing to the last element.
    // UB if called on the past-the-end Cursor.
    const Diff* NextDiff() const {
      if (next_diff_ == end_) return nullptr;
      return &*next_diff_;
    }

    // Returns whether this is the past-the-end Cursor.
    bool IsEnd() const { return !current_; }

    // Returns whether this is a Cursor to the last element.
    bool IsLast() const { return current_.has_value() && next_diff_ == end_; }

   private:
    friend DiffVec;
    // A valid Cursor is represented with current_ being set to the
    // dereferenceable value and next_diff_ an iterator into diff_ toward the
    // next diff.
    // If next_diff_ == end_ and current_ is set, the Cursor points to the
    // last element of the DiffVec.
    // To represent the past-the-end Cursor, current_ is set to nullopt and
    // next_diff_ stays equal to end_.
    absl::optional<T> current_;
    typename std::vector<Diff>::const_iterator next_diff_;
    typename std::vector<Diff>::const_iterator end_;
  };

  // Returns a cursor to the Initial element of the DiffVec. This performs a
  // heavy copy of the element. Thus this should not be used to do
  // `*vec.BeginCursor()`. Use `front` instead.
  Cursor BeginCursor() const {
    Cursor res;
    res.current_ = initial_;  // heavy copy here.
    res.next_diff_ = diffs_.begin();
    res.end_ = diffs_.end();
    return res;
  }

  // Returns a past-the-end cursor. This is cheap to build.
  Cursor EndCursor() const {
    Cursor res;
    res.next_diff_ = res.end_ = diffs_.end();
    return res;
  }

  // Returns the element at `index`.
  // This method is very slow and is mostly used for self documenting code.
  T at(size_t index) const {
    S6_CHECK(initial_.has_value());
    S6_CHECK_LE(index, diffs_.size());
    T result = *initial_;
    for (const Diff& diff : absl::MakeSpan(diffs_).subspan(0, index)) {
      Traits::Apply(result, diff);
    }
    return result;
  }

  // Returns the initial element. Beware that mutating the initial element
  // invalidates all existing Cursors. This method cannot be used to set an
  // initial element if the current size is 0. Use `set_front` in that case.
  T& front() { return *initial_; }
  const T& front() const { return *initial_; }

  // Clears the DiffVec including removing the initial element.
  void clear() {
    diffs_.clear();
    initial_ = absl::nullopt;
  }

  // Gets the size of the DiffVec.
  size_t size() const {
    if (!initial_) return 0;
    return diffs_.size() + 1;
  }
  bool empty() const { return !initial_; }

  // Adds a new diff at the end of the DiffVec. Invalidates all Cursors.
  void push_back(const Diff& d) {
    S6_DCHECK(initial_.has_value());
    diffs_.push_back(d);
  }
  void push_back(Diff&& d) {
    S6_DCHECK(initial_.has_value());
    diffs_.push_back(std::move(d));
  }

  // Sets the initial element. Invalidates all Cursors.
  void set_front(const T& t) { initial_.emplace(t); }
  void set_front(T&& t) { initial_.emplace(std::move(t)); }

  // push_back a diff but maintains the position of a specified Cursor.
  // Invalidates all other Cursors.
  void push_back_maintain(const Diff& d, Cursor& cur) {
    size_t pos = cur.next_diff_ - diffs_.begin();
    push_back(d);
    cur.next_diff_ = diffs_.begin() + pos;
    cur.end_ = diffs_.end();
  }
  void push_back_maintain(Diff&& d, Cursor& cur) {
    size_t pos = cur.next_diff_ - diffs_.begin();
    push_back(std::move(d));
    cur.next_diff_ = diffs_.begin() + pos;
    cur.end_ = diffs_.end();
  }

  // Resizes the DiffVec to `size` element. Only Downsizing is supported.
  // Reduces the container to only its first `size` elements.
  // Invalidates all Cursors.
  void resize(size_t size) {
    S6_CHECK_LE(size, this->size());
    if (size == 0) return clear();
    diffs_.resize(size - 1);
  }

  // Finds the first element satisfying a predicate and returns a Cursor to it.
  template <typename Pred>
  Cursor Find(Pred&& pred) const {
    Cursor cur = BeginCursor();
    while (!cur.IsEnd()) {
      if (pred(*cur)) return cur;
      cur.StepForward();
    }
    return cur;  // Here cur is EndCursor().
  }

  DiffVec() {}
  explicit DiffVec(const T& t) : initial_(t) {}
  explicit DiffVec(T&& t) : initial_(std::move(t)) {}

 private:
  // CLASS INVARIANT: If diffs_ is not empty, then initial_ must be filled.
  absl::optional<T> initial_;
  std::vector<Diff> diffs_;
};

// This class allow to port the Diffability of a type Element to the diffability
// of a pair-like struct of a Tag and an Element. The tag is represented in the
// diff by just being the next value to replace the old one.
//
// This class provide the correct interface to be usable with the default
// implementation of DiffVecTrait.
template <typename Tag, typename Element,
          typename Traits = DiffVecTraits<Element>>
struct TaggedDiffable {
  Tag tag;
  Element elem;

  using EDiff = typename Traits::Diff;

  struct Diff {
    Tag tag;
    EDiff diff;

    EDiff& operator*() & { return diff; }
    const EDiff& operator*() const& { return diff; }
    EDiff&& operator*() && { return std::move(diff); }
    EDiff* operator->() { return &diff; }
    const EDiff* operator->() const { return &diff; }
  };

  void Apply(const Diff& d) {
    tag = d.tag;
    Traits::Apply(elem, d.diff);
  }

  Element& operator*() & { return elem; }
  const Element& operator*() const& { return elem; }
  Element&& operator*() && { return std::move(elem); }
  Element* operator->() { return &elem; }
  const Element* operator->() const { return &elem; }
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_DIFFS_H_
