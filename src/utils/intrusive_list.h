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

// An IntrusiveList<> is a doubly-linked list where the pointer links between
// elements are part of the elements themselves.
//
// Elements in the list must derive from IntrusiveLink<> which defines the next
// and prev pointers used to link elements in the list.
//
// IntrusiveList does not allocate memory and does not own the elements in the
// list.
//
// As IntrusiveList does not own its elements and links are stored in the
// elements, IntrusiveList cannot be copied.
//
// Insertion and removal into an IntrusiveList is a constant-time operation.
//
// Note that IntrusiveList::size() runs in O(N) time.

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_INTRUSIVE_LIST_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_INTRUSIVE_LIST_H_

#include <iterator>
#include <type_traits>

namespace deepmind::s6 {

template <typename T>
class IntrusiveList;

template <typename T>
class IntrusiveLink {
 protected:
  // Only derived types and friends should be able to construct an
  // IntrusiveLink.
  IntrusiveLink() = default;

  // Prevent copy.
  IntrusiveLink(const IntrusiveLink&) = delete;
  IntrusiveLink& operator=(const IntrusiveLink&) = delete;

 private:
  T* cast_to_derived() { return static_cast<T*>(this); }
  const T* cast_to_derived() const { return static_cast<const T*>(this); }

  friend class IntrusiveList<T>;

  IntrusiveLink* prev_ = nullptr;
  IntrusiveLink* next_ = nullptr;
};

template <typename T>
class IntrusiveList {
  template <typename QualifiedType, typename QualifiedLink>
  class IteratorImpl;

 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  using link_type = IntrusiveLink<T>;
  using iterator = IteratorImpl<T, link_type>;
  using const_iterator = IteratorImpl<const T, const link_type>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  IntrusiveList() { clear(); }

  // Prevent copy and assignment.
  IntrusiveList(const IntrusiveList&) = delete;
  IntrusiveList& operator=(const IntrusiveList&) = delete;

  IntrusiveList(IntrusiveList&& src) noexcept {
    // Updates links so that:
    // src:
    //  PREV <-> SENTINEL <-> NEXT
    // becomes:
    //  SENTINEL <-> SENTINEL <-> SENTINEL
    // this:
    //  OLD_PREV <-> SENTINEL <-> OLD_NEXT
    // becomes:
    //  PREV <-> SENTINEL <-> NEXT
    //
    clear();
    if (src.empty()) {
      return;
    }
    link_type* next_link = src.sentinel_link_.next_;
    link_type* prev_link = src.sentinel_link_.prev_;

    // Connect next to the sentinel.
    sentinel_link_.next_ = next_link;
    next_link->prev_ = &sentinel_link_;

    // Connect prev to the sentinel.
    sentinel_link_.prev_ = prev_link;
    prev_link->next_ = &sentinel_link_;

    src.clear();
  }

  iterator begin() { return iterator(sentinel_link_.next_); }

  const_iterator begin() const { return const_iterator(sentinel_link_.next_); }

  iterator end() { return iterator(&sentinel_link_); }

  const_iterator end() const { return const_iterator(&sentinel_link_); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  bool empty() const { return (sentinel_link_.next_ == &sentinel_link_); }

  // `size` runs in O(N) time.
  size_type size() const { return std::distance(begin(), end()); }

  size_type max_size() const { return size_type(-1); }

  reference front() { return *begin(); }

  const_reference front() const { return *begin(); }

  reference back() { return *(--end()); }

  const_reference back() const { return *(--end()); }

  static iterator insert(iterator position, T* obj) {
    return insert_link(position.link(), obj);
  }

  void push_front(T* obj) { insert(begin(), obj); }

  void push_back(T* obj) { insert(end(), obj); }

  static iterator erase(T* obj) {
    // Removes obj so that:
    //  PREV <-> OBJ <-> NEXT
    // becomes
    //  PREV <-> NEXT
    link_type* obj_link = obj;
    link_type* const next_link = obj_link->next_;
    link_type* const prev_link = obj_link->prev_;

    // Connect prev and next links.
    next_link->prev_ = prev_link;
    prev_link->next_ = next_link;

    // Set prev and next to nullptr on obj.
    // Future attempts to remove the same element from the list will lead to a
    // nullptr dereference.
    obj_link->next_ = nullptr;
    obj_link->prev_ = nullptr;
    return iterator(next_link);
  }

  static iterator erase(iterator position) {
    return erase(position.operator->());
  }

  void pop_front() { erase(begin()); }

  void pop_back() { erase(--end()); }

  // Checks if an element is linked into a list.
  // This will return true if the element was removed from a list with `clear`.
  static bool is_linked(const T* obj) {
    return obj->link_type::next_ != nullptr;
  }

  // Clears the list by connecting the sentinel up to itself.
  // This does nothing to the links that were in the list.
  void clear() {
    sentinel_link_.next_ = &sentinel_link_;
    sentinel_link_.prev_ = &sentinel_link_;
  }

  friend void swap(IntrusiveList& lhs, IntrusiveList& rhs) {
    IntrusiveList tmp;
    tmp.splice(tmp.begin(), lhs);
    lhs.splice(lhs.begin(), rhs);
    rhs.splice(rhs.begin(), tmp);
  }

  void splice(iterator pos, IntrusiveList& src) {
    splice(pos, src, src.begin(), src.end());
  }

  void splice(iterator pos, IntrusiveList& src, iterator first, iterator last) {
    // Moves a section of one list into another so that
    //  src: PREV <-> FIRST <-> MIDDLE <-> LAST_PREV <-> LAST
    // becomes:
    //  src: PREV <-> LAST
    // and
    //  dest: PREV <-> POS <-> NEXT
    // becomes:
    //  dest: PREV <-> FIRST <-> MIDDLE <-> LAST_PREV <-> POS <-> NEXT
    if (first == last) return;

    link_type* last_prev = last.link()->prev_;
    link_type* prev = pos.link()->prev_;

    // Remove section from src.
    first.link()->prev_->next_ = last.link();
    last.link()->prev_ = first.link()->prev_;

    // Connect first and prev
    first.link()->prev_ = prev;
    prev->next_ = first.link();

    // Connect last_prev and pos
    last_prev->next_ = pos.link();
    pos.link()->prev_ = last_prev;
  }

 private:
  static iterator insert_link(link_type* next_link, T* obj) {
    // Inserts obj so that:
    //  PREV <-> NEXT
    // becomes
    //  PREV <-> OBJ <-> NEXT
    link_type* prev_link = next_link->prev_;
    link_type* obj_link = obj;

    // Connect obj link to prev
    prev_link->next_ = obj_link;
    obj_link->prev_ = prev_link;

    // Connect obj link to next
    obj_link->next_ = next_link;
    next_link->prev_ = obj_link;

    return iterator(obj_link);
  }

  template <typename QualifiedType, typename QualifiedLink>
  class IteratorImpl
      : public std::iterator<std::bidirectional_iterator_tag, QualifiedType> {
    static_assert(
        std::is_same_v<std::remove_const_t<QualifiedType>, value_type>,
        "QualifiedType should be T or const T");

   public:
    using base = std::iterator<std::bidirectional_iterator_tag, QualifiedType>;

    IteratorImpl() = default;
    IteratorImpl(const IteratorImpl& other) = default;
    IteratorImpl(QualifiedLink* link)  // NOLINT (google-explicit-constructor)
        : link_(link) {}

    // Allow construction of a const iterator from a non-const iterator.
    template <typename U, typename V>
    explicit IteratorImpl(const IteratorImpl<U, V>& x) : link_(x.link_) {
      static_assert(std::is_same_v<std::remove_const_t<U>, value_type>);
      static_assert(std::is_same_v<std::remove_const_t<V>, link_type>);
    }

    // Allow comparison between const iterators and non-const iterators.
    template <typename U, typename V>
    bool operator==(const IteratorImpl<U, V>& x) const {
      static_assert(std::is_same_v<std::remove_const_t<U>, value_type>);
      static_assert(std::is_same_v<std::remove_const_t<V>, link_type>);
      return link_ == x.link_;
    }

    // Allow comparison between const iterators and non-const iterators.
    template <typename U, typename V>
    bool operator!=(const IteratorImpl<U, V>& x) const {
      static_assert(std::is_same_v<std::remove_const_t<U>, value_type>);
      static_assert(std::is_same_v<std::remove_const_t<V>, link_type>);
      return link_ != x.link_;
    }

    typename base::reference operator*() const { return *operator->(); }

    typename base::pointer operator->() const {
      return link_->cast_to_derived();
    }

    QualifiedLink* link() const { return link_; }

    IteratorImpl& operator++() {
      link_ = link_->next_;
      return *this;
    }

    IteratorImpl operator++(int /*unused*/) {
      IteratorImpl tmp = *this;
      ++*this;
      return tmp;
    }

    IteratorImpl& operator--() {
      link_ = link_->prev_;
      return *this;
    }

    IteratorImpl operator--(int /*unused*/) {
      IteratorImpl tmp = *this;
      --*this;
      return tmp;
    }

   private:
    // Iterators can acccess other iterators directly.
    template <typename U, typename V>
    friend class IteratorImpl;

    QualifiedLink* link_ = nullptr;
  };

  link_type sentinel_link_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_INTRUSIVE_LIST_H_
