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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_NO_DESTRUCTOR_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_NO_DESTRUCTOR_H_

#include <utility>

namespace deepmind::s6 {

// NoDestructor<T> is a wrapper around an object of type T that stores the
// object in an internal buffer and allows access through optional-like
// const/non-const accessors.
//
// NoDestructor<T> never calls T's destructor, NoDestructor<T> objects created
// on the stack or as member variables will lead to memory/resource leaks.
//
// NoDestructor<T> is useful for on-demand construction of objects with static
// storage where destruction never happens.
// using NoDestructor<T> avoids having to use heap-storage and memory
// indirection.
//
//  const std::string& () {
//    static const std::string* x = new std::string("hello");
//    return *x;
//  }
//
// Can be replaced with:
//
//  const std::string& MyString() {
//    static const NoDestructor<std::string> x("hello");
//    return *x;
//  }
//
template <typename T>
class NoDestructor {
 public:
  template <typename... Ts>
  explicit NoDestructor(Ts&&... ts) {
    new (&buffer_) T(std::forward<Ts>(ts)...);
  }

  // Prevent copy and move.
  NoDestructor(const NoDestructor&) = delete;
  NoDestructor& operator=(const NoDestructor&) = delete;
  NoDestructor(NoDestructor&&) = delete;
  NoDestructor& operator=(NoDestructor&&) = delete;

  // Forwards move construction for T.
  // Needed for initialiser-list construction.
  explicit NoDestructor(T&& t) { new (&buffer_) T(std::move(t)); }

  // std::optional-like const accessors.
  // Never returns nullptr.
  const T* get() const { return reinterpret_cast<const T*>(&buffer_); }
  const T* operator->() const { return get(); }
  const T& operator*() const { return *get(); }

  // std::optional-like non-const accessors.
  // Never returns nullptr.
  T* get() { return reinterpret_cast<T*>(&buffer_); }
  T* operator->() { return get(); }
  T& operator*() { return *get(); }

 private:
  alignas(T) unsigned char buffer_[sizeof(T)];
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_NO_DESTRUCTOR_H_
