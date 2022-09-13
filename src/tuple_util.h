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

#ifndef THIRD_PARTY_DEEPMIND_S6_TUPLE_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_TUPLE_UTIL_H_

#include <stddef.h>

#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core_util.h"

// A tuple manipulation library used by S6

// TODO: This library is standalone and has nothing to do with S6: It
// would be nice to open source it.

namespace deepmind::s6::tuple {

// Predicate type that evaluates to true if T is a std::tuple potentially with
// a const or a reference attached to it.
template <typename>
struct is_tuple : std::false_type {};
template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};
template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;
template <typename T>
struct is_tuple<const T> : std::bool_constant<is_tuple_v<T>> {};
template <typename T>
struct is_tuple<T&> : std::bool_constant<is_tuple_v<T>> {};
template <typename T>
struct is_tuple<T&&> : std::bool_constant<is_tuple_v<T>> {};

// Applies the function f on all the members of a tuple, and returns the tuple
// of the results.
//
// transform(f, {1, 2.0, "a"}) returns the tuple {f(1), f(2.0), f("a")}.
//
// This function preserves the exact return type of f, so if f(int) -> double&
// then transform(f, std::tuple<int>{...}) will be of type std::tuple<double&>.
//
// If f has side-effects, the order in which f is called is unspecified.
template <typename F, typename Tuple>
auto transform(F&& f, Tuple&& t) {
  static_assert(is_tuple_v<Tuple>);
  return std::apply(
      [&](auto&&... x) {
        static_assert((std::is_invocable_v<F, decltype(x)> && ...));
        // I'm not letting tuple template argument deduction run, because if f
        // returns a T&, I want the tuple to contain a T& and not a T
        return std::tuple<std::invoke_result_t<F, decltype(x)>...>(
            std::invoke(std::forward<F>(f), std::forward<decltype(x)>(x))...);
      },
      std::forward<Tuple>(t));
}

// Applies the function f in order on all the members of a tuple.
//
// for_each(f, {1, 2.0, "a"}) will call f(1) then f(2.0) then f("a");
template <typename F, typename Tuple>
void for_each(F&& f, Tuple&& t) {
  static_assert(is_tuple_v<Tuple>);
  std::apply(
      [&](auto&&... x) {
        (std::invoke(std::forward<F>(f), std::forward<decltype(x)>(x)), ...);
      },
      std::forward<Tuple>(t));
}

// Forwards a std::tuple by moving the ref const qualifier inside of the tuple:
// For example:
//  - std::tuple<int, double>& becomes std::tuple<int&, double&>.
//  - const std::tuple<int, double&>& becomes
//          std::tuple<const int&, const double&>
//  - std::tuple<int, double&>&& becomes std::tuple<int&&, double&>.
template <typename Tuple>
auto forward(Tuple&& tuple) {
  static_assert(is_tuple_v<Tuple>);
  return transform(
      [](auto&& x) -> decltype(x) { return std::forward<decltype(x)>(x); },
      std::forward<Tuple>(tuple));
}

// Given a tuple of tuples, gets the element I of each tuple in the main tuple.
//
// multi_get<1>({{1, 2, 3}, {4, 5, 6}}) returns the tuple {2, 5}.
template <std::size_t I, typename Tuple>
auto multi_get(Tuple&& t) {
  static_assert(is_tuple_v<Tuple>);
  return transform(
      [](auto&& x)
          -> std::tuple_element_t<I, std::remove_reference_t<decltype(x)>> {
        return std::get<I>(std::forward<decltype(x)>(x));
      },
      std::forward<Tuple>(t));
}

// Private namespace for implementation details.
namespace impl {

template <typename Tuple, size_t... Is>
auto transpose_impl(Tuple&& tuple, std::index_sequence<Is...>) {
  return std::make_tuple(multi_get<Is>(std::forward<Tuple>(tuple))...);
}

}  // namespace impl

// Transposes a tuple of tuple. This can also be seen as a zip but for tuples.
//
// All the tuples in the main tuple must have the same size otherwise there is a
// compilation error.
//
// transpose({{1, 2, 3}, {4, 5, 6}}) returns {{1, 4}, {2, 5}, {3, 6}}.
template <typename Tuple>
auto transpose(Tuple&& tuple) {
  static_assert(is_tuple_v<Tuple>);
  if constexpr (std::tuple_size_v<std::remove_reference_t<Tuple>> == 0) {
    return std::tuple<>{};
  } else {
    // Check all tuples in tuple have the same size.
    constexpr size_t size = std::tuple_size_v<std::remove_reference_t<
        std::tuple_element_t<0, std::remove_reference_t<Tuple>>>>;
    for_each(
        [size](auto&& tuple) {
          static_assert(
              std::tuple_size_v<std::remove_reference_t<decltype(tuple)>> ==
              size);
        },
        std::forward<Tuple>(tuple));
    return impl::transpose_impl(std::forward<Tuple>(tuple),
                                std::make_index_sequence<size>{});
  }
}

// Multi tuple version of transform. See the single tuple version earlier in
// the file. The function f is called as if the tuple were zipped.
//
// transform(f, {1, 2, 3}, {4, 5, 6}) returns {f(1,4), f(2,5), f(3,6)};
//
// The order in which f is called is unspecified.
template <typename F, typename Tuple, typename... Tuples>
auto transform(F&& f, Tuple&& t, Tuples&&... tuples) {
  static_assert(is_tuple_v<Tuple>);
  static_assert((is_tuple_v<Tuples> && ...));
  return transform(
      [&](auto x) -> decltype(auto) {
        static_assert(is_tuple_v<decltype(x)>);
        return std::apply(std::forward<F>(f), std::move(x));
      },
      transpose(std::make_tuple(forward(std::forward<Tuple>(t)),
                                forward(std::forward<Tuples>(tuples))...)));
}

// Multi tuple version of for_each. See the single tuple version earlier in
// the file. The function f is called as if the tuple were zipped.
//
// for_each(f, {1, 2, 3}, {4, 5, 6}) will run f(1,4), then f(2,5) then f(3,6).
template <typename F, typename Tuple, typename... Tuples>
void for_each(F&& f, Tuple&& t, Tuples&&... tuples) {
  static_assert(is_tuple_v<Tuple>);
  static_assert((is_tuple_v<Tuples> && ...));
  for_each(
      [&](auto x) { std::apply(std::forward<F>(f), std::move(x)); },
      transpose(std::make_tuple(forward(std::forward<Tuple>(t)),
                                forward(std::forward<Tuples>(tuples))...)));
}

// Converts a tuple to an array. All tuple types must be identical as per
// std::is_same. This function will not deduce a common type,
template <typename Tuple>
auto to_array(Tuple&& t) {
  return std::apply(
      [](auto&&... x) { return std::array{std::forward<decltype(x)>(x)...}; },
      std::forward<Tuple>(t));
}

// If the parameter is not a tuple, returns a singleton tuple from it.
// Otherwise does a perfect forwarding of the tuple.
template <typename T>
decltype(auto) expand_to_tuple(T&& t) {
  if constexpr (is_tuple_v<T>) {
    return std::forward<T>(t);
  } else {
    return std::tuple<T>(std::forward<T>(t));
  }
}

// Returns a flattened tuple. For example:
//
// decltype(flatten<A, std::tuple<B, C>, D>) -> std::tuple(A, B, C, D)
template <typename... Args>
auto flatten(Args&&... args) {
  return std::tuple_cat(expand_to_tuple(std::forward<Args>(args))...);
}

namespace impl {

// TODO: Replace that by a lambda in unwrap_StatusOr, when C++20
// lands. To do that we need lambdas in an unevaluated context.
struct dereference {
  template <typename T>
  auto operator()(T&& t) {
    return *std::forward<T>(t);
  }
};
}  // namespace impl

// Given a tuple of StatusOr, returns a StatusOr of a tuple of the contained
// elements.
// unwarp_statusOr(tuple<StatusOr<int>, StatusOr<double>>) returns
// StatusOr<tuple<int, double>>.
//
// If there more than one error in the original tuple, an unspecified error
// among the present error is returned.
//
// An okStatus with a tuple of contained elements in only returned if there were
// no errors in any of the argument tuple elements.
template <typename Tuple>
auto unwrap_StatusOr(Tuple&& tuple)
    -> absl::StatusOr<decltype(transform(impl::dereference{},
                                         std::forward<Tuple>(tuple)))> {
  static_assert(is_tuple_v<Tuple>);
  absl::Status error = absl::OkStatus();
  for_each(
      [&](const auto& status_or) {
        if (status_or.ok()) return;
        error = status_or.status();
      },
      tuple);
  if (error.ok()) {
    return transform(impl::dereference{}, std::forward<Tuple>(tuple));
  } else {
    return error;
  }
}

template <typename F, typename... Tuples>
auto transform_StatusOr(F&& f, Tuples&&... tuples) {
  static_assert((is_tuple_v<Tuples> && ...));
  return unwrap_StatusOr(
      transform(std::forward<F>(f), std::forward<Tuples>(tuples)...));
}

template <typename F, typename... Tuples>
absl::Status for_each_Status(F&& f, Tuples&&... tuples) {
  static_assert((is_tuple_v<Tuples> && ...));
  return FirstError(
      to_array(transform(std::forward<F>(f), std::forward<Tuples>(tuples)...)));
}

}  // namespace deepmind::s6::tuple

#endif  // THIRD_PARTY_DEEPMIND_S6_TUPLE_UTIL_H_
