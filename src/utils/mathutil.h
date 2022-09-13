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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_MATHUTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_MATHUTIL_H_

#include <numeric>
#include <type_traits>

#include "utils/logging.h"

namespace deepmind::s6 {
// Returns the maximum integer value which is a multiple of rounding_value,
// and less than or equal to input_value.
// The input_value must be greater than or equal to zero, and the
// rounding_value must be greater than zero.
template <typename T>
static T RoundDownTo(T input_value, T rounding_value) {
  static_assert(std::is_integral_v<T>, "Operand type is not an integer");
  S6_DCHECK_GE(input_value, 0);
  S6_DCHECK_GT(rounding_value, 0);
  return (input_value / rounding_value) * rounding_value;
}

// Returns the minimum integer value which is a multiple of rounding_value,
// and greater than or equal to input_value.
// The input_value must be greater than or equal to zero, and the
// rounding_value must be greater than zero.
template <typename T>
static T RoundUpTo(T input_value, T rounding_value) {
  static_assert(std::is_integral_v<T>, "Operand type is not an integer");
  S6_DCHECK_GE(input_value, 0);
  S6_DCHECK_GT(rounding_value, 0);
  const T remainder = input_value % rounding_value;
  return (remainder == 0) ? input_value
                          : (input_value - remainder + rounding_value);
}
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_MATHUTIL_H_
