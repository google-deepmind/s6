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

#ifndef THIRD_PARTY_DEEPMIND_S6_ARITHMETIC_H_
#define THIRD_PARTY_DEEPMIND_S6_ARITHMETIC_H_

#include <cstdint>
#include <limits>

#include "absl/numeric/int128.h"
#include "utils/logging.h"

namespace deepmind::s6 {

namespace arithmetic {

struct Result {
  int64_t result;
  bool overflowed = false;
  Result(int64_t v) : result(v) {}  // NOLINT(google-explicit-constructor)
  Result() {}
  static Result Overflowed(int64_t v) {
    Result result = v;
    result.overflowed = true;
    return result;
  }
};

Result ResultFromUint128(absl::uint128 from);

inline Result Add(int64_t lhs, int64_t rhs) {
  absl::uint128 big_lhs = lhs;
  absl::uint128 big_rhs = rhs;
  return ResultFromUint128(big_lhs + big_rhs);
}

inline Result Subtract(int64_t lhs, int64_t rhs) {
  absl::uint128 big_lhs = lhs;
  absl::uint128 big_rhs = rhs;
  return ResultFromUint128(big_lhs - big_rhs);
}

inline Result Negate(int64_t op) {
  return ResultFromUint128(-absl::uint128(op));
}

inline Result Multiply(int64_t lhs, int64_t rhs) {
  absl::uint128 big_lhs = lhs;
  absl::uint128 big_rhs = rhs;
  return ResultFromUint128(big_lhs * big_rhs);
}

// This is a round-down division which is different from the normal C++/x86
// semantics. For example: -5 / 3 = -2
inline Result Divide(int64_t lhs, int64_t rhs) {
  S6_CHECK(rhs != 0) << "Dividing by 0 is UB";
  bool different_sign = (lhs < 0) ^ (rhs < 0);

  if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()) {
    return Result::Overflowed(lhs);
  }
  return lhs % rhs == 0 ? lhs / rhs : lhs / rhs - different_sign;
}

// This is a round-down remainder which is different from the normal C++/x86
// semantics. For example: -5 % 3 = 1. The remainder will always have the
// same sign as the divisor.
inline Result Remainder(int64_t lhs, int64_t rhs) {
  S6_CHECK(rhs != 0) << "Dividing by 0 is UB";
  bool different_sign = (lhs < 0) ^ (rhs < 0);

  if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()) {
    return 0;
  }

  int64_t result = lhs % rhs;
  if (result) result += different_sign * rhs;
  return result;
}

// For ShiftLeft, the result is undefined if the overflowed flag is raised.
// Shifting by a negative amount is UB.
inline Result ShiftLeft(int64_t lhs, int64_t rhs) {
  S6_CHECK(rhs >= 0) << "Shifting by a negative value is UB";
  Result result;
  if (rhs >= 64) {
    if (lhs == 0) {
      return 0;
    } else {
      return Result::Overflowed(0);
    }
  }

  absl::uint128 big_lhs = lhs;

  big_lhs << static_cast<int>(rhs);
  result.result = absl::Uint128Low64(big_lhs);
  if (lhs >= 0) {
    result.overflowed =
        !(absl::Uint128High64(big_lhs) == 0 && result.result >= 0);
  } else {
    result.overflowed =
        !(absl::Uint128High64(big_lhs) == -1LL && result.result < 0);
  }
  return result;
}

// ShiftRight uses the same API as the others but can't overflow.
// Shifting by a negative amount is UB.
inline Result ShiftRight(int64_t lhs, int64_t rhs) {
  static_assert(
      -1LL >> 63 == -1,
      "ShiftRight will not work with this compiler, because it performs right "
      "shifts in an unexpected manner");

  S6_CHECK(rhs >= 0) << "Shifting by a negative value is UB";

  if (rhs >= 64) {
    return lhs >= 0 ? 0 : -1;
  }
  return lhs >> rhs;
}

}  // namespace arithmetic

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_ARITHMETIC_H_
