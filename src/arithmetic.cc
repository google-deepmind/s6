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

#include "arithmetic.h"

#include <cstdint>

#include "absl/numeric/int128.h"

namespace deepmind::s6 {

namespace arithmetic {

static_assert(static_cast<int64_t>(absl::uint128(-1LL)) == -1LL,
              "S6 will not work with this compiler, because absl::uint128"
              "does not round trip properly");

Result ResultFromUint128(absl::uint128 from) {
  Result result;
  result.result = static_cast<int64_t>(from);
  result.overflowed = from != absl::uint128(result.result);
  return result;
}

}  // namespace arithmetic

}  // namespace deepmind::s6
