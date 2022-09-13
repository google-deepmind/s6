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

#include "strongjit/value_map.h"

#include <cstdint>

#include "strongjit/function.h"
#include "utils/logging.h"

namespace deepmind::s6 {
ValueMap::ValueMap(const Function& f) {
  if (S6_VLOG_IS_ON(1)) {
    // We need a value numbering if we're expected to log values.
    value_numbering_ = ComputeValueNumbering(f);
  }
}

void ValueMap::Set(const Value* v, int64_t value) {
  EVLOG(2) << absl::StrFormat("  %%%d <- %16d %#16x", value_numbering_->at(v),
                              value, value);
  values_[v] = value;
}

int64_t ValueMap::Get(const Value* v) const {
  S6_CHECK(values_.contains(v))
      << "Requested value " << value_numbering_->at(v);
  int64_t value = values_.at(v);
  EVLOG(3) << absl::StrFormat("  %%%d -> %16d %#16x", value_numbering_->at(v),
                              value, value);
  return value;
}
}  // namespace deepmind::s6
