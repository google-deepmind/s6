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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_MAP_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_MAP_H_

#include <cstdint>

#include "absl/types/optional.h"
#include "strongjit/base.h"

// Like S6_VLOG(n), but prepends the recursion depth. This allows distinguishing
// log output when evaluating function calls.
#define EVLOG(n) S6_VLOG(n) << PyThreadState_GET()->recursion_depth << "> "

namespace deepmind::s6 {

using ValueNumbering = absl::flat_hash_map<const Value*, int64_t>;

// Maps a virtual register file of Value*s to 64-bit values at runtime. If given
// a RegisterAllocation object, this also simulates the physical register file.
class ValueMap {
 public:
  explicit ValueMap(const Function& f);

  // Sets a value.
  void Set(const Value* v, int64_t value);

  template <typename T>
  void Set(const Value* v, T value) {
    static_assert(sizeof(T) == sizeof(int64_t), "T must be int64_t-sized!");
    Set(v, reinterpret_cast<int64_t>(value));
  }

  // Returns the current content of a virtual register `v`.
  int64_t Get(const Value* v) const;

  template <typename T>
  T Get(const Value* v) const {
    static_assert(sizeof(T) == sizeof(int64_t), "T must be int64_t-sized!");
    return reinterpret_cast<T>(Get(v));
  }

 private:
  absl::optional<ValueNumbering> value_numbering_;
  absl::flat_hash_map<const Value*, int64_t> values_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_MAP_H_
