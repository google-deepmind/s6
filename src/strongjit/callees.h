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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CALLEES_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CALLEES_H_

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "core_util.h"

// This file manages callees and callee information. It maintains a list of
// allowed native callees for call_native instructions. It is the only file
// with callees.cc that should access the callee list in callees.inc.
//
// There is an exception for callee_address(.h/.cc), because they need to see
// the declarations of all the callees whereas this file should have as few
// dependencies as possible.

namespace deepmind::s6 {

// This enumeration is an allowlist of all the callee to use in a
// CallNativeInst.
enum class Callee : uint16_t {
#define CALLEE(symbol) k##symbol,
#define CPP_CALLEE(namespace, symbol) k##symbol,
// callees.inc undefs CALLEE and CPP_CALLEE.
#include "strongjit/callees.inc"
};

// This is a list of the name of all the callees in respective order. If
// you need the name of a specific callee, use ToString instead of this list
// directly.
inline constexpr absl::string_view kCalleeNames[] = {
#define CALLEE(symbol) #symbol,
#define CPP_CALLEE(namespace, symbol) #namespace "::" #symbol,
// callees.inc undefs CALLEE and CPP_CALLEE.
#include "strongjit/callees.inc"
};

// Returns the name of a callee.
inline absl::string_view ToString(Callee callee) {
  return kCalleeNames[static_cast<uint16_t>(callee)];
}

// This class represent static information known about a callee.
// Currently it contains:
// - Information about the possible nullness of arguments and return value.
// - How the callee interact with reference counting (does it steal references
//   or give back a reference with its return value).
//
// Other static information can be added but it must then be filled for most
// callees in callees.inc.
class CalleeInfo {
 public:
  // The static information about a specific argument of the callee.
  struct ArgInfo {
    // Whether this argument can be null. If Nullness::kNotNull is specified
    // then calling this function with nullptr as that argement is UB.
    Nullness nullness;

    // Whether this argument is stolen, that means that callee steals a
    // reference to it.
    bool stolen = false;

    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr ArgInfo(Nullness nullness) : nullness(nullness) {}
    constexpr ArgInfo(Nullness nullness, bool stolen)
        : nullness(nullness), stolen(stolen) {}
  };

  // Returns the CalleeInfo of callee. Not all callees have an attached
  // CalleeInfo, so this may return nullopt.
  static absl::optional<CalleeInfo> Get(Callee callee);

  // Returns whether the callee return value can be null.
  Nullness return_nullness() const { return return_nullness_; }

  // Returns whether the callee returns a fresh reference with its return value.
  bool return_new_ref() const { return return_new_ref_; }

  // Returns the information about argument arg.
  ArgInfo argument(int64_t arg) const {
    S6_CHECK_GE(arg, 0);
    S6_CHECK(!arguments_.empty());
    if (arg >= arguments_.size()) return arguments_.back();
    return arguments_[arg];
  }

  explicit CalleeInfo(Nullness return_nullness, bool return_new_ref,
                      std::initializer_list<ArgInfo> arguments)
      : return_nullness_(return_nullness),
        return_new_ref_(return_new_ref),
        arguments_(arguments.begin(), arguments.end()) {}
  explicit CalleeInfo(bool return_new_ref,
                      std::initializer_list<ArgInfo> arguments)
      : CalleeInfo(Nullness::kMaybeNull, return_new_ref, arguments) {}
  CalleeInfo(std::initializer_list<ArgInfo> arguments)
      : CalleeInfo(Nullness::kMaybeNull, false, arguments) {}

 private:
  Nullness return_nullness_;
  bool return_new_ref_;  // Does the callee return a new owning reference?

  // The arguments information. If any argument is beyond the end of that array,
  // then the information from the arguments_.back() is repeated.
  absl::InlinedVector<ArgInfo, 8> arguments_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_CALLEES_H_
