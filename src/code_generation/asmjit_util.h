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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_ASMJIT_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_ASMJIT_UTIL_H_

#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "asmjit/asmjit.h"
#include "utils/status_macros.h"

#define RETURN_IF_ASMJIT_ERROR(x) S6_RETURN_IF_ERROR(AsmjitErrorToStatus(x));

// Converts an asmjit::Error to an absl::Status.
inline absl::Status AsmjitErrorToStatus(asmjit::Error error) {
  if (error == asmjit::kErrorOk) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(
      "error in asmjit: ", asmjit::DebugUtils::errorAsString(error)));
}

namespace asmjit {
// Defines operator<< on std::ostream for asmjit Operands. This allows macros
// such as EXPECT_EQ, S6_RET_CHECK_EQ etc to print operands.
inline std::ostream& operator<<(std::ostream& os, const Operand& operand) {
  ::asmjit::String s;
  ::asmjit::Logging::formatOperand(s, /*flags=*/0, /*emitter=*/nullptr,
                                   ::asmjit::ArchInfo::kIdX64, operand);
  os << std::string(s.data(), s.size());
  return os;
}

namespace x86 {
inline bool operator<(const Gp& a, const Gp& b) { return a.id() < b.id(); }
}  // namespace x86

}  // namespace asmjit

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_ASMJIT_UTIL_H_
