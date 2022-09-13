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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_CALLEE_ADDRESS_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_CALLEE_ADDRESS_H_

#include "absl/status/statusor.h"
#include "strongjit/callees.h"

namespace deepmind::s6 {

// Attempts to obtain the address of a Callee. This will return
// FAILED_PRECONDITION if the symbol is not linked in. It will never return
// nullptr.
absl::StatusOr<void*> GetCalleeSymbolAddress(Callee callee);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_CALLEE_ADDRESS_H_
