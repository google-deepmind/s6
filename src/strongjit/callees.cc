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

#include "strongjit/callees.h"

#include "core_util.h"

namespace deepmind::s6 {

absl::optional<CalleeInfo> CalleeInfo::Get(Callee callee) {
  switch (callee) {
#define CALLEE(symbol)    \
  case Callee::k##symbol: \
    return absl::nullopt;
#define CPP_CALLEE(namespace, symbol) \
  case Callee::k##symbol:             \
    return absl::nullopt;
#define CALLEE_INFO(symbol, info) \
  case Callee::k##symbol:         \
    return info;
#define CPP_CALLEE_INFO(namespace, symbol, info) \
  case Callee::k##symbol:                        \
    return info;
#define CINFO CalleeInfo
#define NEWREF true
#define PLAIN false
#define NOTNULL Nullness::kNotNull
#define MNULL Nullness::kMaybeNull
#define STOLEN ArgInfo(Nullness::kNotNull, true)
#define STOLEN_MNULL ArgInfo(Nullness::kMaybeNull, true)

// callees.inc undefs all the previous macros.
#include "strongjit/callees.inc"
  }
  S6_UNREACHABLE();
}

}  // namespace deepmind::s6
