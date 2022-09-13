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

#include "runtime/callee_address.h"

#include <Python.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "classes/getsetattr.h"
#include "runtime/runtime.h"

namespace deepmind::s6 {

absl::StatusOr<void*> GetCalleeSymbolAddress(Callee callee) {
  switch (callee) {
#define CALLEE(symbol)    \
  case Callee::k##symbol: \
    return reinterpret_cast<void*>(&symbol);
#define CPP_CALLEE(namespace, symbol) \
  case Callee::k##symbol:             \
    return reinterpret_cast<void*>(&namespace ::symbol);
// callees.inc undefs CALLEE and CPP_CALLEE.
#include "strongjit/callees.inc"
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Symbol not linked: ", ToString(callee)));
  }
}

}  // namespace deepmind::s6
