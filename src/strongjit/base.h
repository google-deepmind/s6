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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BASE_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BASE_H_

#include <array>
#include <deque>
#include <map>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "strongjit/block.h"
#include "strongjit/cursor.h"
#include "strongjit/instruction.h"
#include "strongjit/util.h"
#include "strongjit/value.h"
#include "utils/intrusive_list.h"

namespace deepmind::s6 {

class Block;
class Function;
class Instruction;
class TerminatorInst;

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_BASE_H_
