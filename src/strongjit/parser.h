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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_PARSER_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_PARSER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "classes/class_manager.h"
#include "strongjit/base.h"

namespace deepmind::s6 {

class OptimizationInfoEntry;

// Parses an Instruction from `str` and returns it. It will be created inside
// `f`.
absl::StatusOr<Instruction*> ParseInstruction(
    absl::string_view str, Function* f,
    const ClassManager& mgr = ClassManager::Instance());

// Parses a Function from `str` and returns it.
absl::StatusOr<Function> ParseFunction(
    absl::string_view str, const ClassManager& mgr = ClassManager::Instance());

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_PARSER_H_
