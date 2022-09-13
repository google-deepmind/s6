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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_HANDLERS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_HANDLERS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "strongjit/ingestion.h"

namespace deepmind::s6 {

// Returns the Analyze function for an opcode.
absl::StatusOr<AnalyzeFunction> GetAnalyzeFunction(int32_t opcode);

// Returns the Translate function for an opcode.
absl::StatusOr<TranslateFunction> GetTranslateFunction(int32_t opcode);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INGESTION_HANDLERS_H_
