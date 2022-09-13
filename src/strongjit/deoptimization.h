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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_DEOPTIMIZATION_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_DEOPTIMIZATION_H_

#include <random>

#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/util.h"

namespace deepmind::s6 {

// Identifies blocks that are only branched to by 'deoptimize' BrInst edges,
// and marks them as 'deoptimized'.
absl::Status MarkDeoptimizedBlocks(Function& f);

// Converts deoptimized edges in non-deoptimized blocks to deoptimize_if
// instructions. Deoptimized blocks are moved to the end of the function.
absl::Status RewriteFunctionForDeoptimization(Function& f);

// Testing only: stress tests deoptimization by inserting random deoptimization
// edges.
absl::Status StressTestByDeoptimizingRandomly(
    Function& f, std::default_random_engine& rng,
    float probability_to_deoptimize_edge = 0.1f);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_DEOPTIMIZATION_H_
