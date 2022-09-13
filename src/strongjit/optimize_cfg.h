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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CFG_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CFG_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer_util.h"

namespace deepmind::s6 {

// Optimizes JmpInsts to targets with a single predecessor by removing the
// JmpInst and splicing the target.
class EliminateTrivialJumpsPattern
    : public PatternT<EliminateTrivialJumpsPattern, JmpInst> {
 public:
  static absl::Status Apply(JmpInst* jmp, Rewriter& rewriter);
};

// Identifies BlockArguments that are unused and removes them.
class EliminateUnusedBlockArgumentsPattern
    : public PatternT<EliminateUnusedBlockArgumentsPattern, Block> {
 public:
  static absl::Status Apply(Block* block, Rewriter& rewriter);
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CFG_H_
