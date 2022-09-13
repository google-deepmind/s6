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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CONSTANTS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CONSTANTS_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer_util.h"

// This file provide some constant folding as patterns. For some of them it is
// somewhat redundant with the nullconst analysis but they can be applied as
// clean-up at any stage of the optimization process so they are still useful.
// Some other, like cleaning up BrInst or DeoptimzizeIfSafepoint are critical
// and relied upon by the nullconst optimization

namespace deepmind::s6 {

// Replaces "cmp i64 (constant), (constant)" by a constant.
class ConstantFoldCompareInstPattern
    : public PatternT<ConstantFoldCompareInstPattern, CompareInst> {
 public:
  static absl::Status Apply(CompareInst* cmp, Rewriter& rewriter);
};

// Replaces (br (constant)) -> (jmp)
class ConstantFoldBrInstPattern
    : public PatternT<ConstantFoldBrInstPattern, BrInst> {
 public:
  static absl::Status Apply(BrInst* br, Rewriter& rewriter);
};

// Replaces (unbox (constant)) -> (constant).
//
// Also fixes up any `overflowed?` that immediately follows the unbox.
class ConstantFoldUnboxPattern
    : public PatternT<ConstantFoldUnboxPattern, UnboxInst> {
 public:
  static absl::Status Apply(UnboxInst* unbox, Rewriter& rewriter);
};

// Delete (deoptimize_if_safepoint (constant)) if possible.
class ConstantFoldDeoptimizeIfSafepointInstPattern
    : public PatternT<ConstantFoldDeoptimizeIfSafepointInstPattern,
                      DeoptimizeIfSafepointInst> {
 public:
  static absl::Status Apply(DeoptimizeIfSafepointInst* deopt,
                            Rewriter& rewriter);
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CONSTANTS_H_
