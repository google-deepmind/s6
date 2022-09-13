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

#include "strongjit/optimize_constants.h"

#include <cstdint>

#include "strongjit/instructions.h"

namespace deepmind::s6 {

absl::Status ConstantFoldCompareInstPattern::Apply(CompareInst* cmp,
                                                   Rewriter& rewriter) {
  if (cmp->IsDoubleType()) {
    return absl::FailedPreconditionError("operand type is double");
  }

  auto lhs_or = GetValueAsConstantInt(cmp->lhs(), rewriter.code_object());
  auto rhs_or = GetValueAsConstantInt(cmp->rhs(), rewriter.code_object());
  if (!lhs_or.has_value() || !rhs_or.has_value()) {
    return absl::FailedPreconditionError("lhs or rhs is not a constant");
  }

  int64_t lhs = *lhs_or;
  int64_t rhs = *rhs_or;

  Builder builder = rewriter.CreateBuilder(cmp->GetIterator());
  rewriter.ReplaceAllUsesWith(*cmp, *builder.Bool(cmp->Evaluate(lhs, rhs)));
  rewriter.erase(*cmp);
  return absl::OkStatus();
}

absl::Status ConstantFoldBrInstPattern::Apply(BrInst* br, Rewriter& rewriter) {
  auto cond = GetValueAsConstantInt(br->condition(), rewriter.code_object());
  if (!cond.has_value()) return absl::FailedPreconditionError("not a constant");
  bool true_successor = *cond != 0;
  rewriter.ConvertBranchToJump(*br, true_successor);
  return absl::OkStatus();
}

absl::Status ConstantFoldUnboxPattern::Apply(UnboxInst* unbox,
                                             Rewriter& rewriter) {
  PyObject* boxed =
      GetValueAsConstantObject(unbox->boxed(), rewriter.code_object());
  if (!boxed) return absl::FailedPreconditionError("not a constant");

  auto unboxed = unbox->Evaluate(boxed);
  if (!unboxed.has_value()) {
    EventCounters::Instance().Add("optimizer.tautological deoptimize", 1);
    return absl::FailedPreconditionError("tautological deoptimize!");
  }

  OverflowedInst* overflowed =
      dyn_cast<OverflowedInst>(&*std::next(unbox->GetIterator()));

  Builder builder = rewriter.CreateBuilder(unbox->GetIterator());
  if (overflowed) {
    // At this point the overflowed? will always return zero because the unbox
    // succeeded.
    rewriter.ReplaceAllUsesWith(*overflowed, *builder.Int64(0));
    rewriter.erase(*overflowed);
  }

  rewriter.ReplaceAllUsesWith(*unbox, *builder.Int64(*unboxed));
  rewriter.erase(*unbox);
  return absl::OkStatus();
}

absl::Status ConstantFoldDeoptimizeIfSafepointInstPattern::Apply(
    DeoptimizeIfSafepointInst* deopt, Rewriter& rewriter) {
  auto cond = GetValueAsConstantInt(deopt->condition(), rewriter.code_object());
  if (!cond.has_value()) return absl::FailedPreconditionError("not a constant");

  bool test_result = deopt->negated() ? *cond == 0 : *cond != 0;
  if (test_result) {
    // Maybe move backward the information that this deoptimize will be taken
    // in all cases.
    EventCounters::Instance().Add("optimizer.tautological deoptimize", 1);
    return absl::FailedPreconditionError("tautological deoptimize");
  }

  rewriter.erase(*deopt);
  return absl::OkStatus();
}

}  // namespace deepmind::s6
