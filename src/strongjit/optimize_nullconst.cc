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

#include "strongjit/optimize_nullconst.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "core_util.h"
#include "cppitertools/imap.hpp"
#include "cppitertools/sorted.hpp"
#include "cppitertools/zip.hpp"
#include "strongjit/base.h"
#include "strongjit/block.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "strongjit/value_map.h"
#include "utils/logging.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

namespace nullconst {

////////////////////////////////////////////////////////////////////////////////
// PointInfo implementation

std::string ToString(const PointInfo& info, const ValueNumbering& numbering) {
  return absl::StrJoin(
      // Sort to get the lower value numbers first.
      iter::sorted(info.data(),
                   [&](const auto& p1, const auto& p2) {
                     return numbering.at(p1.first) < numbering.at(p2.first);
                   }),
      ", ", [&](std::string* out, const auto& p) {
        auto [v, n] = p;
        absl::StrAppend(out, "%", numbering.at(v), " is ", ToString(n));
      });
}

std::string ToString(const PointInfo::Diff& diff, const ValueNumbering& vn) {
  return absl::StrJoin(diff, ", ", [&](std::string* out, const auto& p) {
    auto [v, opt_null] = p;
    if (opt_null) {
      absl::StrAppend(out, "%", vn.at(v), " will be ", ToString(*opt_null));
    } else {
      absl::StrAppend(out, "%", vn.at(v), " will be deleted");
    }
  });
}

// Adds a new value nullness to the available information.
// REQUIRES: `!contains(v)`
void PointInfo::NewValue(const Value* v, NullConst nullness) {
  S6_CHECK(!contains(v));
  data_.insert({v, nullness});
}

// Adds possible values for `v` that are in `nullness`.
// Returns true if it has indeed added new possible values for `v` and
// false if it didn't change the possible values for `v`.
bool PointInfo::JoinValue(const Value* v, NullConst nullness) {
  if (at(v) >= nullness) return false;
  at(v).JoinEq(nullness);
  return true;
}

// Splits the universe of possibilities depending on `v` to represent a
// a condition taken on `v`.
// The current block must be passed in order not to make invalid assumptions.
// Returns two diffs, the first one is to apply in case if `v` is false and
// the second one it apply if `v` is true (not null). This is smarter than
// just checking if `v` is Null() or NotNull().
auto PointInfo::SplitOn(const Block& b, const Value* v) const
    -> std::array<Diff, 2> {
  std::array<Diff, 2> result;
  // First if we know the result of the condition, this is easy.
  if (at(v) <= NullConst::Null()) {
    result[1].push_back({v, NullConst::Impossible()});
    return result;
  }
  if (at(v) <= NullConst::NotNull()) {
    result[0].push_back({v, NullConst::Impossible()});
    return result;
  }
  // Here we don't know the result of the condition.
  // All we can for generic code it this:
  result[0].push_back({v, Meet(at(v), NullConst::Null())});
  result[1].push_back({v, Meet(at(v), NullConst::NotNull())});

  // However if the conditional instruction is an equality check we can do
  // better.
  const CompareInst* comp = dyn_cast<CompareInst>(v);
  if (!comp || !comp->IsEquality()) return result;

  // The comparaison doesn't come from the same block as the condition
  // evaluation, this is not supported yet.
  if (comp->parent() != &b) return result;

  bool isEqual = comp->comparison() == CompareInst::kEqual;
  size_t eq_index = isEqual ? 1 : 0;
  size_t neq_index = isEqual ? 0 : 1;

  NullConst left_nullness = at(comp->lhs());
  NullConst right_nullness = at(comp->rhs());
  S6_CHECK(!left_nullness.IsConstant() || !right_nullness.IsConstant())
      << "Both comparison operands are constants but the comparison result "
         "is not.";

  NullConst meet = Meet(left_nullness, right_nullness);
  S6_CHECK(meet != NullConst::Impossible())
      << "Comparison result should have been computed, no SplitOn required";

  result[eq_index].push_back({comp->rhs(), meet});
  result[eq_index].push_back({comp->lhs(), meet});
  if (left_nullness == NullConst::Null()) {
    result[neq_index].push_back(
        {comp->rhs(), Meet(right_nullness, NullConst::NotNull())});
  }
  if (right_nullness == NullConst::Null()) {
    result[neq_index].push_back(
        {comp->lhs(), Meet(left_nullness, NullConst::NotNull())});
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// GetUpdate implementation
//
// It is in this section that instruction-specific diffs are computed.

// The NullConst instruction evaluator.
class Evaluator {
 public:
  using Diff = PointInfo::Diff;

  template <typename InstrType>
  static std::optional<absl::StatusOr<Diff>> Visit(Evaluator& evaluator,
                                                   const Instruction& inst) {
    if (inst.kind() != InstrType::kKind) return {};
    if constexpr (InstrType::kProducesValue) {
      absl::StatusOr<NullConst> r = evaluator.Evaluate(cast<InstrType>(inst));
      if (r.ok()) {
        evaluator.diff_.push_back({&inst, *r});
        return std::move(evaluator.diff_);
      }
      return r.status();
    } else {
      S6_RETURN_IF_ERROR(evaluator.Evaluate(cast<InstrType>(inst)));
      return std::move(evaluator.diff_);
    }
  }

  static absl::StatusOr<Diff> Default(Evaluator&, const Instruction& inst) {
    return absl::UnimplementedError(absl::StrCat(
        FormatOrDie(inst), " has no nullconst info implementation"));
  }

  explicit Evaluator(const PointInfo& info, const PyCodeObject* code)
      : info_(info), code_(code) {}

 private:
  const PointInfo& info() const { return info_; }
  const PyCodeObject* code() const { return code_; }

  void AssumeNotNull(const Value* v) {
    NullConst nullness = info_.Nullness(v);
    if (nullness <= NullConst::NotNull()) return;
    S6_VLOG(3)
        << "Assuming " << FormatOrDie(*v)
        << " is null where it was not before. This may be a symptom of a "
           "segfault.";
    diff_.push_back({v, Meet(nullness, NullConst::NotNull())});
  }

  absl::Status Evaluate(const TerminatorInst&) {
    // Nothing to do here.
    return absl::OkStatus();
  }

  absl::StatusOr<NullConst> Evaluate(const ConstantInst& inst) {
    return NullConst::Constant(inst.value());
  }

  absl::StatusOr<NullConst> Evaluate(const CompareInst& inst) {
    NullConst nullconst_lhs = info().at(inst.lhs());
    NullConst nullconst_rhs = info().at(inst.rhs());
    if (inst.IsDoubleType()) {
      if (nullconst_lhs.IsConstant() && nullconst_rhs.IsConstant()) {
        return NullConst::Constant(inst.Evaluate(
            absl::bit_cast<double>(*nullconst_lhs.ConstantValue()),
            absl::bit_cast<double>(*nullconst_rhs.ConstantValue())));
      }
      return NullConst::Any();
    }
    // Operand type is int64_t here.
    if (nullconst_lhs.IsConstant() && nullconst_rhs.IsConstant()) {
      return NullConst::Constant(inst.Evaluate(*nullconst_lhs.ConstantValue(),
                                               *nullconst_rhs.ConstantValue()));
    }
    if (inst.comparison() == CompareInst::kEqual &&
        !IsCompatible(nullconst_lhs, nullconst_rhs)) {
      return NullConst::Constant(0);
    }
    if (inst.comparison() == CompareInst::kNotEqual &&
        !IsCompatible(nullconst_lhs, nullconst_rhs)) {
      return NullConst::Constant(1);
    }
    return NullConst::Any();
  }
  absl::Status Evaluate(const RefcountInst& inst) {
    if (inst.nullness() == Nullness::kNotNull) {
      AssumeNotNull(inst.operand());
    }
    return absl::OkStatus();
  }
  absl::StatusOr<NullConst> Evaluate(const LoadInst& inst) {
    AssumeNotNull(inst.pointer());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const LoadGlobalInst&) {
    return NullConst::Any();
  }
  absl::Status Evaluate(const StoreInst& inst) {
    AssumeNotNull(inst.pointer());
    return absl::OkStatus();
  }
  absl::StatusOr<NullConst> Evaluate(const FrameVariableInst& inst) {
    // TODO: Figure out if a null pointer is possible.
    switch (inst.frame_variable_kind()) {
      case FrameVariableInst::FrameVariableKind::kConsts:
        return NullConst::Constant(reinterpret_cast<int64_t>(
            PyTuple_GET_ITEM(code()->co_consts, inst.index())));
      case FrameVariableInst::FrameVariableKind::kNames:
        return NullConst::Constant(reinterpret_cast<int64_t>(
            PyTuple_GET_ITEM(code()->co_names, inst.index())));
      default:
        break;
    }
    return NullConst::Any();
  }

  template <typename SubBinaryInst,
            std::enable_if_t<std::is_base_of_v<BinaryInst, SubBinaryInst>,
                             bool> = true>
  absl::StatusOr<NullConst> Evaluate(const SubBinaryInst& inst) {
    absl::optional<int64_t> const_lhs = info().ConstantValue(inst.lhs());
    absl::optional<int64_t> const_rhs = info().ConstantValue(inst.rhs());
    if (!const_lhs || !const_rhs) return NullConst::Any();
    if (inst.IsDoubleType()) {
      return NullConst::Constant(absl::bit_cast<int64_t>(
          inst.Evaluate(absl::bit_cast<double>(*const_lhs),
                        absl::bit_cast<double>(*const_rhs))));
    }
    return NullConst::Constant(inst.Evaluate(*const_lhs, *const_rhs));
  }

  template <
      typename SubUnaryInst,
      std::enable_if_t<std::is_base_of_v<UnaryInst, SubUnaryInst>, bool> = true>
  absl::StatusOr<NullConst> Evaluate(const SubUnaryInst& inst) {
    if (auto const_val = info().ConstantValue(inst.operand())) {
      if (inst.IsDoubleType()) {
        return NullConst::Constant(absl::bit_cast<int64_t>(
            inst.Evaluate(absl::bit_cast<double>(*const_val))));
      }
      return NullConst::Constant(inst.Evaluate(*const_val));
    }
    return NullConst::Any();
  }

  absl::StatusOr<NullConst> Evaluate(const IntToFloatInst& inst) {
    if (auto val = info().ConstantValue(inst.operand())) {
      return NullConst::Constant(absl::bit_cast<int64_t>(inst.Evaluate(*val)));
    }
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const CallNativeInst&) {
    // TODO: Use CalleInfo
    return NullConst::Any();
  }

  absl::StatusOr<NullConst> Evaluate(const CallPythonInst&) {
    // TODO: Check if this guarantees that arguments are not null.
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const CallVectorcallInst& inst) {
    AssumeNotNull(inst.callee());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const CallNativeIndirectInst& inst) {
    // TODO: Use CalleInfo
    AssumeNotNull(inst.callee());
    return NullConst::Any();
  }
  absl::Status Evaluate(const BytecodeBeginInst& inst) {
    return absl::OkStatus();
  }
  absl::StatusOr<NullConst> Evaluate(const YieldValueInst& inst) {
    return NullConst::Any();
  }
  absl::Status Evaluate(const DeoptimizeIfSafepointInst& inst) {
    auto [diff_false, diff_true] =
        info().SplitOn(*inst.parent(), inst.condition());
    diff_ = inst.negated() ? diff_true : diff_false;
    return absl::OkStatus();
  }
  absl::Status Evaluate(const AdvanceProfileCounterInst&) {
    return absl::OkStatus();
  }
  template <typename SubProfileInst,
            std::enable_if_t<std::is_base_of_v<ProfileInst, SubProfileInst>,
                             bool> = true>
  auto Evaluate(const SubProfileInst&) {
    if constexpr (SubProfileInst::kProducesValue) {
      return absl::StatusOr<NullConst>(NullConst::Any());
    } else {
      return absl::OkStatus();
    }
  }

  absl::Status Evaluate(const ProfileInst&) { return absl::OkStatus(); }

  absl::StatusOr<NullConst> Evaluate(const BoxInst&) {
    // Box can fail to allocate but we ignore it. The generated code may
    // segfault if boxing fails but that means we are OOM which should not be
    // possible on a normal linux.
    return NullConst::NotNull();
  }
  absl::StatusOr<NullConst> Evaluate(const UnboxInst& inst) {
    if (auto val = info().ConstantValue(inst.boxed())) {
      PyObject* boxed = reinterpret_cast<PyObject*>(*val);
      auto res = inst.Evaluate(boxed);
      if (res) return NullConst::Constant(*res);

      EventCounters::Instance().Add("optimizer.tautological deoptimize", 1);
      return NullConst::Any();
    }
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const OverflowedInst& inst) {
    if (auto ubi = dyn_cast<UnboxInst>(inst.arithmetic_value())) {
      if (auto val = info().ConstantValue(ubi->boxed())) {
        PyObject* boxed = reinterpret_cast<PyObject*>(*val);
        return NullConst::Constant(inst.Evaluate(*ubi, boxed));
      }
      return NullConst::Any();
    }
    if (auto ui = dyn_cast<UnaryInst>(inst.arithmetic_value())) {
      if (auto val = info().ConstantValue(ui->operand())) {
        return NullConst::Constant(inst.Evaluate(*ui, *val));
      }
      return NullConst::Any();
    }
    if (auto bi = dyn_cast<BinaryInst>(inst.arithmetic_value())) {
      auto lhs = info().ConstantValue(bi->lhs());
      auto rhs = info().ConstantValue(bi->rhs());
      if (lhs && rhs) {
        return NullConst::Constant(inst.Evaluate(*bi, *lhs, *rhs));
      }
      return NullConst::Any();
    }
    return NullConst::Any();
  }

  absl::StatusOr<NullConst> Evaluate(const FloatZeroInst& inst) {
    absl::optional<int64_t> const_float_value =
        info().ConstantValue(inst.float_value());
    if (!const_float_value) return NullConst::Any();
    return NullConst::Constant(
        inst.Evaluate(absl::bit_cast<double>(*const_float_value)));
  }

  absl::StatusOr<NullConst> Evaluate(const RematerializeInst&) {
    return NullConst::Any();
  }

  absl::StatusOr<NullConst> Evaluate(const GetClassIdInst& inst) {
    AssumeNotNull(inst.object());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const GetObjectDictInst& inst) {
    AssumeNotNull(inst.object());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const GetInstanceClassIdInst& inst) {
    AssumeNotNull(inst.dict());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const CheckClassIdInst& inst) {
    AssumeNotNull(inst.object());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const LoadFromDictInst& inst) {
    AssumeNotNull(inst.dict());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const StoreToDictInst& inst) {
    AssumeNotNull(inst.dict());
    return NullConst::Any();
  }
  absl::StatusOr<NullConst> Evaluate(const ConstantAttributeInst& inst) {
    return NullConst::Constant(reinterpret_cast<int64_t>(
        inst.LookupAttribute(ClassManager::Instance()).value()));
  }
  absl::Status Evaluate(const SetObjectClassInst& inst) {
    AssumeNotNull(inst.object());
    AssumeNotNull(inst.dict());
    return absl::OkStatus();
  }

  absl::StatusOr<NullConst> Evaluate(const DeoptimizedAsynchronouslyInst&) {
    return NullConst::Any();
  }

  const PointInfo& info_;
  const PyCodeObject* code_;
  Diff diff_;
};

absl::StatusOr<PointInfo::Diff> GetUpdate(const PointInfo& info,
                                          const Instruction& inst,
                                          const PyCodeObject* code) {
  Evaluator evaluator(info, code);
  return ForAllInstructionKinds<Evaluator>(evaluator, inst);
}

////////////////////////////////////////////////////////////////////////////////
// Rewriter implementation

void Rewriter::RewriteBlockArgument(s6::Rewriter& rewriter, PointInfos infos,
                                    BlockArgument* arg) {
  auto [info] = infos;
  if (auto val = info.ConstantValue(arg)) {
    // If a block argument is known to be a constant, replace it by
    // that constant.
    Builder builder = Builder::FromStart(arg->parent());
    rewriter.ReplaceAllUsesWith(*arg, *builder.Int64(*val));
  }
}

void Rewriter::RewriteLiveValue(s6::Rewriter& rewriter, PointInfos infos,
                                Block& block, Value* v) {
  auto [info] = infos;
  if (auto val = info.ConstantValue(v)) {
    if (IsConstantInstruction(v)) {
      // Do not replace a constant by a constant.
      return;
    }
    Builder builder = Builder::FromStart(&block);
    rewriter.ReplaceUsesWith(block, *v, *builder.Int64(*val));
  }
}

void Rewriter::RewriteInstruction(s6::Rewriter& rewriter, PointInfos infos,
                                  Diffs diffs, Instruction& inst) {
  auto [info_] = infos;
  // HACK: structured bindings can't be captured by lambdas, and I need
  // to capture info. I suppose this will be resolved in C++20 or C++23.
  auto& info = info_;

  auto [diff] = diffs;

  if (InstructionTraits::ProducesValue(inst)) {
    S6_CHECK(diff);
    auto valit = absl::c_find_if(
        *diff, [&](const auto& binding) { return binding.first == &inst; });
    S6_CHECK(valit != diff->end());
    S6_CHECK(valit->second.has_value());
    if (auto val = valit->second->ConstantValue()) {
      if (!IsConstantInstruction(&inst)) {
        // Non constant value producing instruction that actually produces
        // a constant is replaced by a ConstantInst.
        Builder b = rewriter.CreateBuilderBefore(inst);
        rewriter.ReplaceAllUsesWith(inst, *b.Int64(*val));
        // Dead code elimination will remove the old instruction if it
        // can be removed (It may have side effects).
      }
    }
  }

  // This function should be applied on all operands that are used a
  // a `condition`. It will perform the appropriate actions to make the
  // condition constant if it is statically known which branch will be
  // taken.
  auto HandleCondition = [&](Value** condition) {
    if (!info.contains(*condition)) return;
    if (info.at(*condition) <= NullConst::NotNull()) {
      // If we know the condition is not null, replace the condition by one.
      Builder b = rewriter.CreateBuilderBefore(inst);
      *condition = b.Int64(1);
    } else if (info.at(*condition) <= NullConst::Null()) {
      // If we know the condition is null, it should already have been
      // replaced by zero.
      ConstantInst* ci = dyn_cast<ConstantInst>(*condition);
      S6_CHECK(ci && ci->value() == 0) << FormatOrDie(**condition);
    }
  };

  if (auto dsi = dyn_cast<DeoptimizeIfSafepointInst>(&inst)) {
    HandleCondition(dsi->mutable_condition());
    return;
  }  // End DeoptimizeIfSafepointInst.

  if (auto rci = dyn_cast<RefcountInst>(&inst)) {
    NullConst nullness = info.Nullness(rci->operand());
    if (nullness <= NullConst::Null()) {
      rewriter.erase(*rci);
    } else if (nullness <= NullConst::NotNull()) {
      *rci->mutable_nullness() = Nullness::kNotNull;
    }
    return;
  }  // End RefCountInst.

  if (auto cti = dyn_cast<ConditionalTerminatorInst>(&inst)) {
    HandleCondition(cti->mutable_condition());
    return;
  }  // End ConditionalTerminatorInst.
}

}  // namespace nullconst

}  // namespace deepmind::s6
