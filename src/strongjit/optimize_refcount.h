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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_REFCOUNT_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_REFCOUNT_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "core_util.h"
#include "strongjit/formatter.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimize_nullconst.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/value.h"
#include "strongjit/value_map.h"

// TODO: Write a top-level comment about refcounting.

namespace deepmind::s6 {

namespace refcount {

////////////////////////////////////////////////////////////////////////////////
// Refcounting information infrastructure

using Update = absl::InlinedVector<std::pair<const Value*, int64_t>, 1>;

using ValueSet = absl::flat_hash_set<const Value*>;

// Single source of truth for the refcounting behavior of any instructions.
//
// If a value is present with a count of zero, it means it is ref-counted but is
// not changed by this operation.
absl::StatusOr<Update> GetUpdate(const Instruction& inst);

// Compute which values are RefCounted. The result is a bitset that has bits
// set to one to represent that a value is refcounted.
absl::StatusOr<ValueSet> RefCountedValues(const Function& f,
                                          const ValueNumbering& numbering);

////////////////////////////////////////////////////////////////////////////////
// PointInfo

// Stores the current refcount of the program Value at a specific point.
// The whole analysis fails if the refcount is not static at each point.
// Not-refcounted value are not in the map, so it is possible for a live value
// to not be in this map. This is different from nullconst::PointInfo.
class PointInfo {
 public:
  // The underlying data representation for a PointInfo.
  using Raw = absl::flat_hash_map<const Value*, int64_t>;

  int64_t at(const Value* v) const { return data_.at(v); }
  int64_t& at(const Value* v) { return data_.at(v); }
  int64_t AtOrZero(const Value* v) const {
    return data_.contains(v) ? data_.at(v) : 0;
  }

  // Test if a Value is contained in the PointInfo.
  // This name came from the C++20 standard.
  bool contains(const Value* v) const { return data_.contains(v); }

  // Return a constant view of the underlying map.
  const Raw& data() const { return data_; }

  // API expected by a DiffVec.
  using Diff = Update;

  // Applies an Update to the PointInfo. When applying an Update, any value that
  // was previously untracked (not in the map) but is present in the update,
  // will start being tracked as if it has 0 owned reference before the update.
  // This is the main mechanism by which new values defined in a block are added
  // to the PointInfo.
  void Apply(const Diff& d);

  explicit PointInfo() {}
  explicit PointInfo(const Raw& data) : data_(data) {}
  explicit PointInfo(Raw&& data) : data_(std::move(data)) {}

 private:
  friend class Analysis;

  // Sets a value to be tracked with a specific refcount. There are three
  // scenarios:
  // - If the value was untracked, it is added to the PointInfo with the
  // specified rc
  //   and this returns true to signal that the PointInfo was modified.
  // - If the value was already tracked with the specified rc, this does nothing
  //   and return false to signal that the PointInfo was not mutated.
  // - If the value was already tracked with a different rc, this returns an
  //   error of inconsistency.
  //
  // This function is used to check that on block entry, all predecessors pass
  // values with the same reference counts.
  absl::StatusOr<bool> Set(const Value* v, int64_t rc);

  // Add a new value to the PointInfo.
  void NewValue(const Value* v, int64_t rc) {
    S6_DCHECK(!data_.contains(v));
    data_.insert({v, rc});
  }

  // Checks if the PointInfo owns any value satisfying the predicate. Owning a
  // value means that the value is in the PointInfo with a non-0 refcount.
  bool OwnsAnyOf(absl::FunctionRef<bool(const Value*)> pred) const;

  Raw data_;
};

// Get a string representation of a PointInfo with a numbering.
// This will be dicovered by ADL in outer namespaces.
std::string ToString(const PointInfo& info, const ValueNumbering& numbering);

////////////////////////////////////////////////////////////////////////////////
// Analysis

// Computes the expected refcounts from the information in a safepoint.
absl::flat_hash_map<const Value*, int64_t> RcFromSafepoint(
    const SafepointInst& safepoint);

// Computes the expected entrypointInfo for a block. This will work only if the
// block contains a safepoint. Otherwise this returns absl::nullopt.
absl::optional<PointInfo> EntryPointInfo(const Block& b);

// Returns if a value can possibly be a valid pointer according to the nullconst
// analysis.
inline bool CanBeValid(nullconst::PointInfo ninfo, const Value* v) {
  nullconst::NullConst nullconstness = ninfo.Nullness(v);
  if (auto val = nullconstness.ConstantValue()) {
    // We use 1 as a special invalid value different from 0 in some contexts.
    if (val == 0 || val == 1) return false;
  }
  return true;
}

// Main body of the RefCount analysis using the analysis framework and thus
// usable in the analysis::Manager. The analysis propagates value by value
// static information about the number of references owned by the function to
// that value. It does not track any values that is not refcounted (plain
// integers or nullptr).
class Analysis : public analysis::Base<PointInfo> {
 public:
  static constexpr absl::string_view kName = "refcount";

  template <typename Context>
  absl::StatusOr<PointInfo> InitializeBlock(Context& ctxt, const Block& b) {
    return EntryPointInfo(b).value_or(PointInfo{});
  }

  template <typename Context>
  absl::Status ProcessFunctionEntry(Context& ctxt, const Block& entry,
                                    PointInfo& info) {
    if (PyTuple_GET_SIZE(ctxt.code_object()->co_cellvars) != 0) {
      return absl::UnimplementedError(
          "Handling of cellvar in refcount analysis is unsupported");
    }
    for (const BlockArgument* arg : entry.block_arguments()) {
      S6_CHECK_OK(info.Set(arg, 1));
    }
    return absl::OkStatus();
  }

  template <typename Context>
  absl::Status ProcessBlockEntry(Context& ctxt, const Block& b,
                                 PointInfo& info) {
    return absl::OkStatus();
  }

  template <typename Context>
  absl::StatusOr<Diff> ProcessInstruction(Context& ctxt,
                                          const Instruction& inst,
                                          const PointInfo& info) {
    // Access the state of the nullconst analysis for that instruction.
    const nullconst::PointInfo& ninfo =
        ctxt.template GetCurrentPointInfo<nullconst::Analysis>();

    // Get the refcount information of that instruction.
    S6_ASSIGN_OR_RETURN(Update update, GetUpdate(inst));

    // If the value is refcounted but the instruction that produced it, did not
    // mark it as refcounted (for example when increffing a constant), then
    // we force it to be refcounted by adding it to the diff with a refcount
    // update of 0.
    if (InstructionTraits::ProducesValue(inst) && refcounted_.contains(&inst)) {
      auto it =
          absl::c_find_if(update, [&](auto p) { return p.first == &inst; });
      if (it == update.end()) update.push_back({&inst, 0});
    }
    // Remove all values from update that cannot be valid.
    STLEraseIf(update, [&](auto p) { return !CanBeValid(ninfo, p.first); });

    // Safepoint check: Check that given the current number of reference owned,
    // taking a safepoint will lead to neither a memory leak nor a dangling
    // reference.
    if (auto safepoint = dyn_cast<SafepointInst>(&inst)) {
      auto safepoint_refcounts = RcFromSafepoint(*safepoint);
      if (auto* yield = dyn_cast<YieldValueInst>(&inst)) {
        // We must also have an extra reference to the yielded value as one
        // owning reference to it will be yielded.
        safepoint_refcounts[yield->yielded_value()]++;
      }
      for (auto [v, rc] : safepoint_refcounts) {
        if (!CanBeValid(ninfo, v)) continue;
        if (isa<RematerializeInst>(v)) continue;
        if (hard_checks_) {
          S6_CHECK(info.contains(v))
              << FormatOrDie(inst) << " for " << FormatOrDie(*v) << " "
              << ToString(info, ctxt.live_info().numbering()) << "\n"
              << FormatOrDie(ctxt.function());
          S6_CHECK(info.at(v) == rc)
              << FormatOrDie(inst) << " for " << FormatOrDie(*v) << " "
              << ToString(info, ctxt.live_info().numbering());
        } else {
          if (!info.contains(v) || info.at(v) != rc) {
            return absl::FailedPreconditionError(
                "Inconsistency with safepoint refcounting info");
          }
        }
      }
      for (auto [v, rc] : info.data()) {
        if (!CanBeValid(ninfo, v)) continue;
        if (rc == 0) continue;
        if (hard_checks_) {
          S6_CHECK(safepoint_refcounts.contains(v))
              << FormatOrDie(inst) << " for " << FormatOrDie(*v) << " "
              << ToString(info, ctxt.live_info().numbering()) << "\n"
              << FormatOrDie(ctxt.function());
        } else {
          if (!safepoint_refcounts.contains(v)) {
            return absl::FailedPreconditionError(
                "Inconsistency with safepoint refcounting info");
          }
        }
      }
    }

    return update;
  }

  template <typename Context>
  absl::StatusOr<TerminatorDiffs> ProcessTerminator(Context& ctxt,
                                                    const TerminatorInst& inst,
                                                    PointInfo& info) {
    const nullconst::PointInfo& ninfo =
        ctxt.template GetTerminatingPointInfo<nullconst::Analysis>();

    // If we are exiting the function (except without successor or return).
    if (auto except = dyn_cast<ExceptInst>(&inst);
        (except && !except->unique_successor()) || isa<ReturnInst>(inst)) {
      // Then, if we own any valid pointer there is a leak.
      if (info.OwnsAnyOf(
              [&](const Value* v) { return CanBeValid(ninfo, v); })) {
        constexpr absl::string_view kMessage =
            "Exiting function without having freed everything";
        if (hard_checks_) {
          S6_LOG(FATAL) << kMessage << " at " << FormatOrDie(inst)
                        << "\nrefcounts: "
                        << ToString(info, ctxt.live_info().numbering())
                        << "\nnullconst: "
                        << ToString(ninfo, ctxt.live_info().numbering());
        }
        return absl::FailedPreconditionError(kMessage);
      }
    }

    return TerminatorDiffs(inst.successor_size(), Diff{});
  }

  template <typename Context>
  absl::Status ProcessSuccessor(Context& ctxt, const TerminatorInst& inst,
                                const Block& successor, PointInfo& info) {
    absl::Status status = ctxt.TransferToSuccessor(successor);
    const nullconst::PointInfo& ninfo =
        ctxt.template GetTerminatingPointInfo<nullconst::Analysis>();

    // The references transferred to the successor are removed from the current
    // successor specific PointInfo during ctxt.TransferToSuccessor so if we own
    // any remaining valid pointer there is a leak.
    if (info.OwnsAnyOf([&](const Value* v) { return CanBeValid(ninfo, v); })) {
      constexpr absl::string_view kMessage =
          "Jumping to successor without transferring all owned values to it";
      if (hard_checks_) {
        S6_CHECK(false) << kMessage << " at " << FormatOrDie(inst)
                        << " on successor " << FormatOrDie(successor)
                        << "\nrefcounts: "
                        << ToString(info, ctxt.live_info().numbering())
                        << "\nnullconst: "
                        << ToString(ninfo, ctxt.live_info().numbering()) << "\n"
                        << FormatOrDie(ctxt.function());
      }
      return absl::FailedPreconditionError(kMessage);
    }
    return status;
  }

  template <typename Context>
  absl::StatusOr<bool> TransferValue(Context& ctxt, const Value* src,
                                     PointInfo& src_info, const Value* dest,
                                     PointInfo& dest_info) {
    const nullconst::PointInfo& ninfo =
        ctxt.template GetTerminatingPointInfo<nullconst::Analysis>();
    if (!src_info.contains(src)) return false;
    if (ninfo.Nullness(src) <= nullconst::NullConst::Null()) return false;
    if (dest_info.contains(dest)) {
      // Transfer ownership to successor by decrementing number of owned
      // references in the source by the amount of expected reference in the
      // destination. This might not set the number of source references to 0 as
      // a value may be transferred twice, being simultaneously a block
      // arguments and a simple live-through value.
      src_info.at(src) -= dest_info.at(dest);
      return false;
    } else {
      // If we don't have information about the expected number of references,
      // we transfer all the reference we have in the source to the destination.
      dest_info.NewValue(dest, src_info.at(src));
      src_info.at(src) = 0;
      return true;
    }
  }

  template <typename Context>
  absl::StatusOr<bool> TransferExceptImplicitValue(Context& ctxt, int64_t index,
                                                   const Value* dest,
                                                   PointInfo& dest_info) {
    // In an ExceptInst, all 6 implicit values start with a refcount of one.
    return dest_info.Set(dest, 1);
  }

  explicit Analysis(const ValueSet& refcounted, bool hard_checks = false)
      : refcounted_(refcounted), hard_checks_(hard_checks) {}

 private:
  const ValueSet& refcounted_;

  // If enabled, any refcounting error will result in a fatal error and
  // immediate abort. This is good for regression tests but not so good for
  // normal production use. If disabled, a refcounting error, will return an
  // absl::Status error and abort the analysis.
  bool hard_checks_;
};

////////////////////////////////////////////////////////////////////////////////
// Rewriter

absl::StatusOr<ValueSet> GetExternallyOwnedValues(const Function& function,
                                                  const ValueSet& refcounted);

// The refcount rewriter. The goal of the rewriter is lower to number of owned
// references for every object to 1 and 0 in hope to reduce the number of
// refcounting operations. We will target 1 except if an object is externally
// owned as specified by `GetExternallyOwnedValues`.
//
// In order to lower the refcount from the original value to the target value.
// I remove or cancel all refcounting operations that would raise the refcount
// above the target value and also remove all those that would decref it from a
// higher target values. I respect the exact positions we a value is decreffed
// below the target values.
//
// This pass requires that there is no critical edges.
class Rewriter : public analysis::RewriterBase<nullconst::Analysis, Analysis> {
 public:
  using analysis::RewriterBase<nullconst::Analysis, Analysis>::RewriterBase;

  void RewriteBlockArgument(s6::Rewriter& rewriter, PointInfos infos,
                            BlockArgument* arg) {}
  void RewriteLiveValue(s6::Rewriter& rewriter, PointInfos infos, Block& block,
                        Value* v) {}
  void RewriteInstruction(s6::Rewriter& rewriter, PointInfos infos, Diffs diffs,
                          Instruction& inst);

  explicit Rewriter(const ValueSet& refcounted,
                    const ValueSet& externally_owned_values)
      : refcounted_(refcounted),
        externally_owned_values_(externally_owned_values) {}

 private:
  const ValueSet& refcounted_;
  const ValueSet& externally_owned_values_;

  int64_t GetTargetCount(const Value* v) {
    S6_DCHECK(refcounted_.contains(v));
    if (externally_owned_values_.contains(v)) return 0;
    return 1;
  }
};

}  // namespace refcount

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_REFCOUNT_H_
