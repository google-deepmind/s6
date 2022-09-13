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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_NULLCONST_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_NULLCONST_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimizer_analysis.h"
#include "strongjit/optimizer_util.h"

namespace deepmind::s6 {

namespace nullconst {

////////////////////////////////////////////////////////////////////////////////
// NullConst lattice.

// Represents static nullness and constness data for Values.
//
// This is a lattice of static information known on a Value.
// It can be seen as a compact representation of set of possible values,
// in which case <= is the inclusion, Meet is the intersection and Join is the
// union.
//
// The Value can be a constant, a non-null value or anything.
// There is also a an impossible value representation. This
// can only appear in unreachable code where a Value has no possible concrete
// value. In the set interpretation, Impossible represents the empty set.
class NullConst {
 public:
  // Returns a NullConst representing a constant value of that constant.
  static constexpr NullConst Constant(int64_t c) {
    return NullConst(kConst, c);
  }
  // Returns a NullConst representing a value that cannot be null
  static constexpr NullConst NotNull() { return NullConst(kNotNull, 0); }
  // Returns a NullConst representing a value that can be anything.
  static constexpr NullConst Any() { return NullConst(kAny, 0); }
  // Returns a NullConst representing a value that is impossible.
  // This means the owning control-flow path is unreachable.
  static constexpr NullConst Impossible() { return NullConst(kImpossible, 0); }
  // Returns a NullConst representing a value that is null.
  static constexpr NullConst Null() { return NullConst(kConst, 0); }

  // If this is a constant, returns the constant value.
  // Otherwise returns nullopt.
  absl::optional<int64_t> ConstantValue() const {
    if (kind_ == kConst) return val_;
    return absl::nullopt;
  }

  // Check if this NullConst is a constant. All other types of NullConst
  // can be checked with a plain operator==.
  bool IsConstant() const { return kind_ == kConst; }

  // Check if the value represented by this NullConst can contain 0.
  // This is equivalent to !(*this <= NotNull()).
  bool MayBeNull() const {
    return kind_ == kAny || (kind_ == kConst && val_ == 0);
  }

  friend bool operator==(NullConst lhs, NullConst rhs) {
    return lhs.kind_ == rhs.kind_ && lhs.val_ == rhs.val_;
  }
  friend bool operator!=(NullConst lhs, NullConst rhs) { return !(lhs == rhs); }

  // Lattice comparison. A NullConst is smaller than another one if it is more
  // precise. In other words, lhs <= rhs means that all values represented by
  // lhs are also represented by rhs.
  // This is NOT a total order.
  friend bool operator<=(NullConst lhs, NullConst rhs) {
    if (lhs.kind_ == kImpossible) return true;
    switch (rhs.kind_) {
      case kAny:
        return true;
      case kImpossible:
        return false;
      case kNotNull:
        if (lhs.kind_ == kAny) return false;
        if (lhs.kind_ == kConst) return lhs.val_ != 0;
        return true;
      case kConst:
        return lhs.kind_ == kConst && lhs.val_ == rhs.val_;
    }
    S6_UNREACHABLE();
  }

  friend bool operator<(NullConst lhs, NullConst rhs) {
    return lhs <= rhs && lhs != rhs;
  }

  friend bool operator>=(NullConst lhs, NullConst rhs) { return rhs <= lhs; }

  friend bool operator>(NullConst lhs, NullConst rhs) {
    return lhs >= rhs && lhs != rhs;
  }

  // Lattice meet. Returns the largest value that is smaller than both `lhs` and
  // `rhs`
  friend NullConst Meet(NullConst lhs, NullConst rhs) {
    switch (lhs.kind_) {
      case kAny:
        return rhs;
      case kImpossible:
        return Impossible();
      case kNotNull:
        if (rhs.kind_ == kAny) return lhs;
        if (rhs == Null()) return Impossible();
        return rhs;
      case kConst:
        if (rhs.IsConstant()) {
          if (lhs.val_ == rhs.val_) return lhs;
          return Impossible();
        }
        return Meet(rhs, lhs);
    }
    S6_UNREACHABLE();
  }

  NullConst MeetEq(NullConst oth) & { return *this = Meet(*this, oth); }

  // Check if `lhs` and `rhs` have common values. This is equivalent to their
  // Meet not being Impossible.
  friend bool IsCompatible(NullConst lhs, NullConst rhs) {
    return Meet(lhs, rhs) != Impossible();
  }

  // Lattice join. Returns the smallest value that is larger than both `lhs` and
  // `rhs`
  friend NullConst Join(NullConst lhs, NullConst rhs) {
    switch (lhs.kind_) {
      case kAny:
        return Any();
      case kImpossible:
        return rhs;
      case kNotNull:
        return rhs.MayBeNull() ? Any() : lhs;
      case kConst:
        if (rhs.IsConstant()) {
          if (lhs.val_ == rhs.val_) return lhs;
          if (lhs.val_ != 0 && rhs.val_ != 0) return NotNull();
          return Any();
        }
        return Join(rhs, lhs);
    }
    S6_UNREACHABLE();
  }

  NullConst JoinEq(NullConst oth) & { return *this = Join(*this, oth); }

  friend std::string ToString(NullConst c) {
    switch (c.kind_) {
      case kAny:
        return "any";
      case kImpossible:
        return "impossible";
      case kNotNull:
        return "not null";
      case kConst:
        return absl::StrCat(c.val_);
    }
    S6_UNREACHABLE();
  }

 private:
  enum Kind {
    kConst,
    kNotNull,
    kAny,
    kImpossible,
  };
  constexpr NullConst(Kind kind, int64_t val) : kind_(kind), val_(val) {}
  Kind kind_;
  int64_t val_;
};

////////////////////////////////////////////////////////////////////////////////
// PointInfo

// Stores information about nullness and constness of all live values at a
// specific point in the program.
class PointInfo {
 public:
  // The underlying data representation for a PointInfo.
  using Raw = absl::flat_hash_map<const Value*, NullConst>;

  // Low-level access to the underlying map. Pefer to use `Nullness` unless
  // you know for certain the value is in the map.
  NullConst at(const Value* v) const { return data_.at(v); }
  NullConst& at(const Value* v) { return data_.at(v); }

  // Test if a Value is contained in the PointInfo.
  // This name came from the C++20 standard.
  bool contains(const Value* v) const { return data_.contains(v); }

  // Return a constant view of the underlying map.
  const Raw& data() const { return data_; }

  // Returns the Nullness of a value. If the value is not in the map, this
  // method will not fail and will do best effort instead.
  // In particular it is guaranteed that it returns a constant when called on
  // a ConstantInst.
  // nullptr is valid and the answer will be NullConst::Any();
  NullConst Nullness(const Value* v) const {
    if (auto constant = dyn_cast<ConstantInst>(v))
      return NullConst::Constant(constant->value());
    if (!contains(v)) return NullConst::Any();
    return at(v);
  }

  // Shortcut for Nullness(v).ConstantValue().
  absl::optional<int64_t> ConstantValue(const Value* v) const {
    return Nullness(v).ConstantValue();
  }

 public:
  // This class is used to represent a `Diff` that can be applied on a
  // NullInfo::PointInfo. It should only be generated by `SplitOn` or
  // `ApplyDiff` and should only be used by `ApplyDiff`.
  using Diff =
      absl::InlinedVector<std::pair<const Value*, absl::optional<NullConst>>,
                          1>;
  friend std::string ToString(const Diff& diff, const ValueNumbering& vn);

  // Applies a diff on a NullInfo::PointInfo.
  // Returns a reverse diff to apply to restore the original PointInfo.
  Diff Apply(const Diff& diff) { return ApplyMapDiff(data_, diff); }

 private:
  friend class Analysis;

  // This class is only defined in optimize_nullconst.cc
  friend class Evaluator;

  // All private methods and classes are documented in optimize_nullconst.cc
  std::array<Diff, 2> SplitOn(const Block& b, const Value* v) const;
  void NewValue(const Value* v, NullConst nullness);
  bool JoinValue(const Value* v, NullConst nullness);

  Raw data_;
};

// Get a string representation of a PointInfo with a numbering.
// This will be dicovered by ADL in outer namespaces.
std::string ToString(const PointInfo& info, const ValueNumbering& numbering);

// Returns the update to apply to a PointInfo when statically running an
// instruction.
absl::StatusOr<PointInfo::Diff> GetUpdate(const PointInfo& info,
                                          const Instruction& inst,
                                          const PyCodeObject* code);

////////////////////////////////////////////////////////////////////////////////
// Analysis

// Main body of the NullConst analysis using the analysis framework and thus
// usable in the analysis::Manager. The analysis propagates value by value
// static information in the NullConst latice. It does not remember any
// relations between values.
class Analysis : public analysis::Base<PointInfo> {
 public:
  static constexpr absl::string_view kName = "nullconst";

  template <typename Context>
  absl::StatusOr<PointInfo> InitializeBlock(Context& ctxt, const Block& b) {
    // By default all block starting points are initialized with Impossible to
    // signify that they are unreachable.
    PointInfo result;
    for (const BlockArgument* arg : b.block_arguments()) {
      result.NewValue(arg, NullConst::Impossible());
    }
    for (const Value* v : ctxt.LiveValues(b)) {
      result.NewValue(v, NullConst::Impossible());
    }
    return result;
  }

  template <typename Context>
  absl::Status ProcessFunctionEntry(Context& ctxt, const Block& entry,
                                    PointInfo& info) {
    // The function arguments are not null except if there are cell arguments.
    // TODO: Figure out which specific arguments can be null.
    if (PyTuple_GET_SIZE(ctxt.code_object()->co_cellvars) == 0) {
      for (const BlockArgument* arg : entry.block_arguments()) {
        info.at(arg) = NullConst::NotNull();
      }
    } else {
      for (const BlockArgument* arg : entry.block_arguments()) {
        info.at(arg) = NullConst::Any();
      }
    }
    return absl::OkStatus();
  }

  template <typename Context>
  absl::Status ProcessBlockEntry(Context& ctxt, const Block& b,
                                 PointInfo& info) {
    S6_VLOG(1) << "Entering " << FormatOrDie(b) << " with "
               << ToString(info, ctxt.live_info().numbering());
    return absl::OkStatus();
  }

  template <typename Context>
  absl::StatusOr<Diff> ProcessInstruction(Context& ctxt,
                                          const Instruction& inst,
                                          const PointInfo& info) {
    S6_ASSIGN_OR_RETURN(auto diff, GetUpdate(info, inst, ctxt.code_object()));
    for (auto [value, nc] : diff) {
      if (nc == NullConst::Impossible()) {
        // We are in an impossible case, that means that instructions after
        // the current one are unreachable.
        ctxt.ExitBlock();
      }
    }
    return diff;
  }

  template <typename Context>
  absl::StatusOr<bool> TransferValue(Context& ctxt, const Value* src,
                                     const PointInfo& src_info,
                                     const Value* dest, PointInfo& dest_info) {
    return dest_info.JoinValue(dest, src_info.Nullness(src));
  }

  template <typename Context>
  absl::StatusOr<bool> TransferExceptImplicitValue(Context& ctxt, int64_t index,
                                                   const Value* dest,
                                                   PointInfo& dest_info) {
    // In an ExceptInst, index 0 to 3 are Any, index 4 and 5 are NotNull.
    // 4 is the current exception value and 5 the current exception type.
    return dest_info.JoinValue(
        dest, index < 4 ? NullConst::Any() : NullConst::NotNull());
  }

  template <typename Context>
  absl::StatusOr<TerminatorDiffs> ProcessTerminator(Context& ctxt,
                                                    const TerminatorInst& inst,
                                                    PointInfo& info) {
    TerminatorDiffs result;
    if (!isa<ConditionalTerminatorInst>(inst)) {
      for (const Block* succ : inst.successors()) {
        (void)succ;
        result.push_back(Diff{});
      }
      return result;
    }
    const auto& br = cast<ConditionalTerminatorInst>(inst);
    NullConst condition_nullness = info.Nullness(br.condition());
    if (condition_nullness == NullConst::Impossible()) {
      return absl::InternalError(
          "Reached NullConst::Impossible() when branching");
    }
    if (condition_nullness <= NullConst::Null()) {
      result.push_back(absl::nullopt);  // true successor
      result.push_back(Diff{});         // false successor
      return result;
    }
    if (condition_nullness <= NullConst::NotNull()) {
      result.push_back(Diff{});         // true successor
      result.push_back(absl::nullopt);  // false successor
      return result;
    }

    auto [diff_false, diff_true] = info.SplitOn(*br.parent(), br.condition());
    result.push_back(std::move(diff_true));
    result.push_back(std::move(diff_false));
    return result;
  }

  template <typename Context>
  absl::Status ProcessSuccessor(Context& ctxt, const TerminatorInst& inst,
                                const Block& successor, PointInfo& info) {
    return ctxt.TransferToSuccessor(successor);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Rewriter

// Rewriter that performs the NullConst optimization. This takes care of
// multiple things:
// - Changes decref null? into decref notnull when the value is known to be
//   not null
// - Removes decref null? when the operand is known to be null.
// - Sets condition in br and deoptimize_if instruction to a constant value
//   when the condition result is known statically.
// - Replaces all uses of an instruction that return a constant by a
//   ConstantInst. The original instruction is left to be removed by dead code
//   elimination if possible.
class Rewriter : public analysis::RewriterBase<Analysis> {
 public:
  using analysis::RewriterBase<Analysis>::RewriterBase;

  void RewriteBlockArgument(s6::Rewriter& rewriter, PointInfos infos,
                            BlockArgument* arg);
  void RewriteLiveValue(s6::Rewriter& rewriter, PointInfos infos, Block& block,
                        Value* v);
  void RewriteInstruction(s6::Rewriter& rewriter, PointInfos infos, Diffs diffs,
                          Instruction& inst);
};

}  // namespace nullconst

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_NULLCONST_H_
