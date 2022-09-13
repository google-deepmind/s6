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

#include "strongjit/optimize_refcount.h"

#include <functional>
#include <iterator>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "cppitertools/enumerate.hpp"
#include "cppitertools/reversed.hpp"
#include "cppitertools/sorted.hpp"
#include "cppitertools/zip.hpp"
#include "strongjit/base.h"
#include "strongjit/block.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimize_nullconst.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "strongjit/value_map.h"
#include "utils/range.h"

namespace deepmind::s6 {

namespace refcount {

////////////////////////////////////////////////////////////////////////////////
// Refcounting information infrastructure
//
// This the section where the static reference counting behavior of instruction
// is computed.

// The Refcount instruction evaluator.
struct Evaluator {
  template <typename InstrType>
  static std::optional<absl::StatusOr<Update>> Visit(const Instruction& inst) {
    if (inst.kind() != InstrType::kKind) return {};
    return Apply(cast<InstrType>(inst));
  }
  static absl::StatusOr<Update> Default(const Instruction& inst) {
    return absl::UnimplementedError(absl::StrCat(
        FormatOrDie(inst), " has no refcount info implementation"));
  }
  static absl::StatusOr<Update> Apply(const ReturnInst& inst) {
    return Update{{inst.returned_value(), -1}};
  }
  static absl::StatusOr<Update> Apply(const YieldValueInst& inst) {
    return Update{{inst.yielded_value(), -1}, {&inst, 1}};
  }
  static absl::StatusOr<Update> Apply(const BytecodeBeginInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const DeoptimizeIfSafepointInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const TerminatorInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const ConstantInst&) { return Update{}; }
  static absl::StatusOr<Update> Apply(const NumericInst&) { return Update{}; }
  static absl::StatusOr<Update> Apply(const IncrefInst& inst) {
    return Update{{inst.operand(), 1}};
  }
  static absl::StatusOr<Update> Apply(const DecrefInst& inst) {
    return Update{{inst.operand(), -1}};
  }
  static absl::StatusOr<Update> Apply(const LoadInst& inst) {
    if (inst.steal()) return Update{{&inst, 1}};
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const StoreInst& inst) {
    if (inst.donate()) return Update{{inst.stored_value(), -1}};
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const LoadGlobalInst&) {
    return absl::FailedPreconditionError(
        "There should not be any load global inst at this point");
  }
  static absl::StatusOr<Update> Apply(const FrameVariableInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const IntToFloatInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> ApplyNative(const CallInst& inst,
                                            const CalleeInfo& info) {
    Update update;
    if (info.return_new_ref()) update.push_back({&inst, 1});
    for (auto [i, arg] : iter::enumerate(inst.call_arguments())) {
      if (info.argument(i).stolen) update.push_back({arg, -1});
    }
    return update;
  }
  static absl::StatusOr<Update> Apply(const CallNativeInst& inst) {
    auto callee_info = CalleeInfo::Get(inst.callee());
    if (!callee_info) {
      return absl::UnimplementedError(
          absl::StrCat("refcounting is not supported for ", inst.CalleeName()));
    }
    return ApplyNative(inst, *callee_info);
  }
  static absl::StatusOr<Update> Apply(const CallPythonInst& inst) {
    Update result{{&inst, 1}, {inst.callee(), -1}};
    for (auto value : inst.call_arguments()) result.push_back({value, -1});
    return result;
  }
  static absl::StatusOr<Update> Apply(const CallAttributeInst& inst) {
    Update result{{&inst, 1}, {inst.object(), -1}};
    for (auto value : inst.call_arguments()) result.push_back({value, -1});
    return result;
  }
  static absl::StatusOr<Update> Apply(const CallVectorcallInst& inst) {
    Update result{{&inst, 1}, {inst.self(), 0}};
    // The callee is not refcounted.
    if (inst.names()) result.push_back({inst.names(), 0});
    for (auto value : inst.call_arguments()) result.push_back({value, 0});
    return result;
  }
  static absl::StatusOr<Update> Apply(const CallNativeIndirectInst& inst) {
    if (!inst.HasInfo()) {
      return absl::UnimplementedError(
          "Call native indirect without callee info");
    }
    return ApplyNative(inst, inst.info());
  }
  static absl::StatusOr<Update> Apply(const AdvanceProfileCounterInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const ProfileInst&) { return Update{}; }
  static absl::StatusOr<Update> Apply(const UnboxInst&) { return Update{}; }
  static absl::StatusOr<Update> Apply(const BoxInst& inst) {
    return Update{{&inst, 1}};
  }
  static absl::StatusOr<Update> Apply(const OverflowedInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const FloatZeroInst&) { return Update{}; }
  static absl::StatusOr<Update> Apply(const RematerializeInst&) {
    // Rematerialized instruction are considered not refcounted even if they
    // appear in safepoints. This is a special case for that.
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const GetClassIdInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const GetInstanceClassIdInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const GetObjectDictInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const CheckClassIdInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const LoadFromDictInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const StoreToDictInst& inst) {
    return Update{{&inst, 1}, {inst.value(), -1}};
  }
  static absl::StatusOr<Update> Apply(const ConstantAttributeInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const SetObjectClassInst&) {
    return Update{};
  }
  static absl::StatusOr<Update> Apply(const DeoptimizedAsynchronouslyInst&) {
    return Update{};
  }
};

absl::StatusOr<Update> GetUpdate(const Instruction& inst) {
  return ForAllInstructionKinds<Evaluator>(inst);
}

absl::StatusOr<ValueSet> RefCountedValues(const Function& f,
                                          const ValueNumbering& numbering) {
  ValueSet res;
  res.reserve(numbering.size() / 4);  // Heuristic estimation.
  auto set_res = [&](const Value* v) {
    if (!isa<RematerializeInst>(v)) res.insert(v);
  };
  for (const Block& b : f) {
    for (const Instruction& inst : b) {
      S6_ASSIGN_OR_RETURN(auto update, GetUpdate(inst));
      for (auto [v, rc] : update) {
        set_res(v);
      }
      if (auto safepoint = dyn_cast<SafepointInst>(&inst)) {
        for (auto arg : safepoint->fastlocals()) set_res(arg);
        for (auto arg : safepoint->value_stack()) set_res(arg);
        for (auto arg : safepoint->increfs()) set_res(arg);
        for (auto arg : safepoint->decrefs()) set_res(arg);
      }
    }
    if (b.IsHandler()) {
      for (auto ba : b.block_arguments().subspan(0, 6)) set_res(ba);
    }
  }

  // Propagate information back through block arguments.
  bool updated;
  do {
    updated = false;

    for (const Block& b : iter::reversed(f)) {
      const TerminatorInst* inst = b.GetTerminator();
      S6_DCHECK(inst);
      for (auto succ : inst->successors()) {
        inst->ForEachArgumentOnEdge(
            succ, [&](const BlockArgument* ba, const Value* v) {
              if (!v) return;
              if (res.contains(ba) && !res.contains(v)) {
                updated = true;
                res.insert(v);
              }
            });
      }
    }
  } while (updated);

  return res;
}

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
        absl::StrAppend(out, "%", numbering.at(v), " : ", n);
      });
}

// Apply a Diff/RefCountUpdate to a PointInfo.
void PointInfo::Apply(const Diff& diff) {
  for (auto [value, update] : diff) {
    auto it = data_.find(value);
    if (it == data_.end()) {
      S6_CHECK_GE(update, 0)
          << "Deleting " << FormatOrDie(*value) << " too soon";
      data_.insert({value, update});
    } else {
      it->second += update;
      S6_CHECK_GE(it->second, 0)
          << "Reaching a negative refcount for: " << FormatOrDie(*value);
    }
  }
}

// Return true if the value is newly set and false otherwise.
// Return an error status if the value was already set with a different value.
absl::StatusOr<bool> PointInfo::Set(const Value* v, int64_t rc) {
  S6_CHECK(!isa<RematerializeInst>(v));
  if (contains(v)) {
    if (rc == at(v)) return false;
    return absl::FailedPreconditionError(
        "A value has multiple reference count at a given place");
  }
  data_.insert({v, rc});
  return true;
}

bool PointInfo::OwnsAnyOf(absl::FunctionRef<bool(const Value*)> pred) const {
  for (auto [v, rc] : data_) {
    if (rc != 0 && pred(v)) return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////
// Analysis implementation

absl::flat_hash_map<const Value*, int64_t> RcFromSafepoint(
    const SafepointInst& safepoint) {
  absl::flat_hash_map<const Value*, int64_t> result;

  for (auto v : safepoint.value_stack()) ++result[v];
  for (auto v : safepoint.fastlocals()) ++result[v];
  for (auto v : safepoint.decrefs()) ++result[v];
  for (auto v : safepoint.increfs()) --result[v];

  absl::erase_if(result, [](const auto& value_and_rc) {
    return isa<RematerializeInst>(value_and_rc.first);
  });

  return result;
}

absl::optional<PointInfo> EntryPointInfo(const Block& b) {
  const SafepointInst* safepoint;
  for (const Instruction& inst : b) {
    safepoint = dyn_cast<SafepointInst>(&inst);
    if (safepoint) break;
  }
  if (!safepoint) return absl::nullopt;

  auto map = RcFromSafepoint(*safepoint);
  auto it = std::make_reverse_iterator(safepoint->GetIterator());
  for (const Instruction& inst : MakeRange(it, b.rend())) {
    auto update = GetUpdate(inst);
    if (!update.ok()) return absl::nullopt;

    for (auto [v, upd] : *update) map[v] -= upd;
    if (InstructionTraits::ProducesValue(inst)) map.erase(&inst);
  }

  return PointInfo(map);
}

////////////////////////////////////////////////////////////////////////////////
// Rewriter

absl::StatusOr<absl::flat_hash_set<const Value*>> GetExternallyOwnedValues(
    const Function& function, const ValueSet& refcounted) {
  absl::flat_hash_set<const Value*> result;
  auto add = [&](const Instruction& inst) {
    if (refcounted.contains(&inst)) result.insert(&inst);
  };
  for (const Block& block : function) {
    for (const Instruction& inst : block) {
      if (!InstructionTraits::ProducesValue(inst)) continue;
      if (auto* finst = dyn_cast<FrameVariableInst>(&inst)) {
        switch (finst->frame_variable_kind()) {
            // All those variables are owned by external entities that outlive
            // the function frame, that means that they will not be deleted
            // before the end of the function and thus the function does not
            // need to take ownership of them.
          case FrameVariableInst::FrameVariableKind::kConsts:
          case FrameVariableInst::FrameVariableKind::kBuiltins:
          case FrameVariableInst::FrameVariableKind::kGlobals:
          case FrameVariableInst::FrameVariableKind::kLocals:
          case FrameVariableInst::FrameVariableKind::kNames:
          case FrameVariableInst::FrameVariableKind::kCodeObject:
          case FrameVariableInst::FrameVariableKind::kThreadState:
            add(inst);
            break;
          default:
            break;
        }
      } else if (isa<ConstantAttributeInst>(inst)) {
        // If the class is alive, then the constant attribute is alive.
        // Otherwise the class check would have triggered and we would have
        // deoptimized.
        add(inst);
      } else if (auto* cinst = dyn_cast<CallNativeInst>(&inst)) {
        if (cinst->CalleeIs(Callee::kLoadGlobal)) add(inst);
      }
    }
  }
  return result;
}

void Rewriter::RewriteInstruction(s6::Rewriter& rewriter, PointInfos infos,
                                  Diffs diffs, Instruction& inst) {
  auto [ninfo, info] = infos;
  auto [ndiffptr, diffptr] = diffs;
  if (diffptr) {
    absl::flat_hash_map<const Value*, int64_t> update;
    for (auto [v, upd] : *diffptr) update[v] += upd;

    for (auto [v, rcupd] : update) {
      S6_DCHECK(refcounted_.contains(v));
      if (rcupd == 0) continue;
      const int64_t old_rc = info.AtOrZero(v);
      const int64_t new_rc = old_rc + rcupd;
      const int64_t target_rc = GetTargetCount(v);
      if (rcupd > 0) {
        if (new_rc <= target_rc) continue;
        if (isa<IncrefInst>(inst)) {
          S6_CHECK_EQ(rcupd, 1);
          S6_CHECK_GE(old_rc, target_rc);
          rewriter.erase(inst);
        } else {
          S6_LOG(FATAL) << "I expected not to have to insert any decref";
        }
      } else {
        S6_DCHECK_LT(rcupd, 0);
        if (old_rc <= target_rc) continue;
        if (isa<DecrefInst>(inst)) {
          S6_CHECK_EQ(rcupd, -1);
          S6_CHECK_GE(new_rc, target_rc);
          // decref goes from above the target to above the target and thus can
          // be safely ignored.
          rewriter.erase(inst);
        } else {
          int64_t num_incref;  // Number of incref to insert.
          if (new_rc < target_rc) {
            // If the final target refcount is below target_rc, then we need
            // to decref the target value from target_rc to old_rc + rcupd.
            // Thus we will raise the refcount from target_rc, to old_rc with
            // increfs, and then let the original instruction take it down to
            // old_rc + rcupd.
            num_incref = old_rc - target_rc;
          } else {
            // Otherwise everything happens above the target_rc, so the refcount
            // effect can be safely cancelled.
            num_incref = -rcupd;
          }
          Builder b = rewriter.CreateBuilderBefore(inst);
          for (int64_t i = 0; i < num_incref; ++i) {
            // We have mutable access to the function via the rewriter, so the
            // const_cast is fine.
            // The incref is null? because the nullconst analysis will run
            // afterward and convert it to notnull if appropriate.
            b.IncrefOrNull(const_cast<Value*>(v));
          }
        }
      }
    }
  }

  // We need to reincref values when taking a safepoint to raise the reference
  // counts to what CPython expects them to be.
  if (auto* safepoint = dyn_cast<SafepointInst>(&inst)) {
    auto safepoint_counts = RcFromSafepoint(*safepoint);
    for (auto [v, rc] : safepoint_counts) {
      const int64_t target_rc = GetTargetCount(v);
      // Since we are lowering the refcount from rc to target_rc, we need to get
      // back to the old refcount when taking the safepoint.
      //
      // If this is a yield, the theory is a bit more complex, but the code is
      // the same. If the yielded value does not live after the yield, then
      // it cannot appear in the safepoint list and thus it has no interaction
      // with the following code. If the yielded value lives after the yield,
      // then the previous code must have added an incref before to cancel it.
      // and in term of refcounts, the yield safepoint behave is if the yielded
      // value had no impact on the safepoint, thus the code can be the same as
      // for other safepoints.
      if (rc > target_rc) {
        for (int64_t i = 0; i < rc - target_rc; ++i) {
          safepoint->incref_value(const_cast<Value*>(v));
        }
      }
    }
  }

  // We need to check terminators because it is possible to go from a target_rc
  // of 0 to a target_rc of one when being passed as a block argument.
  if (auto* terminator = dyn_cast<TerminatorInst>(&inst)) {
    // If the block has more than one successor, we insert the incref after the
    // jump because the target block will have at most one successor given that
    // there is no critical edge.
    const bool insert_after = terminator->successor_size() > 1;
    for (Block* succ : terminator->successors()) {
      auto live_values = live_info().LiveValues(*succ);
      absl::flat_hash_set<const Value*> already_seen(live_values.begin(),
                                                     live_values.end());
      terminator->ForEachArgumentOnEdge(succ, [&](BlockArgument* ba, Value* v) {
        if (!v) return;
        if (!refcounted_.contains(v)) return;
        if (externally_owned_values_.contains(v) || already_seen.contains(v)) {
          if (insert_after) {
            S6_CHECK_EQ(succ->predecessors().size(), 1) << "Critical edge";
            Builder b = Builder::FromStart(succ);
            b.IncrefOrNull(ba);
          } else {
            Builder b = rewriter.CreateBuilderBefore(inst);
            b.IncrefOrNull(v);
          }
        } else {
          already_seen.insert(v);
        }
      });
    }
  }
}

}  // namespace refcount

}  // namespace deepmind::s6
