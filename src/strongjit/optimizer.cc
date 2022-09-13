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

#include "strongjit/optimizer.h"

#include <Python.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "classes/attribute.h"
#include "classes/class.h"
#include "classes/class_manager.h"
#include "core_util.h"
#include "event_counters.h"
#include "runtime/python_function_info_table.h"
#include "strongjit/base.h"
#include "strongjit/builder.h"
#include "strongjit/formatter.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/optimize_calls.h"
#include "strongjit/optimize_cfg.h"
#include "strongjit/optimize_constants.h"
#include "strongjit/optimize_liveness.h"
#include "strongjit/optimize_nullconst.h"
#include "strongjit/optimize_refcount.h"
#include "strongjit/optimize_type_construction.h"
#include "strongjit/optimizer_analysis.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "type_feedback.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"
#include "utils/path.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

// Emits:
//   %0 = frame_variable globals
//   %1 = frame_variable builtins
//   %2 = frame_variable names $(offset)
//   %y = call_native s6::strongjit::runtime::LoadGlobal(%3, %0, %1)
Value* EmitLoadGlobalNoCache(Builder& b, int64_t index,
                             const OptimizerOptions& options) {
  Value* globals =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kGlobals);
  Value* builtins =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kBuiltins);
  Value* name =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames, index);
  if (options.use_event_counters) {
    b.IncrementEventCounter("optimizer.globals.slow");
  }
  return b.Call(Callee::kLoadGlobal, {name, globals, builtins});
}

Value* EmitLoadGlobalSpeculated(Builder& b, Function& f, LoadGlobalInst* li,
                                PyCodeObject* code,
                                const OptimizerOptions& options) {
  if (!f.GetTypeFeedbackForBytecodeOffset(li->bytecode_offset()))
    return nullptr;
  ClassDistributionSummary summary =
      *f.GetTypeFeedbackForBytecodeOffset(li->bytecode_offset());
  Class* cls = summary.MonomorphicClass(options.mgr);
  if (!cls || cls->invalid()) {
    return nullptr;
  }
  int64_t names_index = li->index();

  PyObject* name_obj = PyTuple_GET_ITEM(code->co_names, names_index);
  absl::string_view name = GetObjectAsCheapString(name_obj);
  if (name.empty()) return nullptr;

  auto it = cls->attributes().find(options.mgr.InternString(name));
  if (it == cls->attributes().end()) {
    EventCounters::Instance().Add(
        absl::StrCat("optimizer.globals.no_attribute:", name), 1);
    return nullptr;
  }

  const Attribute* attribute = it->second.get();
  if (attribute->value() == nullptr) {
    // Value is not known to be constant.
    EventCounters::Instance().Add("optimizer.globals.not_constant", 1);
    return nullptr;
  }
  // TODO: Make LoadGlobalInst a SafepointInst.
  auto safepoint_or = BeginningOfBytecodeInstruction(li);
  if (!safepoint_or.ok()) {
    EventCounters::Instance().Add("optimizer.globals.no_safepoint", 1);
    return nullptr;
  }

  Builder builder(li->parent(), li->GetIterator());
  Value* instance_id = builder.GetInstanceClassId(
      builder.FrameVariable(FrameVariableInst::FrameVariableKind::kGlobals));
  builder.DeoptimizeIfSafepoint(
      builder.IsNotEqual(instance_id, builder.Int64(cls->id())),
      /*negated=*/false, "Globals dict has been modified",
      safepoint_or.value());
  Value* result = builder.ConstantAttribute(cls, attribute->name());

  if (options.use_event_counters) {
    builder.IncrementEventCounter("optimizer.globals.speculate");
  }
  f.AddReliedUponClass(cls);
  builder.DeoptimizedAsynchronously(safepoint_or.value());

  return result;
}

// Transforms `load_global` either into a speculated constant based on the
// globals class or a slow path.
void TransformLoadGlobal(Function& f, LoadGlobalInst* li, UseLists& uses,
                         PyCodeObject* code, const OptimizerOptions& options) {
  Builder b(li->parent(), li->GetIterator());

  Value* result = EmitLoadGlobalSpeculated(b, f, li, code, options);
  if (!result) {
    result = EmitLoadGlobalNoCache(b, li->index(), options);
  }

  // Use ReplaceAllUsesWith to replace uses of `ci` with the result of the
  // conditional.
  ReplaceAllUsesWith(uses, li, result);
  li->erase();
}

absl::Status OptimizeLoadGlobal(Function& f, PyCodeObject* code,
                                OptimizerOptions options) {
  // Precompute the use lists.
  auto uses = ComputeUses(f);

  for (auto cursor = f.FirstInstruction(); !cursor.Finished();
       cursor.StepForward()) {
    LoadGlobalInst* li = dyn_cast<LoadGlobalInst>(cursor.GetInstruction());
    if (!li) continue;
    TransformLoadGlobal(f, li, uses, code, options);
  }

  return absl::OkStatus();
}

namespace {

struct OptimizableAttributeLoadStoreInfo {
  bool valid = false;
  const Class* initial_class = nullptr;
  const Class* final_class = nullptr;
  const Attribute* attribute = nullptr;
};

// A store operation may possible transition a class. Is this optimizable?
OptimizableAttributeLoadStoreInfo IsOptimizableAttributeStoreTransition(
    InternedString name, const Class* cls, ClassManager& mgr,
    Value* stored_value, PyCodeObject* code) {
  PyObject* value = GetValueAsConstantObject(stored_value, code);

  // Look up the transition for the addition of `name`. We first look up a
  // transition for a combined dict. If that fails, we look up for split dict.
  // Once a type has started creating combined dicts, it never switches back to
  // split dicts.
  AttributeDescription attr_description(
      name, value, AttributeDescription::InstanceOrType::kInstance);

  DictKind dict_kind = DictKind::kCombined;
  ClassTransition transition =
      ClassTransition::Add(attr_description, dict_kind);
  const Class* new_cls = cls->LookupTransition(transition);
  if (!new_cls) {
    dict_kind = DictKind::kSplit;
    ClassTransition transition =
        ClassTransition::Add(attr_description, dict_kind);
    new_cls = cls->LookupTransition(transition);

    if (!new_cls) {
      return {};
    }
  }

  auto it = new_cls->attributes().find(name);
  S6_CHECK(it != new_cls->attributes().end());
  const Attribute* attr = it->second.get();
  if (attr->value() != value) {
    return {};
  }

  // We know the transitioned-to class and we know the stored value won't
  // invalidate the class.
  return OptimizableAttributeLoadStoreInfo{.valid = true,
                                           .initial_class = cls,
                                           .final_class = new_cls,
                                           .attribute = attr};
}

// Given type feedback for an attribute load or store (GetAttr/SetAttr),
// determines if this is optimizable by a simple dict load or store.
absl::StatusOr<std::vector<OptimizableAttributeLoadStoreInfo>>
IsOptimizableAttributeLoadOrStore(const ClassDistributionSummary& summary,
                                  InternedString name, bool is_store,
                                  ClassManager& mgr,
                                  Value* stored_value = nullptr,
                                  PyCodeObject* code = nullptr) {
  std::vector<OptimizableAttributeLoadStoreInfo> no_infos;
  if (!summary.IsMonomorphic() && !summary.IsPolymorphic()) return no_infos;
  if (!summary.stable()) return no_infos;

  std::vector<OptimizableAttributeLoadStoreInfo> infos;
  // For all classes we are going to specialize over, does the attribute exist?
  // and is it unknown? we don't want to optimize sites that cause a class
  // transition.
  for (int32_t class_id : summary.common_class_ids()) {
    const Class* cls = mgr.GetClassById(class_id);
    S6_RET_CHECK(cls) << "wanted class " << class_id;
    auto it = cls->attributes().find(name);
    if (it == cls->attributes().end()) {
      // This class does not already have this attribute. This is either a
      // lookup failure or a store that causes a transition.
      if (!is_store) {
        EventCounters::Instance().Add("optimizer.loadstore.no_attribute", 1);
        return no_infos;
      }
      infos.push_back(IsOptimizableAttributeStoreTransition(
          name, cls, mgr, stored_value, code));
      if (!infos.back().valid) return no_infos;
      continue;
    }
    const Attribute* attr = it->second.get();
    if (attr->value() && is_store) {
      if (attr->value() != GetValueAsConstantObject(stored_value, code)) {
        EventCounters::Instance().Add("optimizer.loadstore.value_not_equal", 1);
        // This attribute has constant known value, at the moment. Optimizing
        // this program point may cause us to write a differing value here and
        // cause the class to change.
        return no_infos;
      }
    }

    if (attr->IsDescriptor() && cls->is_type_class()) {
      // The descriptor logic for type classes is nontrivial; if the attribute
      // came from the type itself, it is not bound, but if it came from a
      // metaclass, it is. We'll add this behavior later, but for now don't
      // optimize as we don't know here whether we should prepend self or not.
      return no_infos;
    }

    if (cls->GetInstanceSlot(*attr) == -1) {
      // This attribute does not have a slot within the instance dict.
      EventCounters::Instance().Add("optimizer.loadstore.no_instance_slot", 1);
      return no_infos;
    }
    if (cls->dict_kind() != DictKind::kSplit &&
        cls->dict_kind() != DictKind::kCombined) {
      EventCounters::Instance().Add(
          "optimizer.loadstore.empty_or_noncontiguous", 1);
      return no_infos;
    }

    infos.push_back(OptimizableAttributeLoadStoreInfo{
        .valid = true, .initial_class = cls, .attribute = attr});
  }
  return infos;
}

int64_t GetCommonDictOffset(absl::Span<int32_t const> class_ids,
                            ClassManager& mgr) {
  int64_t common_dictoffset = -1;
  for (int32_t class_id : class_ids) {
    const Class* cls = mgr.GetClassById(class_id);
    if (common_dictoffset == -1) {
      common_dictoffset = cls->dictoffset();
    } else if (common_dictoffset != cls->dictoffset()) {
      return 0;
    }
  }
  return common_dictoffset;
}

// Transforms a fast attribute load or store from an object dict. Emits:
//
// Monomorphic case:
//   %dict = get_object_dict %receiver
//   %check = check_class_instance %dict, $class_id
//   deoptimize_if_safepoint not %check, ...
//   %result = load_from_split_dict %dict, $index
//
// Polymorphic case:
//   %dict = get_object_dict %receiver
//   %check1 = check_class_instance %dict, $class_id1
//   br %check1, &success, &fail
//  &success:
//   %result1 = load_from_split_dict %dict, $index1
//   jmp &out [%result1]
//  &fail:
//   //... repeat initial check for all polymorphic classes ...
//
//  &fail_again:
//   deoptimize_if_safepoint...
// &out: [&result]
//   ...
absl::Status OptimizeAttributeLoad(Function& f, CallNativeInst* getset_call,
                                   absl::string_view name_raw,
                                   OptimizerOptions options) {
  ClassManager& mgr = options.mgr;
  InternedString name = mgr.InternString(name_raw);

  if (!f.GetTypeFeedbackForBytecodeOffset(getset_call->bytecode_offset()))
    return {};

  ClassDistributionSummary summary =
      *f.GetTypeFeedbackForBytecodeOffset(getset_call->bytecode_offset());
  S6_ASSIGN_OR_RETURN(
      std::vector<OptimizableAttributeLoadStoreInfo> infos,
      IsOptimizableAttributeLoadOrStore(summary, name, false, mgr));
  if (infos.empty()) return {};

  UseLists uses = ComputeUses(f);
  S6_ASSIGN_OR_RETURN(SafepointInst * safepoint,
                      BeginningOfBytecodeInstruction(getset_call));

  int64_t common_dictoffset =
      GetCommonDictOffset(summary.common_class_ids(), mgr);
  Value* receiver = getset_call->operands()[0];
  Builder builder(getset_call->parent(), getset_call->GetIterator());
  Value* dict = builder.GetObjectDict(receiver, common_dictoffset);
  builder.DeoptimizeIfSafepoint(dict, /*negated=*/true,
                                "Specialized instance attribute load received "
                                "an object without __dict__",
                                safepoint);

  auto emit_load = [&](Builder builder, int32_t class_id) -> Value* {
    Class* cls = mgr.GetClassById(class_id);
    S6_CHECK(cls);
    f.AddReliedUponClass(cls);
    builder.DeoptimizedAsynchronously(safepoint);

    int64_t index = cls->GetInstanceSlot(*cls->attributes().at(name));
    S6_CHECK_GE(index, 0);
    return builder.LoadFromDict(dict, index, cls->dict_kind());
  };

  std::string reason_prefix = absl::StrCat(
      "Specialized instance attribute load (of attribute ", name_raw, ")");

  // The code we emit for monomorphic, polymorphic and skewed megamorphic is
  // similar but sufficiently different that it's easier to emit them
  // separately.
  Value* result = nullptr;
  if (summary.IsMonomorphic()) {
    int32_t class_id = summary.common_class_ids().front();
    Value* instance_id = builder.GetInstanceClassId(dict);
    builder.DeoptimizeIfSafepoint(
        builder.IsNotEqual(instance_id, builder.Int64(class_id)),
        /*negated=*/false,
        absl::StrCat(reason_prefix, " was incorrectly monomorphic on class ",
                     summary.MonomorphicClass(options.mgr)->name()),
        safepoint);
    result = emit_load(builder, class_id);
  } else {
    S6_CHECK(summary.IsPolymorphic())
        << "Skewed megamorphic not yet implemented!";
    Value* instance_id = builder.GetInstanceClassId(dict);

    result = builder.Switch<int32_t>(
        summary.common_class_ids(), safepoint,
        [&](Builder b, int32_t class_id) -> Value* {
          return b.IsEqual(instance_id, b.Int64(class_id));
        },
        emit_load, absl::StrCat(reason_prefix, " was incorrectly polymorphic"));
  }

  builder.IncrefNotNull(result);
  ReplaceAllUsesWith(uses, getset_call, result);
  getset_call->erase();
  return absl::OkStatus();
}

absl::Status OptimizeAttributeStore(Function& f, CallNativeInst* getset_call,
                                    absl::string_view name_raw,
                                    OptimizerOptions options,
                                    PyCodeObject* code) {
  ClassManager& mgr = options.mgr;
  InternedString name = mgr.InternString(name_raw);

  if (!f.GetTypeFeedbackForBytecodeOffset(getset_call->bytecode_offset()))
    return {};

  ClassDistributionSummary summary =
      *f.GetTypeFeedbackForBytecodeOffset(getset_call->bytecode_offset());
  S6_ASSIGN_OR_RETURN(
      std::vector<OptimizableAttributeLoadStoreInfo> infos,
      IsOptimizableAttributeLoadOrStore(summary, name, true, mgr,
                                        getset_call->operands()[2], code));
  if (infos.empty()) return {};

  UseLists uses = ComputeUses(f);
  S6_ASSIGN_OR_RETURN(SafepointInst * safepoint,
                      BeginningOfBytecodeInstruction(getset_call));

  int64_t common_dictoffset =
      GetCommonDictOffset(summary.common_class_ids(), mgr);
  Value* receiver = getset_call->operands()[0];
  Builder builder(getset_call->parent(), getset_call->GetIterator());
  Value* dict = builder.GetObjectDict(receiver, common_dictoffset);

  std::string reason_prefix = absl::StrCat(
      "Specialized instance attribute store (of attribute ", name_raw, ")");

  auto emit_store =
      [&](Builder builder,
          const OptimizableAttributeLoadStoreInfo& info) -> Value* {
    const Class* cls = info.initial_class;
    S6_CHECK(cls);
    f.AddReliedUponClass(cls);
    builder.DeoptimizedAsynchronously(safepoint);

    Value* value = getset_call->operands()[2];
    if (!info.final_class) {
      // Simple set of attribute with no transition.
      builder.DeoptimizeIfSafepoint(dict, /*negated=*/true,
                                    "Specialized instance attribute store "
                                    "received an object without __dict__",
                                    safepoint);
      int64_t index = cls->GetInstanceSlot(*cls->attributes().at(name));
      S6_CHECK_GE(index, 0);
      builder.IncrefNotNull(value);
      builder.DecrefOrNull(
          builder.StoreToDict(value, dict, index, cls->dict_kind()),
          safepoint->bytecode_offset());
      return nullptr;
    }

    if (info.initial_class->dict_kind() == DictKind::kEmpty) {
      // The dict was empty, so it must be initialized.
      Value* name = getset_call->operands()[1];
      Value* result = builder.Call(
          Callee::kInitializeObjectDict,
          {receiver, name, value, builder.Constant(info.final_class)},
          getset_call->bytecode_offset());
      builder.DeoptimizeIfSafepoint(
          result, /*negated=*/true,
          "Expected split dict but obtained combined dict.", safepoint);
      return nullptr;
    }

    // Otherwise there is a class transition. We cannot statically determine
    // at this point if there are enough empty slots in general, and neither
    // do we want to bake in such implementation details as the hash bucket
    // location for the key, so we call PyDict_SetItem here.
    Value* name = getset_call->operands()[1];
    builder.Call(Callee::kPyDict_SetItem, {dict, name, value},
                 getset_call->bytecode_offset());
    // PyDict_SetItem does *not* steal a reference to `value`.
    builder.SetObjectClass(receiver, dict, info.final_class->id());
    return nullptr;
  };

  // The code we emit for monomorphic, polymorphic and skewed megamorphic is
  // similar but sufficiently different that it's easier to emit them
  // separately.
  if (summary.IsMonomorphic()) {
    const OptimizableAttributeLoadStoreInfo& info = infos.front();
    if (info.initial_class->dict_kind() == DictKind::kEmpty) {
      // We don't expect the object dict to have anything. Deoptimize if it's
      // non-nullptr or the type doesn't match.
      Value* type_not_ok =
          builder.IsNotEqual(builder.GetType(receiver),
                             builder.Constant(info.initial_class->type()));
      builder.DeoptimizeIfSafepoint(
          builder.Or(builder.IsNotZero(dict), type_not_ok),
          /*negated=*/false,
          absl::StrCat(reason_prefix, " expected an empty __dict__"),
          safepoint);
    } else {
      builder.DeoptimizeIfSafepoint(
          dict, /*negated=*/true,
          "Specialized instance attribute store received "
          "an object without __dict__",
          safepoint);
      Value* instance_id = builder.GetInstanceClassId(dict);
      builder.DeoptimizeIfSafepoint(
          builder.IsNotEqual(instance_id,
                             builder.Int64(info.initial_class->id())),
          /*negated=*/false,
          absl::StrCat(reason_prefix, " was incorrectly monomorphic on class ",
                       summary.MonomorphicClass(options.mgr)->name()),
          safepoint);
    }
    emit_store(builder, info);
  } else {
    S6_CHECK(summary.IsPolymorphic())
        << "Skewed megamorphic not yet implemented!";
    builder.DeoptimizeIfSafepoint(
        dict, /*negated=*/true,
        "Specialized instance attribute store received "
        "an object without __dict__",
        safepoint);
    Value* instance_id = builder.GetInstanceClassId(dict);
    builder.Switch<OptimizableAttributeLoadStoreInfo>(
        infos, safepoint,
        [&](Builder b,
            const OptimizableAttributeLoadStoreInfo& info) -> Value* {
          if (info.initial_class->dict_kind() == DictKind::kEmpty) {
            // We don't expect the object dict to have anything. Deoptimize if
            // it's non-nullptr or the type doesn't match.
            builder.DeoptimizeIfSafepoint(
                builder.IsNotZero(dict),
                /*negated=*/false,
                absl::StrCat(reason_prefix, " expected an empty __dict__"),
                safepoint);
            return builder.And(
                builder.IsEqual(builder.GetType(receiver),
                                builder.Constant(info.initial_class->type())),
                builder.IsZero(dict));
          }
          return b.IsEqual(instance_id, b.Int64(info.initial_class->id()));
        },
        emit_store,
        absl::StrCat(reason_prefix, " was incorrectly polymorphic"));
  }

  // "return" zero for success.
  ReplaceAllUsesWith(uses, getset_call, builder.Zero());
  getset_call->erase();
  return absl::OkStatus();
}
}  // namespace

const FunctionAttribute* LookupFunction(ClassManager& mgr, const Class* cls,
                                        absl::string_view name) {
  auto it = cls->attributes().find(mgr.InternString(name));
  if (it == cls->attributes().end()) {
    return nullptr;
  }
  if (it->second->kind() != Attribute::kFunction) {
    return nullptr;
  }
  // TODO: Plug Attribute into the isa/dyn_cast machinery.
  return static_cast<const FunctionAttribute*>(it->second.get());
}

const CFunctionAttribute* LookupCFunction(ClassManager& mgr, const Class* cls,
                                          absl::string_view name) {
  auto it = cls->attributes().find(mgr.InternString(name));
  if (it == cls->attributes().end()) {
    return nullptr;
  }
  if (it->second->kind() != Attribute::kCFunction) {
    return nullptr;
  }
  // TODO: Plug Attribute into the isa/dyn_cast machinery.
  return static_cast<const CFunctionAttribute*>(it->second.get());
}

absl::Status SpeculateCallsPattern::Apply(Value* value,
                                          Rewriter& rewriter) const {
  CallAttributeInst& call = cast<CallAttributeInst>(*value);

  absl::optional<ClassDistributionSummary> summary =
      rewriter.function().GetTypeFeedbackForBytecodeOffset(
          call.bytecode_offset());
  if (!summary.has_value())
    return absl::FailedPreconditionError("no type information");
  if (!summary->IsMonomorphic() && !summary->IsPolymorphic()) {
    return absl::FailedPreconditionError("not mono or polymorphic");
  }

  S6_ASSIGN_OR_RETURN(SafepointInst * safepoint,
                      BeginningOfBytecodeInstruction(&call));

  ClassManager& mgr = rewriter.class_manager();
  std::vector<std::pair<const Class*, bool>> attributes;
  for (int32_t id : summary->common_class_ids()) {
    const Class* cls = mgr.GetClassById(id);
    S6_CHECK(cls);
    if (cls->invalid()) return absl::FailedPreconditionError("invalid class");
    const FunctionAttribute* attr =
        LookupFunction(mgr, cls, call.attribute_str());
    if (attr && attr->value()) {
      if (cls->is_type_class()) {
        // The descriptor logic for type classes is nontrivial; if the attribute
        // came from the type itself, it is not bound, but if it came from a
        // metaclass, it is. We'll add this behavior later, but for now don't
        // optimize as we don't know here whether we should prepend self or not.
        return absl::FailedPreconditionError(
            "at least one class is a type class");
      }
      attributes.emplace_back(cls, attr->bound());
      continue;
    }
    const CFunctionAttribute* c_attr =
        LookupCFunction(mgr, cls, call.attribute_str());
    if (c_attr && c_attr->value()) {
      attributes.emplace_back(cls, false);
      continue;
    }
    return absl::FailedPreconditionError(
        "at least one class did not have optimizable methods");
  }

  std::vector<Value*> original_operands(call.call_arguments().begin(),
                                        call.call_arguments().end());
  // The operands to use if the function is bound (i.e. a method call of a
  // PyFunctionObject descriptor). This has `self` prepended.
  std::vector<Value*> bound_operands(original_operands.begin(),
                                     original_operands.end());
  bound_operands.insert(bound_operands.begin(), call.object());

  std::string reason_prefix = absl::StrCat(
      "Call of attribute ", call.attribute_str(), " was speculated to be ");

  Builder builder = rewriter.CreateBuilder(call.GetIterator());
  // We're about to emit code that assumes the behavior of classes, so bail out
  // if those have been broken (if the class has been modified).
  builder.DeoptimizedAsynchronously(safepoint);
  if (summary->IsMonomorphic()) {
    const Class* cls = attributes.front().first;
    Value* class_okay = builder.CheckClassId(call.object(), cls);
    builder.DeoptimizeIfSafepoint(
        class_okay, /*negated=*/true,
        absl::StrCat(reason_prefix, "monomorphic on class ", cls->name(),
                     " incorrectly"),
        safepoint);
    auto& operands =
        attributes.front().second ? bound_operands : original_operands;
    Value* v = builder.CallConstantAttribute(
        cls, call.attribute_str(), operands, call.names(),
        call.call_python_bytecode_offset());
    if (!attributes.front().second) {
      // If the receiver object is not bound, we must decref it.
      builder.DecrefNotNull(call.object(), call.call_python_bytecode_offset());
    }
    rewriter.AddReliedUponClass(cls);
    rewriter.ReplaceAllUsesWith(call, *v);
    rewriter.erase(call);
    return absl::OkStatus();
  }

  Value* class_id = builder.GetClassId(call.object());
  Value* result = builder.Switch<std::pair<const Class*, bool>>(
      attributes, safepoint,
      [&](Builder b, const auto& cls_and_attr) {
        return b.IsEqual(class_id, b.Int64(cls_and_attr.first->id()));
      },
      [&](Builder b, const auto& cls_and_attr) {
        rewriter.AddReliedUponClass(cls_and_attr.first);
        auto& operands =
            cls_and_attr.second ? bound_operands : original_operands;
        Value* v = b.CallConstantAttribute(
            cls_and_attr.first, call.attribute_str(), operands, call.names(),
            call.call_python_bytecode_offset());
        if (!cls_and_attr.second) {
          // If the receiver object is not bound, we must decref it.
          b.DecrefNotNull(call.object(), call.call_python_bytecode_offset());
        }
        return v;
      },
      absl::StrCat(reason_prefix, "polymorphic, incorrectly"));
  rewriter.ReplaceAllUsesWith(call, *result);
  rewriter.erase(call);
  return absl::OkStatus();
}

// Attempts to merge positional and keyword arguments, plus defaults, into a
// single argument list. Returns nullopt on failure.
absl::optional<std::vector<Value*>> RemapKeywordArguments(
    Builder& builder, absl::Span<Value* const> positional_arguments,
    absl::Span<Value* const> keyword_argument_values,
    absl::Span<absl::string_view const> keyword_argument_names,
    const FunctionAttribute& attr) {
  const int64_t argcount = attr.code()->co_argcount;
  const bool varargs = attr.code()->co_flags & CO_VARARGS;
  const bool varkeywords = attr.code()->co_flags & CO_VARKEYWORDS;
  if (positional_arguments.size() > argcount && !varargs) {
    return {};
  }

  std::vector<Value*> args;
  args.resize(argcount + varargs + varkeywords);

  // Positional arguments come first.
  int64_t n = std::min<int64_t>(positional_arguments.size(), argcount);
  absl::c_copy_n(positional_arguments, n, args.begin());

  // Pack the rest into *args.
  std::vector<Value*> to_pack_varargs;
  if (n != positional_arguments.size()) {
    S6_CHECK(varargs);
    to_pack_varargs.reserve(n);
    absl::c_copy(positional_arguments.subspan(n),
                 std::back_inserter(to_pack_varargs));
  }

  // Any keyword arguments come next.
  S6_CHECK_EQ(keyword_argument_names.size(), keyword_argument_values.size());
  for (int64_t i = 0; i < keyword_argument_names.size(); ++i) {
    auto it = attr.argument_names().find(keyword_argument_names[i]);
    if (it == attr.argument_names().end()) {
      // TODO: Handle varkeywords.
      return {};
    }
    int64_t index = it->second;
    if (index >= argcount) {
      // TODO: Understand why this can happen.
      return {};
    }
    if (args[index]) return {};
    args[index] = keyword_argument_values[i];
  }

  // Then positional defaults.
  for (int64_t default_i = 0; default_i < attr.defaults().size(); ++default_i) {
    int64_t arg_i = argcount - attr.defaults().size() + default_i;
    if (args[arg_i]) continue;
    args[arg_i] = builder.Constant(attr.defaults()[default_i]);
    builder.IncrefNotNull(args[arg_i]);
  }

  // Now that there's no going back, if there's a *args tuple to make, make it
  // now.
  if (varargs) {
    Value* tuple = builder.Call(Callee::kPyTuple_New,
                                {builder.Int64(to_pack_varargs.size())});
    for (int64_t i = 0; i < to_pack_varargs.size(); ++i) {
      // Steal the reference.
      builder.TupleSetItem(to_pack_varargs[i], tuple, i);
    }
    // We made room for this earlier.
    args[argcount] = tuple;
  }

  if (varkeywords) {
    // Just make an empty dict for now. The keywords loop doesn't yet handle
    // extra kwargs.
    Value* dict = builder.Call(Callee::kPyDict_New, {});
    // We made room for this earlier.
    args[argcount + varargs] = dict;
  }

  // Now all arguments should be filled in or we made a bad mistake.
  S6_CHECK(absl::c_all_of(args, [](Value* p) { return p != nullptr; }));

  return args;
}

absl::Status ApplyFastcallPattern::ApplyKeywordArgumentRemapping(
    CallPythonInst& call, Rewriter& rewriter, Builder& builder,
    const FunctionAttribute& method_attr) const {
  std::vector<absl::string_view> keyword_names;
  if (call.names()) {
    auto keyword_names_opt =
        GetValueAsConstantTupleOfStrings(call.names(), rewriter.code_object());
    if (!keyword_names_opt.has_value()) {
      return absl::FailedPreconditionError("Invalid names tuple");
    }
    keyword_names = std::move(*keyword_names_opt);
  }

  int64_t num_keyword = keyword_names.size();
  int64_t num_positional = call.call_arguments().size() - num_keyword;
  absl::optional<std::vector<Value*>> final_arguments = RemapKeywordArguments(
      builder, call.call_arguments().first(num_positional),
      call.call_arguments().last(num_keyword), keyword_names, method_attr);
  if (!final_arguments.has_value()) {
    return absl::FailedPreconditionError(
        "Couldn't map keyword arguments or defaults");
  }

  *call.mutable_names() = nullptr;
  call.mutable_call_arguments().resize(final_arguments->size());
  absl::c_copy(*final_arguments, call.mutable_call_arguments().begin());
  EventCounters::Instance().Add("optimizer.fastcall.keywords", 1);
  return absl::OkStatus();
}

absl::Status ApplyFastcallPattern::Apply(Value* value,
                                         Rewriter& rewriter) const {
  CallPythonInst& call = cast<CallPythonInst>(*value);
  if (call.fastcall()) return absl::FailedPreconditionError("already fastcall");
  ConstantAttributeInst* constant_attribute =
      dyn_cast<ConstantAttributeInst>(call.callee());
  if (!constant_attribute)
    return absl::FailedPreconditionError("No constant_attribute");

  const Attribute& attr =
      constant_attribute->LookupAttribute(rewriter.class_manager());
  if (attr.kind() != Attribute::kFunction)
    return absl::FailedPreconditionError("Not a bound method");
  const FunctionAttribute& method_attr =
      static_cast<const FunctionAttribute&>(attr);

  if (method_attr.code()->co_kwonlyargcount > 0) {
    return absl::FailedPreconditionError("Uses keyword-only args.");
  }

  Builder builder = rewriter.CreateBuilder(&call);
  PyCodeObject* co = method_attr.code();

  S6_RETURN_IF_ERROR(
      ApplyKeywordArgumentRemapping(call, rewriter, builder, method_attr));

  if (co->co_flags & CO_GENERATOR) {
    std::vector<Value*> operands(1, builder.Constant(method_attr.value()));
    operands.push_back(
        builder.FrameVariable(FrameVariableInst::FrameVariableKind::kBuiltins));
    absl::c_copy(call.call_arguments(), std::back_inserter(operands));
    Value* runtime_call = builder.Call(Callee::kCreateGenerator, operands,
                                       call.bytecode_offset());
    builder.DecrefNotNull(constant_attribute, call.bytecode_offset());
    rewriter.ReplaceAllUsesWith(call, *runtime_call);
    rewriter.erase(call);
    return absl::OkStatus();
  }

  call.set_fastcall(true);
  return absl::OkStatus();
}

absl::Status OptimizeGetSetAttr(Function& f, PyCodeObject* code,
                                OptimizerOptions options) {
  UseLists uses = ComputeUses(f);
  for (auto cursor = f.FirstInstruction(); !cursor.Finished();
       cursor.StepForward()) {
    CallNativeInst* ci = dyn_cast<CallNativeInst>(cursor.GetInstruction());
    if (ci == nullptr || !ci->CalleeIsAnyOf({Callee::kPyObject_GetAttr,
                                             Callee::kPyObject_SetAttr}))
      continue;

    auto name = GetValueAsConstantString(ci->operands()[1], code);
    if (!name.has_value()) continue;
    if (ci->CalleeIs(Callee::kPyObject_GetAttr)) {
      S6_RETURN_IF_ERROR(OptimizeAttributeLoad(f, ci, *name, options));
    } else {
      S6_RETURN_IF_ERROR(OptimizeAttributeStore(f, ci, *name, options, code));
    }
  }
  return absl::OkStatus();
}

namespace {

bool IsUnboxable(CallNativeInst* inst, UnboxableType type) {
  switch (inst->callee()) {
    case Callee::kPyNumber_MatrixMultiply:
    case Callee::kPyNumber_InPlaceMatrixMultiply:
      return false;

    case Callee::kPyNumber_And:
    case Callee::kPyNumber_InPlaceAnd:
    case Callee::kPyNumber_FloorDivide:
    case Callee::kPyNumber_InPlaceFloorDivide:
    case Callee::kPyNumber_Invert:
    case Callee::kPyNumber_Lshift:
    case Callee::kPyNumber_InPlaceLshift:
    case Callee::kPyNumber_Or:
    case Callee::kPyNumber_InPlaceOr:
    case Callee::kPyNumber_Remainder:
    case Callee::kPyNumber_InPlaceRemainder:
    case Callee::kPyNumber_Rshift:
    case Callee::kPyNumber_InPlaceRshift:
    case Callee::kPyNumber_Xor:
    case Callee::kPyNumber_InPlaceXor:
      // We only have integer optimisations for these.
      return type != UnboxableType::kPyFloat;

    case Callee::kPyNumber_Power:
    case Callee::kPyNumber_InPlacePower: {
      // Only optimise if the exponent is 2.
      Value* exponent = inst->operands()[1];
      ConstantInst* constExponent = dyn_cast<ConstantInst>(exponent);
      return constExponent != nullptr && constExponent->value() == 2;
    }

    case Callee::kPyObject_RichCompare:
      return true;

    default:
      return inst->callee() >= Callee::kPyNumber_Add &&
             inst->callee() <= Callee::kPyNumber_Xor;
  }
}

absl::StatusOr<Value*> UnboxedInst(
    Builder& builder, Callee callee, UnboxableType unboxable_type,
    const absl::InlinedVector<Value*, 3>& operands, SafepointInst* safepoint) {
  // Determine the operand type of the strongjit instruction.
  // PyBools are treated as int64_t via the convention that all bits
  // are equal, so True is represented as -1.
  NumericInst::NumericType type = unboxable_type == UnboxableType::kPyFloat
                                      ? NumericInst::kDouble
                                      : NumericInst::kInt64;

  // Treat 'InPlace' variants as synonymous with their non-in-place
  // counterparts. This is valid because none of PyLong, PyFloat, or PyBool
  // implement in-place functions `__iadd__` etc.
  switch (callee) {
    case Callee::kPyNumber_Add:
    case Callee::kPyNumber_InPlaceAdd:
      return builder.Add(type, operands[0], operands[1], safepoint);
    case Callee::kPyNumber_And:
    case Callee::kPyNumber_InPlaceAnd:
      S6_RET_CHECK(type == NumericInst::kInt64) << "Bad operand type for AND!";
      return builder.And(operands[0], operands[1]);
    case Callee::kPyNumber_FloorDivide:
    case Callee::kPyNumber_InPlaceFloorDivide:
      S6_RET_CHECK(type == NumericInst::kInt64)
          << "Bad operand type for floordiv";
      return builder.Divide(type, operands[0], operands[1], safepoint);
    case Callee::kPyNumber_Invert:
      S6_RET_CHECK(type == NumericInst::kInt64) << "Bad operand type for NOT!";
      return builder.Not(operands[0]);
    case Callee::kPyNumber_Lshift:
    case Callee::kPyNumber_InPlaceLshift:
      S6_RET_CHECK(type == NumericInst::kInt64)
          << "Bad operand type for shift!";
      return builder.ShiftLeft(operands[0], operands[1], safepoint);
    case Callee::kPyNumber_Multiply:
    case Callee::kPyNumber_InPlaceMultiply:
      return builder.Multiply(type, operands[0], operands[1], safepoint);
    case Callee::kPyNumber_Negative:
      return builder.Negate(type, operands[0], safepoint);
    case Callee::kPyNumber_Positive:
      // "Unary positive" is a no-op for primitive types.
      return operands[0];
    case Callee::kPyNumber_Or:
    case Callee::kPyNumber_InPlaceOr:
      S6_RET_CHECK(type == NumericInst::kInt64) << "Bad operand type for OR!";
      return builder.Or(operands[0], operands[1]);
    case Callee::kPyNumber_Power:
    case Callee::kPyNumber_InPlacePower:
      // Power is unboxed only for squaring.
      return builder.Multiply(type, operands[0], operands[0], safepoint);
    case Callee::kPyNumber_Remainder:
    case Callee::kPyNumber_InPlaceRemainder:
      S6_RET_CHECK(type == NumericInst::kInt64)
          << "Bad operand type for remainder!";
      return builder.Remainder(type, operands[0], operands[1], safepoint);
    case Callee::kPyNumber_Rshift:
    case Callee::kPyNumber_InPlaceRshift:
      S6_RET_CHECK(type == NumericInst::kInt64)
          << "Bad operand type for shift!";
      return builder.ShiftRightSigned(operands[0], operands[1], safepoint);
    case Callee::kPyNumber_Subtract:
    case Callee::kPyNumber_InPlaceSubtract:
      return builder.Subtract(type, operands[0], operands[1], safepoint);
    case Callee::kPyNumber_TrueDivide:
    case Callee::kPyNumber_InPlaceTrueDivide:
      if (type == NumericInst::kInt64) {
        // Convert integer arguments to float64, and do a float64 division.
        BlockInserter inserter = builder.inserter();
        Value* float_lhs = inserter.Create<IntToFloatInst>(operands[0]);
        Value* float_rhs = inserter.Create<IntToFloatInst>(operands[1]);
        return builder.Divide(NumericInst::kDouble, float_lhs, float_rhs,
                              safepoint);
      } else {
        return builder.Divide(type, operands[0], operands[1], safepoint);
      }
    case Callee::kPyNumber_Xor:
    case Callee::kPyNumber_InPlaceXor:
      S6_RET_CHECK(type == NumericInst::kInt64) << "Bad operand type for XOR!";
      return builder.Xor(operands[0], operands[1]);

    case Callee::kPyObject_RichCompare: {
      // Expect the third argument (predicate condition) to have been
      // supplied by an immediate constant.
      ConstantInst* condition = dyn_cast<ConstantInst>(operands[2]);
      S6_RET_CHECK(condition != nullptr)
          << "Non-constant expression received"
             " for PyObject_RichCompare predicate!";
      switch (condition->value()) {
        case Py_EQ:
          return builder.IsEqual(type, operands[0], operands[1]);
        case Py_NE:
          return builder.IsNotEqual(type, operands[0], operands[1]);
        case Py_LT:
          return builder.IsLessThan(type, operands[0], operands[1]);
        case Py_LE:
          return builder.IsLessEqual(type, operands[0], operands[1]);
        case Py_GT:
          return builder.IsGreaterThan(type, operands[0], operands[1]);
        case Py_GE:
          return builder.IsGreaterEqual(type, operands[0], operands[1]);
        default:
          S6_RET_CHECK_FAIL() << "Unknown predicate condition";
      }
    }

    default:
      S6_RET_CHECK_FAIL() << "Unknown python callee " << ToString(callee);
  }
}

absl::Status RemovePyNumberResultChecks(CallNativeInst* pynum_op,
                                        Rewriter& rewriter) {
  Block::iterator it = pynum_op->GetIterator();
  ++it;

  // The next instruction should be to decref the inputs.
  // This will be retained after the new unbox-op-box instructions.
  while (true) {
    DecrefInst* decref_op = dyn_cast<DecrefInst>(&*++it);
    if (!decref_op) break;
    if (absl::c_linear_search(pynum_op->call_arguments(),
                              decref_op->operand())) {
      continue;
    }
    return absl::OkStatus();
  }

  if (ConstantInst* zero_op = dyn_cast<ConstantInst>(&*it);
      zero_op && zero_op->value() == 0) {
    ++it;

    // There may be a non-zero validation check on the return value.
    CompareInst* cmp_op = dyn_cast<CompareInst>(&*it);
    S6_RET_CHECK(cmp_op && cmp_op->comparison() == CompareInst::kEqual &&
                 cmp_op->lhs() == pynum_op && cmp_op->rhs() == zero_op)
        << "Malformed call_native validation check!";
    ++it;

    BrInst* br_op = dyn_cast<BrInst>(&*it);
    S6_RET_CHECK(br_op != nullptr);
    S6_RET_CHECK(br_op->condition() == cmp_op);
    it = br_op->false_successor()->begin();

    // Remove this validation check, as we trust that our replacement
    // (the `box` op) will be successful.
    rewriter.ConvertBranchToJump(*br_op, false);
    rewriter.erase(*cmp_op);
    rewriter.erase(*zero_op);
  }

  return absl::OkStatus();
}

absl::optional<UnboxableType> UnboxableTypeOf(const Class* cls) {
  if (cls->type() == &PyLong_Type) {
    return UnboxableType::kPyLong;
  } else if (cls->type() == &PyFloat_Type) {
    return UnboxableType::kPyFloat;
  } else if (cls->type() == &PyBool_Type) {
    return UnboxableType::kPyBool;
  } else {
    return absl::nullopt;
  }
}

}  // namespace

absl::Status UnboxPyNumberOpsPattern::Apply(CallNativeInst* pynum_op,
                                            Rewriter& rewriter,
                                            SafepointInst* safepoint,
                                            Class* cls) {
  if (!rewriter.options().enable_unboxing_optimization)
    return absl::FailedPreconditionError("Unboxing optimisation disabled");

  // Seek a call_native whose type feedback indicates that it can be unboxed.
  if (cls->is_globals_class())
    return absl::FailedPreconditionError("Globals class");
  absl::optional<UnboxableType> type_opt = UnboxableTypeOf(cls);
  if (!type_opt.has_value() || !IsUnboxable(pynum_op, *type_opt))
    return absl::FailedPreconditionError("Not unboxable");
  UnboxableType type = *type_opt;

  Builder builder = rewriter.CreateBuilderBefore(*pynum_op);

  S6_RETURN_IF_ERROR(RemovePyNumberResultChecks(pynum_op, rewriter));

  // Insert unbox instructions on each operand.
  absl::InlinedVector<Value*, 3> unboxed_operands;
  if (pynum_op->callee() == Callee::kPyNumber_Power) {
    // Unbox the lhs.
    Value* unbox_lhs_op =
        builder.Unbox(type, pynum_op->operands()[0], safepoint);
    unboxed_operands.push_back(unbox_lhs_op);
    // Exponent must be the integer 2. Unbox it as a PyLong.
    // Defer the overflow check, as will return +/-1 on overflow.
    Value* unbox_rhs_op = builder.inserter().Create<UnboxInst>(
        UnboxableType::kPyLong, pynum_op->operands()[1]);
    Value* invalid_rhs = builder.IsNotEqual(unbox_rhs_op, builder.Int64(2));
    builder.DeoptimizeIfSafepoint(
        invalid_rhs, /*negated=*/false,
        "Attempted to unbox object of expected type 'long'", safepoint);
  } else {
    for (Value* operand : pynum_op->operands()) {
      if (unboxed_operands.size() < 2) {
        // Unbox the operand.
        Value* unbox_op = builder.Unbox(type, operand, safepoint);
        unboxed_operands.push_back(unbox_op);
      } else {
        // The only three-arg unboxable function is PyObject_RichCompare.
        S6_RET_CHECK(pynum_op->callee() == Callee::kPyObject_RichCompare)
            << "Too many arguments received for numeric operation!";
        // Its third arg is a constant (not a PyLong) specifying the
        // comparison operation. Do not unbox it.
        unboxed_operands.push_back(operand);
      }
    }
  }

  // The output type is occasionally different from the operands' type.
  UnboxableType output_type;
  switch (pynum_op->callee()) {
    case Callee::kPyObject_RichCompare:
      output_type = UnboxableType::kPyBool;
      break;
    case Callee::kPyNumber_TrueDivide:
      output_type = UnboxableType::kPyFloat;
      break;
    case Callee::kPyNumber_And:
    case Callee::kPyNumber_Or:
    case Callee::kPyNumber_Xor:
      output_type = type;
      break;
    default:
      // Arithmetic operators, and bitwise not, promote Booleans to longs.
      output_type =
          type == UnboxableType::kPyBool ? UnboxableType::kPyLong : type;
  }

  // Insert the instruction corresponding to the primitive arithmetic op.
  S6_ASSIGN_OR_RETURN(Value * unboxed_num_op,
                      UnboxedInst(builder, pynum_op->callee(), type,
                                  unboxed_operands, safepoint));

  // Insert a box instruction on the output.
  Value* box_op = builder.Box(output_type, unboxed_num_op);
  rewriter.ReplaceAllUsesWith(*pynum_op, *box_op);

  rewriter.erase(*pynum_op);

  return absl::OkStatus();
}

absl::Status BypassBoxUnboxPattern::Apply(UnboxInst* unbox_op,
                                          Rewriter& rewriter) {
  // Seek a box-unbox instruction sequence.
  BoxInst* box_op = dyn_cast<BoxInst>(unbox_op->boxed());
  if (!box_op) return absl::FailedPreconditionError("Not unboxing a box op");

  // Take a copy of the uses, as subsequent operations will mutate it.
  Uses uses = rewriter.GetUsesOf(*unbox_op);

  for (auto [user, operand_index] : uses) {
    OverflowedInst* of_op = dyn_cast<OverflowedInst>(user);
    if (!of_op) continue;
    // We can't overflow any more since we unbox a just boxed value.
    Builder builder = rewriter.CreateBuilderAfter(*of_op);
    rewriter.ReplaceAllUsesWith(*of_op, *builder.Int64(0));
    // Any deoptimize_if instructions dependent on that overflow instruction
    // will be automatically cleaned up.
  }

  // Reconfigure the unbox instruction's users to bypass the box/unbox and
  // work directly on the input to the box.
  rewriter.ReplaceAllUsesWith(*unbox_op, *box_op->content());

  return absl::OkStatus();
}

absl::Status RemoveUnusedBoxOpPattern::Apply(BoxInst* box_op,
                                             Rewriter& rewriter) {
  // The box op is redundant if it is only used by incref/decref op and
  // safepoint instructions.
  for (auto use : rewriter.GetUsesOf(*box_op)) {
    if (IsSafepointUse(use) || IsRefcountUse(use)) continue;

    return absl::FailedPreconditionError("Still in use");
  }

  // Maybe this box operation was used in safepoints, so create an equivalent
  // rematerialize instruction and replace the box by the rematerialize
  // instruction in all safepoints. If the box wasn't used by safepoints,
  // the rematerializeInst will be deleted as part of dead code elimination.
  Builder builder = rewriter.CreateBuilderBefore(*box_op);
  RematerializeInst* remat_op =
      builder.Rematerialize(box_op->EquivalentCallee(), {box_op->content()});
  rewriter.ReplaceAllSafepointUsesWith(*box_op, *remat_op);
  rewriter.EraseAllRefcountUses(*box_op);

  // box_op shouldn't be used anymore here so we can erase it.
  rewriter.erase(*box_op);

  return absl::OkStatus();
}

absl::Status AddTraceInstructions(Function& f, PyCodeObject* code) {
  GlobalInternTable::InternedString name =
      PythonFunctionInfoTable::Instance().Lookup(code).qualified_name;
  for (Block& b : f) {
    if (b.predecessors().empty()) {
      Builder(&b, b.begin()).TraceBegin(name);
    }
    for (Instruction& inst : b) {
      if (isa<YieldValueInst>(inst)) {
        Builder(&b, &inst).TraceEnd(f.name());
        Builder(&b, std::next(inst.GetIterator())).TraceBegin(name);
      }
    }
    Instruction& terminator = *std::prev(b.end());
    if (isa<ReturnInst>(terminator) ||
        (isa<ExceptInst>(terminator) &&
         !cast<ExceptInst>(terminator).unique_successor())) {
      Builder(&b, &terminator).TraceEnd(name);
    }
  }
  f.MarkTraced();
  return absl::OkStatus();
}

// Identifies a deoptimized_asynchronously? that is followed by another
// deoptimized_asynchronously? with no intervening calls. Removes the latter,
// as the dominating check should be sufficient.
class RemoveDominatingDeoptimizedAsynchronouslyPattern : public Pattern {
 public:
  absl::string_view name() const override {
    return "RemoveDominatingDeoptimizedAsynchronouslyPattern";
  }
  Value::Kind anchor() const override {
    return Value::kDeoptimizedAsynchronously;
  }
  absl::Status Apply(Value* value, Rewriter& rewriter) const override {
    DeoptimizedAsynchronouslyInst& deopt =
        cast<DeoptimizedAsynchronouslyInst>(*value);
    Block& parent = *deopt.parent();
    auto it = std::next(deopt.GetIterator());
    for (; it != parent.end(); ++it) {
      if (isa<CallNativeInst>(*it) || isa<CallPythonInst>(*it) ||
          isa<CallNativeIndirectInst>(*it) || isa<BoxInst>(*it) ||
          isa<CallAttributeInst>(*it) || isa<DecrefInst>(*it) ||
          isa<LoadGlobalInst>(*it)) {
        return absl::FailedPreconditionError(
            "side-effecting instruction found");
      }

      if (DeoptimizedAsynchronouslyInst* follower =
              dyn_cast<DeoptimizedAsynchronouslyInst>(&*it)) {
        Builder builder = rewriter.CreateBuilder(it);
        rewriter.ReplaceAllUsesWith(*follower, *builder.Int64(0));
        rewriter.erase(*follower);
        return absl::OkStatus();
      }
    }

    return absl::FailedPreconditionError("didn't find any candidates");
  }
};

// Identifies blocks with multiple BytecodeBeginInsts. Only keeps the first
// (for deoptimization).
class RemoveDominatingBytecodeBeginPattern : public Pattern {
 public:
  absl::string_view name() const override {
    return "RemoveDominatingBytecodeBeginPattern";
  }
  Value::Kind anchor() const override { return Value::kBytecodeBegin; }
  absl::Status Apply(Value* value, Rewriter& rewriter) const override {
    BytecodeBeginInst& deopt = cast<BytecodeBeginInst>(*value);
    Block& parent = *deopt.parent();
    auto it = std::next(deopt.GetIterator());
    for (; it != parent.end(); ++it) {
      if (BytecodeBeginInst* follower = dyn_cast<BytecodeBeginInst>(&*it)) {
        rewriter.erase(*follower);
        return absl::OkStatus();
      }
    }

    return absl::FailedPreconditionError("didn't find any candidates");
  }
};

// Identifies (call_python (constant_attribute)) where the constant attribute
// is a CFunctionAttribute. Rewrites to call_native_indirect.
class OptimizeCFunctionCallsPattern : public Pattern {
 public:
  absl::string_view name() const override {
    return "OptimizeCFunctionCallsPattern";
  }

  Value::Kind anchor() const override { return Value::kCallPython; }

  absl::Status Apply(Value* value, Rewriter& rewriter) const override {
    CallPythonInst& call_python = cast<CallPythonInst>(*value);
    if (call_python.fastcall()) {
      // Fastcall always means true Python targets, not C functions.
      return absl::FailedPreconditionError("Not a candidate");
    }

    ConstantAttributeInst* constant_attr =
        dyn_cast<ConstantAttributeInst>(call_python.callee());
    if (!constant_attr) {
      return absl::FailedPreconditionError("Not a constant attribute");
    }

    ClassManager& mgr = rewriter.class_manager();
    const CFunctionAttribute* c_function_attr =
        LookupCFunction(mgr, mgr.GetClassById(constant_attr->class_id()),
                        constant_attr->attribute_str());
    if (!c_function_attr) {
      return absl::FailedPreconditionError("Not a CFunctionAttribute");
    }

    Builder builder = rewriter.CreateBuilder(call_python.GetIterator());
    if (c_function_attr->flags() == METH_O) {
      if (call_python.call_arguments().size() != 1) {
        return absl::FailedPreconditionError(
            "METH_O takes exactly one argument.");
      }
      EventCounters::Instance().Add("optimizer.c_function.o", 1);

      static NoDestructor<CalleeInfo> meth_o_info(CalleeInfo(
          /*return_new_ref*/ true, {CalleeInfo::ArgInfo(Nullness::kMaybeNull),
                                    CalleeInfo::ArgInfo(Nullness::kNotNull)}));

      void* method = reinterpret_cast<void*>(c_function_attr->method());
      Value* self = builder.Constant(c_function_attr->self());
      Value* arg = call_python.call_arguments().front();
      Value* result =
          builder.CallIndirect(builder.Constant(method), {self, arg},
                               *meth_o_info, call_python.bytecode_offset());
      // call_python decrefs its own args (it takes stolen references).
      builder.DecrefNotNull(arg, call_python.bytecode_offset());
      builder.DecrefNotNull(constant_attr, call_python.bytecode_offset());

      rewriter.ReplaceAllUsesWith(call_python, *result);
      rewriter.erase(call_python);
      return absl::OkStatus();
    }

    if (c_function_attr->flags() == METH_NOARGS) {
      if (!call_python.call_arguments().empty()) {
        return absl::FailedPreconditionError(
            "METH_NOARGS takes zero arguments.");
      }
      EventCounters::Instance().Add("optimizer.c_function.noargs", 1);

      static NoDestructor<CalleeInfo> meth_noarg_info(CalleeInfo(
          /*return_new_ref*/ true,
          {CalleeInfo::ArgInfo(Nullness::kMaybeNull),
           CalleeInfo::ArgInfo(Nullness::kMaybeNull)}));

      void* method = reinterpret_cast<void*>(c_function_attr->method());
      Value* self = builder.Constant(c_function_attr->self());
      Value* result = builder.CallIndirect(
          builder.Constant(method), {self, builder.Int64(0)}, *meth_noarg_info,
          call_python.bytecode_offset());

      builder.DecrefNotNull(constant_attr, call_python.bytecode_offset());
      rewriter.ReplaceAllUsesWith(call_python, *result);
      rewriter.erase(call_python);
      return absl::OkStatus();
    }

    if (c_function_attr->flags() == METH_VARARGS ||
        c_function_attr->flags() == (METH_VARARGS | METH_KEYWORDS)) {
      if (call_python.names()) {
        return absl::FailedPreconditionError(
            "We can't pass keyword arguments.");
      }
      EventCounters::Instance().Add("optimizer.c_function.varargs", 1);

      int64_t num_arguments = call_python.call_arguments().size();
      Value* tuple =
          builder.Call(Callee::kPyTuple_New, {builder.Int64(num_arguments)},
                       call_python.bytecode_offset());
      for (int64_t i = 0; i < num_arguments; ++i) {
        Value* v = call_python.call_arguments()[i];
        // Note that call_python takes stolen references, so here we steal a
        // reference into the tuple.
        builder.TupleSetItem(v, tuple, i);
      }

      static NoDestructor<CalleeInfo> meth_vararg_info(CalleeInfo(
          /*return_new_ref*/ true,
          {CalleeInfo::ArgInfo(Nullness::kMaybeNull),
           CalleeInfo::ArgInfo(Nullness::kNotNull),
           CalleeInfo::ArgInfo(Nullness::kMaybeNull)}));

      void* method = reinterpret_cast<void*>(c_function_attr->method());
      Value* self = builder.Constant(c_function_attr->self());
      // Even if the callee doesn't take keywords, there's no harm in passing
      // nullptr as the third parameter. So we just always pass it.
      Value* result = builder.CallIndirect(
          builder.Constant(method), {self, tuple, builder.Int64(0)},
          *meth_vararg_info, call_python.bytecode_offset());
      builder.DecrefNotNull(tuple, call_python.bytecode_offset());
      builder.DecrefNotNull(constant_attr, call_python.bytecode_offset());

      rewriter.ReplaceAllUsesWith(call_python, *result);
      rewriter.erase(call_python);
      return absl::OkStatus();
    }

    if (c_function_attr->flags() == METH_FASTCALL) {
      EventCounters::Instance().Add("optimizer.c_function.fastcall", 1);
      void* method = reinterpret_cast<void*>(c_function_attr->method());
      Value* self = builder.Constant(c_function_attr->self());
      Value* result = builder.CallVectorcall(
          builder.Constant(method), self, call_python.names(),
          call_python.call_arguments(), call_python.bytecode_offset());

      // call_python takes stolen references, so we need to decref the arguments
      // afterwards.
      builder.DecrefNotNull(call_python.call_arguments(),
                            call_python.bytecode_offset());
      builder.DecrefNotNull(constant_attr, call_python.bytecode_offset());

      rewriter.ReplaceAllUsesWith(call_python, *result);
      rewriter.erase(call_python);
      return absl::OkStatus();
    }

    // Unhandled cases, categorized so we emit event counters and can tell which
    // cases to optimize next.
    if (c_function_attr->flags() == (METH_VARARGS | METH_KEYWORDS)) {
      // We only get here if call_python.names() != nullptr.
      EventCounters::Instance().Add(
          "optimizer.c_function.bail.vararg_keywords.kw", 1);
    } else {
      EventCounters::Instance().Add("optimizer.c_function.bail.unknown_reason",
                                    1);
    }
    return absl::FailedPreconditionError("unimplemented case");
  }
};

absl::Status OptimizeFunctionWithAnalysis(Function& f, PyCodeObject* code,
                                          OptimizerOptions options) {
  bool has_finally =
      absl::c_any_of(f, [](const Block& b) { return b.IsFinallyHandler(); });

  if (has_finally || PyTuple_GET_SIZE(code->co_cellvars) != 0) {
    LiveInfo live_info(f);

    // The refcount analysis does not support finally handlers.
    if (has_finally) {
      EventCounters::Instance().Add(
          "optimizer.refcount analysis avoided because of finally handlers", 1);
    } else {
      EventCounters::Instance().Add(
          "optimizer.refcount analysis avoided because of cell vars", 1);
    }

    analysis::Manager manager(f, code, live_info, nullconst::Analysis());
    S6_ASSIGN_OR_RETURN(auto result, manager.Analyze());

    Rewriter rewriter(f, code, options);
    analysis::RewriteAnalysis(rewriter, live_info, nullconst::Rewriter{},
                              result);
    return absl::OkStatus();
  }

  SplitCriticalEdges(f);

  LiveInfo live_info(f);

  S6_ASSIGN_OR_RETURN(refcount::ValueSet refcounted,
                      refcount::RefCountedValues(f, live_info.numbering()));

  analysis::Manager manager(
      f, code, live_info, nullconst::Analysis(),
      refcount::Analysis(refcounted, options.harden_refcount_analysis));
  S6_ASSIGN_OR_RETURN(auto result, manager.Analyze());

  S6_ASSIGN_OR_RETURN(auto externally_owned_values,
                      refcount::GetExternallyOwnedValues(f, refcounted));

  Rewriter rewriter(f, code, options);
  analysis::RewriteAnalysis(
      rewriter, live_info,
      refcount::Rewriter(refcounted, externally_owned_values), result);

#ifndef NDEBUG
  {
    // When debugging, we rerun the refcount analysis to find refcounting
    // error in the result of the optimisation.
    LiveInfo live_info2(f);
    S6_ASSIGN_OR_RETURN(refcount::ValueSet refcounted2,
                        refcount::RefCountedValues(f, live_info2.numbering()));

    analysis::Manager manager2(
        f, code, live_info2, nullconst::Analysis(),
        refcount::Analysis(refcounted2, options.harden_refcount_analysis));
    S6_ASSIGN_OR_RETURN(auto result2, manager2.Analyze());
    (void)result2;
  }
#endif

  analysis::RewriteAnalysis(rewriter, live_info, nullconst::Rewriter(), result);

  return absl::OkStatus();
}

absl::Status OptimizeFunction(Function& f, PyCodeObject* code,
                              OptimizerOptions options) {
  // Patterns that should be run whenever cleanup is required.
  using CleanupPatterns =
      std::tuple<ApplyFastcallPattern, ConstantFoldCompareInstPattern,
                 ConstantFoldBrInstPattern, ConstantFoldUnboxPattern,
                 ConstantFoldDeoptimizeIfSafepointInstPattern,
                 RemoveDominatingDeoptimizedAsynchronouslyPattern,
                 EliminateTrivialJumpsPattern,
                 EliminateUnusedBlockArgumentsPattern, BypassBoxUnboxPattern,
                 RemoveUnusedBoxOpPattern>;

  // We optimize in stages.

  // Stage 1: Expand all patterns that rely on safepoints and expand meta-
  // instruction like load global

  S6_RETURN_IF_ERROR(OptimizeLoadGlobal(f, code, options));

  // Then run to fixpoint all patterns that depends on safepoints being
  // available.
  S6_RETURN_IF_ERROR(
      (RewritePatterns<CreateCallAttributePattern, SpeculateCallsPattern,
                       OptimizeTypeConstructionPattern,
                       OptimizeMakeGeneratorFunctionPattern,
                       OptimizeGeneratorIterIdentityPattern,
                       OptimizeDundersPattern, OptimizeCallDunderPattern,
                       OptimizePyObjectGetItemPattern, UnboxPyNumberOpsPattern,
                       OptimizeMathFunctionsPattern, CleanupPatterns>(
          f, code, options)));

  // Run OptimizeCFunctionCallsPattern separately as it can make some other
  // patterns (OptimizeMathFunctionsPattern) harder to match.
  S6_RETURN_IF_ERROR(RewritePatterns<OptimizeCFunctionCallsPattern>(f, code));
  // TODO: Fold these into Patterns so they can be run as part of the
  // above fixpointing.
  S6_RETURN_IF_ERROR(OptimizeGetSetAttr(f, code, options));

  // Stage 2: Aggressively optimize away BytecodeBeginInsts and run the constant
  // folding passes again.
  S6_RETURN_IF_ERROR(
      (RewritePatterns<RemoveDominatingBytecodeBeginPattern, CleanupPatterns>(
          f, code)));

  // Stage 3: Run the global analyses and their optimizations.
  absl::Status status = OptimizeFunctionWithAnalysis(f, code, options);
  if (status.ok()) {
    EventCounters::Instance().Add("optimizer.analysis success", 1);
  } else {
    EventCounters::Instance().Add(
        absl::StrCat("optimizer.analysis failure(", status.ToString(), ")"), 1);
    // We continue normal optimization after an analysis failure,
    // we just have lost time and changed nothing.
  }

  // Stage 4: Clean up code again
  S6_RETURN_IF_ERROR(
      (RewritePatterns<RemoveDominatingBytecodeBeginPattern, CleanupPatterns>(
          f, code)));

  // Finally add trace instructions.
  if (options.enable_function_tracing) {
    S6_RETURN_IF_ERROR(AddTraceInstructions(f, code));
  }
  S6_CHECK_OK(VerifyFunction(f));
  return absl::OkStatus();
}

}  // namespace deepmind::s6
