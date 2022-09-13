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

#include "strongjit/optimize_calls.h"

#include <math.h>

#include <cstdint>
#include <iterator>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "core_util.h"
#include "runtime/pyframe_object_cache.h"
#include "strongjit/callees.h"
#include "strongjit/formatter.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer.h"
#include "strongjit/optimizer_util.h"
#include "strongjit/value_casts.h"
#include "type_feedback.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

absl::Status OptimizeMakeGeneratorFunctionPattern::Apply(
    CallPythonInst* call_python, Rewriter& rewriter,
    CallNativeInst* make_function) {
  if (!make_function->CalleeIs(Callee::kPyFunction_NewWithQualName)) {
    return absl::FailedPreconditionError(
        "Not a call_python of a kPyFunction_NewWithQualName");
  }

  // Look up the code object being bound. Is it a generator?
  Value* code_value = make_function->call_arguments()[0];
  PyObject* code = GetValueAsConstantObject(code_value, rewriter.code_object());
  if (!code || !PyCode_Check(code)) {
    return absl::FailedPreconditionError(
        "PyFunction_NewWithQualName did not bind a constant PyCodeObject");
  }

  if ((reinterpret_cast<PyCodeObject*>(code)->co_flags & CO_GENERATOR) == 0) {
    return absl::FailedPreconditionError(
        "Bound code object was not of generator type.");
  }

  // Okay, we can replace `call_python` with a call to s6::CreateGenerator.
  Builder builder = rewriter.CreateBuilder(call_python->GetIterator());
  std::vector<Value*> args;
  args.push_back(make_function);
  args.push_back(
      builder.FrameVariable(FrameVariableInst::FrameVariableKind::kBuiltins));
  absl::c_copy(call_python->call_arguments(), std::back_inserter(args));
  Value* result = builder.Call(Callee::kCreateGenerator, args,
                               call_python->bytecode_offset());

  builder.DecrefNotNull(make_function, call_python->bytecode_offset());
  rewriter.ReplaceAllUsesWith(*call_python, *result);
  rewriter.erase(*call_python);
  EventCounters::Instance().Add("optimizer.make_generator", 1);
  return absl::OkStatus();
}

absl::Status EnsureGenerator(Instruction* bytecode_inst,
                             int64_t bytecode_offset, Value* arg,
                             Rewriter& rewriter, Builder& builder) {
  if (auto call = dyn_cast<CallNativeInst>(arg);
      call && call->callee() == Callee::kCreateGenerator) {
    return absl::OkStatus();
  }

  absl::optional<ClassDistributionSummary> summary =
      rewriter.function().GetTypeFeedbackForBytecodeOffset(bytecode_offset);
  if (!summary.has_value()) {
    return absl::FailedPreconditionError("No type feedback");
  }
  PyTypeObject* type = summary->MonomorphicType(rewriter.options().mgr);
  if (type != &PyGen_Type &&
      type != GetPyGeneratorFrameObjectCache()->GetGenType()) {
    return absl::FailedPreconditionError("Type feedback says not a generator");
  }
  auto safepoint_or = BeginningOfBytecodeInstruction(bytecode_inst);
  if (!safepoint_or.ok()) {
    return absl::FailedPreconditionError("No safepoint");
  }
  // It's worthwhile speculating. Note that we don't rely on the
  // generator class because we know it is immutable.
  builder.DeoptimizeIfSafepoint(
      builder.CheckClassId(arg, summary->MonomorphicClass()),
      /*negated=*/true, "Speculated iterator of generator type", *safepoint_or);
  return absl::OkStatus();
}

absl::Status OptimizeGeneratorIterIdentityPattern::Apply(
    CallNativeInst* get_iter, Rewriter& rewriter) {
  if (!get_iter->CalleeIs(Callee::kPyObject_GetIter)) {
    return absl::FailedPreconditionError("Not a call to PyObject_GetIter");
  }

  // Do we know that the first argument is a generator?
  Value* arg = get_iter->call_arguments()[0];

  Builder builder = rewriter.CreateBuilder(get_iter->GetIterator());
  S6_RETURN_IF_ERROR(EnsureGenerator(get_iter, get_iter->bytecode_offset(), arg,
                                     rewriter, builder));

  // Okay, just skip the call to GetIter.
  builder.IncrefNotNull(arg);
  rewriter.ReplaceAllUsesWith(*get_iter, *arg);
  rewriter.erase(*get_iter);
  return absl::OkStatus();
}

absl::Status OptimizeDundersPattern::Apply(CallNativeInst* call,
                                           Rewriter& rewriter,
                                           SafepointInst* safepoint,
                                           Class* cls) {
  // Note, only binary functions are here.
  static const NoDestructor<absl::flat_hash_map<Callee, absl::string_view>>
      kDunderCallees({{Callee::kPyNumber_Add, "__add__"},
                      {Callee::kPyNumber_And, "__and__"},
                      {Callee::kPyNumber_FloorDivide, "__floordiv__"},
                      {Callee::kPyNumber_InPlaceAdd, "__iadd__"},
                      {Callee::kPyNumber_InPlaceAnd, "__iand__"},
                      {Callee::kPyNumber_InPlaceFloorDivide, "__ifloordiv__"},
                      {Callee::kPyNumber_InPlaceLshift, "__ilshift__"},
                      {Callee::kPyNumber_InPlaceMultiply, "__imul__"},
                      {Callee::kPyNumber_InPlaceOr, "__ior__"},
                      {Callee::kPyNumber_InPlacePower, "__ipow__"},
                      {Callee::kPyNumber_InPlaceRemainder, "__irem__"},
                      {Callee::kPyNumber_InPlaceRshift, "__irshift__"},
                      {Callee::kPyNumber_InPlaceSubtract, "__isub__"},
                      {Callee::kPyNumber_InPlaceTrueDivide, "__itruediv__"},
                      {Callee::kPyNumber_InPlaceXor, "__ixor__"},
                      {Callee::kPyNumber_Lshift, "__lshift__"},
                      {Callee::kPyNumber_Multiply, "__mul__"},
                      {Callee::kPyNumber_Or, "__or__"},
                      {Callee::kPyNumber_Power, "__pow__"},
                      {Callee::kPyNumber_Remainder, "__rem__"},
                      {Callee::kPyNumber_Rshift, "__rshift__"},
                      {Callee::kPyNumber_Subtract, "__sub__"},
                      {Callee::kPyNumber_TrueDivide, "__truediv__"},
                      {Callee::kPyNumber_Xor, "__xor__"}});

  auto it = kDunderCallees->find(call->callee());
  if (it == kDunderCallees->end()) {
    return absl::FailedPreconditionError("Not a call to a dunder-like method");
  }
  absl::string_view dunder = it->second;

  auto attr_it =
      cls->attributes().find(rewriter.class_manager().InternString(dunder));
  if (attr_it == cls->attributes().end()) {
    return absl::FailedPreconditionError("No dunder attribute");
  }

  const FunctionAttribute* attr =
      dynamic_cast<FunctionAttribute*>(&*attr_it->second);
  if (!attr) {
    return absl::FailedPreconditionError("Dunder was not a function attribute");
  }

  if (!attr->bound()) {
    return absl::FailedPreconditionError("Dunder was not a bound method");
  }

  if (attr->code()->co_argcount != call->call_arguments().size()) {
    return absl::FailedPreconditionError(
        "Dunder had wrong number of arguments.");
  }

  Value* self = call->call_arguments()[0];
  Value* rhs = call->call_arguments()[1];
  S6_CHECK_EQ(call->call_arguments().size(), 2) << dunder;
  Builder builder = rewriter.CreateBuilder(call->GetIterator());
  builder.DeoptimizeIfSafepoint(
      builder.And(builder.CheckClassId(rhs, cls),
                  builder.CheckClassId(self, cls)),
      /*negated=*/true, "Expected dunder method to have monomorphic type",
      safepoint);
  rewriter.AddReliedUponClass(cls);
  builder.DeoptimizedAsynchronously(safepoint);

  builder.IncrefNotNull(self);
  builder.IncrefNotNull(rhs);
  Value* result = builder.CallConstantAttribute(
      cls, dunder, {self, rhs}, nullptr, call->bytecode_offset());

  rewriter.ReplaceAllUsesWith(*call, *result);
  rewriter.erase(*call);
  return absl::OkStatus();
}

absl::Status OptimizeCallDunderPattern::Apply(CallPythonInst* call,
                                              Rewriter& rewriter,
                                              SafepointInst* safepoint,
                                              Class* cls) {
  Value* callee = call->callee();
  if (auto call = dyn_cast<CallNativeInst>(callee);
      call && call->callee() == Callee::kPyObject_GetAttr) {
    return absl::FailedPreconditionError(
        "Won't prematurely optimize call of attribute get.");
  }

  auto attr_it =
      cls->attributes().find(rewriter.class_manager().InternString("__call__"));
  if (attr_it == cls->attributes().end()) {
    return absl::FailedPreconditionError("No dunder attribute");
  }

  const FunctionAttribute* attr =
      dynamic_cast<FunctionAttribute*>(&*attr_it->second);
  if (!attr) {
    return absl::FailedPreconditionError("Dunder was not a function attribute");
  }

  Builder builder = rewriter.CreateBuilder(call->GetIterator());
  builder.DeoptimizeIfSafepoint(
      builder.CheckClassId(callee, cls),
      /*negated=*/true, "Expected dunder method to have monomorphic type",
      safepoint);
  rewriter.AddReliedUponClass(cls);
  builder.DeoptimizedAsynchronously(safepoint);
  std::vector<Value*> args(call->call_arguments().begin(),
                           call->call_arguments().end());
  if (attr->bound()) {
    args.insert(args.begin(), callee);
  }

  Value* result = builder.CallConstantAttribute(
      cls, "__call__", args, call->names(), call->bytecode_offset());

  rewriter.ReplaceAllUsesWith(*call, *result);
  rewriter.erase(*call);
  return absl::OkStatus();
}

absl::Status OptimizePyObjectGetItemPattern::Apply(
    CallNativeInst* call, Rewriter& rewriter, SafepointInst* safepoint,
    const ClassDistributionSummary& summary) {
  // We want to match x[y] where x is (list or tuple) and y is int.
  // The type feedback is then expected to be polymorphic; one class must be int
  // and the other must be list or tuple. We can't rely on which is which in
  // the class distribution summary though.

  if (!summary.IsPolymorphic() || summary.common_class_ids().size() != 2) {
    return absl::FailedPreconditionError("Type is not polymorphic");
  }
  Class* lhs_cls =
      rewriter.class_manager().GetClassById(summary.common_class_ids()[1]);
  Class* rhs_cls =
      rewriter.class_manager().GetClassById(summary.common_class_ids()[0]);
  if (lhs_cls->type() == &PyLong_Type) {
    // Canonicalize on lhs_cls being (list|tuple), rhs_cls being int.
    std::swap(lhs_cls, rhs_cls);
  }

  bool is_tuple = lhs_cls->type() == &PyTuple_Type;
  bool is_list = lhs_cls->type() == &PyList_Type;
  if (!is_tuple && !is_list) {
    return absl::FailedPreconditionError("LHS type was not tuple or list");
  }

  if (rhs_cls->type() != &PyLong_Type) {
    return absl::FailedPreconditionError("RHS type was not int");
  }

  Builder builder = rewriter.CreateBuilder(call->GetIterator());
  Value* lhs = call->call_arguments()[0];
  Value* rhs = call->call_arguments()[1];

  Value* index = builder.Unbox(UnboxableType::kPyLong, rhs, safepoint);

  builder.DeoptimizeIfSafepoint(
      builder.CheckClassId(lhs, lhs_cls), /*negated=*/true,
      "Optimized indexing of list or tuple received non-list or tuple type",
      safepoint);

  Value* container_size = builder.GetSize(lhs);

  // If the index is negative, recalculate based on the container size.
  index = builder
              .Conditional(
                  builder.IsNegative(index),
                  [&](Builder builder) {
                    // index = size + 1 + index; (note index is negative).
                    return Builder::ValueList{builder.Add(
                        NumericInst::kInt64,
                        builder.Add(container_size, builder.Zero()), index)};
                  },
                  Builder::ValueList{index})
              .front();

  // TODO: An unsigned comparison could get rid of the >= 0 here.
  Value* inbounds = builder.And(
      builder.IsGreaterEqual(NumericInst::kInt64, index, builder.Zero()),
      builder.IsLessThan(NumericInst::kInt64, index, container_size));

  builder.DeoptimizeIfSafepoint(inbounds, /*negated=*/true,
                                "List or tuple index out of bounds", safepoint);

  Value* value;
  if (is_tuple) {
    value = builder.TupleGetItem(lhs, index);
  } else {
    S6_CHECK(is_list);
    // PyListType stores items in a hungoff array, so we need to dereference
    // ob_item.
    Value* ob_item = builder.Load64(lhs, offsetof(PyListObject, ob_item));
    // Then index into ob_item to get the value.
    value = builder.Load64(ob_item, index, LoadInst::Shift::k8);
  }
  builder.IncrefNotNull(value);

  rewriter.ReplaceAllUsesWith(*call, *value);
  rewriter.erase(*call);

  return absl::OkStatus();
}

absl::Status OptimizeMathFunctionsPattern::Apply(CallPythonInst* call,
                                                 Rewriter& rewriter,
                                                 SafepointInst* safepoint) {
  PyObject* callee =
      GetValueAsConstantObject(call->callee(), rewriter.code_object());
  if (!callee) {
    return absl::FailedPreconditionError("Not a constant callee");
  }

  using BuiltinMap = absl::flat_hash_map<PyObject*, Callee>;
  static NoDestructor<BuiltinMap> map(BuiltinMap{
      {BuiltinObjects::Instance().LookupBuiltin("math.sin"),
       Callee::kSinDouble},
      {BuiltinObjects::Instance().LookupBuiltin("math.cos"),
       Callee::kCosDouble},
  });

  auto it = map->find(callee);
  if (it == map->end()) {
    return absl::FailedPreconditionError("Not a known builtin");
  }

  Value* arg = call->call_arguments()[0];
  Builder builder = rewriter.CreateBuilder(call->GetIterator());
  Value* unboxed = builder.Unbox(UnboxableType::kPyFloat, arg, safepoint);
  Value* result = builder.Call(it->second, {unboxed}, call->bytecode_offset());
  builder.DecrefNotNull(arg, call->bytecode_offset());
  builder.DecrefNotNull(call->callee(), call->bytecode_offset());
  Value* boxed = builder.Box(UnboxableType::kPyFloat, result);
  rewriter.ReplaceAllUsesWith(*call, *boxed);
  rewriter.erase(*call);

  return absl::OkStatus();
}

absl::Status CreateCallAttributePattern::Apply(CallPythonInst* call_python,
                                               Rewriter& rewriter,
                                               CallNativeInst* call_native) {
  if (!call_native->CalleeIs(Callee::kPyObject_GetAttr)) {
    return absl::FailedPreconditionError("Call native not a PyObject_GetAttr");
  }

  absl::optional<absl::string_view> attr_str = GetValueAsConstantString(
      call_native->call_arguments()[1], rewriter.code_object());

  if (!attr_str.has_value()) {
    return absl::FailedPreconditionError("Attribute was not a constant string");
  }

  // The call_native instruction must only be used by the call_python
  // instruction, and a compare for the validity check.
  //
  // It can also be used by safepoints, and refcounting operations.
  for (Use use : rewriter.GetUsesOf(*call_native)) {
    if (IsSafepointUse(use) || IsRefcountUse(use) || IsCompareWithZeroUse(use))
      continue;

    if (use.user == call_python &&
        use.operand_index == CallPythonInst::kCalleeOperandIndex)
      continue;

    return absl::FailedPreconditionError("Has unsafe uses");
  }

  // The call_native instruction's first operand (the receiver) must be
  // decreffed immediately afterwards.
  Value* receiver = call_native->operands()[0];
  DecrefInst* receiver_decref =
      dyn_cast<DecrefInst>(&*std::next(call_native->GetIterator()));
  if (!receiver_decref || receiver_decref->operand() != receiver) {
    return absl::FailedPreconditionError("Not followed by decref");
  }

  // Okay. Let's go.

  // First we delay the decref to the point of the call_python instruction.
  S6_RETURN_IF_ERROR(DelayDecref(*receiver_decref, {call_python}));
  // Now the call_python instruction shall be preceded by a decref.
  receiver_decref =
      dyn_cast<DecrefInst>(&*std::prev(call_python->GetIterator()));
  if (!receiver_decref || receiver_decref->operand() != receiver) {
    return absl::FailedPreconditionError("DelayDecref misbehaved");
  }

  // Then we can replace the block of 2 instruction:
  //   decref notnull %receiver
  //   call_python %attribute (...)
  // by
  //   call_attribute %receiver :: "attribute_name" (...).
  // Note that call_attribute instruction steals the receiver.
  BlockInserter inserter(call_python->parent(), call_python->GetIterator());
  CallAttributeInst* new_call = inserter.Create<CallAttributeInst>(
      receiver, rewriter.InternString(*attr_str), call_python->call_arguments(),
      call_python->names(), call_native->bytecode_offset(),
      call_python->bytecode_offset());

  rewriter.ReplaceAllUsesWith(*call_python, *new_call);
  rewriter.erase(*call_python);
  rewriter.erase(*receiver_decref);

  // Now that the call_python instruction has been eliminated, we need to
  // clean-up remaining uses of %attribute from the call_native instruction. In
  // particular we need to generate a "rematerialised" version of the get_attr,
  // to be executed only in the case of deoptimisation.
  int64_t names_index =
      cast<FrameVariableInst>(call_native->call_arguments()[1])->index();
  Builder call_native_builder =
      rewriter.CreateBuilder(call_native->GetIterator());
  Value* remat_getattr = call_native_builder.Rematerialize(
      Callee::kRematerializeGetAttr,
      {receiver, call_native_builder.Int64(names_index)});

  rewriter.ReplaceAllSafepointUsesWith(*call_native, *remat_getattr);
  rewriter.EraseAllRefcountUses(*call_native);

// Now the only remaining uses of call_native instruction should be comparison
// with zero that should be removed. We can just replace call_native by one so
// that all those comparisons will be trivialized as returning false by constant
// folding.
#ifndef NDEBUG
  for (Use use : rewriter.GetUsesOf(*call_native))
    S6_CHECK(IsCompareWithZeroUse(use));
#endif
  rewriter.ReplaceAllUsesWith(*call_native, *call_native_builder.Int64(1));

  // And we can finally erase the call_native
  rewriter.erase(*call_native);

  return absl::OkStatus();
}

}  // namespace deepmind::s6
