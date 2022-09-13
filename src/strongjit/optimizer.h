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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_H_

#include <Python.h>

#include <cstdint>

#include "absl/flags/declare.h"
#include "absl/status/status.h"
#include "classes/class_manager.h"
#include "strongjit/base.h"
#include "strongjit/optimizer_util.h"

namespace deepmind::s6 {

// Runs all optimizations on `f`.
absl::Status OptimizeFunction(Function& f, PyCodeObject* code,
                              OptimizerOptions options = {});

// Replaces instances of load_global macro instructions with a code sequence
// that inlines a cached value from globals, with a fallback to a native call if
// the cache entry is stale.
absl::Status OptimizeLoadGlobal(Function& f, PyCodeObject* code,
                                OptimizerOptions options = {});

// Adds inline caches to call_native s6::GetAttr and s6::SetAttr, and optimizes
// monomorphic fast-attribute loads and stores.
//
// `safe_metadata` is the metadata for `code` that can be inspected at compile
// time (a thread-safe snapshot of the live metadata).
absl::Status OptimizeGetSetAttr(Function& f, PyCodeObject* code,
                                OptimizerOptions options = {});

// Identifies any numeric op tagged (in the inline cache) as being a candidate
// for boxing:
//   bytecode_begin
//   %z = call_native <binary_op> %x %y
//   decref notnull %x, %y
//   bytecode_begin
//
// And lifts it to an 'unboxed' implementation as follows:
//   %a = unbox %x
//   %o = overflowed? %a
//   br %o, deoptimised &ad, &af
// &ad:
//   bytecode_begin
//   return
// &af:
//   %b = unbox %y
//   %p = overflowed? %b
//   br %p, deoptimised &bd, &bf
// &bd:
//   bytecode_begin
//   return
// &bf:
//   %c = <binary-inst> %a %b
//   %q = overflowed? %c
//   br %q, deoptimised &cd, &cf
// &cd:
//   bytecode_begin
//   return
// &cf:
//   %z = box %c
//   decref notnull %x, %y
//   bytecode_begin
//
// `safe_metadata` is the metadata for `code` that can be inspected at compile
// time (a thread-safe snapshot of the live metadata).
class UnboxPyNumberOpsPattern
    : public PatternT<UnboxPyNumberOpsPattern, CallNativeInst, NeedsSafepoint,
                      NeedsMonomorphicClass<>> {
 public:
  static absl::Status Apply(CallNativeInst* pynum_op, Rewriter& rewriter,
                            SafepointInst* safepoint, Class* cls);
};

// Identifies sequences of:
//   %b = box %a
//   %c = unbox %b
//   %o = overflowed? %c
//   br %o, deoptimised &deopt, &fast
// &deopt:
//   bytecode_begin
//   return
// &fast:
//   ...
//   <instruction> %c
//
// And converts to:
//   %b = box %a
//   ...
//   <instruction> %a
//
// Additionally, the `box` instruction (%b) is deleted if it becomes unused
// as a result of this replacement.
class BypassBoxUnboxPattern
    : public PatternT<BypassBoxUnboxPattern, UnboxInst> {
 public:
  static absl::Status Apply(UnboxInst* unbox_op, Rewriter& rewriter);
};
class RemoveUnusedBoxOpPattern
    : public PatternT<RemoveUnusedBoxOpPattern, BoxInst> {
 public:
  static absl::Status Apply(BoxInst* box_op, Rewriter& rewriter);
};

// Add trace event instructions to function entry and exit.
absl::Status AddTraceInstructions(Function& f);

// Given type feedback for a call_attribute instruction,
// speculates the call under a type guard.
class SpeculateCallsPattern : public Pattern {
 public:
  absl::string_view name() const override { return "SpeculateCallsPattern"; }

  Value::Kind anchor() const override { return Value::kCallAttribute; }

  absl::Status Apply(Value* value, Rewriter& rewriter) const override;
};

// Optimizes (call_python (constant_attribute)) by adding the fastcall flag
// where possible.
class ApplyFastcallPattern : public Pattern {
 public:
  absl::string_view name() const override { return "ApplyFastcallPattern"; }

  Value::Kind anchor() const override { return Value::kCallPython; }

  absl::Status Apply(Value* value, Rewriter& rewriter) const override;

 private:
  absl::Status ApplyKeywordArgumentRemapping(
      CallPythonInst& call, Rewriter& rewriter, Builder& builder,
      const FunctionAttribute& method_attr) const;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_H_
