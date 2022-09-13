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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CALLS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CALLS_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer_util.h"
#include "type_feedback.h"

namespace deepmind::s6 {

// Identifies a call to an object returned by PyFunction_NewWithQualName where
// the code object is a generator, and replaces with s6::CreateGenerator.
class OptimizeMakeGeneratorFunctionPattern
    : public PatternT<OptimizeMakeGeneratorFunctionPattern, CallPythonInst,
                      NeedsCallee<CallNativeInst>> {
 public:
  static absl::Status Apply(CallPythonInst* call_python, Rewriter& rewriter,
                            CallNativeInst* make_function);
};

// Identifies a call to PyObject_GetIter from an object that is known to be
// a generator, and elides it (because PyObject_GetIter on a generator is the
// identity).
class OptimizeGeneratorIterIdentityPattern
    : public PatternT<OptimizeGeneratorIterIdentityPattern, CallNativeInst> {
 public:
  static absl::Status Apply(CallNativeInst* get_iter, Rewriter& rewriter);
};

// Identifies calls to known functions that map to __dunder__ methods. For
// example, calls to PyObject_Subtract on an object that implements __sub__
// directly calls the class's __sub__ method.
//
// The monomorphism is important as we don't want to deal with the case of two
// different classes for LHS and RHS - the fallback protocol is awkward. If we
// know LHS and RHS have the same class, then we never have to deal with
// __radd__ and friends.
class OptimizeDundersPattern
    : public PatternT<OptimizeDundersPattern, CallNativeInst, NeedsSafepoint,
                      NeedsMonomorphicClass<>> {
 public:
  static absl::Status Apply(CallNativeInst* call, Rewriter& rewriter,
                            SafepointInst* safepoint, Class* cls);
};

// Identifies calls to objects that implement __call__ and replaces them by a
// direct method call to `__call__`
class OptimizeCallDunderPattern
    : public PatternT<OptimizeCallDunderPattern, CallPythonInst, NeedsSafepoint,
                      NeedsMonomorphicClass<>> {
 public:
  static absl::Status Apply(CallPythonInst* call, Rewriter& rewriter,
                            SafepointInst* safepoint, Class* cls);
};

// Identifies calls to PyObject_GetItem with a lhs type of (list or tuple) and
// a rhs type of int. Transforms to a tuple/list load.
class OptimizePyObjectGetItemPattern
    : public PatternT<
          OptimizePyObjectGetItemPattern, CallNativeInst, NeedsSafepoint,
          NeedsNativeCallee<Callee::kPyObject_GetItem>, NeedsTypeFeedback> {
 public:
  static absl::Status Apply(CallNativeInst* call, Rewriter& rewriter,
                            SafepointInst* safepoint,
                            const ClassDistributionSummary& summary);
};

// Identifies calls to trancendental builtin functions (sin/cos) and calls
// libm on the unboxed value.
class OptimizeMathFunctionsPattern
    : public PatternT<OptimizeMathFunctionsPattern, CallPythonInst,
                      NeedsSafepoint> {
 public:
  static absl::Status Apply(CallPythonInst* call, Rewriter& rewriter,
                            SafepointInst* safepoint);
};

// Identifies sequences of:
//   %a = call_native PyObject_GetAttr %x, "attr"
//   decref notnull %x
//   ...
//   %b = call_python %a (...)
//
// And converts to
//   %b = call_attribute %x :: "attr" (...)  // At the location of call_python
//
// This optimization is one of the rare cases where the order of destructor
// calls may be modified. Indeed in the rare case where %a does not own a
// reference to %x and the current function owns the only reference to %x, then
// the `decref notnull %x` will actually trigger a deletion. In such a case, if
// any other side-effect happened between the call_native PyObject_GetAttr and
// the call_python %a, then moving the deletion of %x to the location of the
// call_python will result in an externally observable behavior change.
class CreateCallAttributePattern
    : public PatternT<CreateCallAttributePattern, CallPythonInst,
                      NeedsCallee<CallNativeInst>> {
 public:
  static absl::Status Apply(CallPythonInst* call_python, Rewriter& rewriter,
                            CallNativeInst* call_native);
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_CALLS_H_
