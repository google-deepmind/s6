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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_TYPE_CONSTRUCTION_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_TYPE_CONSTRUCTION_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "strongjit/base.h"
#include "strongjit/optimizer_util.h"

namespace deepmind::s6 {

// Identifies calling a type object to construct a new object instance.
// Determines the required __new__ and __init__ functions.
class OptimizeTypeConstructionPattern
    : public PatternT<OptimizeTypeConstructionPattern, CallPythonInst,
                      NeedsCallee<ConstantAttributeInst>, NeedsSafepoint> {
 public:
  static absl::Status Apply(CallPythonInst* call, Rewriter& rewriter,
                            ConstantAttributeInst* constant_attribute,
                            SafepointInst* safepoint);

 private:
  static Value* CallInitDunder(Builder& builder, const Class& cls,
                               const FunctionAttribute& attr,
                               CallPythonInst& call_python, Value* object);
  static Value* CallNewDunder(Builder& builder, const Class& cls,
                              const FunctionAttribute& attr,
                              CallPythonInst& call_python);
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZE_TYPE_CONSTRUCTION_H_
