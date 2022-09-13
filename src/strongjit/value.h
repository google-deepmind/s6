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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_H_

#include <cstdint>
#include <iterator>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "strongjit/util.h"

namespace deepmind::s6 {
// Value is the base of the strongjit hierarchy. All instructions, blocks and
// block arguments are Values. A Value can be used by Instruction as an operand.
// All Values are discriminated by Kind, which is an opcode. This
// kind can be used by the helper functions isa<>, dyn_cast<> and cast<> to
// perform run-time downcasting without RTTI overhead.
class Value {
 public:
  // LINT.IfChange
  enum Kind : uint8_t {
    // Invalid value, distinct from the tombstone value. This will be
    // the default-constructed value and allows to determine invalid
    // iterators such as end(), distinct from tombstones.
    kSentinel = 0,
    kBlock,
    kBlockArgument,

    // Instructions
    kConstant,
    kLoad,
    kLoadGlobal,
    kStore,
    kIncref,
    kDecref,

    kCompare,

    // Unary instructions
    kNegate,
    kNot,
    kSext,

    // Binary instructions
    kAdd,
    kAnd,
    kDivide,
    kMultiply,
    kOr,
    kRemainder,
    kShiftLeft,
    kShiftRightSigned,
    kSubtract,
    kXor,

    kIntToFloat,
    kFrameVariable,
    kCallNative,
    kCallPython,
    kCallAttribute,
    kCallVectorcall,
    kBytecodeBegin,
    kDeoptimizeIfSafepoint,
    kCallNativeIndirect,
    kYieldValue,
    kBox,
    kUnbox,
    kOverflowed,
    kFloatZero,
    kRematerialize,
    kGetClassId,
    kGetObjectDict,
    kGetInstanceClassId,
    kCheckClassId,
    kLoadFromDict,
    kStoreToDict,
    kConstantAttribute,
    kDeoptimizedAsynchronously,
    kSetObjectClass,

    // Terminators
    kDeoptimizeIf,
    kJmp,
    kBr,
    kExcept,
    kUnreachable,
    kReturn,

    // Profiling instructions
    kAdvanceProfileCounter,
    kIncrementEventCounter,
    kTraceBegin,
    kTraceEnd,

    kMaxValue = kTraceEnd,
  };
  // LINT.ThenChange(value_traits.h)

  Value() = default;
  Value& operator=(const Value&) = default;
  Value(const Value&) = default;
  Value& operator=(Value&&) = default;
  Value(Value&&) = default;

  virtual ~Value() = default;

  // Returns the kind of this Value.
  virtual Kind kind() const = 0;

  static constexpr Kind kKind = kSentinel;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_H_
