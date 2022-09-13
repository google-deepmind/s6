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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_TRAITS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_TRAITS_H_

#include "strongjit/value_traits.h"

namespace deepmind::s6 {

namespace detail {

// Predicate whose value is true for all subclasses of Instruction, not
// including Instruction itself.
template <typename ValueCls>
struct IsInstructionSubclass {
  static constexpr bool value = std::is_base_of<Instruction, ValueCls>::value &&
                                !std::is_same<Instruction, ValueCls>::value;
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// ForAllInstructionKinds
//
// This allows iteration at compile time over all instruction kinds.

template <typename F, typename... Args>
constexpr auto ForAllInstructionKinds(Args&&... args) {
  return ForFilteredValues<detail::IsInstructionSubclass, F, Args...>(
      std::forward<Args>(args)...);
}

class InstructionTraits {
 public:
  static bool ClobbersAllRegisters(const Value& value) {
    return kClobbersAllRegisters.Contains(value.kind());
  }

  static bool ProducesValue(const Value& value) {
    return kProducesValue.Contains(value.kind());
  }

  static bool HasSideEffects(const Value& value) {
    return kHasSideEffects.Contains(value.kind());
  }

  static bool HasPreciseLocation(const Value& value) {
    return ValueTraits::IsA<PreciseLocationInst>(value);
  }

 private:
  template <typename InsnCls>
  struct ProducesValuePredicate {
    static const bool value = InsnCls::kProducesValue;
  };

  template <typename InsnCls>
  struct HasSideEffectsPredicate {
    static const bool value = InsnCls::kHasSideEffects;
  };

  template <typename InsnCls>
  struct ClobbersAllRegistersPredicate {
    static const bool value = InsnCls::kClobbersAllRegisters;
  };

  static constexpr ValueSet kProducesValue =
      ValueSet::FromPredicate<detail::IsInstructionSubclass,
                              ProducesValuePredicate>();

  static constexpr ValueSet kHasSideEffects =
      ValueSet::FromPredicate<detail::IsInstructionSubclass,
                              HasSideEffectsPredicate>();

  static constexpr ValueSet kClobbersAllRegisters =
      ValueSet::FromPredicate<detail::IsInstructionSubclass,
                              ClobbersAllRegistersPredicate>();
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_INSTRUCTION_TRAITS_H_
