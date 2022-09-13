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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_TRAITS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_TRAITS_H_

#include <initializer_list>
#include <iterator>
#include <utility>

#include "absl/types/optional.h"
#include "strongjit/base.h"
#include "strongjit/instructions.h"
#include "strongjit/util.h"

namespace deepmind::s6 {

namespace detail {

// Filter that passes only subclasses of instruction on to F.
template <template <typename FilterCls> class Filter, typename F,
          typename... Args>
struct ValueFilter {
  typedef decltype(F::Default(std::declval<Args>()...)) result_type;

  template <typename Cls, std::enable_if_t<Filter<Cls>::value, bool> = true>
  static constexpr absl::optional<result_type> Visit(Args... args) {
    return F::template Visit<Cls>(args...);
  }

  template <typename Cls, std::enable_if_t<!Filter<Cls>::value, bool> = true>
  static constexpr absl::optional<result_type> Visit(Args... args) {
    return absl::optional<result_type>();
  }

  static constexpr result_type Default(Args... args) {
    return F::Default(args...);
  }
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// ForAllValues
//
// Add any new Value subclasses here. This allows iteration at compile time over
// all Value subclasses.
//
// F must be a type templated on an instruction subclass, and provide the
// following static methods:
//  template<typename ValueCls> std::optional<result_type> Visit(Args...)
//    - Process a value class, and return an optional value. If the optional
//      has a value, then this is what is returned by ForAllValues.
//  result_type Default(Args..)
//    - Return a default value in case no call to Visit produced a value.

template <typename F, typename... Args>
constexpr auto ForAllValues(Args... args) {
#define VALUE_CLASS(cls)                      \
  do {                                        \
    auto v = F::template Visit<cls>(args...); \
    if (v.has_value()) return v.value();      \
  } while (0)
  // LINT.IfChange
  VALUE_CLASS(BlockArgument);
  VALUE_CLASS(Block);
  VALUE_CLASS(Instruction);
  VALUE_CLASS(JmpInst);
  VALUE_CLASS(BrInst);
  VALUE_CLASS(ConstantInst);
  VALUE_CLASS(CompareInst);
  VALUE_CLASS(ExceptInst);
  VALUE_CLASS(DeoptimizeIfInst);
  VALUE_CLASS(UnreachableInst);
  VALUE_CLASS(ReturnInst);
  VALUE_CLASS(IncrefInst);
  VALUE_CLASS(DecrefInst);
  VALUE_CLASS(LoadInst);
  VALUE_CLASS(LoadGlobalInst);
  VALUE_CLASS(StoreInst);
  VALUE_CLASS(FrameVariableInst);
  VALUE_CLASS(NegateInst);
  VALUE_CLASS(NotInst);
  VALUE_CLASS(AddInst);
  VALUE_CLASS(SubtractInst);
  VALUE_CLASS(MultiplyInst);
  VALUE_CLASS(DivideInst);
  VALUE_CLASS(RemainderInst);
  VALUE_CLASS(AndInst);
  VALUE_CLASS(OrInst);
  VALUE_CLASS(XorInst);
  VALUE_CLASS(ShiftLeftInst);
  VALUE_CLASS(ShiftRightSignedInst);
  VALUE_CLASS(IntToFloatInst);
  VALUE_CLASS(SextInst);
  VALUE_CLASS(CallNativeInst);
  VALUE_CLASS(CallPythonInst);
  VALUE_CLASS(CallAttributeInst);
  VALUE_CLASS(CallNativeIndirectInst);
  VALUE_CLASS(BytecodeBeginInst);
  VALUE_CLASS(AdvanceProfileCounterInst);
  VALUE_CLASS(IncrementEventCounterInst);
  VALUE_CLASS(TraceBeginInst);
  VALUE_CLASS(TraceEndInst);
  VALUE_CLASS(YieldValueInst);
  VALUE_CLASS(BoxInst);
  VALUE_CLASS(UnboxInst);
  VALUE_CLASS(OverflowedInst);
  VALUE_CLASS(FloatZeroInst);
  VALUE_CLASS(DeoptimizeIfSafepointInst);
  VALUE_CLASS(RematerializeInst);
  VALUE_CLASS(GetClassIdInst);
  VALUE_CLASS(GetObjectDictInst);
  VALUE_CLASS(GetInstanceClassIdInst);
  VALUE_CLASS(CheckClassIdInst);
  VALUE_CLASS(LoadFromDictInst);
  VALUE_CLASS(StoreToDictInst);
  VALUE_CLASS(ConstantAttributeInst);
  VALUE_CLASS(DeoptimizedAsynchronouslyInst);
  VALUE_CLASS(CallVectorcallInst);
  VALUE_CLASS(SetObjectClassInst);
  // LINT.ThenChange(value.h)

  return F::Default(args...);
#undef VALUE_CLASS
}

// ForAllValues wrapper that takes an additional filter predicate to
// compile-time limit the set of classes iterated over. F is only instantiated
// and called for Value subclasses that match FilterPredicate.
template <template <typename ValueCls> class FilterPredicate, typename F,
          typename... Args>
constexpr auto ForFilteredValues(Args&&... args) {
  return ForAllValues<detail::ValueFilter<FilterPredicate, F, Args...>,
                      Args...>(std::forward<Args>(args)...);
}

// A compile-time generated opaque bit set indexed by value kind, used to
// represent a set of Value classes matching some predicate.
class ValueSet {
  // A matcher for use with ForAllValues that evaluates to true for the Value
  // class associated with kind if Predicate evaluates to true for that class.
  template <template <typename ValueCls> class Predicate>
  struct Matcher {
    template <typename ValueCls>
    static constexpr absl::optional<bool> Visit(int kind) {
      if (ValueCls::kKind == kind) {
        return Predicate<ValueCls>::value;
      }
      return {};
    }
    static constexpr bool Default(int) { return false; }
  };

 public:
  // Construct an ValueSet from a predicate, evaluated for all Value subclasses.
  template <template <typename ValueCls> class Predicate>
  static constexpr ValueSet FromPredicate() {
    ValueSet set{};
    for (int kind = 0; kind <= Value::kMaxValue; ++kind) {
      if (ForAllValues<Matcher<Predicate>>(kind)) set.bits_.SetBit(kind);
    }
    return set;
  }

  // Construct an ValueSet from a predicate, evaluated for all Value subclasses
  // that match FilterPredicate.
  template <template <typename ValueCls> class FilterPredicate,
            template <typename ValueCls> class Predicate>
  static constexpr ValueSet FromPredicate() {
    ValueSet set{};
    for (int kind = 0; kind <= Value::kMaxValue; ++kind) {
      if (ForFilteredValues<FilterPredicate, Matcher<Predicate>>(kind))
        set.bits_.SetBit(kind);
    }
    return set;
  }

  // Construct a ValueSet from an initializer_list of instruction kinds.
  constexpr ValueSet(const std::initializer_list<size_t>& insns)
      : bits_(insns) {}

  // Test whether a given value kind is a member of the set.
  bool Contains(Value::Kind insn) const { return bits_.IsSet(insn); }

 private:
  FixedLengthBitVector<Value::kMaxValue + 1> bits_;
};

// Traits valid for all Value subclasses.
class ValueTraits {
 public:
  template <typename Base>
  static bool IsA(const Value& value) {
    return BaseOf<Base>::kDerivedClasses.Contains(value.kind());
  }

 private:
  template <typename Base>
  struct BaseOf {
    template <typename Derived>
    struct DerivesFrom {
      static constexpr bool value = std::is_base_of<Base, Derived>::value;
    };
    static constexpr ValueSet kDerivedClasses =
        ValueSet::FromPredicate<DerivesFrom>();
  };
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_TRAITS_H_
