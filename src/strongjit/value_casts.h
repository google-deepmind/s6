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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_CASTS_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_CASTS_H_

#include "absl/types/span.h"
#include "strongjit/value_traits.h"
#include "utils/logging.h"
namespace deepmind::s6 {
class Value;

////////////////////////////////////////////////////////////////////////////////
// Casting - isa, dyn_cast, cast

// These helpers mirror those in LLVM - they can be used for lightweight
// dynamic_casting. Note that all of these functions return sane values on
// nullptr.

template <typename T>
bool isa(const Value* v) {
  if (!v) return false;
  return ValueTraits::IsA<T>(*v);
}

template <typename T>
bool isa(const Value& v) {
  return ValueTraits::IsA<T>(v);
}

template <typename T>
const T* cast(const Value* v) {
  if (!v) return nullptr;
  S6_DCHECK(isa<T>(v));
  return reinterpret_cast<const T*>(v);
}

template <typename T>
const T& cast(const Value& v) {
  S6_DCHECK(isa<T>(v));
  return reinterpret_cast<const T&>(v);
}

template <typename T>
T* cast(Value* v) {
  if (!v) return nullptr;
  S6_DCHECK(isa<T>(v));
  return reinterpret_cast<T*>(v);
}

template <typename T>
T& cast(Value& v) {  // NOLINT
  S6_DCHECK(isa<T>(v));
  return reinterpret_cast<T&>(v);
}

template <typename T>
const T* dyn_cast(const Value* v) {
  return isa<T>(v) ? reinterpret_cast<const T*>(v) : nullptr;
}

template <typename T>
T* dyn_cast(Value* v) {
  return isa<T>(v) ? reinterpret_cast<T*>(v) : nullptr;
}

template <typename T, typename U>
absl::Span<T*> span_cast(absl::Span<U*> s) {
  return absl::MakeSpan(reinterpret_cast<T**>(s.data()), s.size());
}

template <typename T, typename U>
absl::Span<T* const> span_cast(absl::Span<U* const> s) {
  return absl::MakeSpan(reinterpret_cast<T* const*>(s.data()), s.size());
}

template <typename T, typename U>
absl::Span<const T* const> span_cast(absl::Span<const U* const> s) {
  return absl::MakeSpan(reinterpret_cast<const T* const*>(s.data()), s.size());
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_VALUE_CASTS_H_
