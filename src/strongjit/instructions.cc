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

#include "strongjit/instructions.h"

#include <cstdint>

#include "strongjit/base.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/value_casts.h"

namespace deepmind::s6 {

namespace {
template <typename Cls>
struct PreciseLocationValues {
  static constexpr bool value =
      std::is_base_of<PreciseLocationInst, Cls>::value;
};

struct PreciseLocationPtr {
  template <typename I>
  static absl::optional<PreciseLocationInst*> Visit(Value* inst) {
    if (I::kKind != inst->kind()) {
      return {};
    }
    return static_cast<PreciseLocationInst*>(static_cast<I*>(inst));
  }
  static PreciseLocationInst* Default(Value*) { return nullptr; }
};
}  // namespace

const PreciseLocationInst* PreciseLocationInst::Get(const Value* inst) {
  return ForFilteredValues<PreciseLocationValues, PreciseLocationPtr>(
      const_cast<Value*>(inst));
}

PreciseLocationInst* PreciseLocationInst::Get(Value* inst) {
  return ForFilteredValues<PreciseLocationValues, PreciseLocationPtr>(inst);
}

absl::Span<Block* const> TerminatorInst::successors() {
  return span_cast<Block>(operands()).subspan(0, successor_size());
}

absl::Span<const Block* const> TerminatorInst::successors() const {
  return span_cast<Block>(operands()).subspan(0, successor_size());
}
absl::Span<Block*> TerminatorInst::mutable_successors() {
  return span_cast<Block>(mutable_operands().subspan(0, successor_size()));
}

const Block* UnconditionalTerminatorInst::unique_successor() const {
  return cast<Block>(operands().front());
}

Block* UnconditionalTerminatorInst::unique_successor() {
  return cast<Block>(operands().front());
}

Block* ConditionalTerminatorInst::true_successor() {
  return cast<Block>(operands().at(0));
}

const Block* ConditionalTerminatorInst::true_successor() const {
  return cast<Block>(operands().at(0));
}

Block* ConditionalTerminatorInst::false_successor() {
  return cast<Block>(operands().at(1));
}

const Block* ConditionalTerminatorInst::false_successor() const {
  return cast<Block>(operands().at(1));
}

}  // namespace deepmind::s6
