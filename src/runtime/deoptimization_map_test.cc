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

#include "runtime/deoptimization_map.h"

#include "absl/algorithm/container.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "runtime/slot_indexes.h"
#include "strongjit/base.h"
#include "strongjit/builder.h"

namespace deepmind::s6 {
namespace {
using ValueAndLocation = DeoptimizationMap::ValueAndLocation;
using testing::UnorderedElementsAre;

TEST(DeoptimizationMapTest, LiveValueIteratorTest) {
  Function f("live_value_iterator_test");
  Builder b(&f);

  std::array<Instruction*, 8> values;
  absl::c_generate(values, [&]() { return b.Int64(0); });

  SlotIndexes slot_indexes(f);
  DeoptimizationMap deopt_map(slot_indexes);

  deopt_map.AddLiveValue(values[0], Location(), values[1]);
  deopt_map.AddLiveValue(values[0], Location(), values[2]);
  deopt_map.AddLiveValue(values[0], Location(), values[3]);
  deopt_map.AddLiveValue(values[1], Location(), values[2]);
  deopt_map.AddLiveValue(values[2], Location(), values[3]);
  deopt_map.AddLiveValue(values[2], Location(), values[4]);
  deopt_map.AddLiveValue(values[2], Location(), values[5]);

  deopt_map.AddInstructionAddress(values[0], 1, 2);
  deopt_map.AddInstructionAddress(values[1], 2, 3);
  deopt_map.AddInstructionAddress(values[2], 3, 4);

  EXPECT_THAT(deopt_map.live_values((ProgramAddress)0), UnorderedElementsAre());
  EXPECT_THAT(deopt_map.live_values(2),
              UnorderedElementsAre(ValueAndLocation{values[0], Location()}));
  EXPECT_THAT(deopt_map.live_values(3),
              UnorderedElementsAre(ValueAndLocation{values[0], Location()},
                                   ValueAndLocation{values[1], Location()}));
}

TEST(DeoptimizationMapTest, LiveFastLocals) {
  Function f("live_fastlocals_test");
  Builder b(&f);

  std::array<Instruction*, 8> values;
  absl::c_generate(values, [&]() { return b.Int64(0); });

  auto make_bytecode_begin = [&](absl::Span<Value* const> vs) {
    return b.inserter().Create<BytecodeBeginInst>(
        0, absl::Span<Value* const>{}, vs, absl::Span<TryHandler const>{});
  };

  // At `b1`, the fastlocals are {values[0], values[1]}.
  BytecodeBeginInst* b1 = make_bytecode_begin({values[0], values[1]});
  // At `b2`, the fastlocals are {values[2], values[1]}.
  BytecodeBeginInst* b2 = make_bytecode_begin({values[2], values[1]});
  // At `b3`, the fastlocals are {values[0]}.
  BytecodeBeginInst* b3 = make_bytecode_begin({values[0]});
  // At `b4`, the fastlocals are {values[0], values[0], values[2], values[4]}.
  BytecodeBeginInst* b4 =
      make_bytecode_begin({values[0], values[0], values[2], values[4]});

  SlotIndexes slot_indexes(f);
  DeoptimizationMap deopt_map(slot_indexes);

  EXPECT_THAT(deopt_map.GetLiveFastLocals(slot_indexes.SlotForValue(b1)),
              testing::ElementsAre(values[0], values[1]));
  EXPECT_THAT(deopt_map.GetLiveFastLocals(slot_indexes.SlotForValue(b2)),
              testing::ElementsAre(values[2], values[1]));
  EXPECT_THAT(deopt_map.GetLiveFastLocals(slot_indexes.SlotForValue(b3)),
              testing::ElementsAre(values[0]));
  EXPECT_THAT(deopt_map.GetLiveFastLocals(slot_indexes.SlotForValue(b4)),
              testing::ElementsAre(values[0], values[0], values[2], values[4]));
}

}  // namespace
}  // namespace deepmind::s6
