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

#include "strongjit/instruction.h"

#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"

namespace deepmind::s6 {

namespace {
struct InstrCloner {
  template <typename InstrType>
  static absl::optional<Instruction*> Visit(
      const Instruction* inst, Function& function,
      const absl::flat_hash_map<const Value*, Value*>& mapping) {
    if (inst->kind() != InstrType::kKind) return {};

    auto cloned_instr = function.Create<InstrType>(*cast<InstrType>(inst));

    // Update operands using the value mapping.
    for (auto& op : cloned_instr->mutable_operands()) {
      auto find_it = mapping.find(op);
      if (find_it != mapping.end()) {
        op = find_it->second;
      }
    }

    return cloned_instr;
  }

  static Instruction* Default(
      const Instruction* instr, Function& function,
      const absl::flat_hash_map<const Value*, Value*>& mapping) {
    S6_LOG(FATAL) << "Unknown instruction kind in clone";
    return nullptr;
  }
};
}  // namespace

// Provide vtable anchor.
InstructionModificationListener::~InstructionModificationListener() {}

void Instruction::CallListenerErased() {
  if (!function_) return;
  if (auto listener = function_->listener()) {
    listener->InstructionErased(this);
  }
}

void Instruction::CallListenerMutated() {
  if (!function_) return;
  if (auto listener = function_->listener()) {
    listener->OperandsMayBeModified(this);
  }
}

Instruction* Instruction::CloneWithNewOperands(
    Function& function,
    const absl::flat_hash_map<const Value*, Value*>& mapping) const {
  auto clone = ForAllInstructionKinds<InstrCloner>(this, function, mapping);
  S6_CHECK(clone != nullptr) << "Null pointer returned by InstrCloner";
  return clone;
}

}  // namespace deepmind::s6
