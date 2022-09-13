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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_REGISTER_ALLOCATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_REGISTER_ALLOCATOR_H_

#include <array>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "asmjit/asmjit.h"
#include "code_generation/live_interval.h"
#include "core_util.h"
#include "runtime/deoptimization_map.h"
#include "runtime/slot_indexes.h"
#include "strongjit/base.h"
#include "strongjit/block.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction.h"
#include "strongjit/instructions.h"
#include "strongjit/ssa.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"

namespace deepmind::s6 {

// LINT.IfChange
// The default set of allocatable registers. Note that r11 is used as a scratch
// register so is unallocatable, and r12 is used as the profile counter
// register.
// TODO: Move out of here into abi.h when it exists.
inline const std::array<asmjit::x86::Reg, 11> kAllocatableRegisters = {
    {asmjit::x86::rax, asmjit::x86::rcx, asmjit::x86::rdx, asmjit::x86::rsi,
     asmjit::x86::rdi, asmjit::x86::r8, asmjit::x86::r9, asmjit::x86::r10,
     asmjit::x86::r14, asmjit::x86::r15, asmjit::x86::r13}};
// LINT.ThenChange(runtime/runtime.cc)

// r13 is used to access spill slots in generator functions.
inline const asmjit::x86::Gp kGeneratorSpillSlotReg = asmjit::x86::r13;

// So the set of allocatable registers for a generator function does not include
// r13.
inline const std::array<asmjit::x86::Reg, 10>
    kAllocatableRegistersForGenerator = {
        {asmjit::x86::rax, asmjit::x86::rcx, asmjit::x86::rdx, asmjit::x86::rsi,
         asmjit::x86::rdi, asmjit::x86::r8, asmjit::x86::r9, asmjit::x86::r10,
         asmjit::x86::r14, asmjit::x86::r15}};

// Parameterizes the register allocator - primarily intended to write easier
// test cases.
struct RegisterAllocationOptions {
  // The set of allocatable registers, in register allocation order.
  absl::Span<asmjit::x86::Reg const> allocatable_registers =
      kAllocatableRegisters;

  // The threshold below which constant instructions will be considered as
  // materializable as immediates.
  int64_t immediate_threshold = 65535;
};

// An allocation of registers to a Function. To allow for different algorithm
// implementations this is an abstract class.
class RegisterAllocation {
 public:
  virtual ~RegisterAllocation();

  // A copy copies from a location to another location.
  using Copy = std::pair<Location, Location>;

  // Returns the output location of `v`.
  virtual Location DestinationLocation(const Value& v) const = 0;

  // Returns the input location of operand `operand` to instruction `inst`.
  virtual Location OperandLocation(const Instruction& inst,
                                   const Value& value) const = 0;

  // Returns the copies that must be performed at the end of `block`. These
  // must be architecturally performed after the block's terminator begins;
  // for example for a BrInst, the condition must be fetched *before* these
  // copies take effect.
  //
  // Copies are sequentially ordered and must be applied in-order for correct
  // behavior.
  //
  // For blocks with multiple successors, pass `successor_index` to get the
  // correct copy list.
  absl::Span<Copy const> block_copies(const Block* b,
                                      int64_t successor_index) const {
    if (block_copies_.contains({b, successor_index}))
      return block_copies_.at({b, successor_index});
    return {};
  }

  // Returns the copies that must be performed before `inst`.
  //
  // Copies are sequentially ordered and must be applied in-order for correct
  // behavior.
  absl::Span<Copy const> inst_copies(const Value* v) const {
    if (inst_copies_.contains(v)) return inst_copies_.at(v);
    return {};
  }

  virtual std::string ToString(const Function& f) const = 0;

  // Returns the number of frame slots used.
  virtual int64_t GetNumFrameSlots() const = 0;

  // Returns the number of used call stack slots.
  virtual int64_t GetNumCallStackSlots() const = 0;

  // Populates the given deoptimization mapping with all live ranges.
  virtual void PopulateDeoptimizationMap(
      DeoptimizationMap& deopt_map) const = 0;

  // Returns the set of all registers used.
  virtual std::vector<asmjit::x86::Gp> ComputeUsedRegisters() const = 0;

 protected:
  // Holds all generated copies. This is indexed by {slot, predicate} -
  // predicate is kAlways apart from BrInst which uses it to determine which
  // outgoing edge is being referred to.
  absl::flat_hash_map<std::pair<const Block*, int64_t>, std::vector<Copy>>
      block_copies_;

  // Holds all generated copies that occur before instructions.
  absl::flat_hash_map<const Value*, std::vector<Copy>> inst_copies_;
};

class RegisterAllocationAnnotator final : public AnnotationFormatter {
 public:
  ~RegisterAllocationAnnotator() final {}
  explicit RegisterAllocationAnnotator(const RegisterAllocation& ra)
      : ra_(ra) {}

  std::string AnnotateBlock(const Block& block, FormatContext* ctx) const {
    std::string s;
    for (const BlockArgument* arg : block.block_arguments()) {
      Location loc = ra_.DestinationLocation(*arg);
      if (loc.IsDefined()) {
        absl::StrAppend(&s, "def(", loc.ToString(), ") ");
      }
    }
    return std::string(absl::StripTrailingAsciiWhitespace(s));
  }

  std::string AnnotateInstruction(const Instruction& inst,
                                  FormatContext* ctx) const {
    Location loc = ra_.DestinationLocation(inst);
    std::string s;
    if (loc.IsDefined()) {
      absl::StrAppend(&s, "def(", loc.ToString(), ") ");
    }

    std::vector<std::string> strs;
    for (const Value* operand : inst.operands()) {
      if (!operand || isa<Block>(operand)) continue;
      Location loc = ra_.OperandLocation(inst, *operand);
      if (loc.IsDefined()) strs.push_back(loc.ToString());
    }
    if (!strs.empty()) {
      absl::StrAppend(&s, "use(", absl::StrJoin(strs, ", "), ") ");
    }
    absl::StrAppend(&s, AnnotateCopies(ra_.inst_copies(&inst), "copies"));

    // If this is a terminator, there may be copies afterwards.
    if (isa<TerminatorInst>(inst)) {
      std::string extra;
      const Block* b = cast<TerminatorInst>(inst).parent();
      if (isa<BrInst>(inst)) {
        absl::StrAppend(&extra,
                        AnnotateCopies(ra_.block_copies(b, 0), "copy-if-true"));
        absl::StrAppend(
            &extra, AnnotateCopies(ra_.block_copies(b, 1), "copy-if-false"));
      } else if (isa<DeoptimizeIfInst>(inst)) {
        absl::StrAppend(&extra,
                        AnnotateCopies(ra_.block_copies(b, 1), "copy-if-okay"));
      } else {
        absl::StrAppend(&extra,
                        AnnotateCopies(ra_.block_copies(b, 0), "copies"));
      }
      if (!extra.empty()) {
        s = std::string(absl::StripTrailingAsciiWhitespace(s));
        absl::StrAppend(&s, "\n", extra);
      }
    }
    return std::string(absl::StripTrailingAsciiWhitespace(s));
  }

  std::string AnnotateCopies(absl::Span<RegisterAllocation::Copy const> copies,
                             absl::string_view label) const {
    std::vector<std::string> strs;
    for (auto [from, to] : copies) {
      strs.push_back(absl::StrCat(from.ToString(), " -> ", to.ToString()));
    }
    if (strs.empty()) {
      return "";
    }
    return absl::StrCat(label, "(", absl::StrJoin(strs, ", "), ") ");
  }

  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final {
    if (const Block* block = dyn_cast<Block>(&v)) {
      return AnnotateBlock(*block, ctx);
    } else if (const Instruction* inst = dyn_cast<Instruction>(&v)) {
      return AnnotateInstruction(*inst, ctx);
    } else {
      return "";
    }
  }

 private:
  const RegisterAllocation& ra_;
};

// Returns the ABI register requirement for operand `operand_index` of `v`.
LocationRequirement GetRegisterRequirement(const Instruction* inst,
                                           int64_t operand_index);

// Returns the ABI register requirement for `v`.
LocationRequirement GetRegisterRequirement(const Value* v);

// Returns the ABI register requirement for `operand_index`.
Location GetAbiLocation(int64_t operand_index);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_REGISTER_ALLOCATOR_H_
