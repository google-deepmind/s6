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

#include "strongjit/function.h"

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "strongjit/instructions.h"
#include "type_feedback.h"

namespace deepmind::s6 {

Function::FixedLengthValue::~FixedLengthValue() {
  Instruction* inst = reinterpret_cast<Instruction*>(padding_.data());
  inst->~Instruction();
}

ValueNumbering ComputeValueNumbering(const Function& f) {
  ValueNumbering value_numbering;

  for (const Block& block : f) {
    value_numbering[&block] = value_numbering.size();
    for (const BlockArgument* arg : block.block_arguments()) {
      value_numbering[arg] = value_numbering.size();
    }
    for (const Instruction& inst : block) {
      value_numbering[&inst] = value_numbering.size();
    }
  }
  return value_numbering;
}

absl::Status VerifyFunction(const Function& f) {
  ValueNumbering vn = ComputeValueNumbering(f);

  for (const Block& b : f) {
    if (!b.GetTerminator()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Block &", vn[&b], " has no terminator"));
    }
    for (const Block* pred : b.predecessors()) {
      if (!vn.contains(pred)) {
        // Unfortunately we can't rely on formatter here, so we can't
        // pretty-print the result. The user can always print the function on
        // failure.
        return absl::FailedPreconditionError(
            "Block has invalid predecessor list");
      }
      if (!absl::c_linear_search(pred->GetTerminator()->successors(), &b)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not predecessor's successor! (succ=&",
                         vn[&b], ", pred=&", vn[pred], ")"));
      }
      const TerminatorInst* pred_term = pred->GetTerminator();
      if (!pred_term) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block &", vn[pred], " has no terminator"));
      }
      if (pred_term->successor_arguments(&b).size() +
              pred_term->num_implicit_arguments() !=
          b.block_arguments_size()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Block &", vn[&b], " has ", b.block_arguments_size(),
            " arguments, but is jumped to from &", vn[pred], " with ",
            pred->GetTerminator()->successor_arguments(&b).size(),
            " arguments"));
      }
      switch (b.handler_kind()) {
        case Block::HandlerKind::kNot:
          if (isa<ExceptInst>(pred_term)) {
            return absl::FailedPreconditionError(
                absl::StrCat("Block &", vn[pred], " is excepting to &", vn[&b],
                             " which is not a handler"));
          }
          break;
        case Block::HandlerKind::kExcept:
          if (!isa<ExceptInst>(pred_term)) {
            return absl::FailedPreconditionError(
                absl::StrCat("Block &", vn[pred], " is jumping normally to &",
                             vn[&b], " which is an except handler"));
          }
          break;
        case Block::HandlerKind::kFinally:
          break;
      }
    }

    for (const Instruction& i : b) {
      for (const Value* v : i.operands()) {
        if (v && !vn.contains(v)) {
          return absl::FailedPreconditionError(
              "Instruction uses undefined value");
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SnapshotTypeFeedback(
    absl::Span<absl::InlinedVector<ClassDistribution, 1> const> type_feedback,
    Function& f) {
  int64_t index = 0;
  for (const absl::InlinedVector<ClassDistribution, 1>& distributions :
       type_feedback) {
    PcValue pc = PcValue::FromIndex(index++);
    int64_t operand_index = 0;
    for (const ClassDistribution& distribution : distributions) {
      if (!distribution.empty()) {
        f.type_feedback()[{pc, operand_index}] = distribution.Summarize();
      }
      ++operand_index;
    }
  }
  return absl::OkStatus();
}

}  // namespace deepmind::s6
