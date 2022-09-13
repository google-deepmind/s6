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

#include "strongjit/optimizer_util.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "core_util.h"
#include "event_counters.h"
#include "strongjit/formatter.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/value.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {

////////////////////////////////////////////////////////////////////////////////
// Uses management

UseLists ComputeUses(Function& f) {
  UseLists uses;
  for (Block& b : f) {
    for (Instruction& inst : b) {
      int64_t i = 0;
      for (Value* operand : inst.operands()) {
        uses[operand].push_back({&inst, i});
        ++i;
      }
    }
  }
  return uses;
}

void ReplaceAllUsesWith(UseLists& uses, Value* from, Value* to) {
  auto it = uses.find(from);
  if (it == uses.end()) return;

  // Copy the current use list so we can mutate `uses`.
  Uses copied_uses = it->second;

  for (const Use& use : copied_uses) {
    use.user->mutable_operands()[use.operand_index] = to;

    // This access invalidates "it".
    uses[to].push_back(use);
  }

  // Don't reuse "it" here, since it has now been invalidated.
  uses.erase(from);
}

////////////////////////////////////////////////////////////////////////////////
// Constant conversion functions

bool IsConstantInstruction(const Value* v) {
  if (isa<ConstantAttributeInst>(v) || isa<ConstantInst>(v)) return true;

  if (auto* frame_inst = dyn_cast<FrameVariableInst>(v)) {
    if (frame_inst->frame_variable_kind() ==
            FrameVariableInst::FrameVariableKind::kConsts ||
        frame_inst->frame_variable_kind() ==
            FrameVariableInst::FrameVariableKind::kNames) {
      return true;
    }
  }
  return false;
}

absl::optional<int64_t> GetValueAsConstantInt(Value* v, PyCodeObject* code) {
  if (auto* attr_inst = dyn_cast<ConstantAttributeInst>(v)) {
    return reinterpret_cast<int64_t>(
        attr_inst->LookupAttribute(ClassManager::Instance()).value());
  }
  if (auto* constant = dyn_cast<ConstantInst>(v)) {
    return constant->value();
  }
  if (auto* frame_inst = dyn_cast<FrameVariableInst>(v)) {
    if (code) {
      switch (frame_inst->frame_variable_kind()) {
        case FrameVariableInst::FrameVariableKind::kConsts:
          return reinterpret_cast<int64_t>(
              PyTuple_GET_ITEM(code->co_consts, frame_inst->index()));
        case FrameVariableInst::FrameVariableKind::kNames:
          return reinterpret_cast<int64_t>(
              PyTuple_GET_ITEM(code->co_names, frame_inst->index()));
        default:
          break;
      }
    }
  }
  return absl::nullopt;
}

absl::optional<absl::string_view> GetPyObjectAsString(PyObject* obj) {
  if (!obj) return absl::nullopt;
  if (!PyUnicode_CHECK_INTERNED(obj)) return absl::nullopt;

  // If the string is interned, its lifetime won't end so it is safe to return
  // it as a string_view without copying.

  Py_ssize_t size;
  const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
  if (!data) return absl::nullopt;
  return absl::string_view(data, size);
}

absl::optional<std::vector<absl::string_view>> GetPyObjectAsTupleOfStrings(
    PyObject* tuple) {
  if (!tuple) return absl::nullopt;
  if (!PyTuple_Check(tuple)) return absl::nullopt;
  std::vector<absl::string_view> result;
  for (int64_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
    PyObject* str = PyTuple_GET_ITEM(tuple, i);
    auto opt = GetPyObjectAsString(str);
    if (!opt) return absl::nullopt;
    result.push_back(*opt);
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Rewriter & Pattern

namespace {
bool ApplyPatterns(
    Value* value, Rewriter& rewriter,
    const absl::flat_hash_map<Value::Kind, std::vector<const Pattern*>>&
        anchors) {
  auto it = anchors.find(value->kind());
  if (it == anchors.end()) return false;
  for (const Pattern* pattern : it->second) {
    absl::Status okay = pattern->Apply(value, rewriter);
    if (okay.ok()) {
      S6_VLOG(3) << "Applied pattern " << pattern->name();
      return true;
    }
    S6_VLOG(10) << "  Could not apply pattern " << pattern->name() << ": "
                << okay;
  }
  return false;
}
}  // namespace

absl::Status RewritePatterns(Function& f, PyCodeObject* code,
                             absl::Span<const Pattern* const> patterns,
                             const OptimizerOptions& options) {
  // Create a map of anchor kind to pattern.
  absl::flat_hash_map<Value::Kind, std::vector<const Pattern*>> anchors;
  for (const Pattern* pattern : patterns) {
    anchors[pattern->anchor()].push_back(pattern);
  }

  Rewriter rewriter(f, code, options);

  // Run until fixpoint (i.e. no more changes occur).
  constexpr int64_t kMaxFixpointIterations = 10;
  int64_t fixpoint_iterations = kMaxFixpointIterations;
  bool changed = true;
  while (fixpoint_iterations > 0 && changed) {
    changed = false;
    Block* previous_block = nullptr;
    for (auto cursor = f.FirstInstruction(); !cursor.Finished();) {
      Block* b = cursor.GetBlock();

      // If we've just changed block, check if the block is orphaned and run
      // patterns on it.
      if (b != previous_block) {
        previous_block = b;
        if (b->predecessors().empty() && b != &f.entry()) {
          // This block is orphaned (and is not the entry block).
          while (!cursor.Finished() && cursor.GetBlock() == b) {
            cursor.StepForward();
          }
          for (Block* succ : b->GetTerminator()->successors()) {
            S6_CHECK(succ);
            succ->RemovePredecessor(b);
          }

          // When removing the block, take care to ensure all blocks that may
          // depend on a value from this block don't depend on invalid values.
          // Any value that this affects will eventually be removed, but
          // temporarily they may still exist.
          // TODO: Implement a Poison value that can be used here.
          Value* arbitrary = &*f.entry().begin();
          while (!b->empty()) {
            rewriter.ReplaceAllUsesWith(*b->begin(), *arbitrary);
            b->begin()->erase();
          }
          for (BlockArgument* arg : b->block_arguments()) {
            rewriter.ReplaceAllUsesWith(*arg, *arbitrary);
          }
          b->erase();

          changed = true;
        }
        if (cursor.Finished()) break;

        rewriter.SetCursor(&cursor);
        changed |= ApplyPatterns(b, rewriter, anchors);
      }
      S6_CHECK(!cursor.Finished());

      rewriter.SetCursor(&cursor);
      Instruction* inst = cursor.GetInstruction();
      if (!InstructionTraits::HasSideEffects(*inst) &&
          rewriter.GetUsesOf(*inst).empty()) {
        // This instruction is dead.
        inst->erase();
        changed = true;
      } else {
        changed |= ApplyPatterns(inst, rewriter, anchors);
      }
      if (!cursor.Finished()) {
        cursor.StepForward();
      }
    }
    --fixpoint_iterations;
  }

  if (changed) {
    return absl::DeadlineExceededError(
        absl::StrCat("Function did not optimize to a fixpoint within ",
                     kMaxFixpointIterations, " iterations"));
  }
  return absl::OkStatus();
}

Pattern::~Pattern() {}

Rewriter::Rewriter(Function& f, PyCodeObject* code_object,
                   const OptimizerOptions& options)
    : function_(f),
      code_object_(code_object),
      options_(options),
      use_lists_(ComputeUses(f)),
      cursor_(nullptr) {
  function_.SetInstructionModificationListener(this);
}

Rewriter::~Rewriter() { function_.ClearInstructionModificationListener(); }

void Rewriter::EnsureUsesValid() {
  if (to_update_.empty()) return;

  EventCounters::Instance().Add("optimizer.rewriter.use_lists_invalidated", 1);
  do {
    Instruction& inst = **to_update_.begin();
    to_update_.erase(&inst);
    AddOperands(inst);
  } while (!to_update_.empty());
}

const Uses& Rewriter::GetUsesOf(Value& v) {
  EnsureUsesValid();
  return use_lists_[&v];
}

void Rewriter::ReplaceAllUsesWith(Value& from, Value& to) {
  EnsureUsesValid();

  auto it = use_lists_.find(&from);
  if (it == use_lists_.end()) return;

  // Copy the current use list so we can mutate `uses`.
  Uses copied_uses = it->second;

  // Remove our listener; we're aware of what we're doing.
  function_.ClearInstructionModificationListener();
  for (Use& use : copied_uses) {
    use.user->mutable_operands()[use.operand_index] = &to;

    // This access invalidates "it".
    use_lists_[&to].push_back(use);
  }

  // Don't reuse "it" here, since it has now been invalidated.
  use_lists_.erase(&from);

  // Listen to new events again.
  function_.SetInstructionModificationListener(this);
}

void Rewriter::ReplaceAllSafepointUsesWith(Value& from, Value& to) {
  EnsureUsesValid();

  auto it = use_lists_.find(&from);
  if (it == use_lists_.end()) return;

  // Copy the current use list so we can mutate `uses`.
  Uses copied_uses = it->second;

  // Remove our listener; we're aware of what we're doing.
  function_.ClearInstructionModificationListener();
  for (Use& use : copied_uses) {
    if (SafepointInst* si = dyn_cast<SafepointInst>(use.user)) {
      auto value_stack = si->mutable_value_stack();
      auto fastlocals = si->mutable_fastlocals();
      if (absl::c_find(value_stack, &from) != value_stack.end() ||
          absl::c_find(fastlocals, &from) != fastlocals.end()) {
        // Invalidate the use lists wherever this safepoint_inst is the user.
        // Do so before the mutation to ensure that it is removed from `from`'s
        // use list.
        OperandsMayBeModified(si);

        absl::c_replace(value_stack, &from, &to);
        absl::c_replace(fastlocals, &from, &to);

        // The `from` may also occur amongst the other operands, e.g.
        // the value to be yielded in yield_value. Leave such
        // occurrences unchanged.
      }
    }
  }

  // Listen to new events again.
  function_.SetInstructionModificationListener(this);
}

void Rewriter::EraseAllRefcountUses(Value& v) {
  EnsureUsesValid();

  auto it = use_lists_.find(&v);
  if (it == use_lists_.end()) return;

  // Copy the current use list so we can mutate the main one.
  Uses copied_uses = it->second;

  for (Use& use : copied_uses) {
    if (isa<RefcountInst>(use.user)) {
      use.user->erase();
    }
    if (SafepointInst* si = dyn_cast<SafepointInst>(use.user)) {
      if (absl::c_linear_search(si->increfs(), &v) ||
          absl::c_linear_search(si->decrefs(), &v)) {
        STLEraseAll(unmove(si->mutable_increfs()), &v);
        STLEraseAll(unmove(si->mutable_decrefs()), &v);
      }
    }
  }
}

void Rewriter::EraseOperands(const Instruction& inst) {
  for (const Value* operand : inst.operands()) {
    auto it = use_lists_.find(operand);
    if (it == use_lists_.end()) continue;
    auto& uses = it->second;
    STLEraseIf(uses, [&](const Use& use) { return use.user == &inst; });
  }
}

void Rewriter::erase(Instruction& inst) {
  // Calls InstructionErased.
  inst.erase();
}

void Rewriter::InstructionErased(Instruction* inst) {
  S6_CHECK(use_lists_[inst].empty())
      << "Uses existed when instruction was erased!";

  EraseOperands(*inst);
  to_update_.erase(inst);

  if (cursor_ && !cursor_->Finished() && cursor_->GetInstruction() == inst) {
    cursor_->StepForward();
  } else if (cursor_ && !cursor_->Finished()) {
    // Keep cursor_ valid.
    *cursor_ = Cursor(cursor_->GetInstruction());
  }
}

void Rewriter::AddOperands(Instruction& inst) {
  int64_t operand_index = 0;
  for (Value* operand : inst.operands()) {
    use_lists_[operand].push_back({&inst, operand_index++});
  }
}

// TODO: May be this should also update the cursor_?
void Rewriter::InstructionAdded(Instruction* inst) { AddOperands(*inst); }

void Rewriter::OperandsMayBeModified(Instruction* inst) {
  EraseOperands(*inst);
  to_update_.insert(inst);
}

// TODO: Changing operand list shape may be unsound. Remove this?
void Rewriter::RemoveUsesOfIn(Instruction& user, Value& use) {
  user.EraseFromOperandList(&use);
}

void Rewriter::ReplaceAt(Instruction& user, int64_t index, Value& to) {
  function_.ClearInstructionModificationListener();
  Value& from = *user.operands()[index];
  user.mutable_operands()[index] = &to;

  Uses& from_uses = use_lists_[&from];
  STLEraseAll(from_uses, Use{&user, index});
  use_lists_[&to].push_back(Use{&user, index});

  function_.SetInstructionModificationListener(this);
}

void Rewriter::ReplaceUsesWith(Instruction& user, Value& from, Value& to) {
  function_.ClearInstructionModificationListener();
  user.ReplaceUsesOfWith(&from, &to);

  STLEraseIf(use_lists_[&from], [&](const Use& u) { return u.user == &user; });

  Uses& to_uses = use_lists_[&to];
  int64_t user_operand_index = 0;
  for (const Value* user_operand : user.operands()) {
    if (user_operand == &to) {
      to_uses.push_back({&user, user_operand_index});
    }
    ++user_operand_index;
  }
  function_.SetInstructionModificationListener(this);
}

void Rewriter::ReplaceUsesWith(Block& block, Value& from, Value& to) {
  for (Instruction& inst : block) ReplaceUsesWith(inst, from, to);
}

void Rewriter::ConvertBranchToJump(BrInst& br, bool true_successor) {
  auto it = br.GetIterator();
  Block* block = br.parent();

  BlockInserter inserter(block, it);
  if (true_successor) {
    inserter.Create<JmpInst>(br.true_successor(), br.true_arguments());
    br.false_successor()->RemovePredecessor(block);
  } else {
    inserter.Create<JmpInst>(br.false_successor(), br.false_arguments());
    br.true_successor()->RemovePredecessor(block);
  }
  erase(br);
}

////////////////////////////////////////////////////////////////////////////////
// Standalone utilities

absl::StatusOr<BytecodeBeginInst*> BeginningOfBytecodeInstruction(
    Instruction* pynum_op) {
  Block::iterator it = pynum_op->GetIterator();
  while (it != pynum_op->parent()->begin()) {
    --it;

    if (auto* bbi = dyn_cast<BytecodeBeginInst>(&*it); bbi != nullptr) {
      return bbi;
    }
    // The opcode can legitimately be preceded by side-effect-free instructions.
    // Re-attempting them will be safe.
    if (InstructionTraits::HasSideEffects(*it) &&
        !isa<AdvanceProfileCounterInst>(*it)) {
      return absl::FailedPreconditionError(
          "Instruction was not preceded by a bytecode_begin!");
    }
  }
  return absl::FailedPreconditionError("No bytecode_begin found in block!");
}

std::vector<const Value*> ReverseValueNumbering(
    const ValueNumbering& numbering) {
  std::vector<const Value*> res;
  res.resize(numbering.size());
  for (auto [value, index] : numbering) res[index] = value;
  return res;
}

absl::Status DelayDecref(DecrefInst& decref,
                         absl::Span<Instruction* const> targets) {
  // First we check that the operation is correct.
  // We want to check that the set of path that goes from the decref to each
  // target does not cross any other targets.
  // To do that we check that going backward from each target, we can only go
  // to the dominating decref without crossing another target.
  // At the same time we check the separate precondition that the Decref must
  // dominate the targets without which this operation is not feasible.

  Function& f = *decref.function();
  Block* decref_block = decref.parent();
  Value* decreffed_object = decref.operand();

  // middle_blocks records all the blocks that are part of such paths.
  // Because of this variable, the first pass of this function is not just a
  // check middle_block is also required to perform the second pass of the
  // algorithm.
  absl::flat_hash_set<Block*> middle_blocks;
  middle_blocks.insert(decref_block);

  absl::flat_hash_map<Block*, Instruction*> target_blocks;
  for (Instruction* target : targets) {
    target_blocks.try_emplace(target->parent(), target);
  }
  if (target_blocks.size() != targets.size()) {
    return absl::FailedPreconditionError(
        "There are two targets in the same block");
  }

  Worklist<Block*> worklist;
  for (auto [target_block, target] : target_blocks) {
    worklist.Push(target_block);
  }

  while (!worklist.empty()) {
    Block* b = worklist.Pop();
    if (b == decref_block) continue;
    if (b == &f.entry()) {
      return absl::FailedPreconditionError(
          "Decref does not dominate some target");
    }
    middle_blocks.insert(b);
    for (Block* pred : b->predecessors()) {
      if (target_blocks.contains(pred)) {
        return absl::FailedPreconditionError(
            "There is a path from one target to another");
      }
      if (middle_blocks.contains(pred)) continue;
      worklist.PushIfNew(pred);
    }
  }

  // Now we know that the path between the decref and the targets are clean.
  // We can commit ourselves to doing the operation.
  for (Instruction* target : targets) {
    Block* target_block = target->parent();
    target_block->insert(target->GetIterator(), decref.Clone());
  }

  // We also need to cover all paths leaving the decref without hitting a target
  // In those path, all safepoint increfs field must receive a pointer to the
  // object because it must be increffed again when taking the safepoint.

  // The set of block where a decref was added at the beginning because any path
  // from the decref through that block will never reach a target.
  absl::flat_hash_set<Block*> decref_added;

  auto DecrefInSafepointUntil = [&](Block& block, Block::iterator end) {
    auto it = block.begin();
    if (&block == decref_block) it = decref.GetIterator();
    for (; it != end; ++it) {
      if (auto* safepoint = dyn_cast<SafepointInst>(&*it)) {
        safepoint->decref_value(decreffed_object);
      }
    }
  };

  for (Block* block : middle_blocks) {
    if (target_blocks.contains(block)) {
      // Any path crossing a target_block will always reach a target.
      // We only need to deal with the safepoint part of the problem.
      Instruction* target = target_blocks.at(block);
      DecrefInSafepointUntil(*block, target->GetIterator());

      continue;
    }

    DecrefInSafepointUntil(*block, block->end());

    TerminatorInst* terminator = block->GetTerminator();
    auto successors = terminator->successors();
    if (successors.empty()) continue;
    if (successors.size() == 1) {
      // If a block is in middle block that means there is one path through it
      // that goes to a target. Since there is only one successor, that
      // successor also can go to the target and so is also in middle_blocks.
      S6_CHECK(middle_blocks.contains(successors.front()));
      continue;
    }
    S6_DCHECK_GE(successors.size(), 2);
    for (auto succ : successors) {
      if (middle_blocks.contains(succ)) continue;
      if (decref_added.contains(succ)) continue;
      decref_added.insert(succ);
      succ->insert(succ->begin(), decref.Clone());
    }
  }
  // Finally we can erase the original decref.
  decref.erase();
  return absl::OkStatus();
}

void SplitCriticalEdges(Function& f) {
  for (Block& pred : f) {
    // If this block has only one successor, the successor edges cannot be
    // critical.
    TerminatorInst* ti = pred.GetTerminator();
    if (ti->successor_size() <= 1) continue;

    for (int64_t i = 0; i < ti->successor_size(); ++i) {
      Block* succ = ti->successors()[i];
      S6_CHECK(succ);
      if (succ->predecessors().size() <= 1) continue;

      // This edge must be split.
      Block* epsilon = f.CreateBlock(succ->GetIterator());
      Builder builder(epsilon);
      builder.Jmp(succ, ti->successor_arguments(i));

      // We're transferring the successor arguments of `ti`, so remove them now.
      ti->mutable_successor_arguments(i).clear();

      epsilon->AddPredecessor(&pred);
      succ->ReplacePredecessor(&pred, epsilon);
      ti->ReplaceUsesOfWith(succ, epsilon);
    }

    // Note that splitting critical edges never introduces more critical edges,
    // so we do not need to revisit any edges.
  }
}

////////////////////////////////////////////////////////////////////////////////
// BuiltinObjects

namespace {
void IterateDict(PyObject* dict,
                 absl::FunctionRef<void(PyObject*, PyObject*)> fn) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(dict, &pos, &key, &value)) {
    fn(key, value);
  }
}
}  // namespace

BuiltinObjects& BuiltinObjects::Instance() {
  static NoDestructor<BuiltinObjects> objects;
  return *objects;
}

void BuiltinObjects::Initialize() {
  // Ensure some builtin modules are imported.
  Py_DECREF(PyImport_ImportModule("math"));

  PyObject* modules = PyThreadState_GET()->interp->modules;
  IterateDict(modules, [&](PyObject* key, PyObject* value) {
    if (!PyUnicode_CheckExact(key)) return;

    absl::string_view key_str = GetObjectAsCheapStringRequiringGil(key);
    builtin_objects_[key_str] = value;

    // We also look one level deeper, to find objects like `math.sin`.
    if (!PyModule_CheckExact(value)) return;
    IterateDict(PyModule_GetDict(value), [&](PyObject* key, PyObject* value) {
      if (!PyUnicode_CheckExact(key)) return;
      absl::string_view subkey_str = GetObjectAsCheapStringRequiringGil(key);
      builtin_objects_[absl::StrCat(key_str, ".", subkey_str)] = value;
    });
  });
}

PyObject* BuiltinObjects::LookupBuiltin(absl::string_view name) {
  auto it = builtin_objects_.find(name);
  return it == builtin_objects_.end() ? nullptr : it->second;
}

}  // namespace deepmind::s6
