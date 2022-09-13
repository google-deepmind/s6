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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_UTIL_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "classes/class_manager.h"
#include "core_util.h"
#include "strongjit/builder.h"
#include "strongjit/function.h"
#include "strongjit/instruction.h"
#include "strongjit/instructions.h"
#include "strongjit/util.h"
#include "strongjit/value.h"
#include "strongjit/value_casts.h"
#include "tuple_util.h"

namespace deepmind::s6 {

////////////////////////////////////////////////////////////////////////////////
// Optimizer Options

// Options for optimization passes.
// TODO: A certain number of them are obsolete, remove them.
struct OptimizerOptions {
  // If true, use optimization info within the function
  // (OptimizationInfoEntries) to inform optimization decisions. If false, do
  // NOT base any decision on OptimizationInfoEntries.
  bool use_optimization_info = true;

  // The ClassManager to use. Only useful for testing.
  ClassManager& mgr = ClassManager::Instance();

  // If true, insert event counter instructions into generated code.
  bool use_event_counters = false;

  // If true, enable insertion of unbox/box instructions to perform successive
  // arithmetic operations using machine instructions.
  bool enable_unboxing_optimization = false;

  // If true, add tracing instructions to function entry and exit points to
  // enable function call trace events to be recorded.
  bool enable_function_tracing = false;

  // If true, the refcounting analysis will crash the program as soon as an
  // inconsistency in reference counts is found.
  bool harden_refcount_analysis = false;
};

////////////////////////////////////////////////////////////////////////////////
// Uses management

// A use of a value.
struct Use {
  Instruction* user;
  int64_t operand_index;

  bool operator==(const Use& oth) const {
    return user == oth.user && operand_index == oth.operand_index;
  }
  bool operator!=(const Use& oth) const { return !(*this == oth); }
};

using Uses = absl::InlinedVector<Use, 2>;
using UseLists = absl::flat_hash_map<Value*, Uses>;

// Analyzes use-def chains. Computes the use list for all instructions in
// a function.
UseLists ComputeUses(Function& f);

// Replaces all uses of value `from` with `to`.
void ReplaceAllUsesWith(UseLists& uses, Value* from, Value* to);

// Returns whether a Use is a safepoint use that can safely contain a
// rematerialize.
inline bool IsSafepointUse(Use use) {
  SafepointInst* safepoint = dyn_cast<SafepointInst>(use.user);
  if (!safepoint) return false;
  return safepoint->IsInStackOrFastlocals(use.operand_index);
}

// Returns whether a use is only related to refcounting.
// That means is it in a refcount instruction or in the increfs/decrefs parts
// of a safepoint.
inline bool IsRefcountUse(Use use) {
  if (isa<RefcountInst>(use.user)) return true;
  SafepointInst* safepoint = dyn_cast<SafepointInst>(use.user);
  if (!safepoint) return false;
  return safepoint->IsInIncrefsOrDecrefs(use.operand_index);
}

// Checks if a use is a comparison with 0.
inline bool IsCompareWithZeroUse(Use use) {
  CompareInst* comp = dyn_cast<CompareInst>(use.user);
  if (!comp) return false;
  int64_t other_index = use.operand_index == 0 ? 1 : 0;
  Value* other_value = comp->operands()[other_index];
  ConstantInst* constant = dyn_cast<ConstantInst>(other_value);
  if (!constant) return false;
  return constant->value() == 0;
}

////////////////////////////////////////////////////////////////////////////////
// Constant conversion functions

// Returns if a Value is constant instruction and thus the output is expected to
// be constant.
bool IsConstantInstruction(const Value* v);

// If `v` is an instruction with a constant value, returns this value, otherwise
// returns absl::nullopt
absl::optional<int64_t> GetValueAsConstantInt(Value* v, PyCodeObject* code);

// If `v` is an instruction with a constant value, returns this value, otherwise
// returns nullptr
inline PyObject* GetValueAsConstantObject(Value* v, PyCodeObject* code) {
  return reinterpret_cast<PyObject*>(
      GetValueAsConstantInt(v, code).value_or(0));
}

// If the PyObject is an *interned* PyUnicode object, returns a reference to it.
// It must be interned so that the string_view stays valid.
absl::optional<absl::string_view> GetPyObjectAsString(PyObject* obj);

// Returns a string from a Value that returns a constant valid PyObject to an
// interned string.
inline absl::optional<absl::string_view> GetValueAsConstantString(
    Value* v, PyCodeObject* code) {
  return GetPyObjectAsString(GetValueAsConstantObject(v, code));
}

// Returns an array of strings from a Python tuple of string. All Python strings
// must be interned to avoid lifetime issues.
absl::optional<std::vector<absl::string_view>> GetPyObjectAsTupleOfStrings(
    PyObject* tuple);

// Returns an array of strings from a Value that returns a constant valid
// PyObject to a tuple of interned strings.
inline absl::optional<std::vector<absl::string_view>>
GetValueAsConstantTupleOfStrings(Value* v, PyCodeObject* code) {
  return GetPyObjectAsTupleOfStrings(GetValueAsConstantObject(v, code));
}

////////////////////////////////////////////////////////////////////////////////
// Rewriter

// A Rewriter is used to mutate Strongjit IR while maintaining use lists.
// It is used principally when using the pattern driver, but also elsewhere.
// Preferably use rewriter methods instead of mutating operands
// directly when possible, it should be faster.
class Rewriter : public InstructionModificationListener {
 public:
  Rewriter(Function& f, PyCodeObject* code_object,
           const OptimizerOptions& options);
  ~Rewriter() override;

  // Returns the uses of `v`.
  const Uses& GetUsesOf(Value& v);

  // Replaces all uses of `from` with `to`.
  void ReplaceAllUsesWith(Value& from, Value& to);

  // Replaces all `IsSafepointUse` uses of `from` with `to`.
  void ReplaceAllSafepointUsesWith(Value& from, Value& to);

  // Erases all use of value v in refcounting operations.
  // This include incref, decrefs and the increfs and decrefs fields of
  // safepoints as defined by `IsRefcountUse`.
  void EraseAllRefcountUses(Value& v);

  // Erases `inst`.
  void erase(Instruction& inst);

  // Returns a Builder that notifies the rewriter on inserted instructions.
  Builder CreateBuilder(Block::iterator iterator) {
    return Builder(iterator->parent(), iterator);
  }

  Builder CreateBuilderBefore(Instruction& inst) {
    return CreateBuilder(inst.GetIterator());
  }
  Builder CreateBuilderAfter(Instruction& inst) {
    auto it = ++inst.GetIterator();
    S6_CHECK(it != inst.parent()->end());
    return CreateBuilder(it);
  }

  // Interns a string.
  StringTable::key_type InternString(absl::string_view str) {
    return function_.GetStringTable()->InternString(str);
  }

  void AddReliedUponClass(const Class* cls) {
    function_.AddReliedUponClass(cls);
  }

  // Removes all uses of `use` within the operand list of `user`.
  void RemoveUsesOfIn(Instruction& user, Value& use);

  // Replace one operand at the specified index.
  void ReplaceAt(Instruction& user, int64_t index, Value& to);

  // Replaces all uses of `from` within `user`'s operand list with `to`.
  void ReplaceUsesWith(Instruction& user, Value& from, Value& to);

  // Replaces all uses of `from` within `block` with `to`.
  void ReplaceUsesWith(Block& block, Value& from, Value& to);

  // Converts a BrInst to an equivalent JmpInst to the given successor,
  // discarding control flow to the other successor.
  void ConvertBranchToJump(BrInst& br, bool true_successor);

  const Function& function() const { return function_; }
  Function& function() { return function_; }
  PyCodeObject* code_object() const { return code_object_; }
  const OptimizerOptions& options() const { return options_; }
  ClassManager& class_manager() const { return options_.mgr; }

  // Sets a Cursor to keep valid with respect to any erase operations.
  void SetCursor(Cursor* cursor) { cursor_ = cursor; }

  // InstructionModificationListener interface.
  void InstructionAdded(Instruction* inst) final;
  void InstructionErased(Instruction* inst) final;
  void OperandsMayBeModified(Instruction* inst) final;

 private:
  void EraseOperands(const Instruction& inst);
  void AddOperands(Instruction& inst);

  Function& function_;
  PyCodeObject* code_object_;
  const OptimizerOptions& options_;
  UseLists use_lists_;
  Cursor* cursor_;

  // Ensures that use_lists_ if valid. Sets use_lists_invalid_ to false.
  void EnsureUsesValid();

  // List of instruction who do not appear in use_lists_ because their operands
  // were modified. Any triggering of GetUses() or similar will repopulate use
  // lists with those instruction operands.
  absl::flat_hash_set<Instruction*> to_update_;
};

////////////////////////////////////////////////////////////////////////////////
// Pattern

// A pattern to rewrite. All patterns have an "anchor" kind. `Apply` will always
// be called with a Value of the anchor kind.
class Pattern {
 public:
  virtual ~Pattern();

  // A name or identifier for this pattern in log messages.
  virtual absl::string_view name() const = 0;

  // The anchor kind. This is a filter that allows the pattern rewriter to only
  // attempt to apply this Pattern on a subset of Values.
  virtual Value::Kind anchor() const = 0;

  // Applies a pattern rewrite. If the rewrite was successful, absl::OkStatus()
  // is returned.
  //
  // Apply may insert any Values, but may not delete any Values. Use
  // rewriter.erase instead.
  virtual absl::Status Apply(Value* value, Rewriter& rewriter) const = 0;
};

// A Pattern with predicate mixins. Client classes should derive from this
// instead of Pattern:
//
//  class MyPat : public PatternT<MyPat, CallNativeInst> {
//    static absl::Status Apply(CallNativeInst* inst, Rewriter& rewriter);
//  };
//
// The anchor() method is filled in automatically, and the instruction is
// casted to the appropriate type.
//
// A PatternT may also contain a list of "mixins" that add additional predicates
// to the match and may also extend the argument list to Apply(). For example,
// the NeedsSafepoint mixin will ensure a SafepointInst is available and will
// plumb it through:
//
// class MyPat : public PatternT<MyPat, CallNativeInst, NeedsSafepoint> {
//   static absl::Status Apply(CallNativeInst* inst, SafepointInst* safepoint,
//                             Rewriter& rewriter);
// };
template <typename Derived, typename Anchor, typename... Mixins>
class PatternT : public Pattern {
 public:
  Value::Kind anchor() const override { return Anchor::kKind; }

  absl::string_view name() const override { return typeid(Derived).name(); }

  absl::Status Apply(Value* value, Rewriter& rewriter) const override {
    return ApplyMixin<Mixins..., Derived>::Apply(static_cast<Anchor*>(value),
                                                 rewriter);
  }

 private:
  // The template arguments <A, B, C> give a sequence of mixins, or transforms,
  // to perform. Applies these in order, morally equivalent to:
  //
  //   ApplyMixin<A, B, C>:
  //     A(B(C(args...)));
  //
  // Although mixins may modify `args`, so this is actually implemented with
  // tail recursion.
  //
  // The first N-1 arguments are expected to be Mixins; that is, they take
  // an additional template parameter that they tail-recurse to:
  //
  //  struct Mixin {
  //    template<typename Fn, typename... Args>
  //    static absl::Status Apply(Args... args) {
  //      ...
  //      return Fn(args...);
  //    }
  template <typename Mixin, typename... Rest>
  struct ApplyMixin {
    template <typename Inst, typename... Args>
    static absl::Status Apply(Inst* inst, Rewriter& rewriter, Args... args) {
      if constexpr (sizeof...(Rest) == 0) {
        return Mixin::Apply(inst, rewriter, args...);
      } else {
        return Mixin::template Apply<ApplyMixin<Rest...>>(inst, rewriter,
                                                          args...);
      }
    }
  };
};

// Mixin for PatternT that ensures a SafepointInst is available. The found
// SafepointInst is injected into the Apply arguments list.
class NeedsSafepoint {
 public:
  template <typename Fn, typename Inst, typename... Args>
  static absl::Status Apply(Inst* inst, Rewriter& rewriter, Args... args) {
    S6_ASSIGN_OR_RETURN(BytecodeBeginInst * safepoint,
                        BeginningOfBytecodeInstruction(inst));
    return Fn::Apply(inst, rewriter, args..., safepoint);
  }
};

// Mixin for PatternT that ensures monomorphic type feedback is available. The
// found Class is injected into the Apply arguments list.
//
// If AllowsTypeClass is true, the found class may have is_type_class() == true.
template <bool AllowsTypeClass = false>
class NeedsMonomorphicClass {
 public:
  template <typename Fn, typename Inst, typename... Args>
  static absl::Status Apply(Inst* inst, Rewriter& rewriter, Args... args) {
    absl::optional<ClassDistributionSummary> summary =
        rewriter.function().GetTypeFeedbackForBytecodeOffset(
            inst->bytecode_offset());
    if (!summary.has_value()) {
      return absl::FailedPreconditionError("No type feedback");
    }
    Class* cls = summary->MonomorphicClass(rewriter.options().mgr);
    if (!cls) {
      return absl::FailedPreconditionError("Not monomorphic class");
    }
    if (!AllowsTypeClass && cls->is_type_class()) {
      return absl::FailedPreconditionError("Was a type class");
    }
    return Fn::Apply(inst, rewriter, args..., cls);
  }
};

// Mixin for PatternT that type feedback is available. The
// ClassDistributionSummary is injected into the pattern's argument list.
class NeedsTypeFeedback {
 public:
  template <typename Fn, typename Inst, typename... Args>
  static absl::Status Apply(Inst* inst, Rewriter& rewriter, Args... args) {
    absl::optional<ClassDistributionSummary> summary =
        rewriter.function().GetTypeFeedbackForBytecodeOffset(
            inst->bytecode_offset());
    if (!summary.has_value()) {
      return absl::FailedPreconditionError("No type feedback");
    }
    return Fn::Apply(inst, rewriter, args..., summary.value());
  }
};

// Mixin for PatternT that ensures that the `callee()` method of Inst returns
// a Value of type `CalleeType`, and injects it into the Apply arguments list.
template <typename CalleeType>
class NeedsCallee {
 public:
  template <typename Fn, typename Inst, typename... Args>
  static absl::Status Apply(Inst* inst, Rewriter& rewriter, Args... args) {
    CalleeType* callee = dyn_cast<CalleeType>(inst->callee());
    if (!callee) {
      return absl::FailedPreconditionError(
          absl::StrCat("Not a call of ", typeid(CalleeType).name()));
    }
    return Fn::Apply(inst, rewriter, args..., callee);
  }
};

// Mixin for PatternT that ensures that the `callee` of a CallNativeInst is a
// given Callee.
template <Callee CalleeKind>
class NeedsNativeCallee {
 public:
  template <typename Fn, typename... Args>
  static absl::Status Apply(CallNativeInst* inst, Rewriter& rewriter,
                            Args... args) {
    if (!inst->CalleeIs(CalleeKind)) {
      return absl::FailedPreconditionError("Not a call of correct callee");
    }
    return Fn::Apply(inst, rewriter, args...);
  }
};

// Given a list of Patterns, greedily applies them to `f` until fixpoint.
absl::Status RewritePatterns(Function& f, PyCodeObject* code,
                             absl::Span<const Pattern* const> patterns,
                             const OptimizerOptions& options = {});

// Given a list of pattern classes, greedily applies them all to `f` until
// fixpoint.
//
// The pattern class list may contain Patterns or tuples of patterns; the tuples
// will be expanded.
//   using MyCommonPatternList = std::tuple<Pass1, Pass2>;
//   RewritePatterns<Pass3, MyCommonPassList>(f, code);
template <typename... Patterns>
absl::Status RewritePatterns(Function& f, PyCodeObject* code,
                             const OptimizerOptions& options = {}) {
  using PatternTuple = decltype(tuple::flatten<Patterns...>(Patterns{}...));
  PatternTuple patterns;
  auto pattern_ptrs = tuple::to_array(
      tuple::transform([](auto& arg) -> Pattern* { return &arg; }, patterns));
  return RewritePatterns(f, code, pattern_ptrs, options);
}

////////////////////////////////////////////////////////////////////////////////
// Standalone utilities

// Scans back to the preceding bytecode_begin op, to be used as
// a safepoint for re-attempting the bytecode instruction.
absl::StatusOr<BytecodeBeginInst*> BeginningOfBytecodeInstruction(
    Instruction* pynum_op);

// Returns a list of values by numbers by reversing a ValueNumbering.
std::vector<const Value*> ReverseValueNumbering(
    const ValueNumbering& numbering);

// Delays a decref instruction to be in front of a set of targets.
//
// Any control-flow path from the decref instruction that does not go through
// any target will also take care of decreffing so that no leak is possible.
//
// It will return an error status in case of failure in which case the function
// will not have been mutated at all. The possible reasons for failures are:
// - A target is not dominated by the decref.
// - A target is before another target in the path from the latter to the
//   decref.
absl::Status DelayDecref(DecrefInst& decref,
                         absl::Span<Instruction* const> targets);

// Splits critical edges in `f`. When this returns, `f` will no longer contain
// critical edges.
//
// A critical edge is an edge which is neither the only edge leaving its source
// block, nor the only edge entering its destination block. Splitting a critical
// edge involves inserting a trivial block on the edge such that the block
// has only one predecessor and one successor.
//
// One way to think about critical edges is that they are edges that we cannot
// insert copies on without inadvertently affecting other edges. In Josef Eisl's
// words (PhD thesis, Definition 15 (Critical Edge):
//
//  Critical edges can be removed by introducing new (empty) blocks. For
//  data-flow resolution, critical edge splitting is of vast importance. The
//  newly introduced blocks provide a place for compensation code, i.e., moves
//  from one location to another.
void SplitCriticalEdges(Function& f);

////////////////////////////////////////////////////////////////////////////////
// Worklist

// A worklist for dataflow algorithms. This is a queue but it remembers what
// is already in it and refuses to add the same element twice.
//
// It also supports `PushIfNew` which is really convenient.
//
// The fact that it is a queue may change later for performance reasons, so
// try not to rely too much on it.
template <typename T>
class Worklist {
 public:
  using value_type = T;

  // Pushes a new element in the worklist. The element must not already in the
  // worklist.
  void Push(T t) {
    S6_CHECK(!contains(t));
    inqueue_.insert(t);
    queue_.push_back(std::move(t));
  }

  // Pops a new element from the worklist which must not be empty.
  T Pop() {
    S6_CHECK(!queue_.empty());
    T t = std::move(queue_.front());
    inqueue_.erase(t);
    queue_.pop_front();
    return t;
  }

  // Pushes a new element in the worklist if it is not already there.
  void PushIfNew(T t) {
    if (!contains(t)) Push(std::move(t));
  }

  // For back_inserter function, Same as Push.
  void push_back(T t) { Push(t); }

  // Checks if an element is in the worklist.
  bool contains(const T& t) { return inqueue_.contains(t); }

  // Checks if the worklist is empty.
  bool empty() { return queue_.empty(); }

 private:
  std::deque<T> queue_;
  absl::flat_hash_set<T> inqueue_;
};

////////////////////////////////////////////////////////////////////////////////
// BuiltinObjects

// Looks up and caches PyObjects from built-in modules.
//
// This is used to identify callees with known semantics whose C symbol has
// hidden linkage.
class BuiltinObjects {
 public:
  // Returns the singleton instance.
  static BuiltinObjects& Instance();

  // Initializes the instance, populating from PyInterpreterState::builtins.
  // REQUIRES: The GIL must be held.
  void Initialize();

  // Returns `name` from the global builtins dict. May return nullptr.
  // `name` should be in the form `module.member`, for example `math.sin`.
  PyObject* LookupBuiltin(absl::string_view name);

 private:
  absl::flat_hash_map<std::string, PyObject*> builtin_objects_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_OPTIMIZER_UTIL_H_
