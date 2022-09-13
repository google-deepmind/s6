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

#include "code_generation/code_generator.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "allocator.h"
#include "asmjit/asmjit.h"
#include "asmjit/core/builder.h"
#include "asmjit/core/codeholder.h"
#include "asmjit/core/func.h"
#include "asmjit/core/globals.h"
#include "asmjit/core/operand.h"
#include "asmjit/x86/x86emitter.h"
#include "asmjit/x86/x86globals.h"
#include "asmjit/x86/x86operand.h"
#include "classes/attribute.h"
#include "classes/class_manager.h"
#include "code_generation/call_stub.h"
#include "code_generation/code_generation_context.h"
#include "code_generation/jit_stub.h"
#include "code_generation/prolog_epilog_insertion.h"
#include "code_generation/register_allocator.h"
#include "code_object.h"
#include "event_counters.h"
#include "metadata.h"
#include "runtime/callee_address.h"
#include "runtime/deoptimization_runtime.h"
#include "runtime/generator.h"
#include "runtime/pyframe_object_cache.h"
#include "runtime/runtime.h"
#include "runtime/stack_frame.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer.h"
#include "utils/logging.h"
#include "utils/path.h"
#include "utils/status_macros.h"

ABSL_FLAG(std::string, s6_dump_dir, "",
          "Dumps strongjit-compiled functions to the given directory.");

namespace deepmind::s6 {
namespace x86 = ::asmjit::x86;

namespace {

// If enabled on the command line (via -s6_dump_dir), emit the IR and assembly
// to files.
absl::Status DumpIfEnabled(const CodeObject& code_object,
                           absl::string_view symbol_name, PyCodeObject* code,
                           const RegisterAllocation& ra) {
  std::string dirname = absl::GetFlag(FLAGS_s6_dump_dir);
  if (dirname.empty()) return absl::OkStatus();
  if (::mkdir(dirname.c_str(), 0777) != 0) {
    // EEXIST is benign. Anything else we abort.
    S6_RET_CHECK(errno == EEXIST)
        << "error creating dump directory: " << strerror(errno);
  }
  auto write_file = [&](absl::string_view extension, std::string content) {
    // We don't use //base:file; we can't guarantee InitGoogle has been called
    // or completed.
    std::ofstream stream(
        file::JoinPath(dirname, absl::StrCat(symbol_name, extension)));
    if (!stream.is_open()) {
      return absl::InternalError("Unable to open dump output file");
    }
    stream << content;
    stream.close();
    return absl::OkStatus();
  };

  S6_RETURN_IF_ERROR(
      write_file(".strongjit",
                 FormatOrDie(code_object.function(),
                             ChainAnnotators(PredecessorAnnotator(),
                                             SourceLocationAnnotator(code)))));
  S6_RETURN_IF_ERROR(
      write_file(".register_allocation", ra.ToString(code_object.function())));
  return write_file(".assembly", code_object.Disassemble());
}

// Helper for MOV and friends, below.
#define EMIT_INST(id, ...) RETURN_IF_ASMJIT_ERROR(emitter.emit(id, __VA_ARGS__))

// Shorthand for emitting a mov instruction and returning on error. We use this
// when we cannot use Emitter::mov(); this happens when one of our operands may
// be in memory, immediate or register. The Emitter::mov() function overloads
// only concrete types; for type-agnostic calls we must use Emitter::emit().
#define MOV(dst, src) EMIT_INST(x86::Inst::kIdMov, dst, src)
#define CMP(s1, s2) EMIT_INST(x86::Inst::kIdCmp, s1, s2)
#define XOR(s1, s2) EMIT_INST(x86::Inst::kIdXor, s1, s2)
#define PUSH(s) EMIT_INST(x86::Inst::kIdPush, s)
#define POP(s) EMIT_INST(x86::Inst::kIdPop, s)

absl::Status GenerateCopy(const RegisterAllocation::Copy& copy,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  asmjit::Operand dst = ctx.Operand(copy.second);
  asmjit::Operand src = ctx.Operand(copy.first);
  if (dst.isMem() && src.isMem()) {
    x86::Gp scratch = ctx.scratch_reg();
    emitter.mov(scratch, src.as<x86::Mem>());
    emitter.mov(dst.as<x86::Mem>(), scratch);
    return absl::OkStatus();
  }
  MOV(dst, src);
  return absl::OkStatus();
}

absl::Status GenerateCopies(const Value& value, x86::Emitter& emitter,
                            CodeGenerationContext& ctx) {
  const RegisterAllocation& ra = ctx.register_allocation();
  for (const RegisterAllocation::Copy& copy : ra.inst_copies(&value)) {
    S6_RETURN_IF_ERROR(GenerateCopy(copy, emitter, ctx));
  }
  return absl::OkStatus();
}

}  // namespace

void GenerateSetOverflowFlag(x86::Gp scratch, x86::Emitter& emitter) {
  emitter.mov(scratch, asmjit::imm(-1));
  // Single-shift SHR sets overflow flag if result's top two bits differ.
  emitter.shr(scratch, 1);  // OF <- 1
}

namespace impl {

// Declare private version of an existing, private Python structure so we can
// access it from generated code.
//
// _PyCodeObjectExtra is different for Python 3.6 and 3.7.
//
#if PY_MINOR_VERSION <= 6
typedef struct {
  Py_ssize_t ce_size;
  void** ce_extras;
} _PyCodeObjectExtra;
#else   // PY_MINOR_VERSION >= 7
typedef struct {
  Py_ssize_t ce_size;
  void* ce_extras[1];
} _PyCodeObjectExtra;
#endif  // PY_MINOR_VERSION-specific code.

}  // namespace impl

namespace {
// `tuple` is a PyTupleObject of PyUnicodeObjects. Scan the tuple, looking for
// `str`.
PyObject* FindStringInTuple(PyObject* tuple, absl::string_view str) {
  for (int64_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
    PyObject* item = PyTuple_GET_ITEM(tuple, i);
    absl::string_view this_string(
        reinterpret_cast<const char*>(PyUnicode_DATA(item)),
        PyUnicode_GET_LENGTH(item));
    if (this_string == str) return item;
  }
  S6_LOG(FATAL) << "Could not find string in tuple!";
}

////////////////////////////////////////////////////////////////////////////////
// Instruction code generation

absl::Status GenerateCode(const ConstantInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  const RegisterAllocation& ra = ctx.register_allocation();
  const Location& loc = ra.DestinationLocation(instr);
  if (loc.IsImmediate()) return absl::OkStatus();

  S6_ASSIGN_OR_RETURN(auto dst, ctx.Destination(instr));
  emitter.mov(dst, asmjit::Imm(instr.value()));
  return absl::OkStatus();
}

absl::Status GenerateCode(const ReturnInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  MOV(x86::rax, ctx.Operand(instr.returned_value()));
  emitter.jmp(ctx.cleanup_point());
  return absl::OkStatus();
}

absl::Status GenerateCode(const YieldValueInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  auto resume = emitter.newLabel();

  // We're going to pause this generator. We return, and when we resume we'd
  // like to pick up where we left off.

  // We call SetUpForYieldValue(), which acts like setjmp; it looks like it
  // returns twice. We use the second element of the returned pair (in rdx)
  // to discriminate between yielding and resuming.
  CallStub<SetUpForYieldValue> stub;

  MOV(stub.result(), ctx.Operand(instr.yielded_value()));
  ctx.stack_frame().Load(stub.stack_frame(), emitter);
  emitter.mov(stub.yi(), asmjit::imm(&instr));
  emitter.call(stub.imm());

  // Now we could either be yielding or resuming. If rdx is zero, we're
  // yielding.
  emitter.test(x86::rdx, x86::rdx);
  emitter.jnz(resume);

  // We're yielding. rax contains the yielded value.
  emitter.jmp(ctx.epilog_point());

  emitter.bind(resume);
  // We're resuming. rax contains the result of the YieldValueInst.

  // TODO: Bail out when recursion depth is too great.
  JitStub<PyThreadState> thread_state =
      ctx.stack_frame().thread_state().Load(x86::rdx, emitter);
  emitter.inc(thread_state.recursion_depth().Mem());

  // The result of the YieldValueInst is already in rax.
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const FrameVariableInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  switch (instr.frame_variable_kind()) {
    case FrameVariableInst::FrameVariableKind::kConsts:
      // Note: We treat co_consts as immutable.
      // Issuing a single instruction here is invalid if the code object has
      // been modified.
      emitter.mov(dst, asmjit::imm(PyTuple_GET_ITEM(
                           ctx.py_code_object()->co_consts, instr.index())));
      break;
    case FrameVariableInst::FrameVariableKind::kFrame:
      emitter.mov(dst, ctx.pyframe_reg());
      break;
    case FrameVariableInst::FrameVariableKind::kBuiltins:
      ctx.pyframe().f_builtins().Load(dst, emitter);
      break;
    case FrameVariableInst::FrameVariableKind::kGlobals:
      ctx.pyframe().f_globals().Load(dst, emitter);
      break;
    case FrameVariableInst::FrameVariableKind::kFastLocals:
      // Note that the fastlocals array is stored inside the frameobject, so
      // we don't dereference the pointer.
      emitter.lea(dst, ctx.pyframe().fastlocals(instr.index()).Mem());
      break;
    case FrameVariableInst::FrameVariableKind::kFreeVars: {
      int64_t index = instr.index() + ctx.py_code_object()->co_nlocals;
      emitter.lea(dst, ctx.pyframe().fastlocals(index).Mem());
      break;
    }
    case FrameVariableInst::FrameVariableKind::kNames:
      // Note: We treat co_names as immutable.
      // Issuing a single instruction here is invalid if the code object has
      // been modified.
      emitter.mov(dst, asmjit::imm(PyTuple_GET_ITEM(
                           ctx.py_code_object()->co_names, instr.index())));
      break;
    case FrameVariableInst::FrameVariableKind::kLocals:
      ctx.pyframe().f_locals().Load(dst, emitter);
      break;
    case FrameVariableInst::FrameVariableKind::kCodeObject:
      ctx.pyframe().f_code().Load(dst, emitter);
      break;
    case FrameVariableInst::FrameVariableKind::kThreadState:
      ctx.stack_frame().thread_state().Load(dst, emitter);
      break;
  }
  return absl::OkStatus();
}

absl::StatusOr<x86::Mem> GenerateMemOp(const MemoryInst& instr,
                                       x86::Emitter& emitter,
                                       CodeGenerationContext& ctx) {
  x86::Gp pointer = ctx.OperandInRegister(instr.pointer());
  if (!instr.index()) {
    return x86::ptr(pointer, instr.offset());
  }
  asmjit::Operand index = ctx.Operand(instr.index());
  int64_t shift = MemoryInst::ShiftToInt(instr.shift());
  if (index.isMem()) {
    return absl::UnimplementedError("Memory index in memory unsupported");
  }
  if (index.isImm()) {
    int64_t offset =
        instr.offset() + (index.as<asmjit::Imm>().i64() * (1ll << shift));
    return x86::ptr(pointer, offset);
  } else {
    S6_CHECK(index.isReg());
    return x86::ptr(pointer, index.as<x86::Gp>(), shift, instr.offset());
  }
}

absl::Status GenerateCode(const LoadInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  if (instr.extension() != LoadInst::kNoExtension) {
    return absl::UnimplementedError("Extending loads not implemented");
  }

  S6_ASSIGN_OR_RETURN(auto dst, ctx.Destination(instr));
  S6_ASSIGN_OR_RETURN(x86::Mem mem, GenerateMemOp(instr, emitter, ctx));
  emitter.mov(dst, mem);
  return absl::OkStatus();
}

absl::Status GenerateCode(const StoreInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  if (instr.truncation() != StoreInst::kNoTruncation) {
    return absl::UnimplementedError("Truncating stores not implemented");
  }

  S6_ASSIGN_OR_RETURN(x86::Mem mem, GenerateMemOp(instr, emitter, ctx));
  asmjit::Operand src = ctx.Operand(instr.stored_value());
  if (src.isMem()) {
    if (mem.baseReg() == ctx.scratch_reg()) {
      return absl::UnimplementedError(
          "Stores requiring two scratch registers not implemented");
    }
    emitter.mov(ctx.scratch_reg(), src.as<x86::Mem>());
    src = ctx.scratch_reg();
  }
  MOV(mem, src);
  return absl::OkStatus();
}

absl::Status GenerateCode(const IncrefInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  x86::Gp gp = ctx.OperandInRegister(instr.operand());
  if (instr.nullness() == Nullness::kNotNull) {
    emitter.inc(x86::qword_ptr(gp, offsetof(PyObject, ob_refcnt)));
    return absl::OkStatus();
  }
  asmjit::Label l = emitter.newLabel();
  emitter.test(gp, gp);
  emitter.jz(l);
  emitter.inc(x86::qword_ptr(gp, offsetof(PyObject, ob_refcnt)));
  emitter.bind(l);
  return absl::OkStatus();
}

absl::Status GenerateCode(const DecrefInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  // DecrefInsts are treated by the register allocator like calls, so the
  // operand is already in call registers.

  x86::Gp gp = AbiLocation<0>();
  S6_RET_CHECK_EQ(ctx.Operand(instr.operand()), gp)
      << "decref operand must be in the first call register";
  asmjit::Label end = emitter.newLabel();
  if (instr.nullness() == Nullness::kMaybeNull) {
    emitter.test(gp, gp);
    emitter.jz(end);
  }
  emitter.dec(x86::qword_ptr(gp, offsetof(PyObject, ob_refcnt)));
  emitter.jnz(end);
  ctx.pyframe().f_lasti().Store(asmjit::imm(instr.bytecode_offset()), emitter);
  emitter.call(asmjit::imm(Dealloc));
  emitter.bind(end);
  return absl::OkStatus();
}

absl::Status GenerateCode(const JmpInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Label target = ctx.block_label(instr.unique_successor());
  const RegisterAllocation& ra = ctx.register_allocation();

  for (const RegisterAllocation::Copy& copy :
       ra.block_copies(instr.parent(), /*successor_index=*/0)) {
    S6_RETURN_IF_ERROR(GenerateCopy(copy, emitter, ctx));
  }

  emitter.jmp(target);
  return absl::OkStatus();
}

namespace {
x86::Condition::Code ComparisonToAsmjitCondCode(
    CompareInst::Comparison comparison) {
  switch (comparison) {
    case CompareInst::kEqual:
      return x86::Condition::kZ;
    case CompareInst::kNotEqual:
      return x86::Condition::kNZ;
    case CompareInst::kGreaterEqual:
      return x86::Condition::kSignedGE;
    case CompareInst::kGreaterThan:
      return x86::Condition::kSignedGT;
    case CompareInst::kLessEqual:
      return x86::Condition::kSignedLE;
    case CompareInst::kLessThan:
      return x86::Condition::kSignedLT;
  }
  S6_UNREACHABLE();
}

x86::Predicate::Cmp ComparisonToAsmjitPredicateCmp(
    CompareInst::Comparison comparison) {
  switch (comparison) {
    case CompareInst::kEqual:
      return x86::Predicate::kCmpEQ;
    case CompareInst::kNotEqual:
      return x86::Predicate::kCmpNEQ;
    case CompareInst::kGreaterEqual:
      return x86::Predicate::kCmpNLT;
    case CompareInst::kGreaterThan:
      return x86::Predicate::kCmpNLE;
    case CompareInst::kLessEqual:
      return x86::Predicate::kCmpLE;
    case CompareInst::kLessThan:
      return x86::Predicate::kCmpLT;
  }
  S6_UNREACHABLE();
}

// Generates code for a CompareInst, OverflowedInst or FloatZeroInst that is
// optimizable; otherwise emits code to compare `instr` with zero.
absl::StatusOr<x86::Condition::Code> GenerateCodeForCondition(
    const Value* instr, CodeGenerationContext& ctx) {
  x86::Emitter& emitter = ctx.emitter();

  if (const auto* compare = dyn_cast<CompareInst>(instr);
      compare && ctx.ConditionIsOptimizable(compare)) {
    S6_CHECK_EQ(compare->type(), NumericInst::kInt64);
    // The condition is an optimizable CompareInst. We generate the
    // CompareInst's comparison inline.
    //
    // Obtain the condition's binary operands but modify `ctx` to query the
    // *compare's* slot, not ours. Else we'll end up with an invalid register
    // location.
    x86::Gp s1 = ctx.OperandInRegister(compare->lhs(), compare);
    asmjit::Operand s2 = ctx.Operand(compare->rhs(), compare);
    CMP(s1, s2);

    return ComparisonToAsmjitCondCode(compare->comparison());
  } else if (const auto* overflowed = dyn_cast<OverflowedInst>(instr);
             overflowed && ctx.ConditionIsOptimizable(overflowed)) {
    // The condition is an optimizable OverflowedInst.
    return x86::Condition::kO;
  } else if (const auto* float_zero = dyn_cast<FloatZeroInst>(instr);
             float_zero && ctx.ConditionIsOptimizable(float_zero)) {
    // The condition is an optimizable FloatZeroInst.
    asmjit::Operand arg = ctx.Operand(float_zero->float_value(), float_zero);
    // Set 'dst' to 0 or 1 based on whether the operand is fp zero.
    // Accept +0 and -0 alike; account for this by shifting out the sign bit.
    x86::Gp scratch = ctx.scratch_reg();
    MOV(scratch, arg);
    emitter.shl(scratch, asmjit::imm(1));
    return x86::Condition::kZ;
  } else {
    // Otherwise we emit code to compare the condition with 0.
    asmjit::Operand cond = ctx.Operand(instr);

    // In degenerate cases we may have an immediate here. Because we can't emit
    // a two-immediate instruction we move the immediate into the scratch
    // register.
    if (cond.isImm()) {
      EventCounters::Instance().Add(
          "code_generation.Comparing an immediate to 0 should not happen", 1);
      emitter.mov(ctx.scratch_reg(), cond.as<asmjit::Imm>());
      cond = ctx.scratch_reg();
    }

    CMP(cond, asmjit::imm(0));
    return x86::Condition::kNZ;
  }
}

absl::Status GenerateCodeForBranch(const ConditionalTerminatorInst& instr,
                                   x86::Emitter& emitter,
                                   CodeGenerationContext& ctx,
                                   asmjit::Label true_target,
                                   asmjit::Label false_target) {
  const RegisterAllocation& ra = ctx.register_allocation();
  // The condition code to jump to the true target.
  S6_ASSIGN_OR_RETURN(x86::Condition::Code cc,
                      GenerateCodeForCondition(instr.condition(), ctx));

  // If this is a DeoptimizeIfInst, honour the "negated()" flag.
  if (auto* di = dyn_cast<DeoptimizeIfInst>(&instr); di && di->negated()) {
    cc = static_cast<x86::Condition::Code>(x86::Condition::negate(cc));
  }

  auto taken_copies = ra.block_copies(instr.parent(), /*successor_index=*/0);
  auto fallthrough_copies =
      ra.block_copies(instr.parent(), /*successor_index=*/1);
  auto taken_target = true_target;
  auto fallthrough_target = false_target;

  // Optimize the polarity of the branch. If:
  //    a) we have copies on the taken branch but not the fallthrough
  // or b) the taken branch is actually the next block in sequence order
  // Now, if our current taken target is actually the next instruction, swap
  // everything around so it becomes the fallthrough target.
  if ((!taken_copies.empty() && fallthrough_copies.empty()) ||
      instr.true_successor() == &*std::next(instr.parent()->GetIterator())) {
    cc = static_cast<x86::Condition::Code>(x86::Condition::negate(cc));
    std::swap(taken_copies, fallthrough_copies);
    std::swap(taken_target, fallthrough_target);
  }

  // We'd ideally plant a "jnz/jmp" sequence. This allows us to insert any
  // copies for the fallthrough branch before jmp, but doesn't allow us to
  // insert any copies for the taken branch. If we have taken branch copies,
  // jump to a temporary label and emit the copies there.
  asmjit::Label taken_copies_label;
  if (taken_copies.empty()) {
    emitter.j(cc, taken_target);
  } else {
    taken_copies_label = emitter.newLabel();
    emitter.j(cc, taken_copies_label);
  }

  for (const RegisterAllocation::Copy& copy : fallthrough_copies) {
    S6_RETURN_IF_ERROR(GenerateCopy(copy, emitter, ctx));
  }
  emitter.jmp(fallthrough_target);

  if (taken_copies_label.isValid()) {
    emitter.bind(taken_copies_label);
    for (const RegisterAllocation::Copy& copy : taken_copies) {
      S6_RETURN_IF_ERROR(GenerateCopy(copy, emitter, ctx));
    }
    emitter.jmp(taken_target);
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status GenerateCode(const BrInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Label true_target = ctx.block_label(instr.true_successor());
  asmjit::Label false_target = ctx.block_label(instr.false_successor());

  return GenerateCodeForBranch(instr, emitter, ctx, true_target, false_target);
}

absl::Status GenerateCode(const DeoptimizeIfInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Label deopt_target = emitter.newLabel();
  asmjit::Label false_target = ctx.block_label(instr.false_successor());

  // Record the deoptimization label we're going to jump to. We'll add the
  // deoptimization trampoline code there later.
  ctx.deopt_labels().emplace_back(&instr, deopt_target);

  return GenerateCodeForBranch(instr, emitter, ctx, deopt_target, false_target);
}

absl::Status GenerateCode(const DeoptimizeIfSafepointInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  asmjit::Label deopt_target = emitter.newLabel();

  // Record the deoptimization label we're going to jump to. We'll add the
  // deoptimization trampoline code there later.
  ctx.deopt_labels().emplace_back(&instr, deopt_target);

  S6_ASSIGN_OR_RETURN(x86::Condition::Code cc,
                      GenerateCodeForCondition(instr.condition(), ctx));
  if (instr.negated()) {
    cc = static_cast<x86::Condition::Code>(x86::Condition::negate(cc));
  }
  emitter.j(cc, deopt_target);
  return absl::OkStatus();
}

absl::Status GenerateCode(const UnreachableInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  emitter.int3();
  return absl::OkStatus();
}

absl::Status GenerateCode(const CallNativeInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  // Set the bytecode location in case we cause an exception.
  ctx.pyframe().f_lasti().Store(asmjit::imm(instr.bytecode_offset()), emitter);

  // The register allocator ensures ABI compliance with copies, so all we need
  // to do is call and potentially move the result from RAX.
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  S6_ASSIGN_OR_RETURN(auto address, GetCalleeSymbolAddress(instr.callee()));
  emitter.call(asmjit::imm(address));
  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const CallNativeIndirectInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  ctx.pyframe().f_lasti().Store(asmjit::imm(instr.bytecode_offset()), emitter);

  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  asmjit::x86::Gp callee_register = ctx.OperandInRegister(instr.callee());
  emitter.call(callee_register);
  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const CompareInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  if (ctx.ConditionIsOptimizable(&instr)) return absl::OkStatus();

  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp s1 = ctx.OperandInRegister(instr.lhs());
  asmjit::Operand s2 = ctx.Operand(instr.rhs());
  switch (instr.type()) {
    case NumericInst::kInt64: {
      x86::Condition::Code condition =
          ComparisonToAsmjitCondCode(instr.comparison());
      // Because dst may equal s1 or s2, instead of using xor to zero out dst
      // here, emit the cmp early and use mov instead. It's a longer encoding
      // but doesn't affect the flags.
      CMP(s1, s2);
      // Set 'dst' to 0 or 1 based on the comparison.
      emitter.set(condition, dst.r8());
      emitter.and_(dst.r32(), 1);  // reset all bits to 0 except the LSB.
      break;
    }

    case NumericInst::kDouble: {
      // First move operands to SSE unit.
      // TODO: Enhance register allocator to allocate SSE regs.
      // Move `s1` first; this frees up `scratch` for relaying `s2`.
      x86::Gp scratch = ctx.scratch_reg();
      emitter.movq(x86::xmm0, s1);
      asmjit::Operand xmm_rhs = x86::xmm1;
      if (s2.isReg()) {
        emitter.movq(x86::xmm1, s2.as<x86::Gp>());
      } else if (s2.isMem()) {
        xmm_rhs = s2;
      } else {
        S6_CHECK(s2.isImm());
        // TODO: Embbed the constant in memory after the end of the
        // function and emit a cmpsd with a memory second operand.
        MOV(scratch, s2);
        emitter.movq(x86::xmm1, scratch);
      }

      EMIT_INST(x86::Inst::kIdCmpsd, x86::xmm0, xmm_rhs,
                ComparisonToAsmjitPredicateCmp(instr.comparison()));
      emitter.movq(dst, x86::xmm0);
      emitter.and_(dst.r32(), 1);  // reset all bits to 0 except the LSB.
    }
  }
  return absl::OkStatus();
}

absl::Status GenerateCode(const ExceptInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  const Block* succ = instr.unique_successor();
  ctx.pyframe().f_lasti().Store(asmjit::imm(instr.bytecode_offset()), emitter);

  // If we have no successor, we still call the runtime but a simpler function
  // that doesn't return anything. We then return zero.
  if (!succ) {
    // Signature: ExceptWithoutHandler(bytecode_offset)
    emitter.mov(AbiLocation<0>(), asmjit::imm(instr.bytecode_offset()));
    emitter.call(asmjit::imm(ExceptWithoutHandler));
    emitter.xor_(x86::rax, x86::rax);
    emitter.jmp(ctx.cleanup_point());
    return absl::OkStatus();
  }

  // The exception handling is done by a runtime call. This returns six items
  // on the stack.
  //
  // Assert that we need room for 6 values on the call stack.
  (void)ctx.call_stack_slot(5);

  // Signature: Except(bytecode_offset, output_values)
  CallStub<Except> except_stub;
  emitter.mov(except_stub.bytecode_offset(),
              asmjit::imm(instr.bytecode_offset()));
  emitter.mov(except_stub.args(), x86::rsp);
  emitter.call(except_stub.imm());

  // Because we've marked ourselves as kClobbersAllRegisters, the source operand
  // of these copies must NEVER be a register.
  //
  // Still, these could use values in frame slots that are clobbered by the
  // below copy loop, so we do this first.
  const RegisterAllocation& ra = ctx.register_allocation();
  for (const RegisterAllocation::Copy& copy :
       ra.block_copies(instr.parent(), /*successor_index=*/0)) {
    if (copy.first.IsInRegister()) {
      S6_LOG_LINES(WARNING, FormatOrDie(ctx.function()));
    }
    S6_CHECK(!copy.first.IsInRegister())
        << "copy after exceptinst somehow in register? it's been clobbered!";
    S6_RETURN_IF_ERROR(GenerateCopy(copy, emitter, ctx));
  }

  // We have mandated that all arguments to this exceptinst go in
  // call_stack_slot(6)+. Therefore we can simply move all values from the call
  // stack to their required locations.
  S6_CHECK_EQ(succ->block_arguments_size(),
              6 + instr.successor_arguments(0).size());
  auto it = succ->block_arguments().begin();
  for (int64_t i = 0; i < 6 + instr.arguments().size(); ++i) {
    const Location& loc = ra.DestinationLocation(**it++);
    if (loc.IsInRegister()) {
      emitter.mov(loc.Register().as<x86::Gp>(), ctx.call_stack_slot(i));
    } else {
      S6_CHECK(loc.IsFrameSlot()) << "Copy to neither register nor frame slot?";
      emitter.mov(ctx.scratch_reg(), ctx.call_stack_slot(i));
      emitter.mov(ctx.spill_slot(loc.FrameSlot()), ctx.scratch_reg());
    }
  }

  emitter.jmp(ctx.block_label(succ));
  return absl::OkStatus();
}

// Catch-all for all unimplemented instructions.
absl::Status GenerateCode(const Instruction& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  return absl::UnimplementedError(
      absl::StrCat("code generation for ", FormatOrDie(instr)));
}

absl::Status GenerateCode(const UnaryInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  static const NoDestructor<absl::flat_hash_map<Value::Kind, x86::Inst::Id>>
      inst_ids({
          {Value::kNegate, x86::Inst::kIdNeg},
          {Value::kNot, x86::Inst::kIdNot},
      });
  S6_RET_CHECK(inst_ids->contains(instr.kind()));
  x86::Inst::Id inst_id = inst_ids->at(instr.kind());

  asmjit::Operand operand = ctx.Operand(instr.operand());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  // inst_id (e.g. not, neg) will modify in place.
  // Move operand to destination then apply the operation there.
  if (dst != operand) MOV(dst, operand);
  EMIT_INST(inst_id, dst);
  return absl::OkStatus();
}

absl::Status GenerateCode(const BinaryInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  static const NoDestructor<absl::flat_hash_map<Value::Kind, x86::Inst::Id>>
      inst_ids({
          {Value::kAdd, x86::Inst::kIdAdd},
          {Value::kSubtract, x86::Inst::kIdSub},
          {Value::kMultiply, x86::Inst::kIdImul},
          {Value::kAnd, x86::Inst::kIdAnd},
          {Value::kOr, x86::Inst::kIdOr},
          {Value::kXor, x86::Inst::kIdXor},
          {Value::kShiftLeft, x86::Inst::kIdShl},
          {Value::kShiftRightSigned, x86::Inst::kIdSar},
      });
  static const NoDestructor<absl::flat_hash_map<Value::Kind, x86::Inst::Id>>
      fp_inst_ids({
          {Value::kAdd, x86::Inst::kIdAddsd},
          {Value::kSubtract, x86::Inst::kIdSubsd},
          {Value::kMultiply, x86::Inst::kIdMulsd},
          {Value::kDivide, x86::Inst::kIdDivsd},
      });

  x86::Gp lhs = ctx.OperandInRegister(instr.lhs());
  asmjit::Operand rhs = ctx.Operand(instr.rhs());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  switch (instr.type()) {
    case NumericInst::kInt64: {
      S6_RET_CHECK(inst_ids->contains(instr.kind()));
      x86::Inst::Id inst_id = inst_ids->at(instr.kind());

      if (dst == rhs && lhs != dst) {
        // Accumulate the result in `scratch` (which may already contain `lhs`)
        // to preserve `rhs` until we need it.
        x86::Gp scratch = ctx.scratch_reg();
        if (lhs != scratch) emitter.mov(scratch, lhs);
        EMIT_INST(inst_id, scratch, rhs);
        emitter.mov(dst, scratch);
      } else {
        // Accumulate the result in `dst` to avoid modifying other registers.
        if (dst != lhs) MOV(dst, lhs);
        EMIT_INST(inst_id, dst, rhs);
      }
      break;
    }

    case NumericInst::kDouble: {
      S6_RET_CHECK(fp_inst_ids->contains(instr.kind()));
      x86::Inst::Id inst_id = fp_inst_ids->at(instr.kind());

      // First move operands to SSE unit.
      // TODO: Enhance register allocator to allocate SSE regs.
      // Move `lhs` first; this frees up `scratch` for relaying `rhs`.
      emitter.movq(x86::xmm0, lhs);
      asmjit::Operand xmm_rhs = x86::xmm1;
      if (rhs.isReg()) {
        emitter.movq(x86::xmm1, rhs.as<x86::Gp>());
      } else if (rhs.isMem()) {
        xmm_rhs = rhs;
      } else {
        S6_CHECK(rhs.isImm());
        // TODO: Embbed the constant in memory after the end of the
        // function and emit an instruction with a memory second operand.
        x86::Gp scratch = ctx.scratch_reg();
        MOV(scratch, rhs);
        emitter.movq(x86::xmm1, scratch);
      }
      // Accumulate result in `xmm0`.
      EMIT_INST(inst_id, x86::xmm0, xmm_rhs);
      emitter.movq(dst, x86::xmm0);
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status GenerateIntegerDivisionCode(const BinaryInst& instr,
                                         x86::Emitter& emitter,
                                         CodeGenerationContext& ctx) {
  asmjit::Operand lhs = ctx.Operand(instr.operands()[0]);
  asmjit::Operand rhs = ctx.Operand(instr.operands()[1]);
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));

  // The idiv instruction leaves the quotient in rax and the remainder in rdx.
  // Determine which of those is the result (dst), and which needs backing up.
  x86::Gp idiv_ignore_reg;
  switch (instr.kind()) {
    case Value::kDivide:
      S6_RET_CHECK(dst == x86::rax);
      idiv_ignore_reg = x86::rdx;
      break;
    case Value::kRemainder:
      S6_RET_CHECK(dst == x86::rdx);
      idiv_ignore_reg = x86::rax;
      break;
    default:
      S6_RET_CHECK_FAIL();
  }

  // Check if the operands are of the same sign.
  x86::Gp scratch = ctx.scratch_reg();
  MOV(scratch, lhs);
  EMIT_INST(x86::Inst::kIdXor, scratch, rhs);

  // Preserve existing contents of whichever of rdx/rax doesn't hold the result.
  // TODO: Express in register_allocator that rax/rdx are clobbered.
  emitter.mov(scratch, idiv_ignore_reg);
  if (rhs == idiv_ignore_reg) rhs = scratch;
  // There are a couple of scenarios in which we need to move rhs to scratch.
  // Firstly, idiv cannot accept an immediate rhs.
  // Secondly, if rhs is dst (e.g. rax = divide i64 lhs, rax) then we need to
  // keep rhs intact while loading lhs into rdx:rax.
  // (We could simplify this by having the register allocator put rhs in rcx;
  // however, that would needlessly produce separate "mov rcx, mem", "idiv rcx"
  // instructions if rhs is in memory.)
  bool rhs_in_scratch = rhs.isImm() || rhs == dst;
  if (rhs_in_scratch) {
    emitter.push(scratch);
    MOV(scratch, rhs);
    rhs = scratch;
  }

  // Move lhs to rdx:rax, sign-extending into rdx.
  if (lhs != x86::rax) MOV(x86::rax, lhs);
  emitter.cqo();

  // The preceding mov/push/cqo instructions don't affect the flags.
  asmjit::Label out = emitter.newLabel();
  asmjit::Label different_signs = emitter.newLabel();
  emitter.jl(different_signs);

  // Operands have the same sign.

  // Perform the division, leaving the quotient in rax and the remainder in rdx.
  EMIT_INST(x86::Inst::kIdIdiv, rhs);

  emitter.jmp(out);
  emitter.bind(different_signs);

  // Operands have opposite signs.

  // Perform the division, leaving the quotient in rax and the remainder in rdx.
  EMIT_INST(x86::Inst::kIdIdiv, rhs);
  emitter.test(x86::rdx, x86::rdx);
  emitter.jz(out);  // There was no remainder.

  // With opposite signs, the quotient will be negative, so x86 rounds up.
  // However, Python always rounds the quotient down.
  // Account for this by decrementing the result.
  if (instr.kind() == Value::kDivide) {
    emitter.dec(x86::rax);
  } else {
    // If lhs < 0 and rhs > 0 then -rhs < remainder < 0, and
    // if lhs > 0 and rhs < 0 then 0 < remainder < -rhs (x86 idiv semantics).
    // Either way, we must now add rhs to change the sign of remainder,
    // to change from x86 to Python behaviour.
    // Combined with decrementing the quotient, this maintains the
    // relation:  lhs = quotient×rhs + remainder
    EMIT_INST(x86::Inst::kIdAdd, x86::rdx, rhs);
  }

  emitter.bind(out);

  // Restore whichever of rdx/rax doesn't hold the result.
  if (rhs_in_scratch) {
    emitter.pop(scratch);
  }
  emitter.mov(idiv_ignore_reg, scratch);
  return absl::OkStatus();
}

absl::Status GenerateCode(const DivideInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  switch (instr.type()) {
    case NumericInst::kInt64:
      return GenerateIntegerDivisionCode(instr, emitter, ctx);

    case NumericInst::kDouble:
      // Floating point division is a 'regular' binary opcode.
      // Use the default implementation.
      return GenerateCode(static_cast<const BinaryInst&>(instr), emitter, ctx);
  }
  S6_UNREACHABLE();
}

absl::Status GenerateCode(const RemainderInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  S6_RET_CHECK(instr.type() == NumericInst::kInt64);
  return GenerateIntegerDivisionCode(instr, emitter, ctx);
}

absl::Status GenerateCode(const NegateInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  switch (instr.type()) {
    case NumericInst::kInt64:
      // Use the default implementation.
      return GenerateCode(static_cast<const UnaryInst&>(instr), emitter, ctx);

    case NumericInst::kDouble: {
      // There is no 'floating point negate' instruction on x86.
      // However we can just get the same result by flipping the sign bit.
      asmjit::Operand operand = ctx.Operand(instr.operand());
      S6_CHECK(!operand.isImm()) << "An immediate before a negate should have "
                                    "been constant propagated";
      S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
      if (dst == operand) {
        x86::Gp scratch = ctx.scratch_reg();
        emitter.mov(scratch, asmjit::imm(uint64_t{1} << 63));
        emitter.xor_(dst, scratch);
      } else {
        // dst != operand
        emitter.mov(dst, asmjit::imm(uint64_t{1} << 63));
        XOR(dst, operand);
      }
      return absl::OkStatus();
    }
  }
  S6_UNREACHABLE();
}

absl::Status GenerateCode(const ShiftLeftInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  S6_RET_CHECK(instr.type() == NumericInst::kInt64);
  x86::Gp lhs = ctx.OperandInRegister(instr.lhs());
  asmjit::Operand rhs = ctx.Operand(instr.rhs());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp scratch = ctx.scratch_reg();

  if (rhs.isImm() && rhs.as<asmjit::Imm>().i64() >= 64) {
    return absl::FailedPreconditionError("Statically creating shift >= 64");
  }

  // The `shl` instruction does not set the overflow flag in general.
  // Do the overflow detection explicitly.
  asmjit::Label out = emitter.newLabel();
  asmjit::Label no_overflow = emitter.newLabel();

  bool can_clobber_dst = dst != lhs && dst != rhs;

  // Use `scratch` to test for overflow.
  // Overflow occurs iff the left-shift can be undone by a right-shift.
  emitter.mov(scratch, lhs);
  EMIT_INST(x86::Inst::kIdShl, scratch, rhs);
  if (can_clobber_dst) {
    // We now have the answer (subject to overflow checking).
    // Move it to the destination. This will preserve lhs and rhs,
    // which we still need intact for the overflow check.
    emitter.mov(dst, scratch);
  }
  EMIT_INST(x86::Inst::kIdSar, scratch, rhs);
  emitter.cmp(scratch, lhs);
  emitter.je(no_overflow);
  GenerateSetOverflowFlag(scratch, emitter);
  emitter.jmp(out);
  emitter.bind(no_overflow);

  if (!can_clobber_dst) {
    // Recompute the left-shift.
    emitter.mov(dst, scratch);
    // Evaluate the left-shift as a conventional binary operator.
    S6_RETURN_IF_ERROR(
        GenerateCode(static_cast<const BinaryInst&>(instr), emitter, ctx));
  }

  emitter.bind(out);
  return absl::OkStatus();
}

absl::Status GenerateCode(const IntToFloatInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  x86::Gp src = ctx.OperandInRegister(instr.operands().front());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  emitter.cvtsi2sd(x86::xmm0, src);
  emitter.movq(dst, x86::xmm0);
  return absl::OkStatus();
}

absl::Status GenerateCode(const SextInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Operand src = ctx.Operand(instr.operands().front());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  if (src.isMem()) {
    RETURN_IF_ASMJIT_ERROR(emitter.movsxd(dst, src.as<x86::Mem>()));
  } else {
    S6_CHECK(src.isReg());
    RETURN_IF_ASMJIT_ERROR(emitter.movsxd(dst, src.as<x86::Gp>().r32()));
  }
  return absl::OkStatus();
}

absl::Status GenerateCode(const BoxInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  x86::Gp value_reg = ctx.OperandInRegister(instr.content());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));

  switch (instr.type()) {
    case UnboxableType::kPyLong:
      emitter.call(asmjit::imm(PyLong_FromLong));
      break;
    case UnboxableType::kPyBool:
      emitter.call(asmjit::imm(PyBool_FromLong));
      break;
    case UnboxableType::kPyFloat:
      // Operand is a 64-bit float.
      // Move it to the low 64 bits of xmm0, as the argument to FromDouble.
      emitter.movq(x86::xmm0, value_reg);
      emitter.call(asmjit::imm(PyFloat_FromDouble));
      break;
  }

  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const UnboxInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  x86::Gp obj_reg = ctx.OperandInRegister(instr.boxed());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp scratch = ctx.scratch_reg();
  asmjit::Label out = emitter.newLabel();

  // Check that the boxed object is of the correct type.
  const PyTypeObject* type_object = nullptr;
  switch (instr.type()) {
    case UnboxableType::kPyLong:
      type_object = &PyLong_Type;
      break;
    case UnboxableType::kPyBool:
      type_object = &PyBool_Type;
      break;
    case UnboxableType::kPyFloat:
      type_object = &PyFloat_Type;
      break;
  }

  asmjit::Label correct_type = emitter.newLabel();
  emitter.mov(scratch, asmjit::imm(type_object));
  emitter.cmp(scratch, x86::ptr(obj_reg, offsetof(PyObject, ob_type)));
  emitter.je(correct_type);
  // Wrong type.
  // Get out of jail free: PyLong can be unboxed as a PyFloat.
  if (instr.type() == UnboxableType::kPyFloat) {
    emitter.mov(scratch, asmjit::imm(&PyLong_Type));
    emitter.cmp(scratch, x86::ptr(obj_reg, offsetof(PyObject, ob_type)));
    emitter.je(correct_type);
  }
  GenerateSetOverflowFlag(scratch, emitter);
  emitter.jmp(out);
  emitter.bind(correct_type);

  switch (instr.type()) {
    case UnboxableType::kPyLong:
      // PyLong_AsLongAndOverflow takes an extra pointer argument,
      // which it populates on exit with overflow state.
      emitter.push(asmjit::imm(0));
      emitter.mov(AbiLocation<1>(), x86::rsp);
      emitter.call(asmjit::imm(PyLong_AsLongAndOverflow));
      emitter.pop(scratch);     // +/-1 if overflow; 0 otherwise
      emitter.ror(scratch, 1);  // MSB <- LSB
      // Single-shift SHR sets overflow flag if result's top two bits differ.
      emitter.shr(scratch, 1);  // OF <- MSB
      if (dst != x86::rax) emitter.mov(dst, x86::rax);
      break;
    case UnboxableType::kPyBool:
      // Because dst may equal obj_reg, instead of using xor to zero out dst,
      // emit the cmp early and use mov instead. It's a longer encoding but
      // doesn't affect the flags.
      emitter.mov(scratch, asmjit::imm(Py_True));
      emitter.cmp(obj_reg, scratch);
      // Set 'dst' to 0 or 1 based on the result of comparison with Py_True.
      emitter.sete(dst.r8());
      emitter.and_(dst.r32(), 1);  // reset all bits to 0 except the LSB.
      break;
    case UnboxableType::kPyFloat:
      emitter.call(asmjit::imm(PyFloat_AsDouble));
      // Return value is a 64-bit float in xmm0.
      emitter.movq(dst, x86::xmm0);
      break;
  }

  emitter.bind(out);
  return absl::OkStatus();
}

absl::Status GenerateCode(const OverflowedInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  if (ctx.ConditionIsOptimizable(&instr)) return absl::OkStatus();

  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  // Set 'dst' to 0 or 1 based on the overflow flag.
  emitter.seto(dst.r8());
  emitter.and_(dst.r32(), 1);  // reset all bits to 0 except the LSB.
  return absl::OkStatus();
}

absl::Status GenerateCode(const FloatZeroInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  if (ctx.ConditionIsOptimizable(&instr)) return absl::OkStatus();

  asmjit::Operand arg = ctx.Operand(instr.float_value());
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  // Set 'dst' to 0 or 1 based on whether the operand is fp zero.
  // Accept +0 and -0 alike; account for this by shifting out the sign bit.
  if (dst != arg) MOV(dst, arg);
  emitter.shl(dst, asmjit::imm(1));
  emitter.setz(dst.r8());
  emitter.and_(dst.r32(), 1);  // reset all bits to 0 except the LSB.
  return absl::OkStatus();
}

absl::Status GenerateCode(const GetClassIdInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Label get_type_class = emitter.newLabel();
  asmjit::Label out = emitter.newLabel();
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.object());

  // Load type into scratch. Note that this could clobber `operand`, but we'll
  // reload it later if that is the case.
  JitStub<PyObject> object(x86::ptr(operand));
  JitStub<PyTypeObject> type =
      object.ob_type().Load(ctx.scratch_reg(), emitter);
  JitStub<int64_t> dictoffset =
      type.tp_dictoffset().Load(ctx.scratch_reg(), emitter);
  emitter.cmp(dictoffset.Reg(), asmjit::imm(0));
  // dictoffset <= 0 -> get the class from the type.
  emitter.jle(get_type_class);

  // Now calculate the address of the dict pointer. We need `operand` and
  // `dictoffset` for this, so if they alias...
  if (operand == ctx.scratch_reg()) {
    // ... add the operand back into dictoffset and dereference.
    emitter.add(dictoffset.Reg(), ctx.Operand(instr.object()).as<x86::Mem>());
    emitter.mov(ctx.scratch_reg(), x86::ptr(dictoffset.Reg()));
  } else {
    // They didn't alias so we can load the dict with a single mov.
    emitter.mov(ctx.scratch_reg(), x86::ptr(operand, dictoffset.Reg()));
  }
  JitStub<PyDictObject> dict(x86::ptr(ctx.scratch_reg()));
  emitter.test(dict.Reg(), dict.Reg());
  emitter.jz(get_type_class);  // Dict was nullptr.

  dict.ma_version_tag().Load(dst, emitter);
  emitter.jmp(out);

  emitter.bind(get_type_class);
  // Reload the type, because scratch is undefined.
  operand = ctx.OperandInRegister(instr.object());
  object = JitStub<PyObject>(x86::ptr(operand));
  type = object.ob_type().Load(ctx.scratch_reg(), emitter);
  type.tp_version_tag().Load(dst, emitter);

  emitter.bind(out);
  emitter.shr(dst, asmjit::imm(44));

  return absl::OkStatus();
}

absl::Status GenerateCode(const GetObjectDictInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  asmjit::Label fail = emitter.newLabel();
  asmjit::Label out = emitter.newLabel();
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.object());
  JitStub<PyObject> object(x86::ptr(operand));

  // There are three versions we emit depending on whether we have type,
  // dictoffset or neither.
  // See go/s6-hidden-classes for full algorithm description.
  if (instr.type() != 0 && object.Reg() != ctx.scratch_reg()) {
    S6_CHECK_GT(instr.dictoffset(), 0);
    emitter.mov(ctx.scratch_reg(), asmjit::imm(instr.type()));
    emitter.cmp(object.ob_type().Mem(), ctx.scratch_reg());
    emitter.jnz(fail);
    emitter.mov(dst, x86::qword_ptr(object.Reg(), instr.dictoffset()));
    emitter.jmp(out);
  } else if (instr.dictoffset()) {
    JitStub<PyTypeObject> type =
        object.ob_type().Load(ctx.scratch_reg(), emitter);
    emitter.cmp(type.tp_dictoffset().Mem(), asmjit::imm(instr.dictoffset()));
    emitter.jnz(fail);
    // Reload if we corrupted object.
    if (object.Reg() == ctx.scratch_reg()) {
      operand = ctx.OperandInRegister(instr.object());
    }
    emitter.mov(dst, x86::qword_ptr(object.Reg(), instr.dictoffset()));
    emitter.jmp(out);
  } else {
    JitStub<PyTypeObject> type =
        object.ob_type().Load(ctx.scratch_reg(), emitter);
    JitStub<int64_t> dictoffset =
        type.tp_dictoffset().Load(ctx.scratch_reg(), emitter);
    emitter.cmp(dictoffset.Reg(), asmjit::imm(0));
    emitter.jle(fail);

    // Now calculate the address of the dict pointer. We need `operand` and
    // `dictoffset` for this, so if they alias...
    if (operand == ctx.scratch_reg()) {
      // ... add the operand back into dictoffset and dereference.
      emitter.add(dictoffset.Reg(), ctx.Operand(instr.object()).as<x86::Mem>());
      emitter.mov(dst, x86::ptr(dictoffset.Reg()));
    } else {
      // They didn't alias so we can load the dict with a single mov.
      emitter.mov(dst, x86::ptr(operand, dictoffset.Reg()));
    }
    emitter.jmp(out);
  }

  emitter.bind(fail);
  emitter.xor_(dst, dst);
  emitter.bind(out);

  return absl::OkStatus();
}

absl::Status GenerateCode(const GetInstanceClassIdInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.dict());
  JitStub<PyDictObject> dict(x86::ptr(operand));
  dict.ma_version_tag().Load(dst, emitter);
  emitter.shr(dst, asmjit::imm(44));
  return absl::OkStatus();
}

absl::Status GenerateCode(const CheckClassIdInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  const Class* cls = instr.class_();
  asmjit::Label fail = emitter.newLabel();
  asmjit::Label out = emitter.newLabel();
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.object());
  JitStub<PyObject> object(x86::ptr(operand));

  // If operand is in scratch, we may clobber it here. But if operand is in
  // scratch, then `dst` cannot alias operand, so use `dst`.
  x86::Gp scratch2 = ctx.scratch_reg();
  if (operand == ctx.scratch_reg()) {
    scratch2 = dst;
  }

  // First, check for a globals class. A globals class doesn't know the type of
  // the object that it is the dict of, however it is a good guess that it is
  // a PyModuleObject so we try that first.
  if (cls->is_globals_class()) {
    asmjit::Label slow_type_lookup = emitter.newLabel();
    emitter.mov(scratch2, asmjit::imm(&PyModule_Type));
    emitter.cmp(object.ob_type().Mem(), scratch2);
    emitter.jnz(slow_type_lookup);

    // The type was PyModule_Type.
    int64_t dictoffset = PyModule_Type.tp_dictoffset;
    emitter.mov(dst, x86::ptr(operand, dictoffset));

    // The below code is shared with the slow path.
    asmjit::Label dict_loaded = emitter.newLabel();
    emitter.bind(dict_loaded);

    emitter.test(dst, dst);
    emitter.jz(fail);
    JitStub<PyDictObject> dict(x86::ptr(dst));
    dict.ma_version_tag().Load(dst, emitter);
    emitter.shr(dst, asmjit::imm(44));
    emitter.cmp(dst, asmjit::imm(instr.class_id()));
    emitter.jnz(fail);
    emitter.mov(dst, asmjit::imm(1));
    emitter.jmp(out);

    // If we get here, we don't know the type so we have to look it up and
    // find the dictoffset.
    emitter.bind(slow_type_lookup);
    JitStub<PyTypeObject> type = object.ob_type().Load(scratch2, emitter);
    type.tp_dictoffset().Load(scratch2, emitter);
    emitter.cmp(scratch2, asmjit::imm(0));
    emitter.jle(fail);
    emitter.mov(dst, x86::ptr(operand, scratch2));
    emitter.jmp(dict_loaded);
  } else {
    // Otherwise we are not a globals class; we have type and dictoffset
    // information.
    emitter.mov(scratch2, asmjit::imm(cls->type()));
    emitter.cmp(object.ob_type().Mem(), scratch2);
    emitter.jnz(fail);

    // See go/s6-hidden-classes for full algorithm description.
    if (cls->is_type_class() || !cls->is_base_class()) {
      emitter.mov(dst, x86::ptr(operand, cls->dictoffset()));
      emitter.test(dst, dst);
      emitter.jz(fail);
      JitStub<PyDictObject> dict(x86::ptr(dst));
      dict.ma_version_tag().Load(dst, emitter);
      emitter.shr(dst, asmjit::imm(44));
      emitter.cmp(dst, asmjit::imm(instr.class_id()));
      emitter.jnz(fail);
      emitter.mov(dst, asmjit::imm(1));
      emitter.jmp(out);
    } else if (cls->dictoffset() > 0) {
      emitter.mov(dst, x86::ptr(operand, cls->dictoffset()));
      emitter.cmp(dst, asmjit::imm(0));
      emitter.jnz(fail);
      emitter.mov(dst, asmjit::imm(1));
      emitter.jmp(out);
    } else {
      S6_CHECK_EQ(cls->dictoffset(), 0);
      emitter.mov(dst, asmjit::imm(1));
      emitter.jmp(out);
    }
  }

  emitter.bind(fail);
  emitter.xor_(dst, dst);
  emitter.bind(out);

  return absl::OkStatus();
}

absl::Status GenerateCodeForPyFunctionCall(const CallPythonInst& instr,
                                           x86::Emitter& emitter,
                                           CodeGenerationContext& ctx) {
  const ConstantAttributeInst* info =
      cast<ConstantAttributeInst>(instr.callee());
  const Attribute& attr = info->LookupAttribute(ClassManager::Instance());
  S6_CHECK(attr.kind() == Attribute::kFunction);
  const FunctionAttribute& method_attr =
      static_cast<const FunctionAttribute&>(attr);

  PyObject* callee = method_attr.value();
  S6_RET_CHECK_EQ(instr.names(), nullptr);

  emitter.mov(AbiLocation<0>(), asmjit::imm(callee));

  // CallPython steals the callee so we must steal the constant attribute.
  // A constant attribute cannot be deleted so we do not need to check if the
  // refcount reaches 0.
  // TODO When changing call_python semantics to not steal the
  // callee, those next two lines must be removed.
  JitStub<PyObject> obj_stub(x86::ptr(AbiLocation<0>()));
  emitter.dec(obj_stub.ob_refcnt().Mem());

  Metadata* m = Metadata::Get(method_attr.code());
  emitter.call(
      asmjit::imm(m->GetOrCreateThunk(ctx.jit_allocator()).GetEntry()));

  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const CallPythonInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  ctx.pyframe().f_lasti().Store(asmjit::imm(instr.bytecode_offset()), emitter);

  // Fast path:
  if (isa<ConstantAttributeInst>(instr.callee()) && instr.fastcall()) {
    return GenerateCodeForPyFunctionCall(instr, emitter, ctx);
  }

  // Slow path:
  // We emit a runtime call: CallPython(callee, num_operands, operands, names)
  // where `operands` is an array of operands.
  //
  // We add one extra operand that we leave undefined (accessible as args[-1])
  // for the runtime to use. This is used to allow the runtime to prepend a
  // `self` argument for method calls.
  //
  // Note that this function also emits CallAttributeInsts, and CallAttribute
  // takes the same arguments as CallPython plus two more.
  CallStub<CallAttribute> stub;
  int64_t num_operands = instr.call_arguments().size();

  // The register allocator has already placed all called_operands prepended by
  // `callee` on the call stack.
  emitter.lea(stub.args(), ctx.call_stack_slot(1));
  ctx.stack_frame().Load(stub.stack_frame(), emitter);

  // Only the names arguments remain, and these should already be in place due
  // to register allocation constraints.
  x86::Gp arg_count = stub.arg_count();
  emitter.mov(arg_count, asmjit::imm(num_operands));

  // Pass nullptr as names if they don't exist.
  x86::Gp names = stub.names();
  if (!instr.names()) {
    emitter.xor_(names, names);
  } else {
    // Otherwise, adjust the argument count to discount the number of arguments
    // passed by name.
    // We inline PyTuple_Size.
    emitter.sub(arg_count,
                x86::qword_ptr(names, offsetof(PyVarObject, ob_size)));
  }

  if (const CallAttributeInst* call_attr =
          dyn_cast<CallAttributeInst>(&instr)) {
    auto str = call_attr->attribute_str();
    PyObject* str_obj = FindStringInTuple(ctx.py_code_object()->co_names, str);
    emitter.mov(stub.attr_str(), asmjit::imm(str_obj));
    emitter.mov(stub.call_python_bytecode_offset(),
                asmjit::imm(call_attr->call_python_bytecode_offset()));
    emitter.call(asmjit::imm(CallAttribute));
  } else {
    emitter.call(asmjit::imm(CallPython));
  }

  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  if (dst != x86::rax) emitter.mov(dst, x86::rax);
  return absl::OkStatus();
}

absl::Status GenerateCode(const AdvanceProfileCounterInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  auto out = emitter.newLabel();
  auto reg = ctx.profile_counter_reg();

  emitter.sub(x86::qword_ptr(reg), asmjit::imm(instr.amount()));
  emitter.jg(out);
  emitter.call(asmjit::imm(ProfileEventReachedZero));
  emitter.bind(out);
  return absl::OkStatus();
}

absl::Status GenerateCode(const IncrementEventCounterInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  int64_t* counter =
      EventCounters::Instance().GetEventCounter(instr.name_str());
  emitter.mov(ctx.scratch_reg(), asmjit::imm(counter));
  emitter.inc(x86::qword_ptr(ctx.scratch_reg()));
  return absl::OkStatus();
}

absl::Status GenerateCode(const TraceBeginInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  emitter.mov(AbiLocation<0>(), asmjit::imm(instr.name().AsPtr()));
  return absl::OkStatus();
}

absl::Status GenerateCode(const TraceEndInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  emitter.mov(AbiLocation<0>(), asmjit::imm(instr.name().AsPtr()));
  return absl::OkStatus();
}

absl::Status GenerateCode(const RematerializeInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  return absl::OkStatus();
}

absl::Status GenerateCode(const LoadFromDictInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.dict());
  JitStub<PyDictObject> dict(x86::ptr(operand));

  if (instr.dict_kind() == DictKind::kSplit) {
    dict.ma_values().Load(ctx.scratch_reg(), emitter);

    x86::Mem location = x86::ptr(ctx.scratch_reg(), 8 * instr.index());
    emitter.mov(dst, location);
    return absl::OkStatus();
  }

  S6_CHECK(instr.dict_kind() == DictKind::kCombined);
  if (operand == ctx.scratch_reg()) {
    return absl::ResourceExhaustedError("Need more scratch regs");
  }
  JitStub<PyDictKeysObject> keys = dict.ma_keys().Load(operand, emitter);
  JitStub<int64_t> size = keys.dk_size().Load(ctx.scratch_reg(), emitter);

  // We are going to clobber operand here, so save it unless it's safe to
  // clobber.
  if (operand != dst) {
    emitter.mov(ctx.call_stack_slot(0), operand);
  }

  // The multiplier to use depends on size.
  asmjit::Label check16 = emitter.newLabel();
  asmjit::Label out = emitter.newLabel();

  emitter.cmp(size.Reg(), asmjit::imm(0xff));
  emitter.jg(check16);
  // dk_indices + (size * 1);
  // Each _PyDictKeyEntry is {hash, key, value}, so add 16 to read value.
  x86::Mem mem = keys.dk_indices().Mem().cloneAdjusted(
      sizeof(_PyDictKeyEntry) * instr.index() + 16);
  mem.setIndex(size.Reg());
  mem.setShift(0);
  emitter.mov(dst, mem);
  emitter.jmp(out);

  emitter.bind(check16);
  // TODO: Implement when we have test coverage.
  emitter.int3();

  emitter.bind(out);
  // restore operand unless operand == dst.
  if (operand != dst) emitter.mov(operand, ctx.call_stack_slot(0));
  return absl::OkStatus();
}

absl::Status GenerateCode(const StoreToDictInst& instr, x86::Emitter& emitter,
                          CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  x86::Gp operand = ctx.OperandInRegister(instr.dict());
  x86::Gp value = ctx.OperandInRegister(instr.value());
  if (operand == ctx.scratch_reg() || value == ctx.scratch_reg()) {
    return absl::InternalError(
        "Not enough registers to emit code; we need a scratch reg!");
  }

  JitStub<PyDictObject> dict(x86::ptr(operand));

  if (instr.dict_kind() == DictKind::kSplit) {
    dict.ma_values().Load(ctx.scratch_reg(), emitter);
    x86::Mem location = x86::ptr(ctx.scratch_reg(), 8 * instr.index());

    if (dst == value) {
      // Emit a more costly sequence that doesn't require an extra register.
      emitter.mov(operand, value);
      emitter.mov(dst, location);
      emitter.mov(location, operand);
    } else {
      // Avoid using xchg, it causes a bus lock.
      emitter.mov(dst, location);
      emitter.mov(location, value);
    }
    return absl::OkStatus();
  }

  S6_CHECK(instr.dict_kind() == DictKind::kCombined);
  if (value == ctx.scratch_reg()) {
    return absl::ResourceExhaustedError("Need more scratch regs");
  }

  // We are going to clobber operand here, so save it unless it's safe to
  // clobber.
  if (operand != dst) {
    emitter.mov(ctx.call_stack_slot(0), operand);
  }

  JitStub<PyDictKeysObject> keys = dict.ma_keys().Load(operand, emitter);
  JitStub<int64_t> size = keys.dk_size().Load(ctx.scratch_reg(), emitter);

  // The multiplier to use depends on size. We perform an LEA to get the
  // location address then jump to common code.
  asmjit::Label check16 = emitter.newLabel();
  asmjit::Label common = emitter.newLabel();

  emitter.cmp(size.Reg(), asmjit::imm(0xff));
  emitter.jg(check16);
  // dk_indices + (size * 1);
  // Each _PyDictKeyEntry is {hash, key, value}, so add 16 to read value.
  x86::Mem mem = keys.dk_indices().Mem().cloneAdjusted(
      sizeof(_PyDictKeyEntry) * instr.index() + 16);
  mem.setIndex(size.Reg());
  mem.setShift(0);
  emitter.lea(ctx.scratch_reg(), mem);
  emitter.jmp(common);

  emitter.bind(check16);
  // TODO: Implement when we have test coverage.
  emitter.int3();

  emitter.bind(common);
  x86::Mem location = x86::qword_ptr(ctx.scratch_reg());
  if (dst == value) {
    // Emit a more costly sequence that doesn't require an extra register.
    emitter.mov(operand, value);
    emitter.mov(dst, location);
    emitter.mov(location, operand);
  } else {
    // Avoid using xchg, it causes a bus lock.
    emitter.mov(dst, location);
    emitter.mov(location, value);
  }

  // restore operand unless operand == dst.
  if (operand != dst) {
    emitter.mov(operand, ctx.call_stack_slot(0));
  }

  return absl::OkStatus();
}

absl::Status GenerateCode(const ConstantAttributeInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  emitter.mov(
      dst,
      asmjit::imm(instr.LookupAttribute(ClassManager::Instance()).value()));
  return absl::OkStatus();
}

absl::Status GenerateCode(const DeoptimizedAsynchronouslyInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));
  emitter.mov(dst, asmjit::imm(CodeObject::DeoptimizedPointer(
                       ctx.s6_code_object_storage())));
  emitter.movzx(dst.r32(), x86::byte_ptr(dst));
  return absl::OkStatus();
}

absl::Status GenerateCode(const CallVectorcallInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  S6_ASSIGN_OR_RETURN(x86::Gp dst, ctx.Destination(instr));

  // The ABI is (self, args, nargs, kwnames).
  // The register allocator has put all arguments in ctx.call_stack_slot(0)+.
  // The names argument is already in RCX if it exists.
  // The self argument is already in RDI.

  // The only thing we need to do is set `nargs` in RSI,put a pointer to
  // the stack in RDX, and zero out RCX if there is no names().
  emitter.mov(AbiLocation<2>(), asmjit::imm(instr.call_arguments().size()));
  emitter.mov(AbiLocation<1>(), x86::rsp);
  if (!instr.names()) {
    emitter.xor_(AbiLocation<3>(), AbiLocation<3>());
  } else {
    // Otherwise, adjust the argument count to discount the number of arguments
    // passed by name.
    // We inline PyTuple_Size.
    emitter.sub(
        AbiLocation<2>(),
        x86::qword_ptr(AbiLocation<3>(), offsetof(PyVarObject, ob_size)));
  }

  emitter.call(ctx.OperandInRegister(instr.callee()));

  if (dst != x86::rax) {
    emitter.mov(dst, x86::rax);
  }
  return absl::OkStatus();
}

absl::Status GenerateCode(const SetObjectClassInst& instr,
                          x86::Emitter& emitter, CodeGenerationContext& ctx) {
  JitStub<PyDictObject> dict(x86::ptr(ctx.OperandInRegister(instr.dict())));
  S6_RET_CHECK(dict.Reg() != ctx.scratch_reg()) << "Need a scratch reg!";

  emitter.mov(ctx.scratch_reg(), asmjit::imm(instr.class_id()));
  // The class ID is stored in the uppermost 20 bits of ma_version_tag.
  emitter.shl(ctx.scratch_reg(), asmjit::imm(44));
  dict.ma_version_tag().Store(ctx.scratch_reg(), emitter);
  return absl::OkStatus();
}

// End instruction code generation
////////////////////////////////////////////////////////////////////////////////

// Functor struct to give to ForAllInstructionKinds.
struct InstrGenerator {
  template <typename InstrType>
  static std::optional<absl::Status> Visit(const Instruction& inst,
                                           x86::Emitter& emitter,
                                           CodeGenerationContext& ctx) {
    if (inst.kind() != InstrType::kKind) return {};
    return GenerateCode(cast<InstrType>(inst), emitter, ctx);
  }

  static absl::Status Default(const Instruction& inst, x86::Emitter& emitter,
                              CodeGenerationContext& ctx) {
    return absl::InvalidArgumentError(
        "unknown instruction kind in instruction printer!");
  }
};

// Removes unconditional jmps that could be converted to fallthroughs.
class TrivialJumpRemoval : public asmjit::Pass {
 public:
  TrivialJumpRemoval() : Pass("trivial-jump-removal") {}

  asmjit::Error run(asmjit::Zone* zone,
                    asmjit::Logger* logger) noexcept override {
    for (auto node = cb()->firstNode(); node != nullptr; node = node->next()) {
      if (!node->isInst()) continue;
      asmjit::InstNode* inst = node->as<asmjit::InstNode>();
      if (inst->id() != x86::Inst::kIdJmp) continue;

      if (!inst->hasLabelOp()) continue;
      int op_index = inst->indexOfLabelOp();
      asmjit::Operand& label_op = inst->operands()[op_index];
      if (!label_op.isLabel()) continue;
      asmjit::Label target = label_op.as<asmjit::Label>();

      if (!node->next() || !node->next()->isLabel()) continue;
      asmjit::LabelNode* label_node = node->next()->as<asmjit::LabelNode>();
      if (label_node->label() == target) {
        this->_cb->removeNode(node);
        node = label_node;
      }
    }
    return asmjit::ErrorCode::kErrorOk;
  }
};

// Returns the set of all CompareInsts and OverflowedInsts that feed a single
// branch. These can be folded into the branch.
absl::flat_hash_set<const Instruction*> FindOptimizableConditionInsts(
    Function& f, const RegisterAllocation& ra) {
  auto uses = ComputeUses(f);

  absl::flat_hash_set<const Instruction*> optimizable;
  for (const auto& [value, use_list] : uses) {
    const CompareInst* ci = dyn_cast<CompareInst>(value);
    const Instruction* opt_instr = nullptr;
    if (ci && ci->type() == NumericInst::kInt64) {
      // An integer compare with a single use, immediately followed by a branch
      // that uses it, can be folded.
      opt_instr = ci;
    } else if (const OverflowedInst* oi = dyn_cast<OverflowedInst>(value)) {
      // An overflow check with a single use, immediately followed by a branch
      // that uses it, can be folded.
      opt_instr = oi;
    } else if (const FloatZeroInst* oi = dyn_cast<FloatZeroInst>(value)) {
      // A zero check with a single use, immediately followed by a branch
      // that uses it, can be folded.
      opt_instr = oi;
    }
    if (!opt_instr || use_list.size() != 1) continue;

    const Instruction* next_instr = &*std::next(opt_instr->GetIterator());

    if (!ra.inst_copies(next_instr).empty()) {
      // The next instruction has prior copies. These could clobber the inputs
      // of the compare.
      continue;
    }

    if (const BrInst* br = dyn_cast<BrInst>(next_instr);
        br && br->condition() == opt_instr) {
      optimizable.insert(opt_instr);
    }

    if (const DeoptimizeIfInst* di = dyn_cast<DeoptimizeIfInst>(next_instr);
        di && di->condition() == opt_instr) {
      optimizable.insert(opt_instr);
    }
    if (const DeoptimizeIfSafepointInst* dsi =
            dyn_cast<DeoptimizeIfSafepointInst>(next_instr);
        dsi && dsi->condition() == opt_instr) {
      optimizable.insert(opt_instr);
    }
  }
  return optimizable;
}

std::vector<x86::Gp> OnlyCalleeSavedRegisters(const std::vector<x86::Gp> regs) {
  std::vector<x86::Gp> r;
  asmjit::CallConv cc;
  cc.init(asmjit::CallConv::kIdX86SysV64);
  uint32_t preserved_regmask = cc.preservedRegs(x86::Gp::kGroupGp);
  for (const x86::Gp& reg : regs) {
    if (preserved_regmask & asmjit::Support::bitMask(reg.id()))
      r.push_back(reg);
  }
  return r;
}
}  // namespace

// Generates machine code for `f`.
absl::StatusOr<std::unique_ptr<CodeObject>> GenerateCode(
    Function&& f, const RegisterAllocation& ra,
    absl::Span<const BytecodeInstruction> program, JitAllocator& allocator,
    PyCodeObject* code_object, Metadata* metadata,
    const CodeGeneratorOptions& options) {
  S6_VLOG_LINES(5, ra.ToString(f));
  asmjit::CodeHolder code;
  code.init(asmjit::CodeInfo(asmjit::ArchInfo::kIdX64));
  code.addEmitterOptions(asmjit::BaseEmitter::kOptionStrictValidation);

  // The final CodeObject location. We bake this in to the code.
  // The CodeObject if not constructed at this point.
  static_assert(alignof(CodeObject) <= alignof(std::max_align_t));
  void* code_object_storage = ::operator new(sizeof(CodeObject));
  absl::Cleanup code_object_storage_cleanup = [&]() {
    ::operator delete(code_object_storage);
  };

  auto moved_f = std::make_shared<Function>(std::move(f));

  // Set a fake signature, we don't use it.
  asmjit::FuncSignatureBuilder signature;
  signature.setRetT<uintptr_t>();
  signature.addArgT<uintptr_t>();
  signature.addArgT<uintptr_t>();
  signature.addArgT<uintptr_t>();

  asmjit::FuncDetail func;
  RETURN_IF_ASMJIT_ERROR(func.init(signature));

  x86::Builder builder(&code);
  x86::Emitter& emitter = *builder.as<x86::Emitter>();
  CodeGenerationContext ctx(emitter, code_object, code_object_storage, ra,
                            allocator, ra.GetNumFrameSlots(),
                            FindOptimizableConditionInsts(*moved_f, ra),
                            *moved_f);
  auto prolog_cursor = builder.cursor();

  std::vector<std::pair<std::string, asmjit::Label>> debug_labels;
  std::vector<std::pair<const Instruction*, asmjit::Label>> instruction_labels;
  ValueNumbering vn = ComputeValueNumbering(*moved_f);
  for (const Block& b : *moved_f) {
    if (b.deoptimized()) continue;
    ctx.BindBlock(&b);
    for (const Instruction& inst : b) {
      asmjit::Label l = emitter.newLabel();
      emitter.bind(l);
      S6_ASSIGN_OR_RETURN(std::string s, Format(inst, vn));
      debug_labels.emplace_back(s, l);
      instruction_labels.emplace_back(&inst, l);

      ctx.SetCurrentInstruction(&inst);
      S6_RETURN_IF_ERROR(GenerateCopies(inst, emitter, ctx));

      if (isa<BytecodeBeginInst>(inst)) {
        continue;
      }
      absl::Status status =
          ForAllInstructionKinds<InstrGenerator, const Instruction&,
                                 x86::Emitter&, CodeGenerationContext&>(
              inst, static_cast<x86::Emitter&>(emitter), ctx);
      S6_RETURN_IF_ERROR(status);
    }
  }

  if (options.add_passes) {
    options.add_passes(builder);
  }
  builder.addPassT<TrivialJumpRemoval>();
  builder.runPasses();

  ctx.FinalizeStackFrameLayout(
      OnlyCalleeSavedRegisters(ra.ComputeUsedRegisters()),
      ra.GetNumCallStackSlots());
  S6_RETURN_IF_ERROR(GenerateCleanup(ctx));

  S6_RETURN_IF_ERROR(GenerateEpilog(ctx));

  // Insert deoptimization trampolines.
  for (auto [inst, label] : ctx.deopt_labels()) {
    S6_ASSIGN_OR_RETURN(std::string s, Format(*inst, vn));
    debug_labels.emplace_back(absl::StrCat("deopt for: ", s), label);

    emitter.bind(label);
    emitter.call(asmjit::imm(Deoptimize));
    emitter.jmp(ctx.cleanup_point());
    instruction_labels.emplace_back(inst, label);
  }

  asmjit::Label constant_pool_label = emitter.newLabel();
  emitter.bind(constant_pool_label);
  instruction_labels.emplace_back(nullptr, constant_pool_label);

  builder.setCursor(prolog_cursor);
  S6_RETURN_IF_ERROR(GenerateProlog(ctx));
  asmjit::Label pyframe_entry_label = ctx.pyframe_entry_point();
  asmjit::Label fast_entry_label = ctx.fast_entry_point();

  RETURN_IF_ASMJIT_ERROR(builder.finalize());
  RETURN_IF_ASMJIT_ERROR(code.flatten());
  int64_t allocated_size = code.codeSize();
  void* ptr = allocator.Alloc(allocated_size, moved_f->name());
  S6_RET_CHECK(ptr);
  RETURN_IF_ASMJIT_ERROR(code.relocateToBase(reinterpret_cast<uint64_t>(ptr)));
  RETURN_IF_ASMJIT_ERROR(code.copyFlattenedData(ptr, allocated_size));

  absl::flat_hash_map<void*, DebugAnnotation> debug_annotations;
  std::vector<std::pair<void*, std::string>> debug_annotation_strs;

  auto add_annotation = [&](DebugAnnotation annotation, asmjit::Label label) {
    uint64_t iptr = reinterpret_cast<uint64_t>(ptr) + code.labelOffset(label);
    debug_annotations[reinterpret_cast<void*>(iptr)] = annotation;
    debug_annotation_strs.emplace_back(reinterpret_cast<void*>(iptr),
                                       annotation.ToString());
  };

  for (const auto& [s, label] : debug_labels) {
    add_annotation(DebugAnnotation(s), label);
  }
  add_annotation(DebugAnnotation("<data>", /*is_code=*/false),
                 constant_pool_label);

  // Give all generated symbols a unique ID.
  static int64_t unique_id = 0;
  std::string symbol_name = absl::StrCat(moved_f->name(), ".", unique_id++);
  allocator.RegisterSymbol(ptr, symbol_name, debug_annotation_strs);

  auto pyframe_entry = reinterpret_cast<CodeObject::PyFrameAbiFunctionType>(
      reinterpret_cast<uint64_t>(ptr) + code.labelOffset(pyframe_entry_label));
  CodeObject::FastAbiFunctionType fast_entry = nullptr;
  if (fast_entry_label.isValid()) {
    fast_entry = reinterpret_cast<CodeObject::FastAbiFunctionType>(
        reinterpret_cast<uint64_t>(ptr) + code.labelOffset(fast_entry_label));
  }

  DeoptimizationMap deopt_map = DeoptimizationMap(SlotIndexes(*moved_f));

  for (auto it = instruction_labels.begin();
       std::next(it) != instruction_labels.end(); ++it) {
    ProgramAddress begin_addr =
        reinterpret_cast<uint64_t>(ptr) + code.labelOffset(it->second);
    // Note that we add a nullptr entry at the end of instruction_labels to
    // support this query.
    ProgramAddress end_addr = reinterpret_cast<uint64_t>(ptr) +
                              code.labelOffset(std::next(it)->second);
    if (begin_addr == end_addr)
      // Don't insert empty ranges.
      continue;
    deopt_map.AddInstructionAddress(it->first, begin_addr, end_addr);
  }
  ra.PopulateDeoptimizationMap(deopt_map);

  S6_CHECK(ptr);
  auto ret_code_object =
      std::unique_ptr<CodeObject>(new (code_object_storage) CodeObject(
          pyframe_entry, fast_entry, ptr, code.codeSize(), moved_f->name(),
          program, std::move(debug_annotations), &allocator, moved_f,
          std::move(deopt_map), ctx.stack_frame_layout(), metadata));
  code_object_storage = nullptr;  // Otherwise there will be a double free.
  S6_CHECK_OK(DumpIfEnabled(*ret_code_object, symbol_name, code_object, ra));
  return ret_code_object;
}

}  // namespace deepmind::s6
