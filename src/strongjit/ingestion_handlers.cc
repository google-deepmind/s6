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

#include "strongjit/ingestion_handlers.h"

#include <Python.h>
#include <opcode.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "code_object.h"
#include "core_util.h"
#include "interpreter.h"
#include "strongjit/builder.h"
#include "strongjit/callees.h"
#include "strongjit/formatter.h"
#include "strongjit/ingestion.h"
#include "strongjit/instructions.h"
#include "strongjit/util.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

////////////////////////////////////////////////////////////////////////////////
// NO-OP

// No-op analysis function.
void AnalyzeNothing(const BytecodeInstruction& inst, Builder b,
                    AnalysisContext* ctx) {}

// No-op translation function.
void TranslateNothing(const BytecodeInstruction& inst, Builder b,
                      TranslateContext* ctx) {}

////////////////////////////////////////////////////////////////////////////////
// LOAD_CONST

void AnalyzeLoadConst(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadConst(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* value = b.FrameVariable(FrameVariableInst::FrameVariableKind::kConsts,
                                 inst.argument());
  b.IncrefNotNull(value);
  ctx->Push(value);
}

////////////////////////////////////////////////////////////////////////////////
// JUMP_ABSOLUTE, JUMP_FORWARD

void AnalyzeJumpAbsolute(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  PcValue target = PcValue::FromOffset(inst.argument());
  b.Jmp(ctx->BlockAt(target));
  ctx->MayJumpTo(target).DoesNotFallthrough();
}

void AnalyzeJumpForward(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  PcValue target = inst.pc_value().AddOffset(inst.argument()).Next();
  b.Jmp(ctx->BlockAt(target));
  ctx->MayJumpTo(target).DoesNotFallthrough();
}

////////////////////////////////////////////////////////////////////////////////
// JUMP_IF_FALSE_OR_POP, JUMP_IF_TRUE_OR_POP, POP_JUMP_IF_TRUE,POP_JUMP_IF_FALSE
//
// These four opcodes are slight variants on each other, so are implemented
// by one set of functions. The template arguments are:
//
// IfTrue: Take the jump if TOS evaluates to True. Otherwise take the jump if
//   TOS evaluates to False.
// PopIfTaken: Pop (decref) TOS if the jump is taken.
//
// Note that all cases pop (decref) TOS if the jump is NOT taken.

template <bool IfTrue, bool PopIfTaken>
void AnalyzeJumpIf(const BytecodeInstruction& inst, Builder b,
                   AnalysisContext* ctx) {
  PcValue target = PcValue::FromOffset(inst.argument());
  PcValue fallthrough = inst.pc_value().Next();

  // If we pop on a taken branch, we pop always (because we always pop a
  // non-taken branch).
  if (PopIfTaken) ctx->Pop();

  BrInst* br = b.Br(nullptr, ctx->BlockAt(target), ctx->BlockAt(fallthrough));
  // Fill in the condition later.
  ctx->Bind(br->mutable_condition());
  // We may jump to target_offset now with the current stack height.
  ctx->MayJumpTo(target);

  // Now that we've possibly jumped, pop TOS if !PopIfTaken such that the
  // fallthrough successor has a different stack height to the target successor.
  if (!PopIfTaken) ctx->Pop();
}

template <bool IfTrue, bool PopIfTaken>
void TranslateJumpIf(const BytecodeInstruction& inst, Builder b,
                     TranslateContext* ctx) {
  PcValue target = PcValue::FromOffset(inst.argument());
  PcValue fallthrough = inst.pc_value().Next();
  Block* target_block = ctx->BlockAt(target);
  Block* fallthrough_block = ctx->BlockAt(fallthrough);

  // Note that we do not pop from the stack; the stack height on output from
  // this function can be different for each successor. As the Analyze function
  // sets the stack entry height per successor, it makes no difference here
  // whether we pop or not.
  Value* v = ctx->Peek(1);

  ClassDistributionSummary type_feedback = ctx->GetTypeFeedback();
  if (type_feedback.MonomorphicType() == &PyBool_Type) {
    // Super easy case. It's either true or false.
    UnboxInst* unbox_op = ctx->UnboxOrDeoptimize(b, UnboxableType::kPyBool, v);
    b.Conditional(unbox_op, [&](Builder b) {
      if (IfTrue) {
        if (PopIfTaken) b.DecrefNotNull(v);
        b.Jmp(target_block);
      } else {
        b.DecrefNotNull(v);
        b.Jmp(fallthrough_block);
      }
      return Builder::DoesNotReturn();
    });

    BrInst* temp_br = cast<BrInst>(b.block()->GetTerminator());
    temp_br->true_successor()->RemovePredecessor(b.block());
    temp_br->false_successor()->RemovePredecessor(b.block());
    b.block()->GetTerminator()->erase();
    b = Builder(b.block(), b.block()->end());
    if (!IfTrue) {
      if (PopIfTaken) b.DecrefNotNull(v);
      b.Jmp(target_block);
    } else {
      b.DecrefNotNull(v);
      b.Jmp(fallthrough_block);
    }
    return;
  }

  // The easy cases are when `v` is exactly Py_True or Py_False, so handle those
  // first.
  b.Conditional(b.IsEqual(v, b.Constant(Py_True)), [&](Builder b) {
    if (IfTrue) {
      if (PopIfTaken) b.DecrefNotNull(v);
      b.Jmp(target_block);
    } else {
      b.DecrefNotNull(v);
      b.Jmp(fallthrough_block);
    }
    return Builder::DoesNotReturn();
  });

  b.Conditional(b.IsEqual(v, b.Constant(Py_False)), [&](Builder b) {
    if (!IfTrue) {
      if (PopIfTaken) b.DecrefNotNull(v);
      b.Jmp(target_block);
    } else {
      b.DecrefNotNull(v);
      b.Jmp(fallthrough_block);
    }
    return Builder::DoesNotReturn();
  });

  // Otherwise we must call PyObject_IsTrue.
  Value* ret = b.Sext(b.Call(Callee::kPyObject_IsTrue, {v}));
  b.Conditional(b.IsEqual(ret, b.Int64(1)), [&](Builder b) {
    // Comparison was True.
    if (IfTrue) {
      if (PopIfTaken) b.DecrefNotNull(v);
      b.Jmp(target_block);
    } else {
      b.DecrefNotNull(v);
      b.Jmp(fallthrough_block);
    }
    return Builder::DoesNotReturn();
  });
  b.Conditional(b.IsZero(ret), [&](Builder b) {
    // Comparison was False.
    if (!IfTrue) {
      if (PopIfTaken) b.DecrefNotNull(v);
      b.Jmp(target_block);
    } else {
      b.DecrefNotNull(v);
      b.Jmp(fallthrough_block);
    }
    return Builder::DoesNotReturn();
  });

  // Otherwise we had an exception. Remove the BrInst we temporarily placed
  // during the analyze function and replace it with an except.
  // Note that `v` is still on the value stack, so is decreffed by ctx->Except.
  ctx->Except(b);

  BrInst* temp_br = cast<BrInst>(b.block()->GetTerminator());
  temp_br->true_successor()->RemovePredecessor(b.block());
  temp_br->false_successor()->RemovePredecessor(b.block());
  b.block()->GetTerminator()->erase();
}

////////////////////////////////////////////////////////////////////////////////
// POP_TOP, DUP_TOP, DUP_TOP_TWO, ROT_TWO, ROT_THREE

void AnalyzePopTop(const BytecodeInstruction& inst, Builder b,
                   AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslatePopTop(const BytecodeInstruction& inst, Builder b,
                     TranslateContext* ctx) {
  b.DecrefOrNull(ctx->Pop());
}

void AnalyzeDupTop(const BytecodeInstruction& inst, Builder b,
                   AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateDupTop(const BytecodeInstruction& inst, Builder b,
                     TranslateContext* ctx) {
  Value* v = ctx->Peek(1);
  b.IncrefOrNull(v);
  ctx->Push(v);
}

void AnalyzeDupTopTwo(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Push().Push();
}

void TranslateDupTopTwo(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* x = ctx->Peek(2);
  Value* y = ctx->Peek(1);
  b.IncrefOrNull(x);
  b.IncrefOrNull(y);
  ctx->Push(x);
  ctx->Push(y);
}

void AnalyzeRotTwo(const BytecodeInstruction& inst, Builder b,
                   AnalysisContext* ctx) {
  ctx->Pop().Pop().Push().Push();
}

void TranslateRotTwo(const BytecodeInstruction& inst, Builder b,
                     TranslateContext* ctx) {
  Value* x = ctx->Pop();
  Value* y = ctx->Pop();
  ctx->Push(x);
  ctx->Push(y);
}

void AnalyzeRotThree(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Pop().Pop().Pop().Push().Push().Push();
}

void TranslateRotThree(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Value* x = ctx->Pop();
  Value* y = ctx->Pop();
  Value* z = ctx->Pop();
  ctx->Push(x);
  ctx->Push(z);
  ctx->Push(y);
}

////////////////////////////////////////////////////////////////////////////////
// RETURN_VALUE

void AnalyzeReturnValue(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  // We pop before returning.
  ctx->Pop();

  if (auto next = ctx->UnwindHandlerStack(Why::kReturn)) {
    S6_CHECK_EQ(next->kind(), TryHandler::kFinally);
    PcValue target = next->pc_value();
    Value* zero = b.Zero();
    Value* discriminator = b.Int64(WhyToDiscriminator(Why::kReturn));
    JmpInst* jmp = b.Jmp(
        ctx->BlockAt(target),
        {zero, zero, zero, zero, /* to be replaced */ zero, discriminator});
    ctx->Bind(&jmp->mutable_arguments()[4]);
    ctx->MayJumpToFinally(target, Why::kReturn).DoesNotFallthrough();
  } else {
    ReturnInst* ri = b.Return(nullptr);
    ctx->Bind(ri->mutable_returned_value()).DoesNotFallthrough();
  }
}

void TranslateReturnValue(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* retval = ctx->Pop();
  auto next = ctx->UnwindHandlerStack(b, Why::kReturn);
  if (next) {
    S6_CHECK(next->kind() == TryHandler::kFinally);
    ctx->Bind(retval);
  } else {
    ctx->Bind(retval);
    ctx->DecrefFastLocals(b);
  }
}

////////////////////////////////////////////////////////////////////////////////
// BREAK_LOOP

void AnalyzeBreakLoop(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  absl::optional<TryHandler> next = ctx->UnwindHandlerStack(Why::kBreak);
  S6_CHECK(next);
  PcValue target = next->pc_value();
  if (next->kind() == TryHandler::kFinally) {
    Value* zero = b.Zero();
    Value* discriminator = b.Int64(WhyToDiscriminator(Why::kBreak));
    b.Jmp(ctx->BlockAt(target), {zero, zero, zero, zero, zero, discriminator});
    ctx->MayJumpToFinally(target, Why::kBreak).DoesNotFallthrough();
  } else {
    S6_CHECK(next->kind() == TryHandler::kLoop);
    b.Jmp(ctx->BlockAt(target));
    ctx->MayJumpTo(target).DoesNotFallthrough();
  }
}

void TranslateBreakLoop(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  auto next = ctx->UnwindHandlerStack(b, Why::kBreak);
  S6_CHECK(next);
}

////////////////////////////////////////////////////////////////////////////////
// CONTINUE_LOOP

void AnalyzeContinueLoop(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  absl::optional<TryHandler> next = ctx->UnwindHandlerStack(Why::kContinue);
  S6_CHECK(next);
  PcValue target = PcValue::FromOffset(inst.argument());
  if (next->kind() == TryHandler::kFinally) {
    ctx->SetContinueTarget(*next, target);

    // Instead of jumping to the loop start, jump to the finally handler.
    target = next->pc_value();
    Value* zero = b.Zero();
    Value* discriminator = b.Int64(WhyToDiscriminator(Why::kContinue));
    b.Jmp(ctx->BlockAt(target), {zero, zero, zero, zero, zero, discriminator});
    ctx->MayJumpToFinally(target, Why::kContinue).DoesNotFallthrough();
  } else {
    S6_CHECK(next->kind() == TryHandler::kLoop);
    b.Jmp(ctx->BlockAt(target));
    ctx->MayJumpTo(target).DoesNotFallthrough();
  }
}

void TranslateContinueLoop(const BytecodeInstruction& inst, Builder b,
                           TranslateContext* ctx) {
  auto next = ctx->UnwindHandlerStack(b, Why::kContinue);
  S6_CHECK(next);
}

////////////////////////////////////////////////////////////////////////////////
// Unary ops

void AnalyzeUnaryOp(const BytecodeInstruction& inst, Builder b,
                    AnalysisContext* ctx) {
  ctx->Pop().Push();
}

void TranslateUnaryOp(const BytecodeInstruction& inst, Builder b,
                      TranslateContext* ctx) {
  Callee callee;
  switch (inst.opcode()) {
    case UNARY_POSITIVE:
      callee = Callee::kPyNumber_Positive;
      break;
    case UNARY_NEGATIVE:
      callee = Callee::kPyNumber_Negative;
      break;
    case UNARY_INVERT:
      callee = Callee::kPyNumber_Invert;
      break;
    default:
      S6_LOG(FATAL) << "Unhandled unary opcode!";
  }
  Value* v = ctx->Pop();
  Value* ret = b.Call(callee, {v});
  b.DecrefNotNull(v);
  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

void TranslateUnaryNot(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Value* v = ctx->Pop();
  Value* ret = b.Sext(b.Call(Callee::kPyObject_IsTrue, {v}));
  b.DecrefNotNull(v);
  ctx->ExceptIf(b, b.IsNegative(ret));

  Value* obj = b.Box(UnboxableType::kPyBool, b.IsZero(ret));
  ctx->Push(obj);
}

////////////////////////////////////////////////////////////////////////////////
// Binary ops

void AnalyzeBinaryOp(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Pop().Pop().Push();
}

void TranslateBinaryOp(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Callee callee;
  switch (inst.opcode()) {
    case BINARY_MATRIX_MULTIPLY:
      callee = Callee::kPyNumber_MatrixMultiply;
      break;
    case INPLACE_MATRIX_MULTIPLY:
      callee = Callee::kPyNumber_InPlaceMatrixMultiply;
      break;
    case BINARY_MULTIPLY:
      callee = Callee::kPyNumber_Multiply;
      break;
    case BINARY_MODULO:
      callee = Callee::kPyNumber_Remainder;
      break;
    case BINARY_ADD:
      callee = Callee::kPyNumber_Add;
      break;
    case BINARY_SUBTRACT:
      callee = Callee::kPyNumber_Subtract;
      break;
    case BINARY_FLOOR_DIVIDE:
      callee = Callee::kPyNumber_FloorDivide;
      break;
    case BINARY_TRUE_DIVIDE:
      callee = Callee::kPyNumber_TrueDivide;
      break;
    case INPLACE_FLOOR_DIVIDE:
      callee = Callee::kPyNumber_InPlaceFloorDivide;
      break;
    case INPLACE_TRUE_DIVIDE:
      callee = Callee::kPyNumber_InPlaceTrueDivide;
      break;
    case INPLACE_ADD:
      callee = Callee::kPyNumber_InPlaceAdd;
      break;
    case INPLACE_SUBTRACT:
      callee = Callee::kPyNumber_InPlaceSubtract;
      break;
    case INPLACE_MULTIPLY:
      callee = Callee::kPyNumber_InPlaceMultiply;
      break;
    case INPLACE_MODULO:
      callee = Callee::kPyNumber_InPlaceRemainder;
      break;
    case BINARY_LSHIFT:
      callee = Callee::kPyNumber_Lshift;
      break;
    case BINARY_RSHIFT:
      callee = Callee::kPyNumber_Rshift;
      break;
    case BINARY_AND:
      callee = Callee::kPyNumber_And;
      break;
    case BINARY_XOR:
      callee = Callee::kPyNumber_Xor;
      break;
    case BINARY_OR:
      callee = Callee::kPyNumber_Or;
      break;
    case INPLACE_LSHIFT:
      callee = Callee::kPyNumber_InPlaceLshift;
      break;
    case INPLACE_RSHIFT:
      callee = Callee::kPyNumber_InPlaceRshift;
      break;
    case INPLACE_AND:
      callee = Callee::kPyNumber_InPlaceAnd;
      break;
    case INPLACE_XOR:
      callee = Callee::kPyNumber_InPlaceXor;
      break;
    case INPLACE_OR:
      callee = Callee::kPyNumber_InPlaceOr;
      break;
    case BINARY_SUBSCR:
      callee = Callee::kPyObject_GetItem;
      break;
    default:
      S6_LOG(FATAL) << "Unhandled binary opcode!";
  }
  Value* rhs = ctx->Pop();
  Value* lhs = ctx->Pop();
  Value* ret = b.Call(callee, {lhs, rhs});
  b.DecrefNotNull({rhs, lhs});
  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

void TranslatePowerOp(const BytecodeInstruction& inst, Builder b,
                      TranslateContext* ctx) {
  Callee callee;
  switch (inst.opcode()) {
    case BINARY_POWER:
      callee = Callee::kPyNumber_Power;
      break;
    case INPLACE_POWER:
      callee = Callee::kPyNumber_InPlacePower;
      break;
    default:
      S6_LOG(FATAL) << "Unhandled power opcode!";
  }
  Value* rhs = ctx->Pop();
  Value* lhs = ctx->Pop();
  Value* ret = b.Call(callee, {lhs, rhs, b.Constant(Py_None)});
  b.DecrefNotNull({rhs, lhs});
  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_METHOD / CALL_METHOD
//
// LOAD_METHOD is a LOAD_ATTR with two differences:
//  1) the compiler emits LOAD_METHOD when the attribute load only has one user,
//  a CALL_METHOD. 2) LOAD_METHOD will attempt to identify a method load. If it
//  finds one, it will push the underlying method function and `self` to the
//  stack.
//
// So on CALL_METHOD, the stack may have one of two forms:
//  NULL | callable | arg1 | ... | argN  : No method found.
//  method | self   | arg1 | ... | argN  : Method found.
//
// Strongjit will perform static analysis to determine method calls, so we don't
// ever need to represent both forms. In ingestion, we always lower to the "no
// method found" variant. We don't call _PyObject_GetMethod, like in the
// interpreter, because we *always* want to perform an attribute load and never
// elide it (note again that a later static optimization pass will remove this
// and replace with a call_attribute).

void AnalyzeLoadMethod(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Pop().Push().Push();
}

void TranslateLoadMethod(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* owner = ctx->Pop();
  Value* value = b.Call(Callee::kPyObject_GetAttr, {owner, name});
  b.DecrefNotNull(owner);
  ctx->ExceptIf(b, b.IsZero(value));
  ctx->Push(b.Zero());
  ctx->Push(value);
}

void AnalyzeCallMethod(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  // The stack contains `oparg` arguments followed by the function object and
  // nullptr.
  ctx->Drop(inst.argument() + 2);
  ctx->Push();
}

void TranslateCallMethod(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  std::vector<Value*> arguments;
  for (int64_t i = 0; i < inst.argument() + 1; ++i)
    arguments.push_back(ctx->Pop());
  // Reverse so that the function comes first and arguments follow.
  absl::c_reverse(arguments);

  // Pop nullptr.
  ctx->Pop();

  Value* ret = b.CallPython(arguments[0], absl::MakeSpan(arguments).subspan(1));
  // Note we don't decref any arguments; we give the callee stolen references
  // to its arguments. This is different from CPython's calling convention.

  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

////////////////////////////////////////////////////////////////////////////////
// CALL_FUNCTION

void AnalyzeCallFunction(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  // The stack contains `oparg` arguments followed by the function object.
  for (int64_t i = 0; i < inst.argument() + 1; ++i) {
    ctx->Pop();
  }
  // Clearly a result gets pushed.
  ctx->Push();
}

void TranslateCallFunction(const BytecodeInstruction& inst, Builder b,
                           TranslateContext* ctx) {
  std::vector<Value*> arguments;
  for (int64_t i = 0; i < inst.argument() + 1; ++i)
    arguments.push_back(ctx->Pop());
  // Reverse so that the function comes first and arguments follow.
  absl::c_reverse(arguments);

  Value* ret = b.CallPython(arguments[0], absl::MakeSpan(arguments).subspan(1));
  // Note we don't decref any arguments; we give the callee stolen references
  // to its arguments. This is different from CPython's calling convention.

  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

void AnalyzeCallFunctionKw(const BytecodeInstruction& inst, Builder b,
                           AnalysisContext* ctx) {
  // The stack contains `names` followed by `oparg` arguments followed by the
  // function object.
  ctx->Drop(inst.argument() + 2);
  // Clearly a result gets pushed.
  ctx->Push();
}

void TranslateCallFunctionKw(const BytecodeInstruction& inst, Builder b,
                             TranslateContext* ctx) {
  Value* names = ctx->Pop();
  std::vector<Value*> arguments;
  for (int64_t i = 0; i < inst.argument() + 1; ++i)
    arguments.push_back(ctx->Pop());
  // Reverse so that the function comes first and arguments follow.
  absl::c_reverse(arguments);

  Value* ret =
      b.CallPython(arguments[0], absl::MakeSpan(arguments).subspan(1), names);
  // Note we don't decref any arguments; we give the callee stolen references
  // to its arguments. This is different from CPython's calling convention.

  // We decref names because this one is not stolen by our CallPython.
  b.DecrefOrNull(names);

  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

void AnalyzeCallFunctionEx(const BytecodeInstruction& inst, Builder b,
                           AnalysisContext* ctx) {
  bool has_mapping = (inst.argument() & 1) != 0;
  if (has_mapping) ctx->Pop();
  ctx->Pop().Pop().Push();
}

void TranslateCallFunctionEx(const BytecodeInstruction& inst, Builder b,
                             TranslateContext* ctx) {
  bool has_mapping = (inst.argument() & 1) != 0;
  Value* mapping;
  if (has_mapping) {
    mapping = ctx->Pop();
  } else {
    mapping = b.Zero();
  }

  Value* positional = ctx->Pop();
  Value* callable = ctx->Pop();
  Value* ret = b.Call(Callee::kCallFunctionEx, {callable, positional, mapping});
  b.DecrefNotNull({callable, positional});
  if (has_mapping) b.DecrefNotNull(mapping);
  ctx->ExceptIf(b, b.IsZero(ret));
  ctx->Push(ret);
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_FAST

void AnalyzeLoadFast(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadFast(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Value* v = ctx->UseFastLocal(inst.argument());
  ctx->DeoptimizeIfNull(b, v, "Fastlocal load was nullptr (unbound local)");
  b.IncrefNotNull(v);
  ctx->Push(v);
}

////////////////////////////////////////////////////////////////////////////////
// STORE_FAST

void AnalyzeStoreFast(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Pop().DefFastLocal(inst.argument());
}

void TranslateStoreFast(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* value = ctx->Pop();
  Value* previous = ctx->UseFastLocal(inst.argument());
  ctx->DefFastLocal(inst.argument(), value);
  b.DecrefOrNull(previous);
}

////////////////////////////////////////////////////////////////////////////////
// DELETE_FAST

void AnalyzeDeleteFast(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->DefFastLocal(inst.argument());
}

void TranslateDeleteFast(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* previous = ctx->UseFastLocal(inst.argument());
  ctx->DeoptimizeIfNull(
      b, previous, "Fastlocal delete was nullptr (deletion of unbound local)");
  ctx->DefFastLocal(inst.argument(), b.Zero());
  b.DecrefNotNull(previous);
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_NAME

void AnalyzeLoadName(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadName(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* locals =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kLocals, 0);

  // Locals must exist!
  ctx->ExceptConditional(b, b.IsZero(locals), [&](Builder& b) {
    b.Call(Callee::kPyErr_Format,
           {b.Constant(PyExc_SystemError),
            b.Constant("no locals while loading %R"), name});
    return Builder::DoesNotReturn();
  });

  // 1. Look in locals. Special-case if it is a dict.
  Value* locals_is_dict =
      b.IsEqual(b.GetType(locals), b.Constant(&PyDict_Type));
  Value* value =
      b.Conditional(
           locals_is_dict,
           [&](Builder b) {
             return Builder::ValueList{
                 b.Call(Callee::kPyDict_GetItem, {locals, name})};
           },
           [&](Builder b) {
             Value* value = b.Call(Callee::kPyObject_GetItem, {locals, name});
             // If the call failed with KeyError, clear the error and
             // try again below.
             b.Conditional(b.IsZero(value), [&](Builder b) {
               Value* exc_matches = b.Call(Callee::kPyErr_ExceptionMatches,
                                           {b.Constant(PyExc_KeyError)});
               ctx->ExceptIf(b, b.IsZero(exc_matches));
               // Clear KeyError.
               b.Call(Callee::kPyErr_Clear, {});
               return Builder::ValueList{};
             });
             return Builder::ValueList{value};
           })
          .front();

  // 2. Look in globals and builtins.
  value = b.Conditional(b.IsZero(value),
                        [&](Builder b) {
                          Value* v = b.LoadGlobal(inst.argument());
                          ctx->ExceptIf(b, b.IsZero(v));
                          return Builder::ValueList{v};
                        },
                        {value})
              .front();

  b.IncrefNotNull(value);
  ctx->Push(value);
}

////////////////////////////////////////////////////////////////////////////////
// STORE_NAME

void AnalyzeStoreName(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslateStoreName(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* value = ctx->Pop();
  Value* locals =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kLocals, 0);

  // Locals must exist!
  ctx->ExceptConditional(b, b.IsZero(locals), [&](Builder& b) {
    b.Call(Callee::kPyErr_Format,
           {b.Constant(PyExc_SystemError),
            b.Constant("no locals found when storing %R"), name});
    b.DecrefNotNull(value);
    return Builder::DoesNotReturn();
  });

  Value* locals_is_dict =
      b.IsEqual(b.GetType(locals), b.Constant(&PyDict_Type));
  Value* err =
      b.Conditional(
           locals_is_dict,
           [&](Builder b) {
             Value* res =
                 b.Call(Callee::kPyDict_SetItem, {locals, name, value});
             return Builder::ValueList{res};
           },
           [&](Builder b) {
             Value* res =
                 b.Call(Callee::kPyObject_SetItem, {locals, name, value});
             return Builder::ValueList{res};
           })
          .front();

  b.DecrefNotNull(value);
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_GLOBAL, STORE_GLOBAL

void AnalyzeLoadGlobal(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadGlobal(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* value = b.LoadGlobal(inst.argument());
  ctx->ExceptIf(b, b.IsZero(value));
  b.IncrefNotNull(value);
  ctx->Push(value);
}

void AnalyzeStoreGlobal(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslateStoreGlobal(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* value = ctx->Pop();
  Value* globals =
      b.FrameVariable(FrameVariableInst::FrameVariableKind::kGlobals, 0);

  Value* err = b.Call(Callee::kSetAttrForGlobalsDict, {globals, name, value});

  b.DecrefNotNull(value);
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_ATTR, STORE_ATTR, DELETE_ATTR

void AnalyzeLoadAttr(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Pop().Push();
}

void TranslateLoadAttr(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* owner = ctx->Pop();
  Value* value = b.Call(Callee::kPyObject_GetAttr, {owner, name});
  b.DecrefNotNull(owner);
  ctx->ExceptIf(b, b.IsZero(value));
  ctx->Push(value);
}

void AnalyzeStoreAttr(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Pop().Pop();
}

void TranslateStoreAttr(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* owner = ctx->Pop();
  Value* value = ctx->Pop();
  // Note the last argument is the inline cache.
  Value* err = b.Call(Callee::kPyObject_SetAttr, {owner, name, value});
  b.DecrefNotNull(owner);
  b.DecrefNotNull(value);
  ctx->ExceptIf(b, b.IsNotZero(err));
}

void AnalyzeDeleteAttr(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslateDeleteAttr(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* name = b.FrameVariable(FrameVariableInst::FrameVariableKind::kNames,
                                inst.argument());
  Value* owner = ctx->Pop();
  // Note the last argument is the inline cache.
  Value* err = b.Call(Callee::kPyObject_SetAttr, {owner, name, b.Zero()});
  b.DecrefNotNull(owner);
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// MAKE_FUNCTION

void AnalyzeMakeFunction(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  ctx->Pop().Pop();

  // Depending on oparg, there will be extra values on the stack that need
  // to be installed in the function object.
  int64_t oparg = inst.argument();
  if (oparg & 0x08) ctx->Pop();
  if (oparg & 0x04) ctx->Pop();
  if (oparg & 0x02) ctx->Pop();
  if (oparg & 0x01) ctx->Pop();

  ctx->Push();
}

void TranslateMakeFunction(const BytecodeInstruction& inst, Builder b,
                           TranslateContext* ctx) {
  Value* qualified_name = ctx->Pop();
  Value* code = ctx->Pop();
  Value* function = b.Call(
      Callee::kPyFunction_NewWithQualName,
      {code, b.FrameVariable(FrameVariableInst::FrameVariableKind::kGlobals, 0),
       qualified_name});
  b.DecrefNotNull({qualified_name, code});
  ctx->ExceptIf(b, b.IsZero(function));

  // Depending on oparg, there will be extra values on the stack that need
  // to be installed in the function object.
  // The ownership is donated to the function object and nothing is overwritten,
  // because the function object is freshly created.
  int64_t oparg = inst.argument();
  if (oparg & 0x08) {
    Value* closure = ctx->Pop();
    b.StoreDonate(closure, function, offsetof(PyFunctionObject, func_closure));
  }
  if (oparg & 0x04) {
    Value* annotations = ctx->Pop();
    b.StoreDonate(annotations, function,
                  offsetof(PyFunctionObject, func_annotations));
  }
  if (oparg & 0x02) {
    Value* keyword_defaults = ctx->Pop();
    b.StoreDonate(keyword_defaults, function,
                  offsetof(PyFunctionObject, func_kwdefaults));
  }
  if (oparg & 0x01) {
    Value* defaults = ctx->Pop();
    b.StoreDonate(defaults, function,
                  offsetof(PyFunctionObject, func_defaults));
  }

  ctx->Push(function);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_SLICE

void AnalyzeBuildSlice(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  int64_t arg = inst.argument();
  S6_CHECK(arg == 2 || arg == 3);
  if (arg == 3) ctx->Pop();
  ctx->Pop();
  ctx->Pop();
  ctx->Push();
}

void TranslateBuildSlice(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  int64_t arg = inst.argument();
  S6_CHECK(arg == 2 || arg == 3);

  Value* step = arg == 3 ? ctx->Pop() : b.Constant(nullptr);
  Value* stop = ctx->Pop();
  Value* start = ctx->Pop();
  Value* slice = b.Call(Callee::kPySlice_New, {start, stop, step});
  b.DecrefNotNull(start);
  b.DecrefNotNull(stop);
  if (arg == 3) b.DecrefNotNull(step);
  ctx->Push(slice);
  ctx->ExceptIf(b, b.IsZero(slice));
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_MAP

void AnalyzeBuildMap(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Drop(2 * inst.argument()).Push();
}

void TranslateBuildMap(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  int64_t oparg = inst.argument();
  Value* map = b.Call(Callee::k_PyDict_NewPresized, {b.Int64(oparg)});
  ctx->ExceptIf(b, b.IsZero(map));

  // Create a target to jump to in case of map building error.
  Block* end = b.Split();

  for (int i = oparg; i > 0; --i) {
    Value* key = ctx->Peek(2 * i);
    Value* value = ctx->Peek(2 * i - 1);
    Value* err = b.Call(Callee::kPyDict_SetItem, {map, key, value});
    // Jump to end of the map construction with the error code `err`.
    b.BrFallthrough(err, end, {err});
  }
  // Continue to the end of the map construction with no error.
  Value* final_err = b.EndSplit(end, {b.Zero()}).front();

  for (int i = 2 * oparg; i > 0; --i) b.DecrefNotNull(ctx->Peek(i));
  ctx->Drop(2 * oparg);

  ctx->ExceptConditional(b, final_err, [&](Builder& b) {
    b.DecrefNotNull(map);
    return Builder::DoesNotReturn{};
  });
  ctx->Push(map);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_CONST_KEY_MAP

void AnalyzeBuildConstKeyMap(const BytecodeInstruction& inst, Builder b,
                             AnalysisContext* ctx) {
  ctx->Drop(inst.argument() + 1).Push();
}

void TranslateBuildConstKeyMap(const BytecodeInstruction& inst, Builder b,
                               TranslateContext* ctx) {
  Value* keys = ctx->Pop();
  int64_t oparg = inst.argument();
  Value* invalid_tuple = b.ShortcircuitOr(
      [&](Builder b) {
        return b.IsNotEqual(b.GetType(keys), b.Constant(&PyTuple_Type));
      },
      [&](Builder b) { return b.IsNotEqual(b.GetSize(keys), b.Int64(oparg)); });
  ctx->ExceptConditional(b, invalid_tuple, [&](Builder& b) {
    b.Call(Callee::kPyErr_SetString,
           {b.Constant(PyExc_SystemError),
            b.Constant("bad BUILD_CONST_KEY_MAP keys argument")});
    return Builder::DoesNotReturn{};
  });
  Value* map = b.Call(Callee::k_PyDict_NewPresized, {b.Int64(oparg)});
  ctx->ExceptIf(b, b.IsZero(map));

  // Create a target to jump to in case of map building error.
  Block* end = b.Split();

  for (int i = oparg; i > 0; --i) {
    Value* key = b.TupleGetItem(keys, oparg - i);
    Value* value = ctx->Peek(i);
    Value* err = b.Call(Callee::kPyDict_SetItem, {map, key, value});
    // Jump to end of the map construction with the error code `err`.
    b.BrFallthrough(err, end, {err});
  }
  // Continue to the end of the map construction with no error.
  Value* final_err = b.EndSplit(end, {b.Zero()}).front();

  for (int i = oparg; i > 0; --i) b.DecrefNotNull(ctx->Peek(i));
  ctx->Drop(oparg);
  b.DecrefNotNull(keys);

  ctx->ExceptConditional(b, final_err, [&](Builder& b) {
    b.DecrefNotNull(map);
    return Builder::DoesNotReturn{};
  });
  ctx->Push(map);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_SET

void AnalyzeBuildSet(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  ctx->Drop(inst.argument()).Push();
}

void TranslateBuildSet(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  int64_t oparg = inst.argument();
  Value* set = b.Call(Callee::kPySet_New, {b.Constant(nullptr)});
  ctx->ExceptIf(b, b.IsZero(set));

  // Create a target to jump to in case of set building error.
  Block* end = b.Split();

  for (int i = oparg; i > 0; --i) {
    Value* item = ctx->Peek(i);
    Value* err = b.Call(Callee::kPySet_Add, {set, item});
    // Jump to end of the set construction with the error code `err`.
    b.BrFallthrough(err, end, {err});
  }
  // Continue to the end of the set construction with no error.
  Value* final_err = b.EndSplit(end, {b.Zero()}).front();

  for (int i = oparg; i > 0; --i) b.DecrefNotNull(ctx->Peek(i));
  ctx->Drop(oparg);

  ctx->ExceptConditional(b, final_err, [&](Builder& b) {
    b.DecrefNotNull(set);
    return Builder::DoesNotReturn{};
  });
  ctx->Push(set);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_TUPLE, BUILD_LIST

void AnalyzeBuildTupleOrList(const BytecodeInstruction& inst, Builder b,
                             AnalysisContext* ctx) {
  ctx->Drop(inst.argument()).Push();
}

void TranslateBuildTuple(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* tuple = b.Call(Callee::kPyTuple_New, {b.Int64(inst.argument())});
  // We assume the tuple creation can't fail, because S6 works under the
  // assumption of having enough memory to allocate.

  int64_t oparg = inst.argument();
  while (--oparg >= 0) b.TupleSetItem(ctx->Pop(), tuple, oparg);
  ctx->Push(tuple);
}

void TranslateBuildList(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* list = b.Call(Callee::kPyList_New, {b.Int64(inst.argument())});
  ctx->ExceptIf(b, b.IsZero(list));

  Value* ob_item = b.Load64(list, offsetof(PyListObject, ob_item));
  int64_t oparg = inst.argument();
  while (--oparg >= 0) {
    // It is okay to donate, because the object is freshly created.
    // We are not overwriting anything.
    b.StoreDonate(ctx->Pop(), ob_item, oparg * sizeof(PyObject*));
  }
  ctx->Push(list);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_TUPLE_UNPACK_WITH_CALL

void AnalyzeBuildTupleUnpackWithCall(const BytecodeInstruction& inst, Builder b,
                                     AnalysisContext* ctx) {
  ctx->Drop(inst.argument()).Push();
}

void TranslateBuildTupleUnpackWithCall(const BytecodeInstruction& inst,
                                       Builder b, TranslateContext* ctx) {
  std::vector<Value*> args(1, b.Int64(inst.argument()));
  args.push_back(ctx->Peek(inst.argument() + 1));
  for (int64_t i = inst.argument(); i > 0; --i) {
    args.push_back(ctx->Peek(i));
  }
  ctx->Drop(inst.argument());
  Value* list = b.Call(Callee::kBuildListUnpackVararg, args);
  b.DecrefNotNull(absl::MakeConstSpan(args).subspan(2));
  ctx->ExceptIf(b, b.IsZero(list));

  Value* tuple = b.Call(Callee::kPyList_AsTuple, {list});
  b.DecrefNotNull(list);
  ctx->ExceptIf(b, b.IsZero(tuple));

  ctx->Push(tuple);
}

////////////////////////////////////////////////////////////////////////////////
// UNPACK_SEQUENCE

void AnalyzeUnpackSequence(const BytecodeInstruction& inst, Builder b,
                           AnalysisContext* ctx) {
  ctx->Pop();
  for (int64_t i = 0; i < inst.argument(); ++i) {
    ctx->Push();
  }
}

void TranslateUnpackSequence(const BytecodeInstruction& inst, Builder b,
                             TranslateContext* ctx) {
  Value* sequence = ctx->Pop();
  int64_t sequence_length = inst.argument();

  // The full UNPACK_SEQUENCE code is very messy. Specialize for the most common
  // cases.
  ClassDistributionSummary type_feedback = ctx->GetTypeFeedback();
  if (type_feedback.MonomorphicType() == &PyTuple_Type ||
      type_feedback.MonomorphicType() == &PyList_Type) {
    bool is_list = type_feedback.MonomorphicType() == &PyList_Type;
    std::vector<Value*> values;
    values.reserve(sequence_length);

    std::string reason =
        absl::StrCat("Unpack sequence was specialized to type: ",
                     type_feedback.MonomorphicType()->tp_name);
    ctx->DeoptimizeIfNull(
        b, b.CheckClassId(sequence, type_feedback.MonomorphicClass()), reason);
    Value* size = b.GetSize(sequence);
    ctx->DeoptimizeIf(b, b.IsNotEqual(size, b.Int64(sequence_length)),
                      "Unpack sequence given incorrect sequence length");

    Value* items = sequence;
    if (is_list) {
      items = b.Load64(sequence, offsetof(PyListObject, ob_item));
    }
    int64_t i = sequence_length;
    while (--i >= 0) {
      Value* element;
      if (is_list) {
        element = b.Load64(items, i * sizeof(PyObject*));
      } else {
        element = b.TupleGetItem(items, i);
      }
      b.IncrefNotNull(element);
      values.push_back(element);
    }

    for (Value* value : values) {
      ctx->Push(value);
    }
    b.DecrefNotNull(sequence);
    return;
  }

  Value* is_sequence_tuple = b.ShortcircuitAnd(
      [&](Builder b) {
        return b.IsEqual(b.GetType(sequence), b.Constant(&PyTuple_Type));
      },
      [&](Builder b) {
        return b.IsEqual(b.GetSize(sequence), b.Int64(sequence_length));
      });

  auto if_tuple_block = [&](Builder b) {
    Builder::ValueList value_list{};
    int64_t i = sequence_length;
    while (--i >= 0) {
      Value* element = b.TupleGetItem(sequence, i);
      b.IncrefNotNull(element);
      value_list.push_back(element);
    }
    return value_list;
  };

  auto elif_list_block = [&](Builder b) {
    Builder::ValueList value_list{};
    int64_t i = sequence_length;
    Value* ob_item = b.Load64(sequence, offsetof(PyListObject, ob_item));
    while (--i >= 0) {
      Value* element = b.Load64(ob_item, i * sizeof(PyObject*));
      b.IncrefNotNull(element);
      value_list.push_back(element);
    }
    return value_list;
  };

  auto else_generic_slow_block = [&](Builder b) {
    Value* iterable = b.Call(Callee::kPyObject_GetIter, {sequence});
    ctx->ExceptConditional(b, b.IsZero(iterable), [&](Builder& b) {
      b.DecrefNotNull(sequence);
      return Builder::DoesNotReturn();
    });
    Builder::ValueList value_list;
    int64_t i = sequence_length;

    // Helper that decrements the retrieved elements, iterator, and sequence if
    // the generic path experiences and error and must bail out.
    auto decref_values = [&](Builder b) {
      for (Value* value : value_list) {
        b.DecrefNotNull(value);
      }
      b.DecrefNotNull(iterable);
      b.DecrefNotNull(sequence);
    };

    while (--i >= 0) {
      Value* next = b.Call(Callee::kPyIter_Next, {iterable});
      // Iteration may have stopped either due to an error or the
      // iterator has been exhausted. If `next` is nullptr with:
      //   - no error set, then raise ValueError.
      //   - an error set, propagate the existing error.
      ctx->ExceptConditional(b, b.IsZero(next), [&](Builder& b) {
        decref_values(b);
        // Check if there is an existing error.
        Value* thread_state =
            b.FrameVariable(FrameVariableInst::FrameVariableKind::kThreadState);
        Value* exception =
            b.Load64(thread_state, offsetof(PyThreadState, curexc_type));
        Value* error_thrown = b.IsNotEqual(exception, b.Zero());
        ctx->ExceptIf(b, error_thrown);
        // Raise ValueError.
        static constexpr const char* kValueErrorMessage =
            "not enough values to unpack (expected %d, got %d)";
        b.Call(Callee::kPyErr_Format,
               {b.Constant(PyExc_ValueError), b.Constant(kValueErrorMessage),
                b.Int64(sequence_length), b.Int64(sequence_length - i - 1)});
        return Builder::DoesNotReturn();
      });
      value_list.push_back(next);
    }

    // Ensure that the iterator is exhausted, else raise ValueError.
    Value* next = b.Call(Callee::kPyIter_Next, {iterable});
    ctx->ExceptConditional(b, b.IsNotZero(next), [&](Builder& b) {
      b.DecrefNotNull(next);
      decref_values(b);
      static constexpr const char* kValueErrorMessage =
          "too many values to unpack (expected %d)";
      b.Call(Callee::kPyErr_Format,
             {b.Constant(PyExc_ValueError), b.Constant(kValueErrorMessage),
              b.Int64(sequence_length)});
      return Builder::DoesNotReturn();
    });

    // But make sure there isn't an error thrown if it is exhausted.
    Value* thread_state =
        b.FrameVariable(FrameVariableInst::FrameVariableKind::kThreadState);
    Value* exception =
        b.Load64(thread_state, offsetof(PyThreadState, curexc_type));
    Value* error_thrown = b.IsNotEqual(exception, b.Zero());
    ctx->ExceptConditional(b, error_thrown, [&](Builder& b) {
      decref_values(b);
      return Builder::DoesNotReturn();
    });

    absl::c_reverse(value_list);
    b.DecrefNotNull(iterable);
    return value_list;
  };

  Builder::ValueList values =
      b.Conditional(is_sequence_tuple, if_tuple_block, [&](Builder b) {
        // Else can be a list, or take the slow generic path.
        Value* is_sequence_list = b.ShortcircuitAnd(
            [&](Builder b) {
              return b.IsEqual(b.GetType(sequence), b.Constant(&PyList_Type));
            },
            [&](Builder b) {
              return b.IsEqual(b.GetSize(sequence), b.Int64(sequence_length));
            });
        return b.Conditional(is_sequence_list, elif_list_block,
                             else_generic_slow_block);
      });

  for (Value* value : values) {
    ctx->Push(value);
  }
  b.DecrefNotNull(sequence);
}

////////////////////////////////////////////////////////////////////////////////
// LIST_APPEND

void AnalyzeListAppend(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslateListAppend(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* element = ctx->Pop();
  Value* list = ctx->Peek(inst.argument());
  Value* err = b.Call(Callee::kPyList_Append, {list, element});
  b.DecrefNotNull(element);
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// COMPARE_OP

void AnalyzeCompareOp(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Pop().Pop().Push();
}

void TranslateCompareOp(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  auto bool_result = [&](Value* condition) -> Value* {
    return b.Box(UnboxableType::kPyBool, condition);
  };

  Value* rhs = ctx->Pop();
  Value* lhs = ctx->Pop();
  Value* retval;

  switch (inst.argument()) {
    case PyCmp_IS:
      retval = bool_result(b.IsEqual(lhs, rhs));
      b.DecrefNotNull({lhs, rhs});
      break;

    case PyCmp_IS_NOT:
      retval = bool_result(b.IsNotEqual(lhs, rhs));
      b.DecrefNotNull({lhs, rhs});
      break;

    case PyCmp_IN: {
      Value* res = b.Sext(b.Call(Callee::kPySequence_Contains, {rhs, lhs}));
      b.DecrefNotNull({lhs, rhs});
      ctx->ExceptIf(b, b.IsNegative(res));
      retval = bool_result(b.IsNotZero(res));
      break;
    }

    case PyCmp_NOT_IN: {
      Value* res = b.Sext(b.Call(Callee::kPySequence_Contains, {rhs, lhs}));
      b.DecrefNotNull({lhs, rhs});
      ctx->ExceptIf(b, b.IsNegative(res));
      retval = bool_result(b.IsZero(res));
      break;
    }

    case PyCmp_EXC_MATCH: {
      Value* v = b.Call(Callee::kCheckedGivenExceptionMatches, {lhs, rhs});
      b.DecrefNotNull({lhs, rhs});
      ctx->ExceptIf(b, b.IsNegative(v));
      retval = bool_result(b.IsNotZero(v));
      break;
    }

    default:
      retval = b.Call(Callee::kPyObject_RichCompare,
                      {lhs, rhs, b.Int64(inst.argument())});
      b.DecrefNotNull({lhs, rhs});
      ctx->ExceptIf(b, b.IsZero(retval));
      break;
  }

  ctx->Push(retval);
}

////////////////////////////////////////////////////////////////////////////////
// LOAD_CLOSURE, LOAD_DEREF, STORE_DEREF, DELETE_DEREF

void AnalyzeLoadClosure(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadClosure(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* cell = b.Load64(b.FrameVariable(
      FrameVariableInst::FrameVariableKind::kFreeVars, inst.argument()));
  b.IncrefNotNull(cell);
  ctx->Push(cell);
}

void AnalyzeLoadDeref(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  ctx->Push();
}

void TranslateLoadDeref(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  Value* cell = b.Load64(b.FrameVariable(
      FrameVariableInst::FrameVariableKind::kFreeVars, inst.argument()));
  Value* value = b.Load64(cell, offsetof(PyCellObject, ob_ref));
  ctx->ExceptConditional(b, b.IsZero(value), [&](Builder& b) {
    b.Call(
        Callee::kFormatUnboundError,
        {b.FrameVariable(FrameVariableInst::FrameVariableKind::kCodeObject, 0),
         b.Int64(inst.argument()), /*is_local=*/b.Zero()});
    return Builder::DoesNotReturn();
  });
  b.IncrefNotNull(value);
  ctx->Push(value);
}

void AnalyzeStoreDeref(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Pop();
}

void TranslateStoreDeref(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  Value* value = ctx->Pop();
  Value* cell = b.Load64(b.FrameVariable(
      FrameVariableInst::FrameVariableKind::kFreeVars, inst.argument()));
  // It is okay to steal since we donate into the same slot immediately
  // afterwards.
  Value* previous = b.LoadSteal(cell, offsetof(PyCellObject, ob_ref));
  b.StoreDonate(value, cell, offsetof(PyCellObject, ob_ref));

  b.DecrefOrNull(previous);
}

void AnalyzeDeleteDeref(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {}

void TranslateDeleteDeref(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* cell = b.Load64(b.FrameVariable(
      FrameVariableInst::FrameVariableKind::kFreeVars, inst.argument()));
  Value* previous = b.LoadSteal(cell, offsetof(PyCellObject, ob_ref));
  ctx->ExceptConditional(b, b.IsZero(previous), [&](Builder& b) {
    b.Call(
        Callee::kFormatUnboundError,
        {b.FrameVariable(FrameVariableInst::FrameVariableKind::kCodeObject, 0),
         b.Int64(inst.argument()), /*is_local=*/b.Zero()});
    return Builder::DoesNotReturn();
  });

  b.Store64(b.Zero(), cell, offsetof(PyCellObject, ob_ref));
  b.DecrefNotNull(previous);
}

////////////////////////////////////////////////////////////////////////////////
// SETUP_EXCEPT, SETUP_LOOP, SETUP_FINALLY, POP_BLOCK, POP_EXCEPT

// The way these work is a little convoluted. SETUP_EXCEPT pushes a TryHandler
// to the handler stack. All exceptions taken while this handler is on the stack
// go to this handler.
//
// POP_BLOCK is used when we exit a try:/except: region without taking an
// exception. It removes the TryHandler from the stack and also pops the stack
// down to the same level as SETUP_EXCEPT (this should not be needed, in
// practice).
//
// When we enter an exception handler six items are pushed to the stack. These
// are the previous exception state exc_{value,type,tb} and the new exception
// state.
// TODO: What state does tstate->exc_* contain?
//
// To handle this, handler blocks identified by SETUP_EXCEPT instructions are
// marked as exception handlers. When the main ingestion driver notices we've
// started an exception handler instruction, it automagically pushes six items
// to the value stack. These six items come from BlockArguments it inserts. It
// is expected that `except` instructions fill in the magic arguments when
// taking an exception. A special TryHandler with kind kExceptHandler is pushed
// onto the handler stack to identify the region handling the exception.
//
// POP_EXCEPT is used when we successfully handle an exception. The stack is
// popped down to the kExceptHandler level *apart from* the last three items,
// which were the previous exc_* state. tstate->exc_* is updated with this state
// and execution continues.

void AnalyzeSetupExcept(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  PcValue target = inst.pc_value().Next().AddOffset(inst.argument());
  ctx->PushExcept(target);
  // Connect the handler into the control flow graph. Before jumping to the
  // handler, except instructions always unwind the stack to the current
  // height. So for dominator tree construction is it correct to model the
  // immediate dominator of the exception handler as the block that pushes it.
  b.Br(nullptr, ctx->BlockAt(inst.pc_value().Next()), ctx->BlockAt(target));
}

void AnalyzeSetupLoop(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  PcValue target = inst.pc_value().Next().AddOffset(inst.argument());
  ctx->PushLoop(target);
  // TODO See if this is still needed for SETUP_LOOP
  // Connect the handler into the control flow graph. Before jumping to the
  // handler, exceptional instructions always unwind the stack to the current
  // height. So for dominator tree construction is it correct to model the
  // immediate dominator of the loop handler as the block that pushes it.
  b.Br(nullptr, ctx->BlockAt(inst.pc_value().Next()), ctx->BlockAt(target));
}

void AnalyzeSetupFinally(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  PcValue target = inst.pc_value().Next().AddOffset(inst.argument());
  ctx->PushFinally(target);
  // Connect the handler into the control flow graph. Before jumping to the
  // handler, exceptional instructions always unwind the stack to the current
  // height. So for dominator tree construction is it correct to model the
  // immediate dominator of the finally handler as the block that pushes it.
  b.Br(nullptr, ctx->BlockAt(inst.pc_value().Next()), ctx->BlockAt(target));
}

void TranslateSetup(const BytecodeInstruction& inst, Builder b,
                    TranslateContext* ctx) {
  // Remove the temporarily placed BrInst and replace with a Jmp.
  Block* block = b.block();
  Block* target = b.block()->GetTerminator()->successors().back();
  b.block()->GetTerminator()->erase();
  target->RemovePredecessor(block);
  block->Create<JmpInst>(ctx->BlockAt(inst.pc_value().Next()));
}

void AnalyzePopBlock(const BytecodeInstruction& inst, Builder b,
                     AnalysisContext* ctx) {
  S6_CHECK(ctx->GetTopHandler().kind() != TryHandler::kFinallyHandler);
  S6_CHECK(ctx->GetTopHandler().kind() != TryHandler::kExceptHandler);
  ctx->PopHandler();
}

void TranslatePopBlock(const BytecodeInstruction& inst, Builder b,
                       TranslateContext* ctx) {
  ctx->PopBlock(b, ctx->GetTopHandler());
}

void AnalyzePopExcept(const BytecodeInstruction& inst, Builder b,
                      AnalysisContext* ctx) {
  S6_CHECK(ctx->GetTopHandler().kind() == TryHandler::kExceptHandler);
  ctx->PopHandler();
}

void TranslatePopExcept(const BytecodeInstruction& inst, Builder b,
                        TranslateContext* ctx) {
  // The translation context needs to know how to pop exception handlers for
  // unwinding the stack anyway, so just delegate.
  ctx->PopExceptHandler(b, ctx->GetTopHandler());
}

////////////////////////////////////////////////////////////////////////////////
// END_FINALLY
//
// END_FINALLY is a very complex bytecode. In the normal interpreter, the stack
// height and block stack shape are not static because of finally blocks.
// In order to manage this we had to make it static again trough multiple means.
//
// Furthermore END_FINALLY can also occur to reraise an exception at the end of
// a normal exception handling block. This is determined by looking at the
// current TryHandler block and whether it is a kExceptHandler or a
// kFinallyHandler. The latter is a fake block that do not exist in the
// normal CPython interpreter. It will correcly manage the stack shape
// described below.
//
//
// In order to make the stack shape static the stack height is fixed on
// entry to the finally handler to be 6 values above the SETUP_FINALLY stack.
//
// The TOS (At the finally handler height + 5) is a discriminator that
// determines the reason for which the finally handler was taken. The
// discriminator is either a PyObject* pointer in case of an exception or a
// constant whoose LSB is 1 in other cases. Since PyObject are always aligned on
// at least 2 bytes, one can determine quickly if we are in an exception or not
// by testing the least signicicant bit of the discriminator.
//
// The possible stack shapes are:
// - On exception, those 6 value are the normal 6 value on entry to an except
//   handler:
//     +5: Dicriminator: current exception type (LSB is 0)
//     +4: Current exception value
//     +3: Current exception traceback
//     +2: Parent exception type
//     +1: Parent exception value
//     +0: Parent exception traceback
//    In that case it unwinding should behave as if there was an
//    EXCEPT_HANDLER
//  - On break, the old stack was (+0: PyLong_FromWhy(Why::kBreak)), the
//    new stack is:
//      +5: Discrimintor: `1 | Why::kBreak` (= WhyToDiscrimintor(Why::kBreak))
//      +0 to +4: 0
//  - On continue, the old stack was:
//      +1: PyLong_FromWhy(Why::kContinue)
//      +0: PcValue to the start of the loop to jump to
//    The new stack is:
//      +5: Discrimintor: `1 | Why::kContinue`
//                                       (= WhyToDiscrimintor(Why::kContinue))
//      +0 to +4: 0
//    The target pointer will be recovered through static means as
//    it is stored in the `pc_continue` field of the kFinallyHandler because
//    each continue calls AnalysisContext::SetContinueTarget at ingestion time.
//  - On return, The old stack was:
//      +1: PyLong_FromWhy(Why::kReturn)
//      +0: the returned value.
//    The new stack is:
//      +5: Discrimintor: `1 | Why::kReturn` (= WhyToDiscrimintor(Why::kReturn))
//      +4: the returned value
//      +0 to +3: 0
//  - On fallthrough, the old stack was (+0: Py_None) and the new stack is:
//      +5: Discrimintor: `1 | Why::kNot` (= WhyToDiscrimintor(Why::kNot))
//      +0 to +4: 0
//  - The silenced case is currently unsupported but SETUP_WITH is not compiled
//    currently.
//
// There is an extra flag in kFinallyHandler that is there to support a common
// case of stack height mismatch. When the `finally_fallthrough_popped_handler`
// flag is set, it means that in the case a fallthrough, the EXCEPT_HANDLER
// right below the current finally handler has already been popped. So no need
// to do it again. If we are in fallthrough the stack shape relative to the
// finally handler origin will be:
//    +5: Discrimintor: `1 | Why::kNot` (= WhyToDiscrimintor(Why::kNot))
//    +0 to +4: 0
//    ------ kFinallyHandler stack height
//    -3 to -1: 0
//    ------ kExceptHandler stack height (There should not be an EXCEPT_HANDLER
//                                        on the block stack of the interpreter)
// If we are not in a fallthorugh the stack shape will be:
//    +5: discrimintor:
//    +0 to +4: usual values as described above without the flag.
//    ------ kFinallyHandler stack height
//    -1: Parent exception type
//    -2: Parent exception value
//    -3: Parent exception traceback
//    ------ kExceptHandler stack height (There should be an EXCEPT_HANDLER
//                                        on the block stack of the interpreter)
//
//
// The job of END_FINALLY is to unwind this new stack shape and resume
// the operation that triggered the finally.

static_assert(alignof(PyObject) >= 2);

void AnalyzeEndFinally(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  if (ctx->GetTopHandler().kind() == TryHandler::kFinallyHandler) {
    // We are in a finally block.
    // The Possible jumping targets are:
    //  - Always: a falltrough or an exception
    //  - If there is a finally block above : that finally block
    //  - If there is a loop block before any finally block:
    //      both the continue and break targets of that loop.
    //  - If there is no finally block, we can also return.

    // Get the possible reason we are in that finally handler:
    WhyFlags why = ctx->GetWhyFinally();

    absl::optional<PcValue> continue_target = ctx->GetContinueTarget();
    S6_CHECK(!why.Has(Why::kContinue) || continue_target);

    // The finally handler calling convention has 6 parameters.
    ctx->Drop(6);

    // END_FINALLY may fallthrough but in that case we jump to the next
    // instruction by hand so we tell the analysis it does not fallthrough.
    ctx->DoesNotFallthrough();

    // Pop the finally handler as finally block is ending.
    auto fhandler = ctx->PopHandler();
    S6_CHECK_EQ(fhandler.kind(), TryHandler::kFinallyHandler);

    // First, handle the fallthrough case.
    if (why.Has(Why::kNot)) {
      BrInst* br =
          b.BrFallthrough(nullptr, ctx->BlockAt(inst.pc_value().Next()));
      if (fhandler.finally_fallthrough_popped_handler()) {
        // Other cases are gonna pop this handler anyway, so we can pop it here.
        auto ehandler = ctx->PopHandler();
        S6_CHECK_EQ(ehandler.kind(), TryHandler::kExceptHandler);
      }
      ctx->Bind(br->mutable_condition());
      ctx->MayJumpTo(inst.pc_value().Next());
    }

    // We don't handle the exception here because that is not required for
    // the analysis phase.

    // Skip the rest if we cannot reach any of those cases.
    WhyFlags whyBRC = Why::kBreak | Why::kContinue | Why::kReturn;
    if (!(why & whyBRC)) {
      b.Unreachable();
      return;
    }

    // From this point on, only return, continue and break are possible.
    // Continue is the one that will trigger the earliest block.
    // TODO: Think if it would be cleaner to support multiple reasons
    // in `UnwindHandlerStack`.
    Value* zero = b.Zero();
    absl::optional<TryHandler> handler =
        ctx->UnwindHandlerStack(Why::kContinue);
    if (!handler) {
      // If there is no handler, then there is no loop handler, so only return
      // is possible.
      S6_CHECK(why.Has(Why::kReturn));
      ReturnInst* ri = b.Return(nullptr);
      ctx->Bind(ri->mutable_returned_value());
    } else if (handler->kind() == TryHandler::kFinally) {
      // If the parent handler is a finally, all three operations go to that
      // finally handler.
      JmpInst* jmp =
          b.Jmp(ctx->BlockAt(handler->pc_value()),
                {zero, zero, zero, zero, /* to be replaced */ zero, zero});
      ctx->Bind(&jmp->mutable_arguments()[4]);
      ctx->Bind(&jmp->mutable_arguments()[5]);
      ctx->MayJumpToFinally(handler->pc_value(), why & whyBRC);
      if (why.Has(Why::kContinue))
        ctx->SetContinueTarget(*handler, continue_target.value());
    } else {
      S6_CHECK(handler->kind() == TryHandler::kLoop);
      // If the parent handler is a loop, all three operation jump to the
      // different targets.

      // Handle the continue using the stored continue data.
      if (why.Has(Why::kContinue)) {
        BrInst* br = b.BrFallthrough(nullptr, ctx->BlockAt(*continue_target));
        ctx->Bind(br->mutable_condition());
        ctx->MayJumpTo(*continue_target);
      }

      ctx->PopHandler();

      // Handle the break: Jump to the loop block target.
      if (why.Has(Why::kBreak)) {
        BrInst* br =
            b.BrFallthrough(nullptr, ctx->BlockAt(handler->pc_value()));
        ctx->Bind(br->mutable_condition());
        ctx->MayJumpTo(handler->pc_value());
      }

      // Handle the return.
      if (why.Has(Why::kReturn)) {
        // Unwind again out of the loop.
        handler = ctx->UnwindHandlerStack(Why::kReturn);
        if (handler) {
          S6_CHECK(handler->kind() == TryHandler::kFinally);
          // There is a finally block,
          // so the return needs to jump to that finally block.
          JmpInst* jmp =
              b.Jmp(ctx->BlockAt(handler->pc_value()),
                    {zero, zero, zero, zero, /* to be replaced */ zero,
                     b.Int64(WhyToDiscriminator(Why::kReturn))});
          ctx->Bind(&jmp->mutable_arguments()[4]);
          ctx->MayJumpToFinally(handler->pc_value(), Why::kReturn);
        } else {
          // There is no finally handler, so we perform an actual return.
          ReturnInst* ri = b.Return(nullptr);
          ctx->Bind(ri->mutable_returned_value());
        }
      } else {
        // Return is impossible.
        b.Unreachable();
      }
    }
  } else {
    S6_CHECK(ctx->GetTopHandler().kind() == TryHandler::kExceptHandler);
    // This is an END_FINALLY for an exception handler. The top three stack
    // items are the exception state to restore (with PyErr_Restore()).

    ctx->Pop().Pop().Pop().DoesNotFallthrough();

    // Temporarily add an unreachable; we'll remove this during translation and
    // replace it with an ExceptInst.
    b.Unreachable();
  }
}

void TranslateEndFinally(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  const TryHandler& top_handler = ctx->GetTopHandler();
  if (top_handler.kind() == TryHandler::kFinallyHandler) {
    // We are in a finally block so all the cases are possible.

    WhyFlags why = ctx->GetWhyFinally();

    Value* discriminator = ctx->Peek(1);
    Value* mayberetval = ctx->Peek(2);

    if (why.Has(Why::kNot)) {
      ctx->Bind(
          b.IsEqual(discriminator, b.Int64(WhyToDiscriminator(Why::kNot))));
      b = Builder::FromStart(b.block()->GetTerminator()->successors().back());
    }

    // Handle the exception case if the LSB of the dicriminator is zero.
    // ExceptConditional preserves the stack height so we can pop inside
    // the conditional lambda.
    ctx->ExceptConditional(
        b, b.IsZero(b.And(discriminator, b.Int64(1))), [&](Builder b) {
          Value* type = ctx->Pop();
          Value* value = ctx->Pop();
          Value* traceback = ctx->Pop();
          b.Call(Callee::kPyErr_Restore, {type, value, traceback});
          ctx->PopExceptHandler(b, top_handler);
          return Builder::DoesNotReturn{};
        });

    // Skip the rest if we cannot reach any of those case
    if (!(why & (Why::kBreak | Why::kContinue | Why::kReturn))) {
      S6_CHECK(isa<UnreachableInst>(b.block()->GetTerminator()));
      return;
    }

    // Pop the stop 6 values from the stack.
    // This also implicitly pops the Finally handler from the block stack.
    ctx->Drop(6);

    // From this point on, only return, continue and break are possible.
    // Continue is the one that will trigger the earliest block.
    absl::optional<TryHandler> handler =
        ctx->UnwindHandlerStack(b, Why::kContinue);
    if (!handler) {
      // If there is no handler, then there is no loop handler, so only return
      // is possible.
      S6_CHECK(why.Has(Why::kReturn));
      ctx->DecrefFastLocals(b);
      ctx->Bind(mayberetval);
    } else if (handler->kind() == TryHandler::kFinally) {
      // If the parent handler is a finally, all three operations go to that
      // finally handler. There is no need to unwind anything else.
      ctx->Bind(mayberetval);
      ctx->Bind(discriminator);
    } else {
      S6_CHECK(handler->kind() == TryHandler::kLoop);
      // If the parent handler is a loop, all three operations jump to the
      // different targets.

      // Handle the continue using the stored continue data.
      if (why.Has(Why::kContinue)) {
        ctx->Bind(b.IsEqual(discriminator,
                            b.Int64(WhyToDiscriminator(Why::kContinue))));
        b = Builder::FromStart(b.block()->GetTerminator()->successors().back());
      }

      ctx->PopBlock(b, *handler);

      // Handle the break: Jump to the loop block target.
      if (why.Has(Why::kBreak)) {
        ctx->Bind(
            b.IsEqual(discriminator, b.Int64(WhyToDiscriminator(Why::kBreak))));
        b = Builder::FromStart(b.block()->GetTerminator()->successors().back());
      }

      // Handle the return.
      if (why.Has(Why::kReturn)) {
        // Unwind again out of the loop.
        if (!ctx->UnwindHandlerStack(b, Why::kReturn).has_value()) {
          // We emitted a ReturnInst.
          S6_CHECK(isa<ReturnInst>(b.block()->GetTerminator()));
          ctx->DecrefFastLocals(b);
        }
        ctx->Bind(mayberetval);
      } else {
        // Return is impossible.
        S6_CHECK(isa<UnreachableInst>(b.block()->GetTerminator()));
      }
    }
  } else {
    S6_CHECK(ctx->GetTopHandler().kind() == TryHandler::kExceptHandler);
    // We are in an Exception Handler.

    // Remove the temporary unreachable.
    S6_CHECK(isa<UnreachableInst>(b.block()->GetTerminator()));
    b.block()->GetTerminator()->erase();
    b = Builder(b.block());

    Value* type = ctx->Pop();
    Value* value = ctx->Pop();
    Value* traceback = ctx->Pop();
    b.Call(Callee::kPyErr_Restore, {type, value, traceback});
    ctx->Except(b);
  }
}

////////////////////////////////////////////////////////////////////////////////
// STORE_SUBSCR

void AnalyzeStoreSubscr(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  // Just pop three items from the stack and fall through.
  ctx->Pop().Pop().Pop();
}

void TranslateStoreSubscr(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* key = ctx->Pop();
  Value* object = ctx->Pop();
  Value* value = ctx->Pop();
  Value* err = b.Call(Callee::kPyObject_SetItem, {object, key, value});
  b.DecrefNotNull({key, object, value});
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// DELETE_SUBSCR

void AnalyzeDeleteSubscr(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  // Just pop two items from the stack and fall through.
  ctx->Pop().Pop();
}

void TranslateDeleteSubscr(const BytecodeInstruction& inst, Builder b,
                           TranslateContext* ctx) {
  Value* key = ctx->Pop();
  Value* object = ctx->Pop();
  Value* err = b.Call(Callee::kPyObject_DelItem, {object, key});
  b.DecrefNotNull({key, object});
  ctx->ExceptIf(b, b.IsNotZero(err));
}

////////////////////////////////////////////////////////////////////////////////
// GET_ITER, FOR_ITER

void AnalyzeGetIter(const BytecodeInstruction& inst, Builder b,
                    AnalysisContext* ctx) {
  ctx->Pop().Push();
}

void TranslateGetIter(const BytecodeInstruction& inst, Builder b,
                      TranslateContext* ctx) {
  Value* top = ctx->Pop();
  Value* v = b.Call(Callee::kPyObject_GetIter, {top});
  b.DecrefNotNull(top);
  ctx->ExceptIf(b, b.IsZero(v));
  ctx->Push(v);
}

void AnalyzeForIter(const BytecodeInstruction& inst, Builder b,
                    AnalysisContext* ctx) {
  // FOR_ITER either:
  //   a) falls through, in which case the dereferenced iterator is pushed.
  //   b) jumps to the loop end, in which case the iterator is popped.

  // Fallthrough case first.
  PcValue fallthrough = inst.pc_value().Next();
  ctx->Push();
  ctx->MayJumpTo(fallthrough);

  // Note the second Pop to undo the Push above.
  PcValue target = inst.pc_value().Next().AddOffset(inst.argument());
  ctx->Pop().Pop();
  ctx->MayJumpTo(target);
  ctx->DoesNotFallthrough();

  // Add a BrInst to ensure the CFG is well-formed for SSA formation. We will
  // remove this later.
  b.Br(nullptr, ctx->BlockAt(fallthrough), ctx->BlockAt(target));
}

void TranslateForIter(const BytecodeInstruction& inst, Builder b,
                      TranslateContext* ctx) {
  // The iterator is at TOS.
  Value* iter = ctx->Peek(1);
  Value* next = b.Call(Callee::kIteratorNext, {iter});
  ctx->ExceptIf(b, b.IsZero(next));

  b.Conditional(b.IsEqual(next, b.Int64(1)), [&](Builder b) {
    // '1' is the result that indicates that the iterator was exhausted.
    b.DecrefNotNull(iter);
    PcValue target = inst.pc_value().Next().AddOffset(inst.argument());
    b.Jmp(ctx->BlockAt(target));
    return Builder::DoesNotReturn();
  });

  // No exception and no StopIteration, so push the next item and fall through.
  ctx->Push(next);
  Block* block = b.block();
  // Remove the BrInst we added in Analyze; replace with a JmpInst.
  S6_CHECK(isa<BrInst>(block->GetTerminator()));
  cast<BrInst>(block->GetTerminator())
      ->false_successor()
      ->RemovePredecessor(block);
  block->GetTerminator()->erase();
  PcValue fallthrough = inst.pc_value().Next();
  block->Create<JmpInst>(ctx->BlockAt(fallthrough));
}

////////////////////////////////////////////////////////////////////////////////
// RAISE_VARARGS

void AnalyzeRaiseVarargs(const BytecodeInstruction& inst, Builder b,
                         AnalysisContext* ctx) {
  switch (inst.argument()) {
    case 0:
      break;
    case 1:
      ctx->Pop();
      break;
    case 2:
      ctx->Pop().Pop();
      break;
    default:
      S6_LOG(FATAL) << "Bad RAISE_VARARGS";
  }
}

void TranslateRaiseVarargs(const BytecodeInstruction& inst, Builder b,
                           TranslateContext* ctx) {
  Value* cause;
  Value* exc;
  switch (inst.argument()) {
    case 0:
      cause = b.Zero();
      exc = cause;
      break;
    case 1:
      cause = b.Zero();
      exc = ctx->Pop();
      break;
    case 2:
      cause = ctx->Pop();
      exc = ctx->Pop();
      break;
    default:
      S6_LOG(FATAL) << "Bad RAISE_VARARGS";
  }
  b.Call(Callee::kHandleRaiseVarargs, {b.Int64(inst.argument()), exc, cause});
  ctx->Except(b);
  // Remove the auto-inserted jmp
  cast<JmpInst>(b.block()->GetTerminator())
      ->unique_successor()
      ->RemovePredecessor(b.block());
  b.block()->GetTerminator()->erase();
}

////////////////////////////////////////////////////////////////////////////////
// YIELD_VALUE

void AnalyzeYieldValue(const BytecodeInstruction& inst, Builder b,
                       AnalysisContext* ctx) {
  ctx->Pop().Push();
}

void TranslateYieldValue(const BytecodeInstruction& inst, Builder b,
                         TranslateContext* ctx) {
  ctx->Push(b.YieldValue(ctx->Pop()));
}

////////////////////////////////////////////////////////////////////////////////
// FORMAT_VALUE

void AnalyzeFormatValue(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  if ((inst.argument() & FVS_MASK) == FVS_HAVE_SPEC) {
    ctx->Pop();
  }
  ctx->Pop().Push();
}

void TranslateFormatValue(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  Value* fmt_spec =
      (inst.argument() & FVS_MASK) == FVS_HAVE_SPEC ? ctx->Pop() : nullptr;
  absl::optional<Callee> conv_func;
  switch (inst.argument() & FVC_MASK) {
    case FVC_STR:
      conv_func = Callee::kPyObject_Str;
      break;
    case FVC_REPR:
      conv_func = Callee::kPyObject_Repr;
      break;
    case FVC_ASCII:
      conv_func = Callee::kPyObject_ASCII;
      break;
  }

  Value* value = ctx->Pop();
  if (conv_func.has_value()) {
    Value* result = b.Call(*conv_func, {value});
    b.DecrefNotNull(value);
    ctx->ExceptConditional(b, b.IsZero(result), [&](Builder b) {
      if (fmt_spec) b.DecrefNotNull(fmt_spec);
      return Builder::DoesNotReturn{};
    });
    value = result;
  }

  Value* result =
      b.Call(Callee::kPyObject_Format, {value, fmt_spec ? fmt_spec : b.Zero()});
  b.DecrefNotNull(value);
  if (fmt_spec) b.DecrefNotNull(fmt_spec);
  ctx->ExceptIf(b, b.IsZero(result));

  ctx->Push(result);
}

////////////////////////////////////////////////////////////////////////////////
// BUILD_STRING

void AnalyzeBuildString(const BytecodeInstruction& inst, Builder b,
                        AnalysisContext* ctx) {
  ctx->Drop(inst.argument());
  ctx->Push();
}

void TranslateBuildString(const BytecodeInstruction& inst, Builder b,
                          TranslateContext* ctx) {
  // The interpreter calls _PyUnicode_JoinArray(empty, args, nargs), where args
  // is passed as an array. This looks identical to the vectorcall calling
  // convention, so use a CallVectorcallInst.
  Value* zero = b.Zero();
  Value* empty = b.Call(Callee::kPyUnicode_New, {zero, zero});
  ctx->ExceptIf(b, b.IsZero(empty));

  std::vector<Value*> args;
  for (int64_t i = 0; i < inst.argument(); ++i) {
    args.push_back(ctx->Pop());
  }
  // Args are passed back-to-front.
  absl::c_reverse(args);

  Value* callee = b.Constant(reinterpret_cast<void*>(_PyUnicode_JoinArray));
  Value* result = b.CallVectorcall(callee,
                                   /*self=*/empty,
                                   /*names=*/nullptr, args);
  b.DecrefNotNull(empty);
  for (Value* v : args) {
    b.DecrefNotNull(v);
  }

  ctx->ExceptIf(b, b.IsZero(result));

  ctx->Push(result);
}

namespace {
std::tuple<int32_t, AnalyzeFunction, TranslateFunction> kFunctions[] = {
    {LOAD_CONST, AnalyzeLoadConst, TranslateLoadConst},
    {JUMP_ABSOLUTE, AnalyzeJumpAbsolute, TranslateNothing},
    {JUMP_FORWARD, AnalyzeJumpForward, TranslateNothing},
    {POP_TOP, AnalyzePopTop, TranslatePopTop},
    {DUP_TOP, AnalyzeDupTop, TranslateDupTop},
    {DUP_TOP_TWO, AnalyzeDupTopTwo, TranslateDupTopTwo},
    {ROT_TWO, AnalyzeRotTwo, TranslateRotTwo},
    {ROT_THREE, AnalyzeRotThree, TranslateRotThree},
    {BINARY_POWER, AnalyzeBinaryOp, TranslatePowerOp},
    {INPLACE_POWER, AnalyzeBinaryOp, TranslatePowerOp},
    {BINARY_MATRIX_MULTIPLY, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_MATRIX_MULTIPLY, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_MULTIPLY, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_MODULO, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_ADD, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_SUBTRACT, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_FLOOR_DIVIDE, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_TRUE_DIVIDE, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_FLOOR_DIVIDE, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_TRUE_DIVIDE, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_ADD, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_SUBTRACT, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_MULTIPLY, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_MODULO, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_LSHIFT, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_RSHIFT, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_AND, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_XOR, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_OR, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_LSHIFT, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_RSHIFT, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_AND, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_XOR, AnalyzeBinaryOp, TranslateBinaryOp},
    {INPLACE_OR, AnalyzeBinaryOp, TranslateBinaryOp},
    {BINARY_SUBSCR, AnalyzeBinaryOp, TranslateBinaryOp},
    {UNARY_POSITIVE, AnalyzeUnaryOp, TranslateUnaryOp},
    {UNARY_NEGATIVE, AnalyzeUnaryOp, TranslateUnaryOp},
    {UNARY_INVERT, AnalyzeUnaryOp, TranslateUnaryOp},
    {UNARY_NOT, AnalyzeUnaryOp, TranslateUnaryNot},
    {CALL_FUNCTION, AnalyzeCallFunction, TranslateCallFunction},
    {CALL_FUNCTION_KW, AnalyzeCallFunctionKw, TranslateCallFunctionKw},
    {LOAD_FAST, AnalyzeLoadFast, TranslateLoadFast},
    {STORE_FAST, AnalyzeStoreFast, TranslateStoreFast},
    {DELETE_FAST, AnalyzeDeleteFast, TranslateDeleteFast},
    {RETURN_VALUE, AnalyzeReturnValue, TranslateReturnValue},
    {BREAK_LOOP, AnalyzeBreakLoop, TranslateBreakLoop},
    {CONTINUE_LOOP, AnalyzeContinueLoop, TranslateContinueLoop},
    {POP_JUMP_IF_TRUE, AnalyzeJumpIf<true, true>, TranslateJumpIf<true, true>},
    {POP_JUMP_IF_FALSE, AnalyzeJumpIf<false, true>,
     TranslateJumpIf<false, true>},
    {JUMP_IF_TRUE_OR_POP, AnalyzeJumpIf<true, false>,
     TranslateJumpIf<true, false>},
    {JUMP_IF_FALSE_OR_POP, AnalyzeJumpIf<false, false>,
     TranslateJumpIf<false, false>},
    {LOAD_NAME, AnalyzeLoadName, TranslateLoadName},
    {STORE_NAME, AnalyzeStoreName, TranslateStoreName},
    {LOAD_GLOBAL, AnalyzeLoadGlobal, TranslateLoadGlobal},
    {STORE_GLOBAL, AnalyzeStoreGlobal, TranslateStoreGlobal},
    {LOAD_ATTR, AnalyzeLoadAttr, TranslateLoadAttr},
    {STORE_ATTR, AnalyzeStoreAttr, TranslateStoreAttr},
    {DELETE_ATTR, AnalyzeDeleteAttr, TranslateDeleteAttr},
    {MAKE_FUNCTION, AnalyzeMakeFunction, TranslateMakeFunction},
    {BUILD_SLICE, AnalyzeBuildSlice, TranslateBuildSlice},
    {BUILD_MAP, AnalyzeBuildMap, TranslateBuildMap},
    {BUILD_CONST_KEY_MAP, AnalyzeBuildConstKeyMap, TranslateBuildConstKeyMap},
    {BUILD_SET, AnalyzeBuildSet, TranslateBuildSet},
    {BUILD_TUPLE, AnalyzeBuildTupleOrList, TranslateBuildTuple},
    {BUILD_TUPLE_UNPACK_WITH_CALL, AnalyzeBuildTupleUnpackWithCall,
     TranslateBuildTupleUnpackWithCall},
    {LIST_APPEND, AnalyzeListAppend, TranslateListAppend},
    {BUILD_LIST, AnalyzeBuildTupleOrList, TranslateBuildList},
    {UNPACK_SEQUENCE, AnalyzeUnpackSequence, TranslateUnpackSequence},
    {COMPARE_OP, AnalyzeCompareOp, TranslateCompareOp},
    {LOAD_CLOSURE, AnalyzeLoadClosure, TranslateLoadClosure},
    {LOAD_DEREF, AnalyzeLoadDeref, TranslateLoadDeref},
    {STORE_DEREF, AnalyzeStoreDeref, TranslateStoreDeref},
    {DELETE_DEREF, AnalyzeDeleteDeref, TranslateDeleteDeref},
    {SETUP_EXCEPT, AnalyzeSetupExcept, TranslateSetup},
    {SETUP_FINALLY, AnalyzeSetupFinally, TranslateSetup},
    {POP_BLOCK, AnalyzePopBlock, TranslatePopBlock},
    {POP_EXCEPT, AnalyzePopExcept, TranslatePopExcept},
    {END_FINALLY, AnalyzeEndFinally, TranslateEndFinally},
    {SETUP_LOOP, AnalyzeSetupLoop, TranslateSetup},
    {STORE_SUBSCR, AnalyzeStoreSubscr, TranslateStoreSubscr},
    {DELETE_SUBSCR, AnalyzeDeleteSubscr, TranslateDeleteSubscr},
    {EXTENDED_ARG, AnalyzeNothing, TranslateNothing},
    {GET_ITER, AnalyzeGetIter, TranslateGetIter},
    {FOR_ITER, AnalyzeForIter, TranslateForIter},
    {RAISE_VARARGS, AnalyzeRaiseVarargs, TranslateRaiseVarargs},
    {YIELD_VALUE, AnalyzeYieldValue, TranslateYieldValue},
    {FORMAT_VALUE, AnalyzeFormatValue, TranslateFormatValue},
    {BUILD_STRING, AnalyzeBuildString, TranslateBuildString},
    {CALL_FUNCTION_EX, AnalyzeCallFunctionEx, TranslateCallFunctionEx},
#if PY_MINOR_VERSION >= 7
    {LOAD_METHOD, AnalyzeLoadMethod, TranslateLoadMethod},
    {CALL_METHOD, AnalyzeCallMethod, TranslateCallMethod},
#endif
};

NoDestructor<
    absl::flat_hash_map<int32_t, std::pair<AnalyzeFunction, TranslateFunction>>>
    function_map;
std::once_flag once;

absl::StatusOr<std::pair<AnalyzeFunction, TranslateFunction>> GetFunction(
    int32_t opcode) {
  std::call_once(once, [&]() {
    for (const auto& [opcode, analyze, translate] : kFunctions) {
      function_map->emplace(opcode, std::make_pair(analyze, translate));
    }
  });
  auto it = function_map->find(opcode);
  if (it == function_map->end()) {
    return absl::UnimplementedError(absl::StrCat(
        "opcode not implemented: ", BytecodeOpcodeToString(opcode)));
  }
  return it->second;
}

}  // namespace

absl::StatusOr<AnalyzeFunction> GetAnalyzeFunction(int32_t opcode) {
  S6_ASSIGN_OR_RETURN(auto pair, GetFunction(opcode));
  return pair.first;
}

absl::StatusOr<TranslateFunction> GetTranslateFunction(int32_t opcode) {
  S6_ASSIGN_OR_RETURN(auto pair, GetFunction(opcode));
  return pair.second;
}

}  // namespace deepmind::s6
