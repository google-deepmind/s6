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

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "core_util.h"
#include "event_counters.h"
#include "interpreter.h"
#include "runtime/callee_address.h"
#include "runtime/evaluator.h"
#include "runtime/interpreter_stub.h"
#include "strongjit/base.h"
#include "strongjit/formatter.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/value_casts.h"
#include "utils/status_macros.h"

// TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION >= 7
#define EXC_INFO(x) exc_state.x
#else
#define EXC_INFO(x) x
#endif

namespace deepmind::s6 {
namespace {
absl::StatusOr<int64_t> EvaluateInst(const ConstantInst& inst,
                                     EvaluatorContext& ctx) {
  return inst.value();
}

absl::StatusOr<int64_t> EvaluateInst(const CompareInst& inst,
                                     EvaluatorContext& ctx) {
  int64_t lhs = ctx.Get(inst.lhs());
  int64_t rhs = ctx.Get(inst.rhs());

  if (inst.IsIntType()) {
    return inst.Evaluate(lhs, rhs) ? 1 : 0;
  }
  // The operands are to be interpreted as float64.
  S6_RET_CHECK(inst.IsDoubleType());
  return inst.Evaluate(absl::bit_cast<double>(lhs), absl::bit_cast<double>(rhs))
             ? 1
             : 0;
}

int64_t* GetMemoryPointer(const MemoryInst& inst, EvaluatorContext& ctx,
                          absl::string_view kind) {
  int64_t base = ctx.Get(inst.pointer());
  int64_t offset = inst.offset();
  int64_t index = inst.index() ? ctx.Get(inst.index()) : 0;
  int64_t shift = MemoryInst::ShiftToInt(inst.shift());
  int64_t pointer = base + offset + (index << shift);
  if (inst.index()) {
    EVLOG(2) << kind << "from [" << reinterpret_cast<void*>(base) << " + "
             << offset << " + " << index << " << " << shift << "]";
  } else {
    EVLOG(2) << kind << "from [" << reinterpret_cast<void*>(base) << " + "
             << offset << "]";
  }
  return reinterpret_cast<int64_t*>(pointer);
}

absl::StatusOr<int64_t> EvaluateInst(const LoadInst& inst,
                                     EvaluatorContext& ctx) {
  S6_RET_CHECK(inst.extension() == LoadInst::kNoExtension)
      << "Extending loads not implemented!";
  int64_t* ptr = GetMemoryPointer(inst, ctx, "Load");
  return *ptr;
}

absl::Status EvaluateInst(const StoreInst& inst, EvaluatorContext& ctx) {
  S6_RET_CHECK(inst.truncation() == StoreInst::kNoTruncation)
      << "Truncating stores not implemented!";
  int64_t* ptr = GetMemoryPointer(inst, ctx, "Store");
  *ptr = ctx.Get(inst.stored_value());
  return absl::OkStatus();
}

absl::Status EvaluateInst(const IncrefInst& inst, EvaluatorContext& ctx) {
  if (inst.nullness() == Nullness::kMaybeNull) {
    Py_XINCREF(ctx.Get<PyObject*>(inst.operand()));
  } else {
    Py_INCREF(ctx.Get<PyObject*>(inst.operand()));
  }
  return absl::OkStatus();
}

absl::Status EvaluateInst(const DecrefInst& inst, EvaluatorContext& ctx) {
  if (inst.nullness() == Nullness::kMaybeNull) {
    Py_XDECREF(ctx.Get<PyObject*>(inst.operand()));
  } else {
    Py_DECREF(ctx.Get<PyObject*>(inst.operand()));
  }
  return absl::OkStatus();
}

template <
    typename SubBinaryInst,
    std::enable_if_t<std::is_base_of_v<BinaryInst, SubBinaryInst>, bool> = true>
absl::StatusOr<int64_t> EvaluateInst(const SubBinaryInst& inst,
                                     EvaluatorContext& ctx) {
  if constexpr (SubBinaryInst::kSupportsDouble) {
    if (inst.IsDoubleType()) {
      return absl::bit_cast<int64_t>(
          inst.Evaluate(absl::bit_cast<double>(ctx.Get(inst.lhs())),
                        absl::bit_cast<double>(ctx.Get(inst.rhs()))));
    }
  }
  return inst.Evaluate(ctx.Get(inst.lhs()), ctx.Get(inst.rhs()));
}

template <
    typename SubUnaryInst,
    std::enable_if_t<std::is_base_of_v<UnaryInst, SubUnaryInst>, bool> = true>
absl::StatusOr<int64_t> EvaluateInst(const SubUnaryInst& inst,
                                     EvaluatorContext& ctx) {
  if (inst.IsDoubleType()) {
    return absl::bit_cast<int64_t>(
        inst.Evaluate(absl::bit_cast<double>(ctx.Get(inst.operand()))));
  }
  return inst.Evaluate(ctx.Get(inst.operand()));
}

absl::StatusOr<double> EvaluateInst(const IntToFloatInst& inst,
                                    EvaluatorContext& ctx) {
  return inst.Evaluate(ctx.Get(inst.operand()));
}

absl::StatusOr<PyObject*> EvaluateInst(const BoxInst& inst,
                                       EvaluatorContext& ctx) {
  int64_t content = ctx.Get(inst.content());
  switch (inst.type()) {
    case UnboxableType::kPyLong:
      return PyLong_FromLong(content);
    case UnboxableType::kPyBool:
      return PyBool_FromLong(content);
    case UnboxableType::kPyFloat:
      return PyFloat_FromDouble(absl::bit_cast<double>(content));
  }
  S6_UNREACHABLE();
}

absl::StatusOr<int64_t> EvaluateInst(const UnboxInst& inst,
                                     EvaluatorContext& ctx) {
  PyObject* boxed = reinterpret_cast<PyObject*>(ctx.Get(inst.boxed()));
  // Return 0 if the unboxing fails, as the failure will be picked up by the
  // overflowed? instruction that is supposed to be next.
  return inst.Evaluate(boxed).value_or(0);
}

absl::StatusOr<int64_t> EvaluateInst(const OverflowedInst& inst,
                                     EvaluatorContext& ctx) {
  if (auto bi = dyn_cast<BinaryInst>(inst.arithmetic_value())) {
    return inst.Evaluate(*bi, ctx.Get(bi->lhs()), ctx.Get(bi->rhs()));
  }
  if (auto ui = dyn_cast<UnaryInst>(inst.arithmetic_value())) {
    return inst.Evaluate(*ui, ctx.Get(ui->operand()));
  }
  if (auto ui = dyn_cast<UnboxInst>(inst.arithmetic_value())) {
    return inst.Evaluate(*ui,
                         reinterpret_cast<PyObject*>(ctx.Get(ui->boxed())));
  }
  return false;
}

absl::StatusOr<int64_t> EvaluateInst(const FloatZeroInst& inst,
                                     EvaluatorContext& ctx) {
  return inst.Evaluate(absl::bit_cast<double>(ctx.Get(inst.float_value())));
}

absl::StatusOr<PyObject*> EvaluateInst(const FrameVariableInst& inst,
                                       EvaluatorContext& ctx) {
  PyCodeObject* co = reinterpret_cast<PyCodeObject*>(ctx.pyframe()->f_code);
  switch (inst.frame_variable_kind()) {
    case FrameVariableInst::FrameVariableKind::kConsts:
      return PyTuple_GET_ITEM(co->co_consts, inst.index());
    case FrameVariableInst::FrameVariableKind::kFrame:
      S6_LOG(FATAL) << "You can't just get a kFrame! You will regret this!";
      break;
    case FrameVariableInst::FrameVariableKind::kBuiltins:
      return ctx.pyframe()->f_builtins;
    case FrameVariableInst::FrameVariableKind::kGlobals:
      return ctx.pyframe()->f_globals;
    case FrameVariableInst::FrameVariableKind::kFastLocals:
      return reinterpret_cast<PyObject*>(
          &ctx.pyframe()->f_localsplus[inst.index()]);
    case FrameVariableInst::FrameVariableKind::kFreeVars:
      return reinterpret_cast<PyObject*>(
          &ctx.pyframe()->f_localsplus[co->co_nlocals + inst.index()]);
    case FrameVariableInst::FrameVariableKind::kNames:
      return PyTuple_GET_ITEM(co->co_names, inst.index());
    case FrameVariableInst::FrameVariableKind::kLocals:
      return ctx.pyframe()->f_locals;
    case FrameVariableInst::FrameVariableKind::kCodeObject:
      return reinterpret_cast<PyObject*>(co);
    case FrameVariableInst::FrameVariableKind::kThreadState:
      return reinterpret_cast<PyObject*>(PyThreadState_GET());
    default:
      std::stringstream ss;
      ss << "FrameVariable unimplemented in evaluator!";
      return absl::UnimplementedError(ss.str());
  }
  S6_UNREACHABLE();
}

absl::StatusOr<int64_t> EvaluateInst(const CallNativeInst& inst,
                                     EvaluatorContext& ctx) {
  std::vector<int64_t> arguments;
  arguments.reserve(inst.operands().size());
  for (const Value* v : inst.operands()) {
    arguments.push_back(ctx.Get(v));
  }
  S6_ASSIGN_OR_RETURN(void* address, GetCalleeSymbolAddress(inst.callee()));
  return CallNative(address, arguments);
}

absl::StatusOr<int64_t> EvaluateInst(const CallNativeIndirectInst& inst,
                                     EvaluatorContext& ctx) {
  std::vector<int64_t> arguments;
  arguments.reserve(inst.call_arguments().size());
  for (const Value* v : inst.call_arguments()) {
    arguments.push_back(ctx.Get(v));
  }
  return CallNative(ctx.Get<void*>(inst.callee()), arguments);
}

absl::Status EvaluateInst(const JmpInst& inst, EvaluatorContext& ctx) {
  const Block* succ = inst.unique_successor();
  S6_RET_CHECK_EQ(inst.arguments().size(), succ->block_arguments_size());
  int64_t i = 0;
  for (const BlockArgument* arg : succ->block_arguments()) {
    ctx.Set(arg, ctx.Get(inst.arguments()[i++]));
  }

  ctx.SetNextInstruction(&*succ->begin());
  EVLOG(2) << "Jumped to " << FormatOrDie(*ctx.GetNextInstruction());
  return absl::OkStatus();
}

absl::Status EvaluateInst(const BrInst& inst, EvaluatorContext& ctx) {
  bool condition = ctx.Get(inst.condition()) != 0;
  const Block* succ =
      condition ? inst.true_successor() : inst.false_successor();
  auto arguments = condition ? inst.true_arguments() : inst.false_arguments();
  S6_RET_CHECK_EQ(arguments.size(), succ->block_arguments_size());
  int64_t i = 0;
  for (const BlockArgument* arg : succ->block_arguments()) {
    ctx.Set(arg, ctx.Get(arguments[i++]));
  }

  ctx.SetNextInstruction(&*succ->begin());
  EVLOG(2) << "Branched to " << FormatOrDie(*ctx.GetNextInstruction());
  return absl::OkStatus();
}

absl::Status EvaluateInst(const ExceptInst& inst, EvaluatorContext& ctx) {
  const Block* succ = inst.unique_successor();
  PyThreadState* tstate = PyThreadState_GET();
  S6_CHECK(PyErr_Occurred());
  // Start by logging a traceback and the actual slowpath of calling
  // exception tracing if enabled.
  PyTraceBack_Here(tstate->frame);
  if (tstate->c_tracefunc) {
    CallExceptionTrace(tstate, tstate->frame);
  }

  if (!succ) {
    EVLOG(1) << "Exception-returned nullptr!";
    ctx.SetReturnValue(nullptr);
    return absl::OkStatus();
  }

  auto it = succ->block_arguments().begin();
  ctx.Set(*it++, tstate->EXC_INFO(exc_traceback));
  ctx.Set(*it++, tstate->EXC_INFO(exc_value));
  if (tstate->EXC_INFO(exc_type)) {
    ctx.Set(*it++, tstate->EXC_INFO(exc_type));
  } else {
    Py_INCREF(Py_None);
    ctx.Set(*it++, Py_None);
  }
  PyObject *exc_type, *exc_value, *exc_traceback;
  PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
  // Make the raw exception data available to the handler, so a program
  // can emulate the Python main loop.
  PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);
  PyException_SetTraceback(exc_value, exc_traceback ? exc_traceback : Py_None);
  Py_INCREF(exc_type);
  tstate->EXC_INFO(exc_type) = exc_type;
  Py_INCREF(exc_value);
  tstate->EXC_INFO(exc_value) = exc_value;
  tstate->EXC_INFO(exc_traceback) = exc_traceback;
  if (!exc_traceback) {
    exc_traceback = Py_None;
  }
  Py_INCREF(exc_traceback);
  ctx.Set(*it++, exc_traceback);
  ctx.Set(*it++, exc_value);
  ctx.Set(*it++, exc_type);

  // Set the rest of the except instruction arguments.
  int64_t i = 0;
  while (it != succ->block_arguments().end()) {
    ctx.Set(*it++, ctx.Get(inst.arguments()[i++]));
  }

  ctx.SetNextInstruction(&*succ->begin());
  EVLOG(2) << "Excepted to " << FormatOrDie(*ctx.GetNextInstruction());
  return absl::OkStatus();
}

absl::Status EvaluateInst(const UnreachableInst& inst, EvaluatorContext& ctx) {
  return absl::FailedPreconditionError("UNREACHABLE executed in evaluator!");
}

absl::Status EvaluateInst(const DeoptimizeIfInst& inst, EvaluatorContext& ctx) {
  bool condition = ctx.Get(inst.condition()) != 0;
  if (inst.negated()) condition = !condition;
  const Block* succ =
      condition ? inst.true_successor() : inst.false_successor();
  auto arguments = condition ? inst.true_arguments() : inst.false_arguments();
  S6_RET_CHECK_EQ(arguments.size(), succ->block_arguments_size());
  int64_t i = 0;

  if (condition) {
    // The evaluator is now running in deoptimized code, so should stop at a
    // bytecode_begin.
    ctx.set_finish_on_bytecode_begin(true);
  }

  // Execute virtual copies.
  for (const BlockArgument* arg : succ->block_arguments()) {
    ctx.Set(arg, ctx.Get(arguments[i++]));
  }

  ctx.SetNextInstruction(&*succ->begin());
  EVLOG(2) << "Branched to " << FormatOrDie(*ctx.GetNextInstruction());
  return absl::OkStatus();
}

absl::StatusOr<PyObject*> EvaluateInst(const ReturnInst& inst,
                                       EvaluatorContext& ctx) {
  PyObject* ret = ctx.Get<PyObject*>(inst.returned_value());
  EVLOG(2) << "Returning " << ret;
  ctx.SetReturnValue(ret);
  return ret;
}

absl::Status EvaluateInst(const BytecodeBeginInst& inst,
                          EvaluatorContext& ctx) {
  return absl::OkStatus();
}

PyObject* CallPython(const CallPythonInst& ci, PyObject* receiver,
                     EvaluatorContext& ctx) {
  if (!receiver) receiver = ctx.Get<PyObject*>(ci.callee());
  std::vector<PyObject*> args;
  for (const Value* v : ci.call_arguments()) {
    args.push_back(ctx.Get<PyObject*>(v));
  }

  PyObject* names = nullptr;
  int64_t positional_count = args.size();
  if (ci.names()) {
    names = reinterpret_cast<PyObject*>(ctx.Get(ci.names()));
    positional_count -= PyTuple_Size(names);
  }

  PyObject* ret;
  if (PyCFunction_Check(receiver)) {
    ret = _PyCFunction_FastCallKeywords(receiver, args.data(), positional_count,
                                        names);
  } else if (PyFunction_Check(receiver)) {
    ret = _PyFunction_FastCallKeywords(receiver, args.data(), positional_count,
                                       names);
  } else {
    ret = _PyObject_FastCallKeywords(receiver, args.data(), positional_count,
                                     names);
  }
  Py_DECREF(receiver);
  for (PyObject* arg : args) {
    Py_DECREF(arg);
  }
  return ret;
}

absl::StatusOr<PyObject*> EvaluateInst(const CallPythonInst& inst,
                                       EvaluatorContext& ctx) {
  if (inst.fastcall()) {
    int64_t n = inst.call_arguments().size();
    PyObject** args =
        reinterpret_cast<PyObject**>(alloca(sizeof(PyObject*) * n));
    for (int64_t i = 0; i < n; ++i) {
      args[i] = ctx.Get<PyObject*>(inst.call_arguments()[i]);
    }
    return StrongjitInterpreterStubArrayArgs(ctx.Get<PyObject*>(inst.callee()),
                                             n, args);
  }

  return CallPython(inst, nullptr, ctx);
}

absl::StatusOr<PyObject*> EvaluateInst(const CallAttributeInst& inst,
                                       EvaluatorContext& ctx) {
  PyObject* obj = ctx.Get<PyObject*>(inst.object());
  std::string str(inst.attribute_str());
  PyObject* str_obj = ctx.GetUnicodeObjectForStringName(str);
  PyObject* attr = PyObject_GetAttr(obj, str_obj);
  if (!attr) {
    return nullptr;
  }
  ctx.pyframe()->f_lasti = inst.call_python_bytecode_offset();
  PyObject* ret = CallPython(inst, attr, ctx);
  Py_DECREF(obj);
  return ret;
}

absl::Status EvaluateInst(const AdvanceProfileCounterInst& inst,
                          EvaluatorContext& ctx) {
  // Don't advance the profile counter from the evaluator, we're not
  // running in optimized code!
  return absl::OkStatus();
}

absl::Status EvaluateInst(const IncrementEventCounterInst& inst,
                          EvaluatorContext& ctx) {
  ++*EventCounters::Instance().GetEventCounter(inst.name_str());
  return absl::OkStatus();
}

absl::Status EvaluateInst(const TraceBeginInst& inst, EvaluatorContext& ctx) {
  return absl::OkStatus();
}

absl::Status EvaluateInst(const TraceEndInst& inst, EvaluatorContext& ctx) {
  return absl::OkStatus();
}

absl::Status EvaluateInst(const YieldValueInst& inst, EvaluatorContext& ctx) {
  PyObject* ret = ctx.Get<PyObject*>(inst.yielded_value());
  EVLOG(2) << "Yielding " << ret;
  ctx.SetReturnValue(ret);
  return absl::OkStatus();
}

absl::Status EvaluateInst(const DeoptimizeIfSafepointInst& inst,
                          EvaluatorContext& ctx) {
  bool condition = ctx.Get(inst.condition()) != 0;
  if (inst.negated()) condition = !condition;
  if (condition) {
    // Finished.
    ctx.SetReturnValue(nullptr);
  }
  return absl::OkStatus();
}

absl::Status EvaluateInst(const RematerializeInst& inst,
                          EvaluatorContext& ctx) {
  // Rematerialize does nothing.
  return absl::OkStatus();
}

PyDictObject* GetDictPtr(PyObject* obj, int64_t dictoffset) {
  return *reinterpret_cast<PyDictObject**>(reinterpret_cast<int64_t>(obj) +
                                           dictoffset);
}

absl::StatusOr<int64_t> EvaluateInst(const GetClassIdInst& inst,
                                     EvaluatorContext& ctx) {
  PyObject* obj = ctx.Get<PyObject*>(inst.object());
  PyTypeObject* type = Py_TYPE(obj);
  int64_t dictoffset = type->tp_dictoffset;
  if (dictoffset <= 0) return type->tp_flags >> 44;
  PyDictObject* dict = GetDictPtr(obj, dictoffset);
  if (dict == nullptr) return type->tp_flags >> 44;
  return dict->ma_version_tag >> 44;
}

absl::StatusOr<PyDictObject*> EvaluateInst(const GetObjectDictInst& inst,
                                           EvaluatorContext& ctx) {
  PyObject* obj = ctx.Get<PyObject*>(inst.object());
  PyTypeObject* type = Py_TYPE(obj);

  if (inst.type()) {
    if (type != reinterpret_cast<PyTypeObject*>(inst.type())) return nullptr;
    PyDictObject* dict = GetDictPtr(obj, inst.dictoffset());
    return dict;
  }

  if (inst.dictoffset()) {
    if (type->tp_dictoffset != inst.dictoffset()) return nullptr;
    PyDictObject* dict = GetDictPtr(obj, inst.dictoffset());
    return dict;
  }

  int64_t dictoffset = type->tp_dictoffset;
  if (dictoffset <= 0) return nullptr;
  PyDictObject* dict = GetDictPtr(obj, dictoffset);
  return dict;
}

absl::StatusOr<int64_t> EvaluateInst(const GetInstanceClassIdInst& inst,
                                     EvaluatorContext& ctx) {
  PyDictObject* dict = ctx.Get<PyDictObject*>(inst.dict());
  return dict->ma_version_tag >> 44;
}

absl::StatusOr<int64_t> EvaluateInst(const CheckClassIdInst& inst,
                                     EvaluatorContext& ctx) {
  PyObject* obj = ctx.Get<PyObject*>(inst.object());

  // Note: it is easier to just write GetClassId(obj) == inst.class_id() here,
  // but we spell out and implement the exact algorithm that the code
  // generator uses here.
  const Class* cls = inst.class_();
  if (cls->is_globals_class()) {
    PyTypeObject* type = Py_TYPE(obj);
    if (type->tp_dictoffset <= 0) return 0;
    PyDictObject* dict = GetDictPtr(obj, type->tp_dictoffset);
    if (dict == nullptr) return 0;
    int64_t class_id = dict->ma_version_tag >> 44;
    return class_id == inst.class_id() ? 1 : 0;
  }

  PyTypeObject* expected_type = cls->type();
  S6_CHECK(expected_type);
  if (!cls->is_base_class()) {
    S6_CHECK_GT(cls->dictoffset(), 0);
    if (obj->ob_type != expected_type) return 0;
    PyDictObject* dict = GetDictPtr(obj, cls->dictoffset());
    if (dict == nullptr) return 0;
    int64_t class_id = dict->ma_version_tag >> 44;
    return class_id == inst.class_id() ? 1 : 0;
  }

  if (cls->dictoffset() > 0) {
    S6_CHECK(cls->is_base_class());
    if (obj->ob_type != expected_type) return 0;
    PyDictObject* dict = GetDictPtr(obj, cls->dictoffset());
    if (dict) return 0;
    return 1;  // Type is correct and no instance dict, so class is correct.
  }

  S6_CHECK_EQ(cls->dictoffset(), 0);
  if (obj->ob_type != expected_type) return 0;
  // The type CANNOT have a dict, so if the type is correct so is the class.
  return 1;
}

// Returns the address of the value within a combined dict. This is a private
// implementation detail of PyDictObject.
PyObject** GetCombinedDictEntry(PyDictObject* dict, int64_t index) {
  _PyDictKeysObject* keys = reinterpret_cast<_PyDictKeysObject*>(dict->ma_keys);
  int64_t index_entry_size;
  int64_t size = keys->dk_size;
  if (size <= 0xff) {
    index_entry_size = 1;
  } else if (size <= 0xffff) {
    index_entry_size = 2;
  } else if (size <= 0xffffffff) {
    index_entry_size = 4;
  } else {
    index_entry_size = 8;
  }

  uint8_t* data = reinterpret_cast<uint8_t*>(keys->dk_indices);
  _PyDictKeyEntry** entry = reinterpret_cast<_PyDictKeyEntry**>(
      data + (index_entry_size * keys->dk_size));
  return &(*entry)->value;
}

absl::StatusOr<PyObject*> EvaluateInst(const LoadFromDictInst& inst,
                                       EvaluatorContext& ctx) {
  PyDictObject* dict_obj = ctx.Get<PyDictObject*>(inst.dict());
  S6_CHECK(dict_obj != nullptr);

  if (inst.dict_kind() == DictKind::kSplit) {
    S6_CHECK(dict_obj->ma_values);
    return dict_obj->ma_values[inst.index()];
  } else {
    S6_CHECK(inst.dict_kind() == DictKind::kCombined);
    return *GetCombinedDictEntry(dict_obj, inst.index());
  }
}

absl::StatusOr<PyObject*> EvaluateInst(const StoreToDictInst& inst,
                                       EvaluatorContext& ctx) {
  PyDictObject* dict_obj = ctx.Get<PyDictObject*>(inst.dict());
  S6_CHECK(dict_obj);
  if (inst.dict_kind() == DictKind::kSplit) {
    S6_CHECK(dict_obj->ma_values);
    PyObject* prev = dict_obj->ma_values[inst.index()];
    dict_obj->ma_values[inst.index()] = ctx.Get<PyObject*>(inst.value());
    return prev;
  } else {
    S6_CHECK(inst.dict_kind() == DictKind::kCombined);
    PyObject** ptr = GetCombinedDictEntry(dict_obj, inst.index());
    PyObject* prev = *ptr;
    *ptr = ctx.Get<PyObject*>(inst.value());
    return prev;
  }
}

absl::StatusOr<int64_t> EvaluateInst(const DeoptimizedAsynchronouslyInst& inst,
                                     EvaluatorContext& ctx) {
  // The evaluator doesn't have an S6CodeObject to query, so assume it's
  // always deoptimized.
  return 1;
}

absl::StatusOr<PyObject*> EvaluateInst(const CallVectorcallInst& inst,
                                       EvaluatorContext& ctx) {
  std::vector<PyObject*> arguments;
  arguments.reserve(inst.call_arguments().size());
  for (const Value* v : inst.call_arguments()) {
    arguments.push_back(ctx.Get<PyObject*>(v));
  }
#if PY_MINOR_VERSION >= 7
  using CalleeType = _PyCFunctionFastWithKeywords;
#else
  using CalleeType = _PyCFunctionFast;
#endif
  CalleeType callee = ctx.Get<CalleeType>(inst.callee());
  return (*callee)(ctx.Get<PyObject*>(inst.self()), arguments.data(),
                   arguments.size(),
                   inst.names() ? ctx.Get<PyObject*>(inst.names()) : nullptr);
}

absl::Status EvaluateInst(const SetObjectClassInst& inst,
                          EvaluatorContext& ctx) {
  PyDictObject* dict = ctx.Get<PyDictObject*>(inst.dict());
  dict->ma_version_tag = inst.class_id() << 44;
  return absl::OkStatus();
}

absl::StatusOr<PyObject*> EvaluateInst(const ConstantAttributeInst& inst,
                                       EvaluatorContext& ctx) {
  return inst.LookupAttribute(ClassManager::Instance()).value();
}

absl::Status EvaluateInst(const LoadGlobalInst& inst, EvaluatorContext& ctx) {
  return absl::UnimplementedError(
      "LoadGlobalInst should have been eliminated by the optimizer "
      "so its evaluation is not implemented");
}

// Functor struct to give to ForAllInstructionKinds.
struct InstructionEvaluator {
  template <typename InstrType>
  static std::optional<absl::Status> Visit(const Instruction& inst,
                                           EvaluatorContext& ctx) {
    if (inst.kind() != InstrType::kKind) return {};
    // We allow EvaluateInst to return either absl::Status or absl::StatusOr.
    auto r = EvaluateInst(cast<InstrType>(inst), ctx);
    if constexpr (std::is_same_v<decltype(r), absl::Status>) {
      return r;
    } else {  // Note constexpr else
      // Return value must therefore be absl::StatusOr.
      if (!r.status().ok()) {
        return r.status();
      }
      ctx.Set(&inst, absl::bit_cast<int64_t>(r.value()));
      return absl::OkStatus();
    }
  }

  static absl::Status Default(const Instruction& inst, EvaluatorContext& ctx) {
    return absl::InvalidArgumentError(
        "unhandled instruction kind in evaluator!");
  }
};
}  // namespace

// Calls `callee`, with arguments and result as int64_t.
int64_t CallNative(void* callee, absl::Span<int64_t const> arguments) {
  // We handle up to 4 arguments.
  S6_CHECK(arguments.size() <= 4) << "Unhandled arity in evaluator!";
  auto arg = +[](absl::Span<int64_t const> arguments, int64_t i) {
    return i >= arguments.size() ? 0LL : arguments[i];
  };

  // It is always safe to call a function with *more* arguments than its
  // signature. We take advantage of this to only have one function type that
  // takes the maximum number of arguments we support, and feed the excess
  // arguments zero.
  using Fn = int64_t (*)(int64_t, int64_t, int64_t, int64_t);
  Fn fn = reinterpret_cast<Fn>(callee);
  return fn(arg(arguments, 0), arg(arguments, 1), arg(arguments, 2),
            arg(arguments, 3));
}

absl::Status Evaluate(const Instruction& inst, EvaluatorContext& ctx) {
  ctx.SetNextInstruction(nullptr);
  ctx.ClearReturnValue();

  if (const PreciseLocationInst* pi = dyn_cast<PreciseLocationInst>(&inst)) {
    ctx.pyframe()->f_lasti = pi->bytecode_offset();
  }

  absl::Status status =
      ForAllInstructionKinds<InstructionEvaluator, const Instruction&,
                             EvaluatorContext&>(inst, ctx);
  S6_RETURN_IF_ERROR(status);

  // If the function did not terminate, and next_instruction was not filled in,
  // fill it in now.
  if (!ctx.IsFinished() && !ctx.GetNextInstruction()) {
    auto it = std::next(inst.GetIterator());
    const Block* b = inst.parent();
    S6_RET_CHECK(it != b->end())
        << "Ran off the end of a block without seeing a terminator!";
    ctx.SetNextInstruction(&*it);
  }
  return absl::OkStatus();
}

}  // namespace deepmind::s6
