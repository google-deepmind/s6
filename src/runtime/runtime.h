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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_RUNTIME_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_RUNTIME_H_

#include <Python.h>

#include <cstdint>

#include "code_object.h"
#include "core_util.h"
#include "runtime/generator.h"
#include "runtime/stack_frame.h"
#include "strongjit/instructions.h"

namespace deepmind::s6 {

// Sets the error indicator and formats an error message if `object` is non-null
// and can be converted to a string.  `exception` should be a Python exception
// class, `format_string` is a C printf format string with one format specifier
// and `object` is the value for that specifier.
void FormatError(PyObject* exception, const char* format_string,
                 PyObject* object);

// Sets the error indicator for an unbound variable error.
// If `is_local` is true, the error occurred when accessing a fastlocal.
// Otherwise the error occurred when accessing a cell or free variable; The kind
// of error depends on whether `index` is an index into the code object's
// closed-over variables or into its free variables.
void FormatUnboundError(PyCodeObject* code, int index, bool is_local);

// Looks up `name` in globals and if that fails, in builtins.  Returns nullptr
// if there was an exception.
PyObject* LoadGlobal(PyObject* name, PyObject* globals, PyObject* builtins);

// Handles a StopIterationException.
int64_t HandleStopIteration();

// `iter` is an iterator. Runs tp->iternext(iter), and returns:
//   * Returns an object if the iterator was not exhausted.
//   * Returns 1 if the iterator was exhausted.
//   * Returns nullptr if any other exception occurred.
PyObject* IteratorNext(PyObject* iter);

// Handles a RAISE_VARARGS instruction. The first argument is the bytecode
// argument (0, 1 or 2).
void HandleRaiseVarargs(int64_t argument, PyObject* exc, PyObject* cause);

// Helper for COMPARE_OP; this implements the checking around
// PyErr_GivenExceptionMatches and calls it. Returns -1 if an exception was
// raised, otherwise the result of PyErr_GivenExceptionMatches.
int64_t CheckedGivenExceptionMatches(PyObject* left, PyObject* right);

// Sets up the free variables for a function call. Always succeeds.
void SetupFreeVars(PyObject* func, PyFrameObject* pyframe);

// Calls _Py_Dealloc.
void Dealloc(PyObject* obj);

// Calls a Python function.
//   `num_args`: The size of `args`.
//   `args`: An array of arguments, with args[-1] being the callee.
//   `names`: The names tuple, or nullptr.
//   `stack_frame`: The prior stack frame.
PyObject* CallPython(int64_t arg_count, PyObject** args, PyObject* names,
                     StackFrame* stack_frame);

// Implements a CallAttributeInst.
// REQUIRES: The bytecode offset inside `stack_frame` corresponds to the
// CallNativeInst(GetAttr).
//
//   `num_args`: The size of `args`.
//   `args`: An array of arguments, with args[-1] being the receiver.
//   `names`: The names tuple, or nullptr.
//   `stack_frame`: The prior stack frame.
//   `attr_str`: The attribute name for getattr, as a PyUnicodeObject*.
//   `call_python_bytecode_offset`: The bytecode offset for the CallPythonInst.
PyObject* CallAttribute(int64_t arg_count, PyObject** args, PyObject* names,
                        StackFrame* stack_frame, PyUnicodeObject* attr_str,
                        int64_t call_python_bytecode_offset);

// Handles an exception where there is no exception handler. Calls
// PyTraceBack_Here.
void ExceptWithoutHandler(int64_t bytecode_offset);

// Handles an exception. Calls PyTraceBack_Here, Latches PyThreadState::curexc_*
// into exc_*, normalizes the exception and returns the six values for the
// exception handler in output_values.
void Except(int64_t bytecode_offset, PyObject** output_values);

// Sets up the StackFrame object for a generator. Returns the object to yield
// and the generator state.
std::pair<PyObject*, GeneratorState*> SetUpStackFrameForGenerator(
    PyFrameObject* pyframe, StackFrame* stack_frame, CodeObject* code_object,
    int64_t num_spill_slots);

// Sets up the StackFrame object for a non-generator.
void SetUpStackFrame(PyFrameObject* pyframe, StackFrame* stack_frame,
                     CodeObject* code_object);

// Cleans up a StackFrame after function execution has completed.
// TODO: It might be wise to inline this into the generated epilog.
void CleanupStackFrame(StackFrame* stack_frame);

// Cleans up a StackFrame after a generator function execution has completed.
void CleanupStackFrameForGenerator(StackFrame* stack_frame);

// A generator function is about to yield `result`.
//
// This runtime function acts like setjmp or fork - it returns twice. We are
// entered during a YieldValueInst, but we also need to support resuming
// a generator after a YieldValueInst. We do this by storing this function's
// return address inside the generator state; when resuming, we'll jump to
// this address, and the code following this call will run twice.
//
// The return value from this function is:
//   If yielding: {yield_result, 0}
//   If resuming: {sent_value, (nonzero)}
std::pair<PyObject*, int64_t> SetUpForYieldValue(PyObject* result,
                                                 StackFrame* stack_frame,
                                                 const YieldValueInst* yi);

// Calls OracleProfileEvent() while saving all registers.
// Note that the `preserve_all` attribute is a clang only feature that avoids
// us having to write assembly:
//   https://clang.llvm.org/docs/AttributeReference.html#calling-conventions
S6_ATTRIBUTE_PRESERVE_CALLER_SAVED_REGISTERS void ProfileEventReachedZero();

// Calls GetPyFrameObjectCache()->AllocateOnHeap(), while saving all general
// purpose registers.
S6_ATTRIBUTE_PRESERVE_CALLER_SAVED_REGISTERS
void AllocateFrameOnHeap();

// Called by a RematerializeInst to perform an attribute lookup. To minimize
// the number of live variables needing to be kept alive, this function takes
// the receiver object and an index into the current PyFrameObject's names
// tuple.
PyObject* RematerializeGetAttr(PyObject* receiver, int64_t names_index);

// Start a function tracing event with the tag `tag`.
void BeginFunctionTrace(const char* tag);

// End a function tracing event with the tag `tag`. Start and end calls must be
// balanced.
void EndFunctionTrace(const char* tag);

// `object` is a fresh object without a __dict__. Add a dict, store
// `stored_value` in it, and transition to `cls`.
//
// Returns nonzero on success, zero if the initialized object dict should have
// been split but ended up combined.
int InitializeObjectDict(PyObject* object, PyObject* name,
                         PyObject* stored_value, const Class* cls);

// Return a new function object associated with the code object `code`.
// This wraps `PyFunction_NewWithQualName`, allowing s6 to record the
// Association between the code object and the qualified name.
PyObject* NewFunctionWithQualName(PyObject* code, PyObject* globals,
                                  PyObject* qualified_name);

// Creates a new PyGeneratorObject.
PyGenObject* CreateGenerator(PyFunctionObject* function, PyObject* builtins,
                             ...);

// Similar to BuildListUnpack in interpreter.cc, but takes the stack as variadic
// arguments.
PyObject* BuildListUnpackVararg(int count, PyObject* func, ...);

// Calls a callable with a sequence and a mapping for positional and keyword
// arguments. This implements CALL_FUNCTION_EX.
PyObject* CallFunctionEx(PyObject* callable, PyObject* positional,
                         PyObject* mapping);

// Wrappers around libm functions that take their input in a general-purpose
// register (not an XMM register).
int64_t SinDouble(int64_t x);
int64_t CosDouble(int64_t x);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_RUNTIME_H_
