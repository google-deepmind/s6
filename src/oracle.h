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

#ifndef THIRD_PARTY_DEEPMIND_S6_ORACLE_H_
#define THIRD_PARTY_DEEPMIND_S6_ORACLE_H_

#include <Python.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "allocator.h"
#include "classes/class_manager.h"
#include "code_generation/register_allocator.h"
#include "code_object.h"
#include "metadata.h"
#include "strongjit/function.h"
#include "utils/channel.h"

namespace deepmind::s6 {

struct OracleDebugOptions {
  // If true, compilation should proceed synchronously, for debugging
  // purposes. Otherwise compilation is asynchronous.
  bool synchronous = false;

  // If true, all possible functions should be compiled, regardless of profile
  // feedback. Otherwise compilation occurs only when profile feedback deems
  // it beneficial.
  bool compile_always = false;

  // The bisection fuel; this is decremented on every compilation request;
  // when it reaches zero no more compilation requests are issued. If -1,
  // bisection is disabled and this option has no effect.
  int64_t bisection_fuel = -1;

  // If true, when bisection fuel reaches zero, log the last function that was
  // compiled.
  bool bisection_verbose = false;

  // The capacity of the channel communicating compilation requests to the
  // background compiler.
  int64_t channel_capacity = 256;

  // The number of bytecode instructions given to ProfileEvent() before
  // a profiling event is taken.
  //
  // Profiling is a two-stage process where every
  // profile_bytecode_instructions_interval instructions, the current
  // PyCodeObject is inspected and `hotness_threshold` decremented; if
  // hotness_threshold reaches zero, the object will be optimized. This
  // two-stage process reduces the frequency of accessing code metadata.
  //
  // TODO: Could this be automatically adjusted to deliver a constant
  // frequency?
  int64_t profile_bytecode_instruction_interval = 10000;

  // The threshold at which optimization will occur. A per-code counter is
  // incremented every time ProfileEvent(code) occurs. Once the counter reaches
  // this threshold, the code object will be compiled.
  int64_t hotness_threshold = 2;

  // If true, event counters will be emitted in generated code and will be
  // dumped to S6_VLOG(1) when the Oracle exits.
  bool enable_event_counters = false;

  // If true, enable insertion of unbox/box instructions to perform successive
  // arithmetic operations using machine instructions.
  bool enable_unboxing_optimization = false;

  // If true, the refcount analysis will crash S6 on any detected reference
  // ownership manipulation errors. Otherwise such an error will just abort
  // some related optimisation and continue to compile the function.
  bool harden_refcount_analysis = false;

  // If true, add tracing instructions to function entry and exit points to
  // enable function call trace events to be recorded.
  bool enable_function_tracing = false;

  // If non-empty, a directory in which a file will be written with a unique
  // name containing the event counters proto.
  //
  // This option implies `enable_event_counters`.
  std::string event_counters_dir = "";
};

class Oracle : public ClassManager::TypeModificationClient {
 public:
  explicit Oracle(OracleDebugOptions debug_options = {});

  // Sets the singleton oracle, which must always be heap allocated.
  static void SetSingleton(Oracle* oracle) { singleton_ = oracle; }

  // Returns the singleton Oracle, which is always heap allocated (or nullptr).
  static Oracle* Get() { return singleton_; }

  // Move constructor moves code_objects_ and metadatas_ and creates a new
  // thread + channel.
  Oracle(Oracle&& other);

  // Virtual because we have subclasses in OracleTest.
  ~Oracle() override;

  // Initializes the background compiler thread. Must be called before any other
  // functions.
  void Initialize();

  // Signals that a periodic profiling event has occurred and `frame` was
  // executing at the time. `num_bytecodes` is an approximate count of the
  // number of bytecode instructions elapsed since the last ProfileEvent call.
  //
  // This call is equivalent to:
  //   int64_t* counter = GetProfileCounter();
  //   *counter -= num_bytecodes;
  //   if (*counter <= 0) ProfileEvent(frame, 0);
  void ProfileEvent(PyFrameObject* frame, int64_t num_bytecodes);

  // Returns the profile counter. This should be decremented for every CPython
  // bytecode executed. When it reaches zero, call ProfileEvent.
  int64_t* GetProfileCounter();

  // Returns the current CodeObject from a PyCodeObject, or nullptr if there is
  // no compiled code.
  //
  // Note: Always use this function as opposed to directly reading the
  // `current_code_object` member of Metadata. This function can compile
  // on-demand for debugging purposes.
  //
  // The `frame` argument is used for context to select a specialization.
  CodeObject* GetCodeObject(PyCodeObject* code, PyFrameObject* frame = nullptr);

  // If `code` is not yet compiled, forces a synchronous compilation and returns
  // the result of the compilation.
  //
  // If `code` has already been (attempted to be) compiled, returns the
  // compilation status.
  absl::Status ForceCompilation(PyCodeObject* code);

  // Drains the compilation request queue and denies any new compilation
  // requests. Requests will be denied until AllowCompilationRequests() is
  // called.
  void DenyCompilationRequestsAndDrainQueue();

  // Allows new compilation requests after having been denied by
  // DenyCompilationRequestsAndDrainQueue().
  void AllowCompilationRequests();

  // Implements the ClassManager::TypeModificationClient interface.
  void WillModifyType() override { DenyCompilationRequestsAndDrainQueue(); }
  void DidModifyType() override { AllowCompilationRequests(); }

  // Obtains the JitAllocator used to allocate all code.
  static JitAllocator& GetAllocator();

 protected:
  // The type of a compilation request.
  struct CompilationRequest {
    CompilationRequest(Metadata* metadata, Function function)
        : metadata(metadata), function(std::move(function)) {}

    // The code to be compiled.
    Metadata* metadata;

    // The function object to compile into.
    Function function;
  };

  // The compilation thread function. This is virtual to allow interposing a
  // testing function.
  virtual void CompileFunctions();

  // Compiles a single function. This is virtual to allow interposing a testing
  // function. CALLED BY: Compiler thread.
  virtual absl::StatusOr<std::unique_ptr<CodeObject>> Compile(
      PyCodeObject* code, Metadata* metadata, Function function);

  // Issues a single compilation request to CompileFunctions.
  // CALLED BY: Main thread.
  void RequestCompilation(PyCodeObject* code, Metadata* metadata);

  // All code objects in existence.
  std::vector<std::unique_ptr<CodeObject>> code_objects_ ABSL_GUARDED_BY(mu_);

  // All Metadatas that could possibly hold a reference to an object in
  // `code_objects_`.
  absl::flat_hash_set<Metadata*> metadatas_ ABSL_GUARDED_BY(mu_);

  // The channel used to communicate CompilationRequests to the compiler thread.
  // WRITTEN BY: Main thread
  // READ BY: Compiler thread.
  thread::Channel<std::unique_ptr<CompilationRequest>> channel_;

  // The debugging options. These are immutable as they are read by multiple
  // threads.
  // ACCESSED BY: Main thread, Compiler thread.
  const OracleDebugOptions debug_options_;

  // All outstanding compilation requests that have not yet been completed.
  absl::flat_hash_set<Metadata*> outstanding_compilation_requests_
      ABSL_GUARDED_BY(mu_);

  // Condition variable signalled when a compilation completes.
  absl::CondVar compilation_completed_;

  // When true, compilation requests must be denied.
  bool deny_compilation_requests_ ABSL_GUARDED_BY(mu_) = false;

  // Guards outstanding_compilation_requests_ and compilation_completed_;
  absl::Mutex mu_;

  // The thread in which the compiler thread runs.
  //
  // We use a unique_ptr because this thread object may have to be recreated if
  // the process forks.
  std::unique_ptr<std::thread> compiler_thread_;

  // The time the last profile event was received. This is only used if
  // S6_VLOG_IS_ON(2)
  absl::Time last_profile_event_ ABSL_GUARDED_BY(mu_);

  // The singleton Oracle, used by the interpreter and evaluator.
  static Oracle* singleton_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_ORACLE_H_
