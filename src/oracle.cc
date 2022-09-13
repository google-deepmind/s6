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

#include "oracle.h"

#include <Python.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "allocator.h"
#include "asmjit/asmjit.h"
#include "classes/class.h"
#include "classes/class_manager.h"
#include "code_generation/code_generator.h"
#include "code_generation/register_allocator.h"
#include "code_generation/thunk_generator.h"
#include "code_generation/trace_register_allocator.h"
#include "code_object.h"
#include "core_util.h"
#include "event_counters.h"
#include "global_intern_table.h"
#include "metadata.h"
#include "strongjit/deoptimization.h"
#include "strongjit/function.h"
#include "strongjit/ingestion.h"
#include "strongjit/optimizer.h"
#include "strongjit/util.h"
#include "type_feedback.h"
#include "utils/inlined_bit_vector.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

ABSL_FLAG(bool, testing_randomly_deoptimize, false,
          "Randomly deoptimizes edges in the control flow graph, causing "
          "aggressive deoptimization. For testing only.");

namespace deepmind::s6 {
namespace {

// Tracks all failure modes for trying to call functions that failed to compile
// in a program already. Key is the function name, while the value is a
// (mode, count) pair.
NoDestructor<absl::flat_hash_map<std::string, std::pair<std::string, int64_t>>>
    call_failure_modes;

// The global allocator for JIT objects. We set the block size quite large
// (50MB) so that JITted objects are physically close to each other.
NoDestructor<JitAllocator> allocator(50 * 1024 * 4096);

// Returns a string of 80 equals signs for logging separation: ================
std::string SeparatorLine() { return std::string(80, '='); }

// Returns true if this Metadata should be specialized based on its first
// argument type.
bool IsProfitableToSpecialize(Metadata* metadata) {
  for (const auto& cls : metadata->type_feedback()) {
    if (cls.empty()) continue;
    // It is sufficient to only look at the first class distribution per type
    // feedback.
    auto summary = cls[0].Summarize();
    // If we don't trust the summary then we don't want to specialize.
    if (!summary.stable()) continue;
    // If the type feedback is poly or megamorphic, it's probably worthwhile
    // trying to specialize.
    if (summary.IsPolymorphic() || summary.IsMegamorphic()) return true;
  }
  return false;
}

}  // namespace

Oracle* Oracle::singleton_ = nullptr;

Oracle::Oracle(OracleDebugOptions debug_options)
    : channel_(debug_options.channel_capacity),
      debug_options_(debug_options),
      compiler_thread_(),  // no thread runs until init is called.
      last_profile_event_(absl::Now()) {
  *GetProfileCounter() = debug_options_.profile_bytecode_instruction_interval;
  ClassManager::Instance().SetTypeModificationClient(this);
}

Oracle::Oracle(Oracle&& other)
    : channel_(other.debug_options_.channel_capacity),
      debug_options_(other.debug_options_),
      compiler_thread_(),  // no thread runs until init is called.
      last_profile_event_(absl::Now()) {
  absl::MutexLock other_lock(&other.mu_);
  absl::MutexLock this_lock(&mu_);
  code_objects_ = std::move(other.code_objects_);
  metadatas_ = std::move(other.metadatas_);
  *GetProfileCounter() = debug_options_.profile_bytecode_instruction_interval;
  ClassManager::Instance().SetTypeModificationClient(this);
}

void Oracle::Initialize() {
  compiler_thread_ =
      absl::make_unique<std::thread>([this] { this->CompileFunctions(); });
}

Oracle::~Oracle() {
  channel_.writer().Close();
  compiler_thread_->join();

  if (S6_VLOG_IS_ON(1)) {
    absl::MutexLock lock(&mu_);
    // Accumulate all compilation statuses for all registered metadata objects.
    absl::flat_hash_map<std::string, int64_t> status_count;
    int64_t ok_count = 0;
    int64_t bad_count = 0;
    for (Metadata* metadata : metadatas_) {
      const absl::Status& status = metadata->compilation_status();
      if (status.ok()) {
        ++ok_count;
      } else {
        ++status_count[status.ToString()];
        ++bad_count;
      }
    }

    std::vector<std::pair<int64_t, std::string>> sorted_status_count;
    sorted_status_count.reserve(status_count.size());
    for (const auto& [msg, count] : status_count) {
      sorted_status_count.emplace_back(count, msg);
    }
    absl::c_sort(sorted_status_count, std::greater<>());

    // Vector is <call_count, func_name, error_msg>.
    std::vector<std::tuple<int64_t, std::string, std::string>>
        call_failure_modes_list;
    call_failure_modes_list.reserve(call_failure_modes->size());
    for (const auto& [func_name, mode_count_pair] : *call_failure_modes) {
      call_failure_modes_list.emplace_back(mode_count_pair.second, func_name,
                                           mode_count_pair.first);
    }
    absl::c_sort(call_failure_modes_list, std::greater<>());

    S6_LOG(INFO) << "S6 attempted to compile " << metadatas_.size()
                 << " functions";
    S6_LOG(INFO) << "                       " << ok_count
                 << " were compiled SUCCESSFULLY";
    S6_LOG(INFO) << "                       " << bad_count
                 << " FAILED to compile";

    if (!sorted_status_count.empty()) {
      S6_LOG(INFO) << SeparatorLine();
      for (auto [count, msg] : sorted_status_count) {
        S6_LOG(INFO) << count << " instances of: " << msg;
      }
    }

    if (!call_failure_modes_list.empty()) {
      S6_LOG(INFO) << SeparatorLine();
      S6_LOG(INFO) << "Function call frequency of functions that failed to "
                   << "compile:";
      for (const auto& [num, func_name, msg] : call_failure_modes_list) {
        S6_LOG(INFO) << absl::StrFormat("  %-5d %s: %s", num, func_name, msg);
      }
    }

    S6_LOG(INFO) << SeparatorLine();
  }

  if (!debug_options_.event_counters_dir.empty()) {
    EventCounters::Instance()
        .DumpToDirectory(debug_options_.event_counters_dir)
        .IgnoreError();
  }
}

JitAllocator& Oracle::GetAllocator() { return *allocator; }

int64_t* Oracle::GetProfileCounter() {
  static thread_local int64_t profile_counter =
      debug_options_.profile_bytecode_instruction_interval;
  return &profile_counter;
}

void Oracle::ProfileEvent(PyFrameObject* frame, int64_t num_bytecodes) {
  int64_t* profile_counter = GetProfileCounter();
  *profile_counter -= num_bytecodes;
  if (*profile_counter > 0) {
    return;
  }
  *profile_counter = debug_options_.profile_bytecode_instruction_interval;
  if (!frame) {
    // Called from generated code.
    return;
  }

  PyCodeObject* code = frame->f_code;
  Metadata* unspecialized_metadata = Metadata::Get(code);
  Metadata* metadata = unspecialized_metadata;
  if (code->co_argcount > 0) {
    metadata = metadata->SelectSpecialization(frame->f_localsplus[0]);
  }

  if (metadata->profile_count() == -1) {
    // This function is undergoing compilation at the moment. Ignore this event.
    return;
  }

  if (metadata->profile_count() >= debug_options_.hotness_threshold) {
    if (!metadata->completion_observed()) {
      // We haven't seen this function complete yet, so the type feedback will
      // be incomplete. It is not a good idea to compile it yet.
      return;
    }

    constexpr int64_t kNumMaxSpecializations = 8;
    if (metadata == unspecialized_metadata && code->co_argcount > 0 &&
        frame->f_localsplus[0] &&
        metadata->specialization_count() < kNumMaxSpecializations &&
        IsProfitableToSpecialize(metadata)) {
      // We should specialize this function.
      Metadata* new_metadata = unspecialized_metadata->CreateSpecialization(
          frame->f_localsplus[0], *allocator);
      S6_VLOG(3) << "Requesting specialization for " << new_metadata->name()
                 << " due to profiling";
      metadata->ResetTypeFeedback();
      metadata->set_completion_observed(false);
      return;
    }

    if (metadata->current_code_object().load()) {
      // This program is already compiled. Nothing to do.
      return;
    }
    S6_VLOG(3) << "Requesting compilation for " << metadata->name()
               << " due to profiling";
    // Set the profile_count to -1 temporarily. The compiler thread will reset
    // it to zero when compilation is complete.
    metadata->set_profile_count(-1);
    RequestCompilation(code, metadata);
  } else {
    metadata->set_profile_count(metadata->profile_count() + 1);
  }
}

CodeObject* Oracle::GetCodeObject(PyCodeObject* code, PyFrameObject* frame) {
  Metadata* metadata = Metadata::Get(code);

  if (frame && code->co_argcount > 0) {
    metadata = metadata->SelectSpecialization(frame->f_localsplus[0]);
  }

  CodeObject* current = metadata->current_code_object().load();
  if (current || !debug_options_.compile_always) {
    // This is the normal case. We just return the current CodeObject.
    return current;
  }

  // This object is not compiled and we must compile. Let's just check it hasn't
  // been attempted already.
  if (!metadata->compilation_status().ok()) {
    if (S6_VLOG_IS_ON(1)) {
      ++(*call_failure_modes)[PyObjectToString(code->co_name)].second;
    }
    return nullptr;
  }
  RequestCompilation(code, metadata);

  // Re-fetch the current code object; we may be compiling synchronously.
  return metadata->current_code_object().load();
}

absl::Status Oracle::ForceCompilation(PyCodeObject* code) {
  Metadata* metadata = Metadata::Get(code);
  if (!metadata->compilation_status().ok()) {
    return metadata->compilation_status();
  }
  if (metadata->current_code_object().load()) {
    return absl::OkStatus();
  }

  RequestCompilation(code, metadata);

  if (debug_options_.synchronous) {
    return metadata->compilation_status();
  }

  absl::MutexLock lock(&mu_);
  while (outstanding_compilation_requests_.contains(metadata)) {
    compilation_completed_.Wait(&mu_);
  }
  return metadata->compilation_status();
}

void Oracle::RequestCompilation(PyCodeObject* code, Metadata* metadata) {
  if (!metadata->compilation_status().ok()) {
    // Program has been compiled before and has failed compilation; don't try
    // again!
    S6_VLOG(2) << "Requested to recompile function that failed to compile!";
    return;
  }

  // We are in the main thread, so reading metadata is safe here.
  Function function(metadata->name());
  S6_CHECK_OK(SnapshotTypeFeedback(metadata->type_feedback(), function));
  auto request =
      absl::make_unique<CompilationRequest>(metadata, std::move(function));

  // We have two paths: synchronous and asynchronous. In synchronous mode we
  // must wait for the compilation to complete, and we know it's impossible to
  // head-of-line block (which simplifies code). In asynchronous mode we never
  // wait, which again simplifies code because we can head-of-line block without
  // having to reacquire mu_.

  if (debug_options_.synchronous) {
    absl::MutexLock lock(&mu_);
    if (metadatas_.insert(metadata).second) {
      // We took a handle to `metadata`, so ensure it does not get deleted.
      Py_INCREF(code);
    }
    bool request_inserted =
        outstanding_compilation_requests_.insert(metadata).second;
    S6_CHECK(request_inserted)
        << "In synchronous mode, requests must always be inserted";
    channel_.writer().Write(std::move(request));
    compilation_completed_.Wait(&mu_);
    if (S6_VLOG_IS_ON(1) && !metadata->compilation_status().ok()) {
      (*call_failure_modes)[PyObjectToString(code->co_name)] =
          std::make_pair(metadata->compilation_status().ToString(), 1);
    }
  } else {
    bool request_inserted;
    {
      absl::MutexLock lock(&mu_);
      if (metadatas_.insert(metadata).second) {
        // We took a handle to `metadata`, so ensure it does not get deleted.
        Py_INCREF(code);
      }
      request_inserted =
          outstanding_compilation_requests_.insert(metadata).second;
    }

    if (request_inserted) {
      // Write with mu_ released in case we head-of-line block.
      channel_.writer().Write(std::move(request));
    }
  }
}

void Oracle::CompileFunctions() {
  thread::Reader<std::unique_ptr<CompilationRequest>>& input =
      channel_.reader();
  int64_t bisection_fuel = debug_options_.bisection_fuel;
  while (true) {
    std::unique_ptr<CompilationRequest> request;
    if (!input.Read(&request)) {
      // Channel is closed.
      return;
    }
    // This is the live version of request.metadata that must be thread-safely
    // mutated.
    Metadata* live_metadata = request->metadata;
    PyCodeObject* code = request->metadata->py_code_object();

    auto cleanup = absl::MakeCleanup([&]() {
      {
        absl::MutexLock lock(&mu_);
        outstanding_compilation_requests_.erase(live_metadata);
      }
      compilation_completed_.Signal();
    });

    if (bisection_fuel == 0) {
      // Fuel has run out!
      live_metadata->set_compilation_status(absl::CancelledError(
          "not attempting compilation because fuel ran out"));
      continue;
    }
    if (bisection_fuel > 0) {
      --bisection_fuel;
    }

    std::string name(request->function.name());
    absl::StatusOr<std::unique_ptr<CodeObject>> code_object_or =
        Compile(code, live_metadata, std::move(request->function));
    if (!code_object_or.ok()) {
      live_metadata->set_compilation_status(code_object_or.status());
      EventCounters::Instance().Add(
          absl::StrCat("oracle.compilation_failure(",
                       code_object_or.status().message(), ")"),
          1);
      continue;
    }
    EventCounters::Instance().Add("oracle.compilation_success", 1);
    std::unique_ptr<CodeObject> code_object = std::move(code_object_or.value());

    if (bisection_fuel == 0 && debug_options_.bisection_verbose) {
      S6_LOG(WARNING) << SeparatorLine();
      S6_LOG(WARNING) << "Last compiled function before fuel ran out: " << name;
      S6_LOG(WARNING) << SeparatorLine();
      for (BytecodeInstruction& inst : ExtractInstructions(code))
        S6_LOG(WARNING) << inst.ToString();
      S6_LOG(WARNING) << SeparatorLine();
      S6_LOG_LINES(WARNING, code_object->ToString());
      S6_LOG(WARNING) << SeparatorLine();
    }

    auto& thunk = live_metadata->GetOrCreateThunk(*allocator);
    for (Class* cls : code_object->function().ConsumeReliedUponClasses()) {
      CodeObject* code_object_ptr = code_object.get();
      cls->AddListener([live_metadata, code_object_ptr]() {
        // If the code object is already deoptimized, nothing to do.
        if (code_object_ptr->deoptimized()) return;

        // Otherwise deoptimize.
        EventCounters::Instance().Add("oracle.listener_deoptimize", 1);
        live_metadata->Deoptimize();
        code_object_ptr->MarkDeoptimized();
      });
    }

    auto fast_fn = code_object->GetFastBody();
    if (fast_fn) {
      auto status = thunk.SetTarget(live_metadata->specialized_class_id(),
                                    reinterpret_cast<void*>(fast_fn));
      if (!status.ok()) {
        live_metadata->set_compilation_status(status);
        EventCounters::Instance().Add(
            absl::StrCat("oracle.compilation_failure(", status.message(), ")"),
            1);
        continue;
      }
    }

    absl::MutexLock lock(&mu_);
    code_objects_.push_back(std::move(code_object));
    live_metadata->current_code_object().store(code_objects_.back().get());
    live_metadata->set_profile_count(0);
  }
}

absl::StatusOr<std::unique_ptr<CodeObject>> Oracle::Compile(PyCodeObject* code,
                                                            Metadata* metadata,
                                                            Function function) {
  CodeFlagsValidator flags_validator(code->co_flags);
  flags_validator.Requires(CO_NEWLOCALS, "CO_NEWLOCALS")
      .Requires(CO_OPTIMIZED, "CO_OPTIMIZED")
      .Allows(CO_NOFREE | CO_NESTED | CO_GENERATOR)
      .Allows(CO_VARARGS | CO_VARKEYWORDS);
  if (absl::Status status = flags_validator.Validate(); !status.ok()) {
    return status;
  }

  bool is_generator = code->co_flags & CO_GENERATOR;
  absl::Span<asmjit::x86::Reg const> registers = kAllocatableRegisters;
  if (is_generator) {
    registers = kAllocatableRegistersForGenerator;
  }
  if (code->co_kwonlyargcount != 0) {
    return absl::UnimplementedError(
        "Can't compile function with keyword-only arguments");
  }

  auto bytecode_insts = ExtractInstructions(code);
  int64_t num_arguments = code->co_argcount;
  if (code->co_flags & CO_VARARGS) ++num_arguments;
  if (code->co_flags & CO_VARKEYWORDS) ++num_arguments;

  S6_RETURN_IF_ERROR(IngestProgram(bytecode_insts, function, code->co_nlocals,
                                   num_arguments, metadata->except_observed()));
  S6_RETURN_IF_ERROR(OptimizeFunction(
      function, code,
      {.use_event_counters = debug_options_.enable_event_counters,
       .enable_unboxing_optimization =
           debug_options_.enable_unboxing_optimization,
       .enable_function_tracing = debug_options_.enable_function_tracing,
       .harden_refcount_analysis = debug_options_.harden_refcount_analysis}));

  if (absl::GetFlag(FLAGS_testing_randomly_deoptimize)) {
    std::default_random_engine rng(43);
    S6_RETURN_IF_ERROR(StressTestByDeoptimizingRandomly(function, rng));
  }
  S6_RETURN_IF_ERROR(MarkDeoptimizedBlocks(function));
  S6_RETURN_IF_ERROR(RewriteFunctionForDeoptimization(function));

  SplitCriticalEdges(function);
  S6_ASSIGN_OR_RETURN(auto ra,
                      AllocateRegistersWithTrace(
                          function, {.allocatable_registers = registers}));
  S6_ASSIGN_OR_RETURN(auto uniq_ptr,
                      GenerateCode(std::move(function), *ra, bytecode_insts,
                                   *allocator, code, metadata));
  S6_CHECK(uniq_ptr->GetPyFrameBody());
  return uniq_ptr;
}

void Oracle::DenyCompilationRequestsAndDrainQueue() {
  absl::MutexLock lock(&mu_);
  S6_CHECK(!deny_compilation_requests_);
  deny_compilation_requests_ = true;
  while (!outstanding_compilation_requests_.empty()) {
    compilation_completed_.Wait(&mu_);
  }
}

void Oracle::AllowCompilationRequests() {
  absl::MutexLock lock(&mu_);
  S6_CHECK(deny_compilation_requests_);
  deny_compilation_requests_ = false;
}

// We provide this entry point for interpreter.cc. We have a dependency cycle
// that is hard to break, so the interpreter externs this function as weak
// without declaring a dependency on Oracle.
void OracleProfileEvent(PyFrameObject* frame, int64_t num_bytecodes) {
  return Oracle::Get()->ProfileEvent(frame, num_bytecodes);
}

}  // namespace deepmind::s6
