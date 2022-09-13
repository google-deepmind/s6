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

#ifndef THIRD_PARTY_DEEPMIND_S6_METADATA_H_
#define THIRD_PARTY_DEEPMIND_S6_METADATA_H_

#include <Python.h>
#include <frameobject.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/declare.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "allocator.h"
#include "classes/util.h"
#include "code_generation/thunk_generator.h"
#include "code_object.h"
#include "core_util.h"
#include "runtime/interpreter_stub.h"
#include "type_feedback.h"

namespace deepmind::s6 {

// Metadata holds information about a PyCodeObject. This is the primary way of
// obtaining S6-specific information about a particular code object.
//
// The Metadata holds the type feedback vector, the current compiled version of
// the code, and the "thunk" that is the S6 fast entry point.
//
// A PyCodeObject may have more than one Metadata associated with it. There
// will always be one Metadata for an unspecialized code object, and potentially
// others specialized on the type of the first argument.
class Metadata {
 public:
  // Returns metadata for a given code object.
  static Metadata* Get(PyCodeObject* co);

  Metadata(const Metadata& other)
      : type_feedback_(other.type_feedback_),
        current_code_object_(other.current_code_object_.load()),
        profile_count_(other.profile_count_),
        name_(other.name_),
        co_(other.co_),
        thunk_generator_(other.thunk_generator_),
        completion_observed_(other.completion_observed_),
        except_observed_(other.except_observed_) {}

  Metadata(Metadata&& other)
      : type_feedback_(std::move(other.type_feedback_)),
        current_code_object_(other.current_code_object_.load()),
        profile_count_(other.profile_count_),
        name_(std::move(other.name_)),
        co_(other.co_),
        thunk_generator_(std::move(other.thunk_generator_)),
        completion_observed_(std::move(other.completion_observed_)),
        except_observed_(std::move(other.except_observed_)) {}

  // Allows access to the private default constructor for creation of
  // placeholder metadatas for unit tests.
  static Metadata CreateForTesting() { return Metadata(); }

  ClassDistribution& GetTypeFeedback(PcValue pc, int64_t operand_index = 0) {
    auto& distributions = type_feedback_[pc.AsIndex()];
    if (distributions.size() <= operand_index)
      distributions.resize(operand_index + 1);
    return distributions[operand_index];
  }

  // Returns the offset given to _PyCode_GetExtra().
  static int GetPyCodeExtraOffset();

  std::atomic<CodeObject*>& current_code_object() {
    return current_code_object_;
  }

  // Gets, and creates if necessary, the entry thunk for this metadata.
  ThunkGenerator& GetOrCreateThunk(JitAllocator& allocator) {
    absl::MutexLock lock(&mu_);
    if (!thunk_generator_) {
      thunk_generator_ = std::make_shared<ThunkGenerator>(
          allocator, reinterpret_cast<void*>(&StrongjitInterpreterStub), name_);
    }
    return *thunk_generator_;
  }

  ThunkGenerator* GetThunk() {
    absl::MutexLock lock(&mu_);
    return thunk_generator_.get();
  }

  std::atomic<CodeObject::FastAbiFunctionType>& current_entry_point() {
    return current_entry_point_;
  }

  // Returns the current compilation status. If this is non-OK, this object has
  // been attempted to be compiled and has failed compilation. Thread-safe.
  absl::Status compilation_status() const {
    absl::MutexLock lock(&mu_);
    return compilation_status_;
  }
  // Sets the current compilation status. If this is non-OK, this object has
  // been attempted to be compiled and has failed compilation. Thread-safe.
  void set_compilation_status(absl::Status status) {
    absl::MutexLock lock(&mu_);
    compilation_status_ = status;
    if (!status.ok()) has_compilation_failed_.store(true);
  }

  // Returns true if compilation has failed for this function. This is thread
  // safe and does not require a lock.
  bool has_compilation_failed() const { return has_compilation_failed_.load(); }

  // Returns the current profile count, which is an opaque integer for use by
  // the Oracle. This function is thread-safe.
  int64_t profile_count() const {
    absl::MutexLock lock(&mu_);
    return profile_count_;
  }

  // Sets the current profile count, which is an opaque integer for use by the
  // Oracle. This function is thread-safe.
  void set_profile_count(int64_t profile_count) {
    absl::MutexLock lock(&mu_);
    profile_count_ = profile_count;
  }

  // Sets that this function has been observed to complete.
  void set_completion_observed(bool b) { completion_observed_ = b; }

  // Returns true if this function has been observed to complete by the
  // interpreter.
  bool completion_observed() const { return completion_observed_; }

  // Sets that this function has been observed to except.
  void set_except_observed(bool b) { except_observed_ = b; }

  // Returns true if this function has been observed to except by the
  // interpreter.
  bool except_observed() const { return except_observed_; }

  const absl::FixedArray<absl::InlinedVector<ClassDistribution, 1>>&
  type_feedback() const {
    return type_feedback_;
  }
  absl::FixedArray<absl::InlinedVector<ClassDistribution, 1>>& type_feedback() {
    return type_feedback_;
  }

  absl::string_view name() const { return name_; }

  // Given the type of the first argument to the function, returns the Metadata
  // for the best existing specialization. If no applicable specializations are
  // found, returns this.
  Metadata* SelectSpecialization(PyObject* first_argument) {
    if (!first_argument) return this;
    int64_t class_id = GetClassIdFromType(Py_TYPE(first_argument));
    for (const auto& [key, meta] : specializations_) {
      if (key == class_id) return meta.get();
    }
    return this;
  }

  // Creates a specialization for the given type of the first argument, or
  // returns the existing metadata if it existed.
  Metadata* CreateSpecialization(PyObject* first_argument,
                                 JitAllocator& allocator) {
    Metadata* existing = SelectSpecialization(first_argument);
    if (existing != this) return existing;
    int64_t class_id = GetClassIdFromType(Py_TYPE(first_argument));
    // Ensure we have a thunk first before creating the specialization.
    GetOrCreateThunk(allocator);
    specializations_.emplace_back(
        class_id, new Metadata(co_, Py_TYPE(first_argument)->tp_name,
                               thunk_generator_, class_id));
    return specializations_.back().second.get();
  }

  // Returns the class ID that this Metadata is specialized for. If this is not
  // specialized, returns zero.
  int64_t specialized_class_id() const { return specialized_class_id_; }

  // Returns the number of specializations.
  int64_t specialization_count() const { return specializations_.size(); }

  // Returns the PyCodeObject that this Metadata represents.
  PyCodeObject* py_code_object() const { return co_; }

  // Resets all type feedback to empty.
  void ResetTypeFeedback() {
    type_feedback_.fill(absl::InlinedVector<ClassDistribution, 1>());
  }

  // Deoptimizes the target compiled function. Only deoptimizes this specific
  // specialization.
  // This should only be called from the main thread.
  void Deoptimize() {
    current_code_object_.store(nullptr);
    if (ThunkGenerator* thunk = GetThunk()) {
      S6_CHECK_OK(thunk->SetTargetToInterpreter(specialized_class_id()));
    }
  }

 private:
  // freefunc callback for _PyEval_RequestCodeExtraIndex.
  static void FreeMetadata(void* meta);

  Metadata() : type_feedback_(0) {}

  explicit Metadata(PyCodeObject* co, const char* specialization = nullptr,
                    std::shared_ptr<ThunkGenerator> thunk_generator = nullptr,
                    int64_t specialized_class_id = 0)
      : type_feedback_(PyBytes_GET_SIZE(co->co_code) / sizeof(_Py_CODEUNIT)),
        co_(co),
        thunk_generator_(thunk_generator),
        specialized_class_id_(specialized_class_id) {
    name_ = GetObjectAsCheapString(co->co_name);
    if (specialization) {
      name_ = absl::StrCat(name_, ".", specialization);
    }
  }

  // The type feedback can be per-operand, but in most cases there is only one
  // type feedback per bytecode offset.
  //
  // Using absl::InlinedVector here adds an extra 8 bytes per entry.
  absl::FixedArray<absl::InlinedVector<ClassDistribution, 1>> type_feedback_;

  // The currently active CodeObject, if any. This can be nullptr if code has
  // not been (successfully) compiled.
  //
  // Note: this field is accessed atomically from generated code, which is why
  // it is atomic rather than guarded by mu_.
  std::atomic<CodeObject*> current_code_object_ = nullptr;

  // The fast ABI entry point of the currently active CodeObject. This is
  // initialized to StrongjitInterpreterStub; a stub function that can accept
  // the Strongjit fast calling convention and forwards to _PyFunction_FastCall.
  //
  // Note: this field is accessed atomically from generated code, which is why
  // it is atomic rather than guarded by mu_.
  std::atomic<CodeObject::FastAbiFunctionType> current_entry_point_ =
      &StrongjitInterpreterStub;

  // The profile count. This is for the Oracle's use only, to decide when to
  // optimize.
  int64_t profile_count_ ABSL_GUARDED_BY(mu_) = 0;

  // If the object has failed to compile, this holds the failure status. This is
  // updated by the Oracle's compilation thread, so is thread-safe.
  absl::Status compilation_status_ ABSL_GUARDED_BY(mu_);

  // A simple boolean that is true if the program has failed to compile. This
  // is redundant with compilation_status_ but can be read without acquiring a
  // lock. This only ever transitions from false to true.
  std::atomic<bool> has_compilation_failed_ = false;

  // The co_name of the PyCodeObject.
  std::string name_;

  // The PyCodeObject.
  PyCodeObject* co_;

  // The thunk for entry to generated code. This is shared between all
  // specializations of a PyCodeObject.
  std::shared_ptr<ThunkGenerator> thunk_generator_;

  // The class ID that this Metadata is specialized over, or zero if it is not
  // specialized.
  int64_t specialized_class_id_ = 0;

  // The specializations available for this Metadata. The key is the class ID of
  // the type of the first argument to the function.
  //
  // Note: NOT the class ID of the first argument! For efficiency, we don't
  // check the dictionary of the argument, only the type's class ID.
  std::vector<std::pair<int64_t, std::unique_ptr<Metadata>>> specializations_;

  // Whether the interpreter has observed this function completing. This is used
  // to determine if we have enough type feedback to compile.
  bool completion_observed_ = false;

  // Whether the interpreter has observed this function taking an exception.
  bool except_observed_ = false;

  mutable absl::Mutex mu_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_METADATA_H_
