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

#ifndef THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_MANAGER_H_
#define THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_MANAGER_H_

#include <Python.h>

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "classes/util.h"
#include "utils/keyed_intern_table.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

// Forward declare.
class Attribute;
class Class;

// The maximum valid class ID.
constexpr int64_t kMaxClassId = (1UL << kNumClassIdBits) - 1;

// All strings used as attribute names must be interned. The use of an intern
// table allows simple pointer comparisons for attribute lookups but also
// cheap mapping from an (interned) PyUnicodeObject.
//
// An InternedString is just a string_view with a protected constructor and
// a simpler hash and equality function.
class InternedString : public absl::string_view {
 public:
  InternedString() {}

 private:
  friend class ClassManager;

  explicit InternedString(absl::string_view s) : absl::string_view(s) {}

  template <typename H>
  friend H AbslHashValue(H h, const InternedString& s) {
    return H::combine(std::move(h), s.data());
  }

  friend bool operator==(const InternedString& self,
                         const InternedString& other) {
    return self.data() == other.data();
  }
};

// Manages the allocation of Classes, Class IDs and Attributes. Classes are
// never deleted.
//
// Also holds the string intern table, which allows all absl::string_views used
// within Classes and Attributes to be compared by data() pointer.
class ClassManager {
 public:
  // A ClassManager may have one and only one client that is informed when
  // types are about to be modified.
  class TypeModificationClient {
   public:
    virtual ~TypeModificationClient();
    virtual void WillModifyType() = 0;
    virtual void DidModifyType() = 0;
  };

  ~ClassManager();

  // Returns a Class, given its ID. Returns nullptr on failure.
  Class* GetClassById(int64_t id);
  const Class* GetClassById(int64_t id) const;

  // Returns the global singleton instance.
  static ClassManager& Instance();

  // Globals and builtins dicts have special Classes associated with them. Each
  // Globals dict has a single class, and this also covers its __builtins__.
  //
  // This function gets the unique Class for `globals` or creates it if it
  // hasn't been seen before. Note that because of layering this function does
  // NOT check if `globals` already has a valid class ID assigned to it.
  Class* GetClassForGlobals(PyDictObject* globals) {
    auto find_it = globals_classes_.find(globals);
    if (find_it != globals_classes_.end()) {
      return find_it->second;
    }
    return nullptr;
  }

  InternedString InternString(absl::string_view str)
      ABSL_LOCKS_EXCLUDED(table_mu_);
  InternedString InternString(PyObject* unicode_object)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Sets the type modification client. The client is not owned by the class
  // manager. Setting this will replace any existing client.
  void SetTypeModificationClient(TypeModificationClient* client) {
    absl::MutexLock lock(&mu_);
    type_modification_client_ = client;
  }

  // Runs the given function, which may call Class::TypeHasBeenModified.
  template <typename F, typename... Args>
  auto SafelyRunTypeHasBeenModified(F&& f, const Args&... args)
      -> decltype(f(args...)) {
    // Obtain the client with the lock held, but don't call `f` with the lock
    // held - that is the way to deadlock.
    TypeModificationClient* client = nullptr;
    {
      absl::MutexLock lock(&mu_);
      client = type_modification_client_;
    }
    if (client) client->WillModifyType();
    f(args...);
    if (client) client->DidModifyType();
  }

 private:
  // Class may use ClassManager to create new Classes.
  friend class Class;

  absl::StatusOr<int64_t> AllocateNewClassId() ABSL_SHARED_LOCKS_REQUIRED(mu_);

  template <typename... Args>
  absl::StatusOr<Class*> CreateClass(Args&&... args) {
    absl::WriterMutexLock lock(&mu_);
    S6_ASSIGN_OR_RETURN(int64_t id, AllocateNewClassId());
    Class* cls =
        absl::make_unique<Class>(id, std::forward<Args>(args)...).release();
    classes_.push_back(cls);
    return cls;
  }

  // Holds the canonical storage for all InternedStrings.
  KeyedInternTable<uint32_t> intern_table_ ABSL_GUARDED_BY(table_mu_);
  mutable absl::Mutex table_mu_;

  // Holds a map from (interned) unicode object to InternedString.
  std::unordered_map<PyObject*, InternedString> interned_unicode_objects_
      ABSL_GUARDED_BY(mu_);
  std::vector<Class*> classes_ ABSL_GUARDED_BY(mu_);
  TypeModificationClient* type_modification_client_ ABSL_GUARDED_BY(mu_) =
      nullptr;
  // Guarded by the GIL.
  absl::flat_hash_map<PyDictObject*, Class*> globals_classes_;
  mutable absl::Mutex mu_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CLASSES_CLASS_MANAGER_H_
