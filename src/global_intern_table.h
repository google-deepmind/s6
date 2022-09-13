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

#ifndef THIRD_PARTY_DEEPMIND_S6_GLOBAL_INTERN_TABLE_H_
#define THIRD_PARTY_DEEPMIND_S6_GLOBAL_INTERN_TABLE_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "utils/keyed_intern_table.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {

// Singleton maintaining a global table of interned null-terminated strings.
//
// Strings in the global intern table are guaranteed to be remain allocated at
// a constant address for the lifetime of the process. This makes them useable
// as trace event names.
class GlobalInternTable {
 public:
  // Fetch the global intern table singleton.
  static GlobalInternTable& Instance();

  // Opaque wrapper for an interned string.
  class InternedString {
   public:
    InternedString() = default;

    absl::string_view get() const { return absl::string_view(value_); }

    // Returns a non-owning raw pointer to the underlying string. This pointer
    // and the memory pointed to will not change or be deallocated in the
    // lifetime of the process, so it is safe to be used as an immediate value
    // in generated code.
    const char* AsPtr() const { return value_; }

   private:
    friend class GlobalInternTable;

    // Only GlobalInternTable can create InternedString instances with non-null
    // values.
    explicit InternedString(const char* value) : value_(value) {}

    // Raw pointer to a null terminated string owned by `intern_table_`.
    const char* value_ = nullptr;
  };

  InternedString Intern(absl::string_view str) {
    absl::MutexLock lock(&mtx_);
    return InternedString(intern_table_->Insert(str).value);
  }

 private:
  GlobalInternTable()
      : intern_table_(std::make_unique<KeyedInternTable<uint32_t>>()) {}
  ~GlobalInternTable() = default;

  friend class NoDestructor<GlobalInternTable>;

  GlobalInternTable(const GlobalInternTable&) = delete;
  GlobalInternTable(const GlobalInternTable&&) = delete;
  GlobalInternTable& operator=(const GlobalInternTable&) = delete;

  std::unique_ptr<KeyedInternTable<uint32_t>> intern_table_
      ABSL_GUARDED_BY(mtx_);
  absl::Mutex mtx_;
};

inline bool operator==(GlobalInternTable::InternedString a,
                       GlobalInternTable::InternedString b) {
  return a.AsPtr() == b.AsPtr();
}

template <typename H>
H AbslHashValue(H h, const GlobalInternTable::InternedString& str) {
  return H::combine(std::move(h), reinterpret_cast<intptr_t>(str.AsPtr()));
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_GLOBAL_INTERN_TABLE_H_
