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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYTHON_FUNCTION_INFO_TABLE_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYTHON_FUNCTION_INFO_TABLE_H_

#include <Python.h>

#include <cstdint>

#include "absl/container/node_hash_map.h"
#include "core_util.h"
#include "event_counters.h"
#include "global_intern_table.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {

// Provides a global map from PyCodeObject instances to function information.
// This allows us to record information such as the fully qualified name
// associated with a python code object at the time that that information is
// available.
//
// Although the singleton map itself must remain thread safe, we keep a thread
// local fixed size hash table as a cache, to skip acquiring a lock as often as
// possible.
class PythonFunctionInfoTable {
 public:
  static constexpr size_t kCacheSize = 10240;

  struct FunctionInfo {
    GlobalInternTable::InternedString qualified_name;
    GlobalInternTable::InternedString filename;
    int64_t line;
  };

  // Retrieve the singleton instance.
  static PythonFunctionInfoTable& Instance();

  // Look up the entry for a code object, creating an entry if it has not been
  // seen before.
  //
  // If this must create a new entry, then its qualified name is not available,
  // so we use the co_name member, which is not qualified.
  FunctionInfo& Lookup(const PyCodeObject* code) {
    size_t cache_elem = absl::Hash<const PyCodeObject*>{}(code) % kCacheSize;
    if (cache_[cache_elem] && cache_[cache_elem]->first == code) {
      ++*cache_hit_count_;
      return cache_[cache_elem]->second;
    }

    absl::MutexLock lock(&mu_);
    ++*cache_miss_count_;
    auto it = map_.find(code);
    if (it != map_.end()) {
      cache_[cache_elem] = &*it;
      return it->second;
    }
    return InsertLocked(code, GetObjectAsCheapString(code->co_name))->second;
  }

  // Interpose PyFunction_NewWithQualName to insert an entry with the correct
  // qualified name.
  static PyObject* NewFunctionWithQualName(PyObject* code, PyObject* globals,
                                           PyObject* qualified_name) {
    S6_DCHECK(PyCode_Check(code));
    S6_DCHECK(PyGILState_Check());
    PyObject* result =
        PyFunction_NewWithQualName(code, globals, qualified_name);
    if (result) {
      Instance().Insert(reinterpret_cast<const PyCodeObject*>(code),
                        GetObjectAsCheapStringRequiringGil(qualified_name));
    }
    return result;
  }

 private:
  // Uses node_hash_map because thread local caches rely on value_type addresses
  // being stable across rehashing.
  using Map = absl::node_hash_map<const PyCodeObject*, FunctionInfo>;

  PythonFunctionInfoTable()
      : cache_hit_count_(EventCounters::Instance().GetEventCounter(
            "code_symbol_map.tls_cache_hit")),
        cache_miss_count_(EventCounters::Instance().GetEventCounter(
            "code_symbol_map.tls_cache_miss")) {}
  ~PythonFunctionInfoTable() = default;

  Map::iterator Insert(const PyCodeObject* code,
                       absl::string_view qualified_name) {
    absl::MutexLock lock(&mu_);
    return InsertLocked(code, qualified_name);
  }

  // Insert a new entry mapping code to its symbol info. Caches the new entry
  // for this thread.
  Map::iterator InsertLocked(const PyCodeObject* code,
                             absl::string_view qualified_name)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    size_t cache_elem = absl::Hash<const PyCodeObject*>{}(code) % kCacheSize;
    auto it =
        map_.emplace(code,
                     FunctionInfo{
                         .qualified_name = GlobalInternTable::Instance().Intern(
                             qualified_name),
                         .filename = GlobalInternTable::Instance().Intern(
                             GetObjectAsCheapString(code->co_filename)),
                         .line = code->co_firstlineno})
            .first;
    cache_[cache_elem] = &*it;
    return it;
  }

  friend class NoDestructor<PythonFunctionInfoTable>;

  PythonFunctionInfoTable(const PythonFunctionInfoTable&) = delete;
  PythonFunctionInfoTable(const PythonFunctionInfoTable&&) = delete;
  PythonFunctionInfoTable& operator=(const PythonFunctionInfoTable&) = delete;

  // Thread local cache of entries.
  static thread_local std::array<Map::value_type*, kCacheSize> cache_;

  // Counters to track cache effectiveness. Owned by `EventCounters`.
  int64_t* cache_hit_count_;
  int64_t* cache_miss_count_;

  absl::Mutex mu_;
  Map map_ ABSL_GUARDED_BY(mu_);
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYTHON_FUNCTION_INFO_TABLE_H_
