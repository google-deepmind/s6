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

#include "classes/class_manager.h"

#include <cstdint>

#include "classes/class.h"
#include "core_util.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {

Class* ClassManager::GetClassById(int64_t id) {
  absl::ReaderMutexLock lock(&mu_);
  // The zero ID is reserved, so the index into classes_ is id - 1.
  int64_t class_index = id - 1;
  if (class_index < 0 || class_index >= classes_.size()) return nullptr;
  return classes_[class_index];
}

const Class* ClassManager::GetClassById(int64_t id) const {
  absl::ReaderMutexLock lock(&mu_);
  // The zero ID is reserved, so the index into classes_ is id - 1.
  int64_t class_index = id - 1;
  if (class_index < 0 || class_index >= classes_.size()) return nullptr;
  return classes_[class_index];
}

absl::StatusOr<int64_t> ClassManager::AllocateNewClassId() {
  int64_t current_num_ids = classes_.size() + 1;
  if (current_num_ids == kMaxClassId) {
    return absl::ResourceExhaustedError("Maximum number of classes reached");
  }
  return current_num_ids;
}

ClassManager& ClassManager::Instance() {
  static NoDestructor<ClassManager> instance;
  return *instance;
}

InternedString ClassManager::InternString(absl::string_view str) {
  absl::MutexLock lock(&table_mu_);
  return InternedString(intern_table_.Insert(str).value);
}

InternedString ClassManager::InternString(PyObject* unicode_object) {
  S6_CHECK(PyUnicode_Check(unicode_object));
  if (!PyUnicode_CHECK_INTERNED(unicode_object)) {
    return InternString(GetObjectAsCheapStringRequiringGil(unicode_object));
  }
  if (PyUnicode_CHECK_INTERNED(unicode_object) == SSTATE_INTERNED_MORTAL) {
    PyUnicode_InternImmortal(&unicode_object);
  }
  absl::MutexLock lock(&mu_);
  if (auto it = interned_unicode_objects_.find(unicode_object);
      it != interned_unicode_objects_.end()) {
    return it->second;
  }
  InternedString s = InternString(GetObjectAsCheapString(unicode_object));
  S6_CHECK(!s.empty());
  // Note that this is racy, but InternString by definition is idempotent, so
  // we could only ever race `s` with `s`.
  interned_unicode_objects_.emplace(unicode_object, s);
  return s;
}

ClassManager::TypeModificationClient::~TypeModificationClient() {}

ClassManager::~ClassManager() {
  for (Class* cls : classes_) {
    delete cls;
  }
}

}  // namespace deepmind::s6
