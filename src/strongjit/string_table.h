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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_STRING_TABLE_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_STRING_TABLE_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "utils/keyed_intern_table.h"

namespace deepmind::s6 {

class StringTable {
 public:
  // The type of an interned string ID in the function's string table.
  using key_type = uint16_t;

  StringTable() : string_table_() {}

  // String tables can't be copied.
  StringTable(const StringTable&) = delete;
  StringTable& operator=(const StringTable&) = delete;

  ~StringTable() = default;

  // Interns a string in this function's string table.
  key_type InternString(absl::string_view s) {
    return string_table_.Insert(s).key;
  }

  // Returns the string content of an InternedString.
  absl::string_view GetInternedString(key_type id) const {
    return string_table_.ToStringView(id);
  }

 private:
  KeyedInternTable<key_type> string_table_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_STRING_TABLE_H_
