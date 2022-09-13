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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_KEYED_INTERN_TABLE_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_KEYED_INTERN_TABLE_H_

#include <deque>
#include <string>
#include <string_view>
#include <type_traits>

#include "absl/container/flat_hash_set.h"
#include "utils/logging.h"

// KeyedInternTable is a set of strings where each string is only stored once.
// This allows reduction in memory consumption where a single string is used
// repeatedly.
//
// Strings can be looked up by key and are stored so that the underlying char*
// pointer is stable after further insertions into the table.
//
// As a result of pointer stability, string-views of interned strings can be
// validly stored for the lifetime of the KeyedInternTable.

template <typename Key>
class KeyedInternTable {
  static_assert(std::is_integral_v<Key> && !std::is_enum_v<Key>);

 public:
  struct KeyValue {
    Key key;
    const char* value;  // null-terminated, non-null.
  };

  // Finds the string relating to the given key.
  //
  // REQUIRES: The string was previously inserted.
  absl::string_view ToStringView(Key key) const {
    S6_DCHECK(key >= 0 && key < values_.size())
        << "Key " << key << " does not correspond to an interned string.";
    return values_[static_cast<std::deque<std::string>::size_type>(key)];
  }

  // Returns the key-value corresponding to the given string.
  // Inserts the string if not already inserted.
  KeyValue Insert(absl::string_view s) {
    auto find_it = keys_.find(s);
    if (find_it != keys_.end()) {
      Key key = *find_it;
      return {key, values_[key].c_str()};
    }
    Key key = values_.size();
    values_.push_back(std::string(s));
    keys_.insert(key);
    return {key, values_.back().c_str()};
  }

  // Returns the key for the specified string if previously inserted, otherwise
  // returns `std::nullopt`.
  std::optional<Key> ToKey(absl::string_view s) const {
    auto find_it = keys_.find(s);
    if (find_it != keys_.end()) {
      return *find_it;
    }
    return std::nullopt;
  }

  // Returns the number of unique string inserted.
  size_t size() const { return values_.size(); }

 private:
  // Implement Hash and Equals to allow (heterogeneous) lookup of strings in the
  // keys set.
  struct Hash {
    using is_transparent = void;

    const std::deque<std::string>& strings;

    size_t operator()(absl::string_view x) const {
      return absl::Hash<absl::string_view>()(x);
    }

    size_t operator()(Key i) const {
      return (*this)(strings[static_cast<size_t>(i)]);
    }
  };

  struct Equals {
    using is_transparent = void;

    const std::deque<std::string>& strings;

    bool operator()(Key a, Key b) const { return a == b; }

    bool operator()(Key a, absl::string_view b) const {
      return strings[static_cast<size_t>(a)] == b;
    }
  };
  using KeySet = absl::flat_hash_set<Key, Hash, Equals>;

  std::deque<std::string> values_;
  KeySet keys_{0, Hash{values_}, Equals{values_}};
};

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_KEYED_INTERN_TABLE_H_
