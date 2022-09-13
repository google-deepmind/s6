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

#include "utils/path.h"

#include <algorithm>
#include <string>

namespace deepmind::s6::file {

std::string JoinPathImpl(std::initializer_list<absl::string_view> fragments) {
  std::string joined_path;

  if (fragments.size() == 0) {
    return joined_path;
  }

  // Assume no fragment has a leading or trailing '/'. This means that we need
  // an extra character between each pair of fragments.
  size_t path_size = fragments.size() - 1;
  for (absl::string_view fragment : fragments) {
    path_size += fragment.size();
  }
  joined_path.resize(path_size);  // Will zero-initialize the string.

  auto out = joined_path.begin();
  const auto begin = joined_path.begin();
  bool path_has_trailing_slash = false;
  for (absl::string_view fragment : fragments) {
    if (fragment.empty()) {
      continue;
    }

    if (fragment.front() == '/') {
      if (path_has_trailing_slash) {
        fragment.remove_prefix(1);
      }
    } else {
      // Allow path to start without a leading '/'.
      if (!path_has_trailing_slash && out != begin) {
        *(out++) = '/';
      }
    }
    out = std::copy(fragment.begin(), fragment.end(), out);
    path_has_trailing_slash = fragment.back() == '/';
  }

  // Trim unnecessary reserved space.
  joined_path.erase(out, joined_path.end());
  return joined_path;
}

absl::string_view Basename(absl::string_view path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return path;
  }
  path.remove_prefix(pos + 1);
  return path;
}

}  // namespace deepmind::s6::file
