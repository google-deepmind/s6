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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_PATH_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_PATH_H_

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/string_view.h"

namespace deepmind::s6::file {

std::string JoinPathImpl(std::initializer_list<absl::string_view> fragments);

template <typename... Ts>
std::string JoinPath(Ts&&... ts) {
  return JoinPathImpl(std::initializer_list<absl::string_view>{ts...});
}

absl::string_view Basename(absl::string_view path);

}  // namespace deepmind::s6::file

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_PATH_H_
