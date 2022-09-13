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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_STATUS_BUILDER_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_STATUS_BUILDER_H_

#include <sstream>
#include <utility>

#include "absl/status/status.h"

namespace deepmind::s6 {

// StatusBuilder allows an absl::Status and message to be built with a streaming
// interface.
//
// StatusBuilder is designed so that it can only be used as a temporary -
// conversion to absl::Status and streaming operations require an rvalue.
//
// StatusBuilder is used in S6 macros to produce error messages:
// S6_RET_CHECK(some_condition) << "Error message " << built << "with streaming"
//                              << interface;
class StatusBuilder {
 public:
  template <typename... Ts>
  explicit StatusBuilder(absl::StatusCode code, Ts&&... ts) : code_(code) {
    AppendMessage(std::forward<Ts>(ts)...);
  }

  template <typename T>
  friend StatusBuilder&& operator<<(StatusBuilder&& s, T&& t) {
    s.stream_ << std::forward<T>(t);
    return std::move(s);
  }

  operator absl::Status() && {  // NOLINT - allow explicit
    return absl::Status(code_, stream_.str());
  }

 private:
  template <typename T, typename... Ts>
  void AppendMessage(T&& t, Ts&&... ts) {
    stream_ << std::forward<T>(t);
    AppendMessage(std::forward<Ts>(ts)...);
  }

  template <typename T>
  void AppendMessage(T&& t) {
    stream_ << std::forward<T>(t);
  }

  void AppendMessage() {}

  std::ostringstream stream_;
  absl::StatusCode code_;
};
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_STATUS_BUILDER_H_
