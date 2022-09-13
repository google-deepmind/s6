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

#ifndef LEARNING_DEEPMIND_S6_UTILS_STATUS_MACROS_H_
#define LEARNING_DEEPMIND_S6_UTILS_STATUS_MACROS_H_

#include "utils/logging.h"
#include "utils/status_builder.h"

#define S6_RETURN_IF_ERROR(...)                          \
  do {                                                   \
    auto _status = (__VA_ARGS__);                        \
    if (S6_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define S6_STATUS_MACROS_CONCAT_NAME(x, y) S6_STATUS_MACROS_CONCAT_IMPL(x, y)
#define S6_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#define S6_ASSIGN_OR_RETURN(lhs, rexpr) \
  S6_ASSIGN_OR_RETURN_IMPL(             \
      S6_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define S6_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                             \
  if (S6_PREDICT_FALSE(!statusor.ok())) {              \
    return statusor.status();                          \
  }                                                    \
  lhs = std::move(*statusor)

#define S6_ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  S6_ASSERT_OK_AND_ASSIGN_IMPL(             \
      S6_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define S6_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                 \
  if (S6_PREDICT_FALSE(!statusor.ok())) {                  \
    S6_LOG(FATAL) << "Assignment failed " << #rexpr;       \
  }                                                        \
  lhs = std::move(*statusor)

#define S6_ASSERT_OK(status_expr)               \
  if (S6_PREDICT_FALSE(!(status_expr).ok())) {  \
    S6_LOG(FATAL) << "Not OK " << #status_expr; \
  }

#define S6_RET_CHECK(condition)                                   \
  if (!(condition))                                               \
  return deepmind::s6::StatusBuilder(absl::StatusCode::kInternal, \
                                     " Check failed: ", #condition)

#define S6_RET_CHECK_FAIL(condition) \
  return deepmind::s6::StatusBuilder(absl::StatusCode::kInternal)

#define S6_RET_CHECK_EQ(lhs, rhs) S6_RET_CHECK(lhs == rhs)

#endif  // LEARNING_DEEPMIND_S6_UTILS_STATUS_MACROS_H_
