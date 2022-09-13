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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_LOGGING_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"

#ifdef __GNUC__
#define S6_NORETURN __attribute__((noreturn))
#define S6_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define S6_PREDICT_TRUE(x) (__builtin_expect(x, 1))
#else
#define S6_NORETURN
#define S6_PREDICT_FALSE(x) (x)
#define S6_PREDICT_TRUE(x) (x)
#endif

namespace deepmind::s6::internal {

// Use a mutex to prevent multiple threads writing to log output at once.
extern absl::Mutex ostream_mutex;

// Returns true if verbose log level is less than or equal to that effective
// verbosity level, otherwise returns false.
// This function is implementation detail, user-code should use `VLOG_IS_ON`
// instead.
extern bool VLogIsEnabled(int verbose_level);

struct CheckOpString {
  explicit CheckOpString(std::string* str_) : str(str_) {}
  explicit operator bool() const { return S6_PREDICT_FALSE(str != nullptr); }
  std::string* const str;
};

template <typename T1, typename T2>
CheckOpString MakeCheckOpString(const T1& v1, const T2& v2,
                                const char* exprtext) {
  std::ostringstream oss;
  oss << exprtext << " (" << v1 << " vs. " << v2 << ")";
  return CheckOpString(new std::string(oss.str()));
}

#define DEFINE_CHECK_OP_IMPL(name, op)                                    \
  template <typename T1, typename T2>                                     \
  inline CheckOpString name##Impl(const T1& v1, const T2& v2,             \
                                  const char* exprtext) {                 \
    if (S6_PREDICT_TRUE(v1 op v2)) {                                      \
      return CheckOpString(nullptr);                                      \
    } else {                                                              \
      return (MakeCheckOpString)(v1, v2, exprtext);                       \
    }                                                                     \
  }                                                                       \
  inline CheckOpString name##Impl(int v1, int v2, const char* exprtext) { \
    return (name##Impl<int, int>)(v1, v2, exprtext);                      \
  }

DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
DEFINE_CHECK_OP_IMPL(Check_NE, !=)
DEFINE_CHECK_OP_IMPL(Check_LE, <=)
DEFINE_CHECK_OP_IMPL(Check_LT, <)
DEFINE_CHECK_OP_IMPL(Check_GE, >=)
DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef DEFINE_CHECK_OP_IMPL

class LogMessage {
 public:
  LogMessage(const char* file, int line, absl::LogSeverity severity) {
    stream_ << "[" << severity << ' ' << ' ' << file << ":" << line << "] ";
  }

  ~LogMessage() {
    absl::MutexLock lock(&ostream_mutex);
    std::clog << stream_.str() << std::endl;
  }

  template <typename T>
  friend LogMessage&& operator<<(LogMessage&& log, T&& t) {
    log.stream_ << std::forward<T>(t);
    return std::move(log);
  }

 private:
  std::ostringstream stream_;
};

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    stream_ << "[" << absl::LogSeverity::kFatal << ' ' << ' ' << file << ":"
            << line << "] ";
  }

  LogMessageFatal(const char* file, int line, const CheckOpString& result)
      : LogMessageFatal(file, line) {
    stream_ << " Check failed: " << *result.str << " ";
  }

  ~LogMessageFatal() S6_NORETURN {
    absl::MutexLock lock(&ostream_mutex);
    std::cerr << stream_.str() << std::endl;
    std::abort();
  }

  template <typename T>
  friend LogMessageFatal&& operator<<(LogMessageFatal&& log, T&& t) {
    log.stream_ << std::forward<T>(t);
    return std::move(log);
  }

 private:
  std::ostringstream stream_;
};

struct NullStream {
  template <typename T>
  friend NullStream&& operator<<(NullStream&& s, T&&) {
    return std::move(s);
  }
};

struct NullStreamFatal {
  template <typename T>
  friend NullStreamFatal&& operator<<(NullStreamFatal&& s, T&&) {
    return std::move(s);
  }

  ~NullStreamFatal() S6_NORETURN { std::abort(); }
};

// Returns a Status or StatusOr as a Status.
// Only for use in template or macro code that must work with both Status and
// StatusOr.
template <typename T>
inline const absl::Status& AsStatus(const absl::StatusOr<T>& status_or) {
  return status_or.status();
}
inline const absl::Status& AsStatus(const absl::Status& status) {
  return status;
}

}  // namespace deepmind::s6::internal

#define S6_CHECK_OP_LOG(name, op, val1, val2, log)                          \
  while (::deepmind::s6::internal::CheckOpString _result =                  \
             ::deepmind::s6::internal::name##Impl(val1, val2,               \
                                                  #val1 " " #op " " #val2)) \
  log(__FILE__, __LINE__, _result)

#define S6_CHECK_OP(name, op, val1, val2) \
  S6_CHECK_OP_LOG(name, op, val1, val2,   \
                  ::deepmind::s6::internal::LogMessageFatal)

#define S6_CHECK_EQ(val1, val2) S6_CHECK_OP(Check_EQ, ==, val1, val2)
#define S6_CHECK_NE(val1, val2) S6_CHECK_OP(Check_NE, !=, val1, val2)
#define S6_CHECK_LE(val1, val2) S6_CHECK_OP(Check_LE, <=, val1, val2)
#define S6_CHECK_LT(val1, val2) S6_CHECK_OP(Check_LT, <, val1, val2)
#define S6_CHECK_GE(val1, val2) S6_CHECK_OP(Check_GE, >=, val1, val2)
#define S6_CHECK_GT(val1, val2) S6_CHECK_OP(Check_GT, >, val1, val2)

#define S6_CHECK(condition)                                        \
  while (auto _result = ::deepmind::s6::internal::CheckOpString(   \
             (condition) ? nullptr : new std::string(#condition))) \
  ::deepmind::s6::internal::LogMessageFatal(__FILE__, __LINE__, _result)

#define S6_CHECK_OK(value) \
  S6_CHECK_EQ(absl::OkStatus(), ::deepmind::s6::internal::AsStatus(value))

#ifndef NDEBUG
#define S6_DCHECK(condition) S6_CHECK(condition)
#define S6_DCHECK_OK(value) S6_CHECK_OK(value)
#define S6_DCHECK_EQ(val1, val2) S6_CHECK_EQ(val1, val2)
#define S6_DCHECK_NE(val1, val2) S6_CHECK_NE(val1, val2)
#define S6_DCHECK_LE(val1, val2) S6_CHECK_LE(val1, val2)
#define S6_DCHECK_LT(val1, val2) S6_CHECK_LT(val1, val2)
#define S6_DCHECK_GE(val1, val2) S6_CHECK_GE(val1, val2)
#define S6_DCHECK_GT(val1, val2) S6_CHECK_GT(val1, val2)
#else
#define S6_DCHECK(condition) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_OK(value) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_EQ(val1, val2) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_NE(val1, val2) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_LE(val1, val2) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_LT(val1, val2) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_GE(val1, val2) ::deepmind::s6::internal::NullStream()
#define S6_DCHECK_GT(val1, val2) ::deepmind::s6::internal::NullStream()
#endif

#define S6_LOG_IMPL_INFO                                   \
  ::deepmind::s6::internal::LogMessage(__FILE__, __LINE__, \
                                       ::absl::LogSeverity::kInfo)

#define S6_LOG_IMPL_WARNING                                \
  ::deepmind::s6::internal::LogMessage(__FILE__, __LINE__, \
                                       ::absl::LogSeverity::kWarning)

#define S6_LOG_IMPL_ERROR                                  \
  ::deepmind::s6::internal::LogMessage(__FILE__, __LINE__, \
                                       ::absl::LogSeverity::kError)

#define S6_LOG_IMPL_FATAL \
  ::deepmind::s6::internal::LogMessageFatal(__FILE__, __LINE__)

#define S6_LOG(x) S6_LOG_IMPL_##x

#define S6_LOG_LINES(severity, text)                                   \
  do {                                                                 \
    std::vector<absl::string_view> lines = absl::StrSplit(text, '\n'); \
    for (auto line : lines) {                                          \
      S6_LOG(severity) << line;                                        \
    }                                                                  \
  } while (0)

#define S6_VLOG_IS_ON(verbose_level) \
  deepmind::s6::internal::VLogIsEnabled(verbose_level)

#define S6_VLOG(verbose_level)                                           \
  for (bool s6_vlog_is_on = S6_VLOG_IS_ON(verbose_level); s6_vlog_is_on; \
       s6_vlog_is_on = false)                                            \
  S6_LOG(INFO)

#ifndef NDEBUG
#define S6_DVLOG(verbose_level)                                          \
  for (bool s6_vlog_is_on = S6_VLOG_IS_ON(verbose_level); s6_vlog_is_on; \
       s6_vlog_is_on = false)                                            \
  S6_LOG(INFO)
#else
#define S6_DVLOG(verbose_level)                           \
  for (bool s6_vlog_is_on = S6_VLOG_IS_ON(verbose_level); \
       false && s6_vlog_is_on; s6_vlog_is_on = false)     \
  S6_LOG(INFO)
#endif

#define S6_VLOG_LINES(verbose_level, text) \
  if (bool s6_vlog_check = false; s6_vlog_check) S6_LOG_LINES(INFO, text)

#define S6_UNREACHABLE() S6_LOG(FATAL) << "Executed unreachable code"

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_LOGGING_H_
