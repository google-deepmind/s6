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

#include "profiler_test_util.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/debugging/symbolize.h"
#include "perftools/profiles/proto/encoder.h"
#include "perftools/profiles/proto/profile.pb.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

TestOnlyProfiler* handler_payload;
absl::Notification* handler_done;

}  // namespace

extern "C" void SignalHandler(int sig, siginfo_t* sinfo, void* ucontext) {
  DeepMindS6SigprofHandler(sig, sinfo, ucontext,
                           static_cast<void*>(handler_payload));
  handler_done->Notify();
}

using perftools::profiles::Location;
using perftools::profiles::Profile;
using perftools::profiles::Sample;

std::vector<std::string> SymbolizeSample(const Profile& profile,
                                         const Sample& sample) {
  std::vector<std::string> stack;
  stack.reserve(sample.location_id_size());
  for (int64_t location_id : sample.location_id()) {
    // Note location_id - 1, to account for one-based indexing.
    const Location& location = profile.location(location_id - 1);
    if (location.line_size() > 0) {
      // This location is already symbolized. Use the zeroth line message
      // to find a function name.
      int64_t function_idx = location.line(0).function_id() - 1;
      int64_t name_idx = profile.function(function_idx).name();
      stack.push_back(profile.string_table(name_idx));
      continue;
    }

    void* pc = reinterpret_cast<void*>(location.address());
    char tmp[1024];
    std::string s = "(unknown)";
    if (absl::Symbolize(pc, tmp, sizeof(tmp))) {
      s = tmp;
    }
    stack.push_back(s);
  }
  return stack;
}

const Sample* GetDeepestSample(const Profile& profile, int64_t includes) {
  const Sample* deepest_sample = nullptr;
  bool deepest_sample_contains_includes = false;
  for (const Sample& sample : profile.sample()) {
    bool contains_includes =
        absl::c_any_of(sample.location_id(), [&](int64_t loc_id) {
          return profile.location(loc_id - 1).address() == includes;
        });
    if (contains_includes && !deepest_sample_contains_includes) {
      deepest_sample = &sample;
      deepest_sample_contains_includes = true;
    } else if (!deepest_sample ||
               sample.location_id_size() > deepest_sample->location_id_size()) {
      // Don't overwrite a sample with `includes` with a sample without.
      if (!includes || contains_includes || !deepest_sample_contains_includes) {
        deepest_sample = &sample;
        deepest_sample_contains_includes = contains_includes;
      }
    }
  }
  return deepest_sample;
}

absl::Status TestOnlyProfiler::StartCollecting() {
  absl::MutexLock lock(&mu_);

  lowest_push_tag_address_ = UINT64_MAX;
  highest_pop_tag_address_ = 0;
  tag_stack_pointer_ = 0;
  is_collecting_ = true;

  struct sigaction sa;
  sa.sa_sigaction = SignalHandler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  S6_RET_CHECK(sigaction(SIGUSR1, &sa, nullptr) == 0) << strerror(errno);

  handler_payload = this;
  return absl::OkStatus();
}

absl::StatusOr<Profiler::Profile> TestOnlyProfiler::StopCollecting() {
  absl::MutexLock lock(&mu_);

  struct sigaction sa;
  sa.sa_handler = SIG_IGN;
  sa.sa_flags = SA_RESTART;
  sigemptyset(&sa.sa_mask);
  S6_RET_CHECK(sigaction(SIGUSR1, &sa, nullptr) == 0) << strerror(errno);
  is_collecting_ = false;

  S6_ASSIGN_OR_RETURN(auto profile, ConsumeProfileData(/*period_micros=*/1));
  return profile;
}

absl::StatusOr<std::unique_ptr<perftools::profiles::Profile>>
TestOnlyProfiler::StopCollectingPprof() {
  S6_ASSIGN_OR_RETURN(auto profile, StopCollecting());
  return profile.ExportAsPpprof();
}

void TestOnlyProfiler::Sample() {
  absl::Notification done;
  handler_done = &done;
  pthread_kill(pthread_self(), SIGUSR1);
  done.WaitForNotification();
}

}  // namespace deepmind::s6
