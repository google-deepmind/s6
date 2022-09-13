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

#ifndef THIRD_PARTY_DEEPMIND_S6_PROFILER_TEST_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_PROFILER_TEST_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "perftools/profiles/proto/profile.pb.h"

namespace deepmind::s6 {

// Symbolizes the stack for `sample`. All unknown entries become "(unknown)".
// Entries that already have symbol data (Location.line_size() != 0) are
// symbolized based on their first .line() entry.
std::vector<std::string> SymbolizeSample(
    const perftools::profiles::Profile& profile,
    const perftools::profiles::Sample& sample);

// Finds the sample with the deepest stack trace in the profile. Return nullptr
// if the profile is empty.
//
// If `includes` is nonzero, attempts to get a sample that contains `includes`.
const perftools::profiles::Sample* GetDeepestSample(
    const perftools::profiles::Profile& profile, int64_t includes = 0);

// A Profiler implementation intended for unit testing. It does not sample;
// instead the unit test calls Sample() to deterministically trigger a SIGPROF
// event.
//
// Only one TestOnlyProfiler may be collecting at any one time.
class TestOnlyProfiler : public Profiler {
 public:
  // As Profiler::StartCollecting(). This is thread-hostile.
  absl::Status StartCollecting();

  // As Profiler::StopCollecting(). This is thread-hostile.
  absl::StatusOr<std::unique_ptr<perftools::profiles::Profile>>
  StopCollectingPprof();

  absl::StatusOr<Profiler::Profile> StopCollecting();

  // Simulates a SIGPROF event. This will occur synchronously on the current
  // thread. This is thread-hostile.
  void Sample();
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_PROFILER_TEST_UTIL_H_
