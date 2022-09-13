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

#include "profiler.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "perftools/profiles/proto/builder.h"
#include "profiler_test_util.h"
#include "testing/base/public/benchmark.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using perftools::profiles::Location;
using perftools::profiles::Profile;
using perftools::profiles::Sample;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::NotNull;

// A deliberately recursive function, that deliberately does not tailcall.
// Tailcalling confuses the stack walker, and the stack walker is not under
// test here.
ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL int64_t
Fib(int64_t n, s6::TestOnlyProfiler* profiler) {
  if (n <= 1) {
    profiler->Sample();
    return 1;
  } else {
    return Fib(n - 1, profiler) + Fib(n - 2, profiler);
  }
}

// A Fib variant that, at various levels of recursion, will produce a profiling
// annotation. The annotation can either be a raw ID as accepted by the profiler
// or a string.
ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL int64_t FibWithAnnotation(
    int64_t n, s6::TestOnlyProfiler* profiler,
    const absl::flat_hash_map<int64_t, int64_t>& recursion_depth_to_tag) {
  if (n <= 1) {
    profiler->Sample();
    return 1;
  }

  if (auto it = recursion_depth_to_tag.find(n);
      it != recursion_depth_to_tag.end()) {
    Profiler::Scope profiler_scope(*profiler, it->second);
    return FibWithAnnotation(n - 1, profiler, recursion_depth_to_tag) +
           FibWithAnnotation(n - 2, profiler, recursion_depth_to_tag);
  }
  return FibWithAnnotation(n - 1, profiler, recursion_depth_to_tag) +
         FibWithAnnotation(n - 2, profiler, recursion_depth_to_tag);
}

// Recurses to the given recursion depth and samples.
ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NO_TAIL_CALL void RecurseAndSleep(
    int64_t n, s6::TestOnlyProfiler* profiler) {
  if (n == 0) {
    profiler->Sample();
    return;
  }
  Profiler::Scope profiler_scope(*profiler, n);
  RecurseAndSleep(n - 1, profiler);
}

TEST(Profiler, CollectsNormalProfiles) {
  TestOnlyProfiler profiler;

  S6_ASSERT_OK(profiler.StartCollecting());

  testing::DoNotOptimize(Fib(20, &profiler));

  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Profile> profile,
                          profiler.StopCollectingPprof());
  ASSERT_THAT(profile, NotNull());

  // Find the deepest stack trace in the profile.
  const Sample* deepest_sample = GetDeepestSample(*profile);
  ASSERT_TRUE(deepest_sample) << "No samples collected";

  std::vector<std::string> stack = SymbolizeSample(*profile, *deepest_sample);

  // Count the number of Fib calls.
  int64_t num_fib_calls =
      std::accumulate(stack.begin(), stack.end(), int64_t{0},
                      [](int64_t count, absl::string_view call) {
                        // Only check the suffix; absl can sometimes insert the
                        // file and line before it.
                        bool is_fib = absl::StrContains(call, "Fib(");
                        return is_fib ? count + 1 : count;
                      });
  // The recursion depth is 20, so in a reasonable sample expect at least a
  // recursion depth of 10.
  EXPECT_THAT(num_fib_calls, Ge(10));
}

TEST(Profiler, CollectsAnnotatedProfiles) {
  TestOnlyProfiler profiler;
  S6_ASSERT_OK(profiler.StartCollecting());

  // We're not testing symbolization in this test, so make up some tag IDs that
  // don't exist but can be checked for later.
  const int64_t kTag1 = 0x56beef;
  const int64_t kTag2 = 0x654231;
  absl::flat_hash_map<int64_t, int64_t> recursion_depth_to_tag_id = {
      {19, kTag1},
      {17, kTag2},
  };
  // Unused result to ensure the function isn't folded away.
  testing::DoNotOptimize(
      FibWithAnnotation(20, &profiler, recursion_depth_to_tag_id));

  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Profile> profile,
                          profiler.StopCollectingPprof());
  ASSERT_THAT(profile, NotNull());

  // We can't guarantee that we have a single sample with both kTag1 and kTag2
  // (although it is likely). So we just ensure that we get a sample with at
  // least kTag1.
  const Sample* deepest_sample = GetDeepestSample(*profile, kTag1);
  ASSERT_TRUE(deepest_sample) << "No samples collected";

  std::vector<std::string> stack = SymbolizeSample(*profile, *deepest_sample);
  std::vector<int64_t> stack_addresses;
  stack_addresses.reserve(deepest_sample->location_id_size());
  for (int64_t loc_id : deepest_sample->location_id()) {
    stack_addresses.push_back(profile->location(loc_id - 1).address());
  }
  // Count the number of Fib calls.
  int64_t fib_calls = std::accumulate(
      stack.begin(), stack.end(), int64_t{0},
      [](int64_t count, absl::string_view call) {
        bool is_fib = absl::StrContains(call, "FibWithAnnotation(");
        return is_fib ? count + 1 : count;
      });
  // The recursion depth is 20, so in a reasonable sample expect at least a
  // recursion depth of 10.
  EXPECT_THAT(fib_calls, Ge(10));
  EXPECT_THAT(stack_addresses, Contains(Eq(kTag1)));
}

TEST(Profiler, CanRecurseDeep) {
  TestOnlyProfiler profiler;
  // Ensure we can recurse deeper than the recursion limit and we get something
  // sane out.
  S6_ASSERT_OK(profiler.StartCollecting());
  RecurseAndSleep(base::ProfileData::kMaxStackDepth * 2, &profiler);

  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Profile> profile,
                          profiler.StopCollectingPprof());
  ASSERT_THAT(profile, NotNull());

  // Find the deepest stack trace in the profile.
  const Sample* deepest_sample = GetDeepestSample(*profile);
  ASSERT_TRUE(deepest_sample) << "No samples collected";
  EXPECT_THAT(deepest_sample->location_id_size(),
              Ge(base::ProfileData::kMaxStackDepth - 1));
}

TEST(Profiler, SymbolizesAnnotatedProfiles) {
  TestOnlyProfiler profiler;
  S6_ASSERT_OK(profiler.StartCollecting());

  const std::string kTag1 = "MyTagOne";
  const std::string kTag2 = "TheTagTwo";
  absl::flat_hash_map<int64_t, int64_t> recursion_depth_to_tag = {
      {19, profiler.InternTag(kTag1)},
      {17, profiler.InternTag(kTag2)},
  };
  // Unused result to ensure the function isn't folded away.
  testing::DoNotOptimize(
      FibWithAnnotation(20, &profiler, recursion_depth_to_tag));

  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Profile> profile,
                          profiler.StopCollectingPprof());
  ASSERT_THAT(profile, NotNull());

  std::vector<std::string> seen_functions;
  for (const Location& loc : profile->location()) {
    if (loc.line_size() == 0) continue;
    int64_t function_index = loc.line(0).function_id() - 1;
    int64_t name_idx = profile->function(function_index).name();
    seen_functions.push_back(profile->string_table(name_idx));
  }

  EXPECT_THAT(seen_functions, Contains(Eq(kTag1)));
  EXPECT_THAT(seen_functions, Contains(Eq(kTag2)));
}

TEST(Profiler, OverwritesCorrectStackFrame) {
  TestOnlyProfiler profiler;
  S6_ASSERT_OK(profiler.StartCollecting());

  // We're not testing symbolization in this test, so make up some tag IDs that
  // don't exist but can be checked for later.
  const int64_t kTag1 = 0x56beef;
  const int64_t kTag2 = 0x654231;
  absl::flat_hash_map<int64_t, int64_t> recursion_depth_to_tag_id = {
      {19, kTag1},
      {17, kTag2},
  };
  testing::DoNotOptimize(
      FibWithAnnotation(20, &profiler, recursion_depth_to_tag_id));

  S6_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Profile> profile,
                          profiler.StopCollectingPprof());
  ASSERT_THAT(profile, NotNull());

  const Sample* deepest_sample = GetDeepestSample(*profile);
  ASSERT_TRUE(deepest_sample) << "No samples collected";

  std::vector<std::string> stack = SymbolizeSample(*profile, *deepest_sample);
  // Ensure that the stack does not contain calls to Profiler::Profile. This is
  // stack frame that we want to overwrite with our tags.
  ASSERT_THAT(stack, Not(Contains(HasSubstr("Profiler::Profile"))));
}

std::vector<std::string> ExtractAllFunctions(
    const Profiler::Profile& raw_profile) {
  std::vector<std::string> seen_functions;
  auto profile = raw_profile.ExportAsPpprof();
  for (const Location& loc : profile->location()) {
    if (loc.line_size() == 0) continue;
    int64_t function_index = loc.line(0).function_id() - 1;
    int64_t name_idx = profile->function(function_index).name();
    seen_functions.push_back(profile->string_table(name_idx));
  }
  return seen_functions;
}

TEST(Profiler, StripsCallingFrames) {
  TestOnlyProfiler profiler;
  S6_ASSERT_OK(profiler.StartCollecting());

  const std::string kTag1 = "MyTagOne";
  const std::string kTag2 = "TheTagTwo";
  absl::flat_hash_map<int64_t, int64_t> recursion_depth_to_tag = {
      {19, profiler.InternTag(kTag1)},
      {17, profiler.InternTag(kTag2)},
  };
  testing::DoNotOptimize(
      FibWithAnnotation(20, &profiler, recursion_depth_to_tag));

  S6_ASSERT_OK_AND_ASSIGN(Profiler::Profile raw_profile,
                          profiler.StopCollecting());

  // The initial profile should have this test function within it.
  std::string function_name = "Profiler_StripsCallingFrames_Test::TestBody";
  std::vector<std::string> seen_functions = ExtractAllFunctions(raw_profile);

  EXPECT_THAT(seen_functions, Contains(HasSubstr(function_name)));

  Profiler::Profile pruned_profile =
      raw_profile.WithCurrentStackTracePrefixRemoved();
  seen_functions = ExtractAllFunctions(pruned_profile);

  EXPECT_THAT(seen_functions, Not(Contains(HasSubstr(function_name))));
}

}  // namespace
}  // namespace deepmind::s6
