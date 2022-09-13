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

#include "utils/logging.h"

#include "absl/flags/flag.h"
#include "absl/synchronization/mutex.h"
#include "utils/no_destructor.h"

ABSL_FLAG(int, s6_vlog_level, 0, "Verbose log-level");

namespace deepmind::s6::internal {

absl::Mutex ostream_mutex(absl::kConstInit);

bool VLogIsEnabled(int verbose_level) {
  return verbose_level <= absl::GetFlag(FLAGS_s6_vlog_level);
}

}  // namespace deepmind::s6::internal
