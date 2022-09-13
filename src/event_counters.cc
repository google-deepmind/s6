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

#include "event_counters.h"

#include <sys/stat.h>

#include <cstdint>
#include <fstream>

#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "utils/no_destructor.h"
#include "utils/path.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

EventCounters& EventCounters::Instance() {
  static NoDestructor<EventCounters> instance;
  return *instance;
}

int64_t* EventCounters::GetEventCounter(absl::string_view counter_name) {
  absl::MutexLock lock(&mu_);
  auto& counter = counters_[GlobalInternTable::Instance().Intern(counter_name)];
  if (!counter) counter = absl::make_unique<int64_t>(0);
  return counter.get();
}

EventCountersProto EventCounters::Snapshot() {
  EventCountersProto proto;
  for (const auto& [name, value_ptr] : counters_) {
    EventCounterProto counter;
    counter.set_name(std::string(name.get()));
    counter.set_value(*value_ptr);
    *proto.add_counters() = std::move(counter);
  }
  return proto;
}

absl::Status EventCounters::DumpToDirectory(absl::string_view directory) {
  if (::mkdir(directory.data(), 0777) != 0) {
    // EEXIST is benign. Anything else we abort.
    if (errno != EEXIST) {
      return absl::InternalError(
          absl::StrCat("error creating profile directory: ", strerror(errno)));
    }
  }
  std::string filename = file::JoinPath(
      directory, absl::StrCat("event_counters.", getpid(), ".pbtxt"));

  // Don't use //base:file; we can't guarantee InitGoogle has been called or
  // completed.
  std::ofstream stream(filename);
  S6_RET_CHECK(stream.is_open()) << "Unable to open profiling output file";

  std::string s;
  S6_RET_CHECK(google::protobuf::TextFormat::PrintToString(Snapshot(), &s));
  stream << s;
  return absl::OkStatus();
}

}  // namespace deepmind::s6
