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

#ifndef THIRD_PARTY_DEEPMIND_S6_EVENT_COUNTERS_H_
#define THIRD_PARTY_DEEPMIND_S6_EVENT_COUNTERS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "event_counters.pb.h"
#include "global_intern_table.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {

// Singleton class for managing event counters.
//
// Event counters count the number of times that an event occurs. This could be
// in compiled code, in the runtime, or during compilation. Each event counter
// has an associated name, which enables the event point to be incremented from
// several locations.
//
// Once allocated, the underlying `int64_t*` counter is guaranteed not to move,
// which means that its value is safe to encode directly in generated asm.
class EventCounters {
 public:
  // Return the `EventCounters` singleton instance.
  static EventCounters& Instance();

  const absl::flat_hash_map<GlobalInternTable::InternedString,
                            std::unique_ptr<int64_t>>&
  counters() const {
    return counters_;
  }

  // Get a event counter by name. The event counter is allocated and initialized
  // to zero if needed.
  int64_t* GetEventCounter(absl::string_view counter_name);

  // Convenience function to get or create a counter and add `n`.
  void Add(absl::string_view counter_name, int64_t n) {
    *GetEventCounter(counter_name) += n;
  }

  // Creates an EventCountersProto.
  EventCountersProto Snapshot();

  // Dumps the event counters to the given directory. A new file is created
  // within this directory. The directory is created if required.
  absl::Status DumpToDirectory(absl::string_view directory);

 private:
  EventCounters() = default;
  ~EventCounters() = delete;

  EventCounters& operator=(const EventCounters&) = delete;
  EventCounters(const EventCounters&) = delete;

  friend class NoDestructor<EventCounters>;

  absl::flat_hash_map<GlobalInternTable::InternedString,
                      std::unique_ptr<int64_t>>
      counters_;
  absl::Mutex mu_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_EVENT_COUNTERS_H_
