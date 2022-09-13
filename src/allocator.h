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

#ifndef THIRD_PARTY_DEEPMIND_S6_ALLOCATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "utils/interval_map.h"

namespace deepmind::s6 {
struct GdbCodeEntry;
// Forward-declare internal implementation.
struct MinimalElf;

// The JitAllocator is a simple arena allocator that mprotects its pages to
// allow execution.
//
// Behavior is undefined if more than one JitAllocator instance is alive at
// any one time (as it interacts with a global linked-list provided to GDB).
class JitAllocator {
 public:
  // Creates a new allocator with an arena blocks size of `block_size`.
  // We provide a default block size of 4K, which is a normal page size.
  explicit JitAllocator(int64_t block_size = 4096);
  ~JitAllocator();

  // Allocates at least `size` bytes and returns a pointer to the allocated
  // space. `symbol` is the name of the code symbol that will be generated
  // inside the allocated space, and will be registered with GDB.
  void* Alloc(int64_t size, absl::string_view symbol);

  // Frees space allocated by `Alloc`.
  void Free(void* memory);

  // Allocates at least `size` bytes and returns a pointer to the allocated
  // space. No symbol is registered with the location.
  void* AllocData(int64_t size);

  // Registers information about a symbol with GDB and optionally Linux perf.
  // `memory` must be the exact pointer allocated by a call to `Alloc()`.
  //
  // When `memory` is freed, GDB will be informed that this symbol has gone.
  // `debug_annotations`, if specified, maps addresses with string identifiers.
  void RegisterSymbol(
      void* memory, absl::string_view symbol_name,
      absl::Span<const std::pair<void*, std::string>> debug_annotations);

  // Returns an interval map with all known symbols.
  //
  // This function is thread-safe.
  IntervalMap<uint64_t, std::string> CreateSymbolMap() const;

 private:
  // Per-allocation information.
  struct AllocInfo {
    std::shared_ptr<MinimalElf> elf;
    std::shared_ptr<GdbCodeEntry> code_entry;
    int64_t size;
    std::string backtrace_symbol;
  };

  // Unregisters `symbol` with GDB.
  void UnregisterSymbol(const AllocInfo* info);

  // Allocates raw data from a region. Creates a new region if the current is
  // too small.
  void* AllocateRaw(int64_t size) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Holds the mmapped regions.
  std::vector<void*> regions_;

  // The current pointer within regions_.back().
  void* region_ptr_ = nullptr;

  // The amount of memory left in the current region.
  int64_t region_available_ = 0;

  int64_t block_size_;

  // Per-alloc information.
  absl::flat_hash_map<void*, AllocInfo> infos_ ABSL_GUARDED_BY(mu_);

  mutable absl::Mutex mu_;
};
}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_ALLOCATOR_H_
