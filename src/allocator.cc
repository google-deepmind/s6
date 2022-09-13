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

#include "allocator.h"

#include <elf.h>
#include <sys/mman.h>

#include <cstdint>
#include <mutex>  // NOLINT
#include <new>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/internal/sysinfo.h"
#include "utils/interval_map.h"
#include "utils/logging.h"
#include "utils/mathutil.h"

namespace deepmind::s6 {

// GDB integration - Linked list of ELF binaries.
struct GdbCodeEntry {
  GdbCodeEntry* next_entry;
  GdbCodeEntry* prev_entry;
  const char* symfile_addr;
  uint64_t symfile_size;
};

// GDB requires a well-formed ELF file to register a symbol. This is the
// smallest well-formed ELF file that GDB will accept. It contains a header and
// four sections (NULL, .symtab, .strtab, .text). The symtab has two entries,
// (NULL, <our jit symbol>). The .text section has SHT_NOBITS so has no content,
// only a start address and length.
struct MinimalElf {
  Elf64_Ehdr header;
  std::array<Elf64_Shdr, 4> section_header;
  Elf64_Sym reserved_symbol;
  Elf64_Sym symbol;
  std::array<char, 256> strtab;
};

namespace {
// GDB integration - action type for __jit_debug_register_code().
enum GdbActionKind : uint32_t {
  kGdbNoAction = 0,
  kGdbRegisterFunction = 1,
  kGdbUnregisterFunction = 2
};

// GDB integration - the type of __jit_debug_descriptor, which is our
// communication method with GDB.
struct GdbDescriptor {
  // Always 1.
  uint32_t version;
  // The action we want GDB to take.
  GdbActionKind action_flag;
  // The GdbCodeEntry we want GDB to take action on.
  GdbCodeEntry* relevant_entry;
  // GDB may choose to traverse the entire linked-list of GdbCodeEntrys instead
  // of just `relevant_entry`. This allows GDB to find the start of the list.
  GdbCodeEntry* first_entry;
};

// GDB integration - GDB puts a breakpoint in this function. when we call it,
// GDB reads the value of __jit_debug_descriptor for action to take.
extern "C" void ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_WEAK
__jit_debug_register_code() {
  // Ensure this function isn't removed by the compiler.
  __asm__ volatile("" : : : "memory");
}

// Mutex guarding modifications to __jit_debug_descriptor and first_code_entry.
absl::Mutex gdb_mu(absl::kConstInit);

// GDB integration - We make sure to specify the version statically, because the
// debugger may check the version before we can set it.
extern "C" ABSL_ATTRIBUTE_WEAK struct GdbDescriptor __jit_debug_descriptor
    ABSL_GUARDED_BY(gdb_mu) = {1, kGdbNoAction, nullptr, nullptr};

// Section header indices for MinimalElf.
enum SectionHeaderIdx { kNull, kSymtab, kStrtab, kText };

// Entry point for the global linked-list.
GdbCodeEntry* first_code_entry ABSL_GUARDED_BY(gdb_mu) = nullptr;

// Creates a MinimalElf. This sets up everything except a symbol name, address
// and length which should be filled in later. Creating this prototype once
// makes creating more Elves minimal-effort.
MinimalElf MakePrototypicalElf() {
  MinimalElf elf;
  std::memset(&elf, 0, sizeof(MinimalElf));

  const std::array<uint8_t, 16> kIdent = {
      ELFMAG0,    ELFMAG1,     ELFMAG2,    ELFMAG3,
      ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_SYSV,
      0,          0,           0,          0,
      0,          0,           0};
  absl::c_copy(kIdent, &elf.header.e_ident[0]);
  elf.header.e_type = ET_REL;
  elf.header.e_machine = EM_X86_64;
  elf.header.e_version = EV_CURRENT;
  elf.header.e_ehsize = sizeof(Elf64_Ehdr);
  elf.header.e_shentsize = sizeof(Elf64_Shdr);
  elf.header.e_shnum = elf.section_header.size();
  // GDB really wants a shstrtab, even though we pass all the sh_name entries
  // as 0.
  elf.header.e_shstrndx = kStrtab;
  elf.header.e_shoff = offsetof(MinimalElf, section_header);

  elf.section_header[kSymtab].sh_type = SHT_SYMTAB;
  elf.section_header[kSymtab].sh_offset = offsetof(MinimalElf, reserved_symbol);
  elf.section_header[kSymtab].sh_size = sizeof(Elf64_Sym) * 2;
  elf.section_header[kSymtab].sh_link = kStrtab;
  // One greater than the last local symbol. We have no local symbols.
  elf.section_header[kSymtab].sh_info = 2;
  elf.section_header[kSymtab].sh_entsize = sizeof(Elf64_Sym);

  elf.section_header[kStrtab].sh_type = SHT_STRTAB;
  elf.section_header[kStrtab].sh_offset = offsetof(MinimalElf, strtab);
  elf.section_header[kStrtab].sh_size = sizeof(MinimalElf::strtab);

  elf.section_header[kText].sh_type = SHT_NOBITS;
  elf.section_header[kText].sh_link = 1;
  elf.section_header[kText].sh_flags = SHF_WRITE | SHF_ALLOC;

  // Offset into string table, ignoring initial NUL byte.
  elf.symbol.st_name = 1;
  // Offset into .text section.
  elf.symbol.st_value = 0;
  elf.symbol.st_info = ELF64_ST_INFO(STB_LOCAL, STT_FUNC);
  elf.symbol.st_other = STV_DEFAULT;
  elf.symbol.st_shndx = kText;

  return elf;
}

// Constructs a new MinimalElf for the given symbol name, address and length.
std::unique_ptr<MinimalElf> MakeElfForSymbol(absl::string_view symbol,
                                             uint64_t start_address,
                                             uint64_t length) {
  auto elf = absl::make_unique<MinimalElf>();

  static MinimalElf prototype = MakePrototypicalElf();
  *elf = prototype;

  elf->section_header[kText].sh_addr = start_address;
  elf->section_header[kText].sh_size = length;
  elf->symbol.st_size = length;

  // The string table must always start with NUL and end with NUL. Therefore
  // we have sizeof(strtab) - 2 bytes to play with.
  absl::string_view truncated_symbol = symbol.substr(0, elf->strtab.size() - 2);
  // strtab has already been NUL-initialized, so copy from index 1 so that
  // strtab[0] == '\0' still.
  absl::c_copy(truncated_symbol, std::next(elf->strtab.begin()));

  return elf;
}

// A jitdump entity. Jitdump is the format used by Linux perf to annotate and
// report on jitted symbols. Spec:
// https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/jitdump-specification.txt
enum JitdumpEntityKind : uint32_t {
  // Registration of new code.
  kCodeLoad = 0,
  // Debug information for existing code.
  kCodeDebugInfo = 2,
};

// File header for a jitdump file.
struct JitdumpHeader {
  // Characters "jItD"
  uint32_t magic = 0x4A695444;
  // Header version
  uint32_t version = 1;
  uint32_t total_size;
  uint32_t elf_mach;
  uint32_t pad1 = 0;
  uint32_t pid;
  uint64_t timestamp;
  uint64_t flags = 0;
};

// Jitdump entity that reports code has been loaded. After this struct is the
// symbol name (NUL-terminated) and then the code content itself.
struct JitdumpLoadCode {
  JitdumpEntityKind kind = kCodeLoad;
  uint32_t total_size;
  uint64_t timestamp;

  uint32_t pid;
  uint32_t tid;
  uint64_t vma;
  uint64_t code_addr;
  uint64_t code_size;
  uint64_t code_index;
};

// Jitdump debug entry. This is followed by a NUL-terminated "filename", which
// is just a string and has no semantics attached.
struct JitdumpDebugEntry {
  uint64_t addr;
  int32_t lineno = 1;
  int32_t discrim = 0;
};

// Jitdump entity holding multiple JitdumpDebugEntries. The JitdumpDebugEntries
// directly follow this struct.
struct JitdumpDebugInfo {
  JitdumpEntityKind kind = kCodeDebugInfo;
  uint32_t total_size;
  uint64_t timestamp;

  uint64_t code_addr;
  uint64_t num_entries;
};

uint64_t NanosecondsFromTimespec(const struct timespec* ts) {
  const uint64_t kNanosecondsPerSecond = 1000000000;
  return (static_cast<uint64_t>(ts->tv_sec) * kNanosecondsPerSecond) +
         ts->tv_nsec;
}

// Obtains a CLOCK_MONOTONIC timestamp.
uint64_t GetTimestamp() {
  struct timespec ts;

  int ret = clock_gettime(CLOCK_MONOTONIC, &ts);
  if (ret) return 0;

  return NanosecondsFromTimespec(&ts);
}

std::string GetJitdumpFilename() {
  // TODO: Respect TMPDIR.
  return absl::StrCat("/tmp/jit-", getpid(), ".dump");
}

void InitializeJitdump() {
  std::string filename = GetJitdumpFilename();
  FILE* stream = std::fopen(filename.c_str(), "w+");
  S6_CHECK(stream != nullptr);

  JitdumpHeader header;
  header.timestamp = GetTimestamp();
  header.elf_mach = EM_X86_64;
  header.pid = getpid();
  header.total_size = sizeof(JitdumpHeader);
  S6_CHECK(fwrite_unlocked(&header, sizeof(JitdumpHeader), 1, stream) > 0);

  int64_t page_size = sysconf(_SC_PAGESIZE);
  S6_CHECK(page_size != -1);

  // Perf requires a MMAP record to locate the file we write from a perf.data
  // file. So here we perform a trivial mmap of the file we just created.
  //
  // PROT_EXEC is required for arbitrary perf related reasons.
  S6_CHECK(mmap(nullptr, page_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE, fileno(stream), 0) != MAP_FAILED);

  S6_CHECK(fclose(stream) == 0);
}

void JitdumpRegisterCodeObject(absl::string_view symbol, uint64_t code,
                               int64_t size) {
  std::string filename = GetJitdumpFilename();
  FILE* stream = fopen(filename.c_str(), "a");
  S6_CHECK(stream != nullptr);

  // code_index is just a monotonically increasing ID.
  static uint64_t code_index = 0;
  JitdumpLoadCode entity;
  entity.total_size = symbol.size() + 1 + size + sizeof(JitdumpLoadCode);
  entity.timestamp = GetTimestamp();
  entity.pid = getpid();
  entity.tid = absl::base_internal::GetTID();
  entity.vma = code;
  entity.code_addr = code;
  entity.code_size = size;
  entity.code_index = code_index++;

  S6_CHECK(fwrite_unlocked(reinterpret_cast<void*>(&entity),
                           sizeof(JitdumpLoadCode), 1, stream) >= 0);
  S6_CHECK(fwrite_unlocked(reinterpret_cast<const void*>(symbol.data()),
                           symbol.size() + 1, 1, stream) >= 0);
  S6_CHECK(fwrite_unlocked(reinterpret_cast<void*>(code), size, 1, stream) >=
           0);

  S6_CHECK(fclose(stream) == 0);
}

void JitdumpRegisterDebugInfo(
    uint64_t code,
    absl::Span<const std::pair<void*, std::string>> debug_annotations) {
  std::string filename = GetJitdumpFilename();
  FILE* stream = fopen(filename.c_str(), "a");
  S6_CHECK(stream != nullptr);

  // Generate all the debug info strings up front so we know the size of
  // the debug entries list.
  std::vector<std::pair<JitdumpDebugEntry, std::string>> strings;
  int64_t bytes = 0;
  // Unfortunately perf won't show records where the filename changes but the
  // line number stays the same. So we create a monotonically increasing (fake)
  // line number.
  int32_t line_number = 1;
  for (const auto& [address, str] : debug_annotations) {
    uint64_t addr = reinterpret_cast<uint64_t>(address);
    strings.emplace_back(
        JitdumpDebugEntry{.addr = addr, .lineno = line_number++}, str);
    // The +1 is to account for the NUL-terminating byte.
    bytes += strings.back().second.size() + 1 + sizeof(JitdumpDebugEntry);
  }

  JitdumpDebugInfo entity;
  entity.total_size = bytes + sizeof(JitdumpDebugInfo);
  entity.timestamp = GetTimestamp();
  entity.code_addr = code;
  entity.num_entries = strings.size();

  S6_CHECK(fwrite_unlocked(reinterpret_cast<void*>(&entity),
                           sizeof(JitdumpDebugInfo), 1, stream) >= 0);
  for (const auto& [debug_entry, str] : strings) {
    S6_CHECK(fwrite_unlocked(reinterpret_cast<const void*>(&debug_entry),
                             sizeof(JitdumpDebugEntry), 1, stream) >= 0);
    S6_CHECK(fwrite_unlocked(reinterpret_cast<const void*>(str.c_str()),
                             str.size() + 1, 1, stream) >= 0);
  }

  S6_CHECK(fclose(stream) == 0);
}

// Returns true if we are running under "perf record".
bool IsRunningUnderPerf() {
  // Run the query once and cache the result, to avoid constantly querying
  // getenv which is not thread safe.
  static bool running_under_perf = ::getenv("PERF_BUILDID_DIR") != nullptr;
  return running_under_perf;
}
}  // namespace

JitAllocator::JitAllocator(int64_t block_size) : block_size_(block_size) {
  // Call IsRunningUnderPerf() from the main thread, to cache the
  // non-thread-safe getenv call.
  (void)IsRunningUnderPerf();
}

void JitAllocator::RegisterSymbol(
    void* memory, absl::string_view symbol_name,
    absl::Span<const std::pair<void*, std::string>> debug_annotations) {
  absl::MutexLock lock(&mu_);
  AllocInfo& info = infos_[memory];
  uint64_t start_address = reinterpret_cast<uint64_t>(memory);

  // Mangle the symbol as "s6-jit::symbolname" for GDB. This makes it clear it
  // comes from the S6 jit (as no C++ namespace could ever have a dash in it!)
  std::string mangled_symbol =
      absl::StrCat("_ZN6s6-jit", symbol_name.size(), symbol_name, "E");

  std::unique_ptr<MinimalElf> elf =
      MakeElfForSymbol(mangled_symbol, start_address, info.size);

  absl::MutexLock gdb_mu_lock(&gdb_mu);
  auto code_entry = absl::make_unique<GdbCodeEntry>();
  code_entry->prev_entry = nullptr;
  code_entry->next_entry = first_code_entry;
  code_entry->symfile_addr = reinterpret_cast<const char*>(elf.get());
  code_entry->symfile_size = sizeof(MinimalElf);
  if (first_code_entry) {
    first_code_entry->prev_entry = code_entry.get();
  }
  first_code_entry = code_entry.get();

  __jit_debug_descriptor.action_flag = kGdbRegisterFunction;
  __jit_debug_descriptor.first_entry = first_code_entry;
  __jit_debug_descriptor.relevant_entry = code_entry.get();
  __jit_debug_register_code();

  if (IsRunningUnderPerf()) {
    static std::once_flag once;
    std::call_once(once, []() { InitializeJitdump(); });

    // It is important that we register debug info before we register the code
    // object. Registering the code object flushes all existing debug info to
    // be associated with that object.
    JitdumpRegisterDebugInfo(start_address, debug_annotations);
    JitdumpRegisterCodeObject(symbol_name, start_address, info.size);
  }

  info = AllocInfo{std::move(elf), std::move(code_entry), info.size,
                   absl::StrCat(symbol_name)};
}

void JitAllocator::UnregisterSymbol(const AllocInfo* info) {
  absl::MutexLock lock(&gdb_mu);

  // Unlink `info->code_entry` from the doubly-linked list.
  if (info->code_entry->prev_entry)
    info->code_entry->prev_entry->next_entry = info->code_entry->next_entry;
  if (info->code_entry->next_entry)
    info->code_entry->next_entry->prev_entry = info->code_entry->prev_entry;
  if (first_code_entry == info->code_entry.get())
    first_code_entry = info->code_entry->next_entry;

  __jit_debug_descriptor.action_flag = kGdbUnregisterFunction;
  __jit_debug_descriptor.first_entry = first_code_entry;
  __jit_debug_descriptor.relevant_entry = info->code_entry.get();
  __jit_debug_register_code();
}

JitAllocator::~JitAllocator() {
  absl::MutexLock lock(&gdb_mu);
  // Unlink the code entry list; all the memory will now be destroyed.
  first_code_entry = nullptr;
}

void* JitAllocator::Alloc(int64_t size, absl::string_view symbol) {
  absl::MutexLock lock(&mu_);
  void* ptr = AllocateRaw(size);
  if (!ptr) return nullptr;

  // The arena allocator is a block-based allocator. We have a guarantee that
  // the returned address lives inside a contiguous space of blocksize() blocks.
  // We can therefore round down the start and up the end to a multiple of
  // blocksize() and pass this to mprotect().
  uint64_t base_address =
      RoundDownTo<uint64_t>(reinterpret_cast<uint64_t>(ptr), 4096);
  uint64_t end_address =
      RoundUpTo<uint64_t>(reinterpret_cast<uint64_t>(ptr) + size, 4096);

  S6_CHECK(mprotect(reinterpret_cast<void*>(base_address),
                    end_address - base_address,
                    PROT_READ | PROT_WRITE | PROT_EXEC) == 0);
  infos_[ptr] = AllocInfo{.size = size};

  return ptr;
}

void JitAllocator::Free(void* memory) {
  absl::MutexLock lock(&mu_);
  S6_CHECK(infos_.contains(memory));
  const JitAllocator::AllocInfo& info = infos_.at(memory);
  // Note, we do not erase infos_[memory]. This is used by CreateSymbolMap
  // at the end of profiling.

  UnregisterSymbol(&info);
}

void* JitAllocator::AllocData(int64_t size) {
  absl::MutexLock lock(&mu_);
  void* ptr = AllocateRaw(size);
  if (!ptr) return nullptr;

  infos_.emplace(ptr, AllocInfo{.size = size});
  return ptr;
}

void* JitAllocator::AllocateRaw(int64_t size) {
  size = RoundUpTo<int64_t>(size, 128);
  if (region_available_ < size) {
    int64_t block_size = std::max(size, block_size_);
    regions_.push_back(::operator new(block_size, std::align_val_t(4096)));
    region_ptr_ = regions_.back();
    region_available_ = block_size;

    S6_CHECK(mprotect(region_ptr_, block_size,
                      PROT_READ | PROT_WRITE | PROT_EXEC) == 0);
  }
  S6_CHECK_LE(size, region_available_);
  region_available_ -= size;
  void* ptr = region_ptr_;
  region_ptr_ =
      reinterpret_cast<void*>(reinterpret_cast<int64_t>(region_ptr_) + size);
  return ptr;
}

IntervalMap<uint64_t, std::string> JitAllocator::CreateSymbolMap() const {
  absl::MutexLock lock(&mu_);
  IntervalMap<uint64_t, std::string> map;
  for (auto [address, info] : infos_) {
    map.Set(reinterpret_cast<uint64_t>(address),
            reinterpret_cast<uint64_t>(address) + info.size,
            info.backtrace_symbol);
  }
  return map;
}

}  // namespace deepmind::s6
