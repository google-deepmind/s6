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

#include "code_generation/thunk_generator.h"

#include <Python.h>

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "asmjit/core/codeholder.h"
#include "code_generation/asmjit_util.h"
#include "utils/logging.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace x86 = ::asmjit::x86;

// The number of jumps we emit (the size of conditional_jumps_).
constexpr int64_t kNumJumps = 10;

// The size of the prolog in 8-byte words.
constexpr int64_t kPrologSize = 2;

// The allocation size in 8-byte words.
constexpr int64_t kAllocationSize = 64;

ThunkGenerator::ThunkGenerator(JitAllocator& allocator,
                               void* interpreter_function,
                               absl::string_view name) {
  std::string symbol_name = absl::StrCat(name, ".thunk");
  allocation_ = allocator.Alloc(kAllocationSize * 8, symbol_name);
  S6_CHECK(allocation_);
  allocator.RegisterSymbol(allocation_, symbol_name, {});
  // 0x90 is NOP.
  std::memset(allocation_, 0x90, kAllocationSize * 8);

  uint64_t* cursor = reinterpret_cast<uint64_t*>(allocation_);

  prolog_ = absl::MakeSpan(cursor, kPrologSize);
  cursor += kPrologSize;

  conditional_jumps_ = absl::MakeSpan(cursor, kNumJumps);
  cursor += kNumJumps;

  long_jumps_ = absl::MakeSpan(cursor, kNumJumps);
  cursor += kNumJumps;

  addresses_ = absl::MakeSpan(cursor, kNumJumps);
  cursor += kNumJumps;

  S6_CHECK_LE(cursor,
              reinterpret_cast<uint64_t*>(allocation_) + kAllocationSize);

  // The class ID 0 is always the interpreter symbol, and this is always at
  // the final address slot.
  addresses_[addresses_.size() - 1] =
      reinterpret_cast<uint64_t>(interpreter_function);
  interpreter_symbol_ = addresses_[addresses_.size() - 1];

  // Prime the prolog with a jump. This will get overwritten by EmitProlog when
  // we transition to a specialized form.
  S6_CHECK_OK(EmitJump(&prolog_[0], &addresses_[addresses_.size() - 1]));

  // Initialize all conditional jump slots with an unconditional jump to the
  // interpreter. (Note, the misnomer here is because this is the fallback case,
  // so we don't need a compare-and-jump, just a normal jump, and we can encode
  // this in 8 bytes).
  for (uint64_t& slot : conditional_jumps_) {
    S6_CHECK_OK(EmitJump(&slot, &addresses_[addresses_.size() - 1]));
  }
  conditional_jump_class_ids_.resize(kNumJumps);
}

int64_t GetRelativeDistance(uint64_t instruction_memory, uint64_t address) {
  return std::abs(static_cast<int64_t>(instruction_memory) -
                  static_cast<int64_t>(address));
}

absl::Status ThunkGenerator::EmitJump(uint64_t* jump_instruction_memory,
                                      uint64_t* address_table_entry) {
  asmjit::CodeHolder code;
  S6_RETURN_IF_ERROR(Emit(code, jump_instruction_memory, [&](x86::Builder& b) {
    int64_t relative =
        GetRelativeDistance(reinterpret_cast<uint64_t>(jump_instruction_memory),
                            *address_table_entry);
    if (relative < 1L << 31) {
      // We can emit a normal jmp (this is faster).
      b.jmp(*address_table_entry);
    } else {
      // We need an indirect jump.
      b.jmp(x86::ptr(reinterpret_cast<uint64_t>(address_table_entry)));
    }
    b.align(asmjit::kAlignCode, 8);
  }));
  S6_CHECK_EQ(code.codeSize(), 8);

  uint64_t buffer;
  code.copyFlattenedData(&buffer, 8);

  // Perform a single atomic 8-byte store rather than relying on memcpy (as
  // copyFlattenedData does).
  *jump_instruction_memory = buffer;
  return absl::OkStatus();
}

absl::Status ThunkGenerator::EmitSpecializationProlog() {
  asmjit::CodeHolder code;
  S6_RETURN_IF_ERROR(Emit(code, prolog_.data(), [&](x86::Builder& b) {
    b.mov(x86::rax, x86::qword_ptr(x86::rsi, offsetof(PyObject, ob_type)));
    b.mov(x86::rax, x86::qword_ptr(x86::rax, offsetof(PyTypeObject, tp_flags)));
    b.shr(x86::rax, 44);
    b.align(asmjit::kAlignCode, 8);
  }));
  S6_CHECK_EQ(code.codeSize(), 16);

  // We copy via a buffer, so we can emit two atomic stores.
  uint64_t buffer[2];
  code.copyFlattenedData(&buffer[0], 16);

  // We store prolog[1] first because it is inaccessible (the unconditional jump
  // from the constructor is in prolog_[0]). Then we overwrite prolog_[0]. This
  // is a safe ordering.
  prolog_[1] = buffer[1];
  prolog_[0] = buffer[0];

  return absl::OkStatus();
}

absl::Status ThunkGenerator::EmitCompareAndJump(uint64_t* instruction_memory,
                                                int64_t class_id,
                                                uint64_t* long_jump_address) {
  asmjit::CodeHolder code;
  S6_RETURN_IF_ERROR(Emit(code, instruction_memory, [&](x86::Builder& b) {
    b.cmp(x86::eax, asmjit::imm(class_id));
    // Note the short_() prefix is to force a 2-byte encoding.
    b.short_().jz(asmjit::imm(reinterpret_cast<uint64_t>(long_jump_address)));
    b.align(asmjit::kAlignCode, 8);
  }));

  S6_CHECK_EQ(code.codeSize(), 8);
  uint64_t buffer;
  code.copyFlattenedData(&buffer, 8);

  // Perform a single atomic 8-byte store rather than relying on memcpy (as
  // copyFlattenedData does).
  instruction_memory[0] = buffer;
  return absl::OkStatus();
}

absl::Status ThunkGenerator::Emit(
    asmjit::CodeHolder& code, uint64_t* memory,
    absl::FunctionRef<void(asmjit::x86::Builder&)> fn) {
  code.init(asmjit::CodeInfo(asmjit::ArchInfo::kIdX64));
  code.addEmitterOptions(asmjit::BaseEmitter::kOptionStrictValidation);
  if (memory) {
    code.relocateToBase(reinterpret_cast<uint64_t>(memory));
  }
  x86::Builder builder(&code);
  fn(builder);
  RETURN_IF_ASMJIT_ERROR(builder.finalize());
  RETURN_IF_ASMJIT_ERROR(code.flatten());
  return absl::OkStatus();
}

absl::Status ThunkGenerator::AllocateSpecializationIndex(int64_t class_id) {
  if (absl::c_linear_search(conditional_jump_class_ids_, class_id)) {
    return absl::OkStatus();
  }
  // Note we don't ever allocate the last conditional jump index; this is
  // reserved so that we always have at least one fallback case.
  for (int64_t i = 0; i < conditional_jump_class_ids_.size() - 1; ++i) {
    if (conditional_jump_class_ids_[i] == 0) {
      conditional_jump_class_ids_[i] = class_id;
      return absl::OkStatus();
    }
  }
  return absl::ResourceExhaustedError("no more specialization slots");
}

absl::Status ThunkGenerator::SetTargetToInterpreter(
    int64_t specialized_class_id) {
  if (specialized_class_id != 0 && !has_been_specialized_) {
    S6_RETURN_IF_ERROR(EmitSpecializationProlog());
    has_been_specialized_ = true;
  }
  if (specialized_class_id != 0) {
    S6_RETURN_IF_ERROR(AllocateSpecializationIndex(specialized_class_id));
  }

  for (int64_t i = 0; i < conditional_jump_class_ids_.size(); ++i) {
    if (conditional_jump_class_ids_[i] != specialized_class_id) continue;

    uint64_t* address_table_entry = &addresses_[i];
    *address_table_entry = interpreter_symbol_;

    S6_RETURN_IF_ERROR(EmitJump(&conditional_jumps_[i], address_table_entry));
  }

  // If the prolog hasn't been specialized, we also have to update the jump
  // in the prolog!
  if (!has_been_specialized_) {
    S6_CHECK_EQ(specialized_class_id, 0);
    uint64_t* address_table_entry = &addresses_[addresses_.size() - 1];
    *address_table_entry = interpreter_symbol_;
    S6_RETURN_IF_ERROR(EmitJump(prolog_.data(), address_table_entry));
  }

  return absl::OkStatus();
}

absl::Status ThunkGenerator::SetTarget(int64_t specialized_class_id,
                                       void* target) {
  if (specialized_class_id != 0 && !has_been_specialized_) {
    S6_RETURN_IF_ERROR(EmitSpecializationProlog());
    has_been_specialized_ = true;
  }

  if (specialized_class_id != 0) {
    S6_RETURN_IF_ERROR(AllocateSpecializationIndex(specialized_class_id));
  }

  for (int64_t i = 0; i < conditional_jump_class_ids_.size(); ++i) {
    if (conditional_jump_class_ids_[i] != specialized_class_id) continue;

    uint64_t* address_table_entry = &addresses_[i];
    *address_table_entry = reinterpret_cast<uint64_t>(target);

    if (specialized_class_id == 0) {
      S6_RETURN_IF_ERROR(EmitJump(&conditional_jumps_[i], address_table_entry));
    } else {
      S6_RETURN_IF_ERROR(EmitJump(&long_jumps_[i], address_table_entry));
      S6_RETURN_IF_ERROR(EmitCompareAndJump(
          &conditional_jumps_[i], specialized_class_id, &long_jumps_[i]));
    }
  }

  // If the prolog hasn't been specialized, we also have to update the jump
  // in the prolog!
  if (!has_been_specialized_) {
    S6_CHECK_EQ(specialized_class_id, 0);
    uint64_t* address_table_entry = &addresses_[addresses_.size() - 1];
    *address_table_entry = reinterpret_cast<uint64_t>(target);

    S6_RETURN_IF_ERROR(EmitJump(prolog_.data(), address_table_entry));
  }

  return absl::OkStatus();
}

}  // namespace deepmind::s6
