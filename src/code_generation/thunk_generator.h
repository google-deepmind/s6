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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_THUNK_GENERATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_THUNK_GENERATOR_H_

#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "allocator.h"
#include "asmjit/asmjit.h"
#include "asmjit/core/builder.h"
#include "asmjit/core/codeholder.h"
#include "code_generation/asmjit_util.h"

namespace deepmind::s6 {

// Generates the entry thunks for strongjit functions. A thunk forwards (jumps)
// to the most recent code generated for a particular function.
//
// Code that calls a strongjit function will actually emit a call instruction
// to this thunk. Then, when the called function is deoptimized, we can modify
// this thunk (to call the interpreter) rather than every callsite.
//
// An alternative strategy would have all callsites be dynamic, loading the
// current CodeObject at runtime, but this is more costly than a jump.
//
// This is used to handle dispatch for specializations.
//
// For atomicity and thread safety, we generate code in chunks of 8 bytes
// each. We can atomically overwrite 8 bytes (aligned) of instructions without
// causing corruption in a parallel thread.
//
// In the simplest case with no specializations, we just need to emit:
//     jmp [real_function_entry]
//
// But this is an indirect jump (jump to 64-bit address is unencodable), so we
// need an address table:
//     jmp [address]
//   address: <address here>
//
// Now if we have specializations, we instead need a prolog to extract the class
// ID of the first argument, then a sequence of test-and-jumps for the possible
// specializations. These require, in the general case, 16 bytes to encode, and
// we can only replace 8 bytes of the instruction stream atomically when making
// updates.
//
// Therefore we use the following layout:
//
//   entry:
//    { prolog_ (16 bytes)
//     mov rax, [rsi + 8]   // Load Py_TYPE(arg[0])
//     mov rax, [rax + ...] // Load tp_flags
//     shr rax, $44         // tp_flags>> 44 == class_id
//     nop...               // Pad to 16 bytes
//    }
//    { conditional_jumps_ (8 bytes each)
//     cmp rax, $123
//     jz long_jumps_[i]    // If class_id match, jump to long_jumps[i]. This
//                          // has a small (8-bit) displacement.
//     nop
//     nop                  // Pad to 8-byte boundary.
//    }
//    { long_jumps_ (8 bytes each)
//     jmp [addresses_[i]]  // Uses address_table in general case but will
//                          // encode displacement directly if possible.
//    }
//    { addresses_ (8 bytes each)
//     <address>
//    }
//
// prolog: Extracts the class_id. If we are unspecialized, this is simply
//   "jmp addresses[i]".
// conditional_jumps_: The mutable region. Each of these segments is 8 bytes
//   exactly and is 8-byte aligned. Note that a conditional jump cannot be
//   indirect, which is why we conditionally jump to a `long_jumps_` entry. That
//   and we can encode this as JZ rel8, which is 2 bytes (and keeps us within
//   8 bytes for a CMP imm32 + JZ rel8.
// long_jumps_: Append-only, contains long indirect jumps.
// addresses_: Append-only, contains the addresses for indirect jumps.
//   addresses_[0] is always the interpreter function.
//
class ThunkGenerator {
 public:
  // Creates a new ThunkAllocator using `allocator` and forwarding to the given
  // interpreter function when no strongjit code is available.
  ThunkGenerator(JitAllocator& allocator, void* interpreter_function,
                 absl::string_view name);

  // Sets the forwarding target. The thunk will contain a `jmp` to this target.
  // Use 0 for the generic, unspecialized version.
  absl::Status SetTarget(int64_t specialized_class_id, void* target);

  // Sets the forwarding target back to the interpreter function.
  // Use 0 for the generic, unspecialized version.
  absl::Status SetTargetToInterpreter(int64_t specialized_class_id);

  // Returns the entry point.
  void* GetEntry() const { return allocation_; }

  // Not copyable or movable.
  ThunkGenerator(const ThunkGenerator&) = delete;
  ThunkGenerator(ThunkGenerator&&) = delete;

 private:
  absl::Status Emit(asmjit::CodeHolder& code, uint64_t* memory,
                    absl::FunctionRef<void(asmjit::x86::Builder&)> fn);
  absl::Status EmitJump(uint64_t* jump_instruction_memory,
                        uint64_t* address_table_entry);
  absl::Status EmitCompareAndJump(uint64_t* instruction_memory,
                                  int64_t class_id,
                                  uint64_t* long_jump_address);
  absl::Status EmitSpecializationProlog();
  absl::Status AllocateSpecializationIndex(int64_t class_id);

  absl::Span<uint64_t> prolog_;
  absl::Span<uint64_t> conditional_jumps_;
  absl::Span<uint64_t> long_jumps_;
  absl::Span<uint64_t> addresses_;

  std::vector<int64_t> conditional_jump_class_ids_;
  bool has_been_specialized_ = false;

  // The interpreter symbol, to be used with SetTargetToInterpreter.
  uint64_t interpreter_symbol_;

  // The master allocation.
  void* allocation_;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_THUNK_GENERATOR_H_
