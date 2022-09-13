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

#include "code_object.h"

#include <cstdint>

#include "absl/strings/str_format.h"
#include "core_util.h"
#include "udis86/udis86.h"

namespace deepmind::s6 {

std::string BytecodeInstruction::ToString() const {
  return absl::StrFormat("@%d:\t%s\t%d", program_offset(),
                         BytecodeOpcodeToString(opcode()), argument());
}

CodeObject::~CodeObject() {
  if (allocator_) allocator_->Free(reinterpret_cast<void*>(body_ptr_));
}

std::string CodeObject::ToString() const {
  std::string s;
  for (const BytecodeInstruction& inst : program_) {
    absl::StrAppend(&s, inst.ToString(), "\n");
  }
  return s;
}

std::string CodeObject::Disassemble() const {
  uint64_t address = reinterpret_cast<uint64_t>(body_ptr_);
  uint64_t end = address + body_size_;

  ud_t ud_obj;
  ud_init(&ud_obj);
  ud_set_mode(&ud_obj, 64);
  ud_set_input_buffer(&ud_obj, reinterpret_cast<uint8_t*>(body_ptr_),
                      body_size_);
  ud_set_pc(&ud_obj, address);
  ud_set_syntax(&ud_obj, UD_SYN_INTEL);

  std::string s;
  while (unsigned int size = ud_disassemble(&ud_obj)) {
    if (auto it = debug_annotations_.find(reinterpret_cast<void*>(address));
        it != debug_annotations_.end()) {
      if (!it->second.is_code()) {
        // Format the rest of the program as data. This is the constant pool.
        absl::StrAppend(&s, "// <data>\n");
        while (address < end) {
          absl::StrAppendFormat(&s, "%#05x: %08x\n", address,
                                *reinterpret_cast<uint64_t*>(address));
          address += sizeof(void*);
        }
        break;
      }
      absl::StrAppendFormat(&s, "// %s\n", it->second.ToString());
    }
    absl::StrAppendFormat(&s, "%#05x: %s\n", address, ud_insn_asm(&ud_obj));
    address += size;
  }
  absl::StrAppendFormat(&s, "%#05x: <end>\n", address);
  return s;
}
}  // namespace deepmind::s6
