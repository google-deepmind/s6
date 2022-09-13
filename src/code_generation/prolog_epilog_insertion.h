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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_PROLOG_EPILOG_INSERTION_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_PROLOG_EPILOG_INSERTION_H_

#include "absl/status/status.h"
#include "code_generation/code_generation_context.h"

namespace deepmind::s6 {

// Generates code for the function prolog. This saves callee-saved registers
// and sets up the stack frame.
//
// Calls ctx.BindPyFrameEntryPoint() and (optionally) ctx.BindFastEntryPoint().
absl::Status GenerateProlog(CodeGenerationContext& ctx);

// Generates code to clean up the stack frame. This includes clearing all
// fastlocals, returning the PyFrameObject to the PyFrameObjectCache and
// changing PyThreadState::frame.
//
// Preserves rax; does not preserve any other registers.
//
// Calls ctx.BindCleanupPoint().
absl::Status GenerateCleanup(CodeGenerationContext& ctx);

// Generates code to return to the caller. This restores the value of callee-
// saved registers, restores the stack pointer and returns.
//
// Calls ctx.BindEpilogPoint().
absl::Status GenerateEpilog(CodeGenerationContext& ctx);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_PROLOG_EPILOG_INSERTION_H_
