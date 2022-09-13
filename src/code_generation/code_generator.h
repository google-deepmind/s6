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

#ifndef THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATOR_H_
#define THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATOR_H_

#include <Python.h>

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "allocator.h"
#include "asmjit/asmjit.h"
#include "code_generation/register_allocator.h"
#include "code_object.h"
#include "core_util.h"
#include "strongjit/base.h"
#include "strongjit/function.h"

namespace deepmind {
namespace s6 {
class Metadata;
}  // namespace s6
}  // namespace deepmind

namespace deepmind::s6 {

using AddPassesFunction = std::function<void(asmjit::x86::Builder&)>;

struct CodeGeneratorOptions {
  // A callback that can modify the Builder object before it is
  // code-generated. The only supported usecase is to call Builder::addPassT<>()
  // to add one or more passes, which will be run before the code is generated.
  const AddPassesFunction& add_passes = {};

  // Generates profiling information for BrInsts.
  bool profile_branches = false;
};

// Generates machine code for `f`. The returned CodeObject takes ownership of
// `f`.
//
// `add_passes` is a callback that can modify the Builder object before it is
// code-generated. The only supported usecase is to call Builder::addPassT<>()
// to add one or more passes, which will be run before the code is generated.
absl::StatusOr<std::unique_ptr<CodeObject>> GenerateCode(
    Function&& f, const RegisterAllocation& ra,
    absl::Span<const BytecodeInstruction> program, JitAllocator& allocator,
    PyCodeObject* code_object, Metadata* metadata = nullptr,
    const CodeGeneratorOptions& options = {});

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_CODE_GENERATION_CODE_GENERATOR_H_
