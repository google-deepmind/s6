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

#include "code_generation/jit_stub.h"

#include <ostream>

#include "asmjit/asmjit.h"
#include "asmjit/core/builder.h"
#include "code_generation/asmjit_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
namespace x86 = ::asmjit::x86;

TEST(JitStubTest, Simple) {
  JitStub<PyObject> stub(x86::ptr(x86::rax));
  JitStub<PyTypeObject> type = stub.ob_type();
  EXPECT_EQ(type.Mem(), x86::qword_ptr(x86::rax, offsetof(PyObject, ob_type)));
  EXPECT_EQ(type.dereference_kind(), DereferenceKind::kDereferenced);

  // Now latch `type` into rbx.
  asmjit::x86::Builder b;
  x86::Emitter& e = *b.as<x86::Emitter>();
  type = type.Load(x86::rbx, e);
  EXPECT_EQ(type.Mem(), x86::qword_ptr(x86::rbx));
  EXPECT_EQ(type.dereference_kind(), DereferenceKind::kEffectiveAddress);

  // Now we should be able to access tp_version_tag.
  JitStub<int32_t> version_tag = type.tp_version_tag();
  EXPECT_EQ(version_tag.Mem(),
            x86::dword_ptr(x86::rbx, offsetof(PyTypeObject, tp_version_tag)));
  EXPECT_EQ(version_tag.dereference_kind(), DereferenceKind::kDereferenced);
}

TEST(JitStubTest, BaseOffset) {
  // This PyObject is at [rax + 24].
  JitStub<PyObject> stub(x86::ptr(x86::rax, 24));
  JitStub<PyTypeObject> type = stub.ob_type();
  // The offset of ob_type and 24 should be summed into the offset field.
  EXPECT_EQ(type.Mem(),
            x86::qword_ptr(x86::rax, 24 + offsetof(PyObject, ob_type)));
  EXPECT_EQ(type.dereference_kind(), DereferenceKind::kDereferenced);
}
}  // namespace

}  // namespace deepmind::s6
