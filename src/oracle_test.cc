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

#include "oracle.h"

#include <algorithm>
#include <cstdint>
#include <mutex>  // NOLINT
#include <utility>

#include "absl/status/statusor.h"
#include "api.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
using testing::IsNull;

PyCodeObject* CreateCodeObject() {
  static std::once_flag once;
  std::call_once(once, []() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });
  PyObject* empty_tuple = PyTuple_New(0);
  PyObject* empty_unicode = PyUnicode_FromString("");
  // Ensure bytes has some content so that the bytecode count is nonzero.
  PyObject* dummy_bytes = PyBytes_FromString("abcdefghijklm");
  return PyCode_New(
      /*argcount=*/0, /*kwonlyargcount=*/0, /*nlocals=*/0,
      /*stacksize=*/0, /*flags=*/0, /*code=*/dummy_bytes,
      /*consts=*/empty_tuple, /*names=*/empty_tuple,
      /*varnames=*/empty_tuple, /*freevars=*/empty_tuple,
      /*cellvars=*/empty_tuple, /*filename=*/empty_unicode,
      /*name=*/empty_unicode, /*firstlineno=*/0, /*lnotab=*/dummy_bytes);
}

class SlowOracle : public Oracle {
 public:
  using Oracle::Oracle;

  void CompileFunctions() override {
    thread::Reader<std::unique_ptr<CompilationRequest>>& input =
        channel_.reader();
    // This compiler is slow, but catches up!
    absl::SleepFor(absl::Seconds(2));

    while (true) {
      std::unique_ptr<CompilationRequest> request;
      if (!input.Read(&request)) {
        // Channel is closed.
        return;
      }
      // Ensure the test is ready to receive this signal by acquiring the lock.
      { absl::MutexLock lock(&mu_); }
      compilation_completed_.Signal();
    }
  }
};

TEST(OracleThreadTest, HeadOfLineBlock_MakesProgress_Async) {
  std::vector<PyCodeObject*> code_objects(5);
  std::generate(code_objects.begin(), code_objects.end(), CreateCodeObject);

  SlowOracle async_oracle({.compile_always = true, .channel_capacity = 2});
  async_oracle.Initialize();
  for (PyCodeObject* code : code_objects) {
    CodeObject* code_object = async_oracle.GetCodeObject(code);
    EXPECT_THAT(code_object, testing::IsNull());
    Py_DECREF(code);
  }
}

TEST(OracleThreadTest, HeadOfLineBlock_MakesProgress_Sync) {
  std::vector<PyCodeObject*> code_objects(5);
  std::generate(code_objects.begin(), code_objects.end(), CreateCodeObject);

  SlowOracle sync_oracle(
      {.synchronous = true, .compile_always = true, .channel_capacity = 2});
  sync_oracle.Initialize();
  for (PyCodeObject* code : code_objects) {
    CodeObject* code_object = sync_oracle.GetCodeObject(code);
    EXPECT_THAT(code_object, testing::IsNull());
    Py_DECREF(code);
  }
}

// An Oracle that bumps the given value up when it compiles something.
class NotifyingOracle : public Oracle {
 public:
  NotifyingOracle(absl::Mutex& mu, int64_t& value,
                  OracleDebugOptions debug_options = {})
      : Oracle(debug_options), value_mu_(mu), value_(value) {}

  absl::StatusOr<std::unique_ptr<CodeObject>> Compile(
      PyCodeObject* code, Metadata* metadata, Function function) override {
    absl::MutexLock lock(&value_mu_);
    ++value_;
    auto f = std::make_shared<Function>(std::move(function));
    // Return a valid CodeObject but with no useful internal data. Calling code
    // assumes that at least the Strongjit entry point is readable.
    return absl::make_unique<CodeObject>(
        nullptr, nullptr, nullptr, 0, "foo",
        absl::Span<const BytecodeInstruction>{},
        absl::flat_hash_map<void*, DebugAnnotation>{}, nullptr, f,
        DeoptimizationMap(SlotIndexes(*f)), StackFrameLayout(), nullptr);
  }

  absl::Mutex& value_mu_;
  int64_t& value_;
};

TEST(OracleThreadTest, AlwaysCompile_AlwaysCompiles) {
  PyCodeObject* code_object = CreateCodeObject();

  absl::Mutex mu;
  int64_t count = 0;
  NotifyingOracle oracle(mu, count,
                         {.synchronous = true, .compile_always = true});
  oracle.Initialize();
  (void)oracle.GetCodeObject(code_object);

  absl::MutexLock lock(&mu);
  EXPECT_EQ(count, 1);
  Py_DECREF(code_object);
}

TEST(OracleThreadTest, ProfiledCode_EventuallyCompiles) {
  PyCodeObject* code_object = CreateCodeObject();
  PyFrameObject frame;
  frame.f_code = code_object;
  Metadata::Get(code_object)->set_completion_observed(true);

  absl::Mutex mu;
  int64_t count = 0;
  NotifyingOracle oracle(mu, count,
                         {.synchronous = true,
                          .profile_bytecode_instruction_interval = 1,
                          .hotness_threshold = 3});
  oracle.Initialize();

  // The first request should not trigger compilation.
  oracle.ProfileEvent(&frame, 1);
  EXPECT_THAT(oracle.GetCodeObject(code_object), IsNull());
  {
    absl::MutexLock lock(&mu);
    EXPECT_EQ(count, 0);
  }

  // Perform 5 more requests; by this point we do expect compilation to have
  // occurred (because the hotness_threshold is set to 3 above, which is less
  // than 5).
  //
  // Also note `profile_bytecode_instructions_interval` is set to 1, so every
  // ProfileEvent call should result in modifying the code_object's hotness.
  for (int64_t i = 0; i < 5; ++i) {
    oracle.ProfileEvent(&frame, 1);
  }

  absl::MutexLock lock(&mu);
  EXPECT_EQ(count, 1);
  Py_DECREF(code_object);
}

TEST(OracleThreadTest, Bisection) {
  std::array<PyCodeObject*, 4> code_objects;
  absl::c_generate(code_objects, CreateCodeObject);

  absl::Mutex mu;
  int64_t count = 0;
  NotifyingOracle oracle(
      mu, count,
      {.synchronous = true, .compile_always = true, .bisection_fuel = 3});
  oracle.Initialize();
  // Compile 3 times; all should be compiled.
  for (int64_t i = 0; i < 3; ++i) {
    oracle.GetCodeObject(code_objects[i]);
  }

  {
    absl::MutexLock lock(&mu);
    EXPECT_EQ(count, 3);
  }

  // The fourth compilation should not happen because bisection_fuel == 3 and
  // we've done 3 compiles above.
  oracle.GetCodeObject(code_objects[3]);
  {
    absl::MutexLock lock(&mu);
    EXPECT_EQ(count, 3);
  }

  for (PyCodeObject* code_object : code_objects) {
    Py_DECREF(code_object);
  }
}

}  // namespace
}  // namespace deepmind::s6
