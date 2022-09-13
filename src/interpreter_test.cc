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

// Note that most of the tests for interpreter.cc are in
// python/interpreter_test.py.

#include "interpreter.h"

#include <Python.h>

#include <mutex>  // NOLINT
#include <vector>

#include "api.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using ::testing::Contains;
using ::testing::NotNull;
using ::testing::StrEq;

// A program that compute Fibonnacci numbers for ~2 seconds. We use a sampling
// profiler, so we need to run for enough time to guarantee a reasonable sample.
const char kProgram[] = R"(
import time

def pyfib(n):
  if n <= 1:
    return 1
  return pyfib(n-1) + pyfib(n-2)

start = time.perf_counter()
# Run Fibonacci for 2 seconds so we can assume to get some reasonable
# profiling samples.
while (time.perf_counter() < start + 2.0):
  pyfib(20)
)";

void InitializeOnce() {
  // Ensure Python is initialized.
  // TODO: Even if we call Py_Finalize, the leak checker thinks we
  // leak a load of Python objects. Understand why. A test as simple as
  // Py_Initialize(); Py_Finalize(); leaks objects.
  static std::once_flag once;
  std::call_once(once, [&]() {
    Py_Initialize();
    S6_ASSERT_OK(s6::Initialize());
  });
}

}  // namespace
}  // namespace deepmind::s6
