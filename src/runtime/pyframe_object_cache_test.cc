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

#include "runtime/pyframe_object_cache.h"

#include <algorithm>
#include <cstdint>
#include <mutex>  // NOLINT
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6 {
namespace {
void PyInit() {
  static std::once_flag once;
  std::call_once(once, []() { Py_Initialize(); });
}

TEST(PyFrameObjectCacheTest, AllocateFreeReclaimsStorage) {
  PyFrameObjectCache cache;
  PyFrameObject* f = cache.Allocate();
  cache.Finished(f);
  ASSERT_EQ(cache.capacity(), 0);
  ASSERT_EQ(cache.size(), 0);
}

TEST(PyFrameObjectCacheTest, StorageIsEventuallyReclaimed) {
  PyFrameObjectCache cache;
  PyFrameObject* f = cache.Allocate();
  EXPECT_EQ(cache.capacity(), 1);
  EXPECT_EQ(cache.size(), 1);

  Py_INCREF(f);
  cache.Finished(f);
  EXPECT_EQ(cache.capacity(), 1);
  EXPECT_EQ(cache.size(), 1);

  Py_DECREF(f);
  Py_DECREF(f);
  EXPECT_EQ(cache.capacity(), 1);
  EXPECT_EQ(cache.size(), 0);

  // Tombstones aren't reclaimed until the next Finished(), so allocate first.
  f = cache.Allocate();
  EXPECT_EQ(cache.capacity(), 2);
  EXPECT_EQ(cache.size(), 1);
  cache.Finished(f);

  // The tombstoned frame should now have been reclaimed.
  EXPECT_EQ(cache.capacity(), 0);
  EXPECT_EQ(cache.size(), 0);
}

TEST(PyFrameObjectCacheTest, RandomDeallocationOrder) {
  const int64_t kNumFramesToAllocate = PyFrameObjectCache::kNumFrames + 10;
  const int64_t kNumTestCycles = 10;
  PyInit();

  PyCodeObject global_dummy_code;
  global_dummy_code.ob_base.ob_refcnt = 2;
  global_dummy_code.co_zombieframe = nullptr;

  for (int64_t i = 0; i < kNumTestCycles; ++i) {
    PyFrameObjectCache cache;
    std::vector<PyFrameObject*> frames(kNumFramesToAllocate);
    absl::c_generate(frames, [&]() {
      PyFrameObject* f = cache.Allocate();
      // f_code, f_builtins and f_globals must be non-nullptr in frame_dealloc.
      Py_INCREF(Py_None);
      f->f_builtins = Py_None;
      Py_INCREF(Py_None);
      f->f_globals = Py_None;
      Py_INCREF(&global_dummy_code);
      f->f_code = &global_dummy_code;
      return f;
    });

    std::default_random_engine rng;
    std::shuffle(frames.begin(), frames.end(), rng);
    while (!frames.empty()) {
      cache.Finished(frames.back());
      frames.pop_back();
    }

    ASSERT_EQ(cache.capacity(), 0);
    ASSERT_EQ(cache.size(), 0);
  }
}

TEST(PyFrameObjectCacheTest, HeapFramesAreDeallocated) {
  PyInit();

  PyCodeObject global_dummy_code;
  global_dummy_code.ob_base.ob_refcnt = 2;
  global_dummy_code.co_zombieframe = nullptr;

  PyFrameObjectCache cache;
  PyFrameObject* f = cache.AllocateOnHeap();
  // f_code, f_builtins and f_globals must be non-nullptr in frame_dealloc.
  Py_INCREF(Py_None);
  f->f_builtins = Py_None;
  Py_INCREF(Py_None);
  f->f_globals = Py_None;
  Py_INCREF(&global_dummy_code);
  f->f_code = &global_dummy_code;

#if PY_MINOR_VERSION < 7
  // We want to test that `f` is deallocated. PyFrameObject cannot hold weak
  // references, so we just create a new PyTypeObject that we embed in
  // f_exc_type. We take a weakref to this, and expect it to be cleaned up when
  // the frame is deallocated.
  f->f_exc_type = PyType_GenericAlloc(&PyType_Type, 0);
  reinterpret_cast<PyTypeObject*>(f->f_exc_type)->tp_flags |=
      Py_TPFLAGS_HEAPTYPE;
  PyObject* weakref = PyWeakref_NewRef(f->f_exc_type, Py_None);
  S6_CHECK(weakref);
  ASSERT_NE(PyWeakref_GET_OBJECT(weakref), Py_None);

  cache.Finished(f);
  // `f` should be freed, so f->f_exc_type should be freed too.
  ASSERT_EQ(PyWeakref_GET_OBJECT(weakref), Py_None);
#endif
}

}  // namespace
}  // namespace deepmind::s6
