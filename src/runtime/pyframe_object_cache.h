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

#ifndef THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYFRAME_OBJECT_CACHE_H_
#define THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYFRAME_OBJECT_CACHE_H_

#include <Python.h>
#include <frameobject.h>

#include <array>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "core_util.h"
#include "runtime/generator.h"
#include "utils/logging.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {
// Clears members of `f`. This includes the value stack, fastlocals, and
// f_exc_* members.
inline void ClearFrameMembers(PyFrameObject& frame) {
  PyObject** valuestack = frame.f_valuestack;
  for (PyObject** p = frame.f_localsplus; p < valuestack; ++p) Py_CLEAR(*p);
  if (frame.f_stacktop) {
    for (PyObject** p = valuestack; p < frame.f_stacktop; ++p) {
      Py_CLEAR(*p);
    }
  }
  if (frame.f_code) {
    for (int64_t i = 0; i < frame.f_code->co_stacksize; ++i) {
      frame.f_valuestack[i] = nullptr;
    }
  }
  Py_XDECREF(frame.f_back);
  Py_XDECREF(frame.f_code);
  Py_XDECREF(frame.f_builtins);
  Py_XDECREF(frame.f_globals);
  Py_CLEAR(frame.f_locals);
  Py_CLEAR(frame.f_trace);
  // TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION < 7
  Py_CLEAR(frame.f_exc_type);
  Py_CLEAR(frame.f_exc_value);
  Py_CLEAR(frame.f_exc_traceback);
#endif
}

// A free list of PyFrameObjects. Similar to `frameobject.h`'s freelist, this
// allows very cheap allocation of PyFrameObjects.
//
// PyFrameObjects are variably sized. Their `f_localsplus` member contains the
// fastlocals, free/cell vars, and the value stack.
//
// PyFrameObjects in this list all have the same number of f_localsplus
// entries so are equally sized. We also define a new Type that is similar to
// PyFrameObjectType but:
//   1) Is declared as not GC'd, because none of these PyFrameObjects are
//        garbage collectable.
//   2) Has a custom tp_dealloc to return a frame to the freelist.
//
// We expose the implementation details so that we can inline allocation and
// deallocation in generated code.
//
// This list is optimized for the common case of a stack of frames whose
// lifetimes do not extend beyond their allocating function. In this usage
// pattern the list can be considered a simple contiguous stack.
//
// The free list is a statically allocated contiguous array. When the array is
// used up, frame objects are allocated on the heap and GCd as normal.
//
// A single head pointer is maintained that points to the next unused frame in
// the list. If the head points at the end of the list, the list is empty and
// a frame should be allocated on the heap.
//
// If a frame's lifetime extends beyond its calling function, it is simply not
// released and stack allocation continues. Prior frames that become free are
// not reused (they are hidden from reclamation the alive frame). When such a
// frame is eventually freed, it is tombstoned. When releasing a frame the head
// pointer is decremented past all tombstoned frames, releasing the storage
// again.
//
// This is costly, but is incredibly rare.
//
// Template parameters:
//   NumFrameSlots: The number of f_fastlocals each frame should have.
//   NumFrames: The number of frames in contiguous storage.
//
// This class is thread-hostile, non-copyable and non-moveable.
template <int64_t NumFrameSlots, int64_t NumFrames>
class PyFrameObjectCacheT {
 public:
  static constexpr int64_t kNumFrameSlots = NumFrameSlots;
  static constexpr int64_t kNumFrames = NumFrames;

  // A PyFrameObject with room for NumItems fastlocals / value stack entries
  // inside the f_localsplus field.
  //
  // It stores an extra boolean that we use to determine if the object has been
  // tombstoned.
  struct SizedFrameObject {
    explicit SizedFrameObject(PyTypeObject* type_object = nullptr) {
      std::memset(this, 0, sizeof(SizedFrameObject));
      PyObject_InitVar(reinterpret_cast<PyVarObject*>(this), type_object,
                       NumFrameSlots);
    }

    absl::Status Verify() {
      if (frame.f_localsplus[0] != nullptr) {
        return absl::FailedPreconditionError("localsplus was nonzero");
      }

      for (int64_t i = 0; i < NumFrameSlots - 1; ++i) {
        if (localsplus_extra[i] != nullptr) {
          return absl::FailedPreconditionError("localsplus was nonzero");
        }
      }
      return absl::OkStatus();
    }

    PyFrameObject frame;

    // Frame ends with `PyObject* f_localsplus[1]`, so the first fastlocal
    // entry is inside `frame`.
    PyObject* localsplus_extra[NumFrameSlots - 1];

    // True if this frame is tombstoned. This means that the frame has been
    // freed while other frames above it on the stack were still allocated.
    bool tombstoned;
  };

  // The tp_dealloc callback for a SizedFrameObject. This mirrors
  // PyFrameObject's destructor method, and returns the object to the cache
  // by setting `tombstoned = true`.
  //
  // Note that getting here should be rare; usually frame objects' lifetimes
  // are not extended beyond their function, so we don't bother inlining a
  // fastpath if this frame is the head of the list. We just tombstone.
  static void DeallocFrame(PyObject* obj) {
    SizedFrameObject* f = reinterpret_cast<SizedFrameObject*>(obj);
    ClearFrameMembers(f->frame);

    Py_REFCNT(&f->frame) = 1;
    S6_DCHECK_OK(f->Verify());
    f->tombstoned = true;
  }

  PyFrameObjectCacheT() {
    // Our frame object type is a clone of PyFrame_Type, except it is not
    // garbage collectable and has a custom deallocation function.
    frame_object_type_ = PyFrame_Type;
    frame_object_type_.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
    frame_object_type_.tp_dealloc = DeallocFrame;

    storage_.fill(SizedFrameObject(&frame_object_type_));
    head_ = &storage_[0];
  }

  // Cached frames hold a pointer to `&frame_object_type_`, so this is not
  // copyable or moveable.
  PyFrameObjectCacheT(PyFrameObjectCacheT&&) = delete;
  PyFrameObjectCacheT(const PyFrameObjectCacheT&) = delete;

  // Expose the head pointer for the code generator to use. If the head pointer
  // is GetEnd(), the freelist is empty.
  //
  // If GetHead() != GetEnd(), the pointed-to PyFrameObject:
  //   * Has ob_refcnt = 1.
  //   * Has a valid ob_size, ob_type and map/fast attributes field.
  //   * Has all fields set to nullptr apart from f_globals, f_code, f_builtins,
  //       f_back which are undefined.
  SizedFrameObject*& GetHead() {
    if (head_ != GetEnd()) S6_DCHECK_EQ(Py_REFCNT(head_), 1);
    return head_;
  }
  SizedFrameObject* GetHead() const {
    if (head_ != GetEnd()) S6_DCHECK_EQ(Py_REFCNT(head_), 1);
    return head_;
  }

  SizedFrameObject** GetHeadPointer() { return &head_; }

  // Returns true if a SizedFrameObject is tombstoned.
  bool IsTombstoned(SizedFrameObject* f) { return f->tombstoned; }

  // Expose the end pointer for the code generator to use. If GetHead() ==
  // GetEnd(), the freelist is empty. The end pointer never changes across the
  // lifetime of this object.
  const SizedFrameObject* GetEnd() const { return &storage_[NumFrames]; }

  // Allocates a PyFrameObject. This always succeeds. This is a fast path
  // that the code generator should inline.
  //
  // Returns a stolen reference.
  PyFrameObject* Allocate() {
    if (GetHead() == GetEnd()) {
      return AllocateOnHeap();
    }
    // GetHead() points to a valid object.
    return reinterpret_cast<PyFrameObject*>(GetHead()++);
  }

  // Returns the type of PyFrameObjects allocated from this cache.
  const PyTypeObject* GetType() const { return &frame_object_type_; }

  // Called when a PyFrameObject is finished with. This is a fast path that the
  // code generator should inline.
  //
  // The object `f` MUST have been allocated by Allocate() or AllocateOnHeap()
  // due to the use of borrowed interior references.
  void Finished(PyFrameObject* f) {
    if (
        // Is this object at the top of stack?
        reinterpret_cast<SizedFrameObject*>(f) == GetHead() - 1 &&
        // Is the object freeable?
        Py_REFCNT(f) == 1 &&
        // Is this a cached frame?
        Py_TYPE(f) == GetType()) {
      // Back up the head pointer. Normally this only backs up one frame,
      // but we back up over tombstoned frames to release them back into the
      // free list.
      --GetHead();
      S6_DCHECK_OK(GetHead()->Verify());
      while (GetHead() != &storage_[0] && IsTombstoned(GetHead() - 1)) {
        --GetHead();
        S6_DCHECK_EQ(Py_REFCNT(GetHead()), 1);
        S6_DCHECK_OK(GetHead()->Verify());
        GetHead()->tombstoned = false;
      }
      return;
    }

    // Slow path.
    FinishedSlow(f);
  }

  void FinishedSlow(PyFrameObject* f) {
    // If this is a cached frame with refcount 1, we have a faster path.
    if (Py_TYPE(f) == GetType()) {
      // If the refcount is 1, we just need to tombstone this frame.
      if (Py_REFCNT(f) == 1) {
        S6_CHECK_NE(reinterpret_cast<SizedFrameObject*>(f), GetHead() - 1)
            << "How else could we get here?";
        S6_DCHECK_OK(reinterpret_cast<SizedFrameObject*>(f)->Verify());
        reinterpret_cast<SizedFrameObject*>(f)->tombstoned = true;
        return;
      }
    }

    // We are either "leaking" a frame from this cache or we're about to call
    // the PyFrameObject destructor. Either way we must convert borrowed
    // references to full-fat references.
    Py_XINCREF(f->f_back);
    Py_XINCREF(f->f_code);
    Py_XINCREF(f->f_builtins);
    Py_XINCREF(f->f_globals);

    if (Py_TYPE(f) == &PyFrame_Type) {
      // This is a PyFrameObject returned from AllocateOnHeap(). We hold a real
      // reference to it.
      Py_DECREF(f);
    }
  }

  // Allocates and returns a new garbage collected PyFrameObject. This object
  // is not allocated from the freelist.
  static PyFrameObject* AllocateOnHeap() {
    PyFrameObject* f =
        PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, NumFrameSlots);
    f->f_back = nullptr;
    f->f_locals = nullptr;
    f->f_code = nullptr;
    f->f_valuestack = nullptr;
    f->f_stacktop = nullptr;
    f->f_trace = nullptr;
    // TODO: API compatibility between 3.6 and 3.7.
#if PY_MINOR_VERSION < 7
    f->f_exc_type = f->f_exc_traceback = f->f_exc_value = nullptr;
#endif
    f->f_gen = nullptr;
    f->f_lasti = -1;
    f->f_iblock = 0;
    f->f_executing = 0;
    for (int64_t i = 0; i < NumFrameSlots; ++i) {
      f->f_localsplus[i] = nullptr;
    }
    S6_DCHECK_EQ(Py_REFCNT(f), 1);
    return f;
  }

  // Returns the number of cache entries used. Some of these entries may be
  // tombstoned.
  int64_t capacity() const { return GetHead() - &storage_[0]; }

  // Returns the number of cache entries used that are NOT tombstoned.
  int64_t size() const {
    int64_t n = 0;
    for (const SizedFrameObject* i = &storage_[0]; i != GetHead(); ++i) {
      if (!i->tombstoned) ++n;
    }
    return n;
  }

  std::string DebugString() {
    return absl::StrJoin(storage_.begin(), head_, ", ",
                         [](std::string* s, auto f) {
                           absl::StrAppend(s, f.tombstoned ? "dead" : "alive");
                         });
  }

 private:
  // Stores all our frame objects.
  std::array<SizedFrameObject, NumFrames> storage_;

  // The head of the free list, which points either into storage_ or
  // one-past-the-end.
  SizedFrameObject* head_;

  // A type object that is similar to PyFrameObject_Type, but is not garbage
  // collected and has a custom tp_dealloc.
  PyTypeObject frame_object_type_;
};

// Our standard cache holds 128 fastlocals and 512 frames.
using PyFrameObjectCache = PyFrameObjectCacheT<128, 512>;

// Returns the global PyFrameObjectCache.
inline PyFrameObjectCache* GetPyFrameObjectCache() {
  static NoDestructor<PyFrameObjectCache> cache;
  return cache.get();
}

// Similar to PyFrameObjectCache, provides frame and generator objects in a
// single allocation for generator functions.
//
// Generator functions do not tend to operate like a stack - generators are
// more likely to outlive their calling frame than a normal frame. However, in
// most programs the number of outstanding generators is quite low, so we
// optimize for a small number (<=64) of active generators using a simple
// bitmap allocator.
//
// Template parameters:
//   NumFrameSlots: The number of f_fastlocals each frame should have.
//   NumSpillSlots: The number of spill slots that each GeneratorState object
//     should have.
//
// Note that the number of frames in fast storage is always 64.
template <int64_t NumFrameSlots, int64_t NumSpillSlots>
class PyGeneratorFrameObjectCacheT {
 public:
  static constexpr int64_t kNumFrameSlots = NumFrameSlots;
  static constexpr int64_t kNumSpillSlots = NumSpillSlots;
  static constexpr int64_t kNumFrames = 64;

  // A PyFrameObject with room for NumFrameSlots fastlocals / value stack
  // entries inside the f_localsplus field, a PyGeneratorObject, and a
  // GeneratorStateObject with room for NumSpillSlots spill slots.
  //
  // There then are three subobjects that reference each other:
  //   * `frame` has a *BORROWED* reference to gen_object.
  //   * `gen_object` has a reference to frame.
  //   * `frame` has a reference to `generator_state` (in f_valuestack[0]).
  //
  // When a generator is exhausted, its `frame` member is cleared. As a result,
  // we must track the availability of `frame` and `gen_object` separately.
  struct SizedFrameObject {
    explicit SizedFrameObject(PyGeneratorFrameObjectCacheT* parent = nullptr,
                              PyTypeObject* frame_type_object = nullptr,
                              PyTypeObject* gen_type_object = nullptr) {
      std::memset(this, 0, sizeof(SizedFrameObject));  // NOLINT
      PyObject_InitVar(reinterpret_cast<PyVarObject*>(this), frame_type_object,
                       NumFrameSlots);
      PyObject_Init(reinterpret_cast<PyObject*>(&gen_object), gen_type_object);
      PyObject_InitVar(reinterpret_cast<PyVarObject*>(&generator_state),
                       &GeneratorState_Type, NumSpillSlots);
      new (&generator_state.state)
          GeneratorState(&generator_state.spill_slots[0]);

      // This object is *never* destroyed, it is always reinitialized in-place,
      // so leak a reference to it here.
      // This is a very special situation:
      // Not using Py_INCREF macro because the GIL is not held here.
      reinterpret_cast<PyObject*>(&generator_state)->ob_refcnt++;
      this->parent = parent;
      Reinitialize();
    }

    SizedFrameObject& operator=(SizedFrameObject&& other) {
      std::memcpy(this, &other, sizeof(SizedFrameObject));  // NOLINT
      Reinitialize();
      return *this;
    }

    void Reinitialize() {
      frame_freed = false;
      gen_object_freed = false;
      frame.f_gen = reinterpret_cast<PyObject*>(&gen_object);

      // gen_object steals our reference to frame.
      gen_object.gi_frame = &frame;
      new (&generator_state.state)
          GeneratorState(&generator_state.spill_slots[0]);

      frame.f_stacktop = frame.f_valuestack;
    }

    absl::Status Verify() {
      if (frame.f_gen != reinterpret_cast<PyObject*>(&gen_object)) {
        return absl::FailedPreconditionError("frame->f_gen");
      }
      if (gen_object.gi_frame != &frame) {
        return absl::FailedPreconditionError("gen_object.gi_frame");
      }
      if (frame.f_localsplus[0] != nullptr) {
        return absl::FailedPreconditionError("localsplus was nonzero");
      }

      for (int64_t i = 0; i < NumFrameSlots - 1; ++i) {
        if (localsplus_extra[i] != nullptr) {
          return absl::FailedPreconditionError(
              absl::StrCat("localsplus was nonzero: ", i, ", ",
                           Py_TYPE(localsplus_extra[i])->tp_name));
        }
      }
      return absl::OkStatus();
    }

    PyFrameObject frame;

    // Frame ends with `PyObject* f_localsplus[1]`, so the first fastlocal
    // entry is inside `frame`.
    PyObject* localsplus_extra[NumFrameSlots - 1];

    // The PyGenObject associated with `frame`.
    PyGenObject gen_object;

    // S6's information about the generator; spill slots, code_object,
    // resume_pc.
    GeneratorStateObject generator_state;

    // GeneratorStateObject ends with spill_slots[1], so the first spill slot
    // is inside `generator_state`.
    uint64_t spill_slots[NumSpillSlots - 1];

    // Is the frame available to be allocated?
    bool frame_freed;

    // Is the generator available to be allocated?
    bool gen_object_freed;

    PyGeneratorFrameObjectCacheT* parent;
  };

  // The tp_dealloc callback for a SizedFrameObject. This mirrors
  // PyFrameObject's destructor method, and returns the object to the cache
  // if the generator object has already been freed.
  static void DeallocFrame(PyObject* obj) {
    SizedFrameObject* f = reinterpret_cast<SizedFrameObject*>(obj);
    S6_DCHECK(!f->frame_freed);
    ClearFrameMembers(f->frame);

    S6_DCHECK_EQ(Py_REFCNT(reinterpret_cast<PyObject*>(&f->frame)), 0);
    Py_REFCNT(&f->frame) = 1;
    f->frame_freed = true;
    if (f->gen_object_freed) {
      FreeSizedFrame(f);
    }
  }

  static void DeallocGenerator(PyObject* obj) {
    uint64_t obj_address = reinterpret_cast<uint64_t>(obj);
    obj_address -= offsetof(SizedFrameObject, gen_object);
    SizedFrameObject* f = reinterpret_cast<SizedFrameObject*>(obj_address);
    S6_DCHECK_EQ(&f->gen_object, reinterpret_cast<PyGenObject*>(obj));

    if (f->gen_object.gi_frame) {
      _PyGen_Finalize(reinterpret_cast<PyObject*>(&f->gen_object));
    }

    Py_CLEAR(f->gen_object.gi_code);
    Py_CLEAR(f->gen_object.gi_name);
    Py_CLEAR(f->gen_object.gi_qualname);
    Py_REFCNT(&f->gen_object) = 1;
    f->gen_object_freed = true;
    if (f->frame_freed) {
      FreeSizedFrame(f);
    } else if (f->gen_object.gi_frame) {
      f->gen_object.gi_frame->f_gen = nullptr;
      S6_CHECK_EQ(f->gen_object.gi_frame, &f->frame);
      DeallocFrame(reinterpret_cast<PyObject*>(&f->frame));
    }
  }

  PyGeneratorFrameObjectCacheT() {
    // Our frame object type is a clone of PyFrame_Type, except it is not
    // garbage collectable and has a custom deallocation function.
    frame_object_type_ = PyFrame_Type;
    frame_object_type_.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
    frame_object_type_.tp_dealloc = DeallocFrame;

    // Similarly our GeneratorObject type is a clone of PyGen_Type with a custom
    // destructor and isn't GC'd.
    gen_object_type_ = PyGen_Type;
    gen_object_type_.tp_flags &= ~Py_TPFLAGS_HAVE_GC;
    gen_object_type_.tp_dealloc = DeallocGenerator;

    // std::array does not have a fill() method that can construct in-place, so
    // we use a loop instead (SizedFrameObject isn't copyable and we don't want
    // it to be).
    for (auto& x : storage_) {
      x = SizedFrameObject(this, &frame_object_type_, &gen_object_type_);
    }
  }

  // Cached frames hold a pointer to `&frame_object_type_`, so this is not
  // copyable or moveable.
  PyGeneratorFrameObjectCacheT(PyGeneratorFrameObjectCacheT&&) = delete;
  PyGeneratorFrameObjectCacheT(const PyGeneratorFrameObjectCacheT&) = delete;

  // Allocates a SizedFrameObject. If no cached objects are available, this
  // returns nullptr.
  //
  // Returns a stolen reference.
  SizedFrameObject* AllocateOrNull() {
    if (ready_ == 0) {
      return nullptr;
    }
    // CountTrailingZeroes gives us the index of the first one in the bitmap.
    int64_t index = absl::countr_zero(ready_);
    ready_ &= ~(1ULL << index);
    auto f = &storage_[index];
    S6_CHECK_EQ(Py_REFCNT(f), 1);
    return f;
  }

  // Returns the type of PyFrameObjects allocated from this cache.
  const PyTypeObject* GetFrameType() const { return &frame_object_type_; }

  // Returns the type of PyGenObjects allocated from this cache.
  const PyTypeObject* GetGenType() const { return &gen_object_type_; }

  // Returns the number of cache entries used.
  int64_t size() const { return kNumFrames - absl::popcount(ready_); }

  std::string DebugString() {
    std::vector<std::string> v;
    for (int64_t i = 0; i < kNumFrames; ++i) {
      v.push_back((ready_ & (1ULL << i)) ? "dead" : "alive");
    }
    return absl::StrJoin(v, ", ");
  }

  GeneratorState* GetOrCreateGeneratorState(PyFrameObject* pyframe,
                                            int64_t num_spill_slots) {
    PyObject* gen = pyframe->f_gen;
    if (Py_TYPE(gen) == &gen_object_type_) {
      SizedFrameObject* sf = reinterpret_cast<SizedFrameObject*>(pyframe);
      pyframe->f_valuestack[0] =
          reinterpret_cast<PyObject*>(&sf->generator_state);
      pyframe->f_stacktop = &pyframe->f_valuestack[1];
      return &sf->generator_state.state;
    } else {
      return GeneratorState::Create(pyframe, num_spill_slots);
    }
  }

 private:
  static void FreeSizedFrame(SizedFrameObject* f) {
    f->Reinitialize();
    PyGeneratorFrameObjectCacheT* parent = f->parent;
    int64_t bitmap_index = std::distance(parent->storage_.begin(), f);
    parent->ready_ |= 1ULL << bitmap_index;
    S6_DCHECK_OK(f->Verify());
  }

  // Stores all our frame objects.
  std::array<SizedFrameObject, kNumFrames> storage_;

  // The storage bitmap. kNumFrames is always 64, so there is one bit per frame.
  // a `1` means the frame is available, `0` means it is active.
  uint64_t ready_ = ~0ULL;

  // A type object that is similar to PyFrameObject_Type, but is not garbage
  // collected and has a custom tp_dealloc.
  PyTypeObject frame_object_type_;

  // A type object that is similar to PyGenObject_Type, but is not garbage
  // collected and has a custom tp_dealloc.
  PyTypeObject gen_object_type_;
};

// Our standard cache holds 128 fastlocals and 512 spill slots.
using PyGeneratorFrameObjectCache = PyGeneratorFrameObjectCacheT<128, 512>;

// Returns the global PyGeneratorFrameObjectCache.
inline PyGeneratorFrameObjectCache* GetPyGeneratorFrameObjectCache() {
  static NoDestructor<PyGeneratorFrameObjectCache> cache;
  return cache.get();
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_RUNTIME_PYFRAME_OBJECT_CACHE_H_
