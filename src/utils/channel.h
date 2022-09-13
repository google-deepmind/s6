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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_CHANNEL_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_CHANNEL_H_

#include <cstddef>
#include <queue>
#include <type_traits>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "utils/logging.h"

namespace deepmind::s6::thread {

// Channel allows typed, unidirectional, synchronized communication.
//
// The type `T` for Channel<T> must be movable.
//
// Channel exposes a Reader and Writer which can be used by sinks and sources.
//
// The Reader blocks while waiting for data or for the channel to be closed.
//
// The Writer closes the channel when it is done writing data.
// Data written is stored in a queue up to the capacity of the channel.
// Once the queue is full, further writes are blocked.
//
// Care must be taken to ensure that writers do not write to a closed channel.

template <typename T>
class Channel;

template <typename T>
class Reader {
 public:
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  Reader(Reader&&) = delete;
  Reader& operator=(Reader&&) = delete;
  ~Reader() = default;

  // Reads the next value from the underlying channel.
  // Blocks until there is a value or the underlying channel is closed.
  // Returns true if a value is produced, false if the underlying channel is
  // closed.
  bool Read(T* t) { return channel_->Read(t); }

 private:
  // Only Channel can create a Reader.
  explicit Reader(Channel<T>* channel) : channel_(channel) {}
  friend class Channel<T>;
  Channel<T>* channel_;
};

template <typename T>
class Writer {
 public:
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  Writer(Writer&&) = delete;
  Writer& operator=(Writer&&) = delete;
  ~Writer() = default;

  // Closes the underlying channel preventing further writes and signals to
  // readers that there are no more values.
  // Can only be called once.
  void Close() { channel_->Close(); }

  // Writes to the underlying channel.
  // Blocks until the value is written.
  // Channel must not be closed.
  void Write(T&& t) { channel_->Write(std::move(t)); }

 private:
  // Only Channel can create a Writer.
  friend class Channel<T>;
  explicit Writer(Channel<T>* channel) : channel_(channel) {}

  Channel<T>* channel_;
};

template <typename T>
class Channel {
  static_assert(sizeof(T), "T in Channel<T> must be a complete type.");

  static_assert(std::is_move_assignable_v<T>,
                "T in Channel<T> must be MoveAssignable.");

 public:
  // Constructs a channel with a queue of the specified capacity.
  // Writes to a queue at full cacapcity will block until values are read.
  explicit Channel(size_t capacity)
      : capacity_(capacity), reader_(this), writer_(this), closed_(false) {}

  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;
  Channel(Channel&&) = delete;
  Channel& operator=(Channel&&) = delete;
  ~Channel() = default;

  // Returns the writer.
  Writer<T>& writer() { return writer_; }

  // Returns the reader.
  Reader<T>& reader() { return reader_; }

 private:
  void Close() {
    absl::ReleasableMutexLock lock(&mutex_);
    S6_CHECK(!closed_) << "Called Close() on a closed channel";
    closed_ = true;
    lock.Release();
    condition_variable_.Signal();
  }

  void Write(T&& t) {
    absl::ReleasableMutexLock lock(&mutex_);
    while (!closed_ && queue_.size() >= capacity_) {
      condition_variable_.Wait(&mutex_);
    }
    S6_CHECK(!closed_) << "Attempted write to a closed Channel";
    queue_.push(std::move(t));
    lock.Release();
    condition_variable_.Signal();
  }

  bool Read(T* t) {
    absl::ReleasableMutexLock lock(&mutex_);
    while (!closed_ && queue_.empty()) {
      condition_variable_.Wait(&mutex_);
    }
    if (closed_) {
      lock.Release();
      condition_variable_.Signal();
      return false;
    }
    *t = std::move(queue_.front());
    queue_.pop();
    lock.Release();
    condition_variable_.Signal();
    return true;
  }

  // Reader and Writer need access to internals.
  friend class Reader<T>;
  friend class Writer<T>;

  const size_t capacity_;
  Reader<T> reader_;
  Writer<T> writer_;
  absl::Mutex mutex_;
  absl::CondVar condition_variable_;
  std::queue<T> queue_ ABSL_GUARDED_BY(mutex_);
  bool closed_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace deepmind::s6::thread

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_CHANNEL_H_
