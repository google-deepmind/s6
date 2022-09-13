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

#include "utils/channel.h"

#include <atomic>
#include <memory>
#include <thread>  // NOLINT

#include "absl/synchronization/mutex.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace deepmind::s6::thread {
namespace {
TEST(ChannelTest, ReadFromAClosedChannel) {
  Channel<int> channel(0);
  channel.writer().Close();

  int i = 0;
  EXPECT_FALSE(channel.reader().Read(&i));
}

// Write numbers [1...count] into the channel with `count` writers.
std::vector<std::thread> GenerateWriters(int count, Channel<int>& channel) {
  std::vector<std::thread> writer_threads;
  writer_threads.reserve(count);
  for (int i = 1; i <= count; ++i) {
    writer_threads.emplace_back([i, &writer = channel.writer()]() mutable {
      writer.Write(std::move(i));
    });
  }
  return writer_threads;
}

// Read from the channel and add read values to the sum.
// Use a mutex and condition variable to synchronize access to `sum` and
// ensure that multiple readers read values.
std::vector<std::thread> GenerateReaders(int count, Channel<int>& channel,
                                         int& sum, absl::Mutex& mtx,
                                         absl::CondVar& cv) {
  std::vector<std::thread> reader_threads;
  reader_threads.reserve(count);

  for (int i = 0; i < count; ++i) {
    reader_threads.emplace_back(
        [&sum, &mtx, &cv, &reader = channel.reader()]() mutable {
          int value = 0;
          while (reader.Read(&value)) {
            absl::ReleasableMutexLock lock(&mtx);
            sum += value;
            cv.SignalAll();
            cv.Wait(&mtx);
          }
        });
  }
  return reader_threads;
}

TEST(ChannelTest, ReadAndWriteFromMultipleThreads) {
  Channel<int> channel(4);

  // Write numbers [1...10] into the channel with 10 writers.
  std::vector<std::thread> writer_threads = GenerateWriters(10, channel);

  // Read from the channel and add read values to the sum.
  int sum = 0;
  absl::Mutex mtx;
  absl::CondVar cv;
  std::vector<std::thread> reader_threads =
      GenerateReaders(2, channel, sum, mtx, cv);

  // Sum [1...10] is 55.
  // Wait for readers to accumulate the correct sum.
  while (true) {
    absl::ReleasableMutexLock lock(&mtx);
    if (sum == 55) break;
    cv.SignalAll();
    cv.Wait(&mtx);
  }
  channel.writer().Close();
  cv.SignalAll();

  for (auto& t : writer_threads) t.join();
  for (auto& t : reader_threads) t.join();

  // Given the wait on the sum above this is somewhat redundant.
  EXPECT_EQ(sum, 55);
}

TEST(ChannelTest, ReadAndWriteFromMultipleThreadsEarlyClose) {
  Channel<int> channel(4);

  // Write numbers [1...10] into the channel with 10 writers.
  std::vector<std::thread> writer_threads = GenerateWriters(10, channel);

  // Read from the channel and add read values to the sum.
  int sum = 0;
  absl::Mutex mtx;
  absl::CondVar cv;
  std::vector<std::thread> reader_threads =
      GenerateReaders(2, channel, sum, mtx, cv);

  for (auto& t : writer_threads) t.join();

  // Close the channel once all writers are done without checking the sum.
  channel.writer().Close();
  cv.SignalAll();

  for (auto& t : reader_threads) t.join();

  EXPECT_LE(sum, 55);
}

}  // namespace
}  // namespace deepmind::s6::thread
