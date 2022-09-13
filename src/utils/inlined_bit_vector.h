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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_INLINED_BIT_VECTOR_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_INLINED_BIT_VECTOR_H_

#include <algorithm>
#include <cstddef>

#include "absl/container/inlined_vector.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "utils/logging.h"
#include "utils/mathutil.h"

namespace deepmind::s6 {

// An InlinedBitVector<NBITS> is a resizable bitvector that provides
// storage for bitvectors of length <= NBITS inline without requiring
// any heap allocation.
// Resizing beyond NBITS may incur heap allocation but is allowed.
//
// InlinedBitVector is implemented with an underlying InlineVector of unsigned
// 32-bit ints and bit manipulation.
//
// InlinedBitVector does not offer iterators or mutating `operator[]`.

template <size_t NBITS>
class InlinedBitVector {
 public:
  InlinedBitVector() = default;

  // Constructs a new bit vector that is "num_bits" bits in size.  All
  // bits are initially set to 0.
  explicit InlinedBitVector(size_t num_bits) { resize(num_bits); }

  size_t count() const {
    size_t count = 0;
    for (uint32_t word : words_) count += absl::popcount(word);
    return count;
  }

  size_t size() const { return num_bits_; }

  // Sets (sets to one) the bit at the specified index.
  //
  // REQUIRES: index < size()
  void set_bit(size_t index) {
    S6_DCHECK_LT(index, num_bits_);

    size_t word_offset = index / 32;
    size_t bit_index = index % 32;
    words_[word_offset] |= (1u << bit_index);
  }

  // Clears (sets to zero) the bit at the specified index.
  //
  // REQUIRES: index < size()
  void clear_bit(size_t index) {
    S6_DCHECK_LT(index, num_bits_);

    size_t word_offset = index / 32;
    size_t bit_index = index % 32;
    words_[word_offset] &= ~(1u << bit_index);
  }

  // Gets the bit at the specified index.
  //
  // REQUIRES: index < size()
  bool get_bit(size_t index) const {
    S6_DCHECK_LT(index, num_bits_);

    const size_t word = index / 32;
    const size_t bit_index = index % 32;
    return (words_[word] & (1u << bit_index)) ? true : false;
  }

  // Convenience method that is the same same as "get_bit".
  bool operator[](size_t index) const { return get_bit(index); }

  // Set (sets to one) the bits at the [start, end) subsequence.
  //
  // REQUIRES: start >= 0
  // REQUIRES: end <= size()
  // REQUIRES: start <= end
  void SetBits(size_t start, size_t end) {
    S6_DCHECK_GE(start, 0);
    S6_DCHECK_LE(start, end);
    S6_DCHECK_LE(end, num_bits_);

    if (num_bits_ == 0) return;

    const size_t start_word = start / 32;
    const size_t start_word_bit_index = start % 32;
    const size_t end_word = end / 32;
    const size_t end_word_bit_index = end % 32;
    const uint32_t start_word_mask = ~((1u << start_word_bit_index) - 1);
    const uint32_t end_word_mask = (1u << end_word_bit_index) - 1;

    // Resize the bit vector so that it is "num_bits" in length.

    if (start_word == end_word) {
      words_[start_word] |= (start_word_mask & end_word_mask);
    } else {
      words_[start_word] |= start_word_mask;
      for (int i = start_word + 1; i < end_word; ++i) {
        words_[i] = ~(0u);
      }
      if (end_word_bit_index != 0) {
        words_[end_word] |= end_word_mask;
      }
    }
  }

  // Clears (sets to zero) the bits at the [start, end) subsequence.
  //
  // REQUIRES: start >= 0
  // REQUIRES: start <= end
  // REQUIRES: end <= size()
  void ClearBits(size_t start, size_t end) {
    S6_DCHECK_GE(start, 0);
    S6_DCHECK_LE(start, end);
    S6_DCHECK_LE(end, num_bits_);

    const size_t start_word_index = start / 32;
    const size_t start_word_bit_index = start % 32;
    const size_t end_word_index = end / 32;
    const size_t end_word_bit_index = end % 32;

    const uint32_t start_word_mask = (1u << start_word_bit_index) - 1;
    const uint32_t end_word_mask = ~((1u << end_word_bit_index) - 1);

    // Begin and end bits may lie in the same word.
    if (start_word_index == end_word_index) {
      words_[start_word_index] &= (start_word_mask | end_word_mask);
    } else {
      words_[start_word_index] &= start_word_mask;
      for (int i = start_word_index + 1; i < end_word_index; ++i) {
        words_[i] = 0;
      }
      if (end_word_bit_index != 0) {
        words_[end_word_index] &= end_word_mask;
      }
    }
  }

  // Resizes the bit vector so that it is "num_bits" in length.
  // If this requires adding extra bits then the extra bits will be set to zero.
  void resize(size_t num_bits) {
    // resize the underlying vector, newly added words will be all zeroes
    const size_t bits_needed = (num_bits + 31) / 32;
    words_.resize(bits_needed);

    const size_t old_num_bits = num_bits_;
    num_bits_ = num_bits;

    // If the bitvector got bigger, we may need to zero bits in the old last
    // word.
    if (num_bits_ <= old_num_bits) return;

    // Newly added words are all zero, so we only need to zero higher bits in
    // existing words.
    size_t old_last_bit_index = old_num_bits % 32;
    if (old_last_bit_index == 0) return;

    // We have some bits to zero out.
    // Create a mask and logical-and its inverse with the last old bit to
    // zero-out the new entries in the last old bit (previous resizes may mean
    // they contain junk).
    size_t mask = (~0u) << old_last_bit_index;
    words_[old_num_bits / 32] &= ~mask;
  }

  // Removes all bits from the bit vector and set its length to 0 bits.
  inline void clear() {
    words_.clear();
    num_bits_ = 0;
  }

  // Sets "this" to be the union of "this" and "other". The bitmaps do not
  // have to be the same size. If "other" is smaller than "this", all the
  // missing bits in "other" are assumed to be zero.
  void Union(const InlinedBitVector& other) {
    const int n_words = std::min(words_.size(), other.words_.size());
    for (int i = 0; i < n_words; ++i) {
      words_[i] |= other.words_[i];
    }
  }

  // Sets "this" to be the intersection of "this" and "other". If "other" is
  // smaller than "this", all the missing bits in "other" are assumed to be
  // zero.
  void Intersection(const InlinedBitVector& other) {
    const int n_words = std::min(words_.size(), other.words_.size());
    for (int i = 0; i < n_words; ++i) {
      words_[i] &= other.words_[i];
    }
    for (int i = n_words; i < words_.size(); ++i) {
      words_[i] = 0;
    }
  }

  // Sets "this" to be the difference of "this" and "other". If "other" is
  // smaller than "this", all the missing bits in "other" are assumed to be
  // zero.
  void Difference(const InlinedBitVector& other) {
    const int n_words = std::min(words_.size(), other.words_.size());
    for (int i = 0; i < n_words; ++i) {
      words_[i] &= ~other.words_[i];
    }
  }

  // Compares the bit vector "this" to "other", returns true if they contain the
  // same bit values.
  bool Equals(const InlinedBitVector& other) const {
    // Compare size.
    if (num_bits_ != other.num_bits_) {
      return false;
    }

    // Compare full-words.
    const int num_full_words = num_bits_ / 32;
    if (absl::Span<const uint32_t>(words_.data(), num_full_words) !=
        absl::Span<const uint32_t>(other.words_.data(), num_full_words))
      return false;

    // Compare non-junk in last word.
    uint32_t last_word_bit_index = num_bits_ % 32;
    if (last_word_bit_index == 0) {
      return true;
    }
    uint32_t last_word_mask = (1u << last_word_bit_index) - 1;
    return ((words_[num_full_words] ^ other.words_[num_full_words]) &
            last_word_mask) == 0;
  }

  friend bool operator==(const InlinedBitVector& lhs,
                         const InlinedBitVector& rhs) {
    return lhs.Equals(rhs);
  }

  friend bool operator!=(const InlinedBitVector& lhs,
                         const InlinedBitVector& rhs) {
    return !lhs.Equals(rhs);
  }

 private:
  absl::InlinedVector<uint32_t, (NBITS + 31) / 32> words_;
  size_t num_bits_ = 0;
};

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_INLINED_BIT_VECTOR_H_
