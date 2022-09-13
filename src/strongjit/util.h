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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_UTIL_H_

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "core_util.h"
#include "utils/logging.h"
#include "utils/range.h"

namespace deepmind::s6 {

// A MultiLineVector is a file-like container of T but where individual lines
// can be accessed in constant time.
//
// For example:
// - E1 E2 E3
// - E4
// -
// - E6
//
// where all Ei are of type T.
//
// `N` is just for specifying the target size of the underlying
// `absl::InlinedVector`.
//
// The MultiLineVector behaves on its own as a contiguous container of all
// the line concatenated but the `line` method can be used to get a specific
// `Line` container.
template <typename T, size_t N>
class MultiLineVector {
 public:
  using underlying = absl::InlinedVector<T, N>;
  using value_type = T;
  using iterator = typename underlying::iterator;
  using const_iterator = typename underlying::const_iterator;
  using span_type = absl::Span<T>;
  using const_span_type = absl::Span<const T>;

  MultiLineVector() = default;
  MultiLineVector(const MultiLineVector&) = default;
  MultiLineVector(MultiLineVector&&) = default;
  MultiLineVector& operator=(const MultiLineVector&) = default;
  MultiLineVector& operator=(MultiLineVector&&) = default;

  iterator begin() { return storage_.begin(); }
  iterator end() { return storage_.end(); }
  const_iterator begin() const { return storage_.begin(); }
  const_iterator end() const { return storage_.end(); }

  size_t size() const { return storage_.size(); }
  bool empty() const { return storage_.empty(); }
  void clear() {
    storage_.clear();
    lines_.clear();
  }

  T* data() { return storage_.data(); }
  const T* data() const { return storage_.data(); }

  span_type span() { return absl::MakeSpan(*this); }
  const_span_type span() const { return *this; }

  // Pushes on the last line.
  void push_back(const T& value) { storage_.push_back(value); }
  void push_back(T&& value) { storage_.push_back(std::move(value)); }

  // Pops from the last line. The last line must not be empty.
  void pop_back() {
    S6_DCHECK(!empty());
    S6_DCHECK(lines_.empty() || lines_.back() < size());
    storage_.pop_back();
  }

  // Returns the first element.
  T& front() { return storage_.front(); }
  const T& front() const { return storage_.front(); }

  // Returns the last element.
  T& back() { return storage_.back(); }
  const T& back() const { return storage_.back(); }

  // Adds a new empty line at the end of the container.
  void push_line() { lines_.push_back(storage_.size()); }

  // Pops the last line and all its elements.
  void pop_line() {
    S6_DCHECK(!lines_.empty());
    S6_DCHECK_GE(storage_.size(), lines_.back());
    storage_.resize(lines_.back());
    lines_.pop_back();
  }

  // Inserts a value at a certain position, in front of the pointed element.
  // It is not possible to insert at the end of a line or on an empty line
  // with this method. Use Line::insert on the correct line to do that.
  iterator insert(const_iterator pos, const T& value) {
    line_move(pos, 1);
    return storage_.insert(pos, value);
  }
  iterator insert(const_iterator pos, T&& value) {
    line_move(pos, 1);
    return storage_.insert(pos, std::move(value));
  }
  iterator insert(const_iterator pos, size_t count, const T& value) {
    line_move(pos, count);
    return storage_.insert(pos, count, value);
  }
  iterator insert_at(size_t index, const T& value) {
    return insert(begin() + index, value);
  }
  iterator insert_at(size_t index, T&& value) {
    return insert(begin() + index, std::move(value));
  }

  // Erases a element at the specified position.
  iterator erase(const_iterator pos) {
    line_move(pos, -1);
    return storage_.erase(pos);
  }
  iterator erase_at(size_t index) { return erase(begin() + index); }

  // Resizes the container.
  // When upsizing, all elements are added to the last line.
  // When downsizing, all lines that started after the new size are deleted.
  void resize(size_t count) {
    storage_.resize(count);
    auto it = absl::c_upper_bound(lines_, count);
    lines_.erase(it, lines_.end());
  }

  T& operator[](size_t i) { return storage_[i]; }
  const T& operator[](size_t i) const { return storage_[i]; }

  // This class represents a single line of a MultiLineVector.
  //
  // In practice it represent an end of line, since it can start at a non-zero
  // column offset.
  //
  // Is is valid only as long as the parent MultiLineVector and stay valid
  // whatever operation (insertion, deletion, ...) is done on the parent
  // container except if the referenced line is deleted.
  class Line {
   public:
    using underlying = MultiLineVector<T, N>::underlying;
    using value_type = MultiLineVector<T, N>::value_type;
    using iterator = MultiLineVector<T, N>::iterator;
    using const_iterator = MultiLineVector<T, N>::const_iterator;
    using span_type = MultiLineVector<T, N>::span_type;
    using const_span_type = MultiLineVector<T, N>::const_span_type;

    // Returns a sub line with the given offset applied to the column.
    // The new offset must be smaller than the line size in order not to
    // go beyond the end of the line. Giving a column offset equal to the line
    // size will result in an empty line view (but one can still push_back at
    // the end)
    Line SubLine(uint32_t column_offset) {
      Line res = *this;
      res.column_ += column_offset;
      S6_CHECK_LE(res.StartIndex(), res.EndIndex());
      return res;
    }

    // Returns the start index of that line in the original container
    size_t StartIndex() const { return sv_.line_start(line_index_) + column_; }

    // Returns the end index (excluded) of that line in the original container.
    size_t EndIndex() const { return sv_.line_end(line_index_); }

    iterator begin() { return sv_.begin() + StartIndex(); }
    iterator end() { return sv_.begin() + EndIndex(); }
    const_iterator begin() const { return sv_.begin() + StartIndex(); }
    const_iterator end() const { return sv_.begin() + EndIndex(); }

    size_t size() const {
      return std::max<size_t>(EndIndex() - StartIndex(), 0);
    }
    bool empty() const { return size() == 0; }
    void clear() { resize(0); }

    // Returns the first element.
    T& front() { return sv_[StartIndex()]; }
    const T& front() const { return sv_[StartIndex()]; }

    // Returns the last element.
    T& back() { return sv_[EndIndex() - 1]; }
    const T& back() const { return sv_[EndIndex() - 1]; }

    T* data() { return sv_.data() + StartIndex(); }
    const T* data() const { return sv_.data() + StartIndex(); }

    span_type span() { return absl::MakeSpan(*this); }
    const_span_type span() const { return *this; }

    span_type subspan(size_t pos, size_t len = span_type::npos) {
      return span().subspan(pos, len);
    }
    const_span_type subspan(size_t pos, size_t len = span_type::npos) const {
      return span().subspan(pos, len);
    }

    iterator insert(const_iterator pos, const T& value) {
      S6_DCHECK(pos >= begin() && pos <= end());
      line_move(pos, 1);
      return sv_.storage_.insert(pos, value);
    }
    iterator insert(const_iterator pos, T&& value) {
      S6_DCHECK(pos >= begin() && pos <= end());
      line_move(pos, 1);
      return sv_.storage_.insert(pos, std::move(value));
    }
    iterator insert(const_iterator pos, size_t count, const T& value) {
      S6_DCHECK(pos >= begin() && pos <= end());
      line_move(pos, count);
      return sv_.storage_.insert(pos, count, value);
    }
    iterator insert_at(size_t index, const T& value) {
      return insert(begin() + index, value);
    }
    iterator insert_at(size_t index, T&& value) {
      return insert(begin() + index, std::move(value));
    }

    iterator erase(const_iterator pos) {
      S6_DCHECK(pos >= begin() && pos < end());
      line_move(pos, -1);
      return sv_.storage_.erase(pos);
    }
    iterator erase(const_iterator first, const_iterator last) {
      S6_DCHECK(first >= begin() && first <= last && last <= end());
      if (last != first) line_move(first, -(last - first));
      return sv_.storage_.erase(first, last);
    }
    iterator erase_at(size_t index) { return erase(begin() + index); }

    void push_back(const T& value) { insert(end(), value); }
    void push_back(T&& value) { insert(end(), std::move(value)); }
    void pop_back() { erase(end() - 1); }

    void resize(size_t count) {
      if (count > size()) {
        insert(end(), count - size(), T());
      } else if (count < size()) {
        erase(begin() + count, end());
      }
    }

    T& operator[](size_t i) { return sv_[i + StartIndex()]; }
    const T& operator[](size_t i) const { return sv_[i + StartIndex()]; }

   private:
    void line_move(const_iterator pos, int32_t offset) {
      for (uint32_t& line_start :
           MakeRange(sv_.lines_.begin() + line_index_, sv_.lines_.end())) {
        line_start += offset;
      }
    }

    Line(MultiLineVector& sv, uint32_t line_index, uint32_t column)
        : sv_(sv), line_index_(line_index), column_(column) {}

    friend MultiLineVector;
    MultiLineVector& sv_;
    uint32_t line_index_;
    uint32_t column_;
  };

  // Returns a line, optionally starting at a non-zero column.
  Line line(size_t line_index, size_t column = 0) {
    return Line(*this, line_index, column);
  }

  // Returns a line as a span, optionally starting at a non-zero column.
  absl::Span<T> line_span(size_t line_index, size_t column = 0) {
    return absl::MakeSpan(data() + line_start(line_index) + column,
                          line_size(line_index) - column);
  }
  absl::Span<T const> line_span(size_t line_index, size_t column = 0) const {
    return absl::MakeSpan(data() + line_start(line_index) + column,
                          line_size(line_index) - column);
  }

  // Returns the start index of a given line.
  size_t line_start(size_t line) const {
    S6_DCHECK_LT(line, line_num());
    if (line) return lines_[line - 1];
    return 0;
  }
  size_t line_end(size_t line) const {
    S6_DCHECK_LT(line, line_num());
    if (line == lines_.size()) return size();
    return lines_[line];
  }
  size_t line_size(size_t line) const {
    return line_end(line) - line_start(line);
  }

  // Returns the number of lines.
  size_t line_num() const { return lines_.size() + 1; }

 private:
  // Move all line after the iterator pos from the given offset.
  void line_move(const_iterator pos, int32_t offset) {
    uint32_t index = pos - begin();
    auto line = absl::c_upper_bound(lines_, index);
    for (uint32_t& line_start : MakeRange(line, lines_.end())) {
      line_start += offset;
    }
  }

  underlying storage_;
  // List of all lines starts. The first line start is 0 and thus is not in
  // `lines_` so there is an offset of one between a line number
  // as provided by the external API (starting at 0) and the indices in this
  // array (where the second line is numbered 0).
  absl::InlinedVector<uint32_t, 4> lines_;
};

// Creates a vector of the individual BytecodeInstructions that make up the
// PyCodeObject.
std::vector<BytecodeInstruction> ExtractInstructions(PyCodeObject* co);

// Represents a Try handler block, used by a SafepointInst. This is a
// corollary to CPython's PyTryBlock struct.
// LINT.IfChange
class TryHandler {
 public:
  enum Kind {
    // Pushed by a SETUP_EXCEPT bytecode.
    kExcept,
    // Pushed by a SETUP_LOOP bytecode.
    kLoop,
    // Pushed by a SETUP_FINALLY bytecode.
    kFinally,
    // Implicitly pushed on entry to an exception handler; marks the start of
    // the exception handler.
    kExceptHandler,
    // Implicitly pushed on entry to a finally handler; marks the start of
    // the finally handler. This block is a static representation of a
    // optional EXCEPT_HANDLER in a finally block. It does not exist as-is
    // in normal CPython.
    //
    // This block my also have a pc_continue_ value to show the target
    // to jump to in case of continue.
    //
    // It also has a boolean flag telling if the falltrough path
    // has already popped the ExceptHandler block below this block.
    // This information is stored here for deoptimisation.
    kFinallyHandler,
  };

  Kind kind() const { return kind_; }

  // Returns the stack height on entry to and exit from the handler. Any
  // pushed items are pushed on top of this stack height.
  int64_t stack_height() const { return stack_height_; }

  // Returns the program address of the handler target.
  PcValue pc_value() const { return pc_value_; }

  // Returns the pc_continue of this handler. This is only relevant for
  // kFinallyHandler. When there exist a continue instruction
  // in a kFinally block, it can call AnalysisContext::SetContinueTarget
  // to set the value of pc_continue inside the finally handler so that
  // the finally handler know where to jump in case of continue.
  absl::optional<PcValue> pc_continue() const { return pc_continue_; }

  // Set the pc_continue value for a kFinallyHandler.
  // This handler must be a kFinallyHandler
  // The value cannot be changed once set. If it is already set, then
  // it is only allowed to set the same value again.
  void set_pc_continue(PcValue pc_continue) {
    S6_CHECK(kind_ == kFinallyHandler);
    if (pc_continue_) {
      S6_CHECK_EQ(pc_continue_->AsOffset(), pc_continue.AsOffset());
    } else {
      pc_continue_ = pc_continue;
    }
  }

  // This flag is only for kFinallyHandler.
  // If this flag is set, it means that there is a kExceptHandler below this
  // finally handler, and that this except handler has already been popped when
  // entering the finally on a fallthrough path, but not when entering the
  // finally on an exceptional path.
  //
  // This means that if the finally was entered on a falltrough path, the
  // kExceptHandler below needs to be skipped when unwinding.
  bool finally_fallthrough_popped_handler() const {
    return finally_fallthrough_popped_handler_;
  }

  // Sets the finally_fallthrough_popped_handler flag. This flag cannot be
  // unset.
  void set_finally_fallthrough_popped_handler() {
    S6_CHECK(kind_ == kFinallyHandler);
    finally_fallthrough_popped_handler_ = true;
  }

  TryHandler(Kind kind, PcValue pc_value, int64_t stack_height)
      : kind_(kind), pc_value_(pc_value), stack_height_(stack_height) {}

 private:
  Kind kind_;
  PcValue pc_value_;
  absl::optional<PcValue> pc_continue_;              // only for kFinallyHandler
  bool finally_fallthrough_popped_handler_ = false;  // only for kFinallyHandler
  int64_t stack_height_;
};
// LINT.ThenChange(formatter.cc, parser.cc)

// A constant length bit vector that is compile-time initializable.
template <size_t length>
class FixedLengthBitVector {
 public:
  constexpr FixedLengthBitVector(
      const std::initializer_list<size_t>& set_bits) {
    for (auto bit : set_bits) {
      bits_[bit >> 6] |= 1UL << (bit & 0x3f);
    }
  }

  constexpr void SetBit(size_t bit) { bits_[bit >> 6] |= 1UL << (bit & 0x3f); }

  constexpr bool IsSet(size_t bit) const {
    return bits_[bit >> 6] & 1UL << (bit & 0x3f);
  }

 private:
  static constexpr size_t kLength = (length + 63) >> 6;
  uint64_t bits_[kLength]{};
};

// Converts a TryHandler::Kind into its CPython equivalent.
int TryHandlerKindToOpcode(TryHandler::Kind kind);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_UTIL_H_
