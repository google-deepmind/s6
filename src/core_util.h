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

#ifndef THIRD_PARTY_DEEPMIND_S6_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_UTIL_H_

#include <Python.h>

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "asmjit/asmjit.h"
#include "utils/logging.h"
#include "utils/status_macros.h"

// Changes a function's calling convention such that it never corrupts
// caller-saved registers. See
// https://clang.llvm.org/docs/AttributeReference.html#preserve-most
//
// NOTE: rax is also preserved across the call, so all functions with this
// attribute must return void. r11 is not preserved, so with careful
// construction it is possible to return a result in r11.
#define S6_ATTRIBUTE_PRESERVE_CALLER_SAVED_REGISTERS \
  __attribute__((__preserve_most__))

namespace deepmind::s6 {

////////////////////////////////////////////////////////////////////////////////
// PcValue

// Encapsulates the program counter for the interpreter. CPython bytecode
// programs are a sequence of 16-bit (sizeof(_Py_CODEUNIT)) values. Program
// offsets are frequently in units of bytes rather than code units, but
// internally our program counter is an index into an array of code units.
//
// This conversion back and forth can make code that modifies PC hard to read.
//
// This class refers to two granules:
// Assume `_Py_CODEUNIT program[]`;
//   `Offset`: Byte offset into the code array: ((char*)program)[offset].
//   `Index`: Index into the code array: program[index].
//
class PcValue {
 public:
  // Instead of requiring internal Python headers to use this class, declare
  // the type of _Py_CODEUNIT here.
  using Py_CODEUNIT = uint16_t;

  static PcValue FromOffset(int offset) { return PcValue(offset); }
  static PcValue FromIndex(int index) {
    return PcValue(index * sizeof(Py_CODEUNIT));
  }

  int AsOffset() const { return offset_; }
  int AsIndex() const { return offset_ / sizeof(Py_CODEUNIT); }

  // Returns a new PcValue with the given offset added.
  PcValue AddOffset(int offset) const { return PcValue(offset_ + offset); }

  // Returns a new PcValue representing the next PcValue (FromIndex(AsIndex() +
  // 1)).
  PcValue Next() const { return PcValue(offset_ + sizeof(Py_CODEUNIT)); }

  // Returns a new PcValue representing the previous PcValue
  // (FromIndex(AsIndex() + 1)).
  PcValue Prev() const { return PcValue(offset_ - sizeof(Py_CODEUNIT)); }

 private:
  template <typename H>
  friend H AbslHashValue(H h, const PcValue& p) {
    return H::combine(std::move(h), p.offset_);
  }

  friend bool operator==(const PcValue& self, const PcValue& other) {
    return self.offset_ == other.offset_;
  }

  explicit PcValue(int offset) : offset_(offset) {}
  int offset_;
};

////////////////////////////////////////////////////////////////////////////////
// Location

// The maximum number of addressable registers.
constexpr int64_t kMaxRegisters = 32;

// Describes where a value is stored. This can either be on the stack (in a
// defined slot location), in a general purpose register, or as an immediate (a
// pseudo-location to hold constants that can fit in immediate fields of
// instructions).
class Location {
 public:
  // Creates a location in the given frame index on the stack.
  static Location FrameSlot(int64_t index) {
    return Location(kFrameSlot, index);
  }
  // Creates a location in the given Register.
  static Location Register(asmjit::x86::Reg reg) {
    return Location(kRegister, 0, reg);
  }
  // Creates a location that is materialized as an immediate.
  static Location Immediate(int64_t imm) { return Location(kImmediate, imm); }
  // Creates an invalid location.
  static Location Undefined() { return Location(kUndefined, 0); }
  // Creates a location in the call stack.
  static Location CallStackSlot(int64_t index) {
    return Location(kCallStackSlot, index);
  }

  bool IsDefined() const { return kind_ != kUndefined; }
  bool IsOnStack() const {
    return kind_ == kFrameSlot || kind_ == kCallStackSlot;
  }
  bool IsInRegister() const { return kind_ == kRegister; }
  bool IsImmediate() const { return kind_ == kImmediate; }
  bool IsFrameSlot() const { return kind_ == kFrameSlot; }
  bool IsCallStackSlot() const { return kind_ == kCallStackSlot; }

  asmjit::x86::Reg Register() const {
    S6_CHECK(IsInRegister());
    return *reg_;
  }

  int64_t FrameSlot() const {
    S6_CHECK(IsFrameSlot());
    return index_;
  }

  int64_t CallStackSlot() const {
    S6_CHECK(IsCallStackSlot());
    return index_;
  }

  int64_t ImmediateValue() const {
    S6_CHECK(IsImmediate());
    return index_;
  }

  std::string ToString() const;

  // Returns a unique identifier for this location. All registers and stack
  // values have a unique (dense) integer value, suitable for indexing into a
  // vector.
  //
  // REQUIRES: IsInRegister() or IsOnStack().
  int64_t id() const {
    if (kind_ == kRegister) {
      S6_CHECK_LT(reg_->id(), kMaxRegisters);
      return reg_->id();
    } else {
      return index_ + kMaxRegisters;
    }
  }

  Location() : kind_(kUndefined), index_(0) {}

  bool operator==(const Location& other) const {
    return kind_ == other.kind_ && index_ == other.index_ && reg_ == other.reg_;
  }
  bool operator!=(const Location& other) const { return !(*this == other); }
  bool operator<(const Location& other) const {
    return std::make_tuple(kind_, index_, reg_.has_value() ? reg_->id() : -1) <
           std::make_tuple(other.kind_, other.index_,
                           other.reg_.has_value() ? other.reg_->id() : -1);
  }

 private:
  enum Kind { kUndefined, kFrameSlot, kCallStackSlot, kRegister, kImmediate };
  Location(Kind kind, int64_t index, absl::optional<asmjit::x86::Reg> reg = {})
      : kind_(kind), index_(index), reg_(std::move(reg)) {}

  Kind kind_;
  int64_t index_;
  absl::optional<asmjit::x86::Reg> reg_;
};

////////////////////////////////////////////////////////////////////////////////
// BytecodeInstruction

// A representation of a CPython bytecode instruction.
class BytecodeInstruction {
 public:
  // `opcode` and `argument` are obvious, but `program_offset` is the *offset*
  // into the program: `instruction_index * sizeof(_Py_CODEUNIT)`.
  BytecodeInstruction(int64_t program_offset, int64_t opcode, int64_t argument)
      : argument_(argument), program_offset_(program_offset), opcode_(opcode) {}

  // Returns the "oparg" of the instruction. While this normally is an
  // 8-bit int, EXTENDED_ARG can increase its size.
  int64_t argument() const { return argument_; }

  void set_argument(int64_t argument) { argument_ = argument; }

  // Returns the offset, in bytes, of the bytecode instruction within its
  // program.
  int64_t program_offset() const { return program_offset_; }

  // Returns a PcValue of the bytecode instruction within its program.
  PcValue pc_value() const { return PcValue::FromOffset(program_offset_); }

  // Returns the CPython bytecode opcode of this instruction.
  int64_t opcode() const { return opcode_; }

  // Returns a stringified representation of this instruction.
  std::string ToString() const;

 private:
  int64_t argument_;
  int64_t program_offset_;
  int64_t opcode_;
};

////////////////////////////////////////////////////////////////////////////////
// Python objects and strings

// Returns the result of calling str() on a given object as an std::string.
std::string PyObjectToString(PyObject* obj);

// Returns the given CPython bytecode opcode as a string.
const absl::string_view BytecodeOpcodeToString(int64_t opcode);

// If `constant` is an interned PyUnicodeObject, returns its data. Otherwise
// returns the empty string.
absl::string_view GetObjectAsCheapString(PyObject* constant);

// If `constant` is a PyUnicodeObject, returns its data. Otherwise
// returns the empty string.
//
// This does not require that the unicode object is interned, but does require
// that the GIL is held because PyUnicode_AsUTF8AndSize may mutate the
// underlying object.
absl::string_view GetObjectAsCheapStringRequiringGil(PyObject* constant);

////////////////////////////////////////////////////////////////////////////////
// Why enum

// WHY_* constants for interpreter unwinding.
// These mirror constants of `why_code` in ceval.c and have the same value.
// They decribe the reason for an unwinding.
enum class Why : int {
  kNot = 0x01,        // No error.
  kException = 0x02,  // Exception occurred.
  kReturn = 0x08,     // 'return' statement.
  kBreak = 0x10,      // 'break' statement.
  kContinue = 0x20,   // 'continue' statement.
  kYield = 0x40,      // 'yield' operator
  kSilenced = 0x80    // Exception silenced by 'with'
};

// Convert a Why value to its representation as a finally handler
// discriminator in the interpreter.
inline PyObject* PyLong_FromWhy(Why why) {
  return PyLong_FromLong(static_cast<long>(why));  // NOLINT(google-runtime-int)
}

// Convert a Why value to its representation as a finally handler
// discriminator in strongJIT
constexpr inline int64_t WhyToDiscriminator(Why why) {
  return 1 | static_cast<int64_t>(why);
}

inline absl::string_view ToString(Why why) {
  switch (why) {
    case Why::kNot:
      return "kNot";
    case Why::kException:
      return "kException";
    case Why::kReturn:
      return "kReturn";
    case Why::kBreak:
      return "kBreak";
    case Why::kContinue:
      return "kContinue";
    case Why::kYield:
      return "kYield";
    case Why::kSilenced:
      return "kSilenced";
  }
}

// This class represents multiple WhyFlags in a single value
class WhyFlags {
 public:
  constexpr WhyFlags() : v_(0) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr WhyFlags(Why why) : v_(static_cast<int>(why)) {}

  constexpr explicit operator bool() const { return v_; }

  constexpr bool Has(Why why) const { return v_ & static_cast<int>(why); }

  constexpr WhyFlags operator|(WhyFlags o) const { return WhyFlags(v_ | o.v_); }
  constexpr WhyFlags operator&(WhyFlags o) const { return WhyFlags(v_ & o.v_); }
  constexpr WhyFlags operator~() const { return WhyFlags(~v_); }
  constexpr WhyFlags& operator|=(WhyFlags o) {
    v_ |= o.v_;
    return *this;
  }
  constexpr WhyFlags& operator&=(WhyFlags o) {
    v_ &= o.v_;
    return *this;
  }

 private:
  int v_;
  constexpr explicit WhyFlags(int v) : v_(v) {}
};

inline constexpr WhyFlags operator|(Why lhs, Why rhs) {
  return WhyFlags(lhs) | WhyFlags(rhs);
}

inline constexpr WhyFlags operator&(Why lhs, Why rhs) {
  return WhyFlags(lhs) & WhyFlags(rhs);
}

inline std::string ToString(WhyFlags why) {
  return absl::StrCat(
      (why.Has(Why::kNot) ? "N" : ""), (why.Has(Why::kException) ? "E" : ""),
      (why.Has(Why::kReturn) ? "R" : ""), (why.Has(Why::kBreak) ? "B" : ""),
      (why.Has(Why::kContinue) ? "C" : ""), (why.Has(Why::kYield) ? "Y" : ""),
      (why.Has(Why::kSilenced) ? "S" : ""));
}

////////////////////////////////////////////////////////////////////////////////
// CodeFlagsValidator

// A PyCodeObject CO_FLAGS validator. Allows the caller to validate that
// certain flags, or combinations of flags, are present or not present.
class CodeFlagsValidator {
 public:
  explicit CodeFlagsValidator(uint64_t co_flags)
      : original_flags_(co_flags), unconsumed_flags_(co_flags), status_() {}

  // The given flag is required to be in co_flags. `name` describes the flag
  // for the error message. If `co_flags & bitmask != bitmask`, the sticky
  // status is set to an error.
  CodeFlagsValidator& Requires(uint64_t bitmask, absl::string_view name) {
    bool ok = (original_flags_ & bitmask) == bitmask;
    unconsumed_flags_ &= ~bitmask;
    if (status_.ok() && !ok) {
      status_ = absl::FailedPreconditionError(
          absl::StrCat("Missing required flag ", name));
    }
    return *this;
  }

  // `bitmask` contains flags that are not mandatory but are allowed.
  CodeFlagsValidator& Allows(uint64_t bitmask) {
    unconsumed_flags_ &= ~bitmask;
    return *this;
  }

  // Returns a non-OK status if:
  //   * A previous call to Requires() failed.
  //   * Any flags are still set in co_flags.
  absl::Status Validate() {
    if (!status_.ok()) return status_;
    if (unconsumed_flags_ != 0) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Unhandled co_flags were present: 0x%x", unconsumed_flags_));
    }
    return absl::OkStatus();
  }

 private:
  uint64_t original_flags_;
  uint64_t unconsumed_flags_;
  absl::Status status_;
};

////////////////////////////////////////////////////////////////////////////////
// Nullness
enum class Nullness : uint8_t { kMaybeNull, kNotNull };

////////////////////////////////////////////////////////////////////////////////
// STL Utilities

// Performs the opposite of std::move: Forces a function that takes a non-const
// lvalue reference to take in a temporary as an argument.
// It is easy to misuse so take care.
// One common use-case is to send a rvalue view (like absl::Span) to
// a algorithm that expect a lvalue container because it's going to mutate it.
template <typename T>
T& unmove(T&& t) {
  return static_cast<T&>(t);
}

// Erases all element that compare equal with the provided element from the
// container. The container must have an erase method taking two iterators, and
// must have at least forward iterators.
template <typename Container, typename Element>
void STLEraseAll(Container& c, const Element& e) {
  c.erase(std::remove(c.begin(), c.end(), e), c.end());
}

// Erases all element for which the predicate return true. The container must
// have an erase method taking two iterators, and must have at least forward
// iterators.
template <typename Container, typename Pred>
void STLEraseIf(Container& c, Pred&& pred) {
  c.erase(std::remove_if(c.begin(), c.end(), std::forward<Pred>(pred)),
          c.end());
}

// Sorts the container and remove all duplicates. Sorting is done with `<`
// which must be a total order. Duplicates are identified with `==` which must
// be an equivalence relation. The container must have an erase method taking
// two iterators, and must have at least forward iterators.
template <typename Container>
void STLSortAndRemoveDuplicates(Container& c) {
  std::sort(c.begin(), c.end());
  c.erase(std::unique(c.begin(), c.end()), c.end());
}

////////////////////////////////////////////////////////////////////////////////
// absl::Status and StatusOr management

// Return the first error in a list of status, or ok if there is no error.
inline absl::Status FirstError(absl::Span<const absl::Status> statuses) {
  for (const absl::Status& status : statuses) {
    S6_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTIL_H_
