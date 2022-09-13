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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FORMATTER_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FORMATTER_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "classes/class_manager.h"
#include "strongjit/base.h"
#include "strongjit/util.h"
#include "strongjit/value_map.h"
#include "tuple_util.h"

namespace deepmind::s6 {

// Context for an AnnotationFormatter.
class FormatContext {
 public:
  explicit FormatContext(const ValueNumbering& numbering,
                         const Function& function)
      : value_numbering_(numbering), function_(function) {}

  // Returns the value number for the given Value, or -1 if it is not numbered.
  int64_t ValueNumber(const Value* v) {
    return value_numbering_.contains(v) ? value_numbering_.at(v) : -1;
  }

  const ValueNumbering& value_numbering() { return value_numbering_; }

  // Returns the function we are formatting.
  const Function& function() { return function_; }

 private:
  const ValueNumbering& value_numbering_;
  const Function& function_;
};

constexpr int64_t kAnnotationColumn = 60;

// Callback class for Formatter. Given a Value, return an optional annotation
// which will be formatted as a comment.
// The string may include new lines, but must not end with a new line.
// If the annotation has multiple lines, blank lines will be inserted before:
//   instruction             // annotation1
//                           // annotation2
//   next instruction
//
// Multiple consecutive new lines in the output of FormatAnnotation will be
// collapsed into a single new line.
class AnnotationFormatter {
 public:
  virtual ~AnnotationFormatter() {}
  virtual std::string FormatAnnotation(const Value& v,
                                       FormatContext* ctx) const = 0;
  // Prints an annotation for that value to output.
  // Assumes the current line starts at `start_columns`.
  void AppendAnnotation(std::string& output, size_t start_column,
                        const Value& v, FormatContext* ctx) const;
};

// Annotates blocks with their list of predecessors.
class PredecessorAnnotator final : public AnnotationFormatter {
 public:
  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final;
};

// Annotates everything with their pointer.
class PointerAnnotator final : public AnnotationFormatter {
 public:
  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final;
};

// Merge multiple annotators into one. The annotation of the underlying
// annotators will be separated by new lines.
template <typename... Annotators>
class ChainAnnotators final : public AnnotationFormatter {
 public:
  std::string FormatAnnotation(const Value& v, FormatContext* ctx) const final {
    return absl::StrJoin(tuple::to_array(tuple::transform(
                             [&](const auto& annotator) {
                               return annotator.FormatAnnotation(v, ctx);
                             },
                             annotators_)),
                         "\n");
  }

  explicit ChainAnnotators(Annotators... annotators)
      : annotators_(std::move(annotators)...) {}

 private:
  std::tuple<Annotators...> annotators_;
};

template <typename... Annotators>
ChainAnnotators(Annotators...) -> ChainAnnotators<Annotators...>;

// Formats a TryHandler as a string.
// This has the same format as the output of the bytecode strongJIT instruction.
std::string Format(const TryHandler& handler);

// Formats a TryHandler sequence as a comma separated string.
// This has the same format as the output of the bytecode strongJIT instruction.
std::string Format(absl::Span<const TryHandler> try_handlers);

// Formats an instruction and returns it as a string. The instruction must
// be linked to a Block unless `f` is non-nullptr.
std::string FormatOrDie(const Instruction& inst, const Function* f = nullptr,
                        const ClassManager& mgr = ClassManager::Instance());

// Formats an instruction and returns it as a string. The instruction must
// be linked to a Block unless `f` is non-nullptr.
absl::StatusOr<std::string> Format(
    const Instruction& inst, const Function* f = nullptr,
    const ClassManager& mgr = ClassManager::Instance());

// Formats an instruction and returns it as a string. The instruction must be
// linked to a Block.
absl::StatusOr<std::string> Format(
    const Instruction& inst, const ValueNumbering& value_numbering,
    const ClassManager& mgr = ClassManager::Instance());

// Formats a block and returns it as a string. The block must be linked to a
// function unless `f` is non-nullptr.
absl::StatusOr<std::string> Format(
    const Block& b, const Function* f = nullptr,
    const ClassManager& mgr = ClassManager::Instance());

// Formats a block and returns it as a string. The block must be linked to a
// function unless `f` is non-nullptr.
std::string FormatOrDie(const Block& b, const Function* f = nullptr,
                        const ClassManager& mgr = ClassManager::Instance());

// Formats a value - block, BlockArgument or Instruction.
std::string FormatOrDie(const Value& v, const Function* f = nullptr,
                        const ClassManager& mgr = ClassManager::Instance());

// Prints a function and returns it as a string.
absl::StatusOr<std::string> Format(
    const Function& f,
    const AnnotationFormatter& annotator = PredecessorAnnotator(),
    const ClassManager& mgr = ClassManager::Instance());

// Formats a function and returns it as a string.
std::string FormatOrDie(
    const Function& f,
    const AnnotationFormatter& annotator = PredecessorAnnotator(),
    const ClassManager& mgr = ClassManager::Instance());

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_FORMATTER_H_
