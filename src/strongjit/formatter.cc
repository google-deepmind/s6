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

#include "strongjit/formatter.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "classes/class_manager.h"
#include "strongjit/base.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "strongjit/util.h"
#include "strongjit/value.h"
#include "utils/no_destructor.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

void AnnotationFormatter::AppendAnnotation(std::string& output,
                                           size_t start_column, const Value& v,
                                           FormatContext* ctx) const {
  static NoDestructor<std::string> padding(kAnnotationColumn, ' ');
  std::vector<std::string> annotations =
      absl::StrSplit(FormatAnnotation(v, ctx), '\n', absl::SkipEmpty());
  if (annotations.empty()) return output.push_back('\n');

  if (start_column >= kAnnotationColumn) {
    absl::StrAppend(&output, " // ", annotations.front(), "\n");
  } else {
    absl::StrAppend(&output, absl::string_view(padding->data() + start_column),
                    "// ", annotations.front(), "\n");
  }
  for (size_t i = 1; i < annotations.size(); ++i) {
    absl::StrAppend(&output, *padding, "// ", annotations[i], "\n");
  }
}

std::string PredecessorAnnotator::FormatAnnotation(const Value& v,
                                                   FormatContext* ctx) const {
  const Block* b = dyn_cast<Block>(&v);
  if (!b) return "";

  if (&ctx->function().entry() == b) {
    return "entry point";
  }
  std::string s = absl::StrJoin(
      b->predecessors(), ", ", [&](std::string* s, const Block* pred) {
        absl::StrAppend(s, "&", ctx->ValueNumber(pred));
      });
  if (s.empty()) return "no predecessors";
  return absl::StrCat("preds: ", s);
}

std::string PointerAnnotator::FormatAnnotation(const Value& v,
                                               FormatContext* ctx) const {
  return absl::StrCat("at: 0x", absl::Hex(&v));
}

std::string Format(const TryHandler& handler) {
  absl::string_view name;
  switch (handler.kind()) {
    case TryHandler::kExcept:
      name = "kExcept";
      break;
    case TryHandler::kLoop:
      name = "kLoop";
      break;
    case TryHandler::kFinally:
      name = "kFinally";
      break;
    case TryHandler::kExceptHandler:
      name = "kExceptHandler";
      break;
    case TryHandler::kFinallyHandler:
      name = "kFinallyHandler";
      break;
  }
  return absl::StrCat(
      "{", name, " @", handler.pc_value().AsOffset(),
      handler.pc_continue()
          ? absl::StrCat(" @", handler.pc_continue()->AsOffset())
          : "",
      handler.finally_fallthrough_popped_handler() ? " fallthrough_popped" : "",
      " $", handler.stack_height(), "}");
}

std::string Format(absl::Span<const TryHandler> try_handlers) {
  std::vector<std::string> tmp;
  tmp.reserve(try_handlers.size());
  for (const auto& th : try_handlers) tmp.push_back(Format(th));
  return absl::StrJoin(tmp, ", ");
}

namespace {
// Formatter defines the interface that is used by each Instruction's
// Format function to format an instruction.
class Formatter : public InstructionFormatter {
 public:
  explicit Formatter(const ValueNumbering& value_numbering, const Function& f,
                     const ClassManager& mgr)
      : value_numbering_(value_numbering),
        f_(f),
        mgr_(mgr),
        silenced_listener_(nullptr) {
    // To format, we use the mutable variants of operand accessors. But we
    // guarantee never to modify any value. So we remove any listener here that
    // might be falsely notified.
    silenced_listener_ =
        const_cast<Function&>(f).ClearInstructionModificationListener();
  }
  ~Formatter() {
    if (silenced_listener_) {
      const_cast<Function&>(f_).SetInstructionModificationListener(
          silenced_listener_);
    }
  }

  // Instruction::Format uses value_type to determine its return type.
  using value_type = std::string;

  // Concatenate all non-empty strings with a single space.
  template <typename... T>
  std::string Concat(T... t) {
    return absl::StrJoin(RemoveEmpty({t...}), " ");
  }

  // Concatenate all non-empty strings with a comma.
  template <typename... T>
  std::string CommaConcat(T... t) {
    return absl::StrJoin(RemoveEmpty({t...}), ", ");
  }

  // Formats an integer with a '$' sigil.
  std::string Imm(int64_t* imm) { return absl::StrCat("$", *imm); }

  // Formats an integer with a '@' sigil.
  std::string BytecodeOffset(int32_t* imm) { return absl::StrCat("@", *imm); }

  // Formats a single value.
  std::string Value(class Value** v) {
    return absl::StrCat("%", ValueNumber(*v));
  }

  std::string Value(Block** v) { return absl::StrCat("&", ValueNumber(*v)); }

  std::string Value(std::function<Block**()> fn) { return Value(fn()); }

  std::string OptionalValue(class Value** v, absl::string_view prefix = "") {
    if (!*v) return "";
    return absl::StrCat(prefix, " ", Value(v));
  }

  std::string OptionalValue(Block** v) {
    if (!*v) return "";
    return Value(v);
  }

  // Formats a comma-separated list of values with custom open and close tokens.
  std::string ValueList(MutableOperandList list,
                        absl::string_view open_bracket = "",
                        absl::string_view close_bracket = "") {
    std::string s =
        absl::StrJoin(list.span(), ", ",
                      [&](std::string* s, class Value* v) { *s += Value(&v); });
    return absl::StrCat(open_bracket, s, close_bracket);
  }

  // As ValueList, but if the list is empty returns nothing (not even the open
  // and close brackets).
  std::string OptionalValueList(MutableOperandList list,
                                absl::string_view open_bracket = "",
                                absl::string_view close_bracket = "") {
    return list.empty() ? "" : ValueList(list, open_bracket, close_bracket);
  }

  // Given an enumeration and that enumeration's string values, format the
  // enumeration as a string.
  template <typename T>
  std::string Enum(T* t, absl::Span<const absl::string_view> values) {
    int64_t i = static_cast<int64_t>(*t);
    if (i < 0 || i >= values.size()) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrCat("Enumeration list out of range: value `", i,
                       "' for type ", typeid(T).name())));
      return "BAD-ENUM";
    }
    return std::string(*std::next(values.begin(), i));
  }

  template <typename T>
  std::string EnumBitfield(T* t, absl::Span<const absl::string_view> values) {
    std::string s;
    uint64_t value = static_cast<uint64_t>(*t);
    for (int64_t bit = 0; value != 0; ++bit, value >>= 1) {
      if ((value & 1) == 0) continue;
      if (bit >= values.size()) {
        status_.Update(absl::InvalidArgumentError(
            absl::StrCat("Enumeration list out of range: value `", bit,
                         "' for type ", typeid(T).name())));
        return "BAD-ENUM";
      }
      if (!s.empty()) absl::StrAppend(&s, "|");
      absl::StrAppend(&s, *std::next(values.begin(), bit));
    }
    return s;
  }

  // A simple string token.
  std::string Str(absl::string_view s) { return std::string(s); }

  std::string InternedGlobalString(
      const GlobalInternTable::InternedString* id) {
    return absl::StrCat("\"", id->get(), "\"");
  }

  std::string InternedString(uint16_t* id) {
    return absl::StrCat("\"", f_.GetStringTable()->GetInternedString(*id),
                        "\"");
  }

  std::string Flag(bool* b, absl::string_view flag) {
    return *b ? std::string(flag) : "";
  }

  std::string TryHandlerList(std::vector<TryHandler>* try_handlers) {
    return Format(*try_handlers);
  }

  std::string ImmList(std::vector<int64_t>* list) {
    return absl::StrCat("[$", absl::StrJoin(*list, ", $"), "]");
  }

  std::string Class(int64_t* class_id) {
    const class Class* cls = mgr_.GetClassById(*class_id);
    return absl::StrCat((cls ? cls->name() : "bad-class"), "#", *class_id);
  }

  std::string Map(const int64_t** map) { return "null-map@0"; }

  absl::Status status() const { return status_; }
  absl::Status& mutable_status() { return status_; }

  int64_t ValueNumber(const class Value* v) {
    return value_numbering_.contains(v) ? value_numbering_.at(v) : -1;
  }

 private:
  // Removes any empty strings from the given list.
  std::vector<std::string> RemoveEmpty(
      std::initializer_list<std::string> list) {
    std::vector<std::string> v;
    for (const std::string& s : list) {
      if (!s.empty()) v.push_back(s);
    }
    return v;
  }

  const ValueNumbering& value_numbering_;
  const Function& f_;
  const ClassManager& mgr_;
  absl::Status status_;
  InstructionModificationListener* silenced_listener_;
};

// Functor struct to give to ForAllInstructionKinds.
struct InstrFormatter {
  template <typename InstrType>
  static absl::optional<std::string> Visit(Instruction* inst,
                                           Formatter* formatter) {
    if (inst->kind() != InstrType::kKind) return {};
    std::string prefix;
    if (InstrType::kProducesValue) {
      prefix = absl::StrCat("%", formatter->ValueNumber(inst), " = ");
    }
    return absl::StrCat(prefix, InstrType::kMnemonic, " ",
                        InstrType::Format(cast<InstrType>(inst), formatter));
  }

  static std::string Default(Instruction* inst, Formatter* formatter) {
    formatter->mutable_status().Update(absl::InvalidArgumentError(
        "unknown instruction kind in instruction formatter!"));
    return "BAD-INST";
  }
};

}  // namespace

absl::StatusOr<std::string> Format(const Instruction& inst, const Function* f,
                                   const ClassManager& mgr) {
  if (!f) {
    S6_CHECK(inst.parent());
    f = inst.parent()->parent();
  }

  ValueNumbering value_numbering = ComputeValueNumbering(*f);
  Formatter formatter(value_numbering, *f, mgr);
  // The const_cast is because our visitor struct, Formatter, has to see mutable
  // instructions. It does not modify the instruction.
  std::string result = ForAllInstructionKinds<InstrFormatter>(
      const_cast<Instruction*>(&inst), &formatter);
  S6_RETURN_IF_ERROR(formatter.status());
  return std::string(absl::StripTrailingAsciiWhitespace(result));
}

absl::StatusOr<std::string> Format(const Instruction& inst,
                                   const ValueNumbering& value_numbering,
                                   const ClassManager& mgr) {
  S6_CHECK(inst.parent());
  const Function* f = inst.parent()->parent();
  Formatter formatter(value_numbering, *f, mgr);
  // The const_cast is because our visitor struct, Formatter, has to see mutable
  // instructions. It does not modify the instruction.
  std::string result = ForAllInstructionKinds<InstrFormatter>(
      const_cast<Instruction*>(&inst), &formatter);
  S6_RETURN_IF_ERROR(formatter.status());
  return std::string(absl::StripTrailingAsciiWhitespace(result));
}

std::string FormatOrDie(const Instruction& inst, const Function* f,
                        const ClassManager& mgr) {
  return *Format(inst, f, mgr);
}

absl::StatusOr<std::string> Format(const Block& b, const Function* f,
                                   const ClassManager& mgr) {
  if (!f) {
    S6_CHECK(b.parent());
    f = b.parent();
  }

  ValueNumbering value_numbering = ComputeValueNumbering(*f);
  return absl::StrCat("&", value_numbering.at(&b));
}

std::string FormatOrDie(const Block& b, const Function* f,
                        const ClassManager& mgr) {
  return *Format(b, f, mgr);
}

std::string FormatOrDie(const Value& v, const Function* f,
                        const ClassManager& mgr) {
  if (!f) {
    if (isa<Block>(v)) {
      f = cast<Block>(v).parent();
    } else if (isa<BlockArgument>(v)) {
      f = cast<BlockArgument>(v).parent()->parent();
    } else if (isa<Instruction>(v)) {
      f = cast<Instruction>(v).parent()->parent();
    } else {
      return absl::StrCat("<bad value: kind=", v.kind(), ">");
    }
  }

  ValueNumbering value_numbering = ComputeValueNumbering(*f);
  return absl::StrCat(isa<Block>(v) ? "&" : "%", value_numbering.at(&v));
}

absl::StatusOr<std::string> Format(const Function& f,
                                   const AnnotationFormatter& annotator,
                                   const ClassManager& mgr) {
  ValueNumbering value_numbering = ComputeValueNumbering(f);

  FormatContext context(value_numbering, f);
  Formatter formatter(value_numbering, f, mgr);
  std::string output;

  std::map<int64_t, std::string> sorted_summaries;
  for (const auto& [pc_value, summary] : f.type_feedback()) {
    std::string sub_offset;
    if (pc_value.second != 0) {
      sub_offset = absl::StrCat(".", pc_value.second);
    }
    sorted_summaries[pc_value.first.AsOffset() * 10 + pc_value.second] =
        absl::StrCat("type_feedback @", pc_value.first.AsOffset(), sub_offset,
                     " ", summary.ToString(&mgr), "\n");
  }
  if (!sorted_summaries.empty()) {
    absl::StrAppend(&output,
                    absl::StrJoin(sorted_summaries, "",
                                  [](std::string* s, const auto& p) {
                                    absl::StrAppend(s, p.second);
                                  }),
                    "\n");
  }

  absl::StrAppend(&output, "function ", f.name(), " {\n");

  for (const Block& block : f) {
    if (&block != &f.entry())
      // Add a newline before each new block that isn't the first.
      output += "\n";

    std::string block_text = absl::StrCat("&", value_numbering.at(&block), ":");
    if (block.IsExceptHandler()) {
      absl::StrAppend(&block_text, " except");
    } else if (block.IsFinallyHandler()) {
      absl::StrAppend(&block_text, " finally");
    }
    if (block.deoptimized()) {
      absl::StrAppend(&block_text, " deoptimized");
    }
    if (!block.block_arguments_empty()) {
      absl::StrAppend(
          &block_text, " [ ",
          absl::StrJoin(block.block_arguments(), ", ",
                        [&](std::string* s, const BlockArgument* a) {
                          absl::StrAppend(s, "%", value_numbering.at(a));
                        }),
          " ]");
    }

    absl::StrAppend(&output, block_text);
    annotator.AppendAnnotation(output, block_text.size(), block, &context);

    for (const Instruction& inst : block) {
      // The const_cast is because our visitor struct, Formatter, has to see
      // mutable instructions. It does not modify the instruction.
      std::string result = ForAllInstructionKinds<InstrFormatter>(
          const_cast<Instruction*>(&inst), &formatter);
      result = std::string(absl::StripTrailingAsciiWhitespace(result));
      S6_RETURN_IF_ERROR(formatter.status());
      absl::StrAppend(&output, "  ", result);
      annotator.AppendAnnotation(output, result.size() + 2, inst, &context);
    }
  }

  absl::StrAppend(&output, "}");
  return output;
}

std::string FormatOrDie(const Function& f, const AnnotationFormatter& annotator,
                        const ClassManager& mgr) {
  return *Format(f, annotator, mgr);
}

}  // namespace deepmind::s6
