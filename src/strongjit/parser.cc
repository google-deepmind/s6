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

#include "strongjit/parser.h"

#include <cstdint>
#include <functional>
#include <initializer_list>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "classes/class_manager.h"
#include "global_intern_table.h"
#include "strongjit/base.h"
#include "strongjit/function.h"
#include "strongjit/instruction_traits.h"
#include "strongjit/instructions.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {

using ParserValueNumbering = absl::flat_hash_map<int64_t, Value*>;

// Strips leading whitespace and returns the string. Does not remove newlines,
// only spaces.
absl::string_view StripLeadingWhitespace(absl::string_view str) {
  int64_t i = 0;
  while (i < str.size() && str[i] == ' ') ++i;
  return str.substr(i);
}

// Consumes an int64_t from `str`, updating it in place.
absl::StatusOr<int64_t> ConsumeInt64(absl::string_view* str) {
  int64_t index = 0;
  if (str->at(0) == '-') ++index;
  while (index < str->size() && std::isdigit(str->at(index))) {
    ++index;
  }
  absl::string_view prefix = str->substr(0, index);
  str->remove_prefix(index);
  int64_t out;
  if (!absl::SimpleAtoi(prefix, &out)) {
    return absl::InvalidArgumentError("expected integer");
  }
  return out;
}

// A simple lexer. This maintains a sticky status that can be queried at any
// time. None of the operations directly return a Status.
// The stream never starts with whitespace, they are automatically stripped.
class Stream {
 public:
  explicit Stream(absl::string_view str) : stream_(str) {}

  // Consumes `s` from the stream and errors if that was not possible.
  void Take(absl::string_view s) {
    if (!status_.ok()) return;
    if (!absl::ConsumePrefix(&stream_, s)) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrCat("expected `", s, "' at:", stream_)));
    }
    stream_ = StripLeadingWhitespace(stream_);
  }

  // Consumes `s` from the stream if possible, otherwise returns false.
  bool TakeIfPossible(absl::string_view s) {
    if (CanTake(s)) {
      Take(s);
      return true;
    }
    return false;
  }

  // Returns true if the stream begins with `s`.
  bool CanTake(absl::string_view s) {
    if (!status_.ok()) return false;
    return absl::StartsWith(stream_, s);
  }

  // Lexes a token - consumes from the stream until a std::isspace() character
  // is found (or the stream ends).
  absl::string_view TakeUntilWhitespace() {
    if (!status_.ok()) return "";
    int64_t i = 0;
    while (i < stream_.size() && !std::isspace(stream_[i])) ++i;
    absl::string_view token = stream_.substr(0, i);
    stream_ = StripLeadingWhitespace(stream_.substr(i));
    return token;
  }

  // Lexes a token - consumes from the stream until the given character is
  // found (or the stream ends).
  absl::string_view TakeUntil(char c) {
    if (!status_.ok()) return "";
    int64_t i = 0;
    while (i < stream_.size() && stream_[i] != c) ++i;
    absl::string_view token = stream_.substr(0, i);
    stream_ = StripLeadingWhitespace(stream_.substr(i));
    return token;
  }

  // Expects an end of line. This can either be a `\n`, or a comment followed by
  // a newline.
  void TakeEndOfLine() {
    if (!status_.ok()) return;

    // A comment is allowed at this point.
    if (CanTake("//")) {
      size_t index = stream_.find_first_of('\n');
      if (index == absl::string_view::npos) {
        stream_ = "";
      } else {
        stream_ = stream_.substr(index + 1);
      }
    } else {
      Take("\n");
    }
    stream_ = StripLeadingWhitespace(stream_);
  }

  // Skips empty lines.
  void SkipEmptyLines() {
    while (CanTake("//") || CanTake("\n")) TakeEndOfLine();
  }

  // Takes an int64_t from the stream.
  int64_t Int64() {
    if (!status_.ok()) return -1;
    absl::StatusOr<int64_t> val = ConsumeInt64(&stream_);
    stream_ = StripLeadingWhitespace(stream_);

    if (val.ok()) return val.value();
    status_.Update(val.status());
    return 0;
  }

  // Returns true if the lexer is at the end of a statement, which is either the
  // empty stream, a newline or a comment.
  bool IsAtEndOfStatement() {
    if (!status_.ok()) return true;
    return stream_.empty() || stream_[0] == '\n' || CanTake("//");
  }

  // Returns true if the lexer is at the end of the stream.
  bool IsAtEndOfStream() {
    if (!status_.ok()) return true;
    return stream_.empty();
  }

  // Records an error in the stream. The stream's status is updated and context
  // appended.
  absl::Status Error(absl::Status status) {
    status_.Update(absl::Status(
        status.code(), absl::StrCat(status.message(), " at: ", stream_)));
    return status_;
  }

  absl::Status status() const { return status_; }
  absl::string_view rest() const { return stream_; }

 private:
  absl::Status status_;
  absl::string_view stream_;
};

absl::StatusOr<int32_t> ParseClass(Stream* stream, const ClassManager& mgr) {
  absl::string_view class_name = stream->TakeUntil('#');
  stream->Take("#");
  int64_t id = stream->Int64();
  S6_RETURN_IF_ERROR(stream->status());

  const Class* cls = mgr.GetClassById(id);
  if (!cls)
    return absl::InvalidArgumentError(
        absl::StrCat("No class found with ID ", id));
  if (cls->name() != class_name) {
    return absl::InvalidArgumentError(
        absl::StrCat("Class with ID ", id, " has non-matching name. Expected '",
                     class_name, "' but got '", cls->name(), "'"));
  }
  return static_cast<int32_t>(id);
}

// ParserFormatter defines the interface that is used by each Instruction's
// Format function to parse an instruction.
class ParserFormatter : public InstructionFormatter {
 public:
  // The value type of this formatter is a function; we build up a tree of
  // functions and call the tree after Format() is complete. This allows us to
  // traverse the stream in order, not in argument evaluation order.
  using value_type = std::function<void()>;

  // Creates a ParserFormatter reading from `stream`.
  ParserFormatter(Stream* stream, ParserValueNumbering* numbering, Function* f,
                  const ClassManager& mgr)
      : stream_(*stream), numbering_(*numbering), f_(*f), mgr_(mgr) {}

  // Concatenate all non-empty strings with a single space.
  template <typename... T>
  value_type Concat(T... t) {
    return [=]() {
      for (auto& fn : {t...}) {
        fn();
      }
    };
  }

  // Concatenate all non-empty strings with a comma.
  template <typename... T>
  value_type CommaConcat(T... t) {
    return [=]() {
      for (auto& fn : {t...}) {
        fn();
        // Technically this should be Take, but getting this correct for the
        // last item is hard.
        stream_.TakeIfPossible(",");
      }
    };
  }

  // Parses an integer with a '$' sigil.
  value_type Imm(int64_t* imm) {
    return [=]() {
      stream_.Take("$");
      *imm = stream_.Int64();
    };
  }

  // Parses an integer with a '@' sigil.
  value_type BytecodeOffset(int32_t* imm) {
    return [=]() {
      stream_.Take("@");
      *imm = stream_.Int64();
    };
  }

  // Parses a single value.
  value_type Value(class Value** v) {
    return [=]() {
      stream_.Take("%");
      *v = ResolveValue(stream_.Int64());
    };
  }

  value_type OptionalValue(class Value** v, absl::string_view prefix = "") {
    return [=]() {
      if (!prefix.empty()) {
        if (!stream_.TakeIfPossible(prefix)) {
          *v = nullptr;
          return;
        }
      }
      if (!stream_.TakeIfPossible("%")) {
        *v = nullptr;
      } else {
        *v = ResolveValue(stream_.Int64());
      }
    };
  }

  // Parses a single block.
  value_type Value(Block** v) {
    return [=]() {
      stream_.Take("&");
      *v = ResolveBlock(stream_.Int64());
    };
  }

  value_type OptionalValue(class Block** v) {
    return [=]() {
      if (!stream_.TakeIfPossible("&")) {
        *v = nullptr;
      } else {
        *v = ResolveBlock(stream_.Int64());
      }
    };
  }

  // Parses a single block with the block location computed lazily.
  value_type Value(std::function<Block**()> fn) {
    return [=]() {
      stream_.Take("&");
      *fn() = ResolveBlock(stream_.Int64());
    };
  }

  // Parses a comma-separated list of values with custom open and close tokens.
  value_type ValueList(MutableOperandList list,
                       absl::string_view open_bracket = "",
                       absl::string_view close_bracket = "") {
    return [=]() mutable {
      if (!open_bracket.empty())
        stream_.Take(absl::StripAsciiWhitespace(open_bracket));

      // Deal with empty list, if brackets are given. If no brackets we can't
      // handle an empty list (should have used OptionalValueList).
      if (!close_bracket.empty() &&
          stream_.TakeIfPossible(absl::StripAsciiWhitespace(close_bracket))) {
        return;
      }

      do {
        stream_.Take("%");
        list.push_back(ResolveValue(stream_.Int64()));
      } while (stream_.TakeIfPossible(","));

      if (!close_bracket.empty())
        stream_.Take(absl::StripAsciiWhitespace(close_bracket));
    };
  }

  // As ValueList, but accepts an empty list.
  // TODO: We need to know the followset to parse this properly.
  // Currently we only accept EOL as the followset if no open bracket is given.
  value_type OptionalValueList(MutableOperandList list,
                               absl::string_view open_bracket = "",
                               absl::string_view close_bracket = "") {
    return [=]() {
      bool exists = !stream_.IsAtEndOfStatement();
      if (!open_bracket.empty()) {
        exists = stream_.CanTake(absl::StripAsciiWhitespace(open_bracket));
      }
      if (exists) {
        ValueList(list, open_bracket, close_bracket)();
      }
    };
  }

  template <typename T>
  int64_t TakeEnum(absl::Span<const absl::string_view> values) {
    int64_t i = 0;
    absl::optional<int64_t> longest_match_index;
    int64_t longest_match_size = -1;
    absl::string_view longest_match = "";
    for (absl::string_view value : values) {
      if (static_cast<int64_t>(value.size()) > longest_match_size &&
          stream_.CanTake(value)) {
        longest_match_index = i;
        longest_match = value;
        longest_match_size = value.size();
      }
      ++i;
    }
    if (longest_match_index) {
      stream_.Take(longest_match);
      return *longest_match_index;
    } else {
      (void)stream_.Error(absl::InvalidArgumentError(
          absl::StrCat("expected an `", typeid(T).name(), "' with values ",
                       absl::StrJoin(values, ","))));
      return 0;
    }
  }

  // Given an enumeration and that enumeration's string values, parse the
  // enumeration as a string.
  template <typename T>
  value_type Enum(T* t, absl::Span<const absl::string_view> values) {
    return [this, t,
            values = std::vector<absl::string_view>(values.begin(),
                                                    values.end())]() {
      int64_t i = TakeEnum<T>(values);
      *t = static_cast<T>(i);
    };
  }

  template <typename T>
  value_type EnumBitfield(T* t, absl::Span<const absl::string_view> values) {
    return [this, t,
            values = std::vector<absl::string_view>(values.begin(),
                                                    values.end())]() {
      int64_t v = 0;
      do {
        int64_t i = TakeEnum<T>(values);
        v |= 1 << i;
      } while (stream_.TakeIfPossible("|"));
      *t = static_cast<T>(v);
    };
  }

  value_type Str(absl::string_view s) {
    return [=]() { stream_.Take(absl::StripAsciiWhitespace(s)); };
  }

  value_type InternedGlobalString(GlobalInternTable::InternedString* id) {
    return [=]() {
      stream_.Take("\"");
      absl::string_view str = stream_.TakeUntil('"');

      if (str.find('\\') != -1) {
        (void)stream_.Error(
            absl::InvalidArgumentError("invalid string literal"));
        return;
      }

      stream_.Take("\"");
      *id = GlobalInternTable::Instance().Intern(str);
    };
  }

  value_type InternedString(uint16_t* id) {
    return [=]() {
      stream_.Take("\"");
      absl::string_view str = stream_.TakeUntil('"');

      if (str.find('\\') != -1) {
        (void)stream_.Error(
            absl::InvalidArgumentError("invalid string literal"));
        return;
      }

      stream_.Take("\"");
      *id = f_.GetStringTable()->InternString(str);
    };
  }

  value_type Flag(bool* b, absl::string_view flag) {
    return [=]() { *b = stream_.TakeIfPossible(flag); };
  }

  value_type TryHandlerList(std::vector<TryHandler>* try_handlers) {
    return [=]() {
      while (stream_.TakeIfPossible("{")) {
        TryHandler::Kind kind;
        Enum(&kind, {"kExcept", "kLoop", "kFinally", "kExceptHandler",
                     "kFinallyHandler"})();
        int32_t bytecode_offset;
        BytecodeOffset (&bytecode_offset)();
        int32_t bytecode_continue_offset = -1;
        if (stream_.CanTake("@")) BytecodeOffset (&bytecode_continue_offset)();
        bool fallthrough_popped = stream_.TakeIfPossible("fallthrough_popped");
        int64_t stack_height;
        Imm (&stack_height)();
        try_handlers->emplace_back(kind, PcValue::FromOffset(bytecode_offset),
                                   stack_height);
        if (bytecode_continue_offset != -1) {
          try_handlers->back().set_pc_continue(
              PcValue::FromOffset(bytecode_continue_offset));
        }
        if (fallthrough_popped) {
          try_handlers->back().set_finally_fallthrough_popped_handler();
        }
        stream_.Take("}");
        stream_.TakeIfPossible(",");
      }
    };
  }

  value_type ImmList(std::vector<int64_t>* list) {
    return [=]() {
      stream_.Take("[");

      // Deal with empty list.
      if (stream_.TakeIfPossible("]")) {
        return;
      }

      do {
        stream_.Take("$");
        list->push_back(stream_.Int64());
      } while (stream_.TakeIfPossible(","));

      stream_.Take("]");
    };
  }

  value_type Class(int64_t* class_id) {
    return [=]() {
      auto value_or = ParseClass(&stream_, mgr_);
      if (!value_or.ok()) {
        (void)stream_.Error(value_or.status());
        return;
      }
      *class_id = value_or.value();
    };
  }

  value_type Map(const int64_t** map) {
    return [=]() {
      stream_.TakeUntil('@');
      stream_.Take("@");
      (void)stream_.Int64();
      // We ignore the map address, and always set `*map` to nullptr. It is very
      // unlikely that the serialized map address is actually valid when parsed,
      // so we try not to create a pointer that we can't sanity check.
      *map = nullptr;
    };
  }

 private:
  class Value* ResolveValue(int64_t value_number) {
    if (numbering_.contains(value_number)) return numbering_.at(value_number);
    (void)stream_.Error(absl::InvalidArgumentError(
        absl::StrCat("value `%", value_number, "' not yet defined")));
    return nullptr;
  }

  class Block* ResolveBlock(int64_t value_number) {
    if (numbering_.contains(value_number)) {
      class Value* v = numbering_.at(value_number);
      if (!isa<Block>(v)) {
        (void)stream_.Error(absl::InvalidArgumentError(
            absl::StrCat("value `%", value_number,
                         "' was defined as an instruction, not a block")));
        return nullptr;
      }
      return cast<Block>(v);
    }

    // Blocks can be forward referenced, so create a new one.
    Block* b = f_.CreateBlock();
    numbering_[value_number] = b;
    return b;
  }

  ParserFormatter(const ParserFormatter&) = delete;
  Stream& stream_;
  ParserValueNumbering& numbering_;
  Function& f_;
  const ClassManager& mgr_;
};

// Functor struct to give to ForAllInstructionKinds.
class InstrParser {
 public:
  template <typename InstrType>
  static absl::optional<absl::StatusOr<Instruction*>> Visit(
      absl::string_view mnemonic, Stream* stream, ParserFormatter* parser,
      Function* f) {
    if (mnemonic != InstrType::kMnemonic)
      return absl::optional<absl::StatusOr<Instruction*>>();

    InstrType* inst = f->Create<InstrType>();
    InstrType::Format(inst, parser)();
    S6_RETURN_IF_ERROR(stream->status());
    return inst;
  }

  static absl::StatusOr<Instruction*> Default(absl::string_view mnemonic,
                                              Stream* stream,
                                              ParserFormatter* parser,
                                              Function* f) {
    return absl::InternalError(
        absl::StrCat("unknown instruction mnemonic `", mnemonic, "'"));
  }
};

// Note that next_value_number != value_numbering.size() when a block is
// forward-referenced.
absl::StatusOr<Instruction*> ParseInstruction(
    Stream* stream, ParserValueNumbering* value_numbering,
    int64_t next_value_number, Function* f, const ClassManager& mgr) {
  if (stream->TakeIfPossible("%")) {
    int64_t value_number = stream->Int64();
    S6_RETURN_IF_ERROR(stream->status());

    if (value_number != next_value_number) {
      return stream->Error(absl::InvalidArgumentError(absl::StrCat(
          "expected instruction to be numbered `", next_value_number, "'")));
      return stream->status();
    }
    stream->Take("=");
  }

  absl::string_view mnemonic = stream->TakeUntilWhitespace();
  ParserFormatter parser(stream, value_numbering, f, mgr);
  S6_ASSIGN_OR_RETURN(Instruction * inst, ForAllInstructionKinds<InstrParser>(
                                              mnemonic, stream, &parser, f));
  if (!stream->IsAtEndOfStatement()) stream->TakeEndOfLine();
  S6_RETURN_IF_ERROR(stream->status());
  return inst;
}

}  // namespace

absl::StatusOr<Instruction*> ParseInstruction(absl::string_view str,
                                              Function* f,
                                              const ClassManager& mgr) {
  ParserValueNumbering value_numbering;
  for (Block& block : *f) {
    value_numbering[value_numbering.size()] = &block;
    for (Instruction& inst : block) {
      value_numbering[value_numbering.size()] = &inst;
    }
  }

  Stream stream(str);
  S6_ASSIGN_OR_RETURN(Instruction * inst,
                      ParseInstruction(&stream, &value_numbering,
                                       value_numbering.size(), f, mgr));

  f->rbegin()->push_back(inst);
  return inst;
}

absl::StatusOr<ClassDistributionSummary> ParseClassDistributionSummary(
    Stream* stream, const ClassManager& mgr) {
  bool stable = true;
  if (stream->TakeIfPossible("UNSTABLE")) {
    stable = false;
  }

  std::array<int32_t, ClassDistribution::kNumClasses> class_ids;
  absl::c_fill(class_ids, 0);
  if (stream->TakeIfPossible("monomorphic,")) {
    S6_ASSIGN_OR_RETURN(class_ids[0], ParseClass(stream, mgr));
    return ClassDistributionSummary(ClassDistributionSummary::kMonomorphic,
                                    class_ids, stable);
  } else if (stream->TakeIfPossible("polymorphic,")) {
    stream->Take("either");
    int64_t i = 0;
    S6_ASSIGN_OR_RETURN(class_ids[i++], ParseClass(stream, mgr));
    while (stream->TakeIfPossible("or")) {
      S6_ASSIGN_OR_RETURN(class_ids[i++], ParseClass(stream, mgr));
    }
    return ClassDistributionSummary(ClassDistributionSummary::kPolymorphic,
                                    class_ids, stable);
  } else if (stream->TakeIfPossible("skewed")) {
    stream->Take("megamorphic,");
    stream->Take("commonly");
    int64_t i = 0;
    S6_ASSIGN_OR_RETURN(class_ids[i], ParseClass(stream, mgr));
    return ClassDistributionSummary(
        ClassDistributionSummary::kSkewedMegamorphic, class_ids, stable);
  } else {
    stream->Take("megamorphic");
    return ClassDistributionSummary(ClassDistributionSummary::kMegamorphic,
                                    class_ids, stable);
  }
}

absl::StatusOr<Function> ParseFunction(absl::string_view str,
                                       const ClassManager& mgr) {
  ParserValueNumbering value_numbering;
  int64_t next_value_number = 0;
  Stream stream(str);
  stream.SkipEmptyLines();

  Function f("");  // We will replace the name later.

  // Parse "type_feedback" "@" BYTECODE_OFFSET ClassDistributionSummary EOL
  while (stream.TakeIfPossible("type_feedback")) {
    stream.Take("@");
    int64_t bytecode_offset = stream.Int64();
    int64_t operand_index = 0;
    if (stream.TakeIfPossible(".")) {
      operand_index = stream.Int64();
    }
    S6_ASSIGN_OR_RETURN(ClassDistributionSummary summary,
                        ParseClassDistributionSummary(&stream, mgr));
    stream.TakeEndOfLine();
    stream.SkipEmptyLines();
    S6_RETURN_IF_ERROR(stream.status());

    f.type_feedback()[{PcValue::FromOffset(bytecode_offset), operand_index}] =
        summary;
  }

  // Parse "function" NAME "{" EOL
  stream.Take("function");
  absl::string_view name = stream.TakeUntilWhitespace();
  f.set_name(name);
  stream.Take("{");
  stream.TakeEndOfLine();
  stream.SkipEmptyLines();
  S6_RETURN_IF_ERROR(stream.status());

  Block* block = nullptr;
  while (!stream.IsAtEndOfStream() && !stream.CanTake("}")) {
    // If possible, parse "&" NUMBER ":" DEOPTIMIZED? EOL
    if (stream.TakeIfPossible("&")) {
      int64_t number = stream.Int64();
      if (number != next_value_number) {
        return stream.Error(absl::InvalidArgumentError(absl::StrCat(
            "expected block to be numbered `", next_value_number, "'")));
      }

      // This block number may have already been encountered.
      if (value_numbering.contains(number)) {
        // The only way we can forward declare is by label, so it is safe to
        // cast this to Block.
        block = cast<Block>(value_numbering.at(number));
      } else {
        block = f.CreateBlock();
        value_numbering[number] = block;
      }
      // Ensure that we sort blocks inside `f` in the order the parser
      // encountered their definitions (now), not their forward declarations.
      f.UnlinkBlock(block);
      f.insert(block, f.end());

      ++next_value_number;
      stream.Take(":");

      if (stream.TakeIfPossible("except")) {
        block->SetExceptHandler();
      } else if (stream.TakeIfPossible("finally")) {
        block->SetFinallyHandler();
      }

      if (stream.TakeIfPossible("deoptimized")) {
        block->set_deoptimized(true);
      }

      if (stream.TakeIfPossible("[")) {
        while (!stream.TakeIfPossible("]")) {
          stream.Take("%");
          int64_t value_number = stream.Int64();
          if (value_number != next_value_number) {
            return stream.Error(absl::InvalidArgumentError(
                absl::StrCat("expected block argument to be numbered `",
                             next_value_number, "'")));
          }
          value_numbering[next_value_number++] = block->CreateBlockArgument();
          stream.TakeIfPossible(",");
          S6_RETURN_IF_ERROR(stream.status());
        }
      }

      stream.TakeEndOfLine();
      S6_RETURN_IF_ERROR(stream.status());

      stream.SkipEmptyLines();
      continue;
    }

    if (!block) {
      return stream.Error(
          absl::InvalidArgumentError("instruction found outside of block"));
    }

    S6_ASSIGN_OR_RETURN(Instruction * inst,
                        ParseInstruction(&stream, &value_numbering,
                                         next_value_number, &f, mgr));
    value_numbering[next_value_number++] = inst;
    block->push_back(inst);

    if (TerminatorInst* ti = dyn_cast<TerminatorInst>(inst)) {
      for (Block* succ : ti->successors()) {
        S6_CHECK(succ);
        succ->AddPredecessor(block);
      }
    }
    stream.SkipEmptyLines();
  }

  stream.Take("}");
  return f;
}

}  // namespace deepmind::s6
