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

#ifndef THIRD_PARTY_DEEPMIND_S6_STRONGJIT_TEST_UTIL_H_
#define THIRD_PARTY_DEEPMIND_S6_STRONGJIT_TEST_UTIL_H_

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "utils/logging.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {

// A trivial subclass of std::string. Strings wrapped in RawString can be
// printed by UniversalPrint. We don't escape newlines to make test output
// easier to read.
struct RawString : std::string {
  using basic_string::basic_string;
  explicit RawString(basic_string s) : basic_string(std::move(s)) {}

  // UniversalPrint is a customization point called by gUnit to pretty-print
  // an object.
  friend void UniversalPrint(const RawString& value, std::ostream* os) {
    *os << "R\"(" << value << ")\"";
  }
};

// A line-oriented, stateful, string matcher tool.
//
// This takes a string corpus as input and treats it as a sequence of lines,
// conceptually with a cursor that starts at the beginning. The cursor holds
// a line and column, so matches on the same line after a prior match are
// possible.
//
// Match methods on this class find matches from the cursor onwards that all
// occur on a single line, and advance the cursor.
//
// On a successful match the cursor is moved to one column after the match.
// If this exhausts the line, the cursor stays at the end of the current line.
// On failure the cursor is undefined and so are all new calls to this matcher
// object.
//
// For example:
//   LineMatcher lm(R"(
//     Lorem ipsum
//     dolor sit amet ipsum
//     Quis custodiet ipsos custodes?)");
//   // Matches the second line, with the cursor after "amet". Note the two
//   // matches were separated by "sit", which was skipped.
//   S6_ASSERT_OK(lm.OnAnyLine("dolor", "amet"));
//
// The arguments that can be given to matcher methods can be more than simple
// strings:
//   lm.Regex(".*")    : matches a regular expression.
//   lm.Def("x", ".*") : matches a regular expression (.*) and assigns the
//                         matched string to the named variable "x". The entire
//                         match is assigned - capture groups are not allowed.
//   lm.Ref("x")       : substitutes in the value of the named variable "x".
//
// These can be stitched together, for example:
//   // Matches "Lorem" and "ipsum", the latter of which is assigned to "v".
//   S6_ASSERT_OK(lm.OnAnyLine("Lorem", lm.Def("v", "\\S+")));
//   // Matches "sit", "amet" and "ipsum" on line 2.
//   S6_ASSERT_OK(lm.OnNextLine("sit", "amet", lm.Ref("v")));
//
// There is also the ability to ensure that tokens are *not* matched:
//   LineMatcher lm(/*as above*/);
//   // Fails, because "sit amet" matches in the second line but "ipsum" matches
//   // in the first line.
//   S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("ipsum"), "sit amet"));
//
// See member documentation for more precise usage examples.
//
// The rationale for this tool is for testing Strongjit IR. Strongjit IR is
// sequential and also encodes a dataflow graph. Tests frequently wish to test
// both; for example:
//
//   %1 = call_python ...
//   %2 = incref %1
//   %3 = decref %1
//
// A gMock matcher could easily identify the first two lines, but the third
// would require storing the matched value %1 and reusing it, something that
// gMock can't do (or at least is an abuse of gMock).
//
// The LineMatcher class is inspired by the LLVM FileCheck tool, used in LLVM
// and MLIR for textual matching of IR:
//   https://llvm.org/docs/CommandGuide/FileCheck.html
//
class LineMatcher {
 public:
  // Scans forward from the cursor to find any line that matches all of the
  // arguments. If no lines match, a NOT_FOUND status is returned.
  //
  // On a successful match the cursor is moved after the match. On failure the
  // cursor is undefined and so are all new calls to this matcher object.
  //
  // For example:
  //   LineMatcher lm("abc\ndef\nghi");
  //   // Matches "de" on the second line. The cursor now points to "f".
  //   lm.OnAnyLine("d", lm.Regex("."));
  template <typename... Tokens>
  absl::Status OnAnyLine(Tokens... tokens);

  // Helper type for WithoutSeeing, used within OnAnyLine(lm.WithoutSeeing())
  // below.
  struct WithoutSeeingType {
    // A regular expression that matches all of the tokens given to
    // WithoutSeeing.
    std::string regex;
  };

  // Helper for OnAnyLine(lm.WithoutSeeing()). This evaluates `tokens` in the
  // current variable context and returns a WithoutSeeingType that can be used
  // as the first argument to OnAnyLine.
  //
  // `tokens` must not contain any Defs.
  template <typename... Tokens>
  WithoutSeeingType WithoutSeeing(Tokens... tokens) {
    return {.regex = MakeRegexFromTokens(tokens...)};
  }

  // Scans forward from the cursor to find any line that matches all of the
  // `tokens` arguments. If no lines match, a NOT_FOUND status is returned.
  // This is termed the "primary" match.
  //
  // The `without_seeing` argument specifies a sequence of tokens that must NOT
  // be matched between the starting and ending cursor points of the primary
  // match above. If a match is found, an ALREADY_EXISTS status is returned.
  //
  // The `without_seeing` argument may contain Regex and Ref tokens, but it is
  // undefined behavior if it contains a Def token.
  //
  // Note that when "without" and the primary match overlap, they are compared
  // based on the cursor position *after* the match, not at the beginning of the
  // match, and if the "without" match cursor precedes the "primary" match
  // cursor then an ALREADY_EXISTS error is returned.
  //
  // For example:
  //
  //   LineMatcher lm("one\ntwo three\nfour five\nsix");
  //
  //   // Returns ALREADY_EXISTS, because "four" "five" matches on line 3 but
  //   // "two" matched on line 2.
  //   S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("two"), "four", "five"));
  //
  //   // Returns ALREADY_EXISTS, because "four" occurs before "five" on line 3.
  //   S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("four"), "five"));
  //
  //   // Returns OK, because "five" does not occur before "four" even on the
  //   // same line.
  //   S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("five"), "four"));
  //
  template <typename... MatchTokens>
  absl::Status OnAnyLine(WithoutSeeingType without_seeing,
                         MatchTokens... match_tokens);

  // Scans forward from the cursor on the current line, attempting to match all
  // of Tokens.
  //
  // On a successful match the cursor is moved after the match. On failure the
  // cursor is undefined and so are all new calls to this matcher object.
  //
  // For example:
  //
  //   LineMatcher lm("abc\ndef\nghi");
  //
  //   // Fails to match, because the cursor is on line 1 and the first "d" is
  //   // on line 2.
  //   lm.OnThisLine("d");
  //
  template <typename... Tokens>
  absl::Status OnThisLine(Tokens... tokens);

  // Moves the cursor to the start of the next line and attempts to match all
  // of Tokens on that line.
  //
  // On a successful match the cursor is moved after the match. On failure the
  // cursor is undefined and so are all new calls to this matcher object.
  //
  // For example:
  //
  //   LineMatcher lm("abc\ndef\nghi");
  //
  //   // Matches "d" on line 2. The cursor is now at "e".
  //   lm.OnNextLine("d");
  //
  template <typename... Tokens>
  absl::Status OnNextLine(Tokens... tokens);

  // Token types, returned by Regex(), Def() and Ref() below.
  struct RegexToken {
    std::string expression;
  };
  struct DefToken {
    std::string variable_name;
    std::string expression;
  };
  struct RefToken {
    std::string variable_name;
  };

  // Matches a regular expression and discards the matched content.
  //
  // For example:
  //
  //   LineMatcher lm("harder\nbetter\nfaster\nstronger");
  //
  //   // Matches "b" "etter", with the cursor on the notional newline.
  //   lm.OnAnyLine("b", lm.Regex(".*"));
  //
  // Note a pitfall:
  //
  //   lm.OnAnyLine(lm.Regex(".*"), "r"); // WRONG: Will never match!
  //
  // The first token regex matches the entire line, and "r" will not match the
  // empty string.
  //
  static RegexToken Regex(absl::string_view expression);

  // Matches a regular expression and saves the matched content in a named
  // variable. The variable can be read by a subsequent Ref() expression.
  //
  // Note that arguments are matched left-to-right, so a Ref may refer to a
  // Def to its left.
  //
  // For example:
  //
  //   LineMatcher lm("harder\nbetter\nfaster\nstronger");
  //
  //   lm.OnAnyLine("be",             // Matches "be" on line 2.
  //                lm.Def("x", "."), // Assigns "t" to variable "x".
  //                lm.Ref("x"));     // Matches "t" as the content of "x".
  //
  static DefToken Def(absl::string_view variable_name,
                      absl::string_view expression);

  // Matches the content of a variable defined by Def().
  static RefToken Ref(absl::string_view variable_name);

  // Creates a new LineMatcher and initializes the cursor to the beginning.
  explicit LineMatcher(std::string corpus)
      : content_(std::move(corpus)), cursor_(content_) {}

  explicit LineMatcher(absl::string_view corpus)
      : LineMatcher(std::string(corpus)) {}

  LineMatcher(const LineMatcher&) = delete;
  LineMatcher& operator=(const LineMatcher&) = delete;

 private:
  // The implementation of a cursor, which maintains a line (from the current
  // column onward) and the tail of the corpus. The cursor can be transitioned
  // into an undefined state after which all operations are undefined.
  class Cursor {
   public:
    // Creates a cursor at the start of `corpus`.
    explicit Cursor(absl::string_view corpus) : rest_(corpus), line_number_(0) {
      // Latch the first line into line_, advance to line number 1.
      AdvanceToNextLine();
    }

    // Returns the line and line number. The line contains the remnants of the
    // current line - the content from the notional column of the cursor up to,
    // but not including, end of the line.
    absl::string_view* mutable_line() { return &line_; }
    absl::string_view line() const { return line_; }
    int64_t line_number() const { return line_number_; }
    int64_t column_number() const {
      return full_line_.size() - line_.size() + 1;
    }

    // Advances the cursor to the next line and returns this.
    Cursor& AdvanceToNextLine() {
      std::pair<absl::string_view, absl::string_view> split =
          absl::StrSplit(rest_, absl::MaxSplits('\n', 1));
      std::tie(line_, rest_) = split;
      ++line_number_;
      full_line_ = line_;
      return *this;
    }

    // Returns true if the cursor it at the end of the corpus.
    bool IsAtEnd() const { return line_.empty() && rest_.empty(); }

    // Returns the status of the cursor; this is OK unless TransitionToUndefined
    // has been called.
    absl::Status status() const { return undefined_status_; }

    // Transitions the cursor to an undefined state. No more operations on the
    // cursor (apart from calls to status()) are valid.
    void TransitionToUndefined() {
      undefined_status_ = absl::FailedPreconditionError(
          "LineMatcher cursor is in an undefined state");
    }

    bool operator<(const Cursor& c) const {
      // Cursors must always point to the same corpus, so we can just perform
      // a pointer comparison.
      return line_.data() < c.line_.data();
    }

    std::string ToString() const {
      return absl::StrCat("\"", line(), "\", line ", line_number(), " column ",
                          column_number());
    }

   private:
    // The partial (remainder) of the line.
    absl::string_view line_;

    // The rest of the corpus after line_.
    absl::string_view rest_;

    // The line_ in full, for calculating column offsets.
    absl::string_view full_line_;

    int64_t line_number_;
    absl::Status undefined_status_;
  };

  // Attempts to match `token`. Updates the cursor on success and returns true,
  // otherwise false with the cursor being implicitly in an undefined state
  // (TransitionToUndefined has not been called).
  bool Check(Cursor* cursor, absl::string_view token) {
    return RE2::FindAndConsume(cursor->mutable_line(), RE2::QuoteMeta(token));
  }
  bool Check(Cursor* cursor, const RegexToken& token) {
    return RE2::FindAndConsume(cursor->mutable_line(), token.expression);
  }
  bool Check(Cursor* cursor, const DefToken& token) {
    std::string& capture_text = variables_[token.variable_name];
    return RE2::FindAndConsume(cursor->mutable_line(),
                               absl::StrCat("(", token.expression, ")"),
                               &capture_text);
  }
  bool Check(Cursor* cursor, const RefToken& token) {
    // TODO: Better explanation text if the variable did not exist?
    auto it = variables_.find(token.variable_name);
    return it != variables_.end() &&
           RE2::FindAndConsume(cursor->mutable_line(),
                               RE2::QuoteMeta(it->second));
  }

  // Writes a textual version of token the output string.
  void Describe(std::string* output, absl::string_view token) {
    absl::StrAppend(output, " ", token);
  }
  void Describe(std::string* output, const RegexToken& token) {
    absl::StrAppend(output, " {{", token.expression, "}}");
  }
  void Describe(std::string* output, const DefToken& token) {
    absl::StrAppend(output, " [[", token.variable_name, ":", token.expression,
                    "]]");
  }
  void Describe(std::string* output, const RefToken& token) {
    absl::StrAppend(output, " [[", token.variable_name, "]]");
  }

  // Creates a regular expression matching token. Used for `without` matching.
  std::string MakeRegex(absl::string_view token) {
    return RE2::QuoteMeta(token);
  }
  std::string MakeRegex(const RegexToken& token) { return token.expression; }
  std::string MakeRegex(const DefToken&) = delete;
  std::string MakeRegex(const RefToken& token) {
    auto it = variables_.find(token.variable_name);
    S6_CHECK(it != variables_.end())
        << "Variable is undefined when used: " << token.variable_name;
    return it->second;
  }

  // Performs a LineMatcher test on line_. Returns true if the check matched,
  // false on failure.
  //
  // The cursor is modified; on success it points to the position of the current
  // line just after the match, on failure it is undefined.
  template <typename... Tokens>
  bool CheckThisLine(Cursor* cursor, Tokens... tokens) {
    return (Check(cursor, tokens) && ... && true);
  }

  // Returns failure text for the given list of tokens in *output.
  template <typename... Tokens>
  std::string TokensText(Tokens... tokens) {
    std::string output;
    (Describe(&output, tokens), ...);
    absl::StripAsciiWhitespace(&output);
    return output;
  }

  // Creates a single regular expression from `tokens`. Tokens must not include
  // any DefTokens.
  template <typename... Tokens>
  std::string MakeRegexFromTokens(Tokens... tokens) {
    return absl::StrJoin({MakeRegex(tokens)...}, ".*");
  }

  // Returns a string with the character `indicator` at `cursor_column`, and
  // `s` either preceding it or after it depending on `cursor_column`.
  std::string ContextMarkerString(char indicator, int64_t cursor_column,
                                  absl::string_view s) {
    const int64_t kIdealNumColumns = 80;
    std::string ret(cursor_column + s.size() + 2, ' ');
    ret[cursor_column] = indicator;

    if (cursor_column + s.size() + 2 < kIdealNumColumns) {
      // There is room to put the cursor indicator on the left of the text.
      absl::c_copy(s, ret.begin() + cursor_column + 2);
    } else {
      // Put the cursor indicator on the right of the text.
      absl::c_copy(s, ret.begin() + cursor_column - 2 - s.size());
    }
    absl::StripTrailingAsciiWhitespace(&ret);
    return ret;
  }

  // Returns a string containing the variable content. This is either empty
  // or contains the prefix "\n  ", ready to be joined to preceding text.
  std::string VariableContentText() {
    if (variables_.empty()) return "";

    auto pf = [](std::string* s, const auto& p) {
      absl::StrAppend(s, p.first, "=\"", p.second, "\"");
    };
    return absl::StrCat("\n  (with ", absl::StrJoin(variables_, ", ", pf), ")");
  }

  std::string CorpusContextText(
      absl::Span<const std::pair<absl::string_view, Cursor>> above_cursors,
      absl::Span<const std::pair<absl::string_view, Cursor>> below_cursors) {
    // Pad to account for the line number prefix: "%03d: ".
    const absl::string_view kPadString = "     ";
    std::string context;
    int64_t line = 1;
    for (auto str : absl::StrSplit(content_, '\n')) {
      for (const auto& [text, cursor] : above_cursors) {
        if (line == cursor.line_number()) {
          absl::StrAppend(
              &context, kPadString,
              ContextMarkerString('v', cursor.column_number() - 1, text), "\n");
        }
      }
      if (str.empty()) {  // Avoid ending a line with whitespace.
        absl::StrAppendFormat(&context, "%03d:\n", line);
      } else {
        absl::StrAppendFormat(&context, "%03d: %s\n", line, str);
      }
      for (const auto& [text, cursor] : below_cursors) {
        if (line == cursor.line_number()) {
          absl::StrAppend(
              &context, kPadString,
              ContextMarkerString('^', cursor.column_number() - 1, text), "\n");
        }
      }
      ++line;
    }
    return context;
  }

  // Outputs a failure message when a match could not be found. Transitions the
  // cursor to undefined. Returns a NOT_FOUND status.
  template <typename... Tokens>
  absl::Status FailedToMatch(const Cursor& from_cursor, Tokens... tokens) {
    std::string variable_message = VariableContentText();
    std::string token_message = TokensText(tokens...);
    std::string corpus_message = CorpusContextText(
        /*above_cursors=*/{std::make_pair("SCANNING FROM HERE", from_cursor)},
        /*below_cursors=*/{});

    cursor_.TransitionToUndefined();
    std::stringstream ss;
    ss << "LineMatcher expression " << token_message << "\n  Matching from "
       << from_cursor.ToString() << variable_message << "\n"
       << corpus_message;
    return absl::NotFoundError(ss.str());
  }

  // Outputs a failure message when a "without" match occurred before the
  // primary match. Transitions the cursor to undefined. Returns an
  // ALREADY_EXISTS status.
  absl::Status Failure(const Cursor& from_cursor,
                       const Cursor& primary_match_cursor,
                       const Cursor& without_match_cursor) {
    std::string variable_message = VariableContentText();
    std::string corpus_message = CorpusContextText(
        /*above_cursors=*/{std::make_pair("SCANNING FROM HERE", from_cursor),
                           std::make_pair("'WITHOUT' MATCH ENDED HERE",
                                          without_match_cursor)},
        /*below_cursors=*/{
            std::make_pair("PRIMARY MATCH ENDED HERE", primary_match_cursor)});

    cursor_.TransitionToUndefined();
    std::stringstream ss;
    ss << "LineMatcher 'Without' expression matched before primary "
          "expression\n"
       << "  Matching from " << from_cursor.ToString()
       << "\n  Primary match finished at " << primary_match_cursor.ToString()
       << "\n  'Without' match finished at " << without_match_cursor.ToString()
       << variable_message << "\n"
       << corpus_message;
    return absl::AlreadyExistsError(ss.str());
  }

  // The text to search.
  std::string content_;

  // The current cursor.
  Cursor cursor_;

  // The current value of all variables defined by Capture(). Because
  // performance is not important and we want a consistent iteration order
  // during printing, this uses std::map rather than an unordered_map.
  std::map<std::string, std::string> variables_;

  // If an operation has failed, this contains the FailedPreconditionError
  // that we will serve to succeeding operations.
  absl::Status cursor_undefined_;
};

// Defined out of line to keep the API readable.

template <typename... Tokens>
absl::Status LineMatcher::OnAnyLine(Tokens... tokens) {
  S6_RETURN_IF_ERROR(cursor_.status());
  // Save the current cursor for the error message below.
  Cursor start_cursor = cursor_;
  while (true) {
    if (CheckThisLine(&cursor_, tokens...)) {
      return absl::OkStatus();
    } else if (cursor_.AdvanceToNextLine().IsAtEnd()) {
      return FailedToMatch(start_cursor, tokens...);
    }
  }
}

template <typename... MatchTokens>
absl::Status LineMatcher::OnAnyLine(WithoutSeeingType without_seeing,
                                    MatchTokens... match_tokens) {
  S6_RETURN_IF_ERROR(cursor_.status());
  Cursor start_cursor = cursor_;
  S6_RETURN_IF_ERROR(OnAnyLine(match_tokens...));
  Cursor match_cursor = cursor_;

  // Rewind the cursor to the start and look for the negative match.
  cursor_ = start_cursor;
  if (OnAnyLine(Regex(without_seeing.regex)).ok()) {
    // The match was found. Was it found before the match cursor?
    if (cursor_ < match_cursor) {
      return Failure(start_cursor, match_cursor, cursor_);
    }
  }
  // Either a without match wasn't found, or it was found after the match.
  cursor_ = match_cursor;
  return absl::OkStatus();
}

template <typename... Tokens>
inline absl::Status LineMatcher::OnNextLine(Tokens... tokens) {
  S6_RETURN_IF_ERROR(cursor_.status());
  cursor_.AdvanceToNextLine();
  return OnThisLine(tokens...);
}

template <typename... Tokens>
inline absl::Status LineMatcher::OnThisLine(Tokens... tokens) {
  S6_RETURN_IF_ERROR(cursor_.status());
  Cursor start_cursor = cursor_;
  if (CheckThisLine(&cursor_, tokens...)) {
    return absl::OkStatus();
  } else {
    return FailedToMatch(start_cursor, tokens...);
  }
}

inline LineMatcher::RegexToken LineMatcher::Regex(
    absl::string_view expression) {
  return {.expression = std::string(expression)};
}

inline LineMatcher::DefToken LineMatcher::Def(absl::string_view variable_name,
                                              absl::string_view expression) {
  return {.variable_name = std::string(variable_name),
          .expression = std::string(expression)};
}

inline LineMatcher::RefToken LineMatcher::Ref(absl::string_view variable_name) {
  return {.variable_name = std::string(variable_name)};
}

// Helpers for common capture kinds in Strongjit IR - blocks and values.

// Captures a block by name - &[0-9]+.
inline LineMatcher::DefToken DefBlock(absl::string_view variable_name) {
  return LineMatcher::Def(variable_name, "&[0-9]+");
}

// Captures a value (not a block) by name - %[0-9]+.
inline LineMatcher::DefToken DefValue(absl::string_view variable_name) {
  return LineMatcher::Def(variable_name, "%[0-9]+");
}

// Matches a block but ignores its number.
inline LineMatcher::RegexToken AnyBlock() {
  return LineMatcher::Regex("&[0-9]+");
}

// Matches a value but ignores its number.
inline LineMatcher::RegexToken AnyValue() {
  return LineMatcher::Regex("%[0-9]+");
}

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_STRONGJIT_TEST_UTIL_H_
