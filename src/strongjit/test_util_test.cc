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

#include "strongjit/test_util.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utils/matchers.h"

namespace deepmind::s6 {
namespace {

using ::deepmind::s6::matchers::StatusIs;
using ::testing::HasSubstr;
using ::testing::StrEq;

// Input string used by all tests.
const absl::string_view kTestInput = R"(Test line 1
line 2

line three follows blank and does not end in newline)";

TEST(LineMatcherTest, CheckSimpleStrings) {
  LineMatcher lm(kTestInput);

  // Check from the first character.
  S6_ASSERT_OK(lm.OnAnyLine("Test"));

  // Check that we can find substrings on the same line.
  S6_ASSERT_OK(lm.OnAnyLine("line 1"));

  // Check that we can find substrings on the next line.
  S6_ASSERT_OK(lm.OnAnyLine("2"));

  // And that we can skip newlines and capture multiple tokens separated by
  // whitespace.
  S6_ASSERT_OK(lm.OnAnyLine("follows", "blank and"));

  // Also while we're here, a simple failure.
  ASSERT_THAT(
      lm.OnAnyLine("does not", "exist"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("does not exist")));
}

TEST(LineMatcherTest, CheckCompoundFailure) {
  LineMatcher lm(kTestInput);

  // The first two expressions should match but the latter should not.
  ASSERT_THAT(lm.OnAnyLine("Test", "line", "3"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(LineMatcherTest, CheckSameLine) {
  LineMatcher lm(kTestInput);

  S6_ASSERT_OK(lm.OnThisLine("line"));

  ASSERT_THAT(lm.OnThisLine("2"), StatusIs(absl::StatusCode::kNotFound));
}

TEST(LineMatcherTest, CheckNextLine) {
  LineMatcher lm(kTestInput);

  // Anchor to the first line.
  S6_ASSERT_OK(lm.OnAnyLine("line 1"));

  S6_ASSERT_OK(lm.OnNextLine("line 2"));

  // Note the empty line is NOT skipped.
  ASSERT_THAT(lm.OnNextLine("line four"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Matching from \"\", line 3")));
}

TEST(LineMatcherTest, Regex) {
  LineMatcher lm(kTestInput);

  S6_ASSERT_OK(lm.OnAnyLine("Test", lm.Regex("line [0-9]")));

  // Capture as much of "line 2..." as possible. This must not cross a line
  // boundary.
  S6_ASSERT_OK(lm.OnAnyLine(lm.Regex("line .*")));

  // Given that the above didn't cross a line boundary, we should be able to
  // match the last line.
  S6_ASSERT_OK(lm.OnAnyLine("line three"));
}

TEST(LineMatcherTest, CaptureAndRef) {
  LineMatcher lm(kTestInput);

  S6_ASSERT_OK(lm.OnAnyLine("Test", lm.Def("myvar", "\\S+"), "1"));

  // "myvar" should have content "line".
  S6_ASSERT_OK(lm.OnNextLine(lm.Ref("myvar"), "2"));
}

TEST(LineMatcherTest, CaptureAndRefNegative) {
  LineMatcher lm(kTestInput);

  S6_ASSERT_OK(lm.OnAnyLine("Test", lm.Def("myvar", ".*")));

  // "myvar" should have content "line 1", and should not be matchable.
  ASSERT_THAT(lm.OnAnyLine(lm.Ref("myvar")),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(LineMatcherTest, FailureMessage) {
  LineMatcher lm(kTestInput);

  // Set up some capture groups that we will check in the error message.
  S6_ASSERT_OK(lm.OnAnyLine(lm.Def("aa", "\\S+")));
  S6_ASSERT_OK(lm.OnAnyLine(lm.Def("bb", "li.")));

  ASSERT_THAT(
      lm.OnAnyLine("foo", lm.Regex("bar"), "baz", lm.Def("too", "abc"),
                   lm.Ref("aa")),
      StatusIs(absl::StatusCode::kNotFound,
               StrEq(
                   R"(LineMatcher expression foo {{bar}} baz [[too:abc]] [[aa]]
  Matching from "e 1", line 1 column 9
  (with aa="Test", bb="lin")
             v SCANNING FROM HERE
001: Test line 1
002: line 2
003:
004: line three follows blank and does not end in newline
)")));
}

TEST(LineMatcherTest, EmptyLineFailureMessage) {
  LineMatcher lm(kTestInput);

  // Progress to the last line.
  S6_ASSERT_OK(lm.OnAnyLine("line three"));

  // Get a match failure on this line. There was an empty line on line 3, so
  // this should be labelled "line 4".
  ASSERT_THAT(lm.OnThisLine("foo"),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr(", line 4")));
}

TEST(LineMatcherTest, Without) {
  LineMatcher lm(kTestInput);

  // Test an overlap; we must see "line" before "line 1".
  S6_ASSERT_OK(lm.OnAnyLine(lm.WithoutSeeing("line", "1"), "line"));

  // Another overlap; this time the negative match is the same as the main
  // match, but the main match is longer so the negative match takes precedence.
  //
  // This is slightly counterintuitive but is documented; it's because we can
  // only compare cursors after a match has completed, rather than where the
  // match started completing.
  ASSERT_THAT(lm.OnAnyLine(lm.WithoutSeeing("line"), "line", "2"),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(LineMatcherTest, Without2) {
  LineMatcher lm(kTestInput);

  // "th" "ee" should match "three".
  ASSERT_THAT(lm.OnAnyLine(lm.WithoutSeeing("th", "ee"), "follows", "blank"),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(LineMatcherTest, WithoutFailureMessage) {
  LineMatcher lm(kTestInput);

  S6_ASSERT_OK(lm.OnAnyLine("line 1"));
  ASSERT_THAT(
      lm.OnAnyLine(lm.WithoutSeeing("line"), "follows"),
      StatusIs(
          absl::StatusCode::kAlreadyExists,
          StrEq(
              R"(LineMatcher 'Without' expression matched before primary expression
  Matching from "", line 1 column 12
  Primary match finished at " blank and does not end in newline", line 4 column 19
  'Without' match finished at " 2", line 2 column 5
                v SCANNING FROM HERE
001: Test line 1
         v 'WITHOUT' MATCH ENDED HERE
002: line 2
003:
004: line three follows blank and does not end in newline
                       ^ PRIMARY MATCH ENDED HERE
)")));
}

TEST(LineMatcherTest, WithoutFailureMessageReallyLongLine) {
  // Line 2 is longer than 80 characters.
  LineMatcher lm(
      absl::string_view("Hello,\nThe lazy fox jumped over the swift brown "
                        "horse and landed in the field pursued by hounds."));

  ASSERT_THAT(
      lm.OnAnyLine(lm.WithoutSeeing("Hello"), "pursued", "by"),
      StatusIs(
          absl::StatusCode::kAlreadyExists,
          StrEq(
              R"(LineMatcher 'Without' expression matched before primary expression
  Matching from "Hello,", line 1 column 1
  Primary match finished at " hounds.", line 2 column 82
  'Without' match finished at ",", line 1 column 6
     v SCANNING FROM HERE
          v 'WITHOUT' MATCH ENDED HERE
001: Hello,
002: The lazy fox jumped over the swift brown horse and landed in the field pursued by hounds.
                                                            PRIMARY MATCH ENDED HERE  ^
)")));
}

}  // namespace
}  // namespace deepmind::s6
