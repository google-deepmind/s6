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

#include "classes/class.h"

#include <cstdint>
#include <mutex>  // NOLINT

#include "absl/hash/hash_testing.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {
using testing::Eq;
using testing::Gt;
using testing::Ne;
using testing::NotNull;
using testing::Pair;
using testing::Pointee;
using testing::StrEq;
using testing::UnorderedElementsAre;

void PyInit() {
  static std::once_flag once;
  std::call_once(once, []() { Py_Initialize(); });
}

TEST(ClassTest, FirstClassIsNotZero) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  ASSERT_THAT(cls, NotNull());
  EXPECT_THAT(cls->name(), StrEq("abc"));
  EXPECT_THAT(cls->id(), Gt(0));
}

TEST(ClassTest, SubsequentClassesHaveMonotonicallyIncreasingId) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls1, Class::Create(mgr, "abc"));
  S6_ASSERT_OK_AND_ASSIGN(Class * cls2, Class::Create(mgr, "abc"));
  ASSERT_THAT(cls1, NotNull());
  ASSERT_THAT(cls2, NotNull());
  EXPECT_THAT(cls2->id(), Eq(cls1->id() + 1));
}

TEST(ClassTest, ClassByIdRoundtrips) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  ASSERT_THAT(cls, NotNull());
  EXPECT_THAT(cls->name(), StrEq("abc"));
  EXPECT_THAT(mgr.GetClassById(cls->id()), Eq(cls));
}

TEST(ClassTest, ApplyTransition) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  ASSERT_THAT(cls, NotNull());
  EXPECT_THAT(cls->attributes(), UnorderedElementsAre());

  AttributeDescription x =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(
      Class * cls_x,
      cls->ApplyTransition(ClassTransition::Add(x, DictKind::kCombined), x,
                           DictKind::kCombined));
  EXPECT_THAT(cls_x->name(), StrEq("abc+x"));
  EXPECT_THAT(cls_x->id(), Ne(cls->id()));
  EXPECT_THAT(cls_x->attributes(), UnorderedElementsAre(Pair("x", Pointee(x))));

  AttributeDescription y =
      AttributeDescription::CreateUnknown(mgr, "y", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(
      Class * cls_y,
      cls_x->ApplyTransition(ClassTransition::Add(y, DictKind::kCombined), y,
                             DictKind::kCombined));
  EXPECT_THAT(cls_y->name(), StrEq("abc+x+y"));
  EXPECT_THAT(cls_y->id(), Ne(cls_x->id()));
  EXPECT_THAT(cls_y->id(), Ne(cls->id()));
  EXPECT_THAT(cls_y->attributes(), UnorderedElementsAre(Pair("x", Pointee(x)),
                                                        Pair("y", Pointee(y))));

  // Note that X->Add(y)->Delete(y) does not result in X again. There is no good
  // reason for this apart from lack of implementing a global transition table.
  S6_ASSERT_OK_AND_ASSIGN(
      Class * cls_noy,
      cls_y->ApplyTransition(ClassTransition::Delete(y.name()),
                             AttributeDescription(), DictKind::kNonContiguous));
  EXPECT_THAT(cls_noy->name(), StrEq("abc+x+y-y"));
  EXPECT_THAT(cls_noy->id(), Ne(cls_x->id()));
  EXPECT_THAT(cls_noy->id(), Ne(cls_y->id()));
  EXPECT_THAT(cls_noy->id(), Ne(cls->id()));
  EXPECT_THAT(cls_noy->attributes(),
              UnorderedElementsAre(Pair("x", Pointee(x))));
}

TEST(ClassTest, Transition) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  ASSERT_THAT(cls, NotNull());
  EXPECT_THAT(cls->attributes(), UnorderedElementsAre());

  // Adding a non-behavioral 'x'.
  AttributeDescription x =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x,
                          cls->Transition(x, DictKind::kCombined));
  EXPECT_THAT(cls_x->name(), StrEq("abc+x"));
  EXPECT_THAT(cls_x->attributes(), UnorderedElementsAre(Pair("x", Pointee(x))));

  // Adding a new attribute with the same name 'x', again nonbehavioral. It has
  // a value. That shouldn't matter.
  AttributeDescription x2 =
      AttributeDescription::CreateUnknown(mgr, "x", false, Py_None);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x2,
                          cls_x->Transition(x2, DictKind::kCombined));
  EXPECT_EQ(cls_x2, cls_x);

  // Setting the attribute to IS behavioral should cause a transition.
  AttributeDescription x3 =
      AttributeDescription::CreateUnknown(mgr, "x", true, Py_None);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x3,
                          cls_x->Transition(x3, DictKind::kCombined));
  EXPECT_NE(cls_x3, cls_x);
  EXPECT_THAT(cls_x3->attributes(),
              UnorderedElementsAre(Pair("x", Pointee(x3))));

  // Changing 'x' to a different Attribute but with the same value should not
  // cause a transition.
  AttributeDescription x4 =
      AttributeDescription::CreateUnknown(mgr, "x", true, Py_None);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x4,
                          cls_x3->Transition(x4, DictKind::kCombined));
  EXPECT_EQ(cls_x4, cls_x3);

  // But changing it back again to x2 (non-behavioral) should again cause a
  // transition.
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x5,
                          cls_x4->Transition(x2, DictKind::kCombined));
  EXPECT_NE(cls_x5, cls_x4);
}

TEST(ClassTest, TransitionSplitDict) {
  PyInit();

  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  EXPECT_THAT(cls->attributes(), UnorderedElementsAre());

  AttributeDescription x =
      AttributeDescription::CreateUnknown(mgr, "x", false, nullptr);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x, cls->Transition(x, DictKind::kSplit));
  EXPECT_THAT(cls_x->GetInstanceSlot(*cls_x->attributes().at(x.name())), Eq(0));

  // Setting the attribute to IS behavioral should cause a transition.
  AttributeDescription x3 =
      AttributeDescription::CreateUnknown(mgr, "x", true, Py_None);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_x3,
                          cls_x->Transition(x3, DictKind::kSplit));
  EXPECT_NE(cls_x3, cls_x);
  // The instance slot should not have changed.
  EXPECT_THAT(cls_x3->GetInstanceSlot(*cls_x3->attributes().at(x.name())),
              Eq(0));

  AttributeDescription y =
      AttributeDescription::CreateUnknown(mgr, "y", true, Py_None);
  S6_ASSERT_OK_AND_ASSIGN(Class * cls_y,
                          cls_x3->Transition(y, DictKind::kSplit));
  EXPECT_THAT(cls_y->GetInstanceSlot(*cls_y->attributes().at(y.name())), Eq(1));
}

TEST(ClassTransitionTest, Hash) {
  ClassManager mgr;
  AttributeDescription a =
      AttributeDescription::CreateUnknown(mgr, "a", true, nullptr);
  AttributeDescription b =
      AttributeDescription::CreateUnknown(mgr, "b", false, nullptr);
  AttributeDescription c = AttributeDescription::CreateUnknown(
      mgr, "c", true, reinterpret_cast<PyObject*>(1UL));
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {ClassTransition::Add(a, DictKind::kCombined),
       ClassTransition::Add(a, DictKind::kSplit),
       ClassTransition::Delete(a.name()),
       ClassTransition::Mutate(b, DictKind::kCombined),
       ClassTransition::Mutate(c, DictKind::kEmpty),
       ClassTransition::Mutate(b, DictKind::kSplit)}));
}

TEST(ClassModificationListenerTest, Simple) {
  ClassManager mgr;
  S6_ASSERT_OK_AND_ASSIGN(Class * cls, Class::Create(mgr, "abc"));
  int64_t event_counter = 0;
  cls->AddListener([&event_counter]() { ++event_counter; });
  cls->UnderlyingTypeHasBeenModified(nullptr);
  cls->UnderlyingTypeHasBeenModified(nullptr);
  EXPECT_EQ(event_counter, 1);
}

}  // namespace
}  // namespace deepmind::s6
