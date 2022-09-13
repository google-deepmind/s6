# Copyright 2021 The s6 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for s6.python.type_feedback."""

import sys

from absl.testing import absltest

from s6.classes.python import classes
from s6.python import type_feedback

classes.adopt_existing_types()
classes.adopt_new_types()


def _type_feedback(f):
  return type_feedback.extract_from_code_object(f.__code__)


def _only_nonempty_type_feedback(f):
  return list(filter(lambda x: x != "empty", _type_feedback(f)))[0]


@absltest.skipIf(sys.version_info.minor >= 7, "CALL_METHOD not supported yet")
class TypeFeedbackTest(absltest.TestCase):

  def test_binary_add_custom_class(self):

    def f(x):
      return x + x

    class C(object):

      def __add__(self, other):
        pass

    self.assertNotEqual(classes.classid(C()), 0)
    f(C())
    f(C())

    self.assertStartsWith(_only_nonempty_type_feedback(f), "monomorphic, C#")

  def test_binary_add_longs(self):

    def f(x):
      return x + x

    # Run f with only longs.
    for i in range(5):
      f(i)

    self.assertStartsWith(_only_nonempty_type_feedback(f), "monomorphic, int#")

  def test_binary_mul_multiple_types_is_polymorphic(self):

    def f(x, y):
      return x * y

    for i in range(5):
      f(i, float(i))

    self.assertRegex(
        _only_nonempty_type_feedback(f),
        r"polymorphic, either float#\d+ or int#")

  def test_many_types_is_megamorphic(self):

    def f(x):
      return x * x

    for _ in range(5):

      # This is a new Class every time through the loop.
      class C(object):

        def __mul__(self, other):
          pass

      f(C())

    self.assertEqual(_only_nonempty_type_feedback(f), "megamorphic")

  def test_getattr(self):

    def f(x):
      return x.__len__

    f("abc")

    self.assertStartsWith(_only_nonempty_type_feedback(f), "monomorphic, str#")

  def test_setattr(self):

    def f(x):
      x.y = 2

    class C(object):
      pass

    class D(object):
      pass

    f(D())
    f(C())

    self.assertRegex(
        _only_nonempty_type_feedback(f), r"polymorphic, either D#\d+ or C#")


if __name__ == "__main__":
  absltest.main()
