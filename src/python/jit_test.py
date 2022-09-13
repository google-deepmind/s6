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

"""Unit tests for correctness of s6 JIT."""

from absl import flags
from absl.testing import absltest

from s6.python import api as s6
from pybind11_abseil import status

FLAGS = flags.FLAGS


class FailedToCompileError(RuntimeError):
  pass


def s6compile(f):
  fjit = s6.jit(f)
  try:
    s6.inspect(f).force_compile()
  except status.StatusNotOk as e:
    raise FailedToCompileError(f'{e}') from None
  return fjit


def expect_compile_failure(f):
  """Raises an AssertionError if f(..) does not raise FailedToCompileError."""

  def wrapper(*args, **kwargs):
    try:
      f(*args, **kwargs)
    except FailedToCompileError as e:
      raise absltest.SkipTest(f'{e}')
    else:
      raise AssertionError(f'Expected {f} to fail with compile error')

  return wrapper


class JitTest(absltest.TestCase):

  def compile_and_make_checker(self, f):
    fc = s6compile(f)

    def assert_output_matches(*a):
      with self.subTest(args=a):
        self.assertEqual(f(*a), fc(*a))

    return assert_output_matches

  def test_add(self):

    def f(a, b):
      return a + b

    assert_output_matches = self.compile_and_make_checker(f)
    assert_output_matches(3, 4)
    assert_output_matches(5, 6)

  def test_finally_return_simple(self):

    def f(i):
      try:
        if i & 1:
          return 1
      finally:
        if i & 2:
          return 2
      return 3

    assert_output_matches = self.compile_and_make_checker(f)
    for i in range(4):
      assert_output_matches(i)

  def test_finally_return(self):

    def f(i):
      try:
        if i & 1:
          return 1
        try:
          if i & 2:
            return 4
        finally:
          if i & 4:
            return 5
      finally:
        if i & 8:
          return 2
        try:
          if i & 16:
            return 6
        finally:
          if i & 32:
            return 7
      return 3

    assert_output_matches = self.compile_and_make_checker(f)
    for i in range(64):
      assert_output_matches(i)

  def test_finally_falltrough(self):

    def f():
      a = 4
      try:
        a += 6
      finally:
        a += 5
      return a

    assert_output_matches = self.compile_and_make_checker(f)
    assert_output_matches()

  def test_finally_loop(self):

    def f(i):
      try:
        x = 0
        for _ in range(5):
          x += 1
          try:
            x += 10
            if i & 2:
              break
            x += 100
            if i & 4:
              continue
            x += 1000
            if i & 8:
              return x
          finally:
            x += 10**4
            if i & 16:
              break
            x += 10**5
            if i & 32:
              return x
          x += 10**6
        return x
      finally:
        if i & 1:
          return -1

    assert_output_matches = self.compile_and_make_checker(f)
    for i in range(64):
      assert_output_matches(i)

  def test_except(self):

    def raiseif(b, a):
      if b:
        raise RuntimeError('raiseif')
      return a

    def f(i):
      a = 1
      try:
        a += 2
        a = raiseif(i & 1, a + 4)
        a += 8
        a = raiseif(i & 2, a + 16)
        a += 32
        raise RuntimeError('raise anyway')
      except RuntimeError:
        a += 64
      return a

    assert_output_matches = self.compile_and_make_checker(f)
    for i in range(4):
      assert_output_matches(i)

  def test_finally_except(self):

    def raiseif(b, a):
      if b:
        raise RuntimeError('raiseif')
      return a

    def f(i):
      try:
        a = 1
        try:
          a += 2
          a = raiseif(i & 1, a + 4)
          a += 8
          raise RuntimeError('raise anyway')
        finally:
          a += 16
          if i & 2:
            return a
      except RuntimeError:
        a += 32
      return a + 64

    assert_output_matches = self.compile_and_make_checker(f)
    for i in range(4):
      assert_output_matches(i)

  def test_using_super(self):

    class X:

      def g(self, a):
        return 2 * a + 3

    class Y(X):

      def f(self, x, y):
        return super().g(x + y)

    y = Y()

    assert_output_matches = self.compile_and_make_checker(Y.f)
    assert_output_matches(y, 3, 5)


if __name__ == '__main__':
  absltest.main()
