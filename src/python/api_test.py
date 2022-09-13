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

"""Tests for s6.python.api."""

import time

from absl import flags
from absl.testing import absltest

from s6.python import api

FLAGS = flags.FLAGS


class ApiTest(absltest.TestCase):

  def test_jit_decorator(self):

    @api.jit
    def f(a, b=2):
      return a + b

    self.assertEqual(f(3, b=1), 4)

  def test_hot_jit_decorator(self):

    @api.jit
    def f(a, b=2):
      return a + b

    # TODO: Does this demonstrate that our barrier to compilation
    # (profile_instruction_interval) is too high?
    for i in range(500000):
      f(i)
    self.assertTrue(api.inspect(f).is_compiled)
    self.assertEqual(f(3, b=1), 4)

  def test_hot_jit_decorator_as_method(self):

    class C(object):

      @api.jit
      def f(self, a, b=2):
        return a + b

    # TODO: Does this demonstrate that our barrier to compilation
    # (profile_instruction_interval) is too high?
    c = C()
    for i in range(500000):
      c.f(i)
    self.assertTrue(api.inspect(c.f).is_compiled)
    self.assertEqual(c.f(3, b=1), 4)

  def test_jit_as_method(self):

    class C(object):

      @api.jit
      def f(self, a):
        return a + 1

    c = C()
    self.assertEqual(c.f(2), 3)

  def test_jit_decorator_on_recursive_function(self):

    @api.jit
    def fib(n):
      return fib(n - 1) + fib(n - 2) if n >= 2 else 1

    for _ in range(1000):
      fib(10)

    self.assertTrue(api.inspect(fib).is_compiled)

  def test_inspect(self):

    @api.jit
    def f(a, b=2):
      return a + b

    f(2)
    i = api.inspect(f)
    self.assertFalse(i.is_compiled)
    self.assertRaises(api.NotCompiledError, lambda: i.strongjit)

    i.force_compile()
    self.assertTrue(i.is_compiled)
    self.assertIn('type_feedback', i.strongjit)
    self.assertNotEmpty('type_feedback', i.x86asm)
    self.assertEqual(f(2), 4)

    i.deoptimize()
    self.assertFalse(i.is_compiled)
    self.assertEqual(f(2), 4)

  def test_jit_and_interpret(self):

    @api.jit
    def f(a, b=2):
      return a + b

    i = api.inspect(f)
    i.force_compile()
    self.assertTrue(i.is_compiled)
    self.assertNotIn('type_feedback', i.strongjit)
    self.assertEqual(f._interpret(2), 4)
    i.deoptimize()
    self.assertFalse(i.is_compiled)
    i.force_compile()
    self.assertTrue(i.is_compiled)
    self.assertIn('type_feedback', i.strongjit)
    self.assertEqual(f(2), 4)

  def test_jit_and_evaluate(self):

    @api.jit
    def f(a, b=2):
      return a + b

    i = api.inspect(f)
    i.force_compile()
    self.assertEqual(f._evaluate(2), 4)
    self.assertEqual(f(2), 4)

  def test_jit_forwards_docstring(self):

    @api.jit
    def f():
      """I am docstring."""
      return None

    self.assertEqual(f.__doc__, 'I am docstring.')

  def test_jit_forwards_name(self):

    @api.jit
    def f():
      return None

    self.assertEqual(f.__name__, 'f')



if __name__ == '__main__':
  absltest.main()
