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

"""Tests for s6.strongjit.dis6."""

from absl.testing import absltest
from absl.testing import parameterized

from s6.strongjit import dis6
from pybind11_abseil import status


# Each test checks if the first line, the definition of the function, is
# correct.
class Dis6Test(parameterized.TestCase):

  def test_dis6_function(self):

    def foo():
      pass

    ir_string = dis6.dis6(foo)
    self.assertNotEmpty(ir_string)
    self.assertTrue(ir_string.startswith('function foo {'))

  def test_dis6_lambda(self):
    x = lambda a: a + 1

    ir_string = dis6.dis6(x)
    self.assertNotEmpty(ir_string)
    self.assertTrue(ir_string.startswith('function <lambda> {'))

  def test_dis6_bound_method(self):

    class Foo:

      def bar(self):
        pass

    foo = Foo()
    ir_string = dis6.dis6(foo.bar)
    self.assertNotEmpty(ir_string)
    self.assertTrue(ir_string.startswith('function bar {'))

  def test_dis6_unbound_method(self):

    class Foo:

      def bar():
        pass

    ir_string = dis6.dis6(Foo.bar)
    self.assertNotEmpty(ir_string)
    self.assertTrue(ir_string.startswith('function bar {'))

  def test_dis6_callable(self):

    class Foo:

      def __call__():
        pass

    foo = Foo()
    ir_string = dis6.dis6(foo.__call__)
    self.assertNotEmpty(ir_string)
    self.assertTrue(ir_string.startswith('function __call__ {'))

  class A:
    pass

  @parameterized.parameters(1, 'test123', A, A())
  def test_dis6_invalid_argument(self, obj):
    if obj == self.A:
      with self.assertRaisesRegex(status.StatusNotOk,
                                  'Argument must be a function or method.'):
        dis6.dis6(obj)
    else:
      with self.assertRaisesRegex(TypeError,
                                  'incompatible function arguments.'):
        dis6.dis6(obj)


if __name__ == '__main__':
  absltest.main()
