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

"""Tests for s6.class.python.class."""

import contextlib

from absl.testing import absltest
from absl.testing import parameterized

from s6.classes.python import classes


@contextlib.contextmanager
def adopt_new_types():
  classes.adopt_new_types()
  yield
  classes.stop_adopting_new_types()


class ClassTest(parameterized.TestCase):

  def assertZeroClass(self, obj):
    self.assertEqual(classes.classid(obj), 0)

  def assertNonzeroClass(self, obj):
    self.assertNotEqual(classes.classid(obj), 0)

  def test_unadopted_object(self):
    # An unadopted object has a zero class.
    self.assertZeroClass(self)

  def test_adoptable(self):
    # An instance of custom type should be adoptable.

    class C(object):
      pass

    self.assertTrue(classes.adoptable(C()))

    # So should longs.
    self.assertTrue(classes.adoptable(42))

    # But modules have a custom setter, so they aren't expected to be
    # adoptable.
    self.assertFalse(classes.adoptable(classes))

    # Objects with existing instance attributes cannot be adopted.
    c = C()
    self.assertTrue(classes.adoptable(c))
    c.x = 2
    self.assertFalse(classes.adoptable(c))

    # Type is special.
    self.assertFalse(classes.adoptable(type))

    # Types derived from unadoptable types aren't adoptable.
    class D(type(classes)):
      pass

    d = D('x')
    self.assertFalse(classes.adoptable(d))

  def test_adopt(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    c_classid = classes.classid(c)
    self.assertNonzeroClass(c)

    # Two objects with the same underlying type should have the same class.
    d = C()
    self.assertEqual(c_classid, classes.classid(d))

    # But objects with different types should not.
    class E(object):
      pass

    e = E()
    classes.adopt(e)
    self.assertNonzeroClass(e)
    self.assertNotEqual(classes.classid(e), c_classid)

    # We should be able to adopt things like floats.
    f = 45.0
    classes.adopt(f)
    self.assertNonzeroClass(f)

  def test_adopt_type(self):

    class C(object):
      pass

    # New instances of C don't have a class.
    self.assertZeroClass(C())

    # But then I adopt C, and new instances now have a class.
    classes.adopt(C())
    self.assertNonzeroClass(C())
    # And the class of multiple objects is still the same.
    self.assertEqual(classes.classid(C()), classes.classid(C()))

    # But instances of (newly) derived types do NOT have a class.
    class D(C):
      pass

    self.assertZeroClass(D())

    # But if I start adopting new types...
    with adopt_new_types():

      # Then new instances of new types *should* have a class.
      class E(C):
        pass

      self.assertNonzeroClass(E())

  def test_adopt_type_object(self):

    class C(object):

      def __init__(self):
        pass

    # New instances of C don't have a class.
    self.assertZeroClass(C())

    # But then I adopt C, and new instances now have a class.
    classes.adopt(C())
    self.assertNonzeroClass(C())

    # Also, C now has a class (it's a type class).
    self.assertNonzeroClass(C)

    # The type class should have __call__, and __init__
    self.assertEqual(classes.get_class_attributes(C)['__call__'], type.__call__)
    self.assertEqual(classes.get_class_attributes(C)['__init__'], C.__init__)

  def test_adopt_metatype(self):
    # I turn on type tracking...
    with adopt_new_types():

      # And I define a metaclass...
      class Meta(type):

        def __new__(cls, name, bases, dct):
          x = super().__new__(cls, name, bases, dct)
          return x

      # And F uses that metaclass.
      class F(metaclass=Meta):
        pass

      # Instances of F should have classes already.
      f = F()
      self.assertNonzeroClass(f)

  def test_add_attribute_transitions_class(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    original_class = classes.classid(c)
    self.assertNonzeroClass(c)

    # I add an attribute, and I get a new class.
    c.x = 2
    self.assertNonzeroClass(c)
    self.assertNotEqual(original_class, classes.classid(c))

    # I add an attribute, but I'm cheeky about it. I ask for the dict and write
    # into that.
    c.__dict__['x'] = 2

    # This kills the class.
    self.assertZeroClass(c)

  def test_modify_attribute_does_not_transition(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    original_class = classes.classid(c)
    self.assertNonzeroClass(c)

    # I add an attribute, and I get a new class.
    c.x = 2
    class_with_x = classes.classid(c)
    self.assertNonzeroClass(c)
    self.assertNotEqual(original_class, class_with_x)

    # Add change c.x. I should have the same class.
    c.x = 3
    self.assertEqual(class_with_x, classes.classid(c))

    # Now I remove it. I should get a new class that isn't zero.
    del c.x
    self.assertNonzeroClass(c)
    self.assertNotEqual(original_class, classes.classid(c))
    self.assertNotEqual(class_with_x, classes.classid(c))

  def test_get_type_attributes(self):

    class C(object):

      def f(self):
        pass

    d = classes.get_type_attributes(C)
    self.assertEqual(d['f'], C.f)

    C.f = 22
    d = classes.get_type_attributes(C)
    self.assertEqual(d['f'], 22)

    class D(C):

      def f(self):
        pass

    d = classes.get_type_attributes(C)
    self.assertEqual(d['f'], 22)
    d = classes.get_type_attributes(D)
    self.assertEqual(d['f'], D.f)

  def test_get_type_attributes_metatype(self):

    # I define a metaclass...
    class Meta(type):

      def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        # All instances will have an `f` attribute.
        x.f = 2
        return x

    # And F uses that metaclass.
    class F(metaclass=Meta):
      pass

    d = classes.get_type_attributes(F)
    self.assertEqual(d['f'], 2)

  def test_type_modifications(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    self.assertNonzeroClass(c)
    original_class = classes.classid(c)
    self.assertNotIn('x', classes.get_class_attributes(c))

    C.x = 2
    self.assertEqual(classes.classid(c), original_class)
    self.assertIn('x', classes.get_class_attributes(c))

  def test_shadowed_attr(self):

    class C(object):

      def f(self):
        pass

    c = C()
    classes.adopt(c)
    original_class_id = classes.classid(c)
    c.f = 2
    f_class_id = classes.classid(c)
    self.assertNotEqual(original_class_id, f_class_id)
    self.assertEqual(classes.get_class_attributes(c)['f'], 2)

    # Setting the attribute to the same value shouldn't change anything.
    c.f = 2
    self.assertEqual(f_class_id, classes.classid(c))
    self.assertEqual(classes.get_class_attributes(c)['f'], 2)

    # But setting it to something else should erase any assumptions about what
    # c.f could hold.
    c.f = 3
    self.assertEqual(f_class_id, classes.classid(c))
    self.assertIsNone(classes.get_class_attributes(c)['f'])

    # And deleting the instance attribute should re-expose the type attribute.
    del c.f
    self.assertEqual(classes.get_class_attributes(c)['f'], C.f)

  def test_shadowed_data_descriptor(self):

    class Descr(object):

      def __get__(self, *args):
        pass

      def __set__(self, *args):
        pass

    descr = Descr()

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    c.x = 2
    self.assertEqual(classes.get_class_attributes(c)['x'], 2)
    original_class_id = classes.classid(c)

    # Let's add a data descriptor to C. That shouldn't cause a transition, but
    # the data descriptor should shadow `c.x`.
    C.x = descr
    self.assertEqual(original_class_id, classes.classid(c))
    self.assertEqual(classes.get_class_attributes(c)['x'], descr)

    # Now let's delete the data descriptor. We should get `c.x` back.
    del C.x
    self.assertEqual(original_class_id, classes.classid(c))
    self.assertEqual(classes.get_class_attributes(c)['x'], 2)

  def test_base_type_update(self):

    class C(object):
      pass

    class D(C):
      pass

    d = D()
    classes.adopt(d)
    self.assertNotIn('x', classes.get_class_attributes(d))

    C.x = 2
    self.assertIn('x', classes.get_class_attributes(d))

  def test_bases_update(self):

    # Deliberately build a very deep hierarchy. Adopt the bottom type in the
    # hierarchy and fiddle with the __bases__ member of D.
    class C(object):
      x = 2

    class D(C):
      pass

    class E(D):
      pass

    class F(E):
      pass

    class G(object):
      pass

    f = F()
    classes.adopt(f)
    self.assertIn('x', classes.get_class_attributes(f))

    D.__bases__ = (G,)
    self.assertNotIn('x', classes.get_class_attributes(f))

  def test_divergent_transition(self):

    class C(object):
      pass

    c1 = C()
    classes.adopt(c1)
    c2 = C()
    classes.adopt(c2)

    c1.x = 42
    c2.x = 23

    # 'x' must be in the attributes list but must not have a known value,
    # because the value differs between c1 and c2.
    self.assertEqual(classes.classid(c1), classes.classid(c2))
    self.assertIsNone(classes.get_class_attributes(c1)['x'])
    self.assertIsNone(classes.get_class_attributes(c2)['x'])

  def test_wrapped_setattr(self):

    class D(type):

      def __setattr__(cls, name, value):
        super().__setattr__(name, value)

      def __delattr__(cls, name):
        super().__delattr__(name)

    class E(object, metaclass=D):
      pass

    e = E()
    try:
      classes.adopt(e)
    except:
      pass

    E.x = 2
    del E.x

  def test_data_descriptor_without_get(self):

    class Descr(object):

      def __set__(self, *args):
        pass

    descr = Descr()

    class C(object):
      pass

    c = C()
    classes.adopt(c)
    c.x = 2
    self.assertEqual(classes.get_class_attributes(c)['x'], 2)
    original_class_id = classes.classid(c)

    # Let's add a data descriptor to C. That shouldn't cause a transition, but
    # the data descriptor should shadow `c.x`.
    C.x = descr
    self.assertEqual(original_class_id, classes.classid(c))
    self.assertEqual(classes.get_class_attributes(c)['x'], descr)

    # Now let's delete the data descriptor. We should get `c.x` back.
    del C.x
    self.assertEqual(original_class_id, classes.classid(c))
    self.assertEqual(classes.get_class_attributes(c)['x'], 2)

  def test_overwritten_getattr(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)

    self.assertTrue(classes.class_is_valid(c))
    C.__getattr__ = lambda *args: 42
    self.assertFalse(classes.class_is_valid(c))

  def test_overwritten_setattr(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)

    self.assertTrue(classes.class_is_valid(c))
    C.__setattr__ = lambda *args: 42
    self.assertFalse(classes.class_is_valid(c))

  def test_overwritten_delattr(self):

    class C(object):
      pass

    c = C()
    classes.adopt(c)

    self.assertTrue(classes.class_is_valid(c))
    C.__delattr__ = lambda *args: 42
    self.assertFalse(classes.class_is_valid(c))

  def test_ndarray(self):
    import numpy

    a = numpy.ndarray([])
    classes.adopt(a)
    self.assertNonzeroClass(a)

    # Test distilled from programs using numpy; ndarray inherits from object
    # but not as the first item in its MRO. The symptom is a TypeError with
    # message "can't apply this __setattr__ to numpy.ndarray object".
    self.assertRaises(AttributeError, lambda: a.__setattr__('x', 2))

  def test_nested_wrapped_setattr(self):

    class C(object):

      def __setattr__(self, name, value):
        pass

    class D(C):

      def xxxx(self, name, value):
        return object.__setattr__(self, name, value)

    d = D()
    try:
      classes.adopt(d)
    except:
      pass
    d.xxxx('x', 3)

  def test_modified_type(self):

    class C(object):

      def x(self):
        return 2

    classes.adopt(C())
    C.y = C()
    c = C()
    self.assertNonzeroClass(c)
    c.a = 2
    self.assertNonzeroClass(c)


if __name__ == '__main__':
  absltest.main()
