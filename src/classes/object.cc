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

#include "classes/object.h"

#include <cstdint>

#include "classes/class_manager.h"
#include "classes/getsetattr.h"
#include "classes/util.h"
#include "event_counters.h"
#include "utils/status_macros.h"

namespace deepmind::s6 {
namespace {
// The original value of PyType_Type.tp_new.
static const auto kTypeNew = PyType_Type.tp_new;

// The original value of PyType_Type.tp_setattro.
static const auto kTypeSetAttro = PyType_Type.tp_setattro;

// Forward declare.
int TypeSetAttro(PyObject* obj, PyObject* name, PyObject* value);

bool TypeHasDeletedAttributeGetter(PyTypeObject* type) {
  return type->tp_getattro == nullptr && type->tp_getattr == nullptr;
}

bool TypeHasDeletedAttributeSetter(PyTypeObject* type) {
  return type->tp_setattro == nullptr && type->tp_setattr == nullptr;
}

bool TypeHasDefaultAttributeGetter(PyTypeObject* type) {
  return type->tp_getattro == PyObject_GenericGetAttr &&
         type->tp_getattr == nullptr;
}

bool TypeHasDefaultAttributeSetter(PyTypeObject* type) {
  return (type->tp_setattro == PyObject_GenericSetAttr ||
          type->tp_setattro == GenericSetAttrUsingClasses) &&
         type->tp_setattr == nullptr;
}

bool MetatypeHasDefaultAttributeSetter(PyTypeObject* type) {
  return (type->tp_setattro == kTypeSetAttro ||
          type->tp_setattro == TypeSetAttro) &&
         type->tp_setattr == nullptr;
}

template <typename F>
void ForAllSubtypes(PyTypeObject* type, F&& f) {
  f(type);
  PyObject* dict = type->tp_subclasses;
  if (!dict || !PyDict_Check(dict)) return;

  Py_ssize_t i = 0;
  PyObject* value;
  while (PyDict_Next(dict, &i, nullptr, &value)) {
    S6_CHECK(PyWeakref_CheckRef(value));
    PyObject* obj = PyWeakref_GET_OBJECT(value);
    if (!obj) continue;
    S6_CHECK(PyType_Check(obj));
    PyTypeObject* subtype = reinterpret_cast<PyTypeObject*>(obj);
    ForAllSubtypes(subtype, f);
  }
}

int TypeSetAttro(PyObject* obj, PyObject* name, PyObject* value) {
  int ret = kTypeSetAttro(obj, name, value);
  if (!PyType_Check(obj))
    // Safety first. This shouldn't be possible but be defensive.
    return ret;

  PyTypeObject* type = reinterpret_cast<PyTypeObject*>(obj);
  ForAllSubtypes(type, [](PyTypeObject* type) {
    int64_t class_id = GetClassIdFromType(type);
    if (class_id == 0) return;
    Class* cls = ClassManager::Instance().GetClassById(class_id);
    if (cls) {
      ClassManager::Instance().SafelyRunTypeHasBeenModified(
          [](Class* cls, PyTypeObject* type) {
            cls->UnderlyingTypeHasBeenModified(type);
          },
          cls, type);
    }
  });
  return ret;
}

// This is a helper for WrapSetAttr and WrapDelAttr that mirrors `hackcheck`
// from typeobject.c. This implements the "Carlo Verre" hack. This
// implementation is less strict than the original version in `typeobject.c`,
// because we allow any of a set of setattro functions.
bool HackCheck(PyObject* self, setattrofunc func, const char* what) {
  PyTypeObject* type = Py_TYPE(self);
  while (type && type->tp_flags & Py_TPFLAGS_HEAPTYPE) type = type->tp_base;

  if (type && type->tp_setattro != func) {
    // Don't worry about mismatches between the identity of these functions.
    // They behave identically, and mismatch may just mean we couldn't adopt a
    // particular type in the hierarchy but could adopt a supertype.
    if (func == PyObject_GenericSetAttr &&
        type->tp_setattro == GenericSetAttrUsingClasses)
      return true;
    if (type->tp_setattro == PyObject_GenericSetAttr &&
        func == GenericSetAttrUsingClasses)
      return true;
    PyErr_Format(PyExc_TypeError, "can't apply this %s to %s object", what,
                 type->tp_name);
    return false;
  }
  return true;
}

// This is `wrap_setattr` from `typeobject.c`, but without `hackcheck`. The
// hackcheck function attempts to validate that the wrapped function is equal
// exactly to `type(self)->tp_setattro`, which is not correct for S6 adopted
// types.
//
// Instead of simply replacing the `wrapped` parameter with our own setattro
// so the check succeeds, we eliminate the check by overwriting the wrapper
// function.
PyObject* WrapSetAttr(PyObject* self, PyObject* args, void* wrapped) {
  setattrofunc func = reinterpret_cast<setattrofunc>(wrapped);
  int res;
  PyObject *name, *value;

  if (!PyArg_UnpackTuple(args, "", 2, 2, &name, &value)) {
    return nullptr;
  }
  if (!HackCheck(self, func, "__setattr__")) {
    return nullptr;
  }
  res = (*func)(self, name, value);
  if (res < 0) {
    return nullptr;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// See above description.
PyObject* WrapDelAttr(PyObject* self, PyObject* args, void* wrapped) {
  setattrofunc func = reinterpret_cast<setattrofunc>(wrapped);
  int res;

  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_SystemError,
                    "PyArg_UnpackTuple() argument list is not a tuple");
    return nullptr;
  }
  if (PyTuple_GET_SIZE(args) != 1) {
    PyErr_Format(PyExc_TypeError, "expected 1 arguments, got %zd",
                 PyTuple_GET_SIZE(args));
    return nullptr;
  }
  if (!HackCheck(self, func, "__delattr__")) {
    return nullptr;
  }

  res = (*func)(self, PyTuple_GET_ITEM(args, 0), nullptr);
  if (res < 0) {
    return nullptr;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// Sets the wrapper function for the given attribute, which is expected to be
// one of "__setattr__" or "__delattr__".
//
// If we find a descriptor pointing to `expected_target`, we replace the
// d_wrapped member with `new_target` and also change the wrapper function
// to `wrapper_function`. We only need to change the wrapper function in
// order to weaken a particular check, `hackcheck()` in typeobject.c.
void SetWrapper(PyTypeObject* type, setattrofunc expected_target,
                setattrofunc new_target, absl::string_view name,
                wrapperfunc wrapper_function) {
  PyObject* str = PyUnicode_FromString(name.data());
  PyObject* descriptor = PyDict_GetItem(type->tp_dict, str);
  Py_DECREF(str);

  if (descriptor && Py_TYPE(descriptor) == &PyWrapperDescr_Type) {
    PyWrapperDescrObject* wrapper =
        reinterpret_cast<PyWrapperDescrObject*>(descriptor);
    if (wrapper->d_wrapped == expected_target) {
      wrapper->d_wrapped = reinterpret_cast<void*>(new_target);
      wrapper->d_base->wrapper = wrapper_function;
    }
  }
}

// Sets type->tp_setattro to `new_target`. Also inspects type.__dict__ for
// '__setattr__' and '__delattr__'. If we find a PyWrapperDescr, modify the
// wrapped target to `new_target`.
void SetTpSetAttrSlot(PyTypeObject* type, setattrofunc expected_target,
                      setattrofunc new_target) {
  if (type->tp_setattro == expected_target) type->tp_setattro = new_target;
  // As well as setting tp_setattro, update the type's __dict__. Type
  // inserts a PyWrapperDescrObject in there that forwards to type_setattro.
  // This is found by either querying type.__dict__['__setattr__'] or
  // via super().__setattr__ on a type subclass.

  // Find all __setattr__, following the MRO.
  for (int64_t i = 0; i < PyTuple_GET_SIZE(type->tp_mro); ++i) {
    PyTypeObject* mro_type =
        reinterpret_cast<PyTypeObject*>(PyTuple_GET_ITEM(type->tp_mro, i));
    SetWrapper(mro_type, expected_target, new_target, "__setattr__",
               WrapSetAttr);
  }
  // Find all __delattr__, following the MRO.
  for (int64_t i = 0; i < PyTuple_GET_SIZE(type->tp_mro); ++i) {
    PyTypeObject* mro_type =
        reinterpret_cast<PyTypeObject*>(PyTuple_GET_ITEM(type->tp_mro, i));
    SetWrapper(mro_type, expected_target, new_target, "__delattr__",
               WrapDelAttr);
  }
}

}  // namespace

// Obtains the Class of a PyObject. Returns nullptr on failure.
Class* GetClass(PyObject* object) {
  int64_t class_id = GetClassId(object);
  if (class_id == 0) return nullptr;
  return ClassManager::Instance().GetClassById(class_id);
}

absl::Status IsAdoptableType(PyTypeObject* type) {
  // We can adopt objects whose types have disallowed getattr/setattr, but not
  // those that have a custom getter or setter.
  if (!TypeHasDeletedAttributeGetter(type) &&
      !TypeHasDefaultAttributeGetter(type))
    return absl::FailedPreconditionError("type has non-default __getattr__");
  if (!TypeHasDeletedAttributeSetter(type) &&
      !TypeHasDefaultAttributeSetter(type))
    return absl::FailedPreconditionError("type has non-default __setattr__");

  if (type->tp_dictoffset < 0) {
    // We don't support obtaining dictionaries from the end of objects.
    return absl::FailedPreconditionError("type has negative tp_dictoffset");
  }

  // We need to be able to observe changes to `type`, as well as instances of
  // `type`.
  PyTypeObject* metatype = Py_TYPE(type);
  if (!TypeHasDeletedAttributeSetter(metatype) &&
      !MetatypeHasDefaultAttributeSetter(metatype))
    return absl::FailedPreconditionError(
        "metatype has non-default __setattr__");

  return absl::OkStatus();
}

absl::Status IsAdoptable(PyObject* object) {
  PyTypeObject* type = Py_TYPE(object);
  S6_RETURN_IF_ERROR(IsAdoptableType(type));
  if (type->tp_dictoffset == 0) return absl::OkStatus();

  PyDictObject* dict = GetObjectDict(object);
  if (!dict || PyDict_Size(reinterpret_cast<PyObject*>(dict)) == 0)
    return absl::OkStatus();

  // Dict has existing attributes. We can't (yet!) work out the correct class.
  return absl::FailedPreconditionError("Object has non-empty __dict__");
}

absl::Status AdoptType(PyTypeObject* type) {
  S6_VLOG(1) << "Adopting type " << type->tp_name;
  S6_RETURN_IF_ERROR(IsAdoptableType(type));

  // If the type has not been adopted before, we create a new class.
  int64_t class_id = 0;
  if (GetClassIdFromType(type) == 0) {
    S6_ASSIGN_OR_RETURN(Class * cls, Class::Create(ClassManager::Instance(),
                                                   type->tp_name, type));
    class_id = cls->id();
    SetClassIdOnType(type, class_id);

    // Note that we never hook tp_setattro to find changes to the __dict__. This
    // will fall off the fast path and invalidate the class.
    S6_ASSIGN_OR_RETURN(
        Class * type_cls,
        Class::CreateType(ClassManager::Instance(), type->tp_name, type));
    SetClassId(reinterpret_cast<PyDictObject*>(type->tp_dict), type_cls->id());

    if (type->tp_setattro && type->tp_setattro != GenericSetAttrUsingClasses) {
      S6_CHECK_EQ(type->tp_setattro, PyObject_GenericSetAttr);
      SetTpSetAttrSlot(type, PyObject_GenericSetAttr,
                       GenericSetAttrUsingClasses);
    }

  } else {
    // We've already inspected this type, so the class ID is already set.
    class_id = GetClassIdFromType(type);
    S6_RET_CHECK(class_id != 0);
  }
  // class_id should now be valid.
  S6_CHECK_GT(class_id, 0);

  // Observe changes to the type by hooking the metatype. IsAdoptableType has
  // already checked that we can do this.
  PyTypeObject* metatype = Py_TYPE(type);
  if (metatype->tp_setattro != TypeSetAttro) {
    SetTpSetAttrSlot(metatype, kTypeSetAttro, TypeSetAttro);
  }

  return absl::OkStatus();
}

absl::Status Adopt(PyObject* object) {
  S6_RETURN_IF_ERROR(IsAdoptable(object));
  PyTypeObject* type = Py_TYPE(object);
  S6_RETURN_IF_ERROR(AdoptType(type));

  if (PyDictObject* dict = GetObjectDict(object)) {
    SetClassId(dict, GetClassIdFromType(type));
  }
  return absl::OkStatus();
}

void AdoptNewTypes() {
  PyType_Type.tp_new =
      +[](PyTypeObject* type, PyObject* args, PyObject* kwargs) {
        PyObject* obj = kTypeNew(type, args, kwargs);

        if (obj && PyType_Check(obj) &&
            PyType_Ready(reinterpret_cast<PyTypeObject*>(obj)) == 0) {
          AdoptType(reinterpret_cast<PyTypeObject*>(obj)).IgnoreError();
        }
        return obj;
      };
}

void StopAdoptingNewTypes() { PyType_Type.tp_new = kTypeNew; }

void AdoptExistingTypes() {
  ForAllSubtypes(&PyType_Type,
                 [](PyTypeObject* type) { AdoptType(type).IgnoreError(); });
  ForAllSubtypes(&PyBaseObject_Type,
                 [](PyTypeObject* type) { AdoptType(type).IgnoreError(); });
}

void AdoptGlobalsDict(PyDictObject* globals) {
  ClassManager& mgr = ClassManager::Instance();
  EventCounters& counters = EventCounters::Instance();
  int64_t class_id = GetClassIdFromObjectDict(globals);
  if (class_id != 0) {
    return;
  }

  if (Class* cls = mgr.GetClassForGlobals(globals)) {
    if (cls->invalid()) {
      // This globals dict has an assigned class but has been modified. Treat it
      // as invalidated.
      counters.Add("classes.globals.invalid_on_adoption", 1);
      return;
    }
    counters.Add("globals.readopted", 1);
    cls->ReinitializeGlobals(globals);
    if (!cls->invalid()) {
      SetClassId(globals, cls->id());
    }
    return;
  }

  auto cls_or = Class::CreateGlobals(mgr, globals);
  if (cls_or.ok()) {
    counters.Add("globals.adopted", 1);
    SetClassId(globals, cls_or.value()->id());
  } else {
    counters.Add("globals.unadoptable", 1);
  }
}

int64_t GetGlobalsClassId(PyDictObject* globals) {
  return GetClassIdFromObjectDict(globals);
}

}  // namespace deepmind::s6
