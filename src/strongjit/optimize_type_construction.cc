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

#include "strongjit/optimize_type_construction.h"

#include <iterator>

#include "absl/status/status.h"
#include "core_util.h"
#include "strongjit/callees.h"
#include "strongjit/instructions.h"
#include "strongjit/optimizer.h"
#include "utils/no_destructor.h"

namespace deepmind::s6 {
namespace {
// Looks up the given attribute on the given type object's tp_dict, which must
// be known to be immutable.
PyObject* LookupAttributeOnImmutableType(PyTypeObject* type,
                                         absl::string_view str) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(type->tp_dict, &pos, &key, &value)) {
    if (!PyUnicode_CheckExact(key)) {
      continue;
    }
    if (GetObjectAsCheapString(key) == str) {
      return value;
    }
  }
  return nullptr;
}
}  // namespace

absl::Status OptimizeTypeConstructionPattern::Apply(
    CallPythonInst* call_python, Rewriter& rewriter,
    ConstantAttributeInst* constant_attribute, SafepointInst* safepoint) {
  Class* cls = constant_attribute->LookupAttribute(ClassManager::Instance())
                   .value_class();
  if (!cls || !cls->is_type_class()) {
    return absl::FailedPreconditionError(
        "not a type class; cannot be a type constructor");
  }

  // Lookup the __call__ dunder. If it's the same as PyType's __call__, this is
  // a type constructor.
  ClassManager& mgr = rewriter.class_manager();
  auto it = cls->attributes().find(mgr.InternString("__call__"));
  if (it == cls->attributes().end()) {
    return absl::FailedPreconditionError("did not have __call__");
  }

  static PyObject* type_call =
      LookupAttributeOnImmutableType(&PyType_Type, "__call__");
  S6_CHECK(type_call) << "Couldn't find __call__ on PyType_Type?";

  if (it->second->value() != type_call) {
    return absl::FailedPreconditionError("Call was not type_call!");
  }

  // Okay, this is a type constructor! Look up the __new__ and __init__
  // attributes.
  it = cls->attributes().find(mgr.InternString("__new__"));
  if (it == cls->attributes().end()) {
    return absl::FailedPreconditionError("Type did not define __new__!");
  }
  Attribute& new_attr = *it->second;

  it = cls->attributes().find(mgr.InternString("__init__"));
  if (it == cls->attributes().end()) {
    return absl::FailedPreconditionError("Type did not define __init__!");
  }
  // This optimization is only going to be a major win if __init__ is a
  // function. This optimization is able to elide method construction (prepend
  // binding) in this case.
  if (it->second->kind() != Attribute::kFunction) {
    return absl::FailedPreconditionError("__init__ was not a function!");
  }
  FunctionAttribute& init_attr = static_cast<FunctionAttribute&>(*it->second);

  // The protocol for type construction is (simplified):
  //   obj = __new__(type, ...)
  //   if (PyType_IsSubType(obj, type)):
  //     type(obj).__init__(obj, ...)
  //
  // Note that __init__ may not be called at all, or a subclass's __init__ may
  // be called.
  //
  // However, most object construction either uses the default object.__new__,
  // which we know will always return an object of the exact type it is given,
  // or a custom __new__ that frequently shims out to object.__new__ but does
  // extra work.
  //
  // Therefore we optimize specially if we know __new__ is exactly
  // object.__new__, because we know at compile time that we can elide the
  // subtype check and call __init__.
  static PyObject* object_new =
      LookupAttributeOnImmutableType(&PyBaseObject_Type, "__new__");
  S6_CHECK(object_new) << "Couldn't find __new__ on PyBaseObject_Type?";

  Builder builder = rewriter.CreateBuilder(call_python->GetIterator());
  if (new_attr.value() == object_new) {
    rewriter.AddReliedUponClass(cls);

    static NoDestructor<CalleeInfo> tp_alloc_info(CalleeInfo(
        /*return_new_ref=*/true, {CalleeInfo::ArgInfo(Nullness::kMaybeNull),
                                  CalleeInfo::ArgInfo(Nullness::kMaybeNull)}));

    // object.__new__ just forwards to type->tp_alloc. We emit the call to
    // tp_alloc here to avoid constructing a pointless tuple for args.
    // TODO: Check this isn't an abstract type.
    Value* type = builder.Constant(cls->type_class_type());
    Value* tp_alloc = builder.Load64(type, offsetof(PyTypeObject, tp_alloc));
    Value* object =
        builder.CallIndirect(tp_alloc, {type, builder.Int64(0)}, *tp_alloc_info,
                             call_python->bytecode_offset());
    // We know that object.__new__ is sideeffect free, so we can restart this
    // bytecode.
    auto* deopt = builder.DeoptimizeIfSafepoint(
        object, /*negated=*/true, "__new__ raised an exception", safepoint);
    // The refcount analysis complains if we do not free object in the
    // safepoint.
    deopt->decref_value(object);

    // CallPython originally stole this but it is now unused.
    // This must be after the safepoint because the safepoint added above
    // targets a safepoint before the original call_python instruction.
    builder.DecrefNotNull(constant_attribute);

    Value* result =
        CallInitDunder(builder, *cls, init_attr, *call_python, object);

    Value* object_or_null = builder
                                .Conditional(
                                    result,
                                    [&](Builder b) {
                                      b.DecrefNotNull(result);
                                      return Builder::ValueList{object};
                                    },
                                    [&](Builder b) {
                                      b.DecrefNotNull(object);
                                      return Builder::ValueList{result};
                                    })
                                .front();
    rewriter.ReplaceAllUsesWith(*call_python, *object_or_null);
    rewriter.erase(*call_python);

    EventCounters::Instance().Add("optimizer.type_construction.elided_new", 1);
    return absl::OkStatus();
  }

  // Otherwise we must call __new__. Let's ensure it's a Function though.
  if (new_attr.kind() != Attribute::kFunction) {
    return absl::FailedPreconditionError(
        "custom __new__ that wasn't a PyFunction.");
  }
  rewriter.AddReliedUponClass(cls);

  // The original (before optimisation) call_python stole this but it is now
  // unused since we call __new__ and __init__ directly.
  builder.DecrefNotNull(constant_attribute);

  Value* object = CallNewDunder(
      builder, *cls, static_cast<FunctionAttribute&>(new_attr), *call_python);
  Value* call_init = builder.ShortcircuitAnd(
      [&](Builder builder) { return builder.IsNotZero(object); },
      [&](Builder builder) {
        return builder.Call(
            Callee::kPyType_IsSubtype,
            {builder.GetType(object), builder.Constant(cls->type_class_type())},
            call_python->bytecode_offset());
      });
  Value* result =
      builder
          .Conditional(
              call_init,
              [&](Builder builder) {
                Value* result = CallInitDunder(builder, *cls, init_attr,
                                               *call_python, object);
                Value* object_or_null =
                    builder
                        .Conditional(
                            result,
                            [&](Builder b) {
                              b.DecrefNotNull(result);
                              return Builder::ValueList{object};
                            },
                            [&](Builder b) {
                              b.DecrefNotNull(object);
                              return Builder::ValueList{result};
                            })
                        .front();
                return Builder::ValueList{object_or_null};
              },
              [&](Builder builder) {
                // The CallPython we are replacing is expected to decref
                // arguments. __init__ above did that, so we have to here as we
                // didn't call __init__.
                builder.DecrefNotNull(call_python->call_arguments(),
                                      call_python->bytecode_offset());
                return Builder::ValueList{object};
              })
          .front();

  rewriter.ReplaceAllUsesWith(*call_python, *result);
  rewriter.erase(*call_python);

  EventCounters::Instance().Add("optimizer.type_construction.nonelided_new", 1);
  return absl::OkStatus();
}

Value* OptimizeTypeConstructionPattern::CallInitDunder(
    Builder& builder, const Class& cls, const FunctionAttribute& attr,
    CallPythonInst& call_python, Value* object) {
  // Prepend object as self.
  std::vector<Value*> arguments(1, object);
  absl::c_copy(call_python.call_arguments(), std::back_inserter(arguments));

  // CallPython steals references. All arguments except self are already
  // accounted for as this is a CallPythonInst we're calling, so just incref
  // self.
  builder.IncrefNotNull(object);
  Value* result = builder.CallConstantAttribute(&cls, attr.name(), arguments,
                                                call_python.names(),
                                                call_python.bytecode_offset());
  return result;
}

Value* OptimizeTypeConstructionPattern::CallNewDunder(
    Builder& builder, const Class& cls, const FunctionAttribute& attr,
    CallPythonInst& call_python) {
  S6_CHECK(cls.type_class_type());
  Value* type = builder.Constant(cls.type_class_type());
  std::vector<Value*> arguments(1, type);
  absl::c_copy(call_python.call_arguments(), std::back_inserter(arguments));

  // CallPython steals references. We're going to call __init__ with these
  // later, so incref them here.
  for (Value* v : arguments) {
    builder.IncrefNotNull(v);
  }
  Value* result = builder.CallConstantAttribute(&cls, attr.name(), arguments,
                                                call_python.names(),
                                                call_python.bytecode_offset());
  return result;
}

}  // namespace deepmind::s6
