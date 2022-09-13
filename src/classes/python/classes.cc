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
#include "classes/object.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/status_casters.h"
#include "utils/status_macros.h"
namespace pybind11 {
using ::deepmind::s6::Adopt;
using ::deepmind::s6::AdoptExistingTypes;
using ::deepmind::s6::AdoptNewTypes;
using ::deepmind::s6::Class;
using ::deepmind::s6::ClassManager;
using ::deepmind::s6::GetClass;
using ::deepmind::s6::GetClassId;
using ::deepmind::s6::GetTypeAttributesAsMap;
using ::deepmind::s6::IsAdoptable;

dict AttributeMapToDict(const Class::AttributeMap& map) {
  dict d;
  for (const auto& [name, attr] : map) {
    handle h = Py_None;
    if (attr->value()) h = attr->value();
    d[str(std::string(name))] = h;
  }
  return d;
}

PYBIND11_MODULE(classes, py_module) {
  pybind11::google::ImportStatusModule();
  py_module.doc() = "Defines internal S6 functions related to classes";

  py_module.def(
      "classid", [](object obj) { return GetClassId(obj.ptr()); },
      "Returns the class ID of an object.");

  py_module.def(
      "class_is_valid",
      [](object obj) {
        Class* cls = GetClass(obj.ptr());
        return cls != nullptr && !GetClass(obj.ptr())->invalid();
      },
      "Returns True if this object has a class and the class is valid.");

  py_module.def(
      "adoptable", [](object obj) { return IsAdoptable(obj.ptr()).ok(); },
      "Returns True if the given object is Adoptable.");

  py_module.def(
      "adopt", [](object obj) { return Adopt(obj.ptr()); },
      "Adopts the given object.");

  py_module.def(
      "adopt_new_types", []() { AdoptNewTypes(); },
      "Attempts to adopt all newly created types.");

  py_module.def(
      "stop_adopting_new_types",
      []() { ::deepmind::s6::StopAdoptingNewTypes(); },
      "Undoes adopt_new_types().");

  py_module.def(
      "get_type_attributes",
      [](object obj) -> absl::StatusOr<dict> {
        S6_ASSIGN_OR_RETURN(
            Class::AttributeMap result,
            GetTypeAttributesAsMap(ClassManager::Instance(),
                                   reinterpret_cast<PyTypeObject*>(obj.ptr())));
        return AttributeMapToDict(result);
      },
      "Reads all attributes from the MRO of an object and returns them as a "
      "dict.");

  py_module.def(
      "get_class_attributes",
      [](object obj) -> absl::StatusOr<dict> {
        Class* cls = GetClass(obj.ptr());
        S6_RET_CHECK(cls);
        return AttributeMapToDict(cls->attributes());
      },
      "Reads all attributes from the MRO of an object and returns them as a "
      "dict.");

  py_module.def(
      "adopt_existing_types", []() { AdoptExistingTypes(); },
      "Attempts to adopt all existing subclasses of Type, transitively.");
}

}  // namespace pybind11
