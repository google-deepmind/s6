#!/bin/bash

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

# Link S6 components into a directory that can be added to PYTHONPATH.
#
# Usage:
#   setup_python_module.py SRC_DIR BUILD_DIR MODULE_DIR
#
# Where:
#   SRC_DIR is the absolute path to S6 source.
#   BUILD_DIR is the absolute path to an S6 build.
#   MODULE_DIR will be added to the environment variable PYTHONPATH.

readonly SRC_DIR=${1}
readonly BUILD_DIR=${2}
readonly MODULE_DIR=${3}

mkdir -p "${MODULE_DIR}"/s6
mkdir -p "${MODULE_DIR}"/s6/classes/python
mkdir -p "${MODULE_DIR}"/s6/python
mkdir -p "${MODULE_DIR}"/s6/strongjit
mkdir -p "${MODULE_DIR}"/pybind11_abseil
ln -s "${SRC_DIR}"/__init__.py "${MODULE_DIR}"/s6/
ln -s "${BUILD_DIR}"/classes/python/classes.so "${MODULE_DIR}"/s6/classes/python/
ln -s "${BUILD_DIR}"/python/api.so "${MODULE_DIR}"/s6/python/
ln -s "${BUILD_DIR}"/python/type_feedback.so "${MODULE_DIR}"/s6/python/
ln -s "${BUILD_DIR}"/strongjit/dis6.so "${MODULE_DIR}"/s6/strongjit/
ln -s "${BUILD_DIR}"/External/pybind11_abseil/lib/status.so "${MODULE_DIR}"/pybind11_abseil
touch "${MODULE_DIR}"/pybind11_abseil/__init__.py
touch "${MODULE_DIR}"/s6/classes/python/__init__.py
touch "${MODULE_DIR}"/s6/python/__init__.py
touch "${MODULE_DIR}"/s6/strongjit/__init__.py
