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

# Build S6 using cmake.
#
# Requires clang and Python 3.7.x
#
# Later Python versions are not compatible with S6.
#
# Usage:
#   build.sh SOURCE_DIR BUILD_DIR EXTRA ARGS
#
# Where:
#   SRC_DIR is the absolute path to S6 source.
#   BUILD_DIR is the absolute path to an S6 build.
#
# Remaining args passed to this script are forwarded to the initial cmake
# invocation.
#
# This script will probably need to be passed -DPython_ROOT_DIR=PYTHON3_7/ROOT


set -eux
set -o pipefail

SOURCE_DIR="${1}"
shift
BUILD_DIR="${1}"
shift

CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -S"${SOURCE_DIR}" -B"${BUILD_DIR}" "${@}"
cmake --build "${BUILD_DIR}"
