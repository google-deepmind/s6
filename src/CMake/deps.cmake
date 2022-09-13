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

# ==============================================================================
# External dependencies
# ==============================================================================

include(FetchContent)

# Python - S6 needs Python 3.7.x
find_package(Python 3.7 EXACT REQUIRED COMPONENTS Interpreter Development)

# ProtoBuf
find_package(Protobuf REQUIRED)

# GoogleTest
fetchcontent_declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    e2239ee6043f73722e7aa812a459f54a28552929
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/googletest
  UPDATE_COMMAND "")
fetchcontent_makeavailable(googletest)

# Abseil
set(ABSL_PROPAGATE_CXX_STD TRUE)
fetchcontent_declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/abseil-cpp
  GIT_TAG 278e0a071885a22dcd2fd1b5576cc44757299343
  UPDATE_COMMAND "")
fetchcontent_makeavailable(abseil)
target_compile_options(absl_strings PUBLIC -Wno-deprecated-copy)

# ASMJit
set(ASMJIT_STATIC TRUE)
fetchcontent_declare(
  asmjit
  GIT_REPOSITORY https://github.com/asmjit/asmjit
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/asmjit
  GIT_TAG ac77dfcd75f043e2fe317133a971040e5b999916
  UPDATE_COMMAND "")
fetchcontent_makeavailable(asmjit)
target_compile_options(
  asmjit PRIVATE -Wno-unused-function -Wno-unused-parameter
                 -Wno-unused-variable)
target_compile_options(
  asmjit PUBLIC -Wno-deprecated-copy -Wno-unused-variable
                -Wno-unused-but-set-variable)
set(asmjit_INCLUDE_DIRS "${asmjit_SOURCE_DIR}/src")

# udis86
include(ExternalProject)
find_program(MAKE_EXE NAMES gmake nmake make)
externalproject_add(
  udis86-external
  GIT_REPOSITORY https://github.com/jbcoe/udis86
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/udis86
  CONFIGURE_COMMAND ./autogen.sh && ./configure --with-python=${Python_EXECUTABLE} --prefix=${CMAKE_CURRENT_BINARY_DIR}/External/udis86-build --disable-shared --enable-static
  BUILD_COMMAND ${MAKE_EXE} CFLAGS='-fPIC'
  INSTALL_COMMAND ${MAKE_EXE} install
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/External/udis86-build/lib/libudis86.a
)
set(udis86_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/External")
add_library(udis86 STATIC IMPORTED GLOBAL)
add_dependencies(udis86 udis86-external)
set_target_properties(udis86 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/External/udis86-build/lib/libudis86.a)

# cppitertools
fetchcontent_declare(
  cppitertools
  GIT_REPOSITORY https://github.com/ryanhaining/cppitertools
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/cppitertools
  GIT_TAG 539a5be8359c4330b3f88ed1821f32bb5c89f5f6  # v2.0.1
  UPDATE_COMMAND "")
fetchcontent_makeavailable(cppitertools)
set(cppitertools_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/External)

# PyBind11
set(PYBIND11_NOPYTHON TRUE)
fetchcontent_declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/pybind11
  GIT_TAG a5ec28ffe8243483dd21e3fbf2e3c8ecb936c4c6
  UPDATE_COMMAND "")
fetchcontent_makeavailable(pybind11)
set(pybind11_INCLUDE_DIRS "${pybind11_SOURCE_DIR}/include")

# pybind11_abseil
fetchcontent_declare(
  pybind11_abseil
  GIT_REPOSITORY https://github.com/pybind/pybind11_abseil
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/pybind11_abseil
  GIT_TAG d9614e4ea46b411d02674305245cba75cd91c1c6
  UPDATE_COMMAND "")
fetchcontent_makeavailable(pybind11_abseil)

add_library(pybind11_abseil SHARED
  ${pybind11_abseil_SOURCE_DIR}/pybind11_abseil/status.cc
  ${pybind11_abseil_SOURCE_DIR}/pybind11_abseil/status_utils.cc
)
target_include_directories(pybind11_abseil PUBLIC
  ${pybind11_abseil_SOURCE_DIR}
  ${pybind11_INCLUDE_DIRS})
target_link_libraries(pybind11_abseil PUBLIC absl::base absl::status Python::Python)
set_target_properties(pybind11_abseil
    PROPERTIES
    OUTPUT_NAME status
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY "${pybind11_abseil_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${pybind11_abseil_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${pybind11_abseil_SOURCE_DIR}/bin"
)

# RE2
fetchcontent_declare(
  re2-external
  GIT_REPOSITORY https://github.com/google/re2
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/re2
  GIT_TAG b15818e
  UPDATE_COMMAND "")
fetchcontent_makeavailable(re2-external)

add_library(re2
  ${re2-external_SOURCE_DIR}/re2/bitmap256.h
  ${re2-external_SOURCE_DIR}/re2/bitstate.cc
  ${re2-external_SOURCE_DIR}/re2/compile.cc
  ${re2-external_SOURCE_DIR}/re2/dfa.cc
  ${re2-external_SOURCE_DIR}/re2/filtered_re2.cc
  ${re2-external_SOURCE_DIR}/re2/mimics_pcre.cc
  ${re2-external_SOURCE_DIR}/re2/nfa.cc
  ${re2-external_SOURCE_DIR}/re2/onepass.cc
  ${re2-external_SOURCE_DIR}/re2/parse.cc
  ${re2-external_SOURCE_DIR}/re2/perl_groups.cc
  ${re2-external_SOURCE_DIR}/re2/pod_array.h
  ${re2-external_SOURCE_DIR}/re2/prefilter.cc
  ${re2-external_SOURCE_DIR}/re2/prefilter.h
  ${re2-external_SOURCE_DIR}/re2/prefilter_tree.cc
  ${re2-external_SOURCE_DIR}/re2/prefilter_tree.h
  ${re2-external_SOURCE_DIR}/re2/prog.cc
  ${re2-external_SOURCE_DIR}/re2/prog.h
  ${re2-external_SOURCE_DIR}/re2/re2.cc
  ${re2-external_SOURCE_DIR}/re2/regexp.cc
  ${re2-external_SOURCE_DIR}/re2/regexp.h
  ${re2-external_SOURCE_DIR}/re2/set.cc
  ${re2-external_SOURCE_DIR}/re2/simplify.cc
  ${re2-external_SOURCE_DIR}/re2/sparse_array.h
  ${re2-external_SOURCE_DIR}/re2/sparse_set.h
  ${re2-external_SOURCE_DIR}/re2/tostring.cc
  ${re2-external_SOURCE_DIR}/re2/unicode_casefold.cc
  ${re2-external_SOURCE_DIR}/re2/unicode_casefold.h
  ${re2-external_SOURCE_DIR}/re2/unicode_groups.cc
  ${re2-external_SOURCE_DIR}/re2/unicode_groups.h
  ${re2-external_SOURCE_DIR}/re2/walker-inl.h
  ${re2-external_SOURCE_DIR}/util/logging.h
  ${re2-external_SOURCE_DIR}/util/rune.cc
  ${re2-external_SOURCE_DIR}/util/strutil.cc
  ${re2-external_SOURCE_DIR}/util/strutil.h
  ${re2-external_SOURCE_DIR}/util/utf.h)
target_include_directories(re2 PUBLIC ${re2-external_SOURCE_DIR})
target_link_libraries(re2 absl::base)
set_target_properties(re2
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${re2-external_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${re2-external_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${re2-external_SOURCE_DIR}/bin"
)
