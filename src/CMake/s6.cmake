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
# Macros to make defining S6 build targets more concise.
# ==============================================================================

# Lazily link some system libraries into all s6 targets.
set(S6_LINK_LIBRARIES dl util pthread)

# ==============================================================================
# s6_cc_library: Allow a C++ library to be defined more concisely.
# The library can be header-only (if SRCS is empty).
#
# s6_cc_library(
#   NAME library-name, string, required
#   SRCS list-of-sources, list of files
#   HDRS list-of-headers, list of files
#   DEPS list-of-libraries-linked, list of targets
#   INCLUDE_DIRS list of additional include dirs to use when compiling)
#   LINKOPTS list-of-strings, Flags list-of-strings, passed to the linker
# ==============================================================================
function(s6_cc_library)
  cmake_parse_arguments(S6_CC_LIB "" "NAME" "HDRS;SRCS;DEPS;INCLUDE_DIRS;LINKOPTS" ${ARGN})
  set(_NAME ${S6_CC_LIB_NAME})
  set(_SRCS ${S6_CC_LIB_SRCS})
  set(_HDRS ${S6_CC_LIB_HDRS})
  set(_DEPS ${S6_CC_LIB_DEPS})
  set(_INCLUDE_DIRS ${S6_CC_LIB_INCLUDE_DIRS})
  set(_LINKOPTS ${S6_CC_LIB_LINKOPTS})

  if(_SRCS)
    add_library(${_NAME} "")
    target_sources(${_NAME} PRIVATE ${_SRCS})
    target_sources(${_NAME} PUBLIC ${_HDRS})
    target_include_directories(${_NAME} BEFORE PUBLIC ${PROJECT_SOURCE_DIR}
                                                      ${PROJECT_BINARY_DIR}
                                                      ${_INCLUDE_DIRS})
    if(_DEPS)
      target_link_libraries(${_NAME} PUBLIC ${_DEPS} ${S6_LINK_LIBRARIES})
    endif()
    target_link_options(${_NAME} PUBLIC ${_LINKOPTS})
  else()
    add_library(${_NAME} INTERFACE)
    target_sources(${_NAME} INTERFACE ${_HDRS})
    target_include_directories(${_NAME} BEFORE INTERFACE ${PROJECT_SOURCE_DIR}
                                                         ${PROJECT_BINARY_DIR}
                                                         ${_INCLUDE_DIRS})
    if(_DEPS)
      target_link_libraries(${_NAME} INTERFACE ${_DEPS} ${S6_LINK_LIBRARIES})
    endif()
    target_link_options(${_NAME} INTERFACE ${_LINKOPTS})
  endif()
endfunction()

# ==============================================================================
# s6_cc_binary: Allow an executable to be defined more concisely.
#
# s6_cc_binary(
#   NAME executable-name, string, required
#   SRCS list-of-sources, list of files
#   DEPS list-of-libraries-linked, list of targets
#   INCLUDE_DIRS list of additional include dirs to use when compiling)
#   LINKOPTS list-of-strings, Flags list-of-strings, passed to the linker
#
# ==============================================================================
function(s6_cc_binary)
  cmake_parse_arguments(S6_CC_BINARY "" "NAME" "HDRS;SRCS;DEPS;INCLUDE_DIRS;LINKOPTS" ${ARGN})
  set(_NAME ${S6_CC_BINARY_NAME})
  set(_SRCS ${S6_CC_BINARY_SRCS})
  set(_DEPS ${S6_CC_BINARY_DEPS})
  set(_INCLUDE_DIRS ${S6_CC_BINARY_INCLUDE_DIRS})
  set(_LINKOPTS ${S6_CC_BINARY_LINKOPTS})
  add_executable(${_NAME} ${_SRCS})
  target_include_directories(${_NAME} BEFORE PRIVATE ${PROJECT_SOURCE_DIR}
                                                     ${PROJECT_BINARY_DIR}
                                                     ${_INCLUDE_DIRS})
  target_link_libraries(${_NAME} PRIVATE ${_DEPS}
                                         ${S6_LINK_LIBRARIES})
  target_link_options(${_NAME} PRIVATE ${_LINKOPTS})
endfunction()

# ==============================================================================
# s6_cc_test: Allow a C++ test using Google Test to be defined more concisely.
#
# s6_cc_test(
#   NAME test-name, string, required
#   SRCS list-of-sources, list of files
#   DEPS list-of-libraries-linked, list of targets
#   INCLUDE_DIRS list of additional include dirs to use when compiling)
#   LINKOPTS list-of-strings, Flags list-of-strings, passed to the linker
#
# ==============================================================================
function(s6_cc_test)
  cmake_parse_arguments(S6_CC_TEST "" "NAME" "HDRS;SRCS;DEPS;INCLUDE_DIRS;LINKOPTS" ${ARGN})
  set(_NAME ${S6_CC_TEST_NAME})
  set(_SRCS ${S6_CC_TEST_SRCS})
  set(_DEPS ${S6_CC_TEST_DEPS})
  set(_INCLUDE_DIRS ${S6_CC_TEST_INCLUDE_DIRS})
  set(_LINKOPTS ${S6_CC_TEST_LINKOPTS})
  add_executable(${_NAME} ${_SRCS})
  target_include_directories(${_NAME} BEFORE PRIVATE ${PROJECT_SOURCE_DIR}
                                                     ${PROJECT_BINARY_DIR}
                                                     ${_INCLUDE_DIRS})
  target_link_libraries(${_NAME} PRIVATE ${_DEPS}
                                         ${S6_LINK_LIBRARIES})
  target_link_options(${_NAME} PRIVATE ${_LINKOPTS})
  add_test(NAME ${_NAME} COMMAND ${_NAME})
endfunction()

# ==============================================================================
# s6_py_test: Allow a Python test to be defined.
#
# s6_py_test(
#   NAME test-name, string, required
#   SRCS list-of-sources, list of files)
#
# ==============================================================================
function(s6_py_test)
  cmake_parse_arguments(S6_PY_TEST "" "NAME" "SRCS" ${ARGN})
  set(_NAME ${S6_PY_TEST_NAME})
  set(_SRCS ${S6_PY_TEST_SRCS})
  set(FULL_SRC_PATHS "")
  foreach(SRC ${_SRCS})
      list(APPEND FULL_SRC_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
  endforeach()
  add_test(NAME ${_NAME} COMMAND Python::Interpreter ${FULL_SRC_PATHS})
endfunction()

# ==============================================================================
# s6_cc_proto_library: Allow a C++ library to be defined from proto source more
# concisely.
#
# Build proto libraries as shared libraries to prevent repeated registration of
# types.
#
# s6_cc_proto_library(
#   NAME library-name, string, required
#   SRCS list-of-sources, list of files)
#
# ==============================================================================
function(s6_cc_proto_library)
  cmake_parse_arguments(S6_CC_PROTO_LIBRARY "" "NAME" "SRCS" ${ARGN})
  set(_NAME ${S6_CC_PROTO_LIBRARY_NAME})
  set(_SRCS ${S6_CC_PROTO_LIBRARY_SRCS})

  protobuf_generate_cpp(_GEN_SRCS _GEN_HDRS ${_SRCS})

  add_library(${_NAME} SHARED ${_GEN_SRCS})
  target_include_directories(${_NAME} BEFORE PUBLIC ${PROJECT_SOURCE_DIR}
                                                    ${PROJECT_BINARY_DIR}
                                                    ${CMAKE_CURRENT_BINARY_DIR}
                                                    ${_INCLUDE_DIRS})
  target_link_libraries(${_NAME} PUBLIC ${Protobuf_LIBRARIES})
endfunction()

# ==============================================================================
# s6_pybind_extension: Allow a Python extension to be defined more concisely.
#
# s6_pybind_extension(
#   NAME target-name, string, required
#   OUTPUT_NAME output-name, string, required
#   SRCS list-of-sources, list of files
#   DEPS list-of-libraries-linked, list of targets
#   INCLUDE_DIRS list of additional include dirs to use when compiling)
#   LINKOPTS list-of-strings, Flags list-of-strings, passed to the linker
#
# ==============================================================================
function(s6_pybind_extension)
  cmake_parse_arguments(S6_PYBIND_EXTENSION "" "NAME;OUTPUT_NAME" "SRCS;DEPS;INCLUDE_DIRS;LINKOPTS" ${ARGN})
  set(_NAME ${S6_PYBIND_EXTENSION_NAME})
  set(_OUTPUT_NAME ${S6_PYBIND_EXTENSION_OUTPUT_NAME})
  set(_SRCS ${S6_PYBIND_EXTENSION_SRCS})
  set(_DEPS ${S6_PYBIND_EXTENSION_DEPS})
  set(_INCLUDE_DIRS ${S6_PYBIND_EXTENSION_INCLUDE_DIRS})
  set(_LINKOPTS ${S6_PYBIND_EXTENSION_LINKOPTS})
  add_library(${_NAME} SHARED "")
  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME ${_OUTPUT_NAME})
  set_target_properties(${_NAME} PROPERTIES PREFIX "")
  target_sources(${_NAME} PRIVATE ${_SRCS})
  target_sources(${_NAME} PUBLIC ${_HDRS})
  target_include_directories(${_NAME} BEFORE PUBLIC ${PROJECT_SOURCE_DIR}
                                                    ${PROJECT_BINARY_DIR}
                                                    ${pybind11_INCLUDE_DIRS}
                                                    ${_INCLUDE_DIRS})
  target_link_libraries(${_NAME} PUBLIC pybind11_abseil
                                        Python::Module
                                        ${_DEPS}
                                        ${S6_LINK_LIBRARIES})
  target_link_options(${_NAME} PUBLIC ${_LINKOPTS})
endfunction()
