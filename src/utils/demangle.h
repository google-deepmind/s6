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

// Function for demangling symbol names.
// Calls a demangler API defined by C++ Itanium ABI.
//
// Reference: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler

#ifndef THIRD_PARTY_DEEPMIND_S6_UTILS_DEMANGLE_H_
#define THIRD_PARTY_DEEPMIND_S6_UTILS_DEMANGLE_H_

#include <string>

namespace deepmind::s6 {

// Demangle a mangled symbol name and return the demangled name.
// REQUIRES: mangled is not null.
// Note: This function uses __cxa_demangle which is prone to vulnerabilities and
// should only be used on known valid input.
std::string Demangle(const char* mangled);

}  // namespace deepmind::s6

#endif  // THIRD_PARTY_DEEPMIND_S6_UTILS_DEMANGLE_H_
