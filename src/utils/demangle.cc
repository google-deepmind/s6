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

#include "utils/demangle.h"

#include <cxxabi.h>

#include <cstdlib>
#include <string>

namespace deepmind::s6 {

std::string Demangle(const char* mangled) {
  int status = 0;
  char* demangled_c_str =
      abi::__cxa_demangle(mangled, nullptr, nullptr, &status);

  if (status == 0 && demangled_c_str != nullptr) {
    std::string demangled(demangled_c_str);
    std::free(demangled_c_str);
    return demangled;
  } else {
    return std::string(mangled);
  }
}

}  // namespace deepmind::s6
