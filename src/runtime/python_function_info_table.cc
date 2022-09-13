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

#include "runtime/python_function_info_table.h"

namespace deepmind::s6 {

thread_local std::array<PythonFunctionInfoTable::Map::value_type*,
                        PythonFunctionInfoTable::kCacheSize>
    PythonFunctionInfoTable::cache_;

PythonFunctionInfoTable& PythonFunctionInfoTable::Instance() {
  static NoDestructor<PythonFunctionInfoTable> instance;
  return *instance;
}

}  // namespace deepmind::s6