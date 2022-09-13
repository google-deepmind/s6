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

"""S6 is a just-in-time compiler and profiler for Python."""

from s6.python.api import CompilationFailedError
from s6.python.api import inspect
from s6.python.api import jit
from s6.python.api import NotCompiledError
from s6.python.api import S6CodeDetail
from s6.python.api import S6JitCallable
__all__ = [
    "CompilationFailedError",
    "NotCompiledError",
    "S6CodeDetail",
    "S6JitCallable",
    "inspect",
    "jit",
]
