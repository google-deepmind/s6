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

"""Modified from unpack sequence benchmark in https://github.com/python/pyperformance.

Modifications include:
  - Removing pyperf (causes seg fault due to S6 de-referencing frame pointer,
    known bug).
  - Run with and without S6 and compare.
  - Move the `do_unpack` for-loop out by one function call.


Microbenchmark for Python's sequence unpacking.
"""


import sys
import time

import s6


def do_unpacking(to_unpack):
    # 400 unpackings
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack
    a, b, c, d, e, f, g, h, i, j = to_unpack


def bench_tuple_unpacking(loops):
    x = tuple(range(10))

    range_it = range(loops)
    t0 = time.time()

    for _ in range_it:
      do_unpacking(x)

    return time.time() - t0


def bench_list_unpacking(loops):
    x = list(range(10))

    range_it = range(loops)
    t0 = time.time()

    for _ in range_it:
      do_unpacking(x)

    return time.time() - t0


def _bench_all(loops):
    dt1 = bench_tuple_unpacking(loops)
    dt2 = bench_list_unpacking(loops)
    return dt1 + dt2


def bench_all(loops):
  return _bench_all(loops)


@s6.jit
def s6_bench_all(loops):
  return bench_all(loops)


def add_cmdline_args(cmd, args):
    if args.benchmark:
        cmd.append(args.benchmark)


if __name__ == "__main__":
    it = 4000
    if len(sys.argv) >= 2:
        it = int(sys.argv[1])

    print(f'Starting unpack sequence (fast) benchmark, running for {it} iterations.')

    py37_time = bench_all(it)

    no_warmup_s6_time = s6_bench_all(it)

    print(f'S6 time without warmup: {no_warmup_s6_time} sec.')
    print(f'Default Python 3.7 time: {py37_time} sec.')
    print(f'Speedup: {(float(py37_time)) / no_warmup_s6_time}')
