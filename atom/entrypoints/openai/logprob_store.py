# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded-memory token records for parser regions not yet classified as text.

A tool-looking region can fall back to visible text at EOS. Spooling preserves
its probabilities without retaining O(tokens * top_k) Python objects in RAM.
Records have fixed width, so source-span lookup needs no growing in-memory index.
"""

import struct
from contextlib import ExitStack
from tempfile import SpooledTemporaryFile


class LogprobStore:
    MEMORY_LIMIT = 64 * 1024

    def __init__(self, top_k):
        self.layout = struct.Struct("<QQqdi" + "qd" * top_k)
        self.top_k = top_k
        self._files = ExitStack()
        self.file = self._files.enter_context(
            SpooledTemporaryFile(  # noqa: SIM115 - owned by the request ExitStack
                max_size=self.MEMORY_LIMIT, mode="w+b"
            )
        )
        self.count = 0

    def append(self, start, end, token, record):
        top = list(record.top[: self.top_k])
        top += [(-1, 0.0)] * (self.top_k - len(top))
        values = [start, end, token, record.logprob, record.rank]
        values.extend(value for pair in top for value in pair)
        self.file.seek(self.count * self.layout.size)
        self.file.write(self.layout.pack(*values))
        self.count += 1

    def read(self, index):
        self.file.seek(index * self.layout.size)
        return self.layout.unpack(self.file.read(self.layout.size))

    def within(self, start, end):
        left, right = 0, self.count
        while left < right:
            middle = (left + right) // 2
            if self.read(middle)[1] <= start:
                left = middle + 1
            else:
                right = middle
        for index in range(left, self.count):
            record = self.read(index)
            if record[0] >= end:
                break
            if record[1] <= end:
                yield record

    def close(self):
        self._files.close()
