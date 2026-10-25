# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A vectorized fast path for tiktoken-based tokenizers' whitespace split.

Kimi's remote-code tokenizer (`tokenization_kimi.TikTokenTokenizer`) cuts
every text into pieces with no run of more than 25K whitespace or
non-whitespace characters before tiktoken sees it (tiktoken issue #195). Its
`_split_whitespaces_or_nonwhitespaces` walks the text one character at a time
in Python: ~220 ms of GIL for a 670K-token prompt, on the executor threads that
tokenize requests, which starves the API server's event loop. Real text almost
never has such a run, and then the split yields the text whole -- which a
numpy pass can establish in a few ms.
"""

import logging

import numpy as np

logger = logging.getLogger("atom")

# Every `str.isspace()` code point is below this.
_WS_LIMIT = 0x3001
_IS_SPACE = np.array([chr(c).isspace() for c in range(_WS_LIMIT)], dtype=bool)
_SPLIT = "_split_whitespaces_or_nonwhitespaces"


def longest_class_run(s: str) -> int:
    """Length of `s`'s longest run of all-whitespace or all-other characters."""
    if not s:
        return 0
    codes = np.frombuffer(s.encode("utf-32-le", "surrogatepass"), dtype="<u4")
    space = np.zeros(codes.size, dtype=bool)
    low = codes < _WS_LIMIT
    space[low] = _IS_SPACE[codes[low]]
    edges = np.flatnonzero(space[1:] != space[:-1])
    bounds = np.concatenate(([-1], edges, [codes.size - 1]))
    return int(np.diff(bounds).max())


def install_fast_whitespace_split(tokenizer) -> bool:
    """Give `tokenizer`'s class the fast path; False if it has no such split.

    The text comes back whole exactly when no run exceeds the limit, as the
    per-character loop would return it; any longer run takes that loop.
    """
    cls = type(tokenizer)
    slow = cls.__dict__.get(_SPLIT)
    if not isinstance(slow, staticmethod):
        return False
    if getattr(slow.__func__, "_atom_fast", False):
        return True
    slow_split = slow.__func__

    def split(s, max_consecutive_slice_len):
        if longest_class_run(s) <= max_consecutive_slice_len:
            return iter((s,))
        return slow_split(s, max_consecutive_slice_len)

    split._atom_fast = True
    setattr(cls, _SPLIT, staticmethod(split))
    logger.info("Vectorized %s.%s", cls.__name__, _SPLIT)
    return True
