# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Text slices retaining their original token-byte coordinates during parsing.

Only logprob requests construct SourceText. Ordinary strings use the existing
parser paths. A part is (text, source-byte start, source-byte end); UTF-8
replacement characters can cover fewer original bytes than their encoding.
"""


class SourceText(str):
    def __new__(cls, parts, owner):
        parts = tuple(p for p in parts if p[0])
        obj = super().__new__(cls, "".join(p[0] for p in parts))
        obj.parts = parts
        obj.owner = owner
        return obj

    @property
    def source_ranges(self):
        return [(lo, hi) for _, lo, hi in self.parts]

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return str.__getitem__(self, key)
        start, end, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Source text requires contiguous slices")
        parts, at = [], 0
        for text, lo, hi in self.parts:
            left, right = max(0, start - at), min(len(text), end - at)
            if left < right:
                if hi - lo == len(text.encode("utf-8")):
                    first = lo + len(text[:left].encode("utf-8"))
                    last = lo + len(text[:right].encode("utf-8"))
                else:
                    # A replacement part is one character mapped to the bytes
                    # that caused it. Never infer its source length from U+FFFD.
                    assert len(text) == 1 and left == 0 and right == 1
                    first, last = lo, hi
                parts.append((text[left:right], first, last))
            at += len(text)
        return SourceText(parts, self.owner)

    def __add__(self, other):
        return join_text((self, other))

    def __radd__(self, other):
        return join_text((other, self))


def join_text(parts):
    parts = list(parts)
    if not parts or type(parts[0]) is str and all(type(p) is str for p in parts):
        return "".join(parts)
    traced = next((p for p in parts if isinstance(p, SourceText)), None)
    if traced is None:
        return "".join(parts)
    spans = []
    for part in parts:
        if not part:
            continue
        if not isinstance(part, SourceText) or part.owner is not traced.owner:
            raise ValueError("Parser lost source-byte provenance")
        spans.extend(part.parts)
    return SourceText(spans, traced.owner)
