# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Declared PR dependencies: after "depends on", "stacked on" or "blocked by",
the references (refs.py) to this repository in the rest of that paragraph, or
in the next one when the paragraph ends at the keyword ("Depends on:", then a
list). However the text is decorated (`**Depends on** PR #12`, a list, a
link), a reference there is a dependency.

Text is not read for meaning: "was stacked on #N" counts too, and so does any
other reference in the paragraph. The hold that follows says which and is
lifted by rewording the sentence.
"""

from __future__ import annotations

import re

from .refs import references

_KEYWORD = re.compile(
    r"\b(?:depends[ \t\-]+on|stacked[ \t\-]+on|blocked[ \t\-]+by)\b", re.IGNORECASE
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def _paragraph_from(text: str, start: int) -> tuple[str, int]:
    """The text from `start` to the end of its paragraph, and where the next
    paragraph starts (len(text) when none does)."""
    end = _PARAGRAPH_BREAK.search(text, start)
    if end is None:
        return text[start:], len(text)
    return text[start : end.start()], end.end()


def declared_dependencies(text: str, own_number: int, repo: str) -> tuple[int, ...]:
    """PR numbers of this repository declared as dependencies, ascending,
    without the PR itself."""
    found: set[int] = set()
    for keyword in _KEYWORD.finditer(text):
        declaration, following = _paragraph_from(text, keyword.end())
        if not re.search(r"\w", declaration):
            # The paragraph ends at the keyword ("Depends on:"): it
            # introduces the next one.
            declaration = _paragraph_from(text, following)[0]
        refs = references(declaration)
        found.update(
            ref.number for ref in refs if ref.is_in(repo) and ref.number != own_number
        )
    return tuple(sorted(found))
