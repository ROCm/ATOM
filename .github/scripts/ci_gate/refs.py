# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""References to pull requests in text, as GitHub writes them: `#N`,
`owner/repo#N` and `https://github.com/owner/repo/pull/N`. Owner and repository
names compare case-insensitively, as GitHub resolves them."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NAME = r"[A-Za-z0-9_.-]+"


# Neither owner/repo#N nor a bare #N may start inside a word, a path or
# another reference.
_REFERENCE = re.compile(
    rf"(?:https?://github\.com/(?P<url_repo>{_NAME}/{_NAME})/pull/(?P<url_number>[0-9]+)"
    rf"|(?<![\w./-])(?P<slug_repo>{_NAME}/{_NAME})#(?P<slug_number>[0-9]+)"
    r"|(?<![\w/#])#(?P<bare_number>[0-9]+))"
    r"(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Reference:
    repo: str | None  # None: a bare #N, which means this repository
    number: int

    def is_in(self, repo: str) -> bool:
        return self.repo is None or self.repo.lower() == repo.lower()


def references(text: str) -> list[Reference]:
    """Every reference in `text`, in order."""
    found = []
    for match in _REFERENCE.finditer(text):
        groups = match.groupdict()
        if groups["url_number"]:
            found.append(Reference(groups["url_repo"], int(groups["url_number"])))
        elif groups["slug_number"]:
            found.append(Reference(groups["slug_repo"], int(groups["slug_number"])))
        else:
            found.append(Reference(None, int(groups["bare_number"])))
    return found
