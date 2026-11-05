# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What one weight sync has written on a rank.

Kept by parameter object, not by name: a trainer sends checkpoint names --
``q_proj``, one expert's ``gate_proj`` -- that ATOM fuses into parameters named
otherwise, and a tied head is one parameter under two names.
"""

from collections.abc import Iterable, Mapping

import torch


class WeightSyncRecord:
    """From a sync's first update on this rank to the step that completes it."""

    def __init__(self) -> None:
        self._sent: set[str] = set()
        self._written: set[int] = set()
        # id(fused parameter) -> (pieces that make it whole, pieces written)
        self._parts: dict[int, tuple[frozenset, set]] = {}

    def take(self, name: str) -> None:
        """Note *name* as sent, refusing one this sync was sent already: its
        second copy would silently replace the first."""
        if name in self._sent:
            raise RuntimeError(f"{name!r} was sent twice in one weight sync")
        self._sent.add(name)

    def credit(self, param: torch.nn.Parameter) -> None:
        self._written.add(id(param))

    def credit_part(self, param: torch.nn.Parameter, part, whole: Iterable) -> None:
        """Count one piece of *param*, which is rewritten once every piece in
        *whole* has been."""
        _, written = self._parts.setdefault(id(param), (frozenset(whole), set()))
        written.add(part)

    def missing(self, params: Mapping[str, torch.nn.Parameter]) -> list[str]:
        """The names in *params* this sync left as they were.

        A static activation scale is not counted: it is calibrated, not
        trained, so no trainer sends one.
        """
        covered = self._written | {
            pid for pid, (whole, written) in self._parts.items() if whole <= written
        }
        return sorted(
            name
            for name, param in params.items()
            if id(param) not in covered and not name.endswith("input_scale")
        )
