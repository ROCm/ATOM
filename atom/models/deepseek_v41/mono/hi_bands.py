# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch bands for the hi hook: more requests a verify step than its static widths, on one ``Program.dyn_rows``
build a band.

A hook serves a step of B requests of ``req_rows`` rows each. Up to ``static_max`` requests it runs a static program
of exactly that width. Past it, a **band** (a number of requests, e.g. 6 or 8) runs one dyn build for every B up to
its size: its programs are built for ``band x req_rows`` rows and launched with the step's live words.

- :func:`plan_widths`: the widths a hook captures -> :class:`Width` (requests, rows a request, the band, the rows its
  programs are built for). Without bands it is the static split, unchanged.
- :class:`LiveWords`: the live words (rows, spec, batch) after a program's step counter, staged each step by a device
  copy (captured with the step's graph), and a band step's dead request slots for the model's own per-slot clears.

The model keeps what is its own: request-slot caches and their clears, peer groups, head launch ids, check pins.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["LiveWords", "Width", "plan_widths"]


@dataclass(frozen=True)
class Width:
    """A captured width: ``requests`` of ``req_rows`` rows; ``band`` the requests of the band that serves it (None:
    a static width)."""

    requests: int
    req_rows: int
    band: int | None = None

    @property
    def rows(self) -> int:
        return self.requests * self.req_rows

    @property
    def prog_rows(self) -> int:
        """The rows the width's programs are built for (a band's widths share its programs)."""
        return (self.band if self.band is not None else self.requests) * self.req_rows


def plan_widths(
    widths,
    req_rows: int,
    static_max: int,
    bands: tuple[int, ...] = (),
    one_request_max: int = 0,
) -> dict[int, Width]:
    """``{width: Width}`` for the step widths (rows) a hook serves.

    - ``w <= req_rows``: one request of ``w`` rows (a decode, a short verify);
    - a multiple of ``req_rows``: ``w / req_rows`` requests, up to ``static_max`` on static programs, then the
      smallest band of at least that many (none: an error);
    - ``one_request_max`` (> ``req_rows``): any other width up to it is one request, but never beside a batch width
      (a request slot's rows are at most ``req_rows`` then).
    """
    top = max((static_max, *bands))
    out: dict[int, Width] = {}
    for w in sorted(set(widths)):
        if 1 <= w <= req_rows:
            out[w] = Width(1, w)
        elif w % req_rows == 0 and w // req_rows <= top:
            b = w // req_rows
            band = None if b <= static_max else min(x for x in bands if x >= b)
            out[w] = Width(b, req_rows, band)
        elif w <= one_request_max:
            out[w] = Width(1, w)
        else:
            raise ValueError(
                f"width {w}: neither one request (<= {max(req_rows, one_request_max)} rows) nor up to {top} requests "
                f"of {req_rows} rows"
            )
    if not out:
        raise ValueError("no width")
    if any(x.requests > 1 for x in out.values()) and any(
        x.requests == 1 and w > req_rows for w, x in out.items()
    ):
        raise ValueError(
            f"widths {sorted(out)}: a one-request width past {req_rows} rows beside a batch width"
        )
    return out


class LiveWords:
    """The live words of a band step: ``step[1:4] = (rows, spec, batch)`` of the step's live requests, after the step
    counter in word 0 (the hi runtime's ``Buffers.set_live`` layout). ``step`` must hold at least 4 words before anything
    binds it (a peer group's counter: the hi runtime's ``bind_group`` adds to word 0 only).
    """

    def __init__(self, step: torch.Tensor, widths: dict[int, Width]):
        if step.numel() < 4 or step.dtype != torch.int32:
            raise ValueError(f"LiveWords: the step tensor needs 4 int32 words (counter + rows, spec, batch), got "
                             f"{step.numel()} {step.dtype}")  # fmt: skip
        self.step, self.widths = step, widths
        bands = {w: x for w, x in widths.items() if x.band is not None}
        self._row = {w: i for i, w in enumerate(sorted(bands))}
        words = [[x.rows, x.req_rows, x.requests] for _, x in sorted(bands.items())]
        self.words = torch.tensor(
            words or [[0, 0, 0]], dtype=torch.int32, device=step.device
        )

    def stage(self, rows: int) -> None:
        """Stage the live words of a band width (a device copy: a graph capture records it). A static width: none."""
        i = self._row.get(rows)
        if i is not None:
            self.step[1:4].copy_(self.words[i])

    def dead(self, rows: int) -> range | None:
        """A band width's dead request slots (``requests .. band - 1``): the model clears its own per-slot state
        there. None for a static width or a full band."""
        x = self.widths.get(rows)
        if x is None or x.band is None or x.requests == x.band:
            return None
        return range(x.requests, x.band)
