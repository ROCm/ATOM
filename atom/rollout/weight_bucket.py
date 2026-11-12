# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layout every bucketed weight path shares: a flat uint8 payload and,
per tensor, where it sits in it."""

from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import pairwise

import torch

_BUCKET_FIELDS = ("shape", "dtype", "offset", "nbytes")


def _is_count(value) -> bool:
    """A size a sender could have written: an integer, not negative, and not
    a bool, which Python counts as an int."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def decode_bucket(
    entries: Iterable[Mapping], payload: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Named views over *payload*, a flat uint8 buffer, one per entry of
    ``{"name", "shape", "dtype", "offset", "nbytes"}``.

    Views, not copies, except for a tensor whose offset is not a multiple of
    its element size, which ``view`` cannot reinterpret in place. Each entry is
    checked as sent, not coerced: ``int()`` made a shape of [2.5] into [2] and
    let -1 through as "infer this dimension". As a set, no name may repeat and
    no two ranges overlap: either hands one tensor's bytes to two parameters.
    A zero-element tensor holds no bytes, so its range overlaps none.
    """
    weights: list[tuple[str, torch.Tensor]] = []
    spans: list[tuple[int, int, str]] = []
    for entry in entries:
        fields = entry if isinstance(entry, Mapping) else {}
        name, shape = fields.get("name"), fields.get("shape")
        start, nbytes = fields.get("offset"), fields.get("nbytes")
        dtype_name = fields.get("dtype")
        dtype = (
            getattr(torch, dtype_name.removeprefix("torch."), None)
            if isinstance(dtype_name, str)
            else None
        )
        if not (
            isinstance(name, str)
            and name
            and isinstance(shape, (list, tuple))
            and all(_is_count(dim) for dim in shape)
            and isinstance(dtype, torch.dtype)
            and _is_count(start)
            and _is_count(nbytes)
        ):
            raise ValueError(f"invalid weight bucket entry: {entry}")
        end = start + nbytes
        if end > payload.numel():
            raise ValueError(
                f"weight bucket range out of bounds for {name}: offset={start} "
                f"nbytes={nbytes} payload={payload.numel()}"
            )
        if torch.Size(shape).numel() * dtype.itemsize != nbytes:
            raise ValueError(
                f"weight bucket size mismatch for {name}: shape={list(shape)} "
                f"dtype={dtype} nbytes={nbytes}"
            )
        raw = payload[start:end]
        if start % dtype.itemsize:
            raw = raw.clone()
        weights.append((name, raw.view(dtype).view(shape)))
        if nbytes:
            spans.append((start, end, name))
    repeated = sorted(
        n for n, seen in Counter(n for n, _ in weights).items() if seen > 1
    )
    if repeated:
        raise ValueError(f"weight bucket repeats names: {repeated[:20]}")
    spans.sort()
    for (_, prev_end, prev_name), (start, _, name) in pairwise(spans):
        if start < prev_end:
            raise ValueError(
                f"weight bucket ranges overlap: {prev_name} ends at byte "
                f"{prev_end}, {name} starts at {start}"
            )
    return weights


def bucket_entries(bucket_meta: Mapping) -> list[dict]:
    """``{name: {shape, dtype, offset, nbytes}}`` as ``decode_bucket`` takes
    it, with those four keys alone: any other the sender put in, a ``name``
    among them, would have overridden the bucket's own."""
    entries = []
    for name, meta in bucket_meta.items():
        fields = meta if isinstance(meta, Mapping) else {}
        entries.append(
            {"name": name, **{key: fields.get(key) for key in _BUCKET_FIELDS}}
        )
    return entries
