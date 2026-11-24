# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-engine A/B of the NVFP4 sparse prefill switches against today's path (``ATOM_M3_PREFILL_CHECK``).

``ATOM_M3_PREFILL_CHECK=<file>``: every prefill layer runs the switched kernels (``ATOM_M3_PREFILL_INSERT``,
``ATOM_M3_PREFILL_FAST`` / ``_ATTN`` / ``_INDEX``) and then aiter's insert / scorer / selector / attention on
the same operands, and appends one line per comparison to ``<file>.<pid>``:
``<kind> layer=<n> rows=<r> equal=<0|1> [bad=<elements> maxabs=<x>]``. Debug only: it doubles the work
and syncs every layer.
"""

import contextlib
import os

import torch

PATH = os.environ.get("ATOM_M3_PREFILL_CHECK", "")
ENABLED = bool(PATH)
_out = None


def record(kind: str, layer: int, rows: int, got: torch.Tensor, want: torch.Tensor) -> None:
    global _out
    if _out is None:
        _out = open(f"{PATH}.{os.getpid()}", "a", buffering=1)
    same_shape = got.shape == want.shape and got.dtype == want.dtype
    a = got.contiguous().view(torch.uint8)
    b = want.contiguous().view(torch.uint8)
    eq = same_shape and torch.equal(a, b)
    line = f"{kind} layer={layer} rows={rows} equal={int(eq)}"
    if not eq and same_shape:
        line += f" bad_bytes={int((a != b).sum())}"
        if got.is_floating_point():
            line += f" maxabs={float((got.float() - want.float()).abs().nan_to_num(1e30).max()):.3e}"
    elif not eq:
        line += f" shapes={tuple(got.shape)}/{tuple(want.shape)}"
    _out.write(line + "\n")


_stash: dict = {}


def stash(name: str, ids: torch.Tensor) -> None:
    _stash[name] = ids.clone()


def compare_ids(layer: int, fwd_ctx=None) -> None:
    """The stashed ``got_ids`` / ``want_ids`` ``[H, rows, 16]``: equal item by item, as sets per row,
    and the first rows whose sets differ."""
    got, want = _stash.pop("got_ids", None), _stash.pop("want_ids", None)
    if got is None or want is None:
        return
    rows = got.shape[1]
    item_rows = (got != want).any(-1).any(0)
    set_rows = (got.sort(-1).values != want.sort(-1).values).any(-1).any(0)
    line = (f"ids layer={layer} rows={rows} equal={int(not bool(item_rows.any()))} "
            f"rows_item_diff={int(item_rows.sum())} rows_set_diff={int(set_rows.sum())}")
    for r in torch.nonzero(set_rows).flatten()[:2].tolist():
        line += f" | row {r}: got {got[:, r].tolist()} want {want[:, r].tolist()}"
    if not bool(set_rows.any()) and bool(item_rows.any()):
        r = int(torch.nonzero(item_rows).flatten()[0])
        line += f" | order row {r}: got {got[:, r].tolist()} want {want[:, r].tolist()}"
    _out.write(line + "\n")


@contextlib.contextmanager
def reference_path():
    """Today's kernels for the calls inside (aiter insert, scorer, selector, attention)."""
    from atom.model_ops.minimax_m3 import prefill_fast, prefill_kernels

    saved = (prefill_fast.ENABLED, prefill_kernels.ATTN, prefill_kernels.INDEX,
             prefill_kernels.INSERT)
    prefill_fast.ENABLED = False
    prefill_kernels.ATTN = prefill_kernels.INDEX = prefill_kernels.INSERT = "aiter"
    try:
        yield
    finally:
        (prefill_fast.ENABLED, prefill_kernels.ATTN, prefill_kernels.INDEX,
         prefill_kernels.INSERT) = saved


def touched_pages(slot_mapping: torch.Tensor, page_size: int = 128) -> torch.Tensor:
    s = slot_mapping[slot_mapping >= 0]
    return torch.unique(torch.div(s, page_size, rounding_mode="floor"))
