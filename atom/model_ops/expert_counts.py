# SPDX-License-Identifier: MIT
"""Per-forward routed-expert counts, for load-imbalance studies (LLM-Booster #740).

Off unless $ATOM_EXPERT_COUNTS_DIR is set. Each MoE layer adds its rows' logical top-k ids into one [layers, experts]
int32 buffer (capture-safe: scatter_add_ only, pad rows are -1 and drop out). At the start of the next forward the
buffer is copied to pinned memory without a host sync and zeroed, both on the current stream, so a forward's counts
are read one forward later. Every `_FLUSH` forwards they go to `<dir>/dp<rank>_<pid>_<seq>.npz`:
counts int16 [n, layers, experts] (layers: up to the highest one counted in the file), label (the profiler run label: kind, bs, tok, dp=[...]) and t (host time.time()).
"""

import atexit
import os
import re
import time

import numpy as np
import torch

DIR = os.environ.get("ATOM_EXPERT_COUNTS_DIR")
_FLUSH = 256
_LAYERS = 128
_buf = None
_pending = []  # (label, t, pinned host copy, event) of forwards not yet read
_rows = []  # (label, t, counts) read and waiting to be written
_label = None
_rank = 0
_seq = 0


def record(layer, topk_ids: torch.Tensor) -> None:
    """Add one MoE layer's routed rows (logical ids) to the buffer."""
    global _buf
    experts = layer.expert_layout.num_logical
    if _buf is None:
        _buf = torch.zeros(_LAYERS, experts, dtype=torch.int32, device=topk_ids.device)
    idx = int(re.search(r"layers\.(\d+)\.", layer.prefix).group(1))
    ids = topk_ids[:, : layer.top_k].reshape(-1).to(torch.int64)
    valid = (ids >= 0) & (ids < experts)
    _buf[idx].scatter_add_(0, ids.clamp(0, experts - 1), valid.to(torch.int32))


def start_forward(label: str, dp_rank: int) -> None:
    """Close the previous forward (copy its counts out, zero the buffer) and open one labelled `label`."""
    global _label, _rank
    _rank = dp_rank
    if _buf is not None:
        if _label is not None:
            host = torch.empty(_buf.shape, dtype=torch.int32, pin_memory=True)
            host.copy_(_buf, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            _pending.append((_label, time.time(), host, ev))
        _buf.zero_()
    while _pending and _pending[0][3].query():
        lab, t, host, _ = _pending.pop(0)
        _rows.append((lab, t, host.numpy().astype(np.int16)))
    if len(_rows) >= _FLUSH:
        _write()
    _label = label


def _write() -> None:
    global _seq
    if not _rows:
        return
    used = max(
        int(np.nonzero(c.any(axis=1))[0].max()) + 1 if c.any() else 0
        for _, _, c in _rows
    )
    np.savez(
        os.path.join(DIR, f"dp{_rank}_{os.getpid()}_{_seq:05d}.npz"),
        counts=np.stack([c[:used] for _, _, c in _rows]),
        label=np.array([lab for lab, _, _ in _rows]),
        t=np.array([t for _, t, _ in _rows]),
    )
    _seq += 1
    _rows.clear()


def _at_exit() -> None:
    start_forward(None, _rank)
    for lab, t, host, ev in _pending:
        ev.synchronize()
        _rows.append((lab, t, host.numpy().astype(np.int16)))
    _pending.clear()
    _write()


if DIR:
    os.makedirs(DIR, exist_ok=True)
    atexit.register(_at_exit)
