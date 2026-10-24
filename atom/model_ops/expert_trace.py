"""Per-step [layer, logical expert] token counts, for routing studies (ATOM_EXPERT_TRACE_DIR)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np
import torch

from atom.utils import envs

_DIR = envs.ATOM_EXPERT_TRACE_DIR
_CHUNK = 512
_CHECK_EVERY = 32


class _ExpertTrace:
    def __init__(self, num_layers: int, num_experts: int, device):
        self.num_layers, self.num_experts = num_layers, num_experts
        # Last bin collects pad rows and invalid ids.
        self.hist = torch.zeros(
            (num_layers, num_experts + 1), dtype=torch.int32, device=device
        )
        self.buf = torch.zeros(
            (_CHUNK, num_layers, num_experts + 1), dtype=torch.int32
        ).pin_memory()
        self.meta: list[tuple] = []
        self.n = self.part = self.steps = 0
        self.enabled = False
        self.rank = torch.cuda.current_device()
        self.pool = ThreadPoolExecutor(max_workers=1)
        os.makedirs(_DIR, exist_ok=True)

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        from atom.utils.forward_context import step_pad_rows

        ids = topk_ids.to(torch.int64)
        pad = step_pad_rows(int(topk_ids.shape[0]))
        if pad is not None:
            ids = torch.where(pad, self.num_experts, ids)
        ids = torch.where(
            (ids < 0) | (ids >= self.num_experts), self.num_experts, ids
        ).reshape(-1)
        self.hist[layer_id].scatter_add_(
            0, ids, torch.ones_like(ids, dtype=torch.int32)
        )

    def end(self, batch) -> None:
        self.steps += 1
        if self.steps % _CHECK_EVERY == 0:
            on = os.path.exists(os.path.join(_DIR, "ENABLE"))
            if self.enabled and not on:
                self.flush()
            self.enabled = on
        if not self.enabled:
            return
        self.buf[self.n].copy_(self.hist, non_blocking=True)
        self.meta.append(
            (
                time.time(),
                int(bool(getattr(batch, "is_dummy_run", False))),
                int(getattr(batch, "total_tokens_num_prefill", 0)),
                int(getattr(batch, "total_tokens_num_decode", 0)),
                int(getattr(batch, "total_seqs_num", 0)),
            )
        )
        self.n += 1
        if self.n == _CHUNK:
            self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return
        # The step copies are non_blocking into pinned memory.
        torch.cuda.current_stream().synchronize()
        hist = self.buf[: self.n].numpy().astype(np.uint16)
        meta = np.array(self.meta, dtype=np.float64)
        path = os.path.join(_DIR, f"rank{self.rank}_part{self.part:05d}")
        self.pool.submit(
            lambda: (
                np.save(path + "_hist.npy", hist),
                np.save(path + "_meta.npy", meta),
            )
        )
        self.part += 1
        self.n = 0
        self.meta = []


_TRACE: _ExpertTrace | None = None


def record_expert_trace(layer: torch.nn.Module, topk_logical: torch.Tensor) -> None:
    global _TRACE
    if not _DIR or topk_logical.numel() == 0:
        return
    layer_id = getattr(layer, "layer_id", None)
    if _TRACE is None:
        from atom.config import get_current_atom_config

        _TRACE = _ExpertTrace(
            int(get_current_atom_config().hf_config.num_hidden_layers),
            int(layer.expert_layout.num_routed),
            topk_logical.device,
        )
    # Drafter MoE layers sit past the target model's layers.
    if not isinstance(layer_id, int) or not 0 <= layer_id < _TRACE.num_layers:
        return
    _TRACE.record(layer_id, topk_logical)


def with_expert_trace(fn):
    if not _DIR:
        return fn

    @wraps(fn)
    def wrapper(self, batch, *args, **kwargs):
        if _TRACE is not None:
            _TRACE.hist.zero_()
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            if _TRACE is not None:
                _TRACE.end(batch)

    return wrapper
