# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PROBE_MEM: each step's activation memory against what startup planned.

The KV cache is sized from one profiled forward (`warmup_model`, a dummy
pure-prefill step) plus the allocator's peak at that point. A step the profile
did not cover -- a mixed prefill+decode step under DSpark and TBO, say -- can
need more, and on DeepSeek-V4-Pro it did: live memory went past the budget and
the allocator OOMed asking for one more MoE output. This probe says, per step
kind, how far real steps go past the profiled activations, and what the
allocator held when it ran out.

It arms only after `get_num_blocks` has read the allocator's peak: per-step
measurement resets that counter, and resetting it during the profile would
change the KV cache size the probe is meant to explain.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

_GiB = float(1 << 30)
# A new high is logged only when it beats the last one by this much, so a run
# prints a handful of lines per step kind rather than one per step.
_NEW_HIGH_MARGIN = 256 << 20
_SUMMARY_EVERY = 200


def step_kind(batch: Any) -> str:
    if getattr(batch, "is_dummy_run", False):
        return "dummy"
    if getattr(batch, "is_mixed", False):
        return "mixed"
    return "prefill" if batch.total_seqs_num_prefill else "decode"


class MemProbe:
    def __init__(self, label: str):
        self.enabled = envs.ATOM_PROBE_MEM
        self.label = label
        self._armed = False
        self._warmup: tuple[int, int] | None = None
        self._profile_activations = 0
        self._base = 0
        self._steps = 0
        self._highs: dict[str, int] = {}

    def wrap(self, forward: Callable[..., Any]) -> Callable[..., Any]:
        """``forward`` itself when the probe is off, so it costs nothing."""
        if not self.enabled:
            return forward

        @functools.wraps(forward)
        def probed(batch, *args, **kwargs):
            self.step_begin()
            try:
                out = forward(batch, *args, **kwargs)
            except torch.OutOfMemoryError:
                self.report_oom()
                raise
            self.step_end(batch)
            return out

        return probed

    def record_warmup(self, num_seqs: int, num_tokens: int) -> None:
        if self.enabled:
            self._warmup = (num_seqs, num_tokens)

    def record_budget(
        self,
        *,
        peak: int,
        current: int,
        non_torch: int,
        budget: int,
        kv_bytes: int,
    ) -> None:
        """What the profile measured and planned; arms per-step measurement."""
        if not self.enabled:
            return
        self._profile_activations = max(peak - current, 0)
        seqs, tokens = self._warmup or (0, 0)
        logger.warning(
            "PROBE_MEM %s profile: warmup %d seq(s) x %d token(s), pure prefill "
            "-> activations %.2f GiB over static %.2f GiB | non_torch %.2f GiB, "
            "budget %.2f GiB, kv %.2f GiB",
            self.label,
            seqs,
            tokens,
            self._profile_activations / _GiB,
            current / _GiB,
            non_torch / _GiB,
            budget / _GiB,
            kv_bytes / _GiB,
        )
        self._armed = True

    def step_begin(self) -> None:
        if not self._armed:
            return
        self._base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    def step_end(self, batch: Any) -> None:
        if not self._armed:
            return
        activations = torch.cuda.max_memory_allocated() - self._base
        kind = step_kind(batch)
        self._steps += 1
        if activations > self._highs.get(kind, 0) + _NEW_HIGH_MARGIN:
            self._highs[kind] = activations
            ratio = (
                activations / self._profile_activations
                if self._profile_activations
                else float("nan")
            )
            logger.warning(
                "PROBE_MEM %s new %s high: activations %.2f GiB (%.2fx profile) "
                "| prefill %d seq / %d tok, decode %d seq / %d tok",
                self.label,
                kind,
                activations / _GiB,
                ratio,
                batch.total_seqs_num_prefill,
                batch.total_tokens_num_prefill,
                batch.total_seqs_num_decode,
                batch.total_tokens_num_decode,
            )
        if self._steps % _SUMMARY_EVERY == 0:
            self._summary()

    def _summary(self) -> None:
        stats = torch.cuda.memory_stats()
        free, _ = torch.cuda.mem_get_info()
        logger.warning(
            "PROBE_MEM %s step %d: reserved %.2f alloc %.2f inactive_split %.2f "
            "device_free %.2f GiB | alloc_retries %d ooms %d | highs %s",
            self.label,
            self._steps,
            stats.get("reserved_bytes.all.current", 0) / _GiB,
            stats.get("allocated_bytes.all.current", 0) / _GiB,
            stats.get("inactive_split_bytes.all.current", 0) / _GiB,
            free / _GiB,
            stats.get("num_alloc_retries", 0),
            stats.get("num_ooms", 0),
            {k: round(v / _GiB, 2) for k, v in sorted(self._highs.items())},
        )

    def report_oom(self) -> None:
        if not self.enabled:
            return
        free, _ = torch.cuda.mem_get_info()
        logger.error(
            "PROBE_MEM %s OOM after %d probed step(s); device_free %.2f GiB\n%s",
            self.label,
            self._steps,
            free / _GiB,
            torch.cuda.memory_summary(),
        )
