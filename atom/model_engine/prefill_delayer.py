"""
PrefillDelayer — cross-DP-rank prefill alignment for ATOM.

Direct port of SGLang's PrefillDelayer (refactored form, post #16269), keeping
the "mixed prefillable status → delay" core that fixes our 1k/1k workload's
~81% pad_waste during prefill forwards.

Mechanism (per scheduler tick):
  1. Each DP rank reports its local state via cpu all_gather:
       (local_prefillable, watermark_force_allow)
  2. Compute `prefillable_status` ∈ {all, none, mixed}:
       - "all"   → every rank has a new prefill ready  → allow (8-way aligned)
       - "none"  → no rank has any prefill             → allow (vacuous)
       - "mixed" → only some ranks have prefill ready  → DELAY
  3. In "mixed", refuse the prefill (return False from `should_allow_prefill`)
     for up to `max_delay_passes` consecutive ticks (default 30, ≈ 255ms at
     8.5ms/decode-tick) OR `max_delay_ms` wall-clock (default 5000ms),
     whichever comes first. After timeout, force-allow to bound worst-case
     TTFT.
  4. Safety valve: if local KV-cache usage drops below
     `token_usage_low_watermark`, the rank reports "force_allow" and the
     delayer falls through immediately (the GPU is idling, don't delay).

Why this fixes our problem:
  Today (no delayer) — when 1 rank has a new prefill and 7 have decodes,
  ATOM runs eager forward NOW. MoE all_to_all is bottlenecked by the
  prefill rank's ~1000 tokens, costing all 8 ranks ~118ms (a1.log: 81.7%
  pad_waste, 67% of wall in moe.gather). 57 such mixed forwards = 6.7s.
  With delayer — the prefilling rank waits up to 255ms for sibling ranks'
  waiting queues to also gain prefills, then all 8 ranks do prefill
  simultaneously (balanced all_to_all). 57 mixed forwards → ~8 aligned
  forwards × ~140ms = 1.1s. Expected: ~5.6s saved per request burst.

What's intentionally NOT ported from SGL (HEAD):
  - prometheus / observability hooks (ATOM has dp_timing instead)
  - NCCL gather path (ATOM's dp_group is gloo cpu_group; sufficient)
  - TBO / hisparse / dllm / spec interactions (N/A)
  - `queue_min_ratio` adaptive trigger (added 2026-05; defer until baseline)
  - `slot_condition` (max_running_requests - global_running_bs check):
        ATOM steady-state running_bs ≈ 16 vs max_num_seqs ≈ 256+, condition
        never true → omitted to keep code simple
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("ATOM_PREFILL_DELAYER_DEBUG", "0") == "1"


class PrefillDelayer:
    __slots__ = (
        "dp_size",
        "cpu_group",
        "max_delay_passes",
        "max_delay_ms",
        "token_usage_low_watermark",
        "_reduce_buf",
        "_delayed_count",
        "_delay_start_ts",
        "_skip_first",
        "_stat_allow",
        "_stat_delay",
        "_stat_timeout",
        "_stat_watermark",
        "_stat_log_every",
    )

    def __init__(
        self,
        dp_size: int,
        cpu_group,
        max_delay_passes: int = 30,
        token_usage_low_watermark: Optional[float] = None,
        max_delay_ms: float = 5000.0,
    ):
        self.dp_size = dp_size
        self.cpu_group = cpu_group
        self.max_delay_passes = max_delay_passes
        self.max_delay_ms = max_delay_ms
        self.token_usage_low_watermark = token_usage_low_watermark

        # 3-slot MAX-reduce buffer (gloo-friendly; mirrors the proven
        # `_sync_dp_state` all_reduce path in engine_core.py rather than
        # relying on all_gather_into_tensor — the stateless gloo group's
        # docstring warns broadcast-like ops are unreliable, and the
        # initial hang we saw with all_gather_into_tensor was consistent
        # with this caveat).
        #
        # Encoding:
        #   slot 0 = local_prefillable          (MAX → "any rank prefillable")
        #   slot 1 = local_force                (MAX → "any rank forces allow")
        #   slot 2 = NOT local_prefillable      (MAX → "any rank lacks prefill")
        # Then prefillable_status:
        #   any_prefillable AND any_not_prefillable → "mixed"
        #   any_prefillable AND NOT any_not_prefillable → "all"
        #   NOT any_prefillable → "none"
        # Single all_reduce, 3 int64s on cpu — negligible overhead.
        self._reduce_buf = torch.zeros(3, dtype=torch.int64, device="cpu")

        self._delayed_count: int = 0
        self._delay_start_ts: float = 0.0
        # Skip first negotiation: during warmup / first burst we want
        # decode batch_size to grow as fast as possible. Mirrors SGL
        # PR #19836 (`skip_first_delayer`).
        self._skip_first: bool = True

        # Aggregate counters for periodic logging
        self._stat_allow = 0
        self._stat_delay = 0
        self._stat_timeout = 0
        self._stat_watermark = 0
        self._stat_log_every = int(
            os.environ.get("ATOM_PREFILL_DELAYER_LOG_EVERY", "1000")
        )

        logger.info(
            f"PrefillDelayer initialized: dp_size={dp_size} "
            f"max_delay_passes={max_delay_passes} "
            f"max_delay_ms={max_delay_ms} "
            f"watermark={token_usage_low_watermark}"
        )

    def should_allow_prefill(
        self,
        local_prefillable: bool,
        token_usage: float,
    ) -> bool:
        """
        Returns True iff this rank is allowed to admit new prefills this tick.

        Args:
            local_prefillable: this rank has at least one new prefill ready
                (i.e. self.waiting non-empty and admission would succeed).
            token_usage: fraction of KV cache blocks currently in use
                (used_blocks / total_blocks ∈ [0, 1]). Used by the
                low-watermark safety valve.
        """
        # Local "force allow" if KV cache is underutilized — don't delay
        # when GPU is starving. Only meaningful if this rank actually has
        # a prefill to push through (otherwise force_allow is a no-op).
        force = False
        if (
            self.token_usage_low_watermark is not None
            and local_prefillable
            and token_usage < self.token_usage_low_watermark
        ):
            force = True

        # Cross-DP MAX-reduce: 3 booleans encoded as int64.
        self._reduce_buf[0] = 1 if local_prefillable else 0
        self._reduce_buf[1] = 1 if force else 0
        self._reduce_buf[2] = 0 if local_prefillable else 1
        torch.distributed.all_reduce(
            self._reduce_buf,
            op=torch.distributed.ReduceOp.MAX,
            group=self.cpu_group,
        )
        any_prefillable = int(self._reduce_buf[0].item()) > 0
        force_max = int(self._reduce_buf[1].item())
        any_not_prefillable = int(self._reduce_buf[2].item()) > 0

        # Derive 3-way status: all / none / mixed.
        prefillable_max = 1 if any_prefillable else 0
        prefillable_min = 0 if any_not_prefillable else 1

        # Watermark short-circuit: ANY rank below the watermark forces all
        # ranks to allow this tick. Without this the delayer can stall a
        # rank with a fresh prefill while the cluster is underloaded.
        if force_max > 0:
            self._stat_watermark += 1
            self._reset_delay()
            self._maybe_log()
            return True

        # Skip first call to maximize initial decode batch size build-up.
        if self._skip_first:
            self._skip_first = False
            self._reset_delay()
            self._stat_allow += 1
            self._maybe_log()
            return True

        # status = "all" or "none" → no skew, just allow
        if prefillable_min == prefillable_max:
            self._reset_delay()
            self._stat_allow += 1
            self._maybe_log()
            return True

        # status = "mixed" → delay if still within budget
        if self._delayed_count == 0:
            self._delay_start_ts = time.perf_counter()
        elapsed_ms = (time.perf_counter() - self._delay_start_ts) * 1000.0

        if (
            self._delayed_count < self.max_delay_passes
            and elapsed_ms < self.max_delay_ms
        ):
            self._delayed_count += 1
            self._stat_delay += 1
            if _DEBUG:
                logger.info(
                    f"[PrefillDelayer] DELAY: count={self._delayed_count} "
                    f"elapsed={elapsed_ms:.1f}ms "
                    f"any_prefillable={any_prefillable} "
                    f"any_not_prefillable={any_not_prefillable}"
                )
            self._maybe_log()
            return False

        # Timed out — force allow to bound worst-case TTFT
        self._stat_timeout += 1
        if _DEBUG:
            logger.info(
                f"[PrefillDelayer] TIMEOUT: count={self._delayed_count} "
                f"elapsed={elapsed_ms:.1f}ms force-allow"
            )
        self._reset_delay()
        self._maybe_log()
        return True

    def _reset_delay(self):
        self._delayed_count = 0
        self._delay_start_ts = 0.0

    def _maybe_log(self):
        total = (
            self._stat_allow
            + self._stat_delay
            + self._stat_timeout
            + self._stat_watermark
        )
        if self._stat_log_every <= 0 or total == 0:
            return
        if total % self._stat_log_every == 0:
            logger.info(
                f"[PrefillDelayer stats] total={total} "
                f"allow={self._stat_allow} delay={self._stat_delay} "
                f"timeout={self._stat_timeout} watermark={self._stat_watermark} "
                f"(delay_rate={self._stat_delay/total:.2%})"
            )
