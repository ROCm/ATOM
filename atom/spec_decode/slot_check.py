"""ATOM_DRAFT_SLOT_CHECK: find draft KV writes that land outside the forward's
own blocks.

The draft's pool rides the target's block ids, so a draft write through a slot
of some other request overwrites that request's draft KV and nothing else --
which is what a P/D push sees when its source's `draft.k/v` regions change
while every target region stays put. Each check compares a write's slot
mapping with the blocks the forward's requests hold, on the GPU, and copies the
verdict back through pinned memory behind an event, so the forward is never
synchronized: a race that a sync would hide stays visible.
"""

import logging
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("atom")

# Full reports before they are thinned to one in `_THIN`.
_FULL_REPORTS = 100
_THIN = 100


def foreign_writes(
    blocks: np.ndarray, foreign: np.ndarray, row_starts: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Of one check's read-back, the offending entries, their blocks, and the
    batch row each belongs to (`row_starts` maps a token index to its row;
    None means one entry per row)."""
    index = np.flatnonzero(foreign)
    if row_starts is None:
        rows = index
    else:
        rows = np.searchsorted(row_starts, index, side="right") - 1
    return index, blocks[index], rows


class DraftSlotCheck:
    """One per runner; `begin_forward` per forward, `check` per draft write."""

    def __init__(
        self,
        block_size: int,
        device: Any,
        describe: Callable[[list[int]], str] | None = None,
    ):
        self.block_size = block_size
        self.device = device
        self.describe = describe
        self._on_gpu = torch.device(device).type == "cuda"
        self.forwards = 0
        self.checks = 0
        self.reports = 0
        self.foreign = 0
        self._allowed: torch.Tensor | None = None
        self._meta: dict[str, Any] = {}
        self._pending: deque = deque()

    def begin_forward(self, batch: Any) -> None:
        """Upload the blocks this forward's requests hold."""
        self.poll()
        self.forwards += 1
        held = sorted({int(b) for table in batch.block_tables for b in table})
        host = torch.tensor(held or [-2], dtype=torch.int64, device="cpu")
        if self._on_gpu:
            host = host.pin_memory()
        self._allowed = host.to(self.device, non_blocking=True)
        scheduled = np.asarray(batch.num_scheduled_tokens, dtype=np.int64)
        self._meta = {
            "forward": self.forwards,
            "req_ids": list(batch.req_ids),
            "row_starts": np.concatenate(([0], np.cumsum(scheduled)[:-1])),
            "prefill_rows": int(getattr(batch, "total_seqs_num_prefill", 0)),
            "dummy": bool(getattr(batch, "is_dummy_run", False)),
            "held": len(held),
            "_host": host,
        }

    def check(self, slots: torch.Tensor | None, label: str, per_token: bool) -> None:
        """Queue the check of one draft write's slot mapping."""
        if slots is None or self._allowed is None:
            return
        if self._on_gpu and torch.cuda.is_current_stream_capturing():
            return
        self.poll()
        blocks = torch.where(
            slots >= 0,
            torch.div(slots, self.block_size, rounding_mode="floor"),
            torch.full_like(slots, -1),
        ).to(torch.int64)
        foreign = (blocks >= 0) & ~torch.isin(blocks, self._allowed)
        packed = torch.stack((blocks, foreign.to(torch.int64)))
        host = torch.empty(
            packed.shape, dtype=torch.int64, device="cpu", pin_memory=self._on_gpu
        )
        host.copy_(packed, non_blocking=True)
        done = None
        if self._on_gpu:
            done = torch.cuda.Event()
            done.record()
        self._pending.append((done, host, label, per_token, self._meta))

    def poll(self) -> None:
        """Report the checks whose read-back has landed."""
        while self._pending and (
            self._pending[0][0] is None or self._pending[0][0].query()
        ):
            _, host, label, per_token, meta = self._pending.popleft()
            self.checks += 1
            blocks, foreign = host.numpy()
            if not foreign.any():
                continue
            self._report(blocks, foreign, label, per_token, meta)

    def _report(self, blocks, foreign, label, per_token, meta) -> None:
        index, hit, rows = foreign_writes(
            blocks, foreign, meta["row_starts"] if per_token else None
        )
        self.reports += 1
        self.foreign += index.size
        if self.reports > _FULL_REPORTS and self.reports % _THIN:
            return
        req_ids = meta["req_ids"]
        owners = sorted({req_ids[r] for r in rows if 0 <= r < len(req_ids)})
        distinct = sorted({int(b) for b in hit})
        logger.error(
            "[draft-slot-check] forward %d %s: %d of %d draft KV writes land "
            "outside the %d blocks this batch holds (blocks %s at entries %s, "
            "written by rows %s, requests %s; %d rows, %d prefill, dummy %s; "
            "those blocks: %s) -- %d such checks, %d writes, in %d checks over "
            "%d forwards so far",
            meta["forward"],
            label,
            index.size,
            blocks.size,
            meta["held"],
            distinct[:16],
            index[:8].tolist(),
            sorted({int(r) for r in rows})[:8],
            owners[:8],
            len(req_ids),
            meta["prefill_rows"],
            meta["dummy"],
            self.describe(distinct) if self.describe else "owners unknown",
            self.reports,
            self.foreign,
            self.checks,
            self.forwards,
        )
