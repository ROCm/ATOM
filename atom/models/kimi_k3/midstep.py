# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 midstep KDA state checkpoints (``ATOM_K3_KDA_STATE_MIDSTEP``).

The part of the midstep path that needs no kernel: the switch, the
`StateTransfer` the backend declares, the snapshot targets of one prefill step,
and how a KDA layer writes them into their slots. The backend
(`atom.model_ops.attentions.kimi_mla_gdn_attn`) and the KDA layer
(`atom.models.kimi_k3.model`) call it; it imports no AITER, so the CPU test gate
reaches it.
"""

import inspect
import os
from dataclasses import dataclass

import torch

from atom.model_engine.state_runtime import StateTransfer, midstep_checkpoint_targets
from atom.utils import envs

#: aiter FlashKDA's recurrence chunk: the state exists only at a chunk's top,
#: counted from each forward's first token.
FLASH_KDA_CHUNK = 32

#: The `chunk_kimi_delta_attn` keywords a snapshot step passes (ROCm/aiter#7413).
SNAPSHOT_KWARGS = ("snapshot_targets", "snapshot_out")


def kda_midstep_enabled() -> bool:
    """ATOM_K3_KDA_STATE_MIDSTEP=1, which needs aiter's FlashKDA recurrence:
    only it can hand back the state at an interior chunk boundary."""
    if not envs.ATOM_K3_KDA_STATE_MIDSTEP:
        return False
    if os.getenv("AITER_FDA_ENABLE", "1").lower() not in ("1", "true", "yes", "on"):
        raise ValueError(
            "ATOM_K3_KDA_STATE_MIDSTEP=1 needs AITER_FDA_ENABLE on: only the "
            "FlashKDA recurrence snapshots the state inside the forward"
        )
    return True


def require_flash_kda_snapshots(chunk_kimi_delta_attn) -> None:
    """Refuse the switch at startup on an aiter that cannot take snapshots.

    Without this, an older aiter fails at the first prefill that has a snapshot
    target, with a `TypeError` from inside the forward. Takes the function
    rather than importing it, so this module stays importable without AITER.
    Named parameters only: a `**kwargs` catch-all would accept the keywords
    and still not store anything.
    """
    params = inspect.signature(chunk_kimi_delta_attn).parameters
    missing = [name for name in SNAPSHOT_KWARGS if name not in params]
    if missing:
        raise RuntimeError(
            "ATOM_K3_KDA_STATE_MIDSTEP=1 needs an aiter whose "
            "chunk_kimi_delta_attn takes snapshot_targets and snapshot_out "
            f"(ROCm/aiter#7413); the installed one has no {', '.join(missing)}"
        )


def kda_midstep_transfer(layout_id: str, conv_kernel_dim: int) -> StateTransfer:
    """The PAGE transfer of a midstep-readable Kimi-K3 state, for `layout_id`.

    Every checkpoint position inside a prefill forward (rung, demand and anchor
    alike) is snapshotted from the FlashKDA recurrence into a scratch slot and
    stored from there, so the chunk is no longer cut onto them. Grid: the
    recurrence's chunk; offset: the conv window this forward has to supply.
    """
    return StateTransfer.copy(
        layout_id,
        readable_midstep=True,
        midstep_grid=FLASH_KDA_CHUNK,
        midstep_min_offset=conv_kernel_dim - 1,
    )


@dataclass(frozen=True)
class KdaSnapshots:
    """Midstep KDA checkpoints of one prefill step, for every KDA layer.

    `targets[i] = (row, off)` are the interior positions, in the order aiter's
    `chunk_kimi_delta_attn(snapshot_targets=...)` stores them: the state of
    prefill row `row` after `off` of this step's tokens. A row can have several.

    The rest are `write_state_checkpoints`' per-target index tensors, interior
    targets first: each checkpoint's scratch slot, batch row and token offset;
    whether it sits at its row's end, where the state is the runtime slot's own
    once the layer has run; and otherwise the row of `snapshot_out` holding it.
    One launch writes both halves of every checkpoint: the recurrent state, and
    the `conv_len` conv-input rows ending at the position. The pool's columns
    past `conv_len` (speculation rollback) are left alone, as GDN leaves them:
    a resumed prefill reads only the first `conv_len`, and a spec-decode verify
    writes the rest before it reads them.
    """

    targets: tuple[tuple[int, int], ...]
    conv_len: int
    rows: torch.Tensor
    slots: torch.Tensor
    offs: torch.Tensor
    is_end: torch.Tensor
    runtime_slots: torch.Tensor
    h_rows: torch.Tensor
    cu_seqlens: torch.Tensor

    def checkpoint_args(
        self,
        snap_out: torch.Tensor | None,
        ssm_state: torch.Tensor,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> dict:
        """`write_state_checkpoints`' arguments for one layer, once it has
        written its runtime slots. `snap_out` is the recurrence's state for
        `targets`, or None when there are none (every target is then an end
        position, read from the pool); `mixed_qkv` is the step's conv input."""
        return {
            "h": ssm_state if snap_out is None else snap_out,
            "ssm_state": ssm_state,
            "x": mixed_qkv,
            "conv_state": conv_state,
            "rows": self.rows,
            "slots": self.slots,
            "offs": self.offs,
            "is_end": self.is_end,
            "runtime_slots": self.runtime_slots,
            "chunk_offsets": None,
            "cu_seqlens": self.cu_seqlens,
            "chunk_size": None,
            "conv_state_len": self.conv_len,
            "h_rows": self.h_rows,
        }


def kda_snapshots(
    saves,
    state_rows,
    num_cached_tokens,
    num_scheduled_tokens,
    runtime_slots,
    *,
    num_slots: int,
    conv_len: int,
    device,
) -> KdaSnapshots | None:
    """The midstep checkpoints one prefill step writes into scratch slots.

    Read off the step's `state_save_all` through `midstep_checkpoint_targets`,
    which raises rather than skips: the scheduler publishes every reserved hash
    after this forward, so a target not written is a findable image of garbage.
    Built once per step; every KDA layer writes the same targets.
    """
    if not saves or not any(saves):
        return None
    found = midstep_checkpoint_targets(
        state_rows,
        num_cached_tokens,
        num_scheduled_tokens,
        runtime_slots,
        saves,
        num_slots=num_slots,
        grid=FLASH_KDA_CHUNK,
        min_offset=conv_len,
    )
    if not found:
        return None
    interior = [t for t in found if not t[3]]
    ordered = interior + [t for t in found if t[3]]
    # First token of each row in this step's flat layout, then the total.
    cu_seqlens = [0]
    for n in num_scheduled_tokens:
        cu_seqlens.append(cu_seqlens[-1] + int(n))

    def mk(col):
        return torch.tensor(col, dtype=torch.int32, device=device)

    return KdaSnapshots(
        targets=tuple((row, off) for row, _dst, off, _end, _rt in interior),
        conv_len=conv_len,
        rows=mk([row for row, *_ in ordered]),
        slots=mk([int(dst) for _row, dst, *_ in ordered]),
        offs=mk([off for _row, _dst, off, *_ in ordered]),
        is_end=mk([int(bool(end)) for *_, end, _rt in ordered]),
        runtime_slots=mk([int(rt) for *_, rt in ordered]),
        h_rows=mk(list(range(len(interior))) + [0] * (len(ordered) - len(interior))),
        cu_seqlens=mk(cu_seqlens),
    )
