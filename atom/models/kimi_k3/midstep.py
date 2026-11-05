# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 midstep KDA state checkpoints (``ATOM_K3_KDA_MIDSTEP_CKPT``).

The part of the midstep path that needs no kernel: the switch, the
`StateTransfer` the backend declares, the snapshot targets of one prefill step,
and how a KDA layer writes them into their slots. The backend
(`atom.model_ops.attentions.kimi_mla_gdn_attn`) and the KDA layer
(`atom.models.kimi_k3.model`) call it; it imports no AITER, so the CPU test gate
reaches it.
"""

import os
from dataclasses import dataclass

import torch

from atom.model_engine.state_runtime import StateTransfer, midstep_checkpoint_targets
from atom.utils import envs

#: aiter FlashKDA's recurrence chunk: the state exists only at a chunk's top,
#: counted from each forward's first token.
FLASH_KDA_CHUNK = 32


def kda_midstep_enabled() -> bool:
    """ATOM_K3_KDA_MIDSTEP_CKPT=1, which needs aiter's FlashKDA recurrence:
    only it can hand back the state at an interior chunk boundary."""
    if not envs.ATOM_K3_KDA_MIDSTEP_CKPT:
        return False
    if os.getenv("AITER_FDA_ENABLE", "1").lower() not in ("1", "true", "yes", "on"):
        raise ValueError(
            "ATOM_K3_KDA_MIDSTEP_CKPT=1 needs AITER_FDA_ENABLE on: only the "
            "FlashKDA recurrence snapshots the state inside the forward"
        )
    return True


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

    Interior targets (`ATOM_K3_KDA_MIDSTEP_CKPT=1`): `targets[i] = (row, off)`
    -- the state of prefill row `row` after `off` of this step's tokens goes to
    scratch slot `slots[i]`. The SSM half comes out of the recurrence
    (`chunk_kimi_delta_attn(snapshot_targets=...)`); the conv half is the
    `conv_len` conv-input rows ending there (`conv_rows`, in target order),
    plus the rollback rows past them copied from `runtime_slots[i]`, which a
    prefill leaves as they were. A row can have several.

    End targets: a position at the forward's end is the runtime slot's own
    state once the layer has run, so `end_slots[j]` is copied whole from
    `end_runtime_slots[j]`.
    """

    targets: tuple[tuple[int, int], ...]
    conv_len: int
    slots: torch.Tensor | None
    runtime_slots: torch.Tensor | None
    conv_rows: torch.Tensor | None
    end_slots: torch.Tensor | None
    end_runtime_slots: torch.Tensor | None

    def write(
        self,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
        snap_out: torch.Tensor | None,
        mixed_qkv: torch.Tensor,
    ) -> None:
        """Write one layer's snapshots into their slots, once the layer has
        written its runtime slots. `snap_out` is the recurrence's state for
        `targets` (None when there are none); `mixed_qkv` is the step's conv
        input."""
        if snap_out is not None:
            ssm_state[self.slots] = snap_out
            # [slot, dim, state_len] here: the window's rows are its last
            # columns' worth of tokens, oldest first, as the conv kernel
            # stores them; the columns past them are speculation rollback,
            # which a prefill leaves alone.
            n = self.conv_len
            window = mixed_qkv[self.conv_rows].view(len(self.targets), n, -1)
            conv_state[self.slots, :, :n] = window.transpose(1, 2).to(conv_state.dtype)
            conv_state[self.slots, :, n:] = conv_state[self.runtime_slots, :, n:]
        if self.end_slots is not None:
            # A checkpoint at this forward's end is the runtime slot's own
            # state, now that both halves of it have been written.
            ssm_state[self.end_slots] = ssm_state[self.end_runtime_slots]
            conv_state[self.end_slots] = conv_state[self.end_runtime_slots]


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
    # Token offset of each row's first token in this step's flat layout.
    bases, base = [], 0
    for n in num_scheduled_tokens:
        bases.append(base)
        base += int(n)
    targets, slots, sources, rows, end_slots, end_src = [], [], [], [], [], []
    for row, dst, off, is_end, runtime in found:
        if is_end:
            end_slots.append(dst)
            end_src.append(int(runtime))
            continue
        targets.append((row, off))
        slots.append(dst)
        sources.append(int(runtime))
        b = bases[row]
        rows.extend(range(b + off - conv_len, b + off))

    def mk(col):
        if not col:
            return None
        return torch.tensor(col, dtype=torch.int64, device=device)

    return KdaSnapshots(
        targets=tuple(targets),
        conv_len=conv_len,
        slots=mk(slots),
        runtime_slots=mk(sources),
        conv_rows=mk(rows),
        end_slots=mk(end_slots),
        end_runtime_slots=mk(end_src),
    )
