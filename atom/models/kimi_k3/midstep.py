# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 midstep KDA state checkpoints (``ATOM_K3_KDA_STATE_MIDSTEP``).

The part of the midstep path that needs no kernel: the switch, the
`StateTransfer` the backend declares, the snapshot targets of one prefill step,
and the writer's arguments for them. The backend
(`atom.model_ops.attentions.kimi_mla_gdn_attn`) calls it and hands each KDA layer
the step's `KdaSnapshots` through the GDN metadata; the layer
(`atom.models.kimi_k3.model`) does not import it. It imports no AITER, so the CPU
test gate reaches it.
"""

from dataclasses import dataclass

import torch

from atom.model_engine.state_runtime import StateTransfer, midstep_checkpoint_targets
from atom.model_ops.gdn_midstep_snapshots import checkpoint_target_columns
from atom.utils import envs

#: aiter FlashKDA's recurrence chunk: the state exists only at a chunk's top,
#: counted from each forward's first token.
FLASH_KDA_CHUNK = 32


def kda_midstep_enabled() -> bool:
    """ATOM_K3_KDA_STATE_MIDSTEP=1. It needs an aiter with ROCm/aiter#7413,
    whose FlashKDA recurrence hands back the state at an interior chunk
    boundary; aiter refuses `snapshot_targets` on any other route."""
    return envs.ATOM_K3_KDA_STATE_MIDSTEP


def refuse_midstep_under_pipeline(pipeline_parallel_size: int) -> None:
    """Refuse the switch at startup under pipeline parallelism.

    Kimi-K3 keeps PAGE checkpoints under PP (ROCm/ATOM#2709), so the switch is
    read there, but its scratch-slot checkpoints have not been validated with
    a pipeline's forwards in flight on several stages. Without the switch, PP
    cuts prefill chunks onto the checkpoint positions.
    """
    if pipeline_parallel_size > 1:
        raise RuntimeError(
            "ATOM_K3_KDA_STATE_MIDSTEP=1 is not supported with pipeline "
            f"parallelism (pipeline_parallel_size={pipeline_parallel_size}); "
            "unset it to cut prefill chunks onto the checkpoint positions instead"
        )


def kda_midstep_transfer(
    layout_id: str, conv_kernel_dim: int, *, decode_keep: str
) -> StateTransfer:
    """The PAGE transfer of a midstep-readable Kimi-K3 state, for `layout_id`.

    Every checkpoint position inside a prefill forward (rung, demand and anchor
    alike) is written into a scratch slot and stored from there, so the chunk is
    no longer cut onto them: an interior position from the FlashKDA recurrence's
    snapshot, one at the forward's end from the runtime slot (`KdaSnapshots`).
    Grid: the recurrence's chunk; offset: the conv window this forward has to
    supply. `decode_keep` is the backend's own (`_decode_keep`): the switch
    changes where a prefill checkpoint is taken, not what a decode boundary
    keeps.
    """
    return StateTransfer.copy(
        layout_id,
        readable_midstep=True,
        midstep_grid=FLASH_KDA_CHUNK,
        midstep_min_offset=conv_kernel_dim - 1,
        decode_keep=decode_keep,
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
        position, read from the pool); `mixed_qkv` is the step's conv input.

        Every interior target names its row of `snap_out` in `h_rows`, so the
        writer never reads `chunk_offsets` or `chunk_size`, which locate a chunk
        kernel's boundaries; they are passed only because it takes them."""
        if snap_out is not None and snap_out.stride(0) != ssm_state.stride(0):
            raise ValueError(
                "write_state_checkpoints reads row i of snap_out at i * "
                "ssm_state.stride(0), so its rows must be as far apart as the slots"
            )
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
            "chunk_offsets": self.h_rows,
            "cu_seqlens": self.cu_seqlens,
            "chunk_size": 1,
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

    rows, slots, offs, is_end, runtime_slots = checkpoint_target_columns(
        ordered, device
    )
    return KdaSnapshots(
        targets=tuple((row, off) for row, _dst, off, _end, _rt in interior),
        conv_len=conv_len,
        rows=rows,
        slots=slots,
        offs=offs,
        is_end=is_end,
        runtime_slots=runtime_slots,
        h_rows=torch.tensor(
            list(range(len(interior))) + [0] * (len(ordered) - len(interior)),
            dtype=torch.int32,
            device=device,
        ),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
    )
