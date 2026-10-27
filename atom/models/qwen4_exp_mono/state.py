# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph-stable buffers and mailbox lifetime for Qwen MonoKernel."""

from dataclasses import dataclass

import torch

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.layout import align as _align
from atom.models.qwen4_exp_mono.layout import (
    gdn_scratch_bytes as scratch_bytes,
)
from atom.models.qwen4_exp_mono.layout import (
    gdn_scratch_layout as scratch_layout,
)
from atom.models.qwen4_exp_mono.layout import (
    qsa_post_scratch_bytes,
    qsa_post_scratch_layout,
)
from atom.models.qwen4_exp_mono.layout import (
    qsa_pre_scratch_bytes as qsa_scratch_bytes,
)
from atom.models.qwen4_exp_mono.peer_layout import peer_bytes
from atom.models.qwen4_exp_mono.shapes import (
    ALIGN,
    HC,
    HIDDEN,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
)
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes

# The accepted TP1 path keeps its measured minimal allocation. TP8 uses the
# compact pull layout; expose the legacy constant for existing TP1 contracts.
PEER_BYTES = ALIGN
LAYER_SCRATCH = scratch_layout()
# Scratch partitions retain the accepted S1 arena offsets.
LAYER_SCRATCH_BYTES = scratch_bytes()
QSA_PRE_SCRATCH_BYTES = qsa_scratch_bytes()


def peer_buffer_bytes(tp: int) -> int:
    return PEER_BYTES if tp == 1 else peer_bytes(1, tp)


@dataclass(frozen=True)
class QsaSeamView:
    """Fixed-address K1 outputs consumed by QSA K2 and common finish."""

    streams: torch.Tensor
    attn_raw: torch.Tensor
    query: torch.Tensor
    gate: torch.Tensor
    index_query: torch.Tensor


@dataclass(frozen=True)
class StateView:
    """Persistent S1 workspace owned by MonoState; addresses survive replay."""

    batched = False
    tokens: int
    scratch: torch.Tensor
    qsa_scratch: torch.Tensor
    qsa_post_scratch: torch.Tensor
    streams_out: tuple[torch.Tensor, torch.Tensor]
    pending_out: torch.Tensor
    pending_raw: torch.Tensor
    moe_out: torch.Tensor
    gdn_out: torch.Tensor
    qsa: QsaSeamView
    qsa_post_attn_out: torch.Tensor


@dataclass(frozen=True)
class _StepState:
    view: StateView
    mailboxes: StepMailboxes


class MonoState:
    """Persistent S1 workspace and owned peer/mailbox lifecycle."""

    def __init__(self, owner, group, rank: int, tp: int, device: torch.device) -> None:
        s1_view = None

        def allocate():
            nonlocal s1_view
            s1_view = self._prepare_local(owner.max_model_len, tp, device)

        # No rank may enter peer registration while another is still allocating
        # or validating its local workspace. bind_agreed also converts OOM to a
        # uniform MonoUnsupported refusal before Native can resume.
        bind_agreed(allocate, group)
        self.peers, self._peer_finalizer = owned_peer_buffer(
            owner, peer_buffer_bytes(tp), group, rank, tp, device
        )

        def bind():
            if self.peers.bytes.data_ptr() % ALIGN:
                raise MonoUnsupported("unaligned Qwen mono peer workspace")
            self._state = self._bind_view(s1_view)

        try:
            # StepMailboxes allocates locally too; all ranks must reject and
            # release their mappings if any rank cannot finish construction.
            bind_agreed(bind, group)
        except Exception:
            self._peer_finalizer()
            raise

    def _prepare_local(self, max_model_len, tp, device):
        if LAYER_SCRATCH_BYTES % ALIGN:
            raise ValueError("Qwen mono scratch is not step-begin aligned")
        self.dims = Dims(tp)
        # TP8 only uses direct Mono attention; its long QSA seam is native.
        capacity = max_model_len if tp == 1 else 8192
        self.qsa_post_layout = qsa_post_scratch_layout(max_model_len=capacity)
        self.qsa_post_bytes = qsa_post_scratch_bytes(capacity)
        # Keep the accepted S=1 GPU allocation order: layer scratch, QSA
        # scratch/seams and outputs precede peer registration. Address order is
        # part of the measured graph contract on this memory-bound path.
        s1_view = self._allocate_view(device)
        self._validate_addresses(s1_view)
        return s1_view

    def _allocate_view(self, device: torch.device) -> StateView:
        scratch = torch.zeros(LAYER_SCRATCH_BYTES, dtype=torch.uint8, device=device)
        qsa_scratch = torch.zeros(
            _align(QSA_PRE_SCRATCH_BYTES) + self.qsa_post_bytes,
            dtype=torch.uint8,
            device=device,
        )
        post_offset = _align(QSA_PRE_SCRATCH_BYTES)
        qsa_post_scratch = qsa_scratch.narrow(0, post_offset, self.qsa_post_bytes)
        bf = torch.bfloat16
        streams_out = tuple(
            torch.empty(1, HC * HIDDEN, dtype=bf, device=device) for _ in range(2)
        )
        pending_out = torch.empty(1, HIDDEN, dtype=bf, device=device)
        pending_raw = torch.empty(1, HC, dtype=bf, device=device)
        moe_out = torch.empty(1, HIDDEN, dtype=bf, device=device)
        qsa = QsaSeamView(
            streams=torch.empty(1, HC * HIDDEN, dtype=bf, device=device),
            attn_raw=torch.empty(1, HC, dtype=bf, device=device),
            query=torch.empty(
                1,
                self.dims.qsa_heads,
                QSA_HEAD_DIM,
                dtype=bf,
                device=device,
            ),
            gate=torch.empty(1, self.dims.qsa_q_size, dtype=bf, device=device),
            index_query=torch.empty(
                1,
                QSA_INDEX_HEADS,
                QSA_INDEX_HEAD_DIM,
                dtype=bf,
                device=device,
            ),
        )
        gdn_out = self._scratch_view(scratch, "gdn_out", bf, (HIDDEN,))
        qsa_post_off, qsa_post_size = self.qsa_post_layout["qsa_attn_out"]
        if qsa_post_size != HIDDEN * torch.empty((), dtype=bf).element_size():
            raise ValueError("Qwen QSA attention output has an unexpected size")
        qsa_post_attn_out = self._partition_view(
            qsa_post_scratch,
            self.qsa_post_bytes,
            qsa_post_off,
            bf,
            (HIDDEN,),
        )
        return StateView(
            1,
            scratch,
            qsa_scratch,
            qsa_post_scratch,
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            gdn_out,
            qsa,
            qsa_post_attn_out,
        )

    def _bind_view(self, view: StateView) -> _StepState:
        mailboxes = StepMailboxes(
            self.peers,
            view.scratch,
            view.qsa_scratch,
        )
        return _StepState(view, mailboxes)

    def view(self, tokens: int) -> StateView:
        if tokens != 1:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token state")
        return self._state.view

    def begin_step(self, tokens: int) -> None:
        if tokens != 1:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token mailbox")
        self._state.mailboxes.begin_step()

    def _scratch_view(
        self,
        scratch: torch.Tensor,
        name: str,
        dtype: torch.dtype,
        tail: tuple[int, ...],
    ) -> torch.Tensor:
        offset, size = LAYER_SCRATCH[name]
        expected = torch.empty((), dtype=dtype).element_size()
        for dim in tail:
            expected *= dim
        if size != expected:
            raise ValueError(f"Qwen mono scratch {name} size mismatch")
        return self._partition_view(scratch, LAYER_SCRATCH_BYTES, offset, dtype, tail)

    @staticmethod
    def _partition_view(
        arena: torch.Tensor,
        partition_bytes: int,
        offset: int,
        dtype: torch.dtype,
        tail: tuple[int, ...],
    ) -> torch.Tensor:
        item_bytes = torch.empty((), dtype=dtype).element_size()
        value = arena.narrow(0, offset, arena.numel() - offset)
        if dtype != torch.uint8:
            value = value.view(dtype)
        strides = []
        stride = 1
        for dim in reversed(tail):
            strides.append(stride)
            stride *= dim
        return torch.as_strided(
            value,
            (1, *tail),
            (partition_bytes // item_bytes, *reversed(strides)),
        )

    def _validate_addresses(self, view: StateView) -> None:
        addresses = {
            "scratch": view.scratch.data_ptr(),
            "qsa_scratch": view.qsa_scratch.data_ptr(),
            "qsa_post_scratch": view.qsa_post_scratch.data_ptr(),
            "streams_out_0": view.streams_out[0].data_ptr(),
            "streams_out_1": view.streams_out[1].data_ptr(),
            "pending_out": view.pending_out.data_ptr(),
            "pending_raw": view.pending_raw.data_ptr(),
            "moe_out": view.moe_out.data_ptr(),
            "qsa_post_attn_out": view.qsa_post_attn_out.data_ptr(),
            "gdn_out": view.gdn_out.data_ptr(),
            **{
                f"qsa_{name}": value.data_ptr()
                for name, value in vars(view.qsa).items()
            },
        }
        bad = {name: ptr for name, ptr in addresses.items() if ptr % ALIGN}
        if bad:
            detail = ", ".join(f"{name}={ptr:#x}" for name, ptr in bad.items())
            raise MonoUnsupported(f"unaligned Qwen mono workspace: {detail}")

    def close(self) -> None:
        """Free peer mappings now; the finalizer makes repeated calls harmless."""

        self._peer_finalizer()
