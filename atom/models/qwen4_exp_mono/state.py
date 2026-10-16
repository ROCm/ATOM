# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph-stable buffers and mailbox lifetime for Qwen MonoKernel."""

from dataclasses import dataclass

import torch

from atom.models.qwen4_exp_mono.config import (
    ALIGN,
    ENABLED_WIDTHS,
    GDN_HEAD_DIM,
    GDN_PROJ_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC,
    HIDDEN,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    SUPPORTED_WIDTHS,
)
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels.gdn import (
    GDR_FULL_READY_BYTES,
    GDR_OUT_PARTIAL_TASKS,
)
from atom.models.qwen4_exp_mono.kernels.gdn_layer import (
    scratch_bytes,
    scratch_layout,
)
from atom.models.qwen4_exp_mono.kernels.qsa_post import (
    scratch_bytes as qsa_post_scratch_bytes,
)
from atom.models.qwen4_exp_mono.kernels.qsa_post import (
    scratch_layout as qsa_post_scratch_layout,
)
from atom.models.qwen4_exp_mono.kernels.qsa_pre import (
    scratch_bytes as qsa_scratch_bytes,
)
from atom.models.qwen4_exp_mono.peer_layout import peer_bytes
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes

# The accepted TP1 path keeps its measured minimal allocation. TP8 uses the
# compact pull layout; expose the legacy constant for existing TP1 contracts.
PEER_BYTES = ALIGN
LAYER_SCRATCH = scratch_layout()
# Per-token sizes. Width builds place one complete token partition after
# another so a persistent task can derive ``token = task // tasks_per_token``
# and address its mailbox/scratch by one compile-time stride.
LAYER_SCRATCH_BYTES = scratch_bytes()
QSA_POST_SCRATCH = qsa_post_scratch_layout()
QSA_PRE_SCRATCH_BYTES = qsa_scratch_bytes()
QSA_POST_SCRATCH_BYTES = qsa_post_scratch_bytes()


def _align(value: int) -> int:
    return (value + ALIGN - 1) // ALIGN * ALIGN


def layer_scratch_bytes(tokens: int) -> int:
    return tokens * LAYER_SCRATCH_BYTES


def qsa_post_scratch_offset(tokens: int) -> int:
    return _align(tokens * QSA_PRE_SCRATCH_BYTES)


def qsa_width_scratch_bytes(tokens: int) -> int:
    return qsa_post_scratch_offset(tokens) + tokens * QSA_POST_SCRATCH_BYTES


def peer_buffer_bytes(tp: int) -> int:
    return PEER_BYTES if tp == 1 else peer_bytes(max(ENABLED_WIDTHS), tp)


# Preserve the S=1 names used by resource/tests while making the width layout
# explicit through the helpers above.
QSA_POST_SCRATCH_OFFSET = qsa_post_scratch_offset(1)
QSA_SCRATCH_BYTES = qsa_width_scratch_bytes(1)


@dataclass(frozen=True)
class GdnWorkspaceView:
    projected: torch.Tensor
    core: torch.Tensor
    normed: torch.Tensor
    out: torch.Tensor
    partials: torch.Tensor
    ready: torch.Tensor


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
    tokens: int
    scratch: torch.Tensor
    qsa_scratch: torch.Tensor
    qsa_post_scratch: torch.Tensor
    streams_out: tuple[torch.Tensor, torch.Tensor]
    pending_out: torch.Tensor
    pending_raw: torch.Tensor
    moe_out: torch.Tensor
    gdn: GdnWorkspaceView
    qsa: QsaSeamView
    qsa_post_attn_out: torch.Tensor


@dataclass(frozen=True)
class _WidthState:
    view: StateView
    mailboxes: StepMailboxes


class MonoState:
    """Graph-stable buffers, independently owned by each compile-time width."""

    def __init__(self, owner, group, rank: int, tp: int, device: torch.device) -> None:
        if LAYER_SCRATCH_BYTES % ALIGN:
            raise ValueError("Qwen mono scratch is not step-begin aligned")
        self.dims = Dims(tp)
        # Keep the accepted S=1 GPU allocation order: layer scratch, QSA
        # scratch/seams and outputs precede peer registration. Address order is
        # part of the measured graph contract on this memory-bound path.
        s1_view = self._allocate_view(1, device)
        self.peers, self._peer_finalizer = owned_peer_buffer(
            owner, peer_buffer_bytes(tp), group, rank, tp, device
        )
        try:
            self._widths = {1: self._bind_view(s1_view)}
        except Exception:
            self._peer_finalizer()
            raise

    def prepare(self, tokens: int, device: torch.device) -> None:
        if tokens not in SUPPORTED_WIDTHS:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token state")
        if tokens not in self._widths:
            self._widths[tokens] = self._bind_view(self._allocate_view(tokens, device))

    def _allocate_view(self, tokens: int, device: torch.device) -> StateView:
        scratch = torch.zeros(
            layer_scratch_bytes(tokens), dtype=torch.uint8, device=device
        )
        qsa_scratch = torch.zeros(
            qsa_width_scratch_bytes(tokens), dtype=torch.uint8, device=device
        )
        post_offset = qsa_post_scratch_offset(tokens)
        qsa_post_scratch = qsa_scratch.narrow(
            0, post_offset, tokens * QSA_POST_SCRATCH_BYTES
        )
        bf = torch.bfloat16
        streams_out = tuple(
            torch.empty(tokens, HC * HIDDEN, dtype=bf, device=device) for _ in range(2)
        )
        pending_out = torch.empty(tokens, HIDDEN, dtype=bf, device=device)
        pending_raw = torch.empty(tokens, HC, dtype=bf, device=device)
        moe_out = torch.empty(tokens, HIDDEN, dtype=bf, device=device)
        qsa = QsaSeamView(
            streams=torch.empty(tokens, HC * HIDDEN, dtype=bf, device=device),
            attn_raw=torch.empty(tokens, HC, dtype=bf, device=device),
            query=torch.empty(
                tokens,
                self.dims.qsa_heads,
                QSA_HEAD_DIM,
                dtype=bf,
                device=device,
            ),
            gate=torch.empty(tokens, self.dims.qsa_q_size, dtype=bf, device=device),
            index_query=torch.empty(
                tokens,
                QSA_INDEX_HEADS,
                QSA_INDEX_HEAD_DIM,
                dtype=bf,
                device=device,
            ),
        )
        gdn = GdnWorkspaceView(
            projected=self._scratch_view(
                scratch, tokens, "gdn_projected", torch.bfloat16, (GDN_PROJ_DIM,)
            ),
            core=self._scratch_view(
                scratch,
                tokens,
                "gdn_core",
                torch.bfloat16,
                (GDN_VALUE_HEADS, GDN_HEAD_DIM),
            ),
            normed=self._scratch_view(
                scratch, tokens, "gdn_normed", torch.bfloat16, (GDN_VALUE_DIM,)
            ),
            out=self._scratch_view(
                scratch, tokens, "gdn_out", torch.bfloat16, (HIDDEN,)
            ),
            partials=self._scratch_view(
                scratch,
                tokens,
                "gdn_partials",
                torch.float32,
                (GDR_OUT_PARTIAL_TASKS, 16),
            ),
            ready=self._scratch_view(
                scratch,
                tokens,
                "gdn_ready",
                torch.uint8,
                (GDR_FULL_READY_BYTES,),
            ),
        )
        qsa_post_off, qsa_post_size = QSA_POST_SCRATCH["qsa_attn_out"]
        if qsa_post_size != HIDDEN * torch.empty((), dtype=bf).element_size():
            raise ValueError("Qwen QSA attention output has an unexpected size")
        qsa_post_attn_out = self._partition_view(
            qsa_post_scratch,
            tokens,
            QSA_POST_SCRATCH_BYTES,
            qsa_post_off,
            bf,
            (HIDDEN,),
        )
        return StateView(
            tokens,
            scratch,
            qsa_scratch,
            qsa_post_scratch,
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            gdn,
            qsa,
            qsa_post_attn_out,
        )

    def _bind_view(self, view: StateView) -> _WidthState:
        self._validate_addresses(view)
        mailboxes = StepMailboxes(
            self.peers,
            view.scratch,
            view.qsa_scratch,
        )
        return _WidthState(view, mailboxes)

    def view(self, tokens: int) -> StateView:
        try:
            return self._widths[tokens].view
        except KeyError as exc:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token state") from exc

    def begin_step(self, tokens: int) -> None:
        try:
            self._widths[tokens].mailboxes.begin_step()
        except KeyError as exc:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token mailbox") from exc

    def _scratch_view(
        self,
        scratch: torch.Tensor,
        tokens: int,
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
        return self._partition_view(
            scratch, tokens, LAYER_SCRATCH_BYTES, offset, dtype, tail
        )

    @staticmethod
    def _partition_view(
        arena: torch.Tensor,
        tokens: int,
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
            (tokens, *tail),
            (partition_bytes // item_bytes, *reversed(strides)),
        )

    def _validate_addresses(self, view: StateView) -> None:
        addresses = {
            "scratch": view.scratch.data_ptr(),
            "qsa_scratch": view.qsa_scratch.data_ptr(),
            "qsa_post_scratch": view.qsa_post_scratch.data_ptr(),
            "peer": self.peers.bytes.data_ptr(),
            "streams_out_0": view.streams_out[0].data_ptr(),
            "streams_out_1": view.streams_out[1].data_ptr(),
            "pending_out": view.pending_out.data_ptr(),
            "pending_raw": view.pending_raw.data_ptr(),
            "moe_out": view.moe_out.data_ptr(),
            "qsa_post_attn_out": view.qsa_post_attn_out.data_ptr(),
            **{
                f"qsa_{name}": value.data_ptr()
                for name, value in vars(view.qsa).items()
            },
            **{
                f"gdn_{name}": value.data_ptr()
                for name, value in vars(view.gdn).items()
            },
        }
        bad = {name: ptr for name, ptr in addresses.items() if ptr % ALIGN}
        if bad:
            detail = ", ".join(f"{name}={ptr:#x}" for name, ptr in bad.items())
            raise MonoUnsupported(f"unaligned Qwen mono workspace: {detail}")

    def close(self) -> None:
        """Free peer mappings now; the finalizer makes repeated calls harmless."""

        self._peer_finalizer()
