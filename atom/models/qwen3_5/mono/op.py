# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper: scratch, symmetric peer memory and the launch of one rank's Qwen3.5 decoder layer
(Gated DeltaNet or gated full attention, then the MoE) over ATOM's weights and state."""

from __future__ import annotations

from typing import ClassVar

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.qwen3_5.mono.config import StepInputError
from atom.models.qwen3_5.mono.kernels.layer import build_qwen35_kernel
from atom.models.qwen3_5.mono.layout import layout, mailbox_regions, shape_dims
from atom.models.qwen3_5.mono.weights import LayerWeights
from atom.mono.runtime.debug import DIAG_BYTES, given_up_waits, region_namer
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.step_begin import ZERO_ALIGN

__all__ = ["Qwen35MonoKernel", "Qwen35Variant"]


class Qwen35Variant:
    """The compiled launch and scratch shared by every layer of one kind at one sample count, and the
    symmetric peer buffer (a runner's ``peers``, or the variant's own, begun by ``begin_step``). Mailbox
    tags are ``layer + 1`` and mailboxes are zeroed per step, so a step's layers take distinct ``layer``.
    """

    _cache: ClassVar[dict] = {}

    def __init__(
        self,
        cfg,
        samples: int,
        full: bool,
        rank: int = 0,
        npes: int = 1,
        group=None,
        tokens_per_seq=1,
        replay_cap=0,
        debug=False,
        timeline=False,
        peers=None,
    ):
        if full and cfg.local_kv_heads != 1:
            raise ValueError("the full-attention kernel holds one K / V head per rank")
        if samples % tokens_per_seq:
            raise ValueError(
                "an MTP verify launch holds whole sequences of tokens_per_seq rows"
            )
        self.cfg, self.S, self.full, self.rank, self.npes = (
            cfg,
            samples,
            full,
            rank,
            npes,
        )
        self.T = tokens_per_seq
        self.replay_cap = replay_cap
        self.debug = debug
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(
            samples, npes, full, tokens_per_seq=tokens_per_seq, **dims
        )
        self.scratch = torch.zeros(
            self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
        )
        # mailbox zeroing works in 16 B units
        assert self.scr_layout["_bytes"] % ZERO_ALIGN == 0
        self.mailboxes = None
        if peers is None:
            # collective over ``group`` (a CPU group)
            peers, self._free_peers = owned_peer_buffer(
                self, self.sym_layout["_bytes"], group, rank, npes, dev
            )
            self.mailboxes = StepMailboxes(peers, self.scratch, debug=debug)
        self.peer_buffer = peers
        self.sym, self.peers = peers.local, peers.addresses
        # target of unused pointer arguments
        self.zero = torch.zeros(1, dtype=torch.int32, device=dev)
        self.key = (
            samples,
            npes,
            full,
            tokens_per_seq,
            replay_cap,
            cfg.eps,
            cfg.rope_dim,
            tuple(sorted(dims.items())),
            debug,
            timeline,
        )
        if self.key not in Qwen35Variant._cache:
            Qwen35Variant._cache[self.key] = build_qwen35_kernel(
                samples,
                npes,
                full,
                eps=cfg.eps,
                rope_dim=cfg.rope_dim,
                tokens_per_seq=tokens_per_seq,
                replay_cap=replay_cap,
                debug=debug,
                timeline=timeline,
                **dims,
            )
        self.launch = Qwen35Variant._cache[self.key]

    def given_up_waits(self) -> list[str]:
        """A debug build's mailbox waits that gave up this step (synchronizes)."""
        off = self.scr_layout["diag"]
        names = [
            d.name
            for d in mailbox_regions(self.S, self.full, self.cfg.weights, self.npes)
        ]
        return given_up_waits(self.scratch[off : off + DIAG_BYTES], region_namer(names))

    def begin_step(self):
        """Zero the variant's own mailboxes and fence peers (one launch; graph-capturable)."""
        assert (
            self.mailboxes is not None
        ), "a runner's variant: the runner begins its steps"
        self.mailboxes.begin_step()


class Qwen35MonoKernel:
    """One rank of one decoder layer: ``packed`` (the kernel's view of ATOM's weights, shared across
    widths) run on a ``Qwen35Variant``. State (GDN conv / SSM, paged KV, RoPE) belongs to the caller.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int,
        npes: int,
        variant: Qwen35Variant,
        packed: dict,
        tokens_per_seq: int = 1,
    ):
        if (variant.S, variant.full, variant.npes, variant.rank, variant.T) != (
            samples,
            W.full,
            npes,
            rank,
            tokens_per_seq,
        ):
            raise StepInputError(
                "the variant was built for another sample count, layer kind or rank"
            )
        self.W, self.S, self.rank, self.npes, self.full = W, samples, rank, npes, W.full
        self.packed = packed
        self.variant = variant

    def forward(
        self,
        h,
        *,
        gdn_state=None,
        kv=None,
        rope=None,
        x_out=None,
        layer=0,
        begin=True,
        tl=0,
        check_only=False,
    ):
        """One layer on the f32 residual stream ``h`` [S, hidden] (``residual + bf16(rank sum)``,
        unrounded) -> the next layer's; updates state in place.

        GDN ``gdn_state``: ``(conv, ssm, slots_in, slots_out)``; MTP verify (T > 1): ``(conv, ssm,
        n_accepted, spec_slots)``, conv window [slots, T + 2, conv_dim]; ReplaySSM (``replay_cap`` > 0):
        ``(conv, ckpt, n_accepted, conv_slots, slot_idx, write_pos, rec_k, rec_u, rec_g)``, k, u bf16
        [slots, hv, cap, 128], g f32 [slots, hv, cap]. Full attention: ``kv = (k_cache, v_cache, k_scale,
        v_scale, positions, context_lens, block_tables, slot_mapping)`` in ATOM's MHA pool layout
        (slot -1 = no write), ``rope = (cos, sin)`` f32 [positions, rope_dim / 2].

        Tensors are checked first (``check_only`` stops there); a mismatch raises ``StepInputError``.
        """
        cfg, S = self.W.cfg, self.S
        if (
            tuple(h.shape) != (S, cfg.hidden)
            or h.dtype != torch.float32
            or not h.is_contiguous()
        ):
            raise StepInputError(
                f"h must be contiguous float32 {(S, cfg.hidden)}, got {tuple(h.shape)} {h.dtype}"
            )

        def i32(name, t, shape):
            if (
                tuple(t.shape[: len(shape)]) != shape
                or t.dtype != torch.int32
                or t.stride(-1) != 1
            ):
                raise StepInputError(
                    f"{name} must be int32 {shape} with unit-stride rows"
                )

        z = self.variant.zero
        conv = ssm = sl_in = sl_out = n_acc = z
        wpos = rk = ru = rg = z
        kc = vc = ks = vs = pos = ctx = bt = slots = cos = sin = z
        bt_stride = 0
        if self.full:
            kc, vc, ks, vs, pos, ctx, bt, slots = kv
            cos, sin = rope
            i32("positions", pos, (S,))
            i32("context_lens", ctx, (S // self.variant.T,))
            i32("slot_mapping", slots, (S,))
            i32("block_tables", bt, (S,))
            if kc.dtype not in (torch.float8_e4m3fn, torch.uint8) or kc.shape[1:] != (
                1,
                cfg.head_dim // 16,
                16,
                16,
            ):
                raise StepInputError(
                    "k_cache must be FP8 [blocks, 1, head_dim / 16, 16, 16] (ATOM's MHA pool)"
                )
            if vc.dtype != kc.dtype or vc.shape[1:] != (1, 1, cfg.head_dim, 16):
                raise StepInputError(
                    "v_cache must be FP8 [blocks, 1, 1, head_dim, 16] (ATOM's MHA pool)"
                )
            blocks = kc.shape[0]
            for name, t in (("k_scale", ks), ("v_scale", vs)):
                if t.dtype != torch.float32 or tuple(t.shape) != (blocks, 1, 16):
                    raise StepInputError(f"{name} must be float32 [{blocks}, 1, 16]")
            for name, t in (
                ("k_cache", kc),
                ("v_cache", vc),
                ("k_scale", ks),
                ("v_scale", vs),
            ):
                if vc.shape[0] != blocks or not t.is_contiguous():
                    raise StepInputError(f"{name} must be contiguous, {blocks} blocks")
            for name, t in (("cos", cos), ("sin", sin)):
                if t.dtype != torch.float32 or t.dim() != 2 or not t.is_contiguous():
                    raise StepInputError(
                        f"{name} must be contiguous float32 [rows, rope_dim / 2]"
                    )
            bt_stride = bt.stride(0)
        else:
            T = self.variant.T
            cap = self.variant.replay_cap
            if cap:
                conv, ssm, n_acc, sl_out, sl_in, wpos, rk, ru, rg = gdn_state
                i32("n_accepted", n_acc, (S // T,))
                if wpos.dtype != torch.int32 or not wpos.is_contiguous():
                    raise StepInputError("write_pos must be contiguous int32 [slots]")
                rec = (cfg.hv, cap)
                if (
                    rk.dtype != torch.bfloat16
                    or ru.dtype != torch.bfloat16
                    or rg.dtype != torch.float32
                    or tuple(rk.shape[1:]) != rec + (cfg.linear_k_dim,)
                    or tuple(ru.shape[1:]) != rec + (cfg.linear_v_dim,)
                    or tuple(rg.shape[1:]) != rec
                    or not (
                        rk.is_contiguous() and ru.is_contiguous() and rg.is_contiguous()
                    )
                ):
                    raise StepInputError(
                        f"record buffers must be contiguous bf16 / f32 [slots, {cfg.hv}, {cap}, ...]"
                    )
            elif T == 1:
                conv, ssm, sl_in, sl_out = gdn_state
                sl_out = sl_in if sl_out is None else sl_out
            else:
                conv, ssm, n_acc, sl_out = gdn_state
                sl_in = sl_out
                i32("n_accepted", n_acc, (S // T,))
            width = cfg.conv_kernel - 1 if T == 1 else T + 2
            if conv.dtype != torch.bfloat16 or tuple(conv.shape[1:]) != (
                width,
                cfg.conv_dim,
            ):
                raise StepInputError(
                    f"conv_state must be bf16 [slots, {width}, {cfg.conv_dim}]"
                )
            if ssm.dtype != torch.bfloat16 or tuple(ssm.shape[1:]) != (
                cfg.hv,
                cfg.linear_k_dim,
                cfg.linear_v_dim,
            ):
                raise StepInputError(
                    f"ssm_state must be bf16 [slots, {cfg.hv}, {cfg.linear_k_dim}, {cfg.linear_v_dim}] (K-major, as ATOM)"
                )
            i32("slots_in", sl_in, (S // T if cap else S,))
            i32("slots_out", sl_out, (S,))
            if not (conv.is_contiguous() and ssm.is_contiguous()):
                raise StepInputError("conv_state and ssm_state must be contiguous")
        if check_only:
            return None
        if x_out is None:
            x_out = torch.empty_like(h)
        p = self.packed

        def ptr(x):
            return x.data_ptr()

        def ptr0(k):
            return p[k].data_ptr() if k in p else 0

        v = self.variant
        if begin:  # standalone: the variant's own mailboxes
            v.begin_step()
        _run_compiled(
            v.launch,
            ptr(h),
            ptr(x_out),
            ptr(sl_in),
            ptr(sl_out),
            ptr(n_acc),
            ptr(wpos),
            ptr(rk),
            ptr(ru),
            ptr(rg),
            ptr(conv),
            ptr(ssm),
            ptr(pos),
            ptr(ctx),
            ptr(bt),
            ptr(slots),
            ptr(kc),
            ptr(vc),
            ptr(ks),
            ptr(vs),
            ptr(cos),
            ptr(sin),
            ptr0("g_qn1"),
            ptr0("g_kn1"),
            ptr(p["g_in1"]),
            ptr(p["w_qkvz"]),
            ptr(p["s_qkvz"]),
            ptr0("w_ba"),
            ptr0("w_conv"),
            ptr0("a_log"),
            ptr0("dt_bias"),
            ptr0("g_norm"),
            ptr(p["w_out"]),
            ptr(p["s_out"]),
            ptr(p["g_post1"]),
            ptr(p["w_r"]),
            ptr(p["w_ug"]),
            ptr(p["s_ug"]),
            ptr(p["w_dn"]),
            ptr(p["s_dn"]),
            ptr0("w_sh_ug"),
            ptr0("w_sh_dn"),
            ptr(v.scratch),
            v.sym,
            ptr(v.peers),
            self.rank,
            layer,
            bt_stride,
            tl,
            torch.cuda.current_stream(),
        )
        return x_out
