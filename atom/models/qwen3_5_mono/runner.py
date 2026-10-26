# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the Qwen3.5 mono decode: binds the model's weights, allocates
scratch / peer memory / counters, builds the per-layer argument tables and launches
one fused kernel a decoder layer.

Activations ping-pong between two buffers by layer parity: layer i reads
(h[i % 2], res[i % 2]) (layer 0: the embedding and a zero residual) and writes
(h[(i + 1) % 2], res[(i + 1) % 2]); the model's final norm takes the last pair.
"""

import logging
import os

import flydsl.expr as fx
import torch

from atom.models.qwen3_5_mono.config import (
    ATT_D,
    ATT_QKVG,
    BLOCKS,
    EPS,
    GDN_BA,
    GDN_QKV,
    GDN_QKVZ,
    GDN_VH,
    HIDDEN,
    MAX_TOKENS,
    N_EXPERTS,
    ROUTER_ROWS,
    SH_INTER,
    SPEC,
    TP,
)
from atom.models.qwen3_5_mono.layout import (
    A1_TILES,
    ATT_ARGS,
    ATT_STAMPS,
    G1_TILES,
    GDN_ARGS,
    GDN_STAMPS,
    M1_RT_TILES,
    MBOX_PAIRS,
    NARGS,
    PEER_BYTES,
    SCRATCH,
    SCRATCH_F32,
    STAMPS,
    TOK,
)

logger = logging.getLogger("atom")


class MonoUnsupported(RuntimeError):
    pass


def _need(cond, why):
    if not cond:
        raise MonoUnsupported(why)


def _w(t, shape, dtype=torch.bfloat16, what=""):
    """A weight's data pointer, after checking its shape, dtype and contiguity."""
    _need(
        tuple(t.shape) == tuple(shape),
        f"{what}: shape {tuple(t.shape)} != {tuple(shape)}",
    )
    _need(t.dtype == dtype, f"{what}: dtype {t.dtype} != {dtype}")
    _need(t.is_contiguous(), f"{what}: not contiguous")
    return t.data_ptr()


def _bytes(t, shape, what):
    """Pointer of a packed byte weight (fp4x2 / e8m0 / uint8) of ``shape`` bytes."""
    _need(t.element_size() == 1, f"{what}: {t.dtype} is not a byte type")
    _need(
        tuple(t.shape) == tuple(shape),
        f"{what}: shape {tuple(t.shape)} != {tuple(shape)}",
    )
    _need(t.is_contiguous(), f"{what}: not contiguous")
    return t.data_ptr()


def _wmma_pack(w):
    """A [R, K] bf16 weight in WMMA A-fragment order (layout.py): [R / 16][K / 32][32]
    [16], lane l = row 16 tile + l % 16, k = 32 s + 16 (l // 16) + 0..15."""
    r, k = w.shape
    rp = (r + 15) // 16 * 16
    if rp != r:
        w = torch.cat([w, w.new_zeros(rp - r, k)])
    return w.reshape(rp // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()


def _ptr(t):
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    return ptr_arg(t, fx.Int32)


class MonoRunner:
    """Construction binds and allocates (collective over the TP group: the peer
    buffer's handle exchange)."""

    def __init__(self, causal_lm, spec=False):
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )

        from atom.models.qwen3_5_mono.peers import PeerBuffer

        self.tp = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        _need(self.tp == TP, f"TP {self.tp} != {TP}")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _need(
            props.multi_processor_count == BLOCKS,
            f"{props.multi_processor_count} CUs != {BLOCKS}",
        )
        self.model = causal_lm.model
        self.layers = list(self.model.layers)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.dev = dev
        self.static = [self._bind(i, layer) for i, layer in enumerate(self.layers)]
        # the norm weights packed [layer][ln1 | ln2]: a layer's ln1_w + 2 HIDDEN elements is
        # the next layer's (the MTP kernels' all-reduce 1 normalizes for the next layer)
        self.lnw = torch.stack(
            [
                torch.stack(
                    [la.input_layernorm.weight, la.post_attention_layernorm.weight]
                )
                for la in self.layers
            ]
        ).contiguous()
        for b, w in zip(self.static, self.lnw):
            b["ln1_w"], b["ln2_w"] = w[0].data_ptr(), w[1].data_ptr()
        self.kinds = [
            "att" if layer.layer_type == "full_attention" else "gdn"
            for layer in self.layers
        ]

        self.scratch = torch.zeros(SCRATCH_F32, dtype=torch.float32, device=dev)
        self.mbox = torch.zeros(
            MBOX_PAIRS * 2, dtype=torch.int32, device=dev
        )  # LL pairs, tag 0 = empty
        self.peers = PeerBuffer(
            PEER_BYTES, get_tp_group().cpu_group, self.rank, TP, dev
        )
        self.peer_bases = [int(a) for a in self.peers.addresses.cpu().tolist()]
        self.counters = torch.zeros(
            64, dtype=torch.int32, device=dev
        )  # gdn at 0, att at 32
        self.h = torch.zeros(2, TOK, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.res = torch.zeros(2, TOK, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.zero_res = torch.zeros(TOK, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.xn1 = (
            self.scratch[SCRATCH["xn1"] : SCRATCH["xn1"] + TOK * HIDDEN // 2]
            .view(torch.bfloat16)
            .view(TOK, HIDDEN)
        )
        import os

        self.timing = os.environ.get("ATOM_QWEN35_MONO_TIMING", "0") == "1"
        self.stamps = torch.zeros(
            len(self.layers), BLOCKS, STAMPS, dtype=torch.int64, device=dev
        )
        self.steps = 0
        self.tables = None  # built on the first step (the caches come from its context)
        self.launchers = {}
        self.spec = (
            spec  # MTP verify steps (kernels_mtp.py) instead of one token a request
        )
        # the recurrent state's storage: V-major when ATOM's fused GDN decode owns it
        # (its cache builder's rule: the layer applies the gated norm in the op, and
        # neither ReplaySSM, which the mono path refuses, nor the lossy decode is on);
        # the caches are checked against this when the tables are built
        from atom.utils import envs as _envs

        impls = [
            getattr(
                getattr(getattr(layer, "linear_attn", None), "attn", None), "impl", None
            )
            for layer in self.layers
        ]
        self.vmajor = any(
            getattr(m, "gated_norm", None) is not None for m in impls
        ) and not getattr(_envs, "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", False)
        self._precompile()
        logger.info(
            "Qwen3.5 mono decode: bound %d layers on TP rank %d",
            len(self.layers),
            self.rank,
        )

    # ------------------------------------------------------------------ binding
    def _bind(self, i, layer):
        b = {
            "ln1_w": _w(layer.input_layernorm.weight, (HIDDEN,), what=f"{i}.ln1"),
            "ln2_w": _w(
                layer.post_attention_layernorm.weight, (HIDDEN,), what=f"{i}.ln2"
            ),
        }
        mlp = layer.mlp
        _need(
            hasattr(mlp, "experts") and mlp.shared_expert is not None,
            f"{i}: not a sparse MoE block",
        )
        _w(mlp.gate.weight, (ROUTER_ROWS, HIDDEN), what=f"{i}.gate")
        _w(
            mlp.shared_expert.gate_up_proj.weight,
            (2 * SH_INTER, HIDDEN),
            what=f"{i}.shared.gate_up",
        )
        # router (row 256 = the shared gate logit) padded to 17 tiles, then the shared
        # expert's gate / up interleaved 8 and 8: a tile's 8 gate rows and their up rows
        gu = mlp.shared_expert.gate_up_proj.weight
        sh = torch.stack(
            [
                gu[:SH_INTER].reshape(16, 8, HIDDEN),
                gu[SH_INTER:].reshape(16, 8, HIDDEN),
            ],
            1,
        ).reshape(2 * SH_INTER, HIDDEN)
        rt = torch.cat(
            [
                mlp.gate.weight,
                mlp.gate.weight.new_zeros(16 * M1_RT_TILES - ROUTER_ROWS, HIDDEN),
            ]
        )
        m1 = _wmma_pack(torch.cat([rt, sh]))
        keep = [m1]
        b["m1_w"] = m1.data_ptr()
        _w(
            mlp.shared_expert.down_proj.weight,
            (HIDDEN, SH_INTER),
            what=f"{i}.shared.down",
        )
        sh2 = _wmma_pack(mlp.shared_expert.down_proj.weight)
        keep.append(sh2)
        b["sh_w2"] = sh2.data_ptr()
        ex = mlp.experts
        b["w13"] = _bytes(ex.w13_weight, (N_EXPERTS, 512, 1024), f"{i}.w13")
        b["w13_s"] = _bytes(
            ex.w13_weight_scale, (N_EXPERTS, 16, 2048), f"{i}.w13_scale"
        )
        b["w2"] = _bytes(ex.w2_weight, (N_EXPERTS, HIDDEN, 128), f"{i}.w2")
        b["w2_s"] = _bytes(ex.w2_weight_scale, (N_EXPERTS, 64, 256), f"{i}.w2_scale")
        if layer.layer_type == "full_attention":
            at = layer.self_attn
            _w(at.qkv_proj.weight, (ATT_QKVG, HIDDEN), what=f"{i}.qkv")
            qkvg = _wmma_pack(at.qkv_proj.weight)
            keep.append(qkvg)
            b["qkvg"] = qkvg.data_ptr()
            b["q_norm_w"] = _w(at.q_norm.weight, (ATT_D,), what=f"{i}.q_norm")
            b["k_norm_w"] = _w(at.k_norm.weight, (ATT_D,), what=f"{i}.k_norm")
            _w(at.o_proj.weight, (HIDDEN, 1024), what=f"{i}.o_proj")
            op = _wmma_pack(at.o_proj.weight)
            keep.append(op)
            b["o_proj"] = op.data_ptr()
            rot = at.rotary_emb
            _need(
                getattr(rot, "rotary_dim", None) == 64
                and getattr(rot, "is_neox_style", False),
                f"{i}: rope",
            )
            # the cos / sin caches as the attention reads them (bf16), kept alive here
            cos = rot.cos_cache.to(self.dev, torch.bfloat16).contiguous()
            sin = rot.sin_cache.to(self.dev, torch.bfloat16).contiguous()
            _need(
                cos.shape[-1] == 32 and cos.numel() == cos.shape[0] * 32,
                f"{i}: cos cache {tuple(cos.shape)}",
            )
            keep += [cos, sin]
            b["cos"], b["sin"] = cos.data_ptr(), sin.data_ptr()
            b["_layer_num"] = at.attn.layer_num if hasattr(at.attn, "layer_num") else i
        else:
            la = layer.linear_attn
            if hasattr(
                la, "in_proj_qkvzba"
            ):  # ATOM's fused bf16 in_proj: rows qkvz | ba
                _w(
                    la.in_proj_qkvzba.weight,
                    (GDN_QKVZ + GDN_BA, HIDDEN),
                    what=f"{i}.in_qkvzba",
                )
                g1 = _wmma_pack(la.in_proj_qkvzba.weight)
            else:
                _w(la.in_proj_qkvz.weight, (GDN_QKVZ, HIDDEN), what=f"{i}.in_qkvz")
                _w(la.in_proj_ba.weight, (GDN_BA, HIDDEN), what=f"{i}.in_ba")
                g1 = _wmma_pack(
                    torch.cat([la.in_proj_qkvz.weight, la.in_proj_ba.weight])
                )
            keep.append(g1)
            b["g1_w"] = g1.data_ptr()
            b["conv_w"] = _w(
                la.conv1d.weight.view(GDN_QKV, 4), (GDN_QKV, 4), what=f"{i}.conv"
            )
            b["dt_bias"] = _w(la.dt_bias, (GDN_VH,), what=f"{i}.dt_bias")
            b["a_log"] = _w(la.A_log, (GDN_VH,), what=f"{i}.A_log")
            b["gnorm_w"] = _w(la.norm.weight, (128,), what=f"{i}.norm")
            _w(la.out_proj.weight, (HIDDEN, 1024), what=f"{i}.out_proj")
            op = _wmma_pack(la.out_proj.weight)
            keep.append(op)
            b["out_proj"] = op.data_ptr()
            b["_layer_num"] = la.attn.layer_num
        b["_keep"] = keep  # the packed copies stay alive with the runner
        return b

    def _build_tables(self, fwd):
        tables = []
        nl = len(self.layers)
        for i, b in enumerate(self.static):
            e = dict(b)
            e["h_in"] = self.h[i % 2].data_ptr()
            e["res_in"] = (self.zero_res if i == 0 else self.res[i % 2]).data_ptr()
            e["h_out"] = self.h[(i + 1) % 2].data_ptr()
            e["res_out"] = self.res[(i + 1) % 2].data_ptr()
            e["scratch"] = self.scratch.data_ptr()
            e["mbox"] = self.mbox.data_ptr()
            e["peers"] = self.peers.addresses.data_ptr()
            e["stamps"] = self.stamps[i].data_ptr()
            kv = fwd.kv_cache_data[f"layer_{b['_layer_num']}"]
            if self.kinds[i] == "att":
                kc, vc = kv.k_cache, kv.v_cache
                _need(
                    kc.dtype == torch.bfloat16
                    and kc.is_contiguous()
                    and vc.is_contiguous(),
                    f"{i}: kv cache",
                )
                _need(
                    tuple(kc.shape[1:]) == (1, 32, 16, 8)
                    and tuple(vc.shape[1:]) == (1, 2, 256, 8),
                    f"{i}: kv layout {tuple(kc.shape)} {tuple(vc.shape)}",
                )
                e["k_cache"], e["v_cache"] = kc.data_ptr(), vc.data_ptr()
                names = ATT_ARGS
            else:
                cs, ss = kv.k_cache, kv.v_cache
                _need(
                    cs.dtype == torch.bfloat16 and ss.dtype == torch.bfloat16,
                    f"{i}: state dtypes",
                )
                _need(
                    tuple(cs.shape[1:]) == ((SPEC + 2) if self.spec else 3, GDN_QKV)
                    and cs.is_contiguous(),
                    f"{i}: conv state {tuple(cs.shape)} {cs.stride()}",
                )
                _need(
                    tuple(ss.shape[1:]) == (GDN_VH, 128, 128),
                    f"{i}: ssm state {tuple(ss.shape)}",
                )
                # K-major [slot][hv][k][v], or ATOM's fused-decode V-major storage seen
                # through a transposed view
                vm = not ss.is_contiguous() and ss.transpose(-1, -2).is_contiguous()
                _need(ss.is_contiguous() or vm, f"{i}: ssm state strides {ss.stride()}")
                _need(
                    self.vmajor == vm,
                    f"{i}: ssm state layout (V-major {vm}) != the built kernels' ({self.vmajor})",
                )
                e["conv_state"], e["ssm_state"] = cs.data_ptr(), ss.data_ptr()
                names = GDN_ARGS
            if i + 1 < len(self.layers):
                nb = self.static[i + 1]
                nx_att = self.kinds[i + 1] == "att"
                e["nx_first"] = nb["qkvg"] if nx_att else nb["g1_w"]
                e["nx_rows"] = (
                    A1_TILES if nx_att else G1_TILES
                )  # tiles of the first projection
                e["nx_out"] = nb["o_proj"] if nx_att else nb["out_proj"]
                e["nx_m1"] = nb["m1_w"]
            else:
                e["nx_first"] = e["nx_out"] = e["nx_m1"] = 0
                e["nx_rows"] = 0
            if os.environ.get("ATOM_QWEN35_MONO_PREFETCH", "1") != "1":
                e["nx_rows"] = 0
            tables.append(
                [int(e[n]) for n in names] + [0] * (NARGS - len(names))
            )  # kernel arguments
        _ = nl
        return tables

    def _precompile(self):
        """Compile every (kind, width) now, not inside a graph capture."""
        import os

        z64 = torch.zeros(64, dtype=torch.int64, device=self.dev)
        z32 = torch.zeros(4, dtype=torch.int32, device=self.dev)
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        before = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            for kind in ("gdn", "att"):
                z = ptr_arg(z32, fx.Int32)
                if self.spec:
                    for g in (1, 2, 4):
                        self.launcher_mtp(kind, g)(
                            *[0] * (NARGS + TP),
                            0,
                            1,
                            z,
                            z,
                            ptr_arg(z64, fx.Int64),
                            z,
                            z,
                            z,
                            0,
                        )
                    continue
                for n in range(1, MAX_TOKENS + 1):
                    self.launcher(kind, n)(
                        *[0] * (NARGS + TP), 0, z, z, ptr_arg(z64, fx.Int64), z, z, z, 0
                    )
        finally:
            if before is None:
                del os.environ["COMPILE_ONLY"]
            else:
                os.environ["COMPILE_ONLY"] = before

    # ------------------------------------------------------------------ step
    def launcher(self, kind, n):
        key = (kind, n)
        if key not in self.launchers:
            from atom.models.qwen3_5_mono.kernels import build_layer

            # ATOM_QWEN35_MONO_MAX_CTX: experiments only (contexts must stay below it)
            max_ctx = int(os.environ.get("ATOM_QWEN35_MONO_MAX_CTX", "16384"))
            self.launchers[key] = build_layer(
                kind, n, max_ctx=max_ctx, timing=self.timing
            )
        return self.launchers[key]

    def launcher_mtp(self, kind, groups):
        key = ("mtp", kind, groups)
        if key not in self.launchers:
            from atom.models.qwen3_5_mono.kernels_mtp import build_layer_mtp

            max_ctx = int(os.environ.get("ATOM_QWEN35_MONO_MAX_CTX", "16384"))
            self.launchers[key] = build_layer_mtp(
                kind, groups, max_ctx=max_ctx, timing=self.timing, vmajor=self.vmajor
            )
        return self.launchers[key]

    def _meta(self, fwd, kind, n):
        """(m0, m1, m2, bt_stride) of a ``kind`` layer for this step."""
        md = fwd.attn_metadata
        if kind == "gdn":
            gm = md.gdn_metadata
            if self.spec:
                return (
                    gm.spec_state_indices_tensor,
                    gm.num_accepted_tokens,
                    gm.num_accepted_tokens,
                    0,
                )
            return (
                gm.non_spec_state_indices_tensor,
                gm.non_spec_state_indices_in_tensor,
                gm.non_spec_state_indices_tensor,
                0,
            )
        bt = md.block_tables
        return (bt, md.context_lens, md.slot_mapping, bt.stride(0))

    def run_layer(self, i, n, positions, fwd, stream):
        kind = self.kinds[i]
        m0, m1, m2, bts = self._meta(fwd, kind, n)
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        if self.spec:
            nreq = n // SPEC
            groups = 1 if nreq <= 4 else (2 if nreq <= 8 else 4)
            self.launcher_mtp(kind, groups)(
                *self.tables[i], *self.peer_bases, self.rank, nreq,
                ptr_arg(self.counters[0:], fx.Int32), ptr_arg(self.counters[32:], fx.Int32),
                ptr_arg(positions, fx.Int64), _ptr(m0), _ptr(m1), _ptr(m2), int(bts),
                stream=stream,
            )  # fmt: skip
            return

        self.launcher(kind, n)(
            *self.tables[i], *self.peer_bases, self.rank,
            ptr_arg(self.counters[0:], fx.Int32), ptr_arg(self.counters[32:], fx.Int32),
            ptr_arg(positions, fx.Int64), _ptr(m0), _ptr(m1), _ptr(m2), int(bts),
            stream=stream,
        )  # fmt: skip

    def forward(self, h_embed, positions):
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        n = h_embed.shape[0]
        if self.tables is None:
            self.tables = self._build_tables(fwd)
        stream = torch.cuda.current_stream()
        self.h[0, :n].copy_(h_embed)
        self._n1_first(n)
        for i in range(len(self.layers)):
            self.run_layer(i, n, positions, fwd, stream)
        hidden = self._final_norm(n)
        self.steps += 1
        if (
            self.timing
            and self.steps % 50 == 20
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.log_timeline(n)
        return hidden

    def _n1_first(self, n):
        """Layer 0's N1 (input RMSNorm, residual 0) into xn1: an MTP layer takes its N1
        from the previous layer's all-reduce 1."""
        if (
            self.spec
        ):  # aiter's Gemma RMSNorm (ATOM's GemmaRMSNorm.forward_cuda): one kernel
            from aiter.ops.fused_qk_rmsnorm_group_quant import (
                fused_qk_rmsnorm_group_quant,
            )

            fused_qk_rmsnorm_group_quant(
                q=self.h[0, :n],
                q_weight=self.lnw[0, 0],
                q_epsilon=EPS,
                q_out_unquantized=self.xn1[:n],
                q_res_out=None,
                q_residual=None,
                gemma_norm=True,
            )

    def _final_norm(self, n):
        """The model's final norm on the last layer's (h, res). The mono layers end
        all-reduced: under ATOM's AR + RMSNorm fusion the norm would all-reduce h again,
        so its fused all-reduce is off for this call."""
        last = len(self.layers) % 2
        norm = self.model.norm
        fused = getattr(norm, "fused_allreduce", False)
        norm.fused_allreduce = False
        try:
            hidden, _ = norm(self.h[last, :n], self.res[last, :n])
        finally:
            norm.fused_allreduce = fused
        return hidden

    def log_timeline(self, n):
        """Per stage: the latest CTA's end minus the earliest CTA's start, median over
        the layers of a kind (timing builds, 100 MHz clock)."""
        torch.cuda.synchronize()
        st = self.stamps.double() * 10.0 / 1e3  # us
        for kind, names in (("gdn", GDN_STAMPS), ("att", ATT_STAMPS)):
            idx = [i for i, k in enumerate(self.kinds) if k == kind]
            s = st[idx]  # [layers, BLOCKS, STAMPS]
            t0 = s[:, :, 0].min(dim=1, keepdim=True).values
            ends = (s - t0.unsqueeze(-1)).max(dim=1).values  # [layers, STAMPS]
            med = ends.median(dim=0).values.tolist()[: len(names)]
            parts = [
                f"{names[k]} {med[k] - med[k - 1]:.1f}" for k in range(1, len(names))
            ]
            skew = (
                (s[:, :, 0].max(dim=1).values - s[:, :, 0].min(dim=1).values)
                .median()
                .item()
            )
            n1_cta = (s[:, :, 1] - s[:, :, 0]).median().item()
            d16 = (s[:, :, 16] - s[:, :, 0]).median().item()
            d17 = (s[:, :, 17] - s[:, :, 0]).median().item()
            d18 = (s[:, :, 18] - s[:, :, 0]).median().item()
            logger.info(
                "mono timeline %s: CTA start skew %.1f us, N1 a CTA (median) %.1f us: table in at %.1f, inputs in at %.1f, first barrier at %.1f",
                kind,
                skew,
                n1_cta,
                d18,
                d16,
                d17,
            )
            what = "last graph replay" if n < 0 else f"n={n}"
            logger.info(
                "mono timeline %s %s (rank %d): total %.1f us | %s",
                kind,
                what,
                self.rank,
                med[-1],
                " | ".join(parts),
            )
            st0 = (
                s[:, :, 0] - s[:, :, 0].min(dim=1, keepdim=True).values
            )  # [layers, BLOCKS]
            m0 = st0.median(dim=0).values
            logger.info(
                "mono timeline %s start by bid %% 8: %s | by bid // 32: %s | latest bids %s",
                kind,
                " ".join(f"{v:.1f}" for v in m0.view(32, 8).mean(0).tolist()),
                " ".join(f"{v:.1f}" for v in m0.view(8, 32).mean(1).tolist()),
                m0.argsort(descending=True)[:12].tolist(),
            )
            fin = s[:, :, 15] - s[:, :, 0].min(dim=1, keepdim=True).values
            logger.info(
                "mono timeline %s end spread (latest - earliest CTA end) %.1f us",
                kind,
                (fin.max(dim=1).values - fin.min(dim=1).values).median().item(),
            )
            if kind == "gdn":
                segs = (
                    ("ag in", 8, 19),
                    ("N2", 19, 9),
                    ("M1", 9, 10),
                    ("logits in", 11, 20),
                    ("top-k + M3", 20, 21),
                    ("act in", 21, 22),
                    ("M4", 22, 14),
                    ("AR moe", 14, 15),
                )
                logger.info(
                    "mono timeline gdn MoE a CTA (median / max): %s",
                    " | ".join(
                        f"{nm} {(s[:, :, b] - s[:, :, a]).median().item():.1f} / {(s[:, :, b] - s[:, :, a]).max(dim=1).values.median().item():.1f}"
                        for nm, a, b in segs
                    ),
                )
            if kind == "att":
                d19 = (
                    (s[:, :, 19] - t0).max(dim=1).values.median().item() - med[3],
                    (s[:, :, 19] - s[:, :, 3]).median().item(),
                )
                logger.info(
                    "mono timeline att A2: q/k/v in (latest CTA) %.1f us after A1, %.1f us a CTA (median)",
                    *d19,
                )
                seg = [
                    (s[:, :, b] - s[:, :, a]).median().item()
                    for a, b in ((3, 19), (19, 20), (20, 21), (21, 4))
                ]
                logger.info(
                    "mono timeline att A2 a CTA (median): q/k/v in %.1f | attention %.1f | partial stores %.1f | flag %.1f",
                    *seg,
                )

    # ------------------------------------------------------------------ check
    def _state_views(self, i, fwd):
        """The cache / state tensors layer i writes this step (to snapshot)."""
        md = fwd.attn_metadata
        kv = fwd.kv_cache_data[f"layer_{self.static[i]['_layer_num']}"]
        if self.kinds[i] == "gdn":
            gm = md.gdn_metadata
            idx = (
                gm.spec_state_indices_tensor.reshape(-1)
                if self.spec
                else gm.non_spec_state_indices_tensor
            )
            idx = torch.unique(idx[idx >= 0].long())
            return [(kv.k_cache, idx), (kv.v_cache, idx)]
        slots = md.slot_mapping[md.slot_mapping >= 0].long()
        blocks = torch.unique(slots // 16)
        return [(kv.k_cache, blocks), (kv.v_cache, blocks)]

    def forward_checked(self, h_embed, positions):
        """Run every layer both ways, compare, keep the mono result (eager only)."""
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        n = h_embed.shape[0]
        if self.tables is None:
            self.tables = self._build_tables(fwd)
        stream = torch.cuda.current_stream()
        self.h[0, :n].copy_(h_embed)
        self._n1_first(n)
        ref_h, ref_res = h_embed, None
        if self.spec:
            md = fwd.attn_metadata
            gm = md.gdn_metadata
            logger.info(
                "mono check step n=%d: positions %s ctx %s slots %s acc %s spec idx %s",
                n,
                positions[0].tolist(),
                md.context_lens[: n // SPEC].tolist(),
                md.slot_mapping[:n].tolist(),
                gm.num_accepted_tokens.tolist(),
                gm.spec_state_indices_tensor.tolist(),
            )
        for i, layer in enumerate(self.layers):
            views = self._state_views(i, fwd)
            snap = [t[idx].clone() for t, idx in views]
            rh, rr = layer(positions, ref_h, ref_res)
            after = [t[idx].clone() for t, idx in views]
            for (t, idx), s in zip(views, snap):
                t[idx] = s
            self.run_layer(i, n, positions, fwd, stream)
            mh = self.h[(i + 1) % 2, :n]
            mr = self.res[(i + 1) % 2, :n]
            rel = lambda a, b: (
                (a.float() - b.float()).norm() / (b.float().norm() + 1e-12)
            ).item()
            st = [rel(t[idx], a) for (t, idx), a in zip(views, after)]
            logger.info(
                "mono check layer %d (%s) n=%d: h rel %.2e  res rel %.2e  state rel %s",
                i, self.kinds[i], n, rel(mh, rh), rel(mr, rr), " ".join(f"{x:.2e}" for x in st),
            )  # fmt: skip
            ref_h, ref_res = mh.clone(), mr.clone()  # continue from the mono values
        return self._final_norm(n)
