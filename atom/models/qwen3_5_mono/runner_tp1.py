# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the Qwen3.5 TP1 mono decode (kernels_tp1.py): binds the model's
weights, allocates scratch / mailbox / counters, builds the per-layer argument tables
and launches one fused kernel a decoder layer on MTP verify steps, GDN on ReplaySSM.

As runner.py: activations ping-pong between two buffers by layer parity; layer 0's N1
is aiter's Gemma RMSNorm; the model's final norm takes the last (h, res).
"""

import logging
import os

import flydsl.expr as fx
import torch

from atom.models.qwen3_5_mono.config_tp1 import (
    ATT_D,
    ATT_KVH,
    ATT_O,
    ATT_QKVG,
    BLOCKS,
    EPS,
    GDN_BA,
    GDN_D,
    GDN_QKV,
    GDN_QKVZ,
    GDN_VH,
    GDN_Z,
    HIDDEN,
    INTER,
    N_EXPERTS,
    ROUTER_ROWS,
    SH_INTER,
    TG,
    TOK,
    W13_ROWS,
)
from atom.models.qwen3_5_mono.layout_tp1 import (
    A1_TILES,
    ATT_ARGS,
    ATT_STAMPS,
    G1_TILES,
    GDN_ARGS,
    GDN_STAMPS,
    M1_RT_TILES,
    MBOX_PAIRS,
    NARGS,
    SCRATCH,
    SCRATCH_F32,
    STAMPS,
)
from atom.models.qwen3_5_mono.runner import (
    MonoUnsupported,
    _bytes,
    _need,
    _w,
    _wmma_pack,
)

logger = logging.getLogger("atom")

__all__ = ["MonoRunnerTP1", "MonoUnsupported"]


def _ptr(t, dtype=fx.Int32):
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    return ptr_arg(t, dtype)


def replay_cache_len(spec_tokens: int) -> int:
    """ATOM's ReplaySSM record capacity for this build (gdn_attn: the env value, at
    least two windows)."""
    from atom.utils import envs

    return max(int(envs.ATOM_REPLAYSSM_CACHE_LEN), 2 * spec_tokens)


class MonoRunnerTP1:
    def __init__(self, causal_lm, spec_tokens: int):
        from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

        _need(get_tensor_model_parallel_world_size() == 1, "TP > 1")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _need(
            props.multi_processor_count == BLOCKS,
            f"{props.multi_processor_count} CUs != {BLOCKS}",
        )
        self.S = spec_tokens  # tokens a request: the anchor and the drafts
        _need(self.S in (4, 5), f"{self.S - 1} draft tokens")
        self.cap = replay_cache_len(self.S)
        self.max_ctx = int(os.environ.get("ATOM_QWEN35_MONO_MAX_CTX", "262144"))
        self.model = causal_lm.model
        self.layers = list(self.model.layers)
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.static = [self._bind(i, layer) for i, layer in enumerate(self.layers)]
        # [layer][ln1 | ln2]: a layer's ln1_w + 2 HIDDEN elements is the next layer's
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
            "att" if la.layer_type == "full_attention" else "gdn" for la in self.layers
        ]
        dev = self.dev
        self.scratch = torch.zeros(SCRATCH_F32, dtype=torch.float32, device=dev)
        self.mbox = torch.zeros(
            MBOX_PAIRS * 2, dtype=torch.int32, device=dev
        )  # tag 0 = empty
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
        self.dummy32 = torch.zeros(64, dtype=torch.int32, device=dev)
        self.timing = os.environ.get("ATOM_QWEN35_MONO_TIMING", "0") == "1"
        self.stamps = torch.zeros(
            len(self.layers), BLOCKS, STAMPS, dtype=torch.int64, device=dev
        )
        self.steps = 0
        self.tables = None
        self.launchers = {}
        self._precompile()
        logger.info(
            "Qwen3.5 TP1 mono decode: bound %d layers, %d tokens a request, ReplaySSM capacity %d",
            len(self.layers),
            self.S,
            self.cap,
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
        gu = mlp.shared_expert.gate_up_proj.weight
        _w(gu, (2 * SH_INTER, HIDDEN), what=f"{i}.shared.gate_up")
        # router (row 256: the shared gate logit) padded to 17 tiles, then the shared
        # expert's gate / up interleaved 8 and 8: a tile's 8 gate rows and their up rows
        sh = torch.stack(
            [
                gu[:SH_INTER].reshape(SH_INTER // 8, 8, HIDDEN),
                gu[SH_INTER:].reshape(SH_INTER // 8, 8, HIDDEN),
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
        b["w13"] = _bytes(ex.w13_weight, (N_EXPERTS, W13_ROWS, HIDDEN // 2), f"{i}.w13")
        b["w13_s"] = _bytes(
            ex.w13_weight_scale, (N_EXPERTS, W13_ROWS // 32, 2048), f"{i}.w13_scale"
        )
        b["w2"] = _bytes(ex.w2_weight, (N_EXPERTS, HIDDEN, INTER // 2), f"{i}.w2")
        b["w2_s"] = _bytes(
            ex.w2_weight_scale,
            (N_EXPERTS, HIDDEN // 32, (INTER // 128) * 128),
            f"{i}.w2_scale",
        )
        if layer.layer_type == "full_attention":
            at = layer.self_attn
            _w(at.qkv_proj.weight, (ATT_QKVG, HIDDEN), what=f"{i}.qkv")
            qkvg = _wmma_pack(at.qkv_proj.weight)
            keep.append(qkvg)
            b["qkvg"] = qkvg.data_ptr()
            b["q_norm_w"] = _w(at.q_norm.weight, (ATT_D,), what=f"{i}.q_norm")
            b["k_norm_w"] = _w(at.k_norm.weight, (ATT_D,), what=f"{i}.k_norm")
            _w(at.o_proj.weight, (HIDDEN, ATT_O), what=f"{i}.o_proj")
            op = _wmma_pack(at.o_proj.weight)
            keep.append(op)
            b["o_proj"] = op.data_ptr()
            rot = at.rotary_emb
            _need(
                getattr(rot, "rotary_dim", None) == 64
                and getattr(rot, "is_neox_style", False),
                f"{i}: rope",
            )
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
            if hasattr(la, "in_proj_qkvzba"):
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
            _need(getattr(la.conv1d, "bias", None) is None, f"{i}: conv bias")
            b["conv_w"] = _w(
                la.conv1d.weight.view(GDN_QKV, 4), (GDN_QKV, 4), what=f"{i}.conv"
            )
            b["dt_bias"] = _w(la.dt_bias, (GDN_VH,), what=f"{i}.dt_bias")
            b["a_log"] = _w(la.A_log, (GDN_VH,), what=f"{i}.A_log")
            b["gnorm_w"] = _w(la.norm.weight, (GDN_D,), what=f"{i}.norm")
            _w(la.out_proj.weight, (HIDDEN, GDN_Z), what=f"{i}.out_proj")
            op = _wmma_pack(la.out_proj.weight)
            keep.append(op)
            b["out_proj"] = op.data_ptr()
            b["_layer_num"] = la.attn.layer_num
        b["_keep"] = keep
        return b

    def _build_tables(self, fwd):
        tables = []
        S = self.S
        for i, b in enumerate(self.static):
            e = dict(b)
            e["h_in"] = self.h[i % 2].data_ptr()
            e["res_in"] = (self.zero_res if i == 0 else self.res[i % 2]).data_ptr()
            e["h_out"] = self.h[(i + 1) % 2].data_ptr()
            e["res_out"] = self.res[(i + 1) % 2].data_ptr()
            e["scratch"] = self.scratch.data_ptr()
            e["mbox"] = self.mbox.data_ptr()
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
                    tuple(kc.shape[1:]) == (ATT_KVH, ATT_D // 8, 16, 8)
                    and tuple(vc.shape[1:]) == (ATT_KVH, 2, ATT_D, 8),
                    f"{i}: kv layout {tuple(kc.shape)} {tuple(vc.shape)}",
                )
                e["k_cache"], e["v_cache"] = kc.data_ptr(), vc.data_ptr()
                names = ATT_ARGS
            else:
                cs, ck = kv.k_cache, kv.v_cache
                _need(
                    cs.dtype == torch.bfloat16 and ck.dtype == torch.bfloat16,
                    f"{i}: state dtypes",
                )
                _need(
                    tuple(cs.shape[1:]) == (S + 2, GDN_QKV) and cs.is_contiguous(),
                    f"{i}: conv state {tuple(cs.shape)} {cs.stride()}",
                )
                # the checkpoint as the ReplaySSM kernels address it: slot-dense [VH][K][V]
                _need(
                    ck.dim() == 4
                    and tuple(ck.shape[1:]) == (GDN_VH, GDN_D, GDN_D)
                    and ck.stride(0) == GDN_VH * GDN_D * GDN_D
                    and ck.stride(1) == GDN_D * GDN_D
                    and sorted(ck.stride()[2:]) == [1, GDN_D],
                    f"{i}: ReplaySSM checkpoint {tuple(ck.shape)} {ck.stride()}",
                )
                rk, ru, rg = kv.replay_buf_k, kv.replay_buf_u, kv.replay_buf_g
                _need(
                    rk is not None and ru is not None and rg is not None,
                    f"{i}: no ReplaySSM records",
                )
                _need(
                    tuple(rk.shape[1:]) == (GDN_VH, self.cap, GDN_D)
                    and tuple(ru.shape[1:]) == (GDN_VH, self.cap, GDN_D)
                    and tuple(rg.shape[1:]) == (GDN_VH, self.cap)
                    and rk.is_contiguous()
                    and ru.is_contiguous()
                    and rg.is_contiguous()
                    and rk.dtype == torch.bfloat16
                    and ru.dtype == torch.bfloat16
                    and rg.dtype == torch.float32,
                    f"{i}: ReplaySSM records {tuple(rk.shape)} {rk.dtype} {tuple(rg.shape)} {rg.dtype}",
                )
                e["conv_state"], e["ckpt"] = cs.data_ptr(), ck.data_ptr()
                e["rec_k"], e["rec_u"], e["rec_g"] = (
                    rk.data_ptr(),
                    ru.data_ptr(),
                    rg.data_ptr(),
                )
                names = GDN_ARGS
            if i + 1 < len(self.layers):
                nb = self.static[i + 1]
                nx_att = self.kinds[i + 1] == "att"
                e["nx_first"] = nb["qkvg"] if nx_att else nb["g1_w"]
                e["nx_rows"] = min(BLOCKS, A1_TILES if nx_att else G1_TILES)
                e["nx_out"] = nb["o_proj"] if nx_att else nb["out_proj"]
                e["nx_m1"] = nb["m1_w"]
            else:
                e["nx_first"] = e["nx_out"] = e["nx_m1"] = 0
                e["nx_rows"] = 0
            if os.environ.get("ATOM_QWEN35_MONO_PREFETCH", "1") != "1":
                e["nx_rows"] = 0
            tables.append([int(e[n]) for n in names] + [0] * (NARGS - len(names)))
        return tables

    def _precompile(self):
        """Compile every (kind, groups) now, not inside a graph capture."""
        before = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        z64 = torch.zeros(64, dtype=torch.int64, device=self.dev)
        z = _ptr(self.dummy32)
        try:
            for kind in ("gdn", "att"):
                for g in (1, 2):
                    self.launcher(kind, g)(
                        *[0] * NARGS, 1, z, z, _ptr(z64, fx.Int64), z, z, z, z, 0
                    )
        finally:
            if before is None:
                del os.environ["COMPILE_ONLY"]
            else:
                os.environ["COMPILE_ONLY"] = before

    # ------------------------------------------------------------------ step
    def launcher(self, kind, groups):
        key = (kind, groups)
        if key not in self.launchers:
            from atom.models.qwen3_5_mono.kernels_tp1 import build_layer_tp1

            self.launchers[key] = build_layer_tp1(
                kind,
                groups,
                self.S,
                cap=self.cap,
                max_ctx=self.max_ctx,
                timing=self.timing,
            )
        return self.launchers[key]

    def _meta(self, fwd, kind):
        """(m0, m1, m2, m3, mstride) of a ``kind`` layer for this step."""
        md = fwd.attn_metadata
        if kind == "gdn":
            gm = md.gdn_metadata
            si = gm.spec_state_indices_tensor
            return si, gm.num_accepted_tokens, gm.slot_idx, gm.write_pos, si.stride(0)
        bt = md.block_tables
        return bt, md.context_lens, md.slot_mapping, md.context_lens, bt.stride(0)

    def check_step(self, fwd, n):
        """This step's metadata fits the kernels (asserted once a shape: a mismatch is a bug)."""
        gm = fwd.attn_metadata.gdn_metadata
        _need(
            int(gm.replayssm_cache_len) == self.cap,
            f"ReplaySSM capacity {gm.replayssm_cache_len} != {self.cap}",
        )
        _need(
            int(gm.replayssm_max_query_len) == self.S,
            f"ReplaySSM window {gm.replayssm_max_query_len} != {self.S}",
        )
        _need(
            gm.slot_idx.dtype == torch.int32 and gm.write_pos.dtype == torch.int32,
            "ReplaySSM index dtypes",
        )

    def run_layer(self, i, n, positions, fwd, stream):
        kind = self.kinds[i]
        m0, m1, m2, m3, ms = self._meta(fwd, kind)
        nreq = n // self.S
        self.launcher(kind, 1 if n <= TG else 2)(
            *self.tables[i], nreq,
            _ptr(self.counters[0:]), _ptr(self.counters[32:]), _ptr(positions, fx.Int64),
            _ptr(m0), _ptr(m1), _ptr(m2), _ptr(m3), int(ms),
            stream=stream,
        )  # fmt: skip

    def forward(self, h_embed, positions):
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        n = h_embed.shape[0]
        if self.tables is None:
            self.check_step(fwd, n)
            self.tables = self._build_tables(fwd)
        stream = torch.cuda.current_stream()
        self.h[0, :n].copy_(h_embed)
        self._n1_first(n)
        for i in range(len(self.layers)):
            self.run_layer(i, n, positions, fwd, stream)
        hidden = self._final_norm(n)
        self.steps += 1
        return hidden

    def _n1_first(self, n):
        from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

        fused_qk_rmsnorm_group_quant(
            q=self.h[0, :n], q_weight=self.lnw[0, 0], q_epsilon=EPS, q_out_unquantized=self.xn1[:n],
            q_res_out=None, q_residual=None, gemma_norm=True,
        )  # fmt: skip

    def _final_norm(self, n):
        last = len(self.layers) % 2
        hidden, _ = self.model.norm(self.h[last, :n], self.res[last, :n])
        return hidden

    # ------------------------------------------------------------------ check
    def _state_views(self, i, fwd):
        """The cache / state tensors layer i writes this step (to snapshot)."""
        md = fwd.attn_metadata
        kv = fwd.kv_cache_data[f"layer_{self.static[i]['_layer_num']}"]
        if self.kinds[i] == "gdn":
            gm = md.gdn_metadata
            n = gm.num_spec_decodes
            conv = gm.spec_state_indices_tensor[:n, 0]
            conv = torch.unique(conv[conv >= 0].long())
            sl = gm.slot_idx[:n]
            sl = torch.unique(sl[sl >= 0].long())
            return [
                ("conv", kv.k_cache, conv), ("ckpt", kv.v_cache, sl),
                ("rec_k", kv.replay_buf_k, sl), ("rec_u", kv.replay_buf_u, sl), ("rec_g", kv.replay_buf_g, sl),
            ]  # fmt: skip
        slots = md.slot_mapping[md.slot_mapping >= 0].long()
        blocks = torch.unique(slots // 16)
        return [("k", kv.k_cache, blocks), ("v", kv.v_cache, blocks)]

    def forward_checked(self, h_embed, positions):
        """Run every layer both ways (ATOM's, then mono's from the same state), compare,
        keep the mono result (eager steps only)."""
        from atom.utils.forward_context import get_forward_context

        fwd = get_forward_context()
        n = h_embed.shape[0]
        if self.tables is None:
            self.check_step(fwd, n)
            self.tables = self._build_tables(fwd)
        stream = torch.cuda.current_stream()
        self.h[0, :n].copy_(h_embed)
        self._n1_first(n)
        ref_h, ref_res = h_embed, None
        rel = lambda a, b: (
            (a.float() - b.float()).norm() / (b.float().norm() + 1e-12)
        ).item()
        md = fwd.attn_metadata
        gm = md.gdn_metadata
        logger.info(
            "mono tp1 check step n=%d: ctx %s acc %s slots %s write_pos %s",
            n, md.context_lens[: n // self.S].tolist(), gm.num_accepted_tokens[: n // self.S].tolist(),
            gm.slot_idx[: n // self.S].tolist(), gm.write_pos[gm.slot_idx[: n // self.S].long().clamp(min=0)].tolist(),
        )  # fmt: skip
        isolated = os.environ.get("ATOM_QWEN35_MONO_CHECK_ISOLATED", "1") == "1"
        for i, layer in enumerate(self.layers):
            if isolated and i > 0:
                # mono layer i from native's layer i - 1 output: h, res and its N1 (xn1)
                self.h[i % 2, :n].copy_(ref_h)
                self.res[i % 2, :n].copy_(ref_res)
                with torch.no_grad():
                    out = layer.input_layernorm(ref_h.clone(), ref_res.clone())
                # (normed, residual), or with the fused quant (fp8, scale, bf16, residual)
                self.xn1[:n].copy_(out[2] if len(out) == 4 else out[0])
            views = self._state_views(i, fwd)
            snap = [t[idx].clone() for _, t, idx in views]
            rh, rr = layer(positions, ref_h, ref_res)
            after = [t[idx].clone() for _, t, idx in views]
            for (_, t, idx), s in zip(views, snap):
                t[idx] = s
            self.run_layer(i, n, positions, fwd, stream)
            mh = self.h[(i + 1) % 2, :n]
            mr = self.res[(i + 1) % 2, :n]
            st = [f"{nm} {rel(t[idx], a):.2e}" for (nm, t, idx), a in zip(views, after)]
            logger.info(
                "mono tp1 check layer %d (%s) n=%d: h rel %.2e  res rel %.2e  %s",
                i,
                self.kinds[i],
                n,
                rel(mh, rh),
                rel(mr, rr),
                " ".join(st),
            )
            ref_h, ref_res = rh, rr
        hidden = self._final_norm(n)
        self.steps += 1
        return hidden

    def log_timeline(self, n):
        torch.cuda.synchronize()
        st = self.stamps.double() * 10.0 / 1e3
        for kind, names in (("gdn", GDN_STAMPS), ("att", ATT_STAMPS)):
            idx = [i for i, k in enumerate(self.kinds) if k == kind]
            s = st[idx]
            t0 = s[:, :, 0].min(dim=1, keepdim=True).values
            ends = (s - t0.unsqueeze(-1)).max(dim=1).values
            med = ends.median(dim=0).values.tolist()[: len(names)]
            parts = [
                f"{names[k]} {med[k] - med[k - 1]:.1f}" for k in range(1, len(names))
            ]
            if (
                kind == "att"
            ):  # A2's sub-stamps: 19 q / k / v staged, 20 the passes done
                sub = ends.median(dim=0).values.tolist()
                parts.append(
                    f"[A2: prep {sub[19] - sub[3]:.1f}, new k/v {sub[22] - sub[19]:.1f}, passes {sub[20] - sub[22]:.1f} "
                    f"(last: qk {sub[23] - sub[25]:.1f}, softmax {sub[24] - sub[23]:.1f}, pv {sub[20] - sub[24]:.1f}), partials {sub[4] - sub[20]:.1f}]"
                )
            logger.info(
                "mono tp1 timeline %s n=%d: total %.1f us | %s",
                kind,
                n,
                med[len(names) - 1],
                " | ".join(parts),
            )
