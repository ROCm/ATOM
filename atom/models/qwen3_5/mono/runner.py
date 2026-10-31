# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One decode step of up to ``MAX_TOKENS`` rows on the Qwen3.5 layer kernel: every decoder layer one
launch (``op.Qwen35MonoKernel``), over ATOM's own weights and state.

State is ATOM's: the GDN conv / recurrent pools and their state slots, the paged FP8 KV cache with its
scales, block tables and slot mapping, read and written in place, so a sequence can switch paths
between steps. The weights are ATOM's tensors, read in place in the layouts aiter's ``shuffle_weight``
(16, 16) gives them: the FP8 checkpoint's matrices and block scales (the shared expert fused into the
bank as expert 512), or the MXFP4 checkpoint's E2M1 bank and E8M0 scales with its BF16 projections
and shared expert. Only small tensors are derived at install: the f32 Gemma gammas ``1 + w``, the
router in MFMA tiles, the GDN b | a projection.

A step launches at its own width (rows): a plain decode one row a request, an MTP verify ``T =
spec_steps + 1`` rows a request on ATOM's GDN state -- ReplaySSM by default (a checkpoint a request
and ``replay_cap`` records of (k, u, g) a v-head; ATOM advances the record cursor before the step),
or the snapshot state (``ATOM_ENABLE_REPLAYSSM=0``: a state slot a drafted token). A width's kernels
and buffers (scratch, peer memory) are built on its first step (``WidthBuilds``); the layers of one
kind (GDN, full attention) share them.
"""

import logging

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.qwen3_5.mono.config import MAX_TOKENS, Q35Config, is_full_attention
from atom.models.qwen3_5.mono.dispatch import final_norm
from atom.models.qwen3_5.mono.layout import (
    BF16_GEMM_RULES,
    FLYDSL_BUDGET,
    FLYDSL_MAX_PARTS,
    TL_POINTS,
    aiter_bf16_rule,
    bf16_gemms,
    layout,
    pa_splits,
    router_rows,
    shape_dims,
)
from atom.models.qwen3_5.mono.op import (
    Qwen35MonoKernel,
    Qwen35Variant,
    StepInputError,
    zero_abi,
)
from atom.models.qwen3_5.mono.packing import pack_bf16
from atom.models.qwen3_5.mono.weights import LayerWeights, rope_tables
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import raise_if_given_up
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.timeline import LayerTimeline
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def baseline_refusal(experts, rows: int = 16) -> str | None:
    """Why the original path's MXFP4 routed experts do not compute what the kernel does -- MXFP4 weights
    on MXFP4 activations (A4W4, aiter's per_1x32), gate / up not interleaved, in aiter's shuffled gfx950
    layout -- on a decode of up to ``rows`` rows, or None. Asked of aiter's own activation rule, so mono
    serves only a deployment whose decode it reproduces."""
    qm = experts.quant_method
    if getattr(qm, "use_triton", False) or getattr(qm, "use_triton_decode", False):
        return "routed experts on the Triton MoE path"
    if getattr(qm, "is_guinterleave", False) or getattr(qm, "fp8_activations", False):
        return "routed experts gate / up interleaved or on FP8 activations (the kernel runs A4W4, GGUU)"
    if not getattr(experts.w13_weight, "is_shuffled", False):
        return "routed experts not in aiter's shuffled layout"
    try:
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode
    except ImportError as e:
        return f"cannot tell the original path's activation format: {e}"
    for m in range(1, rows + 1):
        got = resolve_activation_dtype(
            qm.quant_type,
            experts.w13_weight.dtype,
            activation=experts.activation,
            gate_mode=GateMode.SEPARATED,
            M=m,
            hidden_dtype=torch.bfloat16,
            has_a1_scale=getattr(experts, "w13_input_scale", None) is not None,
        )
        if got != getattr(torch, "float4_e2m1fn_x2", None):
            return f"the original path's {m}-row decode runs the routed experts on {got}, not MXFP4"
    return None


def flydsl_refusal(cfg) -> str | None:
    """Under ATOM_PA_FLYDSL, why the original decode attention is not the FlyDSL paged decode the kernel
    follows (kernels/attn_flydsl.py), or None: ATOM runs gluon instead where aiter predates the explicit
    work plan or the GPU has no FlyDSL decode, plans otherwise than the kernel, or aiter's head-256 decode
    is not the one copied (its K128 pipe up to 32 rows, aiter 59fae2078)."""
    import importlib

    from atom.model_ops import base_attention as ba

    try:
        # the module before ATOM's probe of it: until then aiter.ops.flydsl.pa_decode names the function
        importlib.import_module("aiter.ops.flydsl.pa_decode")
        from aiter.ops.flydsl.kernels.pa_decode import traits
    except ImportError:
        return "ATOM_PA_FLYDSL: this aiter has no FlyDSL paged decode"
    if not ba._flydsl_arch_supported() or not ba._flydsl_explicit_plan_api():
        return "ATOM_PA_FLYDSL: the original path runs gluon here (no FlyDSL decode with a work plan)"
    shape = {"max_partitions": FLYDSL_MAX_PARTS, "workgroup_budget": FLYDSL_BUDGET}
    if ba.flydsl_plan_shape(cfg.head_dim) != shape:
        return (
            f"ATOM_PA_FLYDSL: the original path plans {ba.flydsl_plan_shape(cfg.head_dim)}, the kernel"
            f" {shape}"
        )
    if getattr(traits, "D256_PIPE_MAX_ROWS", None) != 32:
        return "ATOM_PA_FLYDSL: this aiter's head-256 FlyDSL decode is not the one the kernel follows"
    return None


def bf16_gemm_refusal(cfg, rows: int = MAX_TOKENS) -> str | None:
    """Why one of the original path's BF16 GEMMs (``layout.bf16_gemms``) rounds its partial sums
    otherwise than the kernel does at some decode M up to ``rows``, or None. Asked of the installed
    aiter's tuned picks (``tgemm.mm``), so a re-tune, an aiter update or ``AITER_CONFIG_GEMM_BF16``
    cannot change the original path's rounding under the kernel."""
    try:
        from aiter.tuned_gemm import get_GEMM_A16W16_config
    except ImportError as e:
        return f"cannot tell the original path's BF16 GEMMs: {e}"
    for name, n, k, rule in bf16_gemms(cfg):
        for m in range(1, rows + 1):
            pick = get_GEMM_A16W16_config(
                m, n, k, False, "torch.bfloat16", "torch.bfloat16", False, False
            )
            got = aiter_bf16_rule(pick)
            want = BF16_GEMM_RULES[rule][m] if rule else ("single",)
            if got != want:
                return (
                    f"the original path's {name} GEMM ({m} x {n} x {k}) rounds as {got}, the kernel"
                    f" as {want}"
                )
    return None


def _need(i, name, t, shape, dtype=None, shuffled=None, nbytes=None):
    """Layer ``i``'s tensor ``name`` as the kernel reads it -- its shape (or byte size), dtype and,
    for a weight the kernel reads in place, whether ATOM holds it shuffled -- or MonoUnsupported.
    """
    sh = bool(getattr(t, "is_shuffled", False))
    ok = (
        tuple(t.shape) == tuple(shape)
        if nbytes is None
        else t.numel() * t.element_size() == nbytes
    )
    ok = ok and (dtype is None or t.dtype == dtype)
    ok = ok and (shuffled is None or sh == shuffled)
    if not ok:
        want = f"{nbytes} bytes" if nbytes is not None else str(tuple(shape))
        raise MonoUnsupported(
            f"layer {i} {name}: {t.dtype} {tuple(t.shape)}{' shuffled' if sh else ''}; the kernel"
            f" reads {dtype or ''} {want}{'' if shuffled is None else (' shuffled' if shuffled else ' row-major')}"
        )


def check_inputs(built, kvd, x, ins) -> None:
    """Every layer's step tensors against what its kernel addresses, before the step's first launch;
    raises ``StepInputError`` (the step keeps the original path). The KV and state pools are fixed once
    bound, so after a width's first step only one layer of each kind is checked (the step's metadata).
    """
    ops = built["ops"]
    each = [i for i, op in enumerate(ops) if op.full][:1] + [
        i for i, op in enumerate(ops) if not op.full
    ][:1]
    layers = each if built.get("pools_checked") is kvd else range(len(ops))
    for i in layers:
        try:
            ops[i].forward(x, **ins[i], check_only=True)
        except StepInputError as e:
            raise StepInputError(f"layer {i}: {e}") from None
    built["pools_checked"] = kvd


def check_splits(md, requests: int, T: int = 1, flydsl: bool = False) -> None:
    """The original decode attention splits each sequence by the count of its metadata's
    ``context_lens`` (ATOM's ``dense_decode_splits``), and each split is a rounding point: a step
    whose metadata counts other than its ``requests`` rows (a graph's padded batch around an eager
    draft) would split differently, so it raises ``StepInputError``. Under ATOM_PA_FLYDSL the split is
    FlyDSL's work plan on the first ``rows // max_seqlen_q`` sequences, which ATOM takes only while
    they divide and fit its metadata (``_flydsl_pa_decode_num_seqs``; else gluon): the kernel plans
    ``requests`` sequences of ``T`` rows, so the two must agree."""
    seqs = md.context_lens.shape[0]
    if flydsl:
        qlen = int(md.max_seqlen_q)
        if qlen != T or requests > seqs or requests > md.block_tables.shape[0]:
            raise StepInputError(
                f"the original attention plans FlyDSL on {requests * T} rows of {qlen} tokens a"
                f" sequence over {seqs} sequences, the kernel on {requests} of {T}"
            )
        return
    if pa_splits(seqs) != pa_splits(requests):
        raise StepInputError(
            f"the original attention splits {seqs} sequences {pa_splits(seqs)} ways, the kernel's"
            f" {requests} {pa_splits(requests)}"
        )


def language_model(causal_lm):
    """The module holding ``embed_tokens``, ``layers`` and ``norm``."""
    for m in causal_lm.modules():
        if all(hasattr(m, a) for a in ("embed_tokens", "layers", "norm")):
            return m
    raise MonoUnsupported("no embed_tokens / layers / norm in the model")


def raise_on_given_up_waits(variants, mailboxes, rank: int, rows: int) -> None:
    """A debug build's step (eager): raise with every mailbox wait of ``variants`` that gave up and every
    rank the step fence of ``mailboxes`` gave up on."""
    waits = [w for var in variants for w in var.given_up_waits()]
    raise_if_given_up("Qwen3.5 mono", rank, rows, waits, mailboxes)


class Qwen35MonoDecodeRunner:
    """Owns the per-layer weight views (bound at install, before the KV cache is sized) and, per
    launch width, the compiled kernels and their buffers.

    Construction is collective over the TP group: every rank binds and checks its shard, then all
    agree (``consensus.bind_agreed``), so a rank that refuses turns mono off on every rank.
    """

    def __init__(
        self, causal_lm, atom_config, spec_steps: int, replay_cap: int = 0
    ) -> None:
        tp = get_tp_group()
        self.group = tp.cpu_group
        self.tp, self.rank = tp.world_size, tp.rank_in_group
        # an MTP verify: T rows a request, the anchor and its drafted tokens
        self.T = spec_steps + 1 if spec_steps else 1
        # ReplaySSM's record buffer length (0: the snapshot GDN state)
        self.replay_cap = replay_cap
        self.debug = envs.ATOM_MONO_DEBUG
        self.max_len = atom_config.max_model_len
        bind_agreed(lambda: self._bind(causal_lm), self.group)
        # ATOM_MONO_TIMELINE: every layer's stamps, saved after the warm-up steps (eager)
        tl_prefix = envs.ATOM_MONO_TIMELINE
        self.timeline = (
            LayerTimeline(tl_prefix, len(self.layers), TL_POINTS, self.rank, self.dev)
            if tl_prefix
            else None
        )
        self.widths = WidthBuilds(
            self._build_width, self._width_kernels, self.group, "Qwen3.5 mono"
        )

    # ------------------------------------------------------------------ setup
    def _bind(self, causal_lm) -> None:
        """This rank's checks and weight views; raises ``MonoUnsupported``."""
        self.lm = language_model(causal_lm)
        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernel needs all {BLOCKS} CTAs co-resident"
            )
        # the checkpoint format, from the expert bank: FP8 (the shared expert fused into it) or MXFP4
        # (amd/Qwen3.5-397B-A17B-MXFP4: A4W4 experts, BF16 projections and shared expert)
        w13 = self.lm.layers[0].mlp.experts.w13_weight
        weights = (
            "mxfp4" if w13.dtype == getattr(torch, "float4_e2m1fn_x2", None) else "fp8"
        )
        self.cfg = Q35Config(
            tp=self.tp,
            rank=self.rank,
            weights=weights,
            qk_fusion=envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION,
            pa_flydsl=envs.ATOM_PA_FLYDSL,
        )
        # the layers' f32 residual stream and their input quant follow the fused all-reduce + RMSNorm
        from atom.models.qwen3_5.model import allreduce_norm_fusion_enabled

        if not allreduce_norm_fusion_enabled():
            raise MonoUnsupported(
                "the original path's all-reduce + RMSNorm fusion is off; the kernel reproduces it"
            )
        why = bf16_gemm_refusal(self.cfg) or (
            flydsl_refusal(self.cfg) if self.cfg.pa_flydsl else None
        )
        if why is not None:
            raise MonoUnsupported(why)
        n = len(self.lm.layers)
        self.full = [is_full_attention(i) for i in range(n)]
        for i, layer in enumerate(self.lm.layers):
            want = "full_attention" if self.full[i] else "linear_attention"
            if layer.layer_type != want:
                raise MonoUnsupported(
                    f"layer {i} is {layer.layer_type}, the kernel's plan {want}"
                )
        # per layer: (LayerWeights (shape only), the kernel's tensors: ATOM's, viewed)
        self.layers = [
            (LayerWeights(self.cfg, self.full[i]), self._pack(i, layer))
            for i, layer in enumerate(self.lm.layers)
        ]
        self.layer_nums = [
            (layer.self_attn.attn if self.full[i] else layer.linear_attn.attn).layer_num
            for i, layer in enumerate(self.lm.layers)
        ]
        rotary = next(
            l for i, l in enumerate(self.lm.layers) if self.full[i]
        ).self_attn.rotary_emb
        if not getattr(rotary, "is_neox_style", False):
            raise MonoUnsupported("RoPE not NeoX style (the kernel rotates halves)")
        rope = rope_tables(
            rotary,
            self.cfg,
            self.max_len,  # positions < max_model_len
            spare=self.T - 1,  # and the verify's drafted tokens past it
        )
        if rope is None:
            raise MonoUnsupported(
                f"the RoPE caches do not cover {self.max_len} positions"
            )
        self.cos, self.sin = rope
        logger.info("Qwen3.5 mono: rank %d bound %d layers", self.rank, n)

    def _pack(self, i, layer) -> dict:
        """The kernel's weight dict over ATOM's tensors of decoder layer ``i``."""
        cfg = self.cfg

        def f32(t):
            return t.detach().float().contiguous()

        mlp = layer.mlp
        ex = mlp.experts
        mx = cfg.weights == "mxfp4"
        E, inter, hidden = cfg.n_experts, cfg.local_inter, cfg.hidden
        model = torch.bfloat16
        _need(i, "router", mlp.gate.weight, (E + 1, hidden), model)
        for k in ("input_layernorm", "post_attention_layernorm"):
            _need(i, k, getattr(layer, k).weight, (hidden,))
        if mx:
            why = baseline_refusal(ex)
            if why is not None:
                raise MonoUnsupported(f"layer {i}: {why}")
            if ex.w13_weight.shape[0] != cfg.n_experts or mlp.shared_expert is None:
                raise MonoUnsupported(
                    f"layer {i}: MXFP4 bank {tuple(ex.w13_weight.shape)} without a BF16 shared expert"
                )
            pad = -(-(hidden // 32) // 8) * 8
            # E2M1 two a byte, E8M0 a byte, each scale row padded to 8
            _need(i, "w13", ex.w13_weight, None, nbytes=E * 2 * inter * hidden // 2)
            _need(i, "w13 scale", ex.w13_weight_scale, None, nbytes=E * 2 * inter * pad)
            _need(i, "w2", ex.w2_weight, None, nbytes=E * hidden * inter // 2)
            _need(
                i,
                "w2 scale",
                ex.w2_weight_scale,
                None,
                nbytes=E * hidden * (-(-(inter // 32) // 8) * 8),
            )
            sh = cfg.shared_inter // cfg.tp
            _need(
                i,
                "shared gate_up",
                mlp.shared_expert.gate_up_proj.weight,
                (2 * sh, hidden),
                model,
                False,
            )
            _need(
                i,
                "shared down",
                mlp.shared_expert.down_proj.weight,
                (hidden, sh),
                model,
                False,
            )
        elif (
            ex.w13_weight.dtype != torch.float8_e4m3fn
            or ex.w13_weight.shape[0] != cfg.n_experts + 1
        ):
            raise MonoUnsupported(
                f"layer {i}: routed experts {ex.w13_weight.dtype} {tuple(ex.w13_weight.shape)}, the kernel's"
                f" FP8 bank of {cfg.n_experts} + the fused shared expert"
            )
        else:
            f8, f32_ = torch.float8_e4m3fn, torch.float32
            _need(i, "w13", ex.w13_weight, (E + 1, 2 * inter, hidden), f8, True)
            _need(
                i,
                "w13 scale",
                ex.w13_weight_scale,
                (E + 1, 2 * inter // 128, hidden // 128),
                f32_,
            )
            _need(i, "w2", ex.w2_weight, (E + 1, hidden, inter), f8, True)
            _need(
                i,
                "w2 scale",
                ex.w2_weight_scale,
                (E + 1, hidden // 128, inter // 128),
                f32_,
            )
        w_r = torch.zeros(
            router_rows(cfg.n_experts),
            cfg.hidden,
            dtype=torch.bfloat16,
            device=self.dev,
        )
        # 512 experts, then the shared-expert gate row (one merged linear in both formats)
        w_r[: cfg.n_experts + 1] = mlp.gate.weight
        p = {
            "g_in1": 1.0 + f32(layer.input_layernorm.weight),
            "g_post1": 1.0 + f32(layer.post_attention_layernorm.weight),
            "w_r": pack_bf16(w_r),
        }
        if mx:
            # ATOM's bank as it holds it: gfx950 E2M1 tiles and E8M0 scales, gate / up not interleaved

            def pad8(n):
                return -(-n // 8) * 8

            p.update(
                w_ug=ex.w13_weight.data.view(torch.uint8).view(
                    E, 2 * inter, hidden // 2
                ),
                s_ug=ex.w13_weight_scale.data.view(torch.uint8).view(
                    E, 2 * inter, pad8(hidden // 32)
                ),
                w_dn=ex.w2_weight.data.view(torch.uint8).view(E, hidden, inter // 2),
                s_dn=ex.w2_weight_scale.data.view(torch.uint8).view(
                    E, hidden, pad8(inter // 32)
                ),
                w_sh_ug=mlp.shared_expert.gate_up_proj.weight,
                w_sh_dn=mlp.shared_expert.down_proj.weight,
            )
        else:
            p.update(
                w_ug=ex.w13_weight,
                s_ug=ex.w13_weight_scale,
                w_dn=ex.w2_weight,
                s_dn=ex.w2_weight_scale,
            )
        # the BF16 projections of the MXFP4 checkpoint have no scales: a placeholder in their slot
        none = torch.zeros(1, dtype=torch.float32, device=self.dev)
        # the projections: FP8 in aiter's shuffled layout with 128 x 128 scales, or (MXFP4 checkpoint)
        # BF16 row-major
        proj = torch.bfloat16 if mx else torch.float8_e4m3fn

        def need_proj(name, lin, n, k):
            _need(i, name, lin.weight, (n, k), proj, not mx)
            if not mx:
                _need(
                    i,
                    name + " scale",
                    lin.weight_scale,
                    (n // 128, k // 128),
                    torch.float32,
                )

        hd, hv = cfg.head_dim, cfg.hv
        if self.full[i]:
            a = layer.self_attn
            need_proj("qkv", a.qkv_proj, cfg.local_heads * 2 * hd + 2 * hd, hidden)
            need_proj("o_proj", a.o_proj, hidden, cfg.local_heads * hd)
            _need(i, "q_norm", a.q_norm.weight, (hd,))
            _need(i, "k_norm", a.k_norm.weight, (hd,))
            p.update(
                w_qkvz=a.qkv_proj.weight,
                s_qkvz=none if mx else a.qkv_proj.weight_scale,
                g_qn1=1.0 + f32(a.q_norm.weight),
                g_kn1=1.0 + f32(a.k_norm.weight),
                w_out=a.o_proj.weight,
                s_out=none if mx else a.o_proj.weight_scale,
            )
        else:
            a = layer.linear_attn
            if not hasattr(a, "in_proj_qkvz") or not hasattr(a, "in_proj_ba"):
                raise MonoUnsupported(
                    f"layer {i}: GDN input projection not the qkvz + b | a pair"
                )
            need_proj(
                "in_proj_qkvz", a.in_proj_qkvz, cfg.conv_dim + cfg.value_dim, hidden
            )
            need_proj("out_proj", a.out_proj, hidden, cfg.value_dim)
            _need(i, "in_proj_ba", a.in_proj_ba.weight, (2 * hv, hidden), model)
            _need(
                i, "conv1d", a.conv1d.weight, (cfg.conv_dim, 1, cfg.conv_kernel), model
            )
            _need(i, "A_log", a.A_log, (hv,))
            _need(i, "dt_bias", a.dt_bias, (hv,))
            _need(i, "norm", a.norm.weight, (cfg.linear_v_dim,))
            p.update(
                w_qkvz=a.in_proj_qkvz.weight,
                s_qkvz=none if mx else a.in_proj_qkvz.weight_scale,
                w_ba=pack_bf16(a.in_proj_ba.weight.detach()),
                w_conv=a.conv1d.weight,
                a_log=f32(a.A_log),
                dt_bias=f32(a.dt_bias),
                g_norm=f32(a.norm.weight),
                w_out=a.out_proj.weight,
                s_out=none if mx else a.out_proj.weight_scale,
            )
        for k, t in p.items():
            if k in ("w_qkvz", "w_out"):
                want = proj
            elif k in ("w_ug", "w_dn", "s_ug", "s_dn") and mx:
                want = torch.uint8
            elif k in ("w_ug", "w_dn"):
                want = torch.float8_e4m3fn
            elif k.startswith(("s_", "g_", "a_log", "dt_")):
                want = torch.float32
            else:
                want = t.dtype  # the packed BF16 tiles and the raw conv weight
            if not t.is_contiguous() or t.dtype != want:
                raise MonoUnsupported(
                    f"layer {i} {k}: {t.dtype} {tuple(t.shape)}, not the kernel's format"
                )
        return p

    def _build_width(self, S: int) -> dict:
        """The S-row launch: a variant (kernel, scratch, peer memory) per layer kind and an op per
        layer. Collective (the peer handshakes): every rank builds the same widths in the same order.
        """
        T = self.T
        self._check_records()
        # one peer buffer for both kinds (the same symmetric regions; a layer's tags are its own),
        # freed with the runner; collective over the CPU group
        dims = shape_dims(self.cfg)
        sym_bytes = layout(S, self.tp, True, tokens_per_seq=T, **dims)[1]["_bytes"]
        peers, free_peers = owned_peer_buffer(
            self, sym_bytes, self.group, self.rank, self.tp, self.dev
        )
        variants = {
            full: Qwen35Variant(
                self.cfg,
                S,
                full,
                self.rank,
                self.tp,
                self.group,
                tokens_per_seq=T,
                replay_cap=0 if full else self.replay_cap,
                debug=self.debug,
                timeline=self.timeline is not None,
                peers=peers,
            )
            for full in (False, True)
        }
        ops = [
            Qwen35MonoKernel(
                W,
                S,
                self.rank,
                self.tp,
                variant=variants[W.full],
                packed=p,
                tokens_per_seq=T,
            )
            for W, p in self.layers
        ]
        # every step zeroes the width's mailboxes (the peer buffer and both scratches), then fences
        mailboxes = StepMailboxes(
            peers, variants[False].scratch, variants[True].scratch, debug=self.debug
        )
        torch.cuda.synchronize()
        return {
            "variants": variants,
            "ops": ops,
            "mailboxes": mailboxes,
            "free_peers": free_peers,
        }

    def _check_records(self) -> None:
        """ReplaySSM's record buffers as the kernel reads them. They bind with the KV
        cache: a width built before (at install) leaves them to the step's own check
        (``check_inputs``), which also compares their shape before the first launch."""
        if not self.replay_cap:
            return
        kvd = get_forward_context().kv_cache_data
        if not kvd:
            return
        want = (self.cfg.hv, self.replay_cap, self.cfg.linear_k_dim)
        for i, full in enumerate(self.full):
            if full:
                continue
            rk = getattr(kvd[f"layer_{self.layer_nums[i]}"], "replay_buf_k", None)
            if rk is None or tuple(rk.shape[1:]) != want:
                got = None if rk is None else tuple(rk.shape)
                raise MonoUnsupported(
                    f"layer {i}: ReplaySSM records {got}, the kernel's [slots, {want}]"
                )

    @staticmethod
    def _width_kernels(built: dict) -> list:
        out = []
        for var in built["variants"].values():
            out.append((var.launch, zero_abi(var.launch)))
        return out

    def launch_width(self, rows: int) -> int | None:
        """The launch width of a ``rows``-row step, or None (not served): whole requests' rows."""
        return rows if 0 < rows <= MAX_TOKENS and rows % self.T == 0 else None

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the launch of a ``rows``-row step."""
        S = self.launch_width(rows)
        return S is not None and self.widths.prepare(S)

    def prebuild(self) -> None:
        """Every width a step can take, built now (collective, at install). A CUDA-graph deployment
        builds them while capturing; an eager one would otherwise stall a serving step on each.
        """
        for rows in range(self.T, MAX_TOKENS + 1, self.T):
            self.prepare(rows)

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The final normed hidden states [rows, hidden], as the model returns."""
        pos, ctx, slots, bt, gdn = self.step_inputs(input_ids, positions)
        x = self._layers(
            self.lm.embed_tokens(input_ids).float(), pos, ctx, slots, bt, gdn
        )
        # the layers' f32 stream is what the original path's final (fused all-reduce) norm normalizes
        return final_norm(self.lm.norm, x)

    def step_inputs(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """The step's (positions, KV lengths, cache slots, block tables, GDN state indices) as the kernel
        takes them. The KV lengths are the attention metadata's (an MRoPE position is the cache index plus
        the request's delta: RoPE takes the position, the cache range the length)."""
        fwd = get_forward_context()
        md = fwd.attn_metadata
        S = input_ids.numel()
        pos = (positions[0] if positions.dim() == 2 else positions)[:S].to(torch.int32)
        g = md.gdn_metadata
        sm = md.slot_mapping
        if sm.numel() < S:
            # GDN's verify-graph capture metadata slices the per-token slot buffer to the sequence
            # count (gdn_attn.build_for_cudagraph_capture); the buffer itself holds every token's slot
            if not (
                sm.is_contiguous()
                and sm.untyped_storage().nbytes()
                >= (sm.storage_offset() + S) * sm.element_size()
            ):
                raise StepInputError(f"slot_mapping holds {sm.numel()} of {S} rows")
            sm = sm.as_strided((S,), (1,))
        slots = sm[:S].to(torch.int32)
        T = self.T
        check_splits(md, S // T, T, self.cfg.pa_flydsl)
        ctx = md.context_lens[: S // T].to(torch.int32)
        if T == 1:
            gdn = (
                g.non_spec_state_indices_in_tensor[:S].to(torch.int32),
                g.non_spec_state_indices_tensor[:S].to(torch.int32),
            )
            bt = md.block_tables[:S]
        else:
            Q = S // T
            spec = g.spec_state_indices_tensor[:Q]
            n_acc = g.num_accepted_tokens[:Q].to(torch.int32)
            if self.replay_cap:
                # the conv window in each request's slot; the state its checkpoint + records
                conv_rows = spec[:, 0].to(torch.int32).repeat_interleave(T)
                gdn = (n_acc, conv_rows, g.slot_idx[:Q].to(torch.int32), g.write_pos)
            else:
                gdn = (n_acc, spec.reshape(-1).to(torch.int32))
            bt = md.block_tables[:Q].repeat_interleave(T, 0)
        return pos, ctx, slots, bt, gdn

    def layer_inputs(self, built, kvd, pos, ctx, slots, bt, gdn) -> list[dict]:
        """Every layer's step tensors, as its ``op.forward`` takes them (``kv`` / ``rope``, or ``gdn_state``)."""
        ins = []
        for i, op in enumerate(built["ops"]):
            lc = kvd[f"layer_{self.layer_nums[i]}"]
            if op.full:
                kv = (lc.k_cache, lc.v_cache, lc.k_scale, lc.v_scale)
                ins.append(
                    {"kv": kv + (pos, ctx, bt, slots), "rope": (self.cos, self.sin)}
                )
            else:
                state = (lc.k_cache, lc.v_cache) + gdn
                if self.replay_cap:
                    state += (lc.replay_buf_k, lc.replay_buf_u, lc.replay_buf_g)
                ins.append({"gdn_state": state})
        return ins

    def _layers(self, x, pos, ctx, slots, bt, gdn):
        """The step's rows through every layer: one launch each."""

        def tl(i):
            return self.timeline.ptr(i) if self.timeline is not None else 0

        kvd = get_forward_context().kv_cache_data
        built = self.widths[x.shape[0]]
        ins = self.layer_inputs(built, kvd, pos, ctx, slots, bt, gdn)
        check_inputs(built, kvd, x, ins)
        built["mailboxes"].begin_step()
        for i, op in enumerate(built["ops"]):
            x = op.forward(x, **ins[i], layer=i, begin=False, tl=tl(i))
        eager = not torch.cuda.is_current_stream_capturing()
        if self.debug and eager:
            raise_on_given_up_waits(
                built["variants"].values(), built["mailboxes"], self.rank, x.shape[0]
            )
        if self.timeline is not None and eager:
            self.timeline.step_done(x.shape[0])
        return x
