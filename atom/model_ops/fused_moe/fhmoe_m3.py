# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 shared expert inside the routed fused MoE (aiter FHMoE).

The Quark MXFP4 export has MXFP4 routed experts and one MXFP8 shared expert.
With ``ATOM_M3_FHMOE`` the shared expert runs as one more sorted expert of the
routed ``fused_moe`` call, from its checkpoint MXFP8 weights (no re-quantization):
the routed experts and the shared one share the step's activation block (MXFP8
or MXFP4, never bf16) and the stage-2 atomic add of the one call writes the sum.

The layer keeps one dummy final routed row (expert 128: the kernel needs
``shared_expert_id == E - 1``), the router appends the shared column
``[id 128, weight 1.0]`` and folds ``routed_scaling_factor`` into the routed
weights (the ``num_fused_shared_experts`` layout). The shared expert's weights
stay in its ``MiniMaxM3MLP``: FHMoE reads them in place, steps above
``MAX_ROWS`` rows (and mono) use the unfused production path over the first 128
rows of the routed tables.
"""

import logging

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

# The tuned FHMoE rows (aiter/configs/tuned_fhmoe.csv) cover 1..2048 rows.
MAX_ROWS = 2048
A4_FROM_ROWS = 256  # ATOM_M3_FHMOE_ACT=auto: MXFP8 activations below this many rows, MXFP4 from here (production routed is a4w4 from 256 too)
_ACTS = ("auto", "a8", "a4")
# Per-rank shape the rows are tuned for: MiniMax-M3.1 at TP4.
_HIDDEN, _INTER, _ROUTED, _TOP_K, _TP = 6144, 768, 128, 4, 4
_refused_logged = False
_notes_logged: set[str] = set()


def act_mode() -> str:
    act = envs.ATOM_M3_FHMOE_ACT
    if act not in _ACTS:
        raise ValueError(f"ATOM_M3_FHMOE_ACT={act!r}: expected one of {_ACTS}")
    return act


def pick_act(rows: int) -> str:
    """The activation dtype ('a8' MXFP8, 'a4' MXFP4) of a step of ``rows`` rows."""
    act = act_mode()
    return ("a8" if rows < A4_FROM_ROWS else "a4") if act == "auto" else act


def refusal(
    *,
    gfx: str,
    tp_size: int,
    hidden: int,
    inter: int,
    experts: int,
    top_k: int,
    n_shared: int,
    activation,
    shared_ok: bool,
    in_use: dict[str, bool],
    gu_interleave: bool,
    has_bias: bool,
    lossy_shared_flag: bool,
    triton: bool,
) -> str | None:
    """Why FHMoE cannot serve this layer (None: it can). ``in_use``: one entry per
    mode that is on and that the tuned rows do not cover (EP, DP, EPLB, ...)."""
    from aiter import ActivationType

    for mode, on in in_use.items():
        if on:
            return f"{mode} is on: FHMoE serves plain TP only"
    if lossy_shared_flag:
        return (
            "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1 asks for the lossy MXFP4 shared expert"
        )
    if gfx != "gfx950":
        return f"{gfx}: FHMoE is gfx950 only"
    if not shared_ok:
        return "the shared expert is not MXFP8 (per-1x32) next to MXFP4 routed experts"
    if activation != ActivationType.Swiglu:
        return f"activation {activation}: expected Swiglu"
    if triton:
        return "the Triton MoE is selected"
    if gu_interleave:
        return "gate/up interleaved layout (ATOM_MOE_GU_ITLV): the rows are SEPARATED"
    if has_bias:
        return "expert biases"
    shape = (tp_size, hidden, inter // tp_size, experts, top_k, n_shared)
    if shape != (_TP, _HIDDEN, _INTER, _ROUTED, _TOP_K, 1):
        return f"shape (tp, hidden, inter/rank, experts, top_k, shared) {shape}: tuned for {(_TP, _HIDDEN, _INTER, _ROUTED, _TOP_K, 1)}"
    return None


def _note_once(key: str, msg: str, *args) -> None:
    if key not in _notes_logged:
        _notes_logged.add(key)
        logger.info(msg, *args)


def log_refusal(reason: str) -> None:
    global _refused_logged
    if not _refused_logged:
        _refused_logged = True
        logger.warning(
            "ATOM_M3_FHMOE=1 refused, keeping the unfused shared expert: %s", reason
        )


class FhmoeM3:
    """One MoE layer's FHMoE state: the shared expert MLP it reads, built lazily."""

    def __init__(self, layer):
        self.layer = layer
        _note_once(
            "on",
            "ATOM_M3_FHMOE=1: MXFP8 shared expert fused into the routed MoE "
            "(129 experts / top-5, ATOM_M3_FHMOE_ACT=%s, up to %d rows a step)",
            act_mode(),
            MAX_ROWS,
        )
        self.shared_mlp = None
        self._fh = None  # (shared w1, w2, w1 scale, w2 scale), FHMoE layout
        self._routed = None  # the first 128 rows of the routed tables

    def bind(self, shared_mlp) -> None:
        self.shared_mlp = shared_mlp

    def fill_dummy_row(self) -> None:
        """The final routed row (expert 128), before the shuffle: zero weights, E8M0 1.0 scales."""
        layer = self.layer
        for w in (layer.w13_weight, layer.w2_weight):
            w.data[_ROUTED:].view(torch.uint8).zero_()
        for s in (layer.w13_weight_scale, layer.w2_weight_scale):
            s.data[_ROUTED:].view(torch.uint8).fill_(0x7F)

    def _build(self) -> None:
        """After loading (the model runner's warmup runs before any graph capture)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FHMoE tables must be built before graph capture")
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_scale

        layer, mlp = self.layer, self.shared_mlp
        method = layer.quant_method
        assert method.hidden_pad == 0 and method.intermediate_pad == 0
        gate_up, down = mlp.gate_up_proj, mlp.down_proj
        w1 = gate_up.weight.data.view(1, 2 * _INTER, _HIDDEN)  # (16,16)-preshuffled
        w2 = down.weight.data.view(1, _HIDDEN, _INTER)
        for w in (w1, w2):
            w.is_shuffled = True
        e8m0 = dtypes.fp8_e8m0
        s1 = gate_up.weight_scale.data.view(e8m0).view(2 * _INTER, _HIDDEN // 32)
        s2 = down.weight_scale.data.view(e8m0).view(_HIDDEN, _INTER // 32)
        self._fh = (w1, w2, shuffle_scale(s1), shuffle_scale(s2))

        def prefix(t):  # the 128-expert tables are a prefix of the 129-expert ones
            n = t.numel() // (_ROUTED + 1) * _ROUTED
            return t.view(-1)[:n].view(-1, t.shape[-1])

        w13, w2r = layer.w13_weight.data[:_ROUTED], layer.w2_weight.data[:_ROUTED]
        for w in (w13, w2r):
            w.is_shuffled = True  # as the 129-row parameters; no gate_up_interleave tag
        self._routed = (
            w13,
            w2r,
            prefix(layer.w13_weight_scale.data),
            prefix(layer.w2_weight_scale.data),
        )

    @property
    def routed_tables(self):
        """(w13, w2, w13 scale, w2 scale) of the 128 routed experts."""
        if self._routed is None:
            self._build()
        return self._routed

    def run(self, x, topk_weights, topk_ids, num_local_tokens):
        """The routed + shared MoE of ``x`` in one fused_moe call; None above
        MAX_ROWS rows (the caller runs the unfused path)."""
        if x.shape[0] > MAX_ROWS:
            _note_once(
                "big",
                "FHMoE: steps above %d rows use the unfused routed + shared path",
                MAX_ROWS,
            )
            return None
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe

        if self._fh is None:
            self._build()
        layer = self.layer
        sw1, sw2, ss1, ss2 = self._fh
        return fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            num_local_tokens=num_local_tokens,
            dtype=x.dtype,
            swiglu_limit=layer.swiglu_limit,
            gate_mode="separated",
            quant_dtype_a=dtypes.fp8 if pick_act(x.shape[0]) == "a8" else dtypes.fp4x2,
            shared_w1=sw1,
            shared_w2=sw2,
            shared_w1_scale=ss1,
            shared_w2_scale=ss2,
            shared_expert_id=_ROUTED,
        )

    def add_shared(self, routed_out, x):
        """The unfused path's shared expert, added to the routed rows' output."""
        return routed_out + self.shared_mlp(x)
