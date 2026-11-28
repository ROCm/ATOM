# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode-shape MoE block through aiter's FlyDSL ``mega_moe_tp`` decode ops.

Opt-in (``ATOM_M3_DECODE_MOE=1``, default 0) for 8-128 rows per step; every other row count, and any aiter
without the ops, keeps the existing ``gate`` + ``FusedMoE`` + shared-expert path. One graph-capturable custom op
per layer and step runs:

1. ``route_quant_small``: split-K router GEMM on the checkpoint's FP32 gate, biased sigmoid top-4 with the routed
   scale folded into the weights, MXFP4 (routed) and MXFP8 (shared expert) quant of the rows;
2. ``decode_expert_gemms``: routed GEMM1 (+ SwiGLU-OAI + MXFP4 quant) and GEMM2 (+ top-k weighting), the shared
   expert folded into both launches, no sort launch (GEMM1 builds its expert blocks from the top-k ids);
3. ``moe_slots_shared_reduce``: the four weighted slot rows plus the shared expert summed to BF16, the TP-partial
   output ``MiniMaxM3MoE.forward`` returns (ATOM's all-reduce follows as usual).

The ops are ``aiter.ops.flydsl.kernels.mega_moe_tp.{decode_router_chain, decode_expert_gemms}`` and
``aiter.ops.flydsl.kernels.moe_slots_shared_reduce``. The kernels read the weights in their own layouts: routed w2
values and scales and the shared expert's values are the bytes ATOM already holds (aiter's (16, 16) preshuffle);
the routed w13 values are read in place when this aiter's ``get_plan`` takes ``w13_split`` (else a copy with the
gate / up 16-row blocks interleaved is kept) and the routed w13 and the shared scales are re-laid copies (built once after weight loading, before graph capture).
"""

import logging

import torch
from torch import nn

from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")

H, I, E, TOPK = 6144, 768, 128, 4
ROWS = (8, 128)  # the expert tiles' row range (the router and reduce cover 1-128)
SWIGLU_OAI = (
    1.702,
    7.0,
    1.0,
)  # alpha, limit, beta (up + beta), as the kernels compute it
ROUTED_SCALE = 2.0  # folded into the top-k weights by the router kernel

_LAYERS: dict[str, "DecodeMoe"] = {}
_logged_extra = False


def enabled() -> bool:
    return envs.ATOM_M3_DECODE_MOE


def load_ops():
    """The aiter decode ops, or None when this aiter does not have all of them."""
    try:
        from aiter.ops.flydsl.kernels.mega_moe_tp import (
            decode_expert_gemms,
            decode_router_chain,
        )
        from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import (
            router_line_layout,
        )
        from aiter.ops.flydsl.kernels.moe_slots_shared_reduce import (
            moe_slots_shared_reduce,
        )
    except ImportError:
        return None
    return (
        decode_router_chain,
        decode_expert_gemms,
        router_line_layout,
        moe_slots_shared_reduce,
    )


def split_layout_supported(gemms) -> bool:
    """Whether this aiter's ``get_plan`` takes ``w13_split`` (the GEMM1 tile reads the routed w13 in the
    preshuffled [E, 2, I/16, block] layout, so no re-laid copy of it is needed)."""
    import inspect

    try:
        return "w13_split" in inspect.signature(gemms.get_plan).parameters
    except (TypeError, ValueError):
        return False


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.data.view(torch.uint8)


def routed_w13_layout(w13: torch.Tensor, e: int = E, i: int = I) -> torch.Tensor:
    """ATOM's preshuffled routed w13 bytes [E, 2, I/16, ...] (gate / up halves) -> [E, I/16, 2, ...]: the 16-row
    gate and up blocks interleaved, as the GEMM1 tile reads them."""
    return w13.view(e, 2, i // 16, -1).transpose(1, 2).contiguous()


def routed_w13_scale_layout(w13_scale: torch.Tensor) -> torch.Tensor:
    """aiter's e8m0-shuffled routed w13 scales [E, 2I, H/32] (gate rows, then up rows) -> the gate / up interleaved
    shuffle GEMM1 reads. Undoes the shuffle to row-major, then shuffles with ``is_guinterleave``.
    """
    from aiter.ops.shuffle import shuffle_scale

    e, n, k32 = w13_scale.shape
    t = w13_scale.reshape(e * n // 32, k32 // 8, 4, 16, 2, 2)
    raw = t.permute(0, 5, 3, 1, 4, 2).reshape(e, n, k32)
    return shuffle_scale(
        raw.reshape(e * n, k32), e, is_guinterleave=True, gate_up=True
    ).view(torch.uint8)


def shared_scale_layouts(s13: torch.Tensor, s2: torch.Tensor):
    """Row-major e8m0 scales of the shared expert's gate_up [1536, 192] and down [6144, 24] -> the tile layouts
    (bit-exact permutations)."""
    b13 = s13.view(torch.uint8).reshape(48, 2, 16, 24, 2, 4).permute(0, 3, 5, 2, 4, 1)
    b2 = s2.view(torch.uint8).reshape(192, 2, 16, 3, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return b13.contiguous(), b2.contiguous()


class DecodeMoe(nn.Module):
    """The decode ops' weights for one ``MiniMaxM3MoE``, registered as its last child: the loader calls
    ``process_weights_after_loading`` parent-first, children in registration order, so this one runs after the
    experts' and the shared expert's own processing."""

    def __init__(self, moe: nn.Module, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.ops = load_ops()
        if self.ops is None:
            logger.warning(
                "ATOM_M3_DECODE_MOE: this aiter lacks the mega_moe_tp decode ops; %s keeps the default MoE path",
                prefix,
            )
        self.rows = ROWS
        self._moe = [moe]  # not a submodule
        self.ready = False
        self._gate_fp32 = None
        _LAYERS[prefix] = self
        # the gate parameter keeps BF16 (the default path reads it); the router kernel reads the FP32 checkpoint
        gate = moe.gate.weight
        load = gate.weight_loader

        def gate_loader(param, loaded_weight, *args, **kwargs):
            self._gate_fp32 = loaded_weight.detach().to(
                param.device, torch.float32, copy=True
            )
            return load(param, loaded_weight, *args, **kwargs)

        gate.weight_loader = gate_loader

    def process_weights_after_loading(self):
        if self.ops is None:
            return
        moe = self._moe[0]
        why = self._unsupported(moe)
        if why is not None:
            logger.warning(
                "ATOM_M3_DECODE_MOE: %s keeps the default MoE path: %s",
                self.prefix,
                why,
            )
            self._gate_fp32 = None
            return
        router_line_layout = self.ops[2]
        ex, sh = moe.experts, moe.shared_experts
        gate = self._gate_fp32
        self._gate_fp32 = None
        self.gate = gate.contiguous()
        self.gate_lines = router_line_layout(self.gate)
        self.bias = moe.e_score_correction_bias.data
        # an aiter whose GEMM1 tile reads the preshuffled [E, 2, I/16, block] layout in place needs no second w13
        self.plan_kw = (
            {"w13_split": True} if split_layout_supported(self.ops[1]) else {}
        )
        self.w13 = (
            _u8(ex.w13_weight)
            if self.plan_kw
            else routed_w13_layout(_u8(ex.w13_weight))
        )
        self.s13 = routed_w13_scale_layout(
            _u8(ex.w13_weight_scale).view(E, 2 * I, H // 32)
        )
        self.w2 = _u8(ex.w2_weight).contiguous()
        self.s2 = _u8(ex.w2_weight_scale).contiguous()
        gu, dn = sh.gate_up_proj, sh.down_proj
        self.sw13 = _u8(gu.weight).contiguous()
        self.sw2 = _u8(dn.weight).contiguous()
        self.ss13, self.ss2 = shared_scale_layouts(
            _u8(gu.weight_scale), _u8(dn.weight_scale)
        )
        global _logged_extra
        if not _logged_extra:
            _logged_extra = True
            extra = self.s13.numel() + (0 if self.plan_kw else self.w13.numel())
            logger.info(
                "ATOM_M3_DECODE_MOE: %s; holds %.0f MiB per MoE layer and rank besides the default path's weights",
                (
                    "routed w13 read in place (aiter w13_split)"
                    if self.plan_kw
                    else "this aiter cannot read w13 in place: a second re-laid w13 copy is kept"
                ),
                extra / 2**20,
            )
        self.ready = True

    def _unsupported(self, moe) -> str | None:
        ex, sh = moe.experts, moe.shared_experts
        qm = ex.quant_method
        if self._gate_fp32 is None or tuple(self._gate_fp32.shape) != (E, H):
            return "no FP32 [128, 6144] router weight loaded"
        if (
            sh is None
            or ex.num_fused_shared_experts
            or moe.e_score_correction_bias is None
        ):
            return "needs the unfused MXFP8 shared expert and the routing bias"
        if (
            getattr(qm, "is_guinterleave", True)
            or getattr(qm, "intermediate_pad", 1)
            or getattr(qm, "hidden_pad", 1)
        ):
            return "needs aiter's GGUU (16, 16) preshuffle, unpadded"
        if not (
            getattr(ex.w13_weight, "is_shuffled", False)
            and getattr(ex.w2_weight, "is_shuffled", False)
        ):
            return "routed weights not preshuffled"
        if (
            _u8(ex.w13_weight).numel() != E * 2 * I * H // 2
            or _u8(ex.w2_weight).numel() != E * H * I // 2
            or _u8(ex.w13_weight_scale).numel() != E * 2 * I * H // 32
            or _u8(ex.w2_weight_scale).numel() != E * H * I // 32
        ):
            return "routed expert shapes"
        if (
            sh.gate_up_proj.weight.dtype != torch.float8_e4m3fn
            or tuple(sh.gate_up_proj.weight.shape) != (2 * I, H)
            or sh.down_proj.weight.dtype != torch.float8_e4m3fn
            or tuple(sh.down_proj.weight.shape) != (H, I)
            or tuple(_u8(sh.gate_up_proj.weight_scale).shape) != (2 * I, H // 32)
            or tuple(_u8(sh.down_proj.weight_scale).shape) != (H, I // 32)
        ):
            return "shared expert must be MXFP8 [1536, 6144] / [6144, 768] with row-major scales"
        if not (
            getattr(sh.gate_up_proj.weight, "is_shuffled", False)
            and getattr(sh.down_proj.weight, "is_shuffled", False)
        ):
            # ATOM_GROUP32_WEIGHT_PRESHUFFLE=0 leaves them row-major; the kernels read (16, 16) preshuffle
            return "shared expert weights not preshuffled"
        if getattr(ex, "use_ep", False):
            return (
                "expert parallelism (the kernels read all 128 experts, no EP routing)"
            )
        if ex.top_k != TOPK or ex.scoring_func != "sigmoid":
            return f"routing top_k={ex.top_k} scoring_func={ex.scoring_func!r}"
        beta = 1.0 if sh.swiglu_beta is None else sh.swiglu_beta
        swiglu = (sh.swiglu_alpha, sh.swiglu_limit, beta)
        if (
            any(v is None for v in (*swiglu, ex.swiglu_limit))
            or any(abs(a - b) > 1e-6 for a, b in zip(swiglu, SWIGLU_OAI))
            or abs(ex.swiglu_limit - SWIGLU_OAI[1]) > 1e-6
        ):
            return f"SwiGLU-OAI {swiglu}"
        if moe.routed_scaling_factor != ROUTED_SCALE:
            return f"routed scale {moe.routed_scaling_factor}"
        return None

    def release_weight_memory(self) -> None:
        """Weight-discard sleep: the kernel-layout views and copies and the FP32 gate copy pin the old weight
        storage outside the Parameters, so drop them. The layer then keeps the default MoE path for good (the FP32
        gate is not a Parameter and cannot be rebuilt after the weights are synced in).
        """
        was_ready, self.ready = self.ready, False
        for n in (
            "gate",
            "gate_lines",
            "bias",
            "w13",
            "s13",
            "w2",
            "s2",
            "sw13",
            "sw2",
            "ss13",
            "ss2",
        ):
            if hasattr(self, n):
                setattr(self, n, None)
        self._gate_fp32 = None
        if was_ready:
            logger.warning(
                "ATOM_M3_DECODE_MOE: %s: weights were discarded by sleep, keeping the default MoE path",
                self.prefix,
            )

    def resume_weight_memory(self) -> None:
        """Nothing to rebuild: the layer stays on the default path (see ``release_weight_memory``)."""
        self.ready = False

    def runs(self, rows: int) -> bool:
        return self.ready and self.rows[0] <= rows <= self.rows[1]


def decode_moe_forward(hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
    """The MoE block's TP-partial output (routed * scale + shared) of normed BF16 rows."""
    dm = _LAYERS[layer_name]
    rows = hidden_states.shape[0]
    if rows == 0:
        return torch.empty_like(hidden_states)
    if not dm.runs(rows):
        return dm._moe[0].default_forward(hidden_states)
    router, gemms, _, reduce = dm.ops
    x = hidden_states.contiguous()
    routed = router.route_quant_small(x, dm.gate, dm.gate_lines, dm.bias, ROUTED_SCALE)
    if routed is None:  # a row layout the router kernels do not cover
        return dm._moe[0].default_forward(hidden_states)
    (xq, xs), (q8, s8), weights, ids = routed
    plan = gemms.get_plan(rows, **dm.plan_kw)
    slots, shared = gemms.decode_expert_gemms(
        plan,
        xq.view(-1),
        xs.view(-1),
        q8.view(-1),
        s8.view(-1),
        (dm.w13, dm.s13, dm.w2, dm.s2),
        (dm.sw13, dm.ss13, dm.sw2, dm.ss2),
        ids,
        weights,
    )
    return reduce(slots, shared)


def _decode_moe_forward_fake(
    hidden_states: torch.Tensor, layer_name: str
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


decode_moe_op = direct_register_custom_op(
    op_name="minimax_m3_decode_moe",
    op_func=decode_moe_forward,
    mutates_args=[],
    fake_impl=_decode_moe_forward_fake,
)
