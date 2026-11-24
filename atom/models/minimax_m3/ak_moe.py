# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1's MoE block through amd-kernels (``amd_kernels.moe.mxfp4_h6144_i768_e128_top4``), opt-in by
``ATOM_M3_AK_MOE=1`` for the rows ``ATOM_M3_AK_MOE_MIN`` .. ``ATOM_M3_AK_MOE_MAX`` (others keep the aiter path;
by default every row, and the routed w13 is then held once, in the entry's layout).

One graph-capturable call per layer and step: split-K router on the checkpoint's FP32 gate, biased top-4 with the
routed scale folded in, MXFP4 routed experts (MXFP4 activations at every M), the MXFP8 shared expert on a side
stream, slots + shared summed to BF16 before ATOM's all-reduce. The entry takes the weights as ATOM holds them after
its own preshuffle: routed w2 values and scales and the shared expert's values are the same bytes (aiter's (16, 16)
shuffle), w13 is ATOM's with its 16-row gate / up blocks interleaved, the w13 and shared scales are re-laid copies.
"""

import logging

import torch
from torch import nn

from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")

H, I, E = 6144, 768, 128
MAX_ROWS = 87381  # the entry's largest M
_LAYERS: dict[str, "AkMoe"] = {}
_warm = False


def enabled() -> bool:
    return envs.ATOM_M3_AK_MOE


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.data.view(torch.uint8)


class AkMoe(nn.Module):
    """The entry's weights for one ``MiniMaxM3MoE``, registered as its last child: the loader calls
    ``process_weights_after_loading`` parent-first, children in registration order, so this one runs after the
    experts' and the shared expert's own processing."""

    def __init__(self, moe: nn.Module, prefix: str):
        super().__init__()
        self.prefix = prefix
        try:
            import amd_kernels.moe.mxfp4_h6144_i768_e128_top4  # noqa: F401
        except ImportError as e:
            raise ImportError("ATOM_M3_AK_MOE=1 needs inferact/amd-kernels on PYTHONPATH") from e
        self.rows = (envs.ATOM_M3_AK_MOE_MIN, min(envs.ATOM_M3_AK_MOE_MAX, MAX_ROWS))
        self._moe = [moe]  # not a submodule
        self.weights = None
        self.shared_full = None  # ATOM_P1_SP_SHARED_LOCAL: the whole shared expert (p1_sls_shared.SharedFull)
        self._gate_fp32 = None
        _LAYERS[prefix] = self
        # the gate parameter keeps BF16 (the aiter router GEMM and mono read it); the entry's router reads FP32
        gate = moe.gate.weight
        load = gate.weight_loader

        def gate_loader(param, loaded_weight, *args, **kwargs):
            self._gate_fp32 = loaded_weight.detach().to(param.device, torch.float32, copy=True)
            return load(param, loaded_weight, *args, **kwargs)

        gate.weight_loader = gate_loader

    def process_weights_after_loading(self):
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.i3072.gemm1_prefill_h6144_i3072_persistent import (
            interleave_w13_scales,
        )
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.prefill_h6144_i768 import Mxfp4PrefillScales

        moe = self._moe[0]
        ex, sh = moe.experts, moe.shared_experts
        why = self._unsupported(moe)
        if why is not None:
            logger.warning("ATOM_M3_AK_MOE: %s keeps the aiter path: %s", self.prefix, why)
            return
        # W13 [E, 2 (gate, up), I/16, 16-row block] -> [E, I/16, 2, block]
        w13 = _u8(ex.w13_weight).view(E, 2, I // 16, -1).transpose(1, 2).contiguous()
        if self.rows == (1, MAX_ROWS) and envs.ATOM_M3_AK_MOE_SHARE_W13:
            # every row runs here, so the aiter kernels never read w13 again: one copy, in the entry's layout
            # (mono reads it too, ``w13_gugu``)
            ex.w13_weight.data = w13.view(ex.w13_weight.dtype).view(ex.w13_weight.shape)
            ex.w13_weight.w13_gugu = True
        w13 = w13.view(E, I // 16, 2, H // 128, 4, 16, 16)
        w2 = _u8(ex.w2_weight).view(E, H // 16, I // 128, 4, 16, 16)
        scales = Mxfp4PrefillScales(interleave_w13_scales(_u8(ex.w13_weight_scale).view(E, 2 * I, H // 32)),
                                    _u8(ex.w2_weight_scale).view(E, H, I // 32))
        gu, dn = sh.gate_up_proj, sh.down_proj
        shared = SharedWeights.from_row_major(gu.weight.data, _u8(gu.weight_scale), dn.weight.data,
                                              _u8(dn.weight_scale))
        shared.w13 = _u8(gu.weight).view(shared.w13.shape)
        shared.w2 = _u8(dn.weight).view(shared.w2.shape)
        gate = self._gate_fp32
        self._gate_fp32 = None
        self.weights = MoeWeights(gate, moe.e_score_correction_bias.data, w13, scales.w13, w2, scales.w2, scales,
                                  float(moe.routed_scaling_factor), router_weight_lines=router_line_layout(gate),
                                  shared_weights=shared)
        if envs.ATOM_P1_SP_SHARED_LOCAL and envs.ATOM_M3_SP_PREFILL:
            # only the SP prefill hand-off reads it (+3.1 GiB per GPU)
            self.shared_full = _gather_shared_full(shared)
        global _warm
        if not _warm:
            if self.shared_full is not None:
                import p1_sls_shared

                p1_sls_shared.warm_up(self.shared_full)
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import warm_up_mxfp4_moe

            warm_up_mxfp4_moe(self.weights)
            if envs.ATOM_P1_MOE_TILES:
                from atom.models.minimax_m3.ak_moe_tiles import warm_up as warm_up_tiles

                warm_up_tiles(self.weights)
            if envs.ATOM_P1_MBF_BACK:
                from atom.models.minimax_m3.mbf_back import warm_up as warm_up_back

                warm_up_back(self.weights, moe_tiles=envs.ATOM_P1_MOE_TILES)
            _warm = True

    def _unsupported(self, moe) -> str | None:
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import SWIGLU_OAI

        ex, sh = moe.experts, moe.shared_experts
        qm = ex.quant_method
        if self._gate_fp32 is None or tuple(self._gate_fp32.shape) != (E, H):
            return "no FP32 [128, 6144] router weight loaded"
        if sh is None or ex.num_fused_shared_experts or moe.e_score_correction_bias is None:
            return "needs the unfused MXFP8 shared expert and the routing bias"
        if getattr(qm, "is_guinterleave", True) or getattr(qm, "intermediate_pad", 1) or getattr(qm, "hidden_pad", 1):
            return "needs aiter's GGUU (16, 16) preshuffle, unpadded"
        if not (getattr(ex.w13_weight, "is_shuffled", False) and getattr(ex.w2_weight, "is_shuffled", False)):
            return "routed weights not preshuffled"
        if (_u8(ex.w13_weight).numel() != E * 2 * I * H // 2 or _u8(ex.w2_weight).numel() != E * H * I // 2
                or _u8(ex.w13_weight_scale).numel() != E * 2 * I * H // 32
                or _u8(ex.w2_weight_scale).numel() != E * H * I // 32):
            return "routed expert shapes"
        if (sh.gate_up_proj.weight.dtype != torch.float8_e4m3fn
                or tuple(sh.gate_up_proj.weight.shape) != (2 * I, H)
                or tuple(sh.down_proj.weight.shape) != (H, I)
                or _u8(sh.gate_up_proj.weight_scale).numel() != 2 * I * H // 32
                or _u8(sh.down_proj.weight_scale).numel() != H * I // 32):
            return "shared expert must be MXFP8 [1536, 6144] / [6144, 768]"
        beta = 1.0 if sh.swiglu_beta is None else sh.swiglu_beta
        swiglu = (sh.swiglu_alpha, sh.swiglu_limit, beta)
        if any(abs(a - b) > 1e-6 for a, b in zip(swiglu, SWIGLU_OAI)) or ex.swiglu_limit != SWIGLU_OAI[1]:
            return f"SwiGLU-OAI {swiglu}"
        if moe.routed_scaling_factor != 2.0:
            return f"routed scale {moe.routed_scaling_factor}"
        return None

    def runs(self, rows: int) -> bool:
        return self.weights is not None and self.rows[0] <= rows <= self.rows[1]


def _gather_shared_full(shared):
    """ATOM_P1_SP_SHARED_LOCAL: the full shared expert on every TP4 rank, assembled from the four ranks' shards
    (all-gathered bytes in the entry's layout; a permutation of the checkpoint, no re-quantization). A collective:
    every rank loads the same layers in the same order."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size != 4:
        return None
    try:
        import p1_sls_shared
    except ImportError as e:
        raise ImportError("ATOM_P1_SP_SHARED_LOCAL=1 needs /shared_nfs/ygan/p1/sp-local-shared/overlay on PYTHONPATH") from e
    parts = []
    for t in (shared.w13, shared.s13, shared.w2, shared.s2):
        t = t.view(torch.uint8).contiguous()
        out = torch.empty((4, *t.shape), dtype=torch.uint8, device=t.device)
        dist.all_gather_into_tensor(out, t, group=tp.device_group)
        parts.append(list(out.unbind(0)))
    return p1_sls_shared.SharedFull.from_tp4_shards(*parts)


def ak_moe_forward(hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
    """The MoE block's TP-partial output (routed * scale + shared) of normed BF16 rows."""
    ak = _LAYERS[layer_name]
    rows = hidden_states.shape[0]
    if rows == 0:
        return torch.empty_like(hidden_states)
    if not ak.runs(rows):
        return ak._moe[0].aiter_forward(hidden_states)
    if envs.ATOM_P1_MBF_BACK:
        from atom.models.minimax_m3 import mbf_back

        if mbf_back.supported(rows):
            out = mbf_back.mxfp4_moe_back(hidden_states.contiguous(), ak.weights, moe_tiles=envs.ATOM_P1_MOE_TILES,
                                          mode="data")
            if out is not None:  # None: no buffer set left for a new stream under capture -> production path
                return out
    if envs.ATOM_P1_MOE_TILES:
        from atom.models.minimax_m3 import ak_moe_tiles

        if ak_moe_tiles.supported(rows):
            return ak_moe_tiles.mxfp4_moe_tiles(hidden_states.contiguous(), ak.weights)
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe

    return mxfp4_moe(hidden_states.contiguous(), ak.weights)


def _ak_moe_forward_fake(hidden_states: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


ak_moe_op = direct_register_custom_op(
    op_name="minimax_m3_ak_moe",
    op_func=ak_moe_forward,
    mutates_args=[],
    fake_impl=_ak_moe_forward_fake,
)
