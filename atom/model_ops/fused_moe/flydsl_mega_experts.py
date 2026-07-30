# SPDX-License-Identifier: Apache-2.0
"""FlyDSL MegaMoE (ghu_moe_stage1) fused EP-MoE integration for ATOM.

``moe_backend="mega"`` replaces the whole EP experts step (dispatch + GEMM1 +
quant + GEMM2 + combine) with the new-PR ``MegaMoE`` single op:
  stage1 = FusedMoEMegaStage1 (single-launch dispatch + GEMM1, atom-logical a2)
  stage2 = FlyDSLMoeGemm2CombineOp (GEMM2 epilogue inlines EP combine)

Large-bs (prefill) works via FUSED_MEGA_COMPACT_ATOM=1 (compact dispatch +
atom-logical a2; stage2 unchanged). Validated bit-exact in FlyDSL notes
(2026-06-16) for v4_flash a8w4 bs 8..8192.

Memory: ONE MegaMoE is shared across all MoE layers (process-level cache keyed
by shape/quant/mtpr/tile); per-layer weights are swapped in before forward
(stage1.w1 / self.w2 are runtime pointer args, not baked into the kernel).
"""

from __future__ import annotations

import logging
import os
import sys

import torch

# compact+atom combo must be on so large bs (prefill) doesn't trip the
# atom_contract-vs-compact assert in FusedMoEMegaStage1.
os.environ.setdefault("FUSED_MEGA_COMPACT_ATOM", "1")
# ATOM EP holds only this rank's epr experts' weights, so index w1/bias by LOCAL
# expert id (kernel patch in fused_moe_gemm_2stage.py). We pass local-epr w1.
os.environ.setdefault("FUSED_MEGA_W1_LOCAL", "1")

logger = logging.getLogger("atom")

_FLYDSL_KERNELS_PATH = os.environ.get("ATOM_FLYDSL_KERNELS_PATH", "/home/yashao/FlyDSL")
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_MEGA_CACHE: dict = {}
_MEGA_BUILD_DBG = False


def _os_env(k):
    return os.environ.get(k, "<unset>")


def _ensure_path():
    if _FLYDSL_KERNELS_PATH and _FLYDSL_KERNELS_PATH not in sys.path:
        sys.path.insert(0, _FLYDSL_KERNELS_PATH)


def _shuffle_fns():
    """The EXACT shuffle ops MegaMoE's reference uses (FlyDSL tests.utils /
    tests.kernels.utils), NOT aiter's shuffle_weight."""
    _ensure_path()
    # FlyDSL mega_moe_v1 refactor #807 removed tests/kernels/utils/fp4_utils.py and
    # folded e8m0_shuffle into gemm_common_utils. Alias it so downstream
    # `fp4_utils.e8m0_shuffle(...)` keeps working against the current-tip kernels
    # (weight-prep MUST match the MegaMoE commit we actually run).
    from tests.kernels.utils import gemm_common_utils as fp4_utils
    from tests.utils import shuffle_weight

    return shuffle_weight, fp4_utils


def build_mega_weights(layer) -> None:
    """From ATOM's RAW (pre-atom-shuffle) mxfp4 w13/w2 + e8m0 scales, build the
    MegaMoE-layout weights and stash on the layer. Must run BEFORE atom's own
    shuffle_weight in process_weights_after_loading (uses raw layout)."""
    shuffle_weight, fp4_utils = _shuffle_fns()

    w13 = layer.w13_weight.data  # [E_local, 2*inter, hidden//2] fp4-packed uint8
    E, two_inter, _h_half = w13.shape
    # stash local expert count NOW (from raw shape) — run_mega_moe must not re-infer
    # it from the shuffled _mega_w1 (shape changes with shuffle_weight_w4/.view(-1)).
    layer._mega_local_E = int(E)
    # a8w4 MegaMoE defaults to gate_mode=INTERLEAVE(g1u1): w1/w1_scale MUST be
    # gate-up interleave-shuffled with shuffle_weight_w4/shuffle_scale_w4(gate_up=True),
    # NOT the generic shuffle_weight/e8m0_shuffle (that is the SEPARATED layout, which
    # pairs gate/up wrong in swiglu -> garbage output, gsm8k~0). Matches the FlyDSL
    # reference tests/kernels/test_mega_moe.py (w_kernel_gui path). w2 has no gate/up
    # so it keeps the generic shuffle below.
    w13_fp4 = w13.view(_FP4) if _FP4 is not None else w13  # [E, 2*inter, hidden//2]
    layer._mega_w1 = (
        fp4_utils.shuffle_weight_w4(w13_fp4, NLane=16, gate_up=True, moe_gemm=True)
        .view(torch.uint8)
        .contiguous()
        .view(-1)
    )

    s1 = layer.w13_weight_scale.data  # [E, 2*inter, hidden//32]
    s1f = s1.reshape(E * two_inter, s1.shape[2])
    layer._mega_w1_scale = (
        fp4_utils.shuffle_scale_w4(s1f, experts_cnt=E, gate_up=True)
        .view(torch.uint8)
        .contiguous()
        .view(-1)
    )

    w2 = layer.w2_weight.data  # [E_local, hidden, inter//2] fp4-packed uint8
    E2, hh, i_half = w2.shape
    w2f = w2.reshape(E2 * hh, i_half)
    if _FP4 is not None:
        w2f = w2f.view(_FP4)
    layer._mega_w2 = shuffle_weight(w2f).view(torch.uint8).contiguous().view(-1)

    s2 = layer.w2_weight_scale.data
    s2f = s2.reshape(E2 * s2.shape[1], s2.shape[2])
    layer._mega_w2_scale = (
        fp4_utils.e8m0_shuffle(s2f).view(torch.uint8).contiguous().view(-1)
    )

    global _MEGA_BUILD_DBG
    if not _MEGA_BUILD_DBG:
        _MEGA_BUILD_DBG = True
        _w13b = getattr(layer, "w13_bias", None)
        _w2b = getattr(layer, "w2_bias", None)
        logger.info(
            f"[MEGA-BUILD] w13={tuple(w13.shape)}{w13.dtype} w13_scale={tuple(s1.shape)}{s1.dtype} "
            f"w2={tuple(w2.shape)}{w2.dtype} w2_scale={tuple(s2.shape)}{s2.dtype} | "
            f"_mega_w1={tuple(layer._mega_w1.shape)} _mega_w1_scale={tuple(layer._mega_w1_scale.shape)} "
            f"_mega_w2={tuple(layer._mega_w2.shape)} _mega_w2_scale={tuple(layer._mega_w2_scale.shape)} | "
            f"w13_bias={None if _w13b is None else tuple(_w13b.shape)} "
            f"w2_bias={None if _w2b is None else tuple(_w2b.shape)} "
            f"GU_ITLV={_os_env('ATOM_MOE_GU_ITLV')}"
        )


def get_or_build_mega_moe(
    *,
    rank,
    world_size,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    mtpr,
    w1,
    w1_scale,
    w2,
    w2_scale,
    gemm2_tile=(-1, -1, -1),
):
    key = (
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        mtpr,
        gemm2_tile,
    )
    m = _MEGA_CACHE.get(key)
    if m is None:
        _ensure_path()
        from kernels.mega_moe import MegaMoE

        tm, tn, tk = gemm2_tile
        m = MegaMoE(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=mtpr,
            gemm2_tile_m=tm,
            gemm2_tile_n=tn,
            gemm2_tile_k=tk,
            gemm2_persist_m=-1,
            # Both stages fused: the whole point of the mega backend. mega_moe.py
            # forbids fused_stage1=True + fused_stage2=False, so they move together.
            enable_fused_stage1=True,
            enable_fused_stage2=True,
            # ATOM no longer builds a tune table; MegaMoE auto-loads its own.
            gemm2_tile_table=None,
        )
        _MEGA_CACHE[key] = m
    return m


def run_mega_moe(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str = "a8w4",
) -> torch.Tensor:
    """Replace EP experts with MegaMoE. x: [tokens, model_dim] bf16 (this rank's
    local tokens, pre-dispatch). topk_ids: global expert ids. Returns
    [tokens, model_dim] bf16."""
    from aiter.dist.parallel_state import get_ep_group

    am = get_ep_group().device_communicator.all2all_manager
    rank, world = int(am.rank), int(am.world_size)

    run_tokens = int(x.shape[0])
    moe_cfg = getattr(layer, "moe_config", None)
    mtpr = int(getattr(moe_cfg, "max_num_tokens", 0))
    assert run_tokens <= mtpr, f"[mega] run_tokens={run_tokens} > max_num_tokens={mtpr}"

    # local expert count is stashed at build time (layer._mega_local_E). Do NOT
    # infer it from _mega_w1.shape: the INTERLEAVE build uses shuffle_weight_w4
    # (6D output) + .view(-1) (1D), so shape[0] is NOT E*2*inter anymore -> the old
    # `_mega_w1.shape[0] // (2*inter_dim)` inference gives garbage (e.g. 172032),
    # blowing up `experts` and sizing a ~76GB mori dispatch buffer -> shmem_malloc fail.
    local_E = int(getattr(layer, "_mega_local_E", 0))
    if local_E <= 0:  # fallback for any pre-existing 2D layout
        local_E = int(layer._mega_w1.shape[0] // (2 * inter_dim))
    experts = local_E * world
    with torch.inference_mode(False), torch.no_grad():
        mega = get_or_build_mega_moe(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            mtpr=mtpr,
            w1=layer._mega_w1,
            w1_scale=layer._mega_w1_scale,
            w2=layer._mega_w2,
            w2_scale=layer._mega_w2_scale,
        )

    # per-layer weight swap (runtime pointer args; shapes identical across layers).
    # mega_moe_v1 refactor (8acf56d): fused stage-1 weights are held as
    # mega._s1_w1 / mega._s1_w1_scale (read at runtime in _run_fused_stage1), NOT
    # mega.stage1.w1. w2/w2_scale remain top-level attrs. forward_bf16 is aliased
    # to forward, so the call below is unchanged.
    mega._s1_w1 = layer._mega_w1
    mega._s1_w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale

    wts = topk_weights.to(torch.float32).contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    with torch.inference_mode(False), torch.no_grad():
        out = mega.forward_bf16(x.contiguous(), wts, ids)
    return out
