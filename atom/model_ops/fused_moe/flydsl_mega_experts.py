# SPDX-License-Identifier: Apache-2.0
"""aiter MegaMoEV2 fused EP-MoE integration for ATOM.

``moe_backend="mega"`` replaces the whole EP experts step (dispatch + GEMM1 +
quant + GEMM2 + combine) with aiter's upstream ``MegaMoEV2`` single op
(``aiter.ops.flydsl.kernels.mega_moe``, ROCm/aiter#4439). This retargets the
former out-of-tree FlyDSL ``kernels.mega_moe.MegaMoE`` binding: the kernel now
lives inside aiter, so there is no ``ATOM_FLYDSL_KERNELS_PATH`` and no
``FUSED_MEGA_*`` env var to set.

MegaMoEV2 auto-selects its execution config from the runtime token count and
``max_tok_per_rank`` (MTPR): MTPR<=255 -> low-latency fixed-slot path,
MTPR>=256 -> compact path (``FIXED_SLOT_MAX_MTPR``). MTPR must be a positive
power of two. It is passed in as ``max_num_tokens`` (ATOM's
``max_num_batched_tokens``) so that every rank uses the same value: MTPR also
selects the p2p wire format, and a rank-dependent MTPR desynchronises it.

Weight prep uses aiter's own shuffles (``aiter.ops.shuffle``):
  w1/w1_scale: ``shuffle_weight_a16w4(., 16, gate_up=True)`` /
               ``shuffle_scale_a16w4(., E, gate_up=True)``  (g1u1 interleave)
  w2/w2_scale: ``shuffle_weight_a16w4(., 16, gate_up=False)`` /
               ``shuffle_scale_a16w4(., E, gate_up=False)``
Shuffled tensors stay expert-major + contiguous, so EPLB can migrate their
expert-major views in place.

Memory: ONE MegaMoEV2 is shared across all MoE layers (process-level cache keyed
by shape/quant/mtpr); per-layer weights are swapped in before forward
(``_s1_w1`` / ``_s1_w1_scale`` / ``w2`` / ``w2_scale`` are runtime pointer args,
not baked into the kernel).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("atom")

_MEGA_CACHE: dict = {}
_MEGA_ROUTE_ROWS: dict[tuple[torch.device, int], torch.Tensor] = {}
_MEGA_BUILD_DBG = False


def _os_env(k):
    return os.environ.get(k, "<unset>")


def build_mega_weights(layer) -> None:
    """From ATOM's RAW (pre-atom-shuffle) mxfp4 w13/w2 + e8m0 scales, build the
    MegaMoEV2-layout weights and stash on the layer. Must run BEFORE atom's own
    shuffle_weight in process_weights_after_loading (uses raw layout)."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    # a8w4 MegaMoEV2 uses the g1u1 gate-up INTERLEAVE layout for w1/w1_scale
    # (gate_up=True); w2 has no gate/up split (gate_up=False). This matches the
    # aiter reference op_tests/multigpu_tests/test_mega_moe_v2.py weight prep.
    w13 = layer.w13_weight.data  # [E, 2*inter, hidden//2] fp4-packed uint8
    E = int(w13.shape[0])
    expected_local = layer.local_num_experts
    if E != expected_local:
        raise RuntimeError(
            "MegaMoE local weight width disagrees with the dispatch layout: "
            f"weights={E}, dispatch={expected_local}. The shared expert must "
            "occupy the fixed tail slot in every Mega weight tensor."
        )
    layer._mega_w1 = shuffle_weight_a16w4(w13, 16, True).contiguous()

    s1 = layer.w13_weight_scale.data  # [E, 2*inter, hidden//32] e8m0
    s1f = s1.reshape(E * s1.shape[1], s1.shape[2])  # 2D (E*2*inter, hidden//32)
    layer._mega_w1_scale = shuffle_scale_a16w4(s1f, E, True).contiguous()

    w2 = layer.w2_weight.data  # [E, hidden, inter//2] fp4-packed uint8
    layer._mega_w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()

    s2 = layer.w2_weight_scale.data  # [E, hidden, inter//32] e8m0
    s2f = s2.reshape(E * s2.shape[1], s2.shape[2])  # 2D (E*hidden, inter//32)
    layer._mega_w2_scale = shuffle_scale_a16w4(s2f, E, False).contiguous()

    global _MEGA_BUILD_DBG
    if not _MEGA_BUILD_DBG:
        _MEGA_BUILD_DBG = True
        logger.info(
            f"[MEGA-BUILD] w13={tuple(w13.shape)}{w13.dtype} "
            f"w13_scale={tuple(s1.shape)}{s1.dtype} "
            f"w2={tuple(w2.shape)}{w2.dtype} w2_scale={tuple(s2.shape)}{s2.dtype} | "
            f"_mega_w1={tuple(layer._mega_w1.shape)} "
            f"_mega_w1_scale={tuple(layer._mega_w1_scale.shape)} "
            f"_mega_w2={tuple(layer._mega_w2.shape)} "
            f"_mega_w2_scale={tuple(layer._mega_w2_scale.shape)} | "
            f"E={E} GU_ITLV={_os_env('ATOM_MOE_GU_ITLV')}"
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
    swiglu_limit,
    w1,
    w1_scale,
    w2,
    w2_scale,
):
    # swiglu_limit belongs in the key: aiter bakes the clamp into the GEMM1
    # kernel at trace time (gemm_util.py `if self._swiglu_limit <= 0`), so two
    # layers with different limits cannot share one instance. Without it in the
    # key they would silently reuse whichever limit was built first -- wrong
    # numerics, no error.
    key = (
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        mtpr,
        swiglu_limit,
    )
    m = _MEGA_CACHE.get(key)
    if m is None:
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

        # MegaMoEV2 auto-selects fixed-slot vs compact and its GEMM tiles from
        # (tokens, mtpr); no gemm2_tile / enable_fused / tune-table args anymore.
        with torch.inference_mode(False), torch.no_grad():
            m = MegaMoEV2(
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
                swiglu_limit=swiglu_limit,
            )
        _MEGA_CACHE[key] = m

    # Bind this layer's weights on EVERY call, not only on build: the cache key
    # has no weight component, so ONE instance is shared by all MoE layers (they
    # are shape-identical) and a cache hit would otherwise keep running whichever
    # layer happened to build it. The kernels read these as runtime pointer args
    # -- _s1_w1 / _s1_w1_scale are uint8 views read in _run_fused_stage1, w2 /
    # w2_scale are read via data_ptr in _run_fused_stage2.
    # CUDAGraph-safe: capture bakes each layer's own pointer into that layer's
    # launch, and EPLB rebalance mutates the weights in place, so the pointers
    # stay valid across replays. Do not switch these to freshly allocated
    # tensors without revisiting capture.
    m._s1_w1 = w1.view(torch.uint8)
    m._s1_w1_scale = w1_scale.view(torch.uint8)
    m.w2 = w2
    m.w2_scale = w2_scale
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
    mtpr: int,
    swiglu_limit: float,
    quant: str = "a8w4",
) -> torch.Tensor:
    """Replace EP experts with MegaMoEV2. x: [tokens, model_dim] bf16 (this rank's
    local tokens, pre-dispatch). topk_ids: global (physical) expert ids. Returns
    [tokens, model_dim] bf16."""
    from aiter.dist.parallel_state import get_ep_group

    # Do NOT "simplify" this to get_ep_group().rank_in_group / .world_size.
    # Reading device_communicator.all2all_manager is load-bearing: that property
    # lazily constructs the all2all manager, and building MoriAll2AllManager is
    # what initializes the mori symmetric shmem heap.
    am = get_ep_group().device_communicator.all2all_manager
    rank, world = int(am.rank), int(am.world_size)

    run_tokens = int(x.shape[0])
    if run_tokens > mtpr:
        raise ValueError(
            f"[mega] run_tokens={run_tokens} exceeds mtpr={mtpr} "
            f"(max_num_tokens); widening mtpr here would diverge the p2p wire "
            f"format across ranks"
        )
    if not hasattr(layer, "_mega_w1"):
        raise RuntimeError("MegaMoE weights were not prepared")

    # Returns the shared MegaMoEV2 already bound to this layer's weights.
    mega = get_or_build_mega_moe(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        mtpr=mtpr,
        swiglu_limit=float(swiglu_limit),
        w1=layer._mega_w1,
        w1_scale=layer._mega_w1_scale,
        w2=layer._mega_w2,
        w2_scale=layer._mega_w2_scale,
    )

    wts = topk_weights.to(torch.float32).contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    with torch.inference_mode(False), torch.no_grad():
        # swiglu_limit is NOT a forward arg -- it is baked into the instance at
        # construction (see the cache key above).
        out = mega.forward(x.contiguous(), wts, ids)
    return out


class MegaFusedExperts:
    """MegaMoE as a whole-pipeline ``fused_experts`` backend.

    Mega owns dispatch + GEMM1 + activation + GEMM2 + combine, so it replaces
    the entire modular kernel rather than plugging into its prepare/finalize
    seam. It is installed on ``quant_method.fused_experts`` and is therefore
    reached through the same ``if self.fused_experts: return self.fused_experts(...)``
    dispatch at the tail of ``Mxfp4MoEMethod.apply`` that the MORI path uses --
    Mega adds no dispatch layer and overrides no method on the quant method.

    The layer is bound at construction (``init_prepare_finalize`` already
    receives it) because Mega reads its weights from ``layer._mega_*``, which
    the modular-kernel call signature does not carry.

    Deliberately NOT an ``nn.Module``: it is reached purely by duck typing
    (``if self.fused_experts: return self.fused_experts(...)``, no isinstance
    checks anywhere), and holding ``layer`` on an ``nn.Module`` would register
    it as a submodule -- layer -> quant_method -> fused_experts -> layer is a
    cycle that breaks module traversal and duplicates the layer in state_dict.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        model_dim: int,
        inter_dim: int,
        mtpr: int,
        quant: str = "a8w4",
    ) -> None:
        self._layer = layer
        self._model_dim = model_dim
        self._inter_dim = inter_dim
        self._mtpr = mtpr
        self._quant = quant

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int = -1,
        activation=None,
        apply_router_weight_on_input: bool = False,
        expert_map: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        from aiter import ActivationType

        if apply_router_weight_on_input:
            raise NotImplementedError(
                "mega does not support apply_router_weight_on_input=True"
            )
        if activation is not None and activation != ActivationType.Silu:
            raise NotImplementedError(
                f"mega hardcodes SwiGLU; got activation={activation}"
            )
        # w1/w2 are the emptied AITER staging buffers -- mega reads layer._mega_*.
        # expert_map is intentionally unused: Mega consumes backend dispatch ids,
        # while expert_map belongs to EPLB's routed physical space.
        del w1, w2, expert_map

        local_weight_experts = int(self._layer._mega_w1.shape[0])
        if local_weight_experts != self._layer.local_num_experts:
            raise RuntimeError(
                "MegaMoE weight/dispatch layout changed after preparation: "
                f"weights={local_weight_experts}, "
                f"expected={self._layer.local_num_experts}"
            )

        return run_mega_moe(
            self._layer,
            hidden_states,
            topk_weights,
            topk_ids,
            model_dim=self._model_dim,
            inter_dim=self._inter_dim,
            experts=global_num_experts,
            # Infer top-k from the routing tensors, same as the standard kernels.
            topk=int(topk_ids.shape[1]),
            # todo decode use running_bs for perf
            mtpr=self._mtpr,
            swiglu_limit=getattr(self._layer, "swiglu_limit", 0.0),
            quant=self._quant,
        )
