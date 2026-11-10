# SPDX-License-Identifier: Apache-2.0
"""aiter MegaMoEInternode (two-node EP16 fused MoE) as ATOM's routed-experts step.

``moe_backend="mega_internode"`` replaces dispatch + GEMM1 + activation + GEMM2 +
combine of an MXFP4 MoE layer with aiter's ``MegaMoEInternode`` (A4W4: the op
quantizes the bf16 rows to MXFP4 itself and sends FP4 on dispatch).

Capacities. The op accepts actual rows up to a fixed capacity (this
adapter chunks prefill at 4096). Instances are shared by every shape-identical
MoE layer (weights are bound per call):

* decode: ``mtpr`` in ``_DECODE_MTPRS``, the smallest that fits the
  group-agreed row count;
* prefill: ``mtpr = _PREFILL_MTPR``; a larger step is split into chunks.

Every rank must issue the same sequence of collective forwards, so the chunk
count comes from the DP group's agreed maximum (``dp_metadata``), never from
this rank's own row count; a rank with fewer rows runs pad chunks.

Pad rows (ATOM routes them to ``-1``) and the rows past a rank's real count
carry ``-1`` in every slot (weight 0). The op treats an out-of-range expert id
as an invalid slot: no dispatch, no GEMM work, and k2 accumulates it as zero,
so a ragged step costs each rank only its real rows.

Weights: w1/w1 scale in the a4w4 GEMM1 layout (``shuffle_weight`` (16,16) +
``e8m0_shuffle``), w2/w2 scale in the a16w4 GEMM2 layout
(``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``), built from the raw
checkpoint layout (gate rows then up rows).
"""

from __future__ import annotations

import logging
from contextlib import ExitStack

import torch

logger = logging.getLogger("atom")

# Decode capacities, smallest first. A unified decode step (incl. speculative
# verify: bs * (1 + num_spec)) runs on the smallest capacity that fits.
_DECODE_MTPRS = (32, 64, 128, 256, 512, 1024)
_PREFILL_MTPR = 4096
_OP_MAX_MTPR = 4096

_INSTANCES: dict = {}


def close_internode_instances() -> None:
    """Close cached operators before ATOM destroys its distributed groups.

    Each capacity owns a CCO communicator. Close in reverse construction
    order, attempting every capacity even when one teardown raises.
    """
    instances = list(_INSTANCES.values())
    _INSTANCES.clear()
    if instances:
        logger.info("[MEGA-INTERNODE] closing %d cached capacities", len(instances))
    with ExitStack() as cleanup:
        for instance in instances:
            cleanup.callback(instance["op"].close)


def build_internode_weights(layer) -> None:
    """Stash the op's weight layouts on the layer from ATOM's raw MXFP4
    tensors. Must run before ATOM's own shuffle."""
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    w13 = layer.w13_weight.data  # [E, 2*inter, hidden//2] fp4x2, gate rows then up rows
    s13 = layer.w13_weight_scale.data  # [E, 2*inter, hidden//32] e8m0
    w2 = layer.w2_weight.data  # [E, hidden, inter//2]
    s2 = layer.w2_weight_scale.data  # [E, hidden, inter//32]
    E = int(w13.shape[0])
    if E != layer.local_num_experts:
        raise RuntimeError(
            f"mega_internode: local weights {E} != local_num_experts "
            f"{layer.local_num_experts}"
        )
    # Same calls as the op's unit test (op_tests/.../megamoe_internode_inputs.py):
    # 3-D packed weights, 2-D (E*rows, K/32) scales.
    layer._mi_w1 = shuffle_weight(w13, layout=(16, 16)).contiguous()
    layer._mi_w1_scale = e8m0_shuffle(
        s13.reshape(E * s13.shape[1], s13.shape[2])
    ).contiguous()
    layer._mi_w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
    layer._mi_w2_scale = shuffle_scale_a16w4(
        s2.reshape(E * s2.shape[1], s2.shape[2]), E, False
    ).contiguous()


def _rank_world():
    from aiter.dist.parallel_state import get_ep_group

    group = get_ep_group()
    return int(group.rank_in_group), int(group.world_size)


def _get_instance(
    mtpr, *, model_dim, inter_dim, experts, topk, layer, activation, swiglu_limit
):
    # activation/swiglu_limit are baked into GEMM1 at compile time: part of the key.
    rank, world = _rank_world()
    key = (
        torch.cuda.current_device(),
        rank,
        world,
        mtpr,
        model_dim,
        inter_dim,
        experts,
        topk,
        activation,
        swiglu_limit,
    )
    inst = _INSTANCES.get(key)
    if inst is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"mega_internode capacity {mtpr} was not built before graph capture"
            )
        from aiter.ops.flydsl.kernels.mega_moe_internode import MegaMoEInternode

        with torch.inference_mode(False), torch.no_grad():
            op = MegaMoEInternode(
                rank=rank,
                world_size=world,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                quant="a4w4",
                w1=layer._mi_w1,
                w1_scale=layer._mi_w1_scale,
                w2=layer._mi_w2,
                w2_scale=layer._mi_w2_scale,
                max_tok_per_rank=mtpr,
                act=activation,
                swiglu_limit=swiglu_limit,
            )
        dev = op.device
        epr = experts // world
        inst = {
            "op": op,
            "x": torch.zeros(mtpr, model_dim, dtype=torch.bfloat16, device=dev),
            "ids": torch.full((mtpr, topk), -1, dtype=torch.int32, device=dev),
            "wts": torch.zeros(mtpr, topk, dtype=torch.float32, device=dev),
        }
        import json
        import os
        from pathlib import Path

        audit = {"rank": rank, "capacity": mtpr, "quant": op.quant,
                 "stage1": op._s1_tile, "paired_tune": op._s2_tuned,
                 "gemm2_bn": op._gemm2_bn, "gemm2_bm": op._gemm2_bm,
                 "gemm2_cu": op._gemm2_cu, "return_chunk_tokens": op._chunk_tokens,
                 "stage2_sdma": op._stage2_sdma,
                 "stage1_kernel": op.stage1_kernel_name,
                 "tune_path": os.environ.get("AITER_CONFIG_MEGAMOE_INTERNODE")}
        audit_dir = Path(os.environ["ATOM_TORCH_PROFILER_DIR"]).parent / "tune_audit"
        audit_dir.mkdir(exist_ok=True)
        (audit_dir / f"rank{rank}_capacity{mtpr}.json").write_text(json.dumps(audit, indent=2) + "\n")
        logger.info("[MEGA-TUNE] %s", json.dumps(audit))
        _INSTANCES[key] = inst
        if rank == 0:
            logger.info(
                "[MEGA-INTERNODE] built capacity=%d ep=%d experts/rank=%d topk=%d",
                mtpr,
                world,
                epr,
                topk,
            )
            logger.info(
                "[MEGA-INTERNODE] stage1=%s window_bytes=%d single_kernel=%s",
                op.stage1_kernel_name,
                op.layout.total_bytes,
                op._single_kernel,
            )
    return inst


def _run_chunk(inst, layer, x, ids, wts, start, n):
    """Run rows [start, start+n) of this rank (n may be 0) through one op call."""
    op = inst["op"]
    bx, bids, bw = inst["x"], inst["ids"], inst["wts"]
    if n > 0:
        bx[:n].copy_(x[start : start + n])
        cid = ids[start : start + n]
        # ATOM's pad rows are already -1; the op skips invalid slots.
        bids[:n].copy_(cid)
        bw[:n].copy_(torch.where(cid >= 0, wts[start : start + n], 0.0))
    op.bind_weights(layer._mi_w1, layer._mi_w1_scale, layer._mi_w2, layer._mi_w2_scale)
    with torch.inference_mode(False), torch.no_grad():
        return op.forward(bx[:n], bw[:n], bids[:n])


def _group_rows(local_rows: int) -> tuple[int, bool]:
    """(rows every rank agrees on, whether the step is unified)."""
    from atom.utils.forward_context import get_forward_context

    fc = get_forward_context()
    ctx = fc.context
    if ctx is not None and ctx.running_tokens_are_unified and ctx.running_tokens > 0:
        return int(ctx.running_tokens), True
    if fc.dp_metadata is not None:
        return int(fc.dp_metadata.max_tokens_across_dp), False
    raise RuntimeError("mega_internode needs dp_metadata to agree on the chunk count")


class MegaInternodeFusedExperts:
    """Whole-pipeline ``fused_experts`` backend (see MegaFusedExperts)."""

    def __init__(self, layer, *, model_dim: int, inter_dim: int, max_tokens: int):
        self._layer = layer
        self._model_dim = model_dim
        self._inter_dim = inter_dim
        self._prefill_mtpr = min(_PREFILL_MTPR, _OP_MAX_MTPR)
        self._decode_mtprs = tuple(m for m in _DECODE_MTPRS if m < self._prefill_mtpr)
        del max_tokens  # any step is chunked to the prefill capacity

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        w1=None,
        w2=None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int = -1,
        activation=None,
        apply_router_weight_on_input: bool = False,
        expert_map=None,
        x_scale=None,
        **_ignored,
    ) -> torch.Tensor:
        from aiter import ActivationType

        assert x_scale is None, "mega_internode takes bf16 rows"
        if apply_router_weight_on_input:
            raise NotImplementedError("mega_internode: apply_router_weight_on_input")
        if activation == ActivationType.Situv2:
            act, limit = "situv2", 0.0
        elif activation in (None, ActivationType.Silu):
            # SwiGLU; a positive swiglu_limit (DeepSeek-V4: 10) clamps it.
            act, limit = "silu", float(getattr(self._layer, "swiglu_limit", 0.0) or 0.0)
        else:
            raise NotImplementedError(
                f"mega_internode: unsupported activation {activation}"
            )
        del w1, w2, expert_map
        layer = self._layer
        x = hidden_states
        rows = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        ids = topk_ids.to(torch.int32)
        wts = topk_weights.to(torch.float32)
        build = {
            "model_dim": self._model_dim,
            "inter_dim": self._inter_dim,
            "experts": global_num_experts,
            "topk": topk,
            "layer": layer,
            "activation": act,
            "swiglu_limit": limit,
        }
        # Same construction order on every rank (symmetric allocations).
        inst_p = _get_instance(self._prefill_mtpr, **build)
        inst_ds = [_get_instance(m, **build) for m in self._decode_mtprs]

        group_rows, _ = _group_rows(rows)
        if rows > group_rows:
            raise RuntimeError(
                f"mega_internode: local rows {rows} > group rows {group_rows}"
            )
        out = torch.empty(rows, self._model_dim, dtype=torch.bfloat16, device=x.device)
        for inst_d in inst_ds:
            if group_rows <= inst_d["op"].mtpr:
                res = _run_chunk(inst_d, layer, x, ids, wts, 0, rows)
                out.copy_(res[:rows])
                return out
        cap = self._prefill_mtpr
        for c in range((group_rows + cap - 1) // cap):
            start = c * cap
            n = max(0, min(rows - start, cap))
            res = _run_chunk(inst_p, layer, x, ids, wts, start, n)
            if n > 0:
                out[start : start + n].copy_(res[:n])
        return out
