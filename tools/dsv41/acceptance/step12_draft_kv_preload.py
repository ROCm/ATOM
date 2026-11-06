"""Preload DSpark's native KV projection owned outside registered linears."""

import functools
import os
from dataclasses import asdict
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_kernel_preload import Family, existing_choices, preload_choices


def context_families(model):
    from aiter.ops.gemm_op_common import mxscale_w_scale_block
    from aiter.utility import dtypes

    from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

    if not isinstance(model, DeepseekV41DSpark):
        return {}
    # Resolve the native lazy binding once after weight loading. This is the
    # same cached object write_context_kv consumes, not a second weight owner.
    fused = model._context_kv_fusion()
    if fused is None:
        bindings = [
            (
                f"context_layers.{i}.wkv_shard",
                s.weight,
                s.weight_scale,
                s.native_a8_group_rows,
                getattr(s.weight, "is_shuffled", False),
            )
            for i, a in enumerate(model.context_layers)
            if (s := getattr(a, "wkv_shard", None)) is not None
        ]
    else:
        weight, scale, group_rows, _, preshuffled = fused
        bindings = [("_context_kv_fused", weight, scale, group_rows, preshuffled)]
    families = {}
    for name, weight, scale, group_rows, preshuffled in bindings:
        if (
            weight.dtype != dtypes.fp8
            or not preshuffled
            or scale.dtype not in (dtypes.fp8_e8m0, dtypes.u8)
        ):
            raise RuntimeError("DSpark KV preload requires the native MXFP8 binding")
        n, k = weight.shape
        block = mxscale_w_scale_block(tuple(scale.shape), n, k)
        if group_rows != int(block.split("x")[0]):
            raise RuntimeError("DSpark KV binding scale geometry disagrees")
        family = Family("gemm", 1, n, k, 32, block)
        families.setdefault(family, []).append("drafter." + name)
    return families


def preload_model(runner):
    from aiter import logger
    from aiter.dist.parallel_state import get_tp_group

    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return
    max_m = runner.config.max_num_batched_tokens
    if not 1 <= max_m <= 16384:
        raise RuntimeError("DSpark KV preload admits 1 <= M <= 16384")
    path = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"]) / (
        f"draft_kv_preload_rank{get_tp_group().rank_in_group}.json"
    )
    rows, seen = [], set()
    previous = logger.disabled
    logger.disabled = True
    try:
        for family, bindings in context_families(drafter.model).items():
            choices, opus = existing_choices(family, max_m)
            keys = preload_choices(family, choices, seen)
            rows.append(
                {
                    **asdict(family),
                    "bindings": bindings,
                    "choices": [(list(k), m) for k, m in choices.items()],
                    "native_opus_choices": opus,
                    "preloaded_keys": keys,
                }
            )
        write_json(
            path,
            {
                "status": "PASS",
                "families": rows,
                "max_m": max_m,
                "unique_specializations": len(seen),
                "jit_preload_gpu_dispatches": 0,
            },
        )
    except BaseException as error:
        write_json(path, {"status": "failed", "error": repr(error)})
        raise
    finally:
        logger.disabled = previous


def install():
    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        preload_model(self)
        return original(self)

    ModelRunner._maybe_warmup = warmup
