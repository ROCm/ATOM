"""Bind the preload plan to native, retained-prefill and private owner weights."""

import functools
import os
from dataclasses import asdict
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_kernel_preload import (
    Family,
    existing_choices,
    model_families,
    preload_choices,
)


def owner_head_families(q_shape, a_shape, b_shape, groups, o_rank, tp):
    """FullOwnerHeads concatenates Q/A along N and stacks four original B shards.

    The B GEMM retains each original K and BF16 partial, so its B=TP family
    cannot be replaced by a merged-K GEMM in this inventory.
    """
    qn, qk = q_shape
    an, ak = a_shape
    bn, bk = b_shape
    if an != groups * o_rank:
        raise ValueError("Loaded group projection disagrees with native geometry")
    return (
        Family("gemm", 1, tp * qn, qk, 32, "32x32"),
        Family("bmm", tp * groups, o_rank, ak, 32, "32x32"),
        Family("bmm", tp, bn, bk, 32, "32x32"),
    )


def runtime_families(runner, tp, owner):
    families = model_families(runner.model)
    drafter = getattr(runner, "drafter", None)
    if drafter is not None:
        for family, modules in model_families(drafter.model).items():
            families.setdefault(family, []).extend(
                "drafter." + name for name in modules
            )
    for name, attention in runner.model.named_modules():
        if not all(
            hasattr(attention, key)
            for key in ("wo_a", "wo_b", "wq_b", "groups", "o_rank")
        ):
            continue
        if owner:
            for family in owner_head_families(
                attention.wq_b.weight.shape,
                attention.wo_a.weight.shape,
                attention.wo_b.weight.shape,
                attention.groups,
                attention.o_rank,
                tp,
            ):
                families.setdefault(family, []).append(name + ".FullOwnerHeads")
        # Native retained-prefill bindings deliberately live in __dict__, not
        # nn.Module registration, to avoid a second checkpoint/weight owner.
        if getattr(attention, "owner_attention_bound", False):
            for field in ("wq_b_owned", "wo_b_owned"):
                n, k = getattr(attention, field).weight.shape
                family = Family("gemm", 1, n, k, 32, "32x32")
                families.setdefault(family, []).append(name + "." + field)
            groups = attention.wo_a_owned.weight.shape[0] // attention.o_rank
            family = Family(
                "bmm",
                groups,
                attention.o_rank,
                attention.wo_a_owned.weight.shape[-1],
                32,
                "32x32",
            )
            families.setdefault(family, []).append(name + ".wo_a_owned")
    return families


def preload_model(runner):
    from aiter import logger
    from aiter.dist.parallel_state import get_tp_group

    group = get_tp_group()
    max_m = runner.config.max_num_batched_tokens
    if max_m > 16384:
        raise RuntimeError("This bounded experiment admits M <= 16384")
    owner = os.environ.get("DSV41_OWNER_SERVING") == "1"
    path = (
        Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
        / f"kernel_preload_rank{group.rank_in_group}.json"
    )
    families = runtime_families(runner, group.world_size, owner)
    report = {"status": "preloading", "max_m": max_m, "owner": owner, "families": []}
    seen = set()
    previous = logger.disabled
    logger.disabled = True
    try:
        for family, modules in sorted(families.items()):
            choices, opus = existing_choices(family, max_m)
            keys = preload_choices(family, choices, seen)
            report["families"].append(
                {
                    **asdict(family),
                    "modules": modules,
                    "dispatch_choices": len(choices),
                    "native_opus_choices": opus,
                    "preloaded_keys": keys,
                }
            )
            write_json(path, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        write_json(path, report)
        raise
    finally:
        logger.disabled = previous
    report.update(status="PASS", unique_specializations=len(seen))
    write_json(path, report)


def install():
    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        preload_model(self)
        return original(self)

    ModelRunner._maybe_warmup = warmup
