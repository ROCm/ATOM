"""Startup-only preload of existing gfx950 dispatch choices for loaded weights.

This experiment changes neither dispatch nor arithmetic. It enumerates the
configured M domain on CPU, deduplicates full native launcher specializations,
then uses AITER's no-dispatch preload. No synthetic request or tuning timing.
Install before ModelRunner construction, outside any traced model function.
"""

import contextlib
import functools
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Family:
    kind: str
    b: int
    n: int
    k: int
    x_block: int
    w_block: str


def model_families(model):
    from aiter.ops.gemm_op_common import mxscale_w_scale_block
    from aiter.utility import dtypes

    from atom.model_ops.linear import LinearBase

    families = {}
    grouped = {}
    for name, module in model.named_modules():
        # This native attention uses wo_a's storage as [groups, o_rank, K].
        if all(hasattr(module, key) for key in ("wo_a", "groups", "o_rank")):
            weight = module.wo_a.weight
            b, n, k = module.groups, module.o_rank, weight.shape[-1]
            grouped[id(module.wo_a)] = Family("bmm", b, n, k, 32, "32x32")
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        weight, scale = module.weight, module.weight_scale
        if (
            weight.dtype != dtypes.fp8
            or scale is None
            or scale.dtype not in (dtypes.fp8_e8m0, dtypes.u8)
        ):
            continue
        if id(module) in grouped:
            family = grouped[id(module)]
        else:
            n, k = weight.shape
            block = mxscale_w_scale_block(tuple(scale.shape), n, k)
            family = Family("gemm", 1, n, k, int(block.split("x")[1]), block)
        families.setdefault(family, []).append(name)
    if not families:
        raise RuntimeError("No loaded native MXFP8 families found")
    return families


def existing_choices(family, max_m):
    """Use exactly the current native lookups/heuristic; retain M=1 semantics."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import pick_bmm_kernel_name
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config
    from aiter.ops.opus.policy import lookup_mxscale_bmm_config

    choices = {}
    skipped = set()
    wn, wk = map(int, family.w_block.split("x"))
    for m in range(1, max_m + 1):
        if family.kind == "gemm":
            row = get_mxscale_bpreshuffle_config(m, family.n, family.k, family.w_block)
            backend = None if row is None else str(row["kernelId"])
            if row is not None and (
                row["libtype"] != "flydsl" or backend not in ("mxfp8", "bmm")
            ):
                raise RuntimeError("Unsupported native GEMM dispatch row")
            if backend == "mxfp8" and family.x_block != (
                128 if family.w_block == "128x128" else 32
            ):
                row = None
                backend = None
        else:
            row = lookup_mxscale_bmm_config(
                family.b,
                m,
                family.n,
                family.k,
                w_scale_block=family.w_block,
                bpreshuffle=True,
            )
            if row is not None and row["libtype"] == "opus":
                skipped.add((int(row["kernelId"]), int(row["splitK"])))
                continue  # Already built native backend; preserve its dispatch.
            if row is not None and row["libtype"] != "flydsl":
                raise RuntimeError("Unsupported native BMM dispatch row")
            backend = "bmm"
        if backend != "mxfp8":
            transposed = family.kind == "gemm" and family.x_block == 128 and m > 1
            name = (
                str(row["kernelName"])
                if row is not None
                else pick_bmm_kernel_name(
                    family.b, m, family.n, family.k, family.x_block, wn, wk, transposed
                )
            )
            key = ("bmm", name, transposed)
        else:
            # M=1 vs multiple rows affects tile16's native specialization.
            key = ("mxfp8", str(row["kernelName"]), m > 1)
        choices.setdefault(key, m)
    return choices, sorted(skipped)


@contextlib.contextmanager
def capture_mxpsh_launches(consume):
    """Record the runtime's complete ABI without calling any compiled function.

    Only the startup thread is intercepted; unrelated threads retain runtime
    behavior. The native mxpsh entry imports this helper at its call site.
    """
    from aiter.ops.flydsl.kernels import tensor_shim

    original = tensor_shim._run_compiled
    owner = threading.get_ident()

    def capture(exe, *args, specialization_key=None):
        if threading.get_ident() != owner:
            return original(exe, *args, specialization_key=specialization_key)
        consume(exe, args, specialization_key)

    tensor_shim._run_compiled = capture
    try:
        yield
    finally:
        tensor_shim._run_compiled = original


def preload_choices(family, choices, seen):
    import flydsl.expr as fx
    import torch
    from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as bmm
    from aiter.ops.flydsl import mxscale_preshuffle_kernels as mx
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import (
        compile_mxfp8_8wave_gemm,
        parse_mxfp8_8wave_kernel_name,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        parse_kernel_name,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, ptr_arg

    completed = []

    def consume(exe, args, specialization):
        source = exe.func
        key = (source.__module__, source.__qualname__, specialization)
        if key in seen:
            return
        if len(seen) >= 512:
            raise RuntimeError("Preload specialization admission exceeded")
        _preload_compiled(exe, *args)
        seen.add(key)
        completed.append(repr(key))

    zero = torch.empty(0, dtype=torch.uint8, device="cpu")
    for (backend, name, variant), m in choices.items():
        if backend == "bmm":
            wn, wk = map(int, family.w_block.split("x"))
            spec = bmm._constexprs(
                name, family.n, family.k, family.b, family.x_block, wn, wk, variant
            )
            args = (*(ptr_arg(zero),) * 7, m, fx.Stream(0), *spec)
            consume(bmm.launch_bmm_a8w8_mxscale, args, spec)
            continue
        config = parse_mxfp8_8wave_kernel_name(name)
        if config is not None:
            factory = {
                "K": family.k,
                "waves_per_eu": config["waves_per_eu"],
                "xcd_swizzle": config["xcd_swizzle"],
                "scale_preshuffled": False,
                "w_scale_block": family.w_block,
            }
            exe = compile_mxfp8_8wave_gemm(**factory)
            # AOT and runtime both pass flat uint8/uint8/bf16/uint8/uint8.
            out = torch.empty(0, dtype=torch.bfloat16, device="cpu")
            consume(
                exe,
                (zero, zero, out, zero, zero, m, family.n, fx.Stream(0)),
                tuple(factory.items()),
            )
            continue
        config = parse_kernel_name(name)
        if config is None:
            raise RuntimeError("Unknown native mxfp8 kernel " + name)
        wn, wk = map(int, family.w_block.split("x"))
        # Meta tensors carry native shape/layout with zero backing storage.
        a = torch.empty((m, family.k), dtype=torch.uint8, device="meta")
        b = torch.empty((family.n, family.k), dtype=torch.uint8, device="meta")
        sa = torch.empty(
            (m, family.k // family.x_block), dtype=torch.uint8, device="meta"
        )
        sb = torch.empty(
            (family.n // wn, family.k // wk), dtype=torch.uint8, device="meta"
        )
        out = torch.empty((m, family.n), dtype=torch.bfloat16, device="meta")
        with capture_mxpsh_launches(consume):
            mx.flydsl_mxscale_preshuffle_gemm(
                a,
                b,
                sa,
                sb,
                out,
                a_dtype="fp8",
                b_dtype="fp8",
                tile_m=config["tile_m"],
                tile_n=config["tile_n"],
                tile_k=config["tile_k"],
                waves_per_eu=config["waves_per_eu"],
                xcd_swizzle=config["xcd_swizzle"],
                split_k=config["split_k"],
                fused_split_k=config["fused_split_k"],
                stream=fx.Stream(0),
                **dict(mx.MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK[family.w_block]),
            )
    return completed


def preload_model(runner):
    from aiter import logger
    from aiter.dist.parallel_state import get_tp_group
    from rebase_graph_audit_r6 import write_json

    max_m = runner.config.max_num_batched_tokens
    if max_m > 16384:
        raise RuntimeError("This bounded experiment admits M <= 16384")
    path = (
        Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
        / f"kernel_preload_rank{get_tp_group().rank_in_group}.json"
    )
    report = {"status": "preloading", "max_m": max_m, "families": []}
    seen = set()
    before = logger.disabled
    logger.disabled = True  # CPU enumeration must not emit one warning per M.
    try:
        families = model_families(runner.model)
        drafter = getattr(runner, "drafter", None)
        if drafter is not None:
            for family, modules in model_families(drafter.model).items():
                families.setdefault(family, []).extend(
                    "drafter." + name for name in modules
                )
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
    finally:
        logger.disabled = before
    report.update(status="PASS", unique_specializations=len(seen), gpu_dispatches=0)
    write_json(path, report)


def install():
    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        preload_model(self)
        return original(self)

    ModelRunner._maybe_warmup = warmup
