"""Preload reachable BF16 FlyDSL choices from actual model/head weights."""

import collections
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class BF16Family:
    n: int
    k: int
    bias: bool
    preshuffled: bool


def model_families(runner):
    import torch

    from atom.model_ops.embed_head import ParallelLMHead
    from atom.model_ops.linear import LinearBase

    models = [("target", runner.model)]
    if getattr(runner, "drafter", None) is not None:
        models.append(("draft", runner.drafter.model))
    result = {}
    for prefix, model in models:
        for name, module in model.named_modules():
            if not isinstance(module, (LinearBase, ParallelLMHead)):
                continue
            weight = module.weight
            if weight.dtype != torch.bfloat16:
                continue
            if weight.ndim != 2:
                raise RuntimeError("BF16 linear/head weight must be a matrix")
            family = BF16Family(
                *weight.shape,
                getattr(module, "bias", None) is not None,
                bool(getattr(weight, "is_shuffled", False)),
            )
            result.setdefault(family, []).append(prefix + "." + name)
    return result


def choices(family, max_m):
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    if not 1 <= max_m <= 16384:
        raise ValueError("BF16 dispatch coverage admits 1 <= M <= 16384")
    selected, backends = {}, collections.Counter()
    # BF16 is the native linear/head default; compressors may request FP32.
    # These are ABI output types, not alternative accuracy or tuning choices.
    for outdtype in ("torch.bfloat16", "torch.float32"):
        for m in range(1, max_m + 1):
            row = get_GEMM_A16W16_config(
                m,
                family.n,
                family.k,
                family.bias,
                "torch.bfloat16",
                outdtype,
                False,
                family.preshuffled,
            )
            backends[outdtype + ":" + row["libtype"]] += 1
            if row["libtype"] == "flydsl":
                key = (str(row["kernelName"]), outdtype)
                selected.setdefault(key, m)
    return selected, dict(backends)


def native_configuration(family, name, outdtype, m):
    from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params
    from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as native

    if outdtype not in ("torch.bfloat16", "torch.float32"):
        raise ValueError("Use the two native BF16 output ABIs")
    config = get_flydsl_hgemm_kernel_params(name)
    if config is None:
        raise RuntimeError("Native lookup selected a missing BF16 kernel")
    kwargs = {
        key: config[key]
        for key in (
            "block_m",
            "block_n",
            "block_k",
            "stages",
            "split_k",
            "m_waves",
            "n_waves",
            "k_waves",
            "group_m",
            "use_half_tile_interleaved",
        )
    }
    fused_bias = family.bias and outdtype == "torch.bfloat16"
    param = native.make_gemm_a16w16_param_and_validate(
        m,
        family.n,
        family.k,
        {
            **kwargs,
            "a_is_transposed": False,
            "b_is_transposed": True,
            "in_dtype_id": native.GEMM_A16W16_DTYPE_BF16,
            "out_dtype_id": (
                native.GEMM_A16W16_DTYPE_BF16
                if outdtype == "torch.bfloat16"
                else native.GEMM_A16W16_DTYPE_FP32
            ),
            "has_bias": fused_bias,
        },
    )
    if param is None:
        raise RuntimeError(
            "Native BF16 configuration does not support its selected shape"
        )
    return kwargs, param


def preload_choice(family, name, outdtype, m):
    import torch
    from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as native

    from atom.mono.runtime.compile import compile_only

    kwargs, param = native_configuration(family, name, outdtype, m)
    key = param.__cache_signature__()
    cache = getattr(native.gemm_a16w16_gfx950, "_compiled_cache", {})
    if key in cache:
        # run_cached's hit path dispatches the compiled function directly, so
        # never call it a second time on temporary uninitialized inputs.
        return {"signature": repr(key), "already_preloaded": True}
    a = torch.empty((m, family.k), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((family.n, family.k), dtype=torch.bfloat16, device="cuda").T
    dtype = torch.bfloat16 if outdtype == "torch.bfloat16" else torch.float32
    out = torch.empty((m, family.n), dtype=dtype, device="cuda")
    bias = (
        torch.empty((family.n,), dtype=torch.bfloat16, device="cuda")
        if family.bias and outdtype == "torch.bfloat16"
        else None
    )
    native.get_split_k_buffers(torch.cuda.current_stream(), a.device, kwargs["split_k"])
    with compile_only():
        native.gemm_a16w16(
            a, b, out=out, bias=bias, user_kwargs=kwargs, out_dtype=dtype
        )
    if key not in native.gemm_a16w16_gfx950._compiled_cache:
        raise RuntimeError("BF16 preload did not populate the native dispatch cache")
    return {"signature": repr(key), "already_preloaded": False}


def preload_model(runner):
    from aiter import logger

    max_m = runner.config.max_num_batched_tokens
    if not 1 <= max_m <= 16384:
        raise RuntimeError("This workload's BF16 preload admits M <= 16384")
    before = logger.disabled
    logger.disabled = True
    report = []
    try:
        for family, modules in sorted(model_families(runner).items()):
            selected, backends = choices(family, max_m)
            rows = []
            for (name, outdtype), m in selected.items():
                rows.append(
                    {
                        "kernel": name,
                        "outdtype": outdtype,
                        "representative_m": m,
                        **preload_choice(family, name, outdtype, m),
                    }
                )
            report.append(
                {
                    "family": vars(family),
                    "modules": modules,
                    "native_dispatch_rows": backends,
                    "preloaded": rows,
                }
            )
    finally:
        logger.disabled = before
    if not report:
        raise RuntimeError("No native BF16 linear/head family found")
    return report
