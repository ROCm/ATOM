# SPDX-License-Identifier: Apache-2.0
"""Prepare request-selected GEMM variants during model warmup."""

import csv
import logging

import torch

logger = logging.getLogger("atom")


def _variant_rows(table, n, k, scale_block, max_tokens, lookup):
    # Resolve through AITER rather than treating CSV rows as launch configs:
    # the active architecture, duplicate rows and padded-M lookup still apply.
    candidates = {1, max_tokens}
    for row in table:
        if (
            int(row["N"]) == n
            and int(row["K"]) == k
            and row["w_scale_block"] == scale_block
        ):
            m = int(row["M"])
            if 0 < m <= max_tokens:
                candidates.add(m)
    seen = set()
    for m in sorted(candidates):
        config = lookup(m, n, k, scale_block)
        if config is None:
            continue  # Existing model/graph warmup covers the heuristic path.
        key = (config["kernelId"], config["kernelName"])
        # The 16-row MXFP8 launcher specializes multi_row (M > 1). Its
        # single-row and multi-row code objects share the same tune name.
        specialization = (key, m > 1 if key[1].startswith("flydsl_mxpsh_16x") else None)
        if specialization not in seen:
            seen.add(specialization)
            yield m, key


@torch.no_grad()
def warmup_native_fp8_gemms(model, max_tokens):
    """Launch each tuned variant once per loaded weight/scale signature.

    A maximum-size model forward does not prepare variants selected by ragged
    prefill. Use real allocated inputs and loaded weights so both the Python
    launcher and device module are ready before requests. No EP collectives or
    model forwards are added, and temporary inputs are released at each shape.
    """
    layers = []
    seen = set()
    for layer in model.modules():
        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        group = getattr(layer, "native_a8_group_rows", None)
        if (
            group not in (1, 32)
            or weight is None
            or scale is None
            or weight.ndim != 2
            or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float8_e8m0fnu
            or not getattr(weight, "is_shuffled", False)
        ):
            continue
        key = (weight.device, tuple(weight.shape), tuple(scale.shape), group)
        if key not in seen:
            seen.add(key)
            layers.append((weight, scale, group))
    if not layers:
        return 0

    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

    from atom.model_ops.blockscale import native_quant_linear

    path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
    with open(path, newline="") as stream:
        table = list(csv.DictReader(stream))
    count = 0
    for weight, scale, group in layers:
        n, k = weight.shape
        variants = list(
            _variant_rows(
                table,
                n,
                k,
                f"{group}x32",
                max_tokens,
                get_mxscale_bpreshuffle_config,
            )
        )
        if not variants:
            continue
        rows = max(m for m, _ in variants)
        x = torch.zeros((rows, k), dtype=weight.dtype, device=weight.device)
        xs = torch.full(
            (rows, k // 32), 127, dtype=torch.uint8, device=weight.device
        ).view(torch.float8_e8m0fnu)
        for m, (_, name) in variants:
            native_quant_linear(
                x[:m],
                weight,
                scale,
                x_scale=xs[:m],
                weight_group_rows=group,
                weight_preshuffled=True,
            )
            count += 1
            logger.info("GEMM warmup M=%d N=%d K=%d: %s", m, n, k, name)
        torch.cuda.synchronize(weight.device)
        del x, xs
    return count


def _a16_variant_rows(table, weight, bias, max_tokens, lookup):
    n, k = weight.shape
    candidates = {1, max_tokens}
    candidates.update(
        int(row["M"])
        for row in table
        if int(row["N"]) == n and int(row["K"]) == k and 0 < int(row["M"]) <= max_tokens
    )
    seen = set()
    for m in sorted(candidates):
        config = lookup(
            m,
            n,
            k,
            bias is not None,
            str(weight.dtype),
            str(torch.bfloat16),
            False,
            getattr(weight, "is_shuffled", False),
        )
        key = tuple(
            config.get(name) for name in ("libtype", "solidx", "kernelName", "splitK")
        )
        if key not in seen:
            seen.add(key)
            yield m


@torch.no_grad()
def warmup_a16_gemms(model, max_tokens):
    """Prepare tuned BF16/FP16 projections omitted by max-batch warmup."""
    from aiter import QuantType

    from atom.model_ops.linear import LinearBase

    layers, seen = [], set()
    for layer in model.modules():
        if (
            not isinstance(layer, LinearBase)
            or layer.quant_type.value != QuantType.No.value
        ):
            continue
        weight, bias = layer.weight, layer.bias
        if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            continue
        key = (
            weight.device,
            tuple(weight.shape),
            weight.dtype,
            bias is not None,
            getattr(weight, "is_shuffled", False),
        )
        if key not in seen:
            seen.add(key)
            layers.append((weight, bias))
    if not layers:
        return 0

    from aiter.jit.core import AITER_CONFIGS
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    with open(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE, newline="") as stream:
        table = list(csv.DictReader(stream))
    count = 0
    for weight, bias in layers:
        variants = list(
            _a16_variant_rows(table, weight, bias, max_tokens, get_GEMM_A16W16_config)
        )
        x = torch.zeros(
            (max(variants), weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        for m in variants:
            tgemm.mm(x[:m], weight, bias, otype=torch.bfloat16)
            count += 1
            logger.info("A16 GEMM warmup M=%d N=%d K=%d", m, *weight.shape)
        torch.cuda.synchronize(weight.device)
        del x
    return count


def _bmm_variant_rows(groups, n, k, max_tokens, lookup, heuristic):
    # Split-K changes at ceil(M/tile_m) boundaries, not just powers of two.
    # Resolve every supported row count on the CPU, then launch each code
    # object once. This also includes the active table's padded-M decisions.
    seen = set()
    for m in range(1, max_tokens + 1):
        config = lookup(groups, m, n, k, w_scale_block="32x32", bpreshuffle=True)
        if config is None:
            key = ("flydsl", heuristic(groups, m, n, k), None, None)
        else:
            key = tuple(
                config.get(name)
                for name in ("libtype", "kernelName", "kernelId", "splitK")
            )
        if key not in seen:
            seen.add(key)
            yield m


@torch.no_grad()
def warmup_v41_bmms(model, max_tokens):
    """Prepare loaded V4.1 grouped FP8 output projections on gfx950."""
    layers, seen = [], set()
    for layer in model.modules():
        groups, n = getattr(layer, "groups", None), getattr(layer, "o_rank", None)
        projection = getattr(layer, "wo_a", None)
        weight = getattr(projection, "weight", None)
        scale = getattr(projection, "weight_scale", None)
        if (
            not groups
            or not n
            or weight is None
            or scale is None
            or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float8_e8m0fnu
            or not getattr(weight, "is_shuffled", False)
        ):
            continue
        key = (weight.device, groups, n, tuple(weight.shape), tuple(scale.shape))
        if key not in seen:
            seen.add(key)
            layers.append((groups, n, weight, scale))
    if not layers:
        return 0

    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        return 0
    from aiter.ops.batched_gemm_op_a8w8 import (
        batched_gemm_a8w8_mxscale_bpreshuffle,
        lookup_mxscale_bmm_config,
    )
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import pick_bmm_kernel_name

    count = 0
    for groups, n, weight, scale in layers:
        k = weight.shape[-1]
        variants = list(
            _bmm_variant_rows(
                groups,
                n,
                k,
                max_tokens,
                lookup_mxscale_bmm_config,
                pick_bmm_kernel_name,
            )
        )
        rows = max(variants)
        x = torch.zeros((rows, groups, k), dtype=weight.dtype, device=weight.device)
        xs = torch.full(
            (rows, groups, k // 32), 127, dtype=torch.uint8, device=weight.device
        )
        for m in variants:
            batched_gemm_a8w8_mxscale_bpreshuffle(
                x[:m],
                weight.view(groups, n, k),
                xs[:m],
                scale.view(groups, n // 32, k // 32),
                dtype=torch.bfloat16,
            )
            count += 1
            logger.info("BMM warmup B=%d M=%d N=%d K=%d", groups, m, n, k)
        torch.cuda.synchronize(weight.device)
        del x, xs
    return count
