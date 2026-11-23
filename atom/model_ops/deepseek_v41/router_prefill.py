# SPDX-License-Identifier: MIT
"""V4.1 wide BF16 router: preserve tuned FT splits, round at the consumer.

Split-K and multiple K-wave accumulation round their BF16 partials before merging. For the supported
FT route, request FP32 output and hand the router's BF16 consumer one final
cast. Other tuned families keep their original implementation. Decode uses
the existing fixed-order router and never enters this helper.
"""

import functools

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

_FIELDS = (
    "block_m", "block_n", "block_k", "stages", "split_k", "m_waves",
    "n_waves", "k_waves", "group_m", "k_stagger", "b_non_temporal",
)


@functools.lru_cache(maxsize=128)
def _configuration(rows, gfx, cu_num):
    import aiter.tuned_gemm as tg

    config = tg.get_GEMM_A16W16_config(
        M=rows, N=384, K=5120, bias=False,
        dtype=str(torch.bfloat16), otype=str(torch.bfloat16),
        scaleAB=False, bpreshuffle=False, gfx=gfx, cu_num=cu_num,
    )
    params = None
    if config["libtype"] == "flydsl":
        parsed = tg._flydsl_hgemm_kernel_params(config["kernelName"])
        if (
            parsed is not None and parsed["policy"] == "ft"
            and not parsed["use_half_tile_interleaved"]
            and (parsed["split_k"] > 1 or parsed["k_waves"] > 1)
            and parsed["split_k"] == config["splitK"]
        ):
            params = {key: parsed[key] for key in _FIELDS}
    return config, params


def configuration(x):
    # Match the original GEMM's device-specific architecture/CU lookup.
    from aiter.ops.opus._arch import SUPPORTED_OPUS_ARCHES, _device_arch_and_cu
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    gfx, cu_num = _device_arch_and_cu(x.device)
    if gfx not in SUPPORTED_OPUS_ARCHES:
        gfx, cu_num = get_gfx(), get_cu_num()
    if gfx != "gfx950":
        return None, None
    return _configuration(x.shape[0], gfx, cu_num)


def compatible(x, gate):
    return (
        x.dim() == 2 and x.shape[0] > 16 and x.shape[1] == 5120
        and x.dtype == torch.bfloat16 and x.is_contiguous()
        and gate.bias is None and str(gate.quant_type) == "QuantType.No"
        and gate.weight.shape == (384, 5120)
        and gate.weight.dtype == torch.bfloat16 and gate.weight.is_contiguous()
        and not getattr(gate.weight, "is_shuffled", False)
    )


def _fake(x, weight):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)


@torch_compile_guard(gen_fake=_fake, mutates_args=[])
def v41_prefill_router_logits_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import aiter.tuned_gemm as tg
    from atom.model_ops.linear import tgemm

    _, params = configuration(x)
    if params is None:
        return tgemm.mm(x, weight, None, otype=torch.bfloat16)
    accum = tg._get_flydsl_gemm_kernels().flydsl_hgemm(
        x, weight, **params, policy="ft", out_dtype=torch.float32,
    )
    return accum.to(torch.bfloat16)
