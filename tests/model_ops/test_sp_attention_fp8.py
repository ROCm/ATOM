# SPDX-License-Identifier: MIT
"""Head-shard FP8 quantization must equal o_proj's per-token quantizer.

Single process: the amax all-gather and head exchange only move bytes, so each
rank's kernel is driven directly with the concatenated per-shard maxima.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "drives Triton and HIP quantizers; needs a GPU", allow_module_level=True
    )

from aiter import dtypes
from aiter.ops.quant import per_token_quant_hip

from atom.model_ops.sp_attention import head_amax, quantize_with_gathered_amax

if dtypes.fp8 != torch.float8_e4m3fn:
    pytest.skip("FP8 head exchange requires E4M3FN", allow_module_level=True)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_shard_quantization_matches_full_row_quantization(world, padded):
    tokens, width = 64 * world, 96
    real = tokens - (world - 1 if padded else 0)
    torch.manual_seed(world)
    full = torch.randn(tokens, world * width, device="cuda").mul_(
        torch.logspace(-3, 3, tokens, device="cuda")[:, None]
    )
    full = full.bfloat16()
    # Each row's maximum lives in a different shard.
    peaks = torch.arange(tokens, device="cuda")
    full[peaks, (peaks % world) * width] = full.abs().amax(1) * 2
    full[1] = 0
    # The divisibility pad, which the shards omit, quantizes as zero rows.
    full[real:] = 0
    expected_q, expected_scale = per_token_quant_hip(full, quant_dtype=dtypes.fp8)
    shards = [s[:real].contiguous() for s in full.chunk(world, dim=1)]
    amax = torch.cat([head_amax(s, tokens) for s in shards])
    local = tokens // world
    for rank, shard in enumerate(shards):
        scale = torch.empty((local, 1), device="cuda")
        q = quantize_with_gathered_amax(shard, amax, world, rank, scale)
        columns = slice(rank * width, (rank + 1) * width)
        assert torch.equal(
            q.view(torch.uint8), expected_q[:, columns].contiguous().view(torch.uint8)
        )
        rows = slice(rank * local, (rank + 1) * local)
        assert torch.equal(scale, expected_scale[rows].view(-1, 1))
