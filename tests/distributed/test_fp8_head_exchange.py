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

from atom.distributed.fp8_head_exchange import head_amax, quantize_with_gathered_amax

if dtypes.fp8 != torch.float8_e4m3fn:
    pytest.skip("FP8 head exchange requires E4M3FN", allow_module_level=True)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_shard_quantization_matches_full_row_quantization(world):
    tokens, width = 64 * world, 96
    torch.manual_seed(world)
    full = torch.randn(tokens, world * width, device="cuda").mul_(
        torch.logspace(-3, 3, tokens, device="cuda")[:, None]
    )
    full = full.bfloat16()
    # Each row's maximum lives in a different shard.
    peaks = torch.arange(tokens, device="cuda")
    full[peaks, (peaks % world) * width] = full.abs().amax(1) * 2
    full[1] = 0
    expected_q, expected_scale = per_token_quant_hip(full, quant_dtype=dtypes.fp8)
    shards = [s.contiguous() for s in full.chunk(world, dim=1)]
    amax = torch.cat([head_amax(s) for s in shards])
    local = tokens // world
    for rank, shard in enumerate(shards):
        q, scale = quantize_with_gathered_amax(shard, amax, world, rank)
        columns = slice(rank * width, (rank + 1) * width)
        assert torch.equal(
            q.view(torch.uint8), expected_q[:, columns].contiguous().view(torch.uint8)
        )
        rows = slice(rank * local, (rank + 1) * local)
        assert torch.equal(scale, expected_scale[rows].view(-1, 1))
