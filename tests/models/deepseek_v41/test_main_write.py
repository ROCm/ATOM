# SPDX-License-Identifier: MIT
"""`write_main_rows_fp4` writes the bytes `quantize_fp4` + `_scatter_rows` do."""

import types

import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import MAIN_FP4
from atom.model_ops.blockscale import quantize_fp4
from atom.model_ops.deepseek_v41.main_write import write_main_rows_fp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DIM = 512
ROW_BYTES = DIM // 2 + DIM // 16


def _plan(ratio, per_page, bs, qlen, live, columns, generator):
    """A decode-shaped plan: `live` requests' boundaries, then sentinels."""
    per_request = -(-qlen // ratio)
    plan = torch.full((bs * per_request, 4), -1, dtype=torch.int32)
    row = 0
    for batch in range(live):
        high = max(1, (columns * per_page - per_request - 2) * ratio)
        start = int(torch.randint(0, high, (), generator=generator)) // ratio
        for k in range(per_request):
            plan[row] = torch.tensor([0, batch, (start + k) * ratio, 0])
            row += 1
    return plan


@pytest.mark.parametrize(
    "ratio, per_page, bs, qlen, live, gap",
    [
        (1, 16, 128, 6, 128, 0),
        (2, 8, 128, 6, 100, 96),
        (1, 16, 32, 6, 17, 7),
        (1, 16, 4, 3000, 4, 64),
    ],
)
def test_fused_write_matches_quantize_then_scatter(
    ratio, per_page, bs, qlen, live, gap
):
    generator = torch.Generator().manual_seed(0)
    columns = 256 if bs <= 4 else 64
    pages = bs * columns + 1
    plan = _plan(ratio, per_page, bs, qlen, live, columns, generator).cuda()
    block_tables = (
        (torch.randperm(pages - 1, generator=generator)[: bs * columns] + 1)
        .view(bs, columns)
        .to(torch.int32)
        .cuda()
    )
    rows = plan.shape[0]
    x = torch.randn(rows, DIM, generator=generator) * torch.exp(
        torch.randn(rows, 1, generator=generator) * 3
    )
    x[0, :16] = 0  # a group of zeros: the scale floor
    x[1, 16:32] = 1e-30
    x = x.to(torch.bfloat16).cuda()

    # Pages with gaps between them, as a PAGE field's view has.
    page_bytes = per_page * ROW_BYTES + gap
    noise = torch.randint(0, 256, (pages * page_bytes,), dtype=torch.uint8)
    pools = []
    for _ in range(2):
        raw = noise.clone().cuda()
        view = raw.as_strided((pages, per_page, ROW_BYTES), (page_bytes, ROW_BYTES, 1))
        pools.append((raw, view))
    (expected, reference), (actual, fused) = pools

    step = types.SimpleNamespace(
        plans={ratio: types.SimpleNamespace(compress_plan_gpu=plan)},
        block_tables=block_tables,
    )
    PagedAttentionCache._scatter_rows(
        None, reference, step, quantize_fp4(x.unsqueeze(0), **MAIN_FP4), ratio
    )
    write_main_rows_fp4(
        x,
        (fused[..., : DIM // 2], fused[..., DIM // 2 :]),
        plan,
        block_tables,
        ratio=ratio,
    )
    assert not torch.equal(expected.cpu(), noise)
    assert torch.equal(actual, expected)
