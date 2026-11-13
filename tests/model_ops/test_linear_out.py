# SPDX-License-Identifier: MIT
"""`Linear.forward(out=)` on the unquantised GEMM: the result is written into
the caller's tensor, as a graph boundary reads it at one address."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU GEMM")


@pytest.mark.parametrize("rows", [1, 8, 384])
def test_an_unquantised_linear_writes_its_result_into_out(monkeypatch, rows):
    pytest.importorskip("aiter")
    from atom.model_ops import linear

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    layer = linear.ReplicatedLinear(
        7168, 64, bias=False, quant_config=None, prefix="weights_proj"
    )
    torch.manual_seed(0)
    layer.weight.data = torch.randn(64, 7168, dtype=torch.bfloat16)
    layer = layer.cuda()
    x = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)

    out = torch.full((rows, 64), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert layer.supports_out()
    got = layer(x, out=out)

    assert got.data_ptr() == out.data_ptr()
    # A split-K backend accumulates in no fixed order, so two plain calls can
    # differ; the result written into `out` may differ from one by no more.
    ref, again = layer(x), layer(x)
    spread = (again.float() - ref.float()).abs().max().item()
    assert not out.isnan().any()
    assert (out.float() - ref.float()).abs().max().item() <= 4 * spread
