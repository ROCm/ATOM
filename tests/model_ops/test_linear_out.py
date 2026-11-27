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


def test_fp8_projection_reuses_growing_mixed_rows(monkeypatch):
    aiter = pytest.importorskip("aiter")
    from atom.model_ops import linear
    from atom.models.deepseek_v4 import _MixedAttentionWorkspace
    from atom.quant_spec import LayerQuantConfig
    from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    target = LayerQuantConfig(
        quant_type=aiter.QuantType.per_1x128,
        quant_dtype=aiter.dtypes.fp8,
        is_dynamic=True,
    )
    config = SimpleNamespace(
        online_quant=False,
        blockscale_e8m0_scale=False,
        get_layer_quant_config=lambda *a, **kw: target,
    )
    layer = linear.ReplicatedLinear(512, 4096, quant_config=config, prefix="wq_b")
    torch.manual_seed(29)
    weight = torch.randn(4096, 512, dtype=torch.bfloat16, device="cuda") * 0.02
    layer.weight.data, layer.weight_scale.data = (
        quantize_weight_to_fp8_128x128_blockscale(weight, aiter.dtypes.fp8)
    )
    layer.process_weights_after_loading()
    workspace = _MixedAttentionWorkspace(512, 8, 512, device="cuda", fp8=True)
    workspace.q.fill_(-11)
    workspace.out.fill_(-13)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    pointers = (workspace.q.data_ptr(), workspace.out.data_ptr())
    for rows in (17, 33, 65, 129, 256):
        for start in (0, 256):
            q, _out = workspace.rows(start, rows)
            got = layer(x[:rows], out=q)
            ref, again = layer(x[:rows]), layer(x[:rows])
            spread = (again.float() - ref.float()).abs().max().item()
            assert got.data_ptr() == q.data_ptr()
            assert (got.float() - ref.float()).abs().max().item() <= 4 * spread
            assert torch.all(workspace.out == -13)  # Q and attention never alias.
            # An allocating decode projection must not overwrite the mixed rows.
            before = q.clone()
            layer(x[:8])
            torch.testing.assert_close(q, before, rtol=0, atol=0)
    assert pointers == (workspace.q.data_ptr(), workspace.out.data_ptr())
