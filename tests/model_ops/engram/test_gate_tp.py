# SPDX-License-Identifier: MIT
"""Branch ownership, rank-major scalar gates, and strided V reconstruction."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires GPU", allow_module_level=True)

from atom.model_ops.engram.device.gate import (
    engram_post_wkv,
    engram_post_wkv_reference,
    engram_post_wkv_tp,
)


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("shape", [(3, 4, 5120), (2, 3, 4, 64)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rotated_value", [False, True])
def test_tp_gate_and_complete_residual(
    tp, shape, fused, masked, rotated_value, monkeypatch
):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", str(int(fused)))
    torch.manual_seed(713)
    hc, dim = shape[-2:]
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(*shape[:-2], (hc + 1) * dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hc, dim, device="cuda")
    mask = torch.rand(shape[:-2], device="cuda") > 0.3 if masked else None
    # Independent FP64 gate oracle; round to the communicated FP32 dtype.
    h = hidden.double()
    k = kv[..., : hc * dim].reshape_as(hidden).double()
    dot = ((h * weight.double()) * k).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + 1e-20)
    dot *= torch.rsqrt(k.square().mean(-1) + 1e-20) * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).float()
    if mask is not None:
        gate.masked_fill_(~mask.unsqueeze(-1), 0)
    c = hc // tp
    gathered = torch.cat(
        [gate[..., r * c : (r + 1) * c].reshape(-1, c) for r in range(tp)]
    )
    expected = (
        hidden.float() + gate.unsqueeze(-1) * kv[..., hc * dim :].float().unsqueeze(-2)
    ).to(hidden.dtype)
    for rank in range(tp):
        value_start = rank * (dim // tp) if rotated_value else 0
        part = torch.cat(
            (
                kv[..., rank * c * dim : (rank + 1) * c * dim],
                torch.roll(kv[..., hc * dim :], -value_start, -1),
            ),
            dim=-1,
        )

        def all_gather(local, *, dim, use_custom, rank=rank):
            assert local.dtype == torch.float32 and dim == 0
            assert not use_custom  # This test supplies a mock collective.
            torch.testing.assert_close(
                local,
                gate[..., rank * c : (rank + 1) * c].reshape(-1, c),
                rtol=2e-5,
                atol=2e-6,
            )
            return gathered

        group = SimpleNamespace(
            world_size=tp, rank_in_group=rank, all_gather=all_gather
        )
        actual = engram_post_wkv_tp(
            hidden, part, weight, group, mask, value_start=value_start
        )
        assert torch.equal(actual, expected)
        if mask is not None:
            assert torch.equal(actual[~mask], hidden[~mask])
    # The original full path remains the numerical reference at TP1.
    torch.testing.assert_close(
        expected,
        engram_post_wkv_reference(hidden, kv, weight, mask),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("shape", [(1, 4, 5120), (15, 4, 5120), (2, 3, 4, 128)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_gathered_key_value_shards_need_no_reorder(
    tp, shape, fused, masked, monkeypatch
):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", str(int(fused)))
    torch.manual_seed(914)
    hc, dim = shape[-2:]
    h = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(*shape[:-2], (hc + 1) * dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hc, dim, device="cuda")
    mask = torch.rand(shape[:-2], device="cuda") > 0.4 if masked else None
    c, v = hc // tp * dim, dim // tp
    gathered = torch.cat(
        [
            torch.cat(
                (
                    kv[..., r * c : (r + 1) * c],
                    kv[..., hc * dim + r * v : hc * dim + (r + 1) * v],
                ),
                -1,
            )
            for r in range(tp)
        ],
        -1,
    )
    expected = engram_post_wkv(h, kv, w, mask)
    actual = engram_post_wkv(h, gathered, w, mask, kv_tp_size=tp)
    assert torch.equal(actual, expected)
    if mask is not None:
        assert torch.equal(actual[~mask], h[~mask])


def test_empty_and_invalid_local_projection():
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    h = torch.empty(0, 4, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(4, 64, device="cuda")
    assert (
        engram_post_wkv_tp(h, torch.empty(0, 128, device="cuda"), weight, group).shape
        == h.shape
    )
    with pytest.raises(ValueError, match="local KV"):
        engram_post_wkv_tp(h, torch.empty(0, 256, device="cuda"), weight, group)
    with pytest.raises(ValueError, match="mask"):
        engram_post_wkv_tp(
            h,
            torch.empty(0, 128, device="cuda"),
            weight,
            group,
            torch.ones(1, device="cuda", dtype=torch.bool),
        )
