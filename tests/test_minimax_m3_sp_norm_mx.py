# SPDX-License-Identifier: MIT
"""MiniMax-M3 SP prefill: the hand-off norm + quant chains through aiter's `norm_mx`
are byte-equal to the norm + quant kernels they replace."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

sp = pytest.importorskip("atom.models.minimax_m3.sp_prefill")


def _norm(hidden: int):
    from atom.model_ops.layernorm import GemmaRMSNorm

    norm = GemmaRMSNorm(hidden).to("cuda").to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(5)
    norm.weight.data.copy_(
        (torch.randn(hidden, generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    )
    return norm


def _rows(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, hidden, generator=g, device="cuda") * 3).to(
        torch.bfloat16
    )


def _flat(t):
    return [t[0], *t[1], *t[2], t[3]]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.fixture
def norm_mx_available(monkeypatch):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_M3_SP_NORM_MX", True, raising=False)
    if sp._norm_mx() is None:
        pytest.skip("this aiter has no m3_tp1_prefill.norm_mx")


@pytest.mark.parametrize("rows", [32, 96, 1024, 4096])
def test_norm_quant_bytes(norm_mx_available, monkeypatch, rows):
    from atom.utils import envs

    norm = _norm(6144)
    h, res = _rows(rows, 6144, 1), _rows(rows, 6144, 2)
    got = sp._norm_quant(norm, h, res)
    monkeypatch.setattr(envs, "ATOM_M3_SP_NORM_MX", False, raising=False)
    want = sp._norm_quant(norm, h, res)
    for g, w in zip(got, want):
        assert torch.equal(_bytes(g), _bytes(w))


@pytest.mark.parametrize("rows", [32, 96, 1024, 4096])
def test_norm_quant_fp4_bytes(norm_mx_available, monkeypatch, rows):
    from atom.utils import envs

    norm = _norm(6144)
    h, res = _rows(rows, 6144, 3), _rows(rows, 6144, 4)
    got = sp._norm_quant_fp4(norm, h, res)
    monkeypatch.setattr(envs, "ATOM_M3_SP_NORM_MX", False, raising=False)
    want = sp._norm_quant_fp4(norm, h, res)
    for g, w in zip(_flat(got), _flat(want)):
        assert torch.equal(_bytes(g), _bytes(w))


def test_off_switch_and_dtype_fallback(monkeypatch):
    from atom.utils import envs

    norm = _norm(6144)
    h, res = _rows(8, 6144, 1), _rows(8, 6144, 2)
    monkeypatch.setattr(envs, "ATOM_M3_SP_NORM_MX", False, raising=False)
    assert not sp._norm_mx_ok(norm, h, res)
    monkeypatch.setattr(envs, "ATOM_M3_SP_NORM_MX", True, raising=False)
    assert not sp._norm_mx_ok(norm, h.float(), res.float())
