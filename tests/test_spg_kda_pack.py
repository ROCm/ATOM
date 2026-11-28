# SPDX-License-Identifier: MIT
"""The KDA prefill's q, k, v, gate and beta gathered
for its rows and heads in one kernel. The kernel runs on the CPU through
Triton's interpreter, in a child process (the flag must precede the triton
import)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the gather is a @triton.jit kernel")

ROOT = Path(__file__).resolve().parents[1]


def _module():
    """`kda_inputs` alone: the package's other modules import AITER."""
    path = ROOT / "atom/model_ops/kimi_k3/kda_inputs.py"
    spec = importlib.util.spec_from_file_location("spg_kda_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tokens=11, heads=4, dim=128, dtype=torch.bfloat16):
    """The prefill's layouts: q, k, v packed, gate a column slice of a wider
    row, beta a column slice of the projection's tail."""
    generator = torch.Generator().manual_seed(5)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    q, k, v = (rand(1, tokens, heads, dim) for _ in range(3))
    gate = rand(tokens, heads * dim + 64)[:, 64:].view(1, tokens, heads, dim)
    beta = rand(tokens, heads + 136)[:, :heads].unsqueeze(0)
    return q, k, v, gate, beta


def _tokens():
    """Rows 0..8, then two heads re-running 2..5 and 6..8."""
    return torch.cat([torch.arange(9), torch.arange(2, 6), torch.arange(6, 9)])


def _reference(tokens, q, k, v, gate, beta):
    q, k, v, gate, beta = (x.index_select(1, tokens) for x in (q, k, v, gate, beta))
    return q, k, v, gate, beta.float()


def run_kernel() -> None:
    kda_inputs = _module()
    for dtype in (torch.bfloat16, torch.float32):
        # 12 heads: K3.1's per-rank count, beta's lanes masked to 16.
        for heads in (4, 12):
            inputs = _inputs(heads=heads, dtype=dtype)
            for tokens in (_tokens(), _tokens().to(torch.int32)):
                got = kda_inputs._gather_inputs(tokens, *inputs)
                want = _reference(tokens, *inputs)
                for g, w in zip(got, want):
                    assert g.dtype == w.dtype and g.shape == w.shape
                    assert g.is_contiguous()
                    assert torch.equal(g, w), (dtype, heads, tokens.dtype)


def test_gather_matches_index_select_on_the_cpu():
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('probe', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m.run_kernel()"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]


def test_cpu_tensors_take_the_torch_ops():
    inputs = _inputs()
    got = _module().gather_kda_inputs(_tokens(), *inputs)
    for g, w in zip(got, _reference(_tokens(), *inputs)):
        assert g.dtype == w.dtype and torch.equal(g, w)


def test_dense_tokens():
    dense = _module()._dense_tokens
    q, _, _, gate, beta = _inputs()
    assert dense(q, (4, 128)) and dense(gate, (4, 128)) and dense(beta, (4,))
    assert not dense(q.transpose(2, 3), (128, 4))
    assert not dense(q[:, :, ::2], (2, 128))
    assert not dense(q.expand(2, -1, -1, -1), (4, 128))
    assert not dense(q, (4, 64))
    assert dense(torch.zeros(1, 5, 1, 8)[:, :, :, :], (1, 8))


@pytest.fixture(scope="module")
def k3():
    from aiter_stub import atom_without_aiter

    with atom_without_aiter() as load:
        yield SimpleNamespace(
            model=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3"),
        )


@pytest.mark.parametrize("switch", ["0", "1"], ids=["per-tensor", "packed"])
def test_prefill_kda_hands_the_recurrence_the_same_inputs(monkeypatch, k3, switch):
    if switch == "0":  # past `_KDA_PACK_MAX_ROWS`: torch's per-tensor gathers
        monkeypatch.setattr(k3.model, "_KDA_PACK_MAX_ROWS", -1)
    seen = {}

    def run_kda(q, k, v, gate, beta, initial, cu, final):
        seen.update(q=q, k=k, v=v, gate=gate, beta=beta, initial=initial)
        n = initial.shape[0]
        return q.new_zeros(q.shape), torch.arange(n).view(n, 1)

    layer = SimpleNamespace(_run_kda=run_kda)
    q, k, v, gate, beta = _inputs()
    tokens = _tokens()
    ckpt = {
        "kda_tokens": tokens,
        "kda_cu": torch.tensor([0, 9, 13, 16]),
        "kda_rows": torch.tensor([0, 0, 0]),
    }
    initial = torch.zeros(1, 4, 128, 128)
    out, last, heads = k3.model.KimiKDAAttention._prefill_kda(
        layer, q, k, v, gate, beta, initial, ckpt["kda_cu"], ckpt
    )
    want = _reference(tokens, q, k, v, gate, beta)
    for name, w in zip(("q", "k", "v", "gate", "beta"), want):
        got = seen[name]
        # The recurrence widens beta itself; the packed gather hands it fp32.
        if name == "beta" and switch == "0":
            got = got.float()
        assert torch.equal(got, w), name
    assert seen["beta"].dtype == (torch.float32 if switch == "1" else torch.bfloat16)
    assert out.shape == (1, 11, 4, 128)
    assert last.tolist() == [[0]] and heads.tolist() == [[1], [2]]


def test_a_long_prefill_keeps_the_per_tensor_gathers(monkeypatch, k3):
    """Past `_KDA_PACK_MAX_ROWS` recurrence rows torch's gathers are faster."""
    tokens = _tokens()
    monkeypatch.setattr(k3.model, "_KDA_PACK_MAX_ROWS", tokens.shape[0] - 1)
    seen = {}

    def run_kda(q, k, v, gate, beta, initial, cu, final):
        seen.update(q=q, beta=beta)
        n = initial.shape[0]
        return q.new_zeros(q.shape), torch.arange(n).view(n, 1)

    layer = SimpleNamespace(_run_kda=run_kda)
    q, k, v, gate, beta = _inputs()
    ckpt = {
        "kda_tokens": tokens,
        "kda_cu": torch.tensor([0, 9, 13, 16]),
        "kda_rows": torch.tensor([0, 0, 0]),
    }
    initial = torch.zeros(1, 4, 128, 128)
    k3.model.KimiKDAAttention._prefill_kda(
        layer, q, k, v, gate, beta, initial, ckpt["kda_cu"], ckpt
    )
    assert seen["beta"].dtype == torch.bfloat16
    assert torch.equal(seen["q"], q.index_select(1, tokens))
