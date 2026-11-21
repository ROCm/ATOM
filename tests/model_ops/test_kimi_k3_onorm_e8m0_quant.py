# SPDX-License-Identifier: MIT
"""KDA o_norm with E8M0 scales: byte-identical to rmsnorm_gated + o_proj's
per_1x128_e8m0_quant."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import QuantType, dtypes

from atom.model_ops.kimi_k3 import rmsnorm_gated
from atom.model_ops.kimi_k3.rmsnorm_gated_gluon import rmsnorm_gated_e8m0
from atom.model_ops.linear import per_1x128_e8m0_quant

# Decode and prefill token counts; 12 heads = 96 / TP8.
TOKENS = [1, 8, 15, 33, 135, 1009, 8065, 16384]
HEADS, DIM = 12, 128


@pytest.mark.parametrize("preshuffle", ["1", "0"])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("tokens", TOKENS)
def test_o_norm_e8m0_matches_rmsnorm_gated_then_o_proj_quant(
    tokens, is_prefill, preshuffle, monkeypatch
):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", preshuffle)
    torch.manual_seed(20261105)
    x = torch.randn(tokens, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    if tokens >= 2:
        x[0, 3] = 0  # all-zero group
        x[1, 5] *= 2.0**30
    # the gate is a column slice of the fused in_proj output, as in the model
    gate = torch.randn(tokens, HEADS, 3 * DIM, dtype=torch.bfloat16, device="cuda")
    gate = gate[..., DIM : 2 * DIM]
    weight = torch.randn(DIM, dtype=torch.bfloat16, device="cuda")

    normed = rmsnorm_gated(x, weight, gate, 1e-6, is_prefill=is_prefill)
    q_ref, s_ref = per_1x128_e8m0_quant(
        normed.reshape(tokens, HEADS * DIM), dtypes.fp8, preshuffle == "1"
    )
    q, s = rmsnorm_gated(
        x,
        weight,
        gate,
        1e-6,
        quant_type=QuantType.per_1x128,
        quant_dtype=dtypes.fp8,
        is_prefill=is_prefill,
        e8m0_scale=True,
    )
    assert s.dtype == dtypes.fp8_e8m0 and s.shape == s_ref.shape
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def _codes(s, tokens, heads, col_major):
    raw = s.view(torch.uint8).reshape(-1)
    return raw.reshape(heads, tokens).t() if col_major else raw.reshape(tokens, heads)


# 6 heads (TP16) with an odd T leaves a partial last workgroup in the row mapping.
@pytest.mark.parametrize("heads", [HEADS, 6])
@pytest.mark.parametrize("col_major", [True, False])
def test_o_norm_e8m0_is_batch_invariant(col_major, heads):
    # The decode launch packs rows per workgroup and long batches take one head of
    # consecutive tokens per workgroup; a token's bytes must not depend on which.
    torch.manual_seed(20261117)
    full = 1024
    x = torch.randn(full, heads, DIM, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(full, heads, 3 * DIM, dtype=torch.bfloat16, device="cuda")
    gate = gate[..., DIM : 2 * DIM]
    weight = torch.randn(DIM, dtype=torch.bfloat16, device="cuda")

    def run(tokens):
        q, s = rmsnorm_gated_e8m0(
            x[:tokens], weight, gate[:tokens], 1e-6, dtypes.fp8, col_major
        )
        return q.view(torch.uint8), _codes(s, tokens, heads, col_major)

    q_full, s_full = run(full)
    for tokens in (1, 3, 4, 5, 58, 545, 767, 768, 769):
        q, s = run(tokens)
        assert torch.equal(q, q_full[:tokens]), tokens
        assert torch.equal(s, s_full[:tokens]), tokens
