# SPDX-License-Identifier: MIT
"""Compare combined QK/RoPE + masked stores with independent segment launches."""

import pytest
import torch

pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.mark.parametrize("prefill_tokens", [1, 7, 32])
def test_combined_qkn_matches_split_and_keeps_prefill_swa(prefill_tokens):
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import (
        qk_norm_rope_maybe_quant,
    )

    torch.manual_seed(23)
    n = prefill_tokens + 8
    q = torch.randn(n, 64 * 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    angles = torch.randn(128, 32, device="cuda")
    cos_cache = angles.cos().to(torch.bfloat16)
    sin_cache = angles.sin().to(torch.bfloat16)
    positions = torch.arange(n, device="cuda", dtype=torch.int64)
    swa_before = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda")
    combined_swa, split_swa = swa_before.clone(), swa_before.clone()
    dest = torch.arange(96, 104, dtype=torch.int32, device="cuda")
    rows = torch.arange(8, dtype=torch.int32, device="cuda")

    def run(sl, swa=None, dest_rows=None, batch_rows=None):
        return qk_norm_rope_maybe_quant(
            q[sl],
            kv[sl],
            weight,
            cos_cache,
            sin_cache,
            positions[sl],
            64,
            512,
            64,
            1e-6,
            swa_kv=swa,
            swa_dest_rows=dest_rows,
            batch_id_per_q_token=batch_rows,
        )

    prefill = run(slice(0, prefill_tokens))
    decode = run(slice(prefill_tokens, n), split_swa, dest, rows)
    combined = run(
        slice(0, n),
        combined_swa,
        torch.cat([dest.new_zeros(prefill_tokens), dest]),
        torch.cat([rows.new_full((prefill_tokens,), -1), rows]),
    )
    torch.testing.assert_close(
        combined.q_sa, torch.cat([prefill.q_sa, decode.q_sa]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        combined.kv, torch.cat([prefill.kv, decode.kv]), rtol=0, atol=0
    )
    torch.testing.assert_close(combined_swa, split_swa, rtol=0, atol=0)
    torch.testing.assert_close(combined_swa[:96], swa_before[:96], rtol=0, atol=0)
