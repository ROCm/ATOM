# SPDX-License-Identifier: MIT
# The Triton MLA decode table must address KV rows past 2**31 / 576.
#
# `TritonMLAMetadataBuilder` hands aiter's `decode_attention_fwd` a dense table
# of token-level KV row ids. The kernel multiplies each id it loads by the row
# stride (576) in the id's own type. An int32 table therefore wraps for rows
# >= 3,728,270, which a pool of more than 233,016 16-token blocks reaches once it
# fills: on gfx1250 Kimi-K3 with prefix caching, that showed up as NaN decodes on
# every DP rank. With int32 ids this reads outside the buffer and can fault the
# GPU, so the test only runs the table dtype the builder actually allocates.

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aiter.ops.triton.attention.mla_decode import decode_attention_fwd

from atom.model_ops.attentions.triton_mla import TRITON_BLOCK_TABLE_DTYPE

D, DV, H, CTX, SPLITS = 576, 512, 16, 96, 4
LIMIT = 2**31 // D  # first row whose int32 byte offset wraps


def _fp8():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return torch.float8_e4m3fn if arch.startswith("gfx12") else torch.float8_e4m3fnuz


def test_decode_reads_rows_past_the_int32_offset_limit():
    torch.manual_seed(0)
    dev = "cuda"
    rows = LIMIT + 200_000
    # About 2.3 GB: one MLA layer's worth of rows past the limit.
    kv = torch.zeros(rows, 1, D, dtype=_fp8(), device=dev)
    low = torch.arange(1_000_000, 1_000_000 + CTX, device=dev)
    high = torch.arange(LIMIT + 100_000, LIMIT + 100_000 + CTX, device=dev)
    for r in (low, high):
        kv[r] = (torch.randn(CTX, 1, D, device=dev) * 0.5).to(kv.dtype)

    q = torch.randn(2, H, D, dtype=torch.bfloat16, device=dev)
    scale = D**-0.5
    table = torch.stack([low, high]).to(TRITON_BLOCK_TABLE_DTYPE)
    seq_len = torch.full((2,), CTX, dtype=torch.int32, device=dev)
    o = torch.empty(2, H, DV, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(2, H, dtype=torch.float32, device=dev)
    logits = torch.empty(2, H, SPLITS, DV + 1, dtype=torch.float32, device=dev)

    k_buffer = kv.unsqueeze(2)  # page_size 1, as attention_mla passes it
    decode_attention_fwd(
        q,
        k_buffer,
        k_buffer[..., :DV],
        o,
        lse,
        table,
        seq_len,
        logits,
        SPLITS,
        scale,
        1,
    )
    torch.cuda.synchronize()

    for i, r in enumerate((low, high)):
        k = kv[r, 0].float()
        ref = torch.softmax(q[i].float() @ k.T * scale, -1) @ k[:, :DV]
        assert torch.isfinite(o[i]).all()
        torch.testing.assert_close(o[i].float(), ref, atol=1e-2, rtol=1e-2)
