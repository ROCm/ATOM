# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 NVFP4 prefill insert: ``prefill_kernels.insert_amdk`` (amd-kernels'
norm_rope_nvfp4_insert) writes the same K / V page bytes, index page bytes, q and index-q as
``aiter.fused_qknorm_idxrqknorm`` in its NVFP4 mode, bit for bit. Needs gfx950 and amd-kernels on
PYTHONPATH (skipped otherwise)."""

import pytest
import torch

pytest.importorskip("amd_kernels")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

import aiter  # noqa: E402

from atom.model_ops.minimax_m3.prefill_kernels import insert_amdk  # noqa: E402

HQ, HKV, HIDX, D, ROT, BLK = 16, 1, 1, 128, 64, 128
ROW = (HQ + 2 * HKV + HIDX + 1) * D


def _cos_sin(max_pos):
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device="cuda").float() / ROT))
    f = torch.outer(torch.arange(max_pos, device="cuda").float(), inv)
    return torch.cat([f.cos(), f.sin()], -1).to(torch.bfloat16).contiguous()


@pytest.mark.parametrize("n,start,tail_pad", [(1, 0, 0), (777, 5000, 0), (4096, 129, 3), (9000, 0, 0)])
def test_insert_matches_aiter(n, start, tail_pad):
    gen = torch.Generator(device="cuda").manual_seed(n)
    qkv = (torch.randn(n, ROW, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    w = [(0.1 * torch.randn(D, device="cuda", generator=gen)).to(torch.bfloat16) for _ in range(4)]
    pages = (start + n) // BLK + 4
    perm = torch.randperm(pages, device="cuda", generator=gen)
    tok = torch.arange(start, start + n, device="cuda")
    slots = (perm[tok // BLK] * BLK + tok % BLK).to(torch.int64)
    if tail_pad:
        slots[-tail_pad:] = -1  # padding at the batch tail
    pos = tok.to(torch.int64)
    cs = _cos_sin(start + n + 1)
    ks = torch.tensor([0.05], device="cuda")
    vs = torch.tensor([0.07], device="cuda")
    outs = []
    for writer in ("aiter", "amdk"):
        kv = torch.zeros(pages, 2 * HKV * 9216, dtype=torch.uint8, device="cuda")
        idx = torch.zeros(pages, 9216, dtype=torch.uint8, device="cuda")
        q_out = torch.empty(n, HQ * D, dtype=torch.bfloat16, device="cuda")
        iq_out = torch.empty(n, HIDX * D, dtype=torch.float8_e4m3fn, device="cuda")
        if writer == "aiter":
            aiter.fused_qknorm_idxrqknorm(
                qkv, w[0], w[1], cs, pos, HQ, HKV, ROT, 1e-6, w[2], w[3], HIDX, slots, kv, kv,
                idx, BLK, q_out, iq_out, slots, kv_cache_dtype="nvfp4",
                index_cache_dtype="nvfp4", k_scale=ks, v_scale=vs, asm_layout=False,
                skip_index_branch=False)
        else:
            insert_amdk(qkv, w[0], w[1], w[2], w[3], cs, pos, slots, kv, idx, ks, vs, q_out,
                        iq_out, HQ, HKV, HIDX, 1e-6)
        torch.cuda.synchronize()
        live = slots >= 0
        outs.append((kv, idx, q_out[live], iq_out[live].view(torch.uint8)))
    for a, b in zip(*outs):
        assert torch.equal(a, b)
