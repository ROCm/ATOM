# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The KDA gating update on aiter's gfx1250 kernel against ATOM's own kernel,
through `_kda_gating_update` as Kimi-K3 decode and spec verify call it.

Each case compares the `o` buffer that the call sites pass and read afterwards,
and the whole state pool, with the last row padded (slot -1) as a CUDA graph
pads the batch.

Runs on gfx1250 only, detected from torch. Importing the model there needs
aiter's entry, so a missing entry fails these tests instead of skipping them.
"""

import pytest
import torch


def _gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250"


pytestmark = pytest.mark.skipif(not _gfx1250(), reason="gfx1250 only")

K = 128
# A rank's share of GLM-5.3-Flash's 64 KDA heads at TP 8, and of Kimi-K3's 96
# at TP 8, 4 and 2.
HEADS = [8, 12, 24, 48]


@pytest.fixture(scope="module")
def gating_update():
    from atom.models.kimi_k3.model import _kda_gating_update

    return _kda_gating_update


def _padded(table):
    table[-1] = -1
    return table


def _compare(gating_update, heads, write, read, accepted, tokens_per_seq):
    torch.manual_seed(heads)
    t = write.size(0) * tokens_per_seq
    act = {"dtype": torch.bfloat16, "device": "cuda"}
    # beta is a column slice of the fused in-projection output in _forward_impl.
    fused = torch.randn(t, 4 * heads * K + heads + K, **act)
    args = {
        "A_log": torch.randn(heads, device="cuda") * 0.1,
        "a": torch.randn(1, t, heads, K, **act),
        "b": fused[:, 4 * heads * K : 4 * heads * K + heads].unsqueeze(0),
        "dt_bias": torch.randn(heads * K, device="cuda") * 0.1,
        "q": torch.randn(1, t, heads, K, **act),
        "k": torch.randn(1, t, heads, K, **act),
        "v": torch.randn(1, t, heads, K, **act),
        "cu_seqlens": torch.arange(
            0, t + 1, tokens_per_seq, dtype=torch.int32, device="cuda"
        ),
        "ssm_state_indices": write,
        "ssm_state_indices_in": read,
        "num_accepted_tokens": accepted,
        "lower_bound": -5.0,
    }
    # Room for a fork's read slots after the write slots.
    state = torch.randn(2 * write.numel(), heads, K, K, device="cuda")
    # Prefilled, so the comparison also covers which rows a kernel leaves alone.
    out = torch.full((t, heads, K), 7.0, **act)

    def run(on_aiter):
        o, s = out.clone(), state.clone()
        gating_update(on_aiter=on_aiter, o=o, initial_state=s, **args)
        return o, s

    (o_ref, s_ref), (o_new, s_new) = run(False), run(True)
    torch.testing.assert_close(o_new, o_ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(s_new, s_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("fork", [False, True], ids=["in-place", "fork"])
def test_decode_matches_atom(gating_update, heads, fork):
    seqs = 72
    write = _padded(torch.arange(seqs, dtype=torch.int32, device="cuda"))
    # Without a fork the model passes the write table as the read table.
    read = _padded(write + seqs) if fork else write
    _compare(gating_update, heads, write, read, None, tokens_per_seq=1)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("fork", [False, True], ids=["no-read-table", "fork"])
def test_spec_verify_matches_atom(gating_update, heads, fork):
    seqs, width = 8, 3
    accepted = torch.tensor([1, 3, 2, 1, 2, 3, 1, 2], dtype=torch.int32).cuda()
    write = torch.arange(seqs * width, dtype=torch.int32, device="cuda")
    write = _padded(write.view(seqs, width))
    # Without a fork a verify ships no read table (gdn_attn.py), and each row
    # resumes from its own slot[num_accepted - 1]. A fork reads slots that no
    # row writes in the same launch.
    read = _padded(write + seqs * width) if fork else None
    _compare(gating_update, heads, write, read, accepted, tokens_per_seq=width)
