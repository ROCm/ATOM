# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The KDA gating update on aiter's gfx1250 kernel against ATOM's own kernel,
through `_kda_gating_update` as Kimi-K3 decode and spec verify call it.

The head counts are a rank's share of Kimi-K3 (96 KDA heads) and of
GLM-5.3-Flash (64 heads, the same layer) at TP 8, 4, 2 and 1, with
K = V = 128. Each case checks the `o` buffer that the call sites pass and
read afterwards, and the whole state pool, with padded rows (slot -1) at the
end of the batch as a CUDA graph pads it.

Runs on gfx1250 only, detected from torch. Importing the model there needs
aiter's entry, so a missing entry fails these tests instead of skipping them.
The routing is pinned on CPU in test_kimi_k3_kda_gating_dispatch.py.
"""

import pytest
import torch


def _gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250"


pytestmark = pytest.mark.skipif(not _gfx1250(), reason="gfx1250 only")

K = 128
LOWER_BOUND = -5.0
FILL = 7.0
# A rank's KDA heads: Kimi-K3 (96) and GLM-5.3-Flash (64) at TP 8, 4, 2 and 1.
HEADS = [8, 12, 16, 24, 32, 48, 64, 96]


@pytest.fixture(scope="module")
def gating_update():
    from atom.models.kimi_k3.model import _kda_gating_update

    return _kda_gating_update


def _inputs(heads, seqs, per_seq, slots):
    t = seqs * per_seq
    act = {"dtype": torch.bfloat16, "device": "cuda"}
    # beta is a column slice of the fused in-projection output in _forward_impl.
    fused = torch.randn(t, 4 * heads * K + heads + K, **act)
    return {
        "A_log": torch.randn(heads, device="cuda") * 0.1,
        "dt_bias": torch.randn(heads * K, device="cuda") * 0.1,
        "q": torch.randn(1, t, heads, K, **act),
        "k": torch.randn(1, t, heads, K, **act),
        "v": torch.randn(1, t, heads, K, **act),
        "a": torch.randn(1, t, heads, K, **act),
        "b": fused[:, 4 * heads * K : 4 * heads * K + heads].unsqueeze(0),
        "state": torch.randn(slots, heads, K, K, device="cuda"),
        "cu": torch.arange(0, t + 1, per_seq, dtype=torch.int32, device="cuda"),
        "out": torch.full((t, heads, K), FILL, **act),
    }


def _run(gating_update, on_aiter, inp, write, read, accepted):
    state = inp["state"].clone()
    out = inp["out"].clone()
    gating_update(
        on_aiter=on_aiter,
        A_log=inp["A_log"],
        a=inp["a"],
        b=inp["b"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        o=out,
        initial_state=state,
        cu_seqlens=inp["cu"],
        ssm_state_indices=write,
        ssm_state_indices_in=read,
        num_accepted_tokens=accepted,
        lower_bound=LOWER_BOUND,
    )
    return out, state


def _compare(gating_update, inp, write, read, accepted, real_tokens):
    o_ref, s_ref = _run(gating_update, False, inp, write, read, accepted)
    o_new, s_new = _run(gating_update, True, inp, write, read, accepted)
    # Real tokens land in the passed buffer; padded rows keep what was there.
    assert (o_new[:real_tokens] != FILL).all()
    assert (o_new[real_tokens:] == FILL).all()
    torch.testing.assert_close(o_new, o_ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(s_new, s_ref, atol=1e-3, rtol=1e-3)


def _padded(table, pad):
    table[table.size(0) - pad :] = -1
    return table


@pytest.mark.parametrize("heads", HEADS)
def test_decode_matches_atom(gating_update, heads):
    seqs, pad = 72, 2
    inp = _inputs(heads, seqs, per_seq=1, slots=seqs)
    slots = _padded(torch.arange(seqs, dtype=torch.int32, device="cuda"), pad)
    # Without a fork the model reads and writes the same table.
    _compare(gating_update, inp, slots, slots, None, seqs - pad)


@pytest.mark.parametrize("heads", HEADS)
def test_decode_fork_matches_atom(gating_update, heads):
    seqs, pad = 72, 2
    inp = _inputs(heads, seqs, per_seq=1, slots=2 * seqs)
    write = torch.arange(seqs, dtype=torch.int32, device="cuda")
    read = _padded(write + seqs, pad)
    _compare(gating_update, inp, _padded(write, pad), read, None, seqs - pad)


def _spec_tables(seqs, width, pad):
    accepted = torch.tensor([1, 3, 2, 1, 2, 3, 1, 2], dtype=torch.int32)[:seqs]
    write = torch.arange(seqs * width, dtype=torch.int32).view(seqs, width)
    return accepted.cuda(), _padded(write.cuda(), pad)


@pytest.mark.parametrize("heads", HEADS)
def test_spec_verify_fork_matches_atom(gating_update, heads):
    seqs, width, pad = 8, 3, 1
    inp = _inputs(heads, seqs, per_seq=width, slots=2 * seqs * width)
    accepted, write = _spec_tables(seqs, width, pad)
    # No row reads a slot that another row writes in the same launch.
    read = _padded(write + seqs * width, pad)
    _compare(gating_update, inp, write, read, accepted, (seqs - pad) * width)


@pytest.mark.parametrize("heads", HEADS)
def test_spec_verify_without_read_table_matches_atom(gating_update, heads):
    # A verify that is not a fork ships no read table (gdn_attn.py): each row
    # resumes from its own slot[num_accepted - 1].
    seqs, width, pad = 8, 3, 1
    inp = _inputs(heads, seqs, per_seq=width, slots=seqs * width)
    accepted, write = _spec_tables(seqs, width, pad)
    _compare(gating_update, inp, write, None, accepted, (seqs - pad) * width)
