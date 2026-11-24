# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""``_packed_max_seqlen`` must cover the trailing pad bin, not just real sequences.

Megatron sizes the RoPE table from max_seqlen, then slices freqs[:len] for each
cu_seqlens bin. A pad bin longer than max_seqlen silently yields a short slice,
and the packed freqs end up shorter than the activation. The second test here
pins that contract against Megatron's own rope kernel rather than restating the
arithmetic.
"""

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron backend is optional in CPU CI")

from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine

max_seqlen = MegatronNativeEngine._packed_max_seqlen


def _offsets(real_lens, align):
    """Offsets _pp_forward_model would build, trailing pad bin included."""
    offs = [0]
    for L in real_lens:
        offs.append(offs[-1] + L)
    pad = (-offs[-1]) % align
    if pad:
        offs.append(offs[-1] + pad)
    return offs, pad


def test_no_padding_is_just_the_longest_sequence():
    assert max_seqlen([0, 10, 25], cp=1) == 15


def test_empty_stream_is_zero_not_a_crash():
    assert max_seqlen([0], cp=1) == 0
    assert max_seqlen([], cp=1) == 0


def test_cp_scales_every_bin():
    assert max_seqlen([0, 10, 25], cp=2) == 30


@pytest.mark.parametrize("real,align", [([2], 16), ([4], 16), ([1, 1], 16), ([3], 32)])
def test_a_pad_bin_longer_than_every_sequence_still_counts(real, align):
    offs, pad = _offsets(real, align)
    assert pad > max(real), "case must actually exercise pad > max real"
    assert max_seqlen(offs, cp=1) == pad


@pytest.mark.parametrize(
    "real,align", [([1941], 16), ([1000, 941], 16), ([64, 64], 16)]
)
def test_a_short_pad_does_not_inflate_max_seqlen(real, align):
    offs, _ = _offsets(real, align)
    assert max_seqlen(offs, cp=1) == max(real)


def test_result_covers_every_bin_so_no_freqs_slice_is_short():
    # The property the rope kernel actually depends on.
    for real, align in [([2], 16), ([5, 3], 16), ([1941], 16), ([7], 32)]:
        offs, _ = _offsets(real, align)
        mx = max_seqlen(offs, cp=1)
        assert all(offs[i + 1] - offs[i] <= mx for i in range(len(offs) - 1))


class _CP:
    def size(self):
        return 1

    def rank(self):
        return 0


@pytest.mark.parametrize("real,align", [([2], 16), ([4], 16), ([1941], 16)])
def test_megatron_rope_accepts_what_this_returns(real, align):
    """End-to-end against the real kernel: the packed freqs must match the tokens."""
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_thd,
    )

    offs, _ = _offsets(real, align)
    total = offs[-1]
    mx = max_seqlen(offs, cp=1)
    cu = torch.tensor(offs, dtype=torch.int32)
    t = torch.randn(total, 1, 8)
    freqs = torch.randn(mx, 1, 1, 8)
    out = _apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=_CP())
    assert out.shape == (total, 1, 8)


def test_megatron_rope_rejects_the_real_sequence_only_max():
    """The bug, pinned: excluding the pad bin reproduces the production error."""
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_thd,
    )

    offs, _pad = _offsets([2], 16)  # -> [0, 2, 16], pad=14
    buggy_mx = 2  # max over real sequences only
    cu = torch.tensor(offs, dtype=torch.int32)
    t = torch.randn(offs[-1], 1, 8)
    freqs = torch.randn(buggy_mx, 1, 1, 8)
    with pytest.raises(RuntimeError, match=r"size of tensor a \(16\).*tensor b \(4\)"):
        _apply_rotary_pos_emb_thd(t, cu, freqs, cp_group=_CP())
