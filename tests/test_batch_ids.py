# SPDX-License-Identifier: MIT
"""The token -> sequence map, against the four transcriptions it replaced.

This map was written out four times in `attentions/` before it had a module.
Three were the same expression (`repeat(arange(bs), lens)`), one wrapped it in
a `-1`-padded buffer, and a fourth site that looks identical is NOT this map at
all -- `gdn_attn`'s `mlist` repeats over Triton *programs* per sequence, not
tokens. So what has to be checked is that the one function reproduces the three
it replaced, on padded and unpadded shapes alike.

A wrong id does not fault. It reads another request's row, and the answer stays
plausible -- which is why this is tested rather than eyeballed.

No GPU, no AITER: the module is numpy-only and loaded by path.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/model_ops/attentions/token_layout/batch_ids.py"
)
_spec = importlib.util.spec_from_file_location("_batch_ids_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
batch_id_per_token = _mod.batch_id_per_token


# ── the transcriptions this replaced, verbatim ─────────────────────────────


def ref_unpadded(seqlens_q):
    """`deepseek_v4_attn.prepare_decode` and `build_for_cudagraph_capture`."""
    return np.repeat(np.arange(len(seqlens_q), dtype=np.int32), seqlens_q)


def ref_padded(seqlens_q, padded_total_tokens):
    """`deepseek_v4_attn._attach_v4_per_fwd_meta`."""
    scheduled_tokens = int(np.sum(seqlens_q))
    out = np.full(padded_total_tokens, -1, dtype=np.int32)
    out[:scheduled_tokens] = np.repeat(
        np.arange(len(seqlens_q), dtype=np.int32), seqlens_q
    )
    return out


SHAPES = [
    [1],
    [1, 1, 1, 1],  # the ordinary decode step: one row per sequence
    [5],
    [3, 1, 4, 1, 5],  # ragged: a DSpark step verifying different draft counts
    [1024],
    [17, 1, 1, 260],
    [0, 3, 0],  # a sequence contributing no token this step
    [],
]


@pytest.mark.parametrize("seqlens", SHAPES, ids=lambda s: f"bs{len(s)}_T{sum(s)}")
def test_matches_the_unpadded_transcription(seqlens):
    sq = np.asarray(seqlens, dtype=np.int32)
    assert np.array_equal(batch_id_per_token(sq), ref_unpadded(sq))


@pytest.mark.parametrize("seqlens", SHAPES, ids=lambda s: f"bs{len(s)}_T{sum(s)}")
@pytest.mark.parametrize("slack", [0, 1, 7, 64])
def test_matches_the_padded_transcription(seqlens, slack):
    sq = np.asarray(seqlens, dtype=np.int32)
    width = int(sq.sum()) + slack
    assert np.array_equal(batch_id_per_token(sq, pad_to=width), ref_padded(sq, width))


def test_the_pad_tail_names_no_sequence():
    """A captured step runs at the bucket width whatever the batch. A padding
    token that named sequence 0 would have every consumer resolve it to that
    request's row and attend on its behalf; `-1` is what the kernels bail on."""
    got = batch_id_per_token(np.asarray([2, 1], np.int32), pad_to=8)
    assert got[:3].tolist() == [0, 0, 1]
    assert (got[3:] == -1).all()


def test_the_dtype_is_the_one_the_buffer_takes():
    """Staged into an int32 mirror; int64 here would truncate on the way in."""
    assert batch_id_per_token(np.asarray([2, 2], np.int32)).dtype == np.int32
    assert batch_id_per_token(np.asarray([2, 2], np.int64)).dtype == np.int32


def test_a_bucket_narrower_than_the_batch_is_refused():
    """Silently truncating would drop real tokens off the end of the step."""
    with pytest.raises(AssertionError):
        batch_id_per_token(np.asarray([4, 4], np.int32), pad_to=7)


def test_writes_into_a_caller_buffer_without_reaching_past_it():
    sq = np.asarray([3, 2], np.int32)
    dst = np.full(32, -7, dtype=np.int32)
    got = batch_id_per_token(sq, pad_to=8, out=dst)
    assert np.shares_memory(got, dst)
    assert got.tolist() == [0, 0, 0, 1, 1, -1, -1, -1]
    assert (dst[8:] == -7).all(), "wrote past the requested width"
