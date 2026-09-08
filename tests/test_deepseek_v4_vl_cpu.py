# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks for DeepSeek-V4 vision.

The in-image visibility math is pure NumPy, so it runs on the CPU-only CI
runner that `.github/scripts/run_unit_tests.sh` uses -- unlike
`tests/test_prefill_indices_paged.py`, which needs a GPU and is skipped there.
"""

import numpy as np
import pytest

from atom.models.deepseek_v4_vl import (
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    image_aware_extend_window,
    image_visible_spans,
    image_window_width,
)

VOCAB = 1000
MAX_IMG = 32
WIN = 8


def _seq(*blocks) -> np.ndarray:
    return np.array([t for b in blocks for t in b], dtype=np.int64)


def _image_block(n: int) -> list[int]:
    return [VOCAB + IMAGE_START] + [VOCAB + IMAGE] * n + [VOCAB + IMAGE_END]


def test_image_sentinel_code_agrees():
    """The literal duplicated in `deepseek_v4.py` must track the real code."""
    # `deepseek_v4` imports aiter at module scope; the rest of this file does
    # not, so only this one check is skipped on a runner without it.
    pytest.importorskip("aiter")
    from atom.models.deepseek_v4 import _IMAGE_SENTINEL

    assert _IMAGE_SENTINEL == IMAGE


def test_text_only_spans_are_all_zero():
    """No image -> (0, 0) everywhere, which collapses to the causal window."""
    ids = _seq([5, 6, 7, 8])
    left, right = image_visible_spans(ids, VOCAB, MAX_IMG)
    assert not left.any() and not right.any()

    start, count = image_aware_extend_window(
        np.arange(ids.size), left, right, WIN, WIN + MAX_IMG
    )
    pos = np.arange(ids.size)
    np.testing.assert_array_equal(start, np.maximum(pos - WIN + 1, 0))
    np.testing.assert_array_equal(count, np.minimum(pos + 1, WIN))


def test_image_end_is_inside_its_own_span():
    """`IMAGE_END` is visible from the block: its left side must be nonzero.

    It is the one token whose membership comes from the `| is_end` disjunct
    rather than the cumsum comparison, so it is easy to drop by accident.
    """
    ids = _seq([1, 2], _image_block(3), [9])
    left, right = image_visible_spans(ids, VOCAB, MAX_IMG)
    end_idx = int(np.flatnonzero(ids == VOCAB + IMAGE_END)[0])
    assert left[end_idx] > 0
    assert right[end_idx] == 0
    # The token right after the block is outside it.
    assert left[end_idx + 1] == 0 and right[end_idx + 1] == 0


def test_image_tokens_see_forward_to_the_block_end():
    ids = _seq([1, 2], _image_block(4), [9, 10])
    left, right = image_visible_spans(ids, VOCAB, MAX_IMG)
    start, count = image_aware_extend_window(
        np.arange(ids.size),
        left,
        right,
        WIN,
        image_window_width(ids.size, WIN, MAX_IMG),
    )
    start_idx = int(np.flatnonzero(ids == VOCAB + IMAGE_START)[0])
    end_idx = int(np.flatnonzero(ids == VOCAB + IMAGE_END)[0])
    for t in range(start_idx, end_idx + 1):
        last = start[t] + count[t] - 1
        assert start[t] <= start_idx, f"token {t} cannot see the block start"
        assert last >= end_idx, f"token {t} cannot see the block end"


def test_spans_do_not_cross_a_sequence_boundary():
    """Ragged batches resolve spans per sequence, not across the concatenation."""
    a = _seq([1], _image_block(2))
    b = _seq([2], _image_block(2))
    ids = np.concatenate([a, b])
    left, right = image_visible_spans(ids, VOCAB, MAX_IMG, seq_lens=[a.size, b.size])
    # The last token of sequence A is its IMAGE_END; it must not see into B.
    assert right[a.size - 1] == 0
    # Sequence B's IMAGE_START must not reach back into A.
    b_start = a.size + int(np.flatnonzero(b == VOCAB + IMAGE_START)[0])
    assert left[b_start] == 0


def test_unbalanced_span_is_rejected():
    ids = _seq([1], [VOCAB + IMAGE_START], [VOCAB + IMAGE])  # no IMAGE_END
    with pytest.raises(ValueError):
        image_visible_spans(ids, VOCAB, MAX_IMG)


def test_extend_window_is_bounded_by_the_column_budget():
    ids = _seq([1, 2], _image_block(MAX_IMG + 40), [9])
    left, right = image_visible_spans(ids, VOCAB, MAX_IMG)
    width = image_window_width(ids.size, WIN, MAX_IMG)
    _, count = image_aware_extend_window(np.arange(ids.size), left, right, WIN, width)
    assert count.max() <= width
