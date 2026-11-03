# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 decoder request-graph buckets (`decoder_graph_sizes`) and the
`ATOM_DECODER_GRAPH_SIZES` parsing they read."""

import numpy as np
import pytest

from atom.model_ops.attentions.deepseek_v41.decoder_buckets import (
    DECODER_GRAPH_DEFAULT_MAX_BS,
    decoder_graph_sizes,
)
from atom.utils import envs

# serve recipes' ladder at --max-num-seqs 64
LADDER = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 40, 48, 56, 64]
FIT = {"max_bs": 64, "max_tokens": 16384, "window": 128}


def _running_bs(capture_sizes, unified_bs):
    """`ForwardMode.decide`'s rounding: the smallest capture size holding the
    group's batch, the batch itself past the ladder."""
    ladder = np.asarray(sorted(capture_sizes), dtype=np.int32)
    rung = int(np.searchsorted(ladder, unified_bs))
    return int(ladder[rung]) if rung < len(ladder) else unified_bs


def test_unset_records_the_ladder_up_to_16():
    assert decoder_graph_sizes(None, LADDER, **FIT) == (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)


def test_unset_covers_every_running_bs_up_to_16():
    sizes = set(decoder_graph_sizes(None, LADDER, **FIT))
    reachable = {_running_bs(LADDER, bs) for bs in range(1, 65)}
    assert {bs for bs in reachable if bs <= DECODER_GRAPH_DEFAULT_MAX_BS} <= sizes
    # ...and records nothing a step cannot settle on.
    assert sizes <= reachable


def test_old_default_left_reachable_buckets_eager():
    old = set(decoder_graph_sizes((1, 2, 3, 4, 8, 16), LADDER, **FIT))
    reachable = {_running_bs(LADDER, bs) for bs in range(1, 17)}
    assert reachable - old == {5, 6, 7, 12}


def test_unset_follows_a_custom_ladder():
    assert decoder_graph_sizes(None, [64, 16, 8, 4, 2, 1], **FIT) == (1, 2, 4, 8, 16)


def test_unset_without_a_ladder_records_nothing():
    # The runner's eager placeholder ladder.
    assert decoder_graph_sizes(None, [0], **FIT) == ()


def test_explicit_sizes_are_honored_and_bounded():
    assert decoder_graph_sizes((16, 3, 1), LADDER, **FIT) == (1, 3, 16)
    assert decoder_graph_sizes((), LADDER, **FIT) == ()
    # No more requests than a batch, one window each within the token buffer.
    assert decoder_graph_sizes(
        (4, 8, 16), LADDER, max_bs=8, max_tokens=16384, window=128
    ) == (4, 8)
    assert decoder_graph_sizes(
        (4, 8, 16), LADDER, max_bs=64, max_tokens=1024, window=128
    ) == (4, 8)


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", ()), ("1,2,4", (1, 2, 4)), (" 1, 8 ,", (1, 8))],
)
def test_env_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ATOM_DECODER_GRAPH_SIZES", raising=False)
    else:
        monkeypatch.setenv("ATOM_DECODER_GRAPH_SIZES", value)
    assert envs.ATOM_DECODER_GRAPH_SIZES == expected
