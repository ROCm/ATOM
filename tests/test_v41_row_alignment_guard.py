"""The V4.1 row-alignment check reports; it does not refuse.

It was written to catch `req_ids` and block tables addressing different
requests than the positions do, and to raise before that became wrong output.
Measured on vLLM 0.31's V2 runner, the array it reads
(`num_computed_tokens_np`) is not a coherent per-step snapshot: one eight-row
batch held this step's start, one token below it, a 9,613-token-stale
frontier, and a zero for a row added that step. The two arrays were not
permutations of each other, so reordering could not explain the difference --
which is the whole premise of the check.

So it counts and logs once. The contract that actually refuses a state slot
keyed by the wrong request is the cursor check in the attention leg, which is
compared against real bytes. These tests pin that this one never raises on a
value, whatever the runner hands it.
"""

import numpy as np
import pytest

from atom.plugin.vllm import deepseek_v41_bridge as bridge
from atom.plugin.vllm.deepseek_v41_bridge import _check_row_alignment


class _Batch:
    def __init__(self, computed, attr="num_computed_tokens_np"):
        setattr(self, attr, np.asarray(computed, dtype=np.int64))
        self.req_ids = [f"r{i}" for i in range(len(computed))]


def _check(batch, computed, query_lens=None):
    """`query_lens` is accepted and ignored: the guard stopped taking it when
    the sibling branch narrowed it to the two arrays it actually compares,
    and the callers here still read better naming what the batch looked
    like."""
    _check_row_alignment(
        batch,
        np.asarray(computed, dtype=np.int64),
        len(computed),
    )


@pytest.fixture(autouse=True)
def _reset_one_shot():
    bridge._V41_ROW_CHECK_NOISY = False
    yield
    bridge._V41_ROW_CHECK_NOISY = False


def test_the_pre_step_convention_is_quiet():
    """V1's InputBatch holds the count BEFORE the step."""
    _check(_Batch([512, 1024]), [512, 1024], [8, 8])
    assert bridge._V41_ROW_CHECK_NOISY is False


def test_the_post_step_convention_is_quiet():
    """V2's holds it after. Naming one of them names every row on the other."""
    _check(_Batch([520, 1032]), [512, 1024], [8, 8])
    assert bridge._V41_ROW_CHECK_NOISY is False


def test_a_disagreement_raises_and_names_the_row():
    """The sibling branch restored raising here, and this follows it.

    These two tests used to assert a one-shot log instead, because this
    guard had been measured firing on batches that were NOT misordered --
    the two arrays were not permutations of each other, so row order could
    not explain the difference. That downgrade is gone with the refactor,
    so the behaviour under test is raising; whether it still fires
    spuriously on the offload path is a question for a live run, not for
    this file.
    """
    batch = _Batch([0, 256, 512])
    batch.num_computed_tokens_cpu = np.asarray([0, 999, 512], dtype=np.int64)
    with pytest.raises(ValueError, match="disagree at row 1"):
        _check(batch, [0, 256, 512])


def test_an_agreeing_batch_passes():
    batch = _Batch([0, 256, 512])
    batch.num_computed_tokens_cpu = np.asarray([0, 256, 512], dtype=np.int64)
    _check(batch, [0, 256, 512])


def test_a_row_count_mismatch_is_still_structural_and_raises():
    """Not a value disagreement: the two descriptions are of different batches,
    and every index into them afterwards would be meaningless."""
    batch = _Batch([1, 2, 3])
    batch.num_computed_tokens_cpu = np.asarray([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="rows but the step describes"):
        _check_row_alignment(batch, np.asarray([1, 2], dtype=np.int64), 3)
