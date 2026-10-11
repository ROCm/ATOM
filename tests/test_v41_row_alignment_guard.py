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


def _check(batch, computed, query_lens):
    _check_row_alignment(
        batch,
        np.asarray(computed, dtype=np.int64),
        np.asarray(query_lens, dtype=np.int64),
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


def test_the_measured_incoherent_batch_is_reported_not_raised():
    """The exact eight rows that killed a server, verbatim.

    Three of them are neither end of their own step. Raising on them cost the
    engine; the array they come from cannot support the claim that was being
    raised.
    """
    batch = _Batch([9594, 9594, 9610, 9603, 9597, 8192, 7936, 0])
    _check(batch, [9594, 9594, 9611, 9604, 9598, 17805, 7936, 0], [1, 1, 2, 2, 2, 1422, 1658, 4096])
    assert bridge._V41_ROW_CHECK_NOISY is True


def test_a_row_added_this_step_reads_zero_and_does_not_raise():
    """Measured: `batch_computed=[0] step_computed=[9598] query_lens=[8192]`
    on a ONE-row batch, where misordering is not a possible explanation."""
    _check(_Batch([0]), [9598], [8192])


def test_nothing_about_the_values_can_raise():
    """Whatever the runner writes, this must not be the thing that stops the
    server. A pathological array is still only a log line."""
    _check(_Batch([-5, 10**9, 0]), [1, 2, 3], [1, 1, 1])


def test_it_reports_only_once():
    batch = _Batch([7777])
    _check(batch, [0], [8])
    assert bridge._V41_ROW_CHECK_NOISY is True
    _check(batch, [0], [8])


def test_a_batch_with_no_computed_array_is_inert_not_fatal():
    """A runner generation this code has not seen must not stop the model."""

    class _Bare:
        req_ids = ["r0"]

    _check(_Bare(), [512], [8])


def test_a_row_count_mismatch_is_still_structural_and_raises():
    """Not a value disagreement: the two descriptions are of different batches,
    and every index into them afterwards would be meaningless."""
    with pytest.raises(ValueError, match="rows but the step describes"):
        _check_row_alignment(
            _Batch([1, 2, 3]),
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([8, 8], dtype=np.int64),
            3,
        )
