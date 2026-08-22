# SPDX-License-Identifier: MIT
"""The DP draft-pass lockstep contract.

Under DP attention every rank must issue the same collective sequence each
step, and an idle rank issues it from ``dummy_execution``. Draft passes are
where that broke twice: an output-less batch skipping ``propose()`` (one pass
short), then EAGLE's context pass running beside the propose that replaced it
(one pass long). Both showed up only as an eight-worker hang on a GPU node.

``ModelRunner._verify_draft_pass_count`` turns that hang into a local error, so
these tests pin the accounting itself: what the drafters declare, that a pass is
counted where it is launched, and that the check fires on a mismatch and only
when lockstep is in force.
"""

import pytest

from atom.model_engine.model_runner import ModelRunner
from atom.spec_decode.drafter import Drafter


class _FakeTokenIDProcessor:
    def __init__(self, is_deferred_out=True):
        self.is_deferred_out = is_deferred_out


class _FakeParallelConfig:
    def __init__(self, data_parallel_size):
        self.data_parallel_size = data_parallel_size


class _FakeConfig:
    def __init__(self, data_parallel_size):
        self.parallel_config = _FakeParallelConfig(data_parallel_size)


class _FakeBatch:
    def __init__(self, produces_output=True):
        self.is_dummy_run = False
        self.total_seqs_num = 2
        self.total_tokens_num = 32
        self._produces_output = produces_output

    def produces_output(self):
        return self._produces_output


class _CountingDrafter:
    """Just the base's counter surface, without building a draft model."""

    def __init__(self, declared):
        self.declared = declared
        self.reset_draft_passes = Drafter.reset_draft_passes.__get__(self)
        self.count_draft_pass = Drafter.count_draft_pass.__get__(self)
        self._draft_passes_counted = 0

    @property
    def draft_passes_per_forward(self):
        return self.declared

    @property
    def draft_passes_counted(self):
        return self._draft_passes_counted


def _runner(drafter, data_parallel_size=8, is_deferred_out=True):
    """A ModelRunner with only what the check reads -- no GPU, no model."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.label = "Model Runner0/8"
    runner.config = _FakeConfig(data_parallel_size)
    runner.tokenID_processor = _FakeTokenIDProcessor(is_deferred_out)
    if drafter is not None:
        runner.drafter = drafter
    return runner


def test_declared_count_is_a_constant_not_a_batch_property():
    """Peers hold you to the declared count and cannot see your batch."""
    for drafter_cls_name, declared in (("eagle", 3), ("dspark", 1)):
        drafter = _CountingDrafter(declared)
        assert drafter.draft_passes_per_forward == declared, drafter_cls_name


def test_base_drafter_refuses_to_guess_the_count():
    """A new drafter must declare; inheriting a default would re-open the bug."""

    class _Undeclared(_CountingDrafter):
        draft_passes_per_forward = Drafter.draft_passes_per_forward

    with pytest.raises(NotImplementedError, match="draft_passes_per_forward"):
        _ = _Undeclared(1).draft_passes_per_forward


def test_counter_resets_per_forward():
    drafter = _CountingDrafter(2)
    drafter.reset_draft_passes()
    drafter.count_draft_pass()
    drafter.count_draft_pass()
    assert drafter.draft_passes_counted == 2
    drafter.reset_draft_passes()
    assert drafter.draft_passes_counted == 0


def test_matching_count_passes():
    drafter = _CountingDrafter(3)
    runner = _runner(drafter)
    drafter.reset_draft_passes()
    for _ in range(3):
        drafter.count_draft_pass()
    runner._verify_draft_pass_count(_FakeBatch())


def test_one_pass_short_raises():
    """The first deadlock: an output-less batch skipped propose entirely."""
    drafter = _CountingDrafter(3)
    runner = _runner(drafter)
    drafter.reset_draft_passes()
    with pytest.raises(RuntimeError, match="ran 0 draft pass"):
        runner._verify_draft_pass_count(_FakeBatch(produces_output=False))


def test_one_pass_long_raises():
    """The second: EAGLE's context pass beside the propose that replaced it."""
    drafter = _CountingDrafter(3)
    runner = _runner(drafter)
    drafter.reset_draft_passes()
    for _ in range(4):
        drafter.count_draft_pass()
    with pytest.raises(RuntimeError, match="ran 4 draft pass"):
        runner._verify_draft_pass_count(_FakeBatch())


def test_skipped_without_dp():
    """At dp_size 1 an output-less batch legitimately runs no draft at all."""
    drafter = _CountingDrafter(3)
    runner = _runner(drafter, data_parallel_size=1)
    drafter.reset_draft_passes()
    runner._verify_draft_pass_count(_FakeBatch(produces_output=False))


def test_skipped_without_deferred_output():
    """PP stages are excluded -- their own return arm runs no draft."""
    drafter = _CountingDrafter(3)
    runner = _runner(drafter, is_deferred_out=False)
    drafter.reset_draft_passes()
    runner._verify_draft_pass_count(_FakeBatch())


def test_skipped_without_a_drafter():
    runner = _runner(None)
    runner._verify_draft_pass_count(_FakeBatch())
