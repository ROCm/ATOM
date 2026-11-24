# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Streaming rollout for fully async training, driven by a fake group backend."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch
from atomrl.trainer import rl_trainer
from atomrl.trainer.fully_async_rollout import StreamingRollout
from atomrl.trainer.rl_trainer import RLTrainer

_TIMEOUT_S = 10.0


def _eventually(predicate, timeout: float = _TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class FakeBackend:
    """Prompt ``index`` is its own handle and rows; the test decides when it finishes."""

    def __init__(
        self, *, auto: bool = True, drop=lambda index: False, fail=None
    ) -> None:
        self._lock = threading.Lock()
        self.submitted: list[int] = []
        self._released: set[int] = set()
        self._auto = auto
        self._drop = drop
        self._fail = fail
        self._paused = False
        self.calls: list[str] = []

    def submit(self, index: int) -> int:
        with self._lock:
            self.submitted.append(index)
            if self._auto:
                self._released.add(index)
        return index

    def release(self, *indices: int) -> None:
        with self._lock:
            self._released.update(indices)

    def wait(self, handles, timeout):
        time.sleep(min(timeout, 0.01))
        with self._lock:
            if self._paused:
                return []
            return [h for h in handles if h in self._released]

    def finish(self, handle):
        if self._fail is not None and handle == self._fail:
            raise RuntimeError("rollout replica died")
        return None if self._drop(handle) else handle

    def pause(self) -> None:
        self._paused = True
        self.calls.append("pause")

    def resume(self) -> None:
        self._paused = False
        self.calls.append("resume")

    def count(self) -> int:
        with self._lock:
            return len(self.submitted)


def _rollout(backend, **kwargs) -> StreamingRollout:
    defaults = {
        "groups_per_step": 2,
        "sync_every": 1,
        "staleness_threshold": 0.0,
        "max_inflight": 8,
        "partial_rollout": False,
        "poll_s": 0.01,
    }
    defaults.update(kwargs)
    rollout = StreamingRollout(backend, list, **defaults)
    rollout.start()
    return rollout


def test_the_trainer_gets_groups_in_the_order_they_finish() -> None:
    backend = FakeBackend(auto=False)
    rollout = _rollout(backend, staleness_threshold=1.0)
    _eventually(lambda: backend.count() == 4)
    backend.release(3)
    time.sleep(0.05)
    backend.release(1)

    batch, _gen_s, metrics = rollout.take(0, 0)

    assert batch == [3, 1]
    assert metrics["async/staleness_max"] == 0.0
    rollout.close()


def test_the_budget_caps_how_far_generation_runs_ahead() -> None:
    backend = FakeBackend()
    rollout = _rollout(backend, staleness_threshold=0.5)  # ceil(1.5 * 2) = 3 groups

    _eventually(lambda: backend.count() == 3)
    time.sleep(0.1)
    assert backend.count() == 3
    rollout.take(0, 0)
    time.sleep(0.1)
    assert backend.count() == 3, "consuming a group must not free the budget"

    rollout.before_sync()
    rollout.after_sync(1)
    # One group carried over from the last period, so two more fit.
    _eventually(lambda: backend.count() == 5)
    time.sleep(0.1)
    assert backend.count() == 5
    rollout.close()


def test_dropped_groups_do_not_use_up_the_budget() -> None:
    backend = FakeBackend(drop=lambda index: index % 2 == 0)
    rollout = _rollout(backend)  # budget: 2 useful groups

    batch, _gen_s, metrics = rollout.take(0, 0)

    assert batch == [1, 3]
    assert metrics["async/keep_rate"] == pytest.approx(0.5, abs=0.2)
    rollout.close()


def test_a_sync_waits_for_the_groups_in_flight() -> None:
    backend = FakeBackend(auto=False)
    rollout = _rollout(backend, groups_per_step=1, staleness_threshold=1.0)
    _eventually(lambda: backend.count() == 2)
    backend.release(0)
    rollout.take(0, 0)

    done = threading.Event()
    threading.Thread(target=lambda: (rollout.before_sync(), done.set())).start()
    assert not done.wait(0.2), "group 1 is still generating"
    backend.release(1)
    assert done.wait(_TIMEOUT_S)
    assert backend.calls == []
    rollout.close()


def test_a_partial_rollout_sync_freezes_the_groups_in_flight() -> None:
    backend = FakeBackend(auto=False)
    rollout = _rollout(
        backend,
        groups_per_step=1,
        staleness_threshold=1.0,
        partial_rollout=True,
    )
    _eventually(lambda: backend.count() == 2)
    backend.release(0)
    rollout.take(0, 0)

    rollout.before_sync()
    assert backend.calls == ["pause"]
    rollout.after_sync(1)
    assert backend.calls == ["pause", "resume"]
    backend.release(1)
    _batch, _gen_s, metrics = rollout.take(1, 1)

    # Started on version 0, finished on version 1.
    assert metrics["async/partial_groups"] == 1.0
    assert metrics["async/staleness_max"] == 1.0
    rollout.close()


def test_a_rollout_failure_reaches_the_trainer() -> None:
    rollout = _rollout(FakeBackend(fail=1))

    with pytest.raises(RuntimeError, match="streaming rollout failed") as info:
        rollout.take(0, 0)

    assert "rollout replica died" in str(info.value.__cause__)
    rollout.close()


def test_too_many_dropped_groups_in_a_row_stop_the_run() -> None:
    rollout = _rollout(FakeBackend(drop=lambda index: True), max_consecutive_drops=3)

    with pytest.raises(RuntimeError) as info:
        rollout.take(0, 0)

    assert "dropped 3 prompt groups in a row" in str(info.value.__cause__)
    rollout.close()


def test_a_sync_is_due_every_sync_every_updates() -> None:
    rollout = _rollout(FakeBackend(), groups_per_step=1, sync_every=2)

    rollout.take(0, 0)
    assert not rollout.sync_due()
    rollout.take(1, 0)
    assert rollout.sync_due()
    rollout.before_sync()
    rollout.after_sync(2)
    assert not rollout.sync_due()
    rollout.close()


def test_the_budget_covers_every_update_of_a_sync_period() -> None:
    backend = FakeBackend()
    rollout = _rollout(backend, groups_per_step=1, sync_every=3)  # ceil(1 * 1 * 3)

    _eventually(lambda: backend.count() == 3)
    time.sleep(0.1)
    assert backend.count() == 3, "one period's worth of groups, no more"
    for step in range(3):
        rollout.take(step, 0)
    time.sleep(0.1)
    assert backend.count() == 3, "consuming the period must not free the budget"

    rollout.before_sync()
    rollout.after_sync(3)
    _eventually(lambda: backend.count() == 6)
    rollout.close()


def test_staleness_grows_through_a_sync_period() -> None:
    backend = FakeBackend()
    rollout = _rollout(backend, groups_per_step=1, sync_every=2)

    staleness = [rollout.take(step, 0)[2]["async/staleness_max"] for step in (0, 1)]
    rollout.before_sync()
    rollout.after_sync(2)
    after_sync = rollout.take(2, 2)[2]["async/staleness_max"]

    assert staleness == [0.0, 1.0], "the period's second update reuses its weights"
    assert after_sync == 0.0, "groups started after the sync carry the new version"
    rollout.close()


def test_the_rollout_reports_the_version_its_weights_hold() -> None:
    rollout = _rollout(FakeBackend(), start_version=3)
    assert rollout.version == 3

    rollout.before_sync()
    rollout.after_sync(4)
    assert rollout.version == 4
    rollout.close()


def test_new_groups_wait_while_submissions_are_paused() -> None:
    backend = FakeBackend(auto=False)
    rollout = _rollout(
        backend,
        groups_per_step=1,
        staleness_threshold=10.0,
        max_inflight=2,
    )
    _eventually(lambda: backend.count() == 2)

    with rollout.submissions_paused():
        backend.release(0, 1)
        time.sleep(0.2)
        assert backend.count() == 2, "both slots are free, but submissions are held"
    _eventually(lambda: backend.count() == 4)
    rollout.close()


# -- trainer side -------------------------------------------------------------


def test_merge_rollout_groups_stacks_groups_of_different_widths() -> None:
    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = SimpleNamespace(pad_token_id=9)

    def group(width: int, value: int):
        return (
            torch.full((2, width), value),
            torch.ones(2, width, dtype=torch.long),
            [1, 1],
            torch.tensor([float(value), 0.0]),
            [f"r{value}a", f"r{value}b"],
            ["gt", "gt"],
            torch.full((2, width - 1), -1.0),
            {"rollout_routing": [value, value]},
        )

    sequences, mask, prompt_lengths, rewards, responses, gts, log_probs, ragged = (
        trainer._merge_rollout_groups([group(3, 1), group(2, 2)])
    )

    assert sequences.tolist() == [[1, 1, 1], [1, 1, 1], [9, 2, 2], [9, 2, 2]]
    assert mask[2].tolist() == [0, 1, 1]
    assert prompt_lengths == [1, 1, 1, 1]
    assert rewards.tolist() == [1.0, 0.0, 2.0, 0.0]
    assert responses == ["r1a", "r1b", "r2a", "r2b"]
    assert gts == ["gt"] * 4
    assert log_probs.tolist()[2] == [0.0, -1.0]
    assert ragged == {"rollout_routing": [1, 1, 2, 2]}


def _layout_trainer(
    *,
    one_step=False,
    vllm=True,
    atom=False,
    partial=False,
    r3=False,
    pause_modes=(),
) -> RLTrainer:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        controller=SimpleNamespace(
            ray=SimpleNamespace(
                one_step_off_policy=one_step,
                fully_async=SimpleNamespace(enabled=True, partial_rollout=partial),
            )
        ),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=r3, rollout_replay=False)),
    )
    trainer._rollout_wg = object()
    trainer._ray_vllm_engine = SimpleNamespace(enable_sleep=False)
    trainer._ray_use_vllm = vllm
    trainer._ray_use_atom = atom
    trainer._ray_rollout_mgr = SimpleNamespace(
        generation_pause_modes=lambda: set(pause_modes)
    )
    return trainer


def test_layout_check_accepts_fully_async_on_a_separate_vllm_pool() -> None:
    assert _layout_trainer()._resolve_rollout_pipeline(True) == "fully_async"


def test_layout_check_accepts_atom_when_syncs_wait_for_the_groups_in_flight() -> None:
    trainer = _layout_trainer(vllm=False, atom=True, partial=False)

    assert trainer._resolve_rollout_pipeline(True) == "fully_async"


def test_layout_check_accepts_partial_rollout_on_an_atom_that_can_pause() -> None:
    trainer = _layout_trainer(
        vllm=False, atom=True, partial=True, pause_modes=("keep",)
    )

    assert trainer._resolve_rollout_pipeline(True) == "fully_async"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"one_step": True}, "are alternatives"),
        ({"vllm": False}, "needs the Ray vLLM or ATOM rollout"),
        ({"vllm": False, "atom": True, "partial": True}, "an ATOM that can pause"),
        ({"r3": True}, "R3 rollout routing"),
    ],
)
def test_layout_check_rejects_what_fully_async_cannot_run(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _layout_trainer(**kwargs)._resolve_rollout_pipeline(True)


def test_ray_backend_sends_a_group_to_the_least_loaded_replica_and_filters_it(
    monkeypatch,
) -> None:
    import ray

    class Server:
        def __init__(self, name):
            self.generate_group = SimpleNamespace(
                remote=lambda prompt, n, sp: (name, n),
            )

    accs = {"r0": [1.0, 1.0], "r1": [1.0, 0.0]}
    trainer = object.__new__(RLTrainer)
    trainer._ray_rollout_mgr = SimpleNamespace(servers=[Server("r0"), Server("r1")])
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            generation=SimpleNamespace(
                vllm_cfg=SimpleNamespace(calculate_log_probs=True)
            ),
        )
    )
    trainer._ray_sampling_params = lambda want_lp: {"max_tokens": 4}
    trainer._get_prompts_range = lambda start, count: ([f"p{start}"], [f"gt{start}"])
    trainer._ray_rollout_prompt_ids = lambda prompts: [[7, 8]]
    trainer._assemble_ray_rollout = lambda results: (
        torch.zeros(2, 3),
        torch.ones(2, 3),
        [2, 2],
        None,
        {},
    )
    trainer._compute_rewards_full = lambda seqs, mask, plen, gts, verbose: (
        torch.zeros(2),
        ["a", "b"],
        accs[current["server"]],
    )
    current = {}

    def fake_get(handle):
        current["server"] = handle[0]
        return []

    monkeypatch.setattr(ray, "get", fake_get)
    backend = rl_trainer._RayRolloutGroups(trainer, num_generations=2, use_filter=True)

    first, second = backend.submit(0), backend.submit(1)
    assert (first, second) == (("r0", 2), ("r1", 2))

    assert backend.finish(first) is None, "an all-correct group carries no signal"
    rows = backend.finish(second)
    assert rows[5] == ["gt1", "gt1"]
    assert backend.submit(2) == ("r0", 2), "r0 is idle again"
