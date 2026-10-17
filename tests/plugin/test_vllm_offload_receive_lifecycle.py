# SPDX-License-Identifier: MIT
"""PAGE/STATE completion across polls, ranks, failures and retries."""

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from atom.kv_transfer.disaggregation.types import KVConnectorOutput as AtomOutput
from atom.plugin.vllm.kv_transfer.connector import (
    AtomLMCacheOffloadConnector,
    AtomOffloadWorkerMetadata,
)
from atom.plugin.vllm.kv_transfer.kda_state import _KdaLoadResult
from atom.plugin.vllm.kv_transfer.mp_worker_adapter import MPOffloadWorkerAdapter


class Page:
    def __init__(self):
        self.out = AtomOutput()

    def get_finished(self):
        out, self.out = self.out, AtomOutput()
        return out

    def take_load_error_blocks(self):
        return set()


class DensePage(Page):
    def __init__(self):
        super().__init__()
        self.errors = set()

    def get_finished(self):
        out = super().get_finished()
        self.errors.update(str(x) for x in out.failed_loading)
        return out

    def take_load_error_reqs(self):
        out, self.errors = self.errors, set()
        return out


class Tier:
    def __init__(self):
        self.results = {}

    def take_store_reports(self):
        return {}, {}

    def take_load_results(self):
        out, self.results = self.results, {}
        return out


def connector(page=None, tier=None, groups=2):
    c = object.__new__(AtomLMCacheOffloadConnector)
    c._worker = page if page is not None else DensePage()
    c._kda_tier = tier
    c._kv_cache_config = SimpleNamespace(
        kv_cache_groups=tuple(object() for _ in range(groups))
    )
    for name in (
        "_abandoned_loads",
        "_abandoned_error_blocks",
        "_kda_error_blocks",
        "_kda_error_reqs",
        "_kda_expect",
        "_kda_dense_done",
        "_releases_in_flight",
        "_pending_failed_recving",
    ):
        setattr(c, name, set())
    for name in (
        "_worker_saved",
        "_worker_load_failed",
        "_worker_state_stored",
        "_worker_state_store_failed",
        "_worker_state_load_failed",
        "_kda_results",
        "_kda_join_waiting",
        "_promised_loads",
        "_load_failure_reports",
        "_state_load_failure_reports",
        "_save_reports",
        "_completion_reports",
    ):
        setattr(c, name, {})
    c._pending_release_ids = []
    c._worker_completions = []
    c._v41_planner = c._kda_planner = None
    c._kda_flush_stores_in_get_finished = False
    c._gpu_block_pool = None
    c._world_size = 1
    c.calls = []
    c._scheduler = SimpleNamespace(
        load_finished_by_request=lambda r: c.calls.append(("ok", r)),
        load_failed_by_request=lambda r: c.calls.append(("fail", r)),
    )
    return c


def tick(c):
    # Real model-runner order, with no replacement of these production methods.
    result = c.get_transfer_results(set())
    return KVConnectorOutput(
        finished_sending=result.finished_sending,
        finished_recving=result.finished_recving,
        failed_recving=result.failed_recving,
        invalid_block_ids=c.get_block_ids_with_load_errors(),
        kv_connector_worker_meta=c.build_connector_worker_meta(),
    )


def aggregate(aggregator, outputs):
    return aggregator.aggregate(
        [
            ModelRunnerOutput(req_ids=[], req_id_to_index={}, kv_connector_output=o)
            for o in outputs
        ]
    ).kv_connector_output


def test_abandoned_promise_reports_failure_once():
    c = connector()
    c._abandoned_loads = {"r"}
    c._abandoned_error_blocks = {12}
    out = tick(c)
    assert out.finished_recving == out.failed_recving == {"r"}
    assert not out.invalid_block_ids
    out = tick(c)
    assert not out.finished_recving and not out.failed_recving


def test_abandoned_page_still_waits_for_inflight_state():
    tier = Tier()
    c, scheduler = connector(tier=tier), connector()
    c._abandoned_loads = {"r"}
    c._kda_expect = {"r"}
    aggregator = KVOutputAggregator(expected_finished_count=1)
    raw = tick(c)
    assert raw.failed_recving == {"r"} and not raw.finished_recving
    scheduler.update_connector_output(aggregate(aggregator, [raw]))
    for _ in range(300):
        out = aggregate(aggregator, [tick(c)])
        scheduler.update_connector_output(out)
        assert not out.finished_recving and not out.failed_recving
    assert not scheduler.calls
    tier.results = {"r": _KdaLoadResult(True)}
    out = aggregate(aggregator, [tick(c)])
    scheduler.update_connector_output(out)
    assert out.finished_recving == out.failed_recving == {"r"}
    assert scheduler.calls == [("fail", "r")]
    assert not tick(c).finished_recving
    assert not c._kda_expect and not c._kda_results
    assert not scheduler._load_failure_reports
    assert not aggregator._failed_recving_pending


def test_mp_failure_without_request_drain_is_not_lost():
    page = Page()
    mp = MPOffloadWorkerAdapter(page)
    mp._load_blocks = {"r": {12}}
    assert not hasattr(mp, "take_load_error_reqs")
    c = connector(mp)
    page.out = AtomOutput(failed_loading={"r"})
    out = tick(c)
    assert out.finished_recving == out.failed_recving == {"r"}
    assert not out.invalid_block_ids
    assert not tick(c).failed_recving


@pytest.mark.parametrize("first", ["page", "state"])
@pytest.mark.parametrize("state_ok", [False, True])
def test_slow_leg_never_releases_a_half_restored_request(first, state_ok):
    page, tier = DensePage(), Tier()
    c = connector(page, tier)
    c._kda_expect = {"r"}
    result = _KdaLoadResult(state_ok, () if state_ok else (12,))
    if first == "page":
        page.out = AtomOutput(finished_loading={"r"})
    else:
        tier.results = {"r": result}
    for _ in range(300):
        out = tick(c)
        assert not out.finished_recving and not out.failed_recving
    assert "r" in c._kda_expect
    if first == "page":
        tier.results = {"r": result}
    else:
        page.out = AtomOutput(finished_loading={"r"})
    out = tick(c)
    assert out.finished_recving == {"r"}
    assert out.failed_recving == (set() if state_ok else {"r"})
    assert not tick(c).finished_recving
    assert not c._kda_expect and not c._kda_dense_done and not c._kda_results


@pytest.mark.parametrize("backend", ["inproc", "mp"])
def test_page_failure_survives_until_state_join(backend):
    page, tier = DensePage(), Tier()
    worker = page if backend == "inproc" else MPOffloadWorkerAdapter(page)
    c, scheduler = connector(worker, tier), connector()
    c._kda_expect = {"r"}
    aggregator = KVOutputAggregator(expected_finished_count=1)
    page.out = AtomOutput(failed_loading={"r"})
    raw = tick(c)
    assert raw.failed_recving == {"r"} and not raw.finished_recving
    out = aggregate(aggregator, [raw])
    scheduler.update_connector_output(out)
    assert not out.finished_recving and not out.failed_recving
    assert aggregator._failed_recving_pending == {"r"}
    assert scheduler._load_failure_reports == {"r": 1}
    for _ in range(300):
        out = aggregate(aggregator, [tick(c)])
        scheduler.update_connector_output(out)
        assert not out.finished_recving and not out.failed_recving
    tier.results = {"r": _KdaLoadResult(True)}
    out = aggregate(aggregator, [tick(c)])
    scheduler.update_connector_output(out)
    assert out.finished_recving == out.failed_recving == {"r"}
    assert scheduler.calls == [("fail", "r")]
    assert not scheduler._load_failure_reports
    assert not aggregator._failed_recving_pending


def test_scheduler_consumes_failed_receive_without_worker_meta():
    c = connector()
    c.update_connector_output(
        KVConnectorOutput(finished_recving={"r"}, failed_recving={"r"})
    )
    assert c.calls == [("fail", "r")]


def test_two_failures_do_not_poison_successful_retry():
    c = connector()
    c.update_connector_output(
        KVConnectorOutput(
            finished_recving={"r"},
            failed_recving={"r"},
            kv_connector_worker_meta=AtomOffloadWorkerMetadata(
                load_failed={"r": 1}, state_load_failed={"r": 1}
            ),
        )
    )
    assert not c._load_failure_reports and not c._state_load_failure_reports
    c.update_connector_output(KVConnectorOutput(finished_recving={"r"}))
    assert c.calls == [("fail", "r"), ("ok", "r")]


def test_four_persistent_workers_join_both_legs_before_failure_quorum():
    pages, tiers = [DensePage() for _ in range(4)], [Tier() for _ in range(4)]
    workers = [connector(page, tier) for page, tier in zip(pages, tiers)]
    for worker in workers:
        worker._kda_expect = {"r"}
    scheduler = connector()
    scheduler._world_size = 4
    aggregator = KVOutputAggregator(expected_finished_count=4)
    for step in range(305):
        if step == 0:
            pages[0].out = AtomOutput(failed_loading={"r"})
        elif step == 1:
            pages[1].out = AtomOutput(finished_loading={"r"})
        elif step == 2:
            tiers[2].results = {"r": _KdaLoadResult(True)}
        elif step == 300:
            tiers[0].results = {"r": _KdaLoadResult(True)}
        elif step == 301:
            tiers[1].results = {"r": _KdaLoadResult(True)}
        elif step == 302:
            pages[2].out = AtomOutput(finished_loading={"r"})
        elif step == 303:
            pages[3].out = AtomOutput(finished_loading={"r"})
            tiers[3].results = {"r": _KdaLoadResult(True)}
        out = aggregate(aggregator, [tick(worker) for worker in workers])
        scheduler.update_connector_output(out)
        if step == 303:
            assert out.finished_recving == out.failed_recving == {"r"}
        else:
            assert not out.finished_recving and not out.failed_recving
    assert scheduler.calls == [("fail", "r")]
    assert not scheduler._load_failure_reports
    assert not aggregator._recv_remaining_count
    assert not aggregator._failed_recving_pending


def test_single_group_keeps_its_precise_invalid_block_channel():
    c = connector(groups=1)
    c._abandoned_loads = {"r"}
    c._abandoned_error_blocks = {12}
    out = tick(c)
    assert out.finished_recving == {"r"}
    assert not out.failed_recving
    assert out.invalid_block_ids == {12}
    out = tick(c)
    assert not out.invalid_block_ids and not out.failed_recving
    assert not c._pending_failed_recving
