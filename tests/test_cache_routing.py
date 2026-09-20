# SPDX-License-Identifier: MIT
"""Contracts shared by catalog producers and the Rust routing index."""

from dataclasses import replace

import pytest

from atom.cache_routing.catalog import CacheCatalog, SnapshotRequired
from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import content_keys, namespace_digest, root_key
from atom.cache_routing.planner import plan_reuse
from atom.distributed.kv_events import BlockRemoved, BlockStored

MANIFEST = {
    "model_revision": "weights@123",
    "tokenizer_revision": "tok@456",
    "template_revision": "template@789",
    "kv_semantics": "glm52-fp8-v1",
    "adapter_revision": None,
    "cache_salt": None,
    "multimodal_identity": None,
}


def catalog(world=1, **kwargs):
    cfg = CacheRoutingConfig(
        "exec",
        "http://127.0.0.1:9999",
        namespace_digest(MANIFEST),
        canonical_block_size=16,
        **kwargs,
    )
    return CacheCatalog(cfg, "layout", 64, world)


def stored(native=1, parent=None, start=0):
    return BlockStored(
        block_hashes=[native],
        parent_block_hash=parent,
        token_ids=list(range(start, start + 64)),
        block_size=64,
    )


def cpu_report(cat, rank, seq, *, epoch="boot", after=0, snapshot=True, readable=True):
    keys = content_keys(cat.config.content_namespace, list(range(256)), 16)
    return {
        "rank": rank,
        "source_epoch": epoch,
        "seq": str(seq),
        "after_seq": str(after),
        "snapshot": snapshot,
        "layout_id": "cpu-layout",
        "chunk_size": 256,
        "content_namespace": cat.config.content_namespace,
        "events": [
            {
                "chunk_id": keys[-1],
                "token_start": 0,
                "token_end": 256,
                "content_keys": keys,
                "size_bytes": 4096,
                "readable": readable,
                "parent_key": root_key(cat.config.content_namespace, 16).hex(),
            }
        ],
    }


def test_content_identity_includes_entire_prefix_and_namespace():
    ns = namespace_digest(MANIFEST)
    first = content_keys(ns, list(range(64)), 16)
    assert len(first) == 4
    assert first[-1] != content_keys(ns, [9] + list(range(1, 64)), 16)[-1]
    assert first != content_keys(
        namespace_digest({**MANIFEST, "model_revision": "other"}), list(range(64)), 16
    )
    assert content_keys(ns, list(range(15)), 16) == []
    with pytest.raises(ValueError):
        namespace_digest({**MANIFEST, "cache_salt": "tenant"})


def test_dcp_span_maps_atomically_and_ancestors_can_be_nonresident():
    cat = catalog()
    cat.hbm_events([stored(), stored(2, 1, 64)])
    assert [len(e["content_keys"]) for e in cat.snapshot()["entries"]] == [4, 4]
    cut = cat.seq
    cat.hbm_events([stored()])
    assert cat.seq == cut  # duplicate Stored is not a second replica
    cat.hbm_events([BlockRemoved(block_hashes=[1])])
    cat.hbm_events([stored(3, 2, 128)])
    assert {e["chunk_id"] for e in cat.entries.values()} == {"2", "3"}


def test_cpu_requires_all_stages_and_shards_and_independent_hbm():
    cat = catalog(world=4)
    cat.hbm_events([stored()])
    for rank in range(3):
        cat.cpu_update(cpu_report(cat, rank, 1))
    assert {e["tier"] for e in cat.entries.values()} == {"HBM"}
    cat.cpu_update(cpu_report(cat, 3, 1))
    assert {e["tier"] for e in cat.entries.values()} == {"HBM", "CPU"}
    cat.cpu_update(cpu_report(cat, 2, 2, after=1, snapshot=False, readable=False))
    assert {e["tier"] for e in cat.entries.values()} == {"HBM"}


def test_cpu_epoch_fencing_gap_and_stale(monkeypatch):
    cat = catalog()
    now = [100.0]
    monkeypatch.setattr("atom.cache_routing.catalog.time.monotonic", lambda: now[0])
    cat.cpu_update(cpu_report(cat, 0, 1))
    now[0] += 4
    assert cat.snapshot()["entries"] == ()
    cat.cpu_update(cpu_report(cat, 0, 0, epoch="new", readable=False))
    with pytest.raises(SnapshotRequired):
        cat.cpu_update(cpu_report(cat, 0, 2, epoch="boot"))
    cat.cpu_update(cpu_report(cat, 0, 1, epoch="new", after=0, snapshot=False))
    with pytest.raises(SnapshotRequired):
        cat.cpu_update(cpu_report(cat, 0, 3, epoch="new", after=2, snapshot=False))
    assert not cat.entries


def test_snapshot_cut_is_immutable_and_replay_revokes_final_eviction():
    cat = catalog()
    cat.hbm_events([stored(), stored(2, 1, 64)])
    snapshot = cat.snapshot(page_size=1)
    cat.hbm_events([BlockRemoved(block_hashes=[2])])
    second = cat.snapshot(snapshot["snapshot_id"], 1, page_size=1)
    assert second["entries"][0]["chunk_id"] == "2"
    assert second["cut_seq"] == snapshot["cut_seq"]
    replay = cat.events(cat.epoch, int(snapshot["cut_seq"]))
    assert replay["events"][-1]["events"][0]["type"] == "residency_remove"
    assert int(replay["cut_seq"]) == cat.seq


def test_replay_and_snapshot_budget_fail_closed():
    cat = catalog(max_log_bytes=1000)
    snapshot = cat.snapshot()
    cat.hbm_events([stored(), stored(2, 1, 64)])
    with pytest.raises(SnapshotRequired):
        cat.events(cat.epoch, int(snapshot["cut_seq"]))
    with pytest.raises(SnapshotRequired):
        cat.snapshot(snapshot["snapshot_id"])


@pytest.mark.parametrize(
    "hbm,cpu,minimum,kind,end",
    [
        (0, 16384, 0, "cpu", 16128),
        (4096, 16384, 8192, "cpu", 16128),
        (4112, 16384, 0, "precompute_cpu", 16128),
        (8192, 8448, 8192, "hbm", 8192),
        (4096, 0, 0, "hbm", 4096),
        (0, 256, 0, "cpu", 256),
    ],
)
def test_planner_obeys_native_chunk_threshold_and_last_token(
    hbm, cpu, minimum, kind, end
):
    plan = plan_reuse(16384, hbm, cpu, 256, minimum)
    assert (plan.kind, plan.reuse_end) == (kind, end)
    if plan.eligible:
        assert plan.load_start % 256 == 0 and plan.transfer_end % 256 == 0


def test_cross_language_golden_vectors():
    import json
    from dataclasses import asdict
    from pathlib import Path

    vectors = json.loads(
        (Path(__file__).parent / "fixtures/cache_routing_golden.json").read_text()
    )
    for row in vectors["hashes"]:
        assert (
            content_keys(row["namespace"], row["tokens"], row["block_size"])
            == row["keys"]
        )
    for row in vectors["plans"]:
        args = {k: v for k, v in row.items() if k != "plan"}
        assert asdict(plan_reuse(**args)) == row["plan"]


def test_skip_hint_does_not_invoke_lookup(monkeypatch, seq_factory):
    from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    seq = Sequence(
        list(range(1024)),
        block_size=16,
        sampling_params=SamplingParams(cache_load_policy="skip"),
    )
    scheduler = object.__new__(DenseOffloadScheduler)
    # A skip has to return before touching lookup or load lifecycle state.
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)


def test_catalog_http_snapshot_and_gap():
    import json
    from urllib.error import HTTPError
    from urllib.request import urlopen

    from atom.cache_routing.server import CatalogServer

    cat = catalog()
    cat.config = replace(cat.config, catalog_url="http://127.0.0.1:0")
    server = CatalogServer(cat)
    try:
        base = f"http://127.0.0.1:{server.server.server_port}/v1/cache"
        cat.hbm_events([stored()])
        with urlopen(base + "/snapshot") as response:
            snapshot = json.load(response)
        cat.hbm_events([BlockRemoved(block_hashes=[1])])
        with urlopen(
            base + f"/events?source_epoch={cat.epoch}&after_seq={snapshot['cut_seq']}"
        ) as response:
            assert (
                json.load(response)["events"][-1]["events"][0]["type"]
                == "residency_remove"
            )
        with pytest.raises(HTTPError) as exc:
            urlopen(base + "/events?source_epoch=old&after_seq=0")
        assert exc.value.code == 410
    finally:
        server.close()


def test_epoch_snapshot_revokes_old_cpu_chunks():
    cat = catalog()
    cat.cpu_update(cpu_report(cat, 0, 1))
    cat.cpu_update(cpu_report(cat, 0, 0, epoch="replacement", readable=False))
    assert cat.snapshot()["entries"] == ()


def test_completed_snapshots_release_their_budget():
    cat = catalog()
    for _ in range(10):
        assert cat.snapshot()["next_page_token"] is None


def test_native_reporter_incremental_binding_and_bounded_recovery(monkeypatch):
    from types import SimpleNamespace

    from atom.cache_routing.native import NativeCPUReporter

    # Public backend contract: metadata exists before ATOM learns its tokens.
    state = SimpleNamespace(seq=1, readable=True, snapshots=0)
    event = SimpleNamespace(key="native", readable=True, size_bytes=4096)

    class Backend:
        def residency_snapshot(self):
            state.snapshots += 1
            return SimpleNamespace(
                source_epoch="boot",
                cut_seq=state.seq,
                entries=[event] if state.readable else [],
            )

        def residency_events(self, epoch, after):
            return SimpleNamespace(
                source_epoch="boot",
                cut_seq=state.seq,
                events=(
                    []
                    if after == state.seq
                    else [SimpleNamespace(key="native", readable=False)]
                ),
            )

    cat = catalog()
    reports = []

    def send(url, report):
        reports.append(report)
        cat.cpu_update(report)

    monkeypatch.setattr("atom.cache_routing.native.post_cpu_report", send)
    monkeypatch.setattr(
        "atom.cache_routing.native.threading.Thread.start", lambda _: None
    )
    engine = SimpleNamespace(
        storage_manager=SimpleNamespace(
            storage_backends={"LocalCPUBackend": Backend()}
        ),
        token_database=SimpleNamespace(
            process_tokens=lambda tokens: [(0, 256, "native")]
        ),
    )
    reporter = NativeCPUReporter(
        engine, SimpleNamespace(worker_id=0, model_name="layout"), cat.config, 256
    )
    reporter.poll()
    assert not cat.snapshot()["entries"]
    reporter.bind(range(256))
    reporter.poll()
    assert len(cat.snapshot()["entries"]) == 1
    for _ in range(3):
        reporter.bind(range(256))
        reporter.poll()
        assert reports[-1]["events"] == []
    assert state.snapshots == 1
    state.seq, state.readable = 2, False
    reporter.poll()
    assert not cat.snapshot()["entries"]
    assert all(len(report["events"]) <= 256 for report in reports)


def test_native_recovery_splits_large_readable_set(monkeypatch):
    from types import SimpleNamespace

    from atom.cache_routing.native import NativeCPUReporter

    count = 600
    events = [
        SimpleNamespace(key=i, readable=True, size_bytes=4096) for i in range(count)
    ]
    backend = SimpleNamespace(
        residency_snapshot=lambda: SimpleNamespace(
            source_epoch="boot", cut_seq=count, entries=events
        )
    )
    engine = SimpleNamespace(
        storage_manager=SimpleNamespace(storage_backends={"LocalCPUBackend": backend}),
        token_database=SimpleNamespace(
            process_tokens=lambda tokens: (
                (i * 256, (i + 1) * 256, i) for i in range(count)
            )
        ),
    )
    cat = catalog()
    reports = []

    def send(url, report):
        reports.append(report)
        cat.cpu_update(report)

    monkeypatch.setattr("atom.cache_routing.native.post_cpu_report", send)
    monkeypatch.setattr(
        "atom.cache_routing.native.threading.Thread.start", lambda _: None
    )
    reporter = NativeCPUReporter(
        engine, SimpleNamespace(worker_id=0, model_name="layout"), cat.config, 256
    )
    reporter.bind(range(count * 256))
    reporter.poll()
    assert reports[0]["snapshot"] and reports[0]["events"] == []
    assert max(len(report["events"]) for report in reports) <= 256
    assert len(cat.snapshot()["entries"]) == count


@pytest.mark.parametrize(
    "extra", [{"medium": "REMOTE"}, {"lora_id": 1}, {"extra_keys": [("salt",)]}]
)
def test_nonlocal_or_unsupported_identity_does_not_advertise_hbm(extra):
    cat = catalog()
    cat.hbm_events(
        [
            BlockStored(
                block_hashes=[1],
                parent_block_hash=None,
                token_ids=list(range(64)),
                block_size=64,
                **extra,
            )
        ]
    )
    assert cat.snapshot()["entries"] == ()


def test_pp_head_publishes_committed_events_on_every_step(monkeypatch):
    from types import SimpleNamespace

    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.pp_engine_core import PPEngineCoreProc
    from atom.model_engine.scheduler import Scheduler

    cat = catalog()
    events = []
    scheduler = SimpleNamespace(
        block_manager=SimpleNamespace(take_events=lambda: list(events)),
        cache_catalog_server=SimpleNamespace(catalog=cat),
        kv_event_publisher=SimpleNamespace(publish=lambda batch: None),
    )
    core = object.__new__(PPEngineCoreProc)
    core.scheduler = SimpleNamespace(
        publish_kv_events=lambda: Scheduler.publish_kv_events(scheduler)
    )
    monkeypatch.setattr(
        PPEngineCoreProc, "_pp_head_step_inner", lambda core: events.append(stored())
    )
    core._pp_head_step()
    assert len(cat.snapshot()["entries"]) == 1
