# SPDX-License-Identifier: MIT
"""DSpark's startup SPS sweep: table shape, cache reuse, and rank agreement.

The GPU replay itself is faked; what is checked is everything around it --
which shapes are timed, that every rank takes the same cached/measured branch,
and that a restart of the same deployment reuses the measurements.
"""

import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from atom.spec_decode import startup_sps
from atom.spec_decode.startup_sps import sps_table_from_points


def test_table_is_a_monotone_latency_envelope_over_rows():
    table = sps_table_from_points([(2, 10.0), (4, 9.0), (6, 14.0)], 6)
    latency = [1000 / v for v in table]
    assert latency[2] == pytest.approx(10.0)
    assert latency[4] == pytest.approx(10.0), "a faster larger graph is not trusted"
    assert latency[6] == pytest.approx(14.0)
    assert all(b >= a for a, b in itertools.pairwise(latency))


def test_uncovered_rows_extend_along_the_last_slope():
    points = [(6, 10.0), (12, 13.0)]
    latency = [1000 / v for v in sps_table_from_points(points, 18)]
    assert latency[18] == pytest.approx(16.0)


def _runner(tmp_path, monkeypatch, *, cached_elsewhere=True):
    monkeypatch.setenv("ATOM_DSPARK_SPS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "atom.spec_decode.calibration.deployment_identity",
        lambda config, device: {"deployment": "x"},
    )
    monkeypatch.setattr(
        "atom.utils.region_graph.TokenGraph.captured_sizes",
        staticmethod(lambda phase=None: (2, 4, 6, 8, 12, 24)),
    )
    # Two ranks: the peer reports whether ITS cache read succeeded.
    monkeypatch.setattr(
        startup_sps, "_all_true", lambda flag: flag and cached_elsewhere
    )
    monkeypatch.setattr(startup_sps, "_all_max", lambda value: value)
    monkeypatch.setattr(startup_sps, "_all_max_list", lambda values: list(values))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a: 0)
    return NS(
        config=NS(max_num_seqs=4),
        device="cpu",
        rank=0,
        capture_sizes=[4, 1, 2],
        attn_metadata_builder=NS(
            build_for_cudagraph_capture=lambda bs, max_q_len=None: None
        ),
    )


def test_a_restart_of_the_same_deployment_reuses_the_sweep(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    timed = []

    def measure(runner, shapes):
        timed.append(list(shapes))
        return [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 5.0 + b * q}
            for b, q in shapes
        ]

    monkeypatch.setattr(startup_sps, "_measure", measure)
    first = startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    # Only captured (bs, q) products, batches ascending.
    assert timed == [[(1, 6), (2, 1), (2, 3), (2, 6), (4, 1), (4, 3), (4, 6)]]
    assert len(list(tmp_path.glob("*.json"))) == 1
    second = startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert len(timed) == 1, "the cached curve must be reused"
    assert torch.equal(first, second)
    assert first.numel() == 25


def test_one_rank_missing_the_cache_makes_every_rank_measure(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 1.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    runner = _runner(tmp_path, monkeypatch, cached_elsewhere=False)
    measured = []
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: measured.append(1)
        or [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 1.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert measured == [1], "a local hit must not skip a sweep its peers run"


def test_no_token_graph_to_time_is_refused(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "atom.utils.region_graph.TokenGraph.captured_sizes",
        staticmethod(lambda phase=None: ()),
    )
    with pytest.raises(RuntimeError, match="TR_HYBRID"):
        startup_sps.startup_sps_table(runner, [1, 3, 6], 24)


def test_a_builder_without_query_buckets_times_the_full_width_only(
    tmp_path, monkeypatch
):
    """Kimi-K3's capture builder takes only `bs`: it captured, and can rebuild,
    the full width alone, so no other q is timed or asked of it."""
    runner = _runner(tmp_path, monkeypatch)
    runner.attn_metadata_builder = NS(build_for_cudagraph_capture=lambda bs: None)
    built = []

    def build_capture(builder, **kwargs):
        built.append(kwargs)
        return builder(**kwargs), NS(positions=None)

    runner._build_capture = build_capture
    timed = []
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: timed.append(list(shapes))
        or [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 5.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert timed == [[(1, 6), (2, 6), (4, 6)]]
    assert startup_sps._shapes(runner, [1, 3, 6]) == [(1, 6), (2, 6), (4, 6)]


def test_every_worker_writes_the_cache_so_each_node_can_reuse_it(tmp_path, monkeypatch):
    """A non-zero global rank (another node's worker) also persists the
    reduced curve, so its node-local cache hits on the next start."""
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a: 3)
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 5.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_cached_timings_are_reduced_across_workers(tmp_path, monkeypatch):
    """Two nodes may hold caches of different runs under one key; a hit still
    builds the table from the MAX over workers, like a fresh sweep."""
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 5.0} for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    monkeypatch.setattr(  # a peer node's cache measured 8 ms everywhere
        startup_sps, "_all_max_list", lambda values: [max(v, 8.0) for v in values]
    )
    table = startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert 1000 / float(table[6]) == pytest.approx(8.0)
