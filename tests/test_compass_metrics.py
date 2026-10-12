"""Compass forward rows and op-window parsing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.metrics import compass
from atom.metrics.compass import CompassRecorder, window_op_rows

CPU = torch.autograd.DeviceType.CPU
CUDA = torch.autograd.DeviceType.CUDA


class Ev:
    def __init__(self, name, device, start, end, corr, linked=0, shapes=(), ua=False):
        self._v = (name, device, start, end, corr, linked, list(shapes), ua)

    def name(self):
        return self._v[0]

    def device_type(self):
        return self._v[1]

    def start_ns(self):
        return self._v[2]

    def end_ns(self):
        return self._v[3]

    def duration_ns(self):
        return self._v[3] - self._v[2]

    def correlation_id(self):
        return self._v[4]

    def linked_correlation_id(self):
        return self._v[5]

    def shapes(self):
        return self._v[6]

    def dtypes(self):
        return ["float"] * len(self._v[6])

    def is_user_annotation(self):
        return self._v[7]

    def device_resource_id(self):
        return 7


def test_kernels_group_by_op_and_step():
    events = [
        Ev("compass_step:4", CPU, 100, 200, 1, ua=True),
        Ev("compass_step:4", CUDA, 150, 260, 1, ua=True),
        # aten::mm (op corr 3) launches two kernels via runtime corr 10, 11.
        Ev("aten::mm", CPU, 110, 130, 3, shapes=[[8, 16], [16, 32]]),
        Ev("hipLaunchKernel", CPU, 112, 114, 10, linked=3),
        Ev("hipLaunchKernel", CPU, 120, 122, 11, linked=3),
        Ev("gemm_a", CUDA, 150, 170, 10, linked=3),
        Ev("gemm_b", CUDA, 172, 180, 11, linked=3),
        # A graph replay: kernels link to no op, only to the launch.
        Ev("hipGraphLaunch", CPU, 140, 150, 12),
        Ev("graph_k0", CUDA, 190, 200, 12),
        Ev("graph_k1", CUDA, 201, 260, 12),
        # Outside any step: dropped.
        Ev("aten::argmax", CPU, 300, 310, 9, shapes=[[8, 32]]),
        Ev("hipLaunchKernel", CPU, 301, 302, 13, linked=9),
        Ev("argmax_k", CUDA, 400, 410, 13, linked=9),
    ]
    rows = window_op_rows(events)
    assert [r["op"] for r in rows] == ["aten::mm", "hipGraphLaunch"]
    mm, graph = rows
    assert mm["step"] == 4 and graph["step"] == 4
    assert mm["shapes"] == [[8, 16], [16, 32]]
    assert (mm["gpu_start_ns"], mm["gpu_end_ns"], mm["gpu_kernel_ns"]) == (150, 180, 28)
    assert [k["name"] for k in mm["kernels"]] == ["gemm_a", "gemm_b"]
    assert not mm["graph_replay"]
    assert graph["graph_replay"] and graph["shapes"] == []
    assert (graph["gpu_start_ns"], graph["gpu_end_ns"]) == (190, 260)
    assert len(graph["kernels"]) == 2


def test_busy_time_counts_overlapping_kernels_once():
    events = [
        Ev("compass_step:0", CPU, 0, 100, 1, ua=True),
        Ev("aten::mm", CPU, 10, 20, 3, shapes=[[2, 2]]),
        Ev("hipLaunchKernel", CPU, 11, 12, 10, linked=3),
        Ev("hipLaunchKernel", CPU, 13, 14, 11, linked=3),
        Ev("hipLaunchKernel", CPU, 15, 16, 12, linked=3),
        Ev("k0", CUDA, 100, 150, 10, linked=3),
        Ev("k1", CUDA, 120, 170, 11, linked=3),  # overlaps k0 on another stream
        Ev("k2", CUDA, 200, 210, 12, linked=3),
    ]
    (row,) = window_op_rows(events)
    assert row["gpu_kernel_ns"] == 110
    assert row["gpu_busy_ns"] == 80


def _template_events():
    """The eager warmup of graph bs1_q1: op A launches k0 and k1, op B k2."""
    return [
        Ev(compass._TEMPLATE_MARK, CPU, 0, 100, 1, ua=True),
        Ev("op::A", CPU, 10, 20, 3, shapes=[[1, 64]]),
        Ev("hipLaunchKernel", CPU, 11, 12, 10, linked=3),
        Ev("hipLaunchKernel", CPU, 13, 14, 11, linked=3),
        Ev("op::B", CPU, 30, 40, 4, shapes=[[1, 128]]),
        Ev("hipLaunchKernel", CPU, 31, 32, 12, linked=4),
        Ev("k0", CUDA, 50, 60, 10, linked=3),
        Ev("k1", CUDA, 61, 70, 11, linked=3),
        Ev("k2", CUDA, 71, 80, 12, linked=4),
        # After the warmup forward: not part of the template.
        Ev("aten::zero_", CPU, 200, 210, 5, shapes=[[1]]),
        Ev("hipLaunchKernel", CPU, 201, 202, 13, linked=5),
        Ev("zero_k", CUDA, 300, 301, 13, linked=5),
    ]


def test_graph_template_lists_warmup_kernels_with_their_ops():
    template = compass.graph_template_entries(_template_events())
    assert template["kernels"] == ["k0", "k1", "k2"]
    assert template["op_index"] == [0, 0, 1]
    assert [o["op"] for o in template["ops"]] == ["op::A", "op::B"]
    assert template["ops"][1]["shapes"] == [[1, 128]]


def test_graph_replay_kernels_take_their_template_ops():
    templates = {"bs1_q1": compass.graph_template_entries(_template_events())}
    events = [
        Ev("compass_step:7", CPU, 0, 500, 1, ua=True),
        Ev("compass_graph:bs1_q1", CPU, 10, 30, 2, ua=True),
        Ev("hipGraphLaunch", CPU, 12, 20, 20),
        Ev("k0", CUDA, 100, 110, 20),
        Ev("k1", CUDA, 111, 120, 20),
        Ev("extra", CUDA, 121, 125, 20),  # not in the template
        Ev("k2", CUDA, 126, 140, 20),
        # A graph launch outside a compass_graph mark keeps one row.
        Ev("hipGraphLaunch", CPU, 40, 45, 21),
        Ev("d0", CUDA, 200, 210, 21),
    ]
    rows = window_op_rows(events, templates)
    by_op = {r["op"]: r for r in rows}
    assert sorted(by_op) == ["hipGraphLaunch", "op::A", "op::B"]
    a, b = by_op["op::A"], by_op["op::B"]
    assert a["shapes"] == [[1, 64]] and a["shapes_from"] == "graph_template"
    assert [k["name"] for k in a["kernels"]] == ["k0", "k1"]
    assert (a["cpu_start_ns"], a["cpu_end_ns"]) == (12, 20)  # the graph launch
    assert [k["name"] for k in b["kernels"]] == ["k2"]
    assert a["graph"] == b["graph"] == "bs1_q1" and a["graph_replay"]
    rest = [r for r in rows if r["op"] == "hipGraphLaunch"]
    assert sorted(k["name"] for r in rest for k in r["kernels"]) == ["d0", "extra"]
    assert all(r["shapes"] == [] and "shapes_from" not in r for r in rest)


class FakeEvent:
    """A device event whose GPU time is one millisecond per record()."""

    clock = 0

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        FakeEvent.clock += 1
        self.t = FakeEvent.clock

    def query(self):
        return self.t is not None

    def elapsed_time(self, other):
        return float(other.t - self.t)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    FakeEvent.clock = 0
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    context = SimpleNamespace(
        is_prefill=False,
        is_dummy_run=False,
        forward_mode=None,
        scheduled_bs=4,
        scheduled_tokens=4,
        running_bs=4,
        running_tokens=4,
    )
    monkeypatch.setattr(
        compass, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    rec = CompassRecorder(str(tmp_path), op_steps=0, anchor_interval_s=1e9)

    def rows():
        rec.close()
        (path,) = (tmp_path / "dp0_pp0_tp0").glob("forward.*.jsonl")
        return [json.loads(line) for line in path.read_text().splitlines()]

    return rec, rows


def test_step_times_the_whole_forward(recorder):
    rec, rows = recorder
    with rec.measure_forward():
        FakeEvent().record()  # input preparation
        with rec.measure(None):  # run_model
            pass
        with rec.measure(None, "draft"):
            pass
        FakeEvent().record()  # sampling
    target, draft = sorted(rows(), key=lambda r: r["phase"] != "target")
    assert (target["step"], target["scope"]) == (0, "forward")
    assert target["gpu_ms"] == 7.0
    assert target["model_gpu_ms"] == 1.0
    assert target["gpu_start_ns"] < target["model_gpu_start_ns"]
    assert target["model_gpu_end_ns"] < target["gpu_end_ns"]
    assert target["scheduled_bs"] == 4
    assert (draft["step"], draft["phase"], draft["scope"]) == (0, "draft", "call")
    assert draft["gpu_ms"] == 1.0


def test_run_model_outside_forward_is_its_own_step(recorder):
    rec, rows = recorder
    with rec.measure_forward(), rec.measure(None):
        pass
    with rec.measure(None):  # RapidServe prefill_forward
        pass
    first, second = rows()
    assert (first["step"], first["scope"]) == (0, "forward")
    assert (second["step"], second["scope"]) == (1, "call")
    assert second["gpu_ms"] == 1.0 and "model_gpu_ms" not in second


def test_failed_forward_leaves_no_row(recorder):
    rec, rows = recorder
    with pytest.raises(RuntimeError), rec.measure_forward(), rec.measure(None):
        raise RuntimeError
    assert rec.active is None
    with rec.measure_forward(), rec.measure(None):
        pass
    (row,) = rows()
    assert row["step"] == 1


class FakeProfile:
    opened = 0

    def __init__(self, **kwargs):
        self.profiler = SimpleNamespace(kineto_results=SimpleNamespace(events=list))

    def __enter__(self):
        FakeProfile.opened += 1
        return self

    def __exit__(self, *exc):
        return False


def _batch(prefill=False, bs=1, context=100):
    return SimpleNamespace(
        total_seqs_num=bs,
        total_tokens_num_prefill=bs * 8 if prefill else 0,
        total_tokens_num=bs * 8 if prefill else bs,
        context_lens=np.full(bs, context),
    )


def test_windows_open_per_bucket_after_warmup_and_drop_the_tail(tmp_path, monkeypatch):
    FakeEvent.clock = 0
    FakeProfile.opened = 0
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(compass, "profile", FakeProfile)
    rec = CompassRecorder(
        str(tmp_path),
        op_steps=2,
        op_every=10**9,
        op_min_seen=1,
        op_per_bucket=1,
        anchor_interval_s=1e9,
    )
    batches = [_batch()] * 6 + [_batch(bs=3), _batch(bs=4), _batch(bs=4)]
    batches += [_batch(prefill=True)] * 4
    for batch in batches:
        with rec.measure_forward(batch):
            pass
    rec.close()
    (path,) = (tmp_path / "dp0_pp0_tp0").glob("windows.*.jsonl")
    windows = [json.loads(line) for line in path.read_text().splitlines()]
    # decode bs1: seen once at step 0, window over steps 1-2, tail step 3;
    # bs3 and bs4 share the bs<=4 bucket, whose second sighting opens at 7.
    assert [(w["bucket"], w["steps"], w["tail_step"]) for w in windows] == [
        (["decode", 1, 0, 128], [1, 2], 3),
        (["decode", 4, 0, 128], [7, 8], 9),
        (["prefill", 1, 8, 128], [10, 11], 12),
    ]
    assert all(w["reason"] == "bucket" for w in windows)
    assert FakeProfile.opened == 3


def test_spool_reads_whole_lines_and_refuses_escapes(tmp_path):
    import pytest

    from atom.metrics import compass_spool

    rank = tmp_path / "dp0_pp0_tp0"
    rank.mkdir()
    (rank / "forward.1.jsonl").write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
    files = compass_spool.list_files(str(tmp_path))
    assert files == [{"file": "dp0_pp0_tp0/forward.1.jsonl", "size": 21}]
    path = compass_spool.resolve(str(tmp_path), files[0]["file"])
    assert compass_spool.read_lines(path, 0, 1 << 20) == (b'{"a":1}\n{"b":2}\n', 16)
    assert compass_spool.read_lines(path, 16, 1 << 20) == (b"", 16)
    # A line longer than max_bytes comes back whole.
    assert compass_spool.read_lines(path, 0, 3) == (b'{"a":1}\n', 8)
    for bad in ("../x.jsonl", "dp0/../../x.jsonl", "dp0_pp0_tp0/x.txt", "a/b/c.jsonl"):
        with pytest.raises(ValueError):
            compass_spool.resolve(str(tmp_path), bad)
