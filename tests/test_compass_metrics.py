"""Compass forward rows and op-window parsing."""

import json
from types import SimpleNamespace

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
