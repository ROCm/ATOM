"""Exercise real observer/controller boundaries with a local clock and workload."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(os.environ.get("C48_TEST_SCRIPTS", Path(__file__).parents[1]))
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return round(self.now * 1e9)

    time_ns = monotonic_ns

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()


def test_admission_triggers_once_only_in_formal_phase(tmp_path, monkeypatch):
    source = module("pd_c48_timeline")
    clock = Clock()
    monkeypatch.setattr(source, "time", clock)
    request = SimpleNamespace(request_id="target", num_prompt_tokens=544099, arrival_time=999)
    marker = object()
    scheduler = SimpleNamespace(
        schedule=lambda: None, update_from_output=lambda _: {},
        add_request=lambda _: marker,
    )
    timeline = source.Timeline(scheduler, tmp_path).install()
    timeline.plan = {
        "phase_start_unix_ns": clock.time_ns(), "phase_seconds": 3930,
        "detail_windows": [[600, 720]], "target_prompt_range": [543000, 545000],
    }
    try:
        clock.now = 999
        assert scheduler.add_request(request) is marker
        assert not (tmp_path / "target-request.json").exists()
        clock.now = 1100
        request.num_prompt_tokens = 542999
        assert scheduler.add_request(request) is marker
        assert not (tmp_path / "target-request.json").exists()
        request.num_prompt_tokens = 544099
        assert scheduler.add_request(request) is marker
        saved = (tmp_path / "target-request.json").read_bytes()
        assert json.loads(saved)["request_id"] == "target"
        assert timeline.active() == (True, True)
        request.request_id = "another"
        assert scheduler.add_request(request) is marker
        assert (tmp_path / "target-request.json").read_bytes() == saved
        clock.now = 1221
        assert timeline.active() == (True, False)
    finally:
        timeline.close()


@pytest.mark.parametrize("stale", [False, True])
def test_full_controller_waits_for_target_and_preserves_workload(tmp_path, monkeypatch, stale):
    source = module("pd_c48_profile")
    clock = Clock()
    monkeypatch.setattr(source, "time", clock)
    timeline = tmp_path / "timeline"
    timeline.mkdir()
    aiperf_log = tmp_path / "aiperf.log"
    aiperf_log.write_text("Phase profiling (profiling) started\n")
    args = SimpleNamespace(
        prefill="http://127.0.0.1:1", output=tmp_path / "capture", timeline=timeline,
        aiperf_log=aiperf_log, phase_timeout=7200, finish_timeout=5400,
        offsets=[60], target_prompt_range=[543000, 545000], target_offsets=[0, 25], ranks=8,
    )
    waits = []
    child = SimpleNamespace(poll=lambda: None, wait=lambda timeout: waits.append(timeout) or 0)
    monkeypatch.setattr(source.subprocess, "Popen", lambda *a, **kw: child)
    capture = source.PrefillCapture(args)
    events, windows = [], []
    monkeypatch.setattr(capture, "event", lambda kind, **kw: events.append((kind, kw)))

    def window(number, workload):
        assert workload is child
        windows.append((number, clock.now))
        clock.sleep(20)
        return [{"rank": rank, "window": number} for rank in range(8)]

    monkeypatch.setattr(capture, "window", window)

    def publish():
        if clock.now >= 1090 and not (timeline / "target-request.json").exists():
            (timeline / "target-request.json").write_text(json.dumps({
                "request_id": "target", "prompt_tokens": 544099,
                "unix_ns": 1 if stale else clock.time_ns(),
            }))

    clock.on_sleep = publish
    rc = capture.run(["local-full-workload"])
    result = json.loads((args.output / "result.json").read_text())
    assert waits == [5400] and result["workload_rc"] == 0
    if stale:
        assert rc == 1 and not result["complete"]
        assert "stale" in result["errors"][0]
        assert windows == [(1, 1060)]
    else:
        assert rc == 0 and result["complete"]
        assert windows == [(1, 1060), (2, 1090), (3, 1115)]
        assert len(result["traces"]) == 24
        assert any(kind == "target_capture_observed" for kind, _ in events)
