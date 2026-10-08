"""CPU preflight of the profiler control lifecycle before spending GPU time."""

import gzip
import importlib.util
import json
import sys
import threading
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/pd_agentic_profile.py"
)
spec = importlib.util.spec_from_file_location("pd_agentic_profile", MODULE)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@contextmanager
def endpoint(root, role, calls, fail_start=False, ranks=8):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"vllm:request_success_total 16\n")

        def do_POST(self):
            calls.append((role, self.path))
            if self.path == "/start_profile" and fail_start:
                self.send_error(500)
                return
            if self.path == "/stop_profile":
                directory = root / role
                directory.mkdir(parents=True, exist_ok=True)
                for rank in range(ranks):
                    with gzip.open(
                        directory / f"dp0_tp{rank}_rank{rank}.123.pt.trace.json.gz",
                        "wt",
                    ) as stream:
                        json.dump(
                            {
                                "traceEvents": [
                                    {
                                        "cat": "cpu_op",
                                        "name": "aten::mm",
                                        "ts": 0,
                                        "dur": 1,
                                    },
                                    {
                                        "cat": "kernel",
                                        "name": "gemm",
                                        "ts": 1,
                                        "dur": 5,
                                    },
                                ]
                            },
                            stream,
                        )
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("failure", [None, "start", "missing_rank"])
def test_capture_lifecycle(tmp_path, failure):
    calls = []
    with ExitStack() as stack:
        prefill = stack.enter_context(endpoint(tmp_path / "traces", "prefill", calls))
        decode = stack.enter_context(
            endpoint(
                tmp_path / "traces",
                "decode",
                calls,
                fail_start=failure == "start",
                ranks=7 if failure == "missing_rank" else 8,
            )
        )
        args = SimpleNamespace(
            prefill=prefill,
            decode=decode,
            output=tmp_path / "control",
            traces=tmp_path / "traces",
            aiperf_log=tmp_path / "aiperf.log",
            seconds=0.01,
            offsets=[0, 0.1],
            ranks=8,
            phase_timeout=5,
            finish_timeout=5,
        )
        command = [
            sys.executable,
            "-u",
            "-c",
            (
                "import time; print('AIPerf System is PROFILING'); "
                "print('Phase warmup (warmup) started'); time.sleep(.05); "
                "print('Phase profiling (profiling) started'); time.sleep(2)"
            ),
        ]
        rc = profile.Capture(args).run(command)
    result = json.loads((args.output / "result.json").read_text())
    assert result["complete"] == (failure is None)
    assert rc == (0 if failure is None else 1)
    assert ("prefill", "/stop_profile") in calls
    assert ("decode", "/stop_profile") in calls
    if failure is None:
        assert len(result["traces"]) == 32
        assert len(list(args.traces.glob("window-*/*/*.gz"))) == 32


def test_cpu_only_trace_rejected(tmp_path):
    path = tmp_path / "rank0.123.pt.trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": [{"cat": "cpu_op", "ts": 0, "dur": 1}]}, stream)
    with pytest.raises(ValueError, match="Missing GPU kernels"):
        profile.inspect_trace(path)


def test_warmup_is_not_profiling_window():
    assert not profile.PHASE_START.search("AIPerf System is PROFILING")
    assert not profile.PHASE_START.search("Phase warmup (warmup) started")
    assert profile.PHASE_START.search(
        "Phase profiling (profiling) started | target: 600.0s"
    )
