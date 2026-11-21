"""Four fixed C validation payloads. The common f280 core owns all cleanup."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evidence import compare, host_summary, read, response_check, sha, write

BASE = Path(__file__).resolve().parent
CASES = {
    "C-eager": ("candidate", 0, "NONE", 16384, 2400),
    "C-zero": ("candidate", 3, "FULL", 16384, 2700),
    "Main-host": ("main", 3, "FULL", 3072, 2400),
    "C-graph": ("candidate", 3, "FULL", 3072, 6000),
}
MODEL, DRAFT = "/input/model-target", "/input/model-draft"
REQUESTS, REFERENCES = Path("/input/requests"), Path("/input/reference-results")


class HttpError(RuntimeError):
    pass


def server_argv(case):
    _, level, graphs, batch, _ = CASES[case]
    return [sys.executable, "-B", "-m", "atom.entrypoints.openai_server",
            "--model", MODEL, "--trust-remote-code", "-tp", "8",
            "--decode-context-parallel-size", "8", "--kv_cache_dtype", "bf16",
            "--index-cache-dtype", "fp8", "--max-model-len", "131072",
            "--max-num-seqs", "16", "--gpu-memory-utilization", "0.9",
            "--block-size", "128", "--no-enable_prefix_caching", "--host", "127.0.0.1", "--server-port", "8000",
            "--method", "dspark", "--draft-model", DRAFT, "--num-speculative-tokens", "7",
            "--level", str(level), "--cudagraph-mode", graphs,
            "--max-num-batched-tokens", str(batch), "--mark-trace",
            "--torch-profiler-dir", f"/output/{case}/traces"]


def evaluator_argv(output):
    return ["bash", str(BASE / "lm_eval_receipt.sh"), "--model", "local-completions",
            "--model_args", f"model={MODEL},base_url=http://localhost:8000/v1/completions,"
            "num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True",
            "--tasks", "gsm8k", "--include_path", str(BASE / "tasks"),
            "--num_fewshot", "3", "--seed", "0,1234,1234,1234", "--log_samples",
            "--output_path", str(output / "lm-eval")]


def remaining(end):
    value = end - time.monotonic()
    if value <= 0:
        raise TimeoutError("stage or whole-leg deadline exceeded")
    return value


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "state": fields[0], "start_ticks": int(fields[19])}


def readonly(path):
    assert os.statvfs(Path(path).resolve()).f_flag & os.ST_RDONLY, f"not read-only: {path}"


class Case:
    def __init__(self, name, admission_path):
        self.name = name
        self.output = Path(f"/output/{name}")
        self.source = Path("/input") / CASES[name][0]
        self.started = time.monotonic()
        self.end = min(float(os.environ["MODEL_CASE_DEADLINE_MONOTONIC"]),
                       self.started + CASES[name][4])
        self.admission = read(admission_path)
        self.admission_path = Path(admission_path)
        self.server = None
        self.native_ids = set()
        self.requests = read(REQUESTS / "manifest.json")
        self.ledger = []
        self.timed = False
        self.timing_gates = []
        self.verified_references = set()

    def stage(self, seconds):
        return min(self.end, time.monotonic() + seconds)

    def preflight(self):
        pins, a = read(BASE / "pins.json"), self.admission
        assert a["case"] == self.name and a["payload_sha256"] == sha(__file__)
        assert a["payload_manifest_sha256"] == sha(BASE / "payload-manifest.json")
        manifest = read(BASE / "payload-manifest.json")
        expected_files = {"payload-manifest.json"}
        for item in manifest["files"]:
            path = BASE / item["file"]
            assert not path.is_symlink() and path.resolve().is_relative_to(BASE)
            assert sha(path) == item["sha256"]
            expected_files.add(item["file"])
        assert {p.relative_to(BASE).as_posix() for p in BASE.rglob("*") if p.is_file()} == expected_files
        assert a["asset_manifest_sha256"] == pins["model_manifest"]
        assert a["image_configuration_id"] == pins["image"] and a["aiter_git"] == pins["aiter"]
        assert a["common_supervisor_sha256"] == pins["supervisor"]
        assert a["source_sha"] == pins["source"][CASES[self.name][0]]
        assert a["compatibility_preflight_passed"] is True and a["model_views_reviewed"] is True
        if self.name != "C-eager":
            assert a["timed_windows_quiet"] is True
            assert a["timed_window_gate_path"] == "/app/model-case-control/timed-window.json"
            assert isinstance(a["caller_policy_sha256"], str) and len(a["caller_policy_sha256"]) == 64
        assert a["requests_manifest_sha256"] == sha(REQUESTS / "manifest.json")
        assert self.requests["asset_manifest_sha256"] == pins["model_manifest"]
        assert self.output.is_dir() and not (self.output / "case-start.json").exists()
        for path in (self.source, BASE, MODEL, DRAFT, "/input/data", REQUESTS):
            readonly(path)
        for variable, expected in (("HOME", f"/scratch/{self.name}/home"),
                                   ("TMPDIR", f"/scratch/{self.name}/tmp")):
            assert os.environ[variable] == expected and Path(expected).is_dir()
        assert sha(Path(sys.executable).resolve()) == pins["python_sha256"]
        for package, version in pins["package_versions"].items():
            assert importlib.metadata.version(package) == version, package
        for item in pins["package_files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]
        task = Path(importlib.metadata.distribution("lm_eval").locate_file("lm_eval/tasks/gsm8k/gsm8k.yaml"))
        assert sha(task) == pins["task_sha256"]
        distribution = importlib.metadata.distribution("lm_eval")
        for file, expected in pins["client_source_files"].items():
            assert sha(distribution.locate_file(file)) == expected, file
        assert shutil.which("lm_eval") == "/opt/venv/bin/lm_eval", "unbound evaluator command"
        for file, expected in pins["source_files"][CASES[self.name][0]].items():
            assert sha(self.source / file) == expected, file
        for item in pins["data"].values():
            if isinstance(item, dict):
                assert sha(Path("/input/data") / item["file"]) == item["sha256"]
        for file, expected in self.requests["files"].items():
            assert sha(REQUESTS / file) == expected
        views = read(BASE / "composition-manifest.json")
        assert sha(BASE / "composition-manifest.json") == pins["model_manifest"]
        for view, mounted in zip(views["views"], (Path(MODEL), Path(DRAFT))):
            for item in view["files"]:
                path = mounted / item["name"]
                readonly(path)
                if item.get("actual_sha256"):
                    assert sha(path) == item["actual_sha256"]
                else:
                    st, old = path.stat(), item["source"]
                    assert (st.st_size, st.st_ino, st.st_mtime_ns) == (old["bytes"], old["inode"], old["mtime_ns"])
        self.env = dict(os.environ)
        self.env.update(PYTHONPATH=str(self.source), PYTHONDONTWRITEBYTECODE="1",
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                        HF_HOME=f"/scratch/{self.name}/hf", HF_DATASETS_CACHE=f"/scratch/{self.name}/datasets",
                        AITER_LOG_LEVEL="WARNING", ATOM_MONO_ENABLE="1", ATOM_ENABLE_REPLAYSSM="1",
                        ATOM_TOKEN_GRAPH_SIZES="256,512,1024,2048,3072",
                        ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD="1024",
                        ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS="8192", ATOM_PROFILER_MORE="0",
                        ATOM_SERVER_LOG=str(self.output / "server.log"))
        for key in ("ATOM_PROFILER_RECORD_SHAPES", "ATOM_PROFILER_WITH_STACK", "ATOM_PROFILER_PROFILE_MEMORY"):
            self.env[key] = "1" if self.name == "C-zero" else "0"
        recorded_env = ("HOME", "TMPDIR", "PYTHONPATH", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE",
                        "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "AITER_LOG_LEVEL", "ATOM_MONO_ENABLE",
                        "ATOM_ENABLE_REPLAYSSM", "ATOM_TOKEN_GRAPH_SIZES", "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD",
                        "ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "ATOM_PROFILER_MORE", "ATOM_SERVER_LOG",
                        "ATOM_PROFILER_RECORD_SHAPES", "ATOM_PROFILER_WITH_STACK", "ATOM_PROFILER_PROFILE_MEMORY")
        recorded_admission = ("case", "payload_sha256", "payload_manifest_sha256", "asset_manifest_sha256",
                              "image_configuration_id", "aiter_git", "common_supervisor_sha256", "source_sha",
                              "compatibility_preflight_passed", "model_views_reviewed", "requests_manifest_sha256",
                              "reference_manifest_sha256", "timed_windows_quiet", "timed_window_gate_path",
                              "caller_policy_sha256")
        write(self.output / "case-start.json", {"case": self.name, "payload_sha256": sha(__file__),
              "admission": {k: a[k] for k in recorded_admission if k in a},
              "admission_sha256": sha(self.admission_path),
              "started_monotonic": self.started, "deadline_monotonic": self.end,
              "server_argv": server_argv(self.name), "explicit_env": {k: self.env[k] for k in recorded_env}})

    def alive(self):
        assert self.server is not None
        # WNOWAIT leaves the direct child unreaped, so the PID given to drain cannot recycle.
        status = os.waitid(os.P_PID, self.server.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is not None:
            raise RuntimeError(f"owned server exited: {status}")

    def http(self, endpoint, body, end):
        self.alive()
        budget = remaining(end)
        argv = ["curl", "--silent", "--show-error", "--fail-with-body", "--max-time", str(max(.001, budget - .05)),
                "-X", "GET" if body is None else "POST", "-H", "Content-Type: application/json"]
        if body is not None:
            argv += ["--data-binary", "@-"]
        argv += ["http://127.0.0.1:8000" + endpoint]
        helper = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = helper.communicate(None if body is None else json.dumps(body).encode(), timeout=remaining(end))
        remaining(end)
        if helper.returncode:
            raise HttpError(f"HTTP {endpoint} failed ({helper.returncode}): {error.decode(errors='replace')} {output[:2000]!r}")
        return output

    def start(self):
        with (self.output / "server.log").open("xb") as log:
            self.server = subprocess.Popen(server_argv(self.name), cwd=self.source, env=self.env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(self.output / "server-identity.json", identity(self.server.pid))
        end = self.stage(1200)
        while True:
            self.alive()
            try:
                self.http("/health", None, min(end, time.monotonic() + 2))
                break
            except HttpError:
                remaining(end)
                time.sleep(.5)
        end = self.stage(300)
        self.send(300, 1, "startup-ready", end)
        if self.name in ("C-graph", "C-zero"):
            log = (self.output / "server.log").read_text(errors="replace")
            assert "Token graphs recorded for prefill buckets" in log
            assert all(str(n) in log.split("Token graphs recorded for prefill buckets")[-1].splitlines()[0]
                       for n in (256, 512, 1024, 2048, 3072))

    def send(self, n, output, probe, end=None):
        if self.timed:
            self.timing_gate(probe)
        request_path = REQUESTS / f"{n}-{output}.json"
        request = read(request_path)
        assert request["model"] == MODEL
        begun = time.monotonic()
        end = min(self.stage(120), end if end is not None else self.end)
        raw = self.http("/v1/completions", request, end)
        with (self.output / f"{probe}.json").open("xb") as stream:
            stream.write(raw)
        response = json.loads(raw)
        response_check(request, response)
        assert response["id"] not in self.native_ids, "native response ID reused"
        self.native_ids.add(response["id"])
        self.ledger.append({"probe_id": f"{self.name}/{probe}", "native_response_id": response["id"],
                            "input_file": request_path.name, "input_sha256": sha(request_path),
                            "response_file": f"{probe}.json", "response_sha256": sha(self.output / f"{probe}.json"),
                            "start_monotonic": begun, "end_monotonic": time.monotonic()})
        return response

    def timing_gate(self, label):
        gate = read(self.admission["timed_window_gate_path"])
        assert gate["case"] == self.name
        assert gate["payload_manifest_sha256"] == self.admission["payload_manifest_sha256"]
        assert gate["caller_policy_sha256"] == self.admission["caller_policy_sha256"]
        assert gate["status"] == "quiet" and gate["ongoing_host_audit"] is True
        assert 0 <= time.time() - gate["checked_epoch"] <= 120, "timing audit expired"
        self.timing_gates.append({"label": label, "observed_epoch": time.time(), "gate": gate})

    def reference(self, case, file):
        manifest = read(REFERENCES / "manifest.json")
        assert self.admission["reference_manifest_sha256"] == sha(REFERENCES / "manifest.json")
        rec = manifest["cases"][case]
        assert rec["entry_exit"] == 0 and rec["cleanup_verified"] is True
        if case not in self.verified_references:
            for name in ("case-start.json", "entry-exit.json", "supervisor-exit.json"):
                assert sha(REFERENCES / case / name) == rec["files"][name]
            start, finish = read(REFERENCES / case / "case-start.json"), read(REFERENCES / case / "entry-exit.json")
            supervisor = read(REFERENCES / case / "supervisor-exit.json")
            assert supervisor["status"] == "completed"
            assert supervisor["kernel_children_empty"] is True and not supervisor["owned_remaining"]
            assert supervisor["lease_held"] is False and supervisor["lease_released_utc"]
            assert start["case"] == finish["case"] == case and finish["entry_exit"] == 0
            assert start["payload_sha256"] == finish["payload_sha256"] == sha(__file__)
            assert start["server_argv"] == server_argv(case)
            prior, current = start["admission"], self.admission
            for key in ("payload_manifest_sha256", "asset_manifest_sha256", "image_configuration_id",
                        "aiter_git", "common_supervisor_sha256", "requests_manifest_sha256"):
                assert prior[key] == current[key], f"reference {case}: {key} differs"
            assert prior["source_sha"] == read(BASE / "pins.json")["source"][CASES[case][0]]
            self.verified_references.add(case)
        path = REFERENCES / case / file
        readonly(path)
        assert path.resolve().is_relative_to(REFERENCES.resolve())
        assert sha(path) == rec["files"][file]
        return read(path)

    def profile(self, label, scheduled, action, *, replay=False, zeros=False, window=90):
        timed = zeros or label.startswith("host-")
        if timed:
            self.timing_gate(label + ":before")
        self.timed = timed
        before = set((self.output / "traces").rglob("*.pt.trace.json*"))
        self.http("/start_profile", {}, self.stage(30))
        action(self.stage(window))
        end = self.stage(120)
        self.http("/stop_profile", {}, end)
        if timed:
            self.timing_gate(label + ":after-stop")
        self.timed = False
        attempt = 0
        while True:
            self.alive()
            paths = set((self.output / "traces").rglob("*.pt.trace.json*")) - before
            if len(paths) == 8:
                attempt += 1
                spec = self.output / f"{label}-inspect-{attempt}.json"
                report = self.output / f"{label}-traces.json"
                write(spec, {"paths": sorted(map(str, paths)), "scheduled": scheduled, "replay": replay,
                             "zeros": zeros, "native_summary": str(self.source / "tools/analyze_trace_summary.py")})
                with (self.output / f"{label}-inspect-{attempt}.log").open("xb") as log:
                    helper = subprocess.Popen([sys.executable, "-B", str(BASE / "evidence.py"),
                                               "traces", str(spec), str(report)], stdout=log, stderr=subprocess.STDOUT)
                code = helper.wait(timeout=min(60, remaining(end)))
                if code == 0:
                    result = read(report)
                    break
                assert code == 75, f"failed {label} trace witness; see inspection log"
            remaining(end)
            time.sleep(.5)
        return result

    def correctness(self):
        def requests(end):
            for n in (300, 512, 513):
                got = self.send(n, 32, f"correct-{n}", end)
                if self.name == "C-graph":
                    compare(read(REQUESTS / f"{n}-32.json"), self.reference("C-eager", f"correct-{n}.json"), got)
            got = self.send(300, 32, "correct-300-repeat", end)
            compare(read(REQUESTS / "300-32.json"), read(self.output / "correct-300.json"), got)
            if self.name == "C-graph":
                compare(read(REQUESTS / "300-32.json"), self.reference("C-eager", "correct-300-repeat.json"), got)
        if self.name == "C-graph":
            self.profile("correctness", [300, 512, 513, 300], requests, replay=True, window=300)
        else:
            requests(self.stage(300))

    def large(self):
        end = self.stage(300)
        for n in (8192, 8193):
            got = self.send(n, 1, f"large-reference-{n}", end)
            if self.name == "C-zero":
                compare(read(REQUESTS / f"{n}-1.json"), self.reference("C-eager", f"large-reference-{n}.json"), got)
        if self.name == "C-zero":
            def requests(end):
                for n in (8192, 8193):
                    for sample in (1, 2, 3):
                        got = self.send(n, 1, f"zero-{n}-{sample}", end)
                        compare(read(REQUESTS / f"{n}-1.json"), self.reference("C-eager", f"large-reference-{n}.json"), got)
            self.profile("zero", [8192] * 3 + [8193] * 3, requests, zeros=True, window=720)

    def host(self):
        warmup_end = self.stage(300)
        for sample in (1, 2, 3):
            self.send(300, 1, f"host-warm-{sample}", warmup_end)
        windows = []
        for window in (1, 2, 3):
            def requests(end):
                for sample in range(1, 11):
                    probe = f"host-{window}-{sample}"
                    got = self.send(300, 1, probe, end)
                    if self.name == "C-graph":
                        compare(read(REQUESTS / "300-1.json"), self.reference("Main-host", probe + ".json"), got)
            windows.append(self.profile(f"host-{window}", [300] * 10, requests,
                                        replay=self.name == "C-graph"))
        summary = host_summary(windows)
        if self.name == "C-graph":
            main = self.reference("Main-host", "host-summary.json")
            summary["main_reference"] = main
            summary["rank0_median_ratio"] = summary["per_rank"]["0"]["median_us"] / main["per_rank"]["0"]["median_us"]
        write(self.output / "host-summary.json", summary)

    def accuracy(self):
        self.reference("C-zero", "zero-traces.json")  # preceding short leg completed and cleaned up
        assert remaining(self.end) >= 2820, "insufficient budget before sole accuracy attempt"
        output = self.output / "accuracy-attempt-1"
        output.mkdir()  # existing attempt is never overwritten or resumed
        env = dict(self.env, EVIDENCE=str(output), WAIT_SERVER_PID=str(self.server.pid),
                   WAIT_SERVER_EXIT_IS_FAULT="1")
        start_offset = (self.output / "server.log").stat().st_size
        argv = evaluator_argv(output)
        with (output / "client.log").open("xb") as log:
            evaluator = subprocess.Popen(argv, cwd=self.source, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
        env["WAIT_CLIENT_PID"] = str(evaluator.pid)
        write(output / "client-start.json", {"argv": argv, "identity": identity(evaluator.pid),
                  "server_identity": identity(self.server.pid), "server_log_start": start_offset})
        end = min(self.end, time.monotonic() + 2700)
        with (output / "drain.log").open("xb") as log:
            drain = subprocess.Popen(["bash", str(self.source / "scripts/wait_infer_drain.sh"),
                       "8000", "45", "10", str(output / "client.log"), "18"],
                       env=env, cwd=self.source, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        drain_rc = drain.wait(timeout=remaining(end))
        if drain_rc:
            write(output / "client-wait.json", {"drain_returncode": drain_rc, "wrapper_returncode": None})
            raise RuntimeError(f"accuracy drain failed: {drain_rc}")
        client_rc = evaluator.wait(timeout=remaining(end))
        write(output / "client-wait.json", {"drain_returncode": drain_rc, "wrapper_returncode": client_rc})
        assert client_rc == 0 and read(output / "lm-eval-exit.json")["exit_code"] == 0
        self.alive()
        spec = output / "check-spec.json"
        write(spec, {"output": str(output), "population": str(REQUESTS / "population.json"),
                     "server_log": str(self.output / "server.log"), "start_offset": start_offset,
                     "end_offset": (self.output / "server.log").stat().st_size})
        with (output / "check.log").open("xb") as log:
            helper = subprocess.Popen([sys.executable, "-B", str(BASE / "evidence.py"), "accuracy", str(spec),
                                       str(output / "accuracy-completeness.json")], stdout=log, stderr=subprocess.STDOUT)
        assert helper.wait(timeout=min(60, remaining(self.end))) == 0, "accuracy receipt check failed"

    def run(self):
        self.preflight()
        self.start()
        if self.name == "C-eager":
            self.correctness()
            self.large()
        elif self.name == "C-zero":
            self.large()
        elif self.name == "Main-host":
            self.host()
        else:
            self.correctness()
            self.host()
            self.accuracy()
        self.alive()
        remaining(self.end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    case, code, error = None, 1, None
    try:
        case = Case(args.case, args.binding)
        case.run()
        code = 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if case is not None:
            write(case.output / "request-ledger.json", case.ledger)
            write(case.output / "entry-exit.json", {"entry_exit": code, "error": error,
                  "case": args.case, "payload_sha256": sha(__file__),
                  "timing_gates": case.timing_gates,
                  "ended_monotonic": time.monotonic(), "cleanup_authority": "common f280 supervisor"})


if __name__ == "__main__":
    main()
