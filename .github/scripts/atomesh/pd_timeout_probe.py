"""Build the pinned diagnostic image and stop after the first native failure."""

# Pool-host Python 3.10 has no datetime.UTC.
# ruff: noqa: UP017

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pd_native_stacks import capture_stacks


def validate_probe(contract):
    if not contract.get("diagnostic_only") or contract.get("formal_sa_point"):
        raise ValueError("Timeout probe must remain diagnostic only")
    for role in ("prefill", "decode"):
        spec = contract["roles"][role]
        kv = json.loads(spec["args"]["kv-transfer-config"])
        configs = kv.get("kv_connector_extra_config", {}).get("connectors", [kv])
        mc = next(x for x in configs if x["kv_connector"] == "MooncakeConnector")
        if mc.get("kv_connector_extra_config", {}).get("num_workers", 10) != 10:
            raise ValueError("Keep ten Mooncake sender workers")
        env = contract["environment"] | spec["env"]
        if any(name in env for name in ("MC_TRANSFER_TIMEOUT", "MC_SLICE_TIMEOUT")):
            raise ValueError("Do not change the native timeout or status-polling path")
        if env.get("ATOMESH_MOONCAKE_PROBE_DIR") != "/logs/timeout-probe":
            raise ValueError("Expected task-scoped probe output")


def records(directory):
    found = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer may currently be appending its last line.
    return found


def failure_evidence(events):
    native = [e for e in events if e.get("event") == "before_free_uncached_batch"]
    failures = [e for e in events if e.get("event") == "failed_native_write"]
    for n in native:
        for f in failures:
            if (n["pid"], n["tid"]) == (f["pid"], f["tid"]) and f[
                "native_call_start_monotonic_ns"
            ] <= n["monotonic_ns"] <= f["monotonic_ns"]:
                return {"native": n, "connector": f}
    # Very large batches exceed the bounded native reader; still capture their
    # live thread stacks rather than waiting through more doomed requests.
    for f in failures:
        if f.get("duration_s", 0) >= 30:
            return {
                "native": {
                    "event": "native_state_unavailable",
                    "pid": f["pid"],
                    "tid": f["tid"],
                },
                "connector": f,
            }
    return None


def main():
    repo, root = Path(os.environ["REPO_ROOT"]), Path(os.environ["RUN_DIR"])
    scripts = repo / ".github/scripts/atomesh"
    contract = json.loads(
        (repo / os.environ["ATOMESH_ENV_ATOMESH_PR5090_CONTRACT"]).read_text()
    )
    validate_probe(contract)
    context = repo / contract["timeout_probe"]["context"]
    manifest = json.loads((context / "manifest.json").read_text())
    assert manifest["base_image"] == contract["image"]
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((context / name).read_bytes()).hexdigest() == expected, (
            name
        )
    subprocess.run(
        [sys.executable, str(scripts / "pd_gpool_guard.py"), "timeout-probe-build"],
        check=True,
    )
    build = root / "worker-build"
    build.mkdir()
    probe_dir = root / "logs/timeout-probe"
    probe_dir.mkdir(parents=True)
    with (build / "build.log").open("w") as log:
        subprocess.run(
            [
                "docker",
                "build",
                "--network=none",
                "--iidfile",
                str(build / "image-id.txt"),
                str(context),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=True,
        )
    image = (build / "image-id.txt").read_text().strip()
    assert image.startswith("sha256:") and len(image) == 71
    runtime = json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--label",
                "srtctl.job=" + os.environ["SRTCTL_JOB_ID"],
                "--entrypoint",
                "python3",
                image,
                "/opt/atomesh-timeout/prepare_worker.py",
                "--verify",
            ],
            timeout=60,
            text=True,
        )
    )
    assert runtime["engine_sha256"] == manifest["engine_sha256"]
    assert runtime["connector_sha256"] == manifest["instrumented_connector_sha256"]
    runtime.update(image=image, manifest=manifest)
    debugger_image = None
    if contract["timeout_probe"].get("capture_native_stacks"):
        debugger = context / "debugger"
        assert (
            hashlib.sha256((debugger / "Dockerfile").read_bytes()).hexdigest()
            == contract["timeout_probe"]["debugger_dockerfile_sha256"]
        )
        subprocess.run(
            [sys.executable, str(scripts / "pd_gpool_guard.py"), "debugger-build"],
            check=True,
        )
        with (build / "debugger-build.log").open("w") as log:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--iidfile",
                    str(build / "debugger-image-id.txt"),
                    str(debugger),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=True,
            )
        debugger_image = (build / "debugger-image-id.txt").read_text().strip()
        runtime["debugger_image"] = debugger_image
        runtime["debugger_version"] = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--label",
                "srtctl.job=" + os.environ["SRTCTL_JOB_ID"],
                "--entrypoint",
                "/usr/bin/gdb",
                debugger_image,
                "--version",
            ],
            text=True,
            timeout=30,
        ).splitlines()[0]
    (build / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    for role in ("prefill", "decode"):
        contract["roles"][role]["container"] = image
    # Existing helper resolves contracts relative to REPO_ROOT.
    resolved = repo / ".sa-srt-runtime/timeout-probe-contract.json"
    resolved.write_text(json.dumps(contract, indent=2) + "\n")
    env = dict(
        os.environ, ATOMESH_ENV_ATOMESH_PR5090_CONTRACT=str(resolved.relative_to(repo))
    )
    process = subprocess.Popen(["bash", str(scripts / "pd_srt_functional.sh")], env=env)
    stopping = False

    def interrupted(*_):
        nonlocal stopping
        stopping = True
        process.terminate()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        first_native = None
        while process.poll() is None:
            events = records(probe_dir)
            native_seen = any(
                e.get("event") == "before_free_uncached_batch" for e in events
            )
            if native_seen and first_native is None:
                first_native = time.monotonic()
            evidence = failure_evidence(events)
            if evidence or (
                first_native is not None and time.monotonic() - first_native > 30
            ):
                result = {
                    "status": "captured_failure_stopping",
                    "evidence": evidence,
                    "native_seen": native_seen,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "interpretation": "Diagnostic snapshot only; root cause not yet established.",
                }
                (root / "timeout-probe-result.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                if evidence and debugger_image:
                    try:
                        result["native_stacks"] = capture_stacks(
                            os.environ["SRTCTL_JOB_ID"],
                            evidence["connector"]["pid"],
                            image,
                            debugger_image,
                            probe_dir / "gdb",
                        )
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ) as error:
                        result["native_stacks_error"] = repr(error)
                    (root / "timeout-probe-result.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                process.terminate()
                process.wait(timeout=90)
                return 86  # Intentional failed-run capture, never a passing benchmark.
            if stopping:
                process.wait(timeout=90)
                return 143
            time.sleep(2)
        events = records(probe_dir)
        result = {
            "status": "workload_ended",
            "workload_rc": process.returncode,
            "hook_active": sum(
                e.get("event") == "hook_active" and e.get("layout_compatible") is True
                for e in events
            ),
            "registered_workers": len(
                {e["pid"] for e in events if e.get("event") == "register"}
            ),
            "evidence": failure_evidence(events),
            "root_cause": "unresolved",
        }
        if process.returncode == 0 and (
            result["hook_active"] < 4 or result["registered_workers"] != 8
        ):
            result["status"] = "instrumentation_incomplete"
        (root / "timeout-probe-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return (
            2
            if result["status"] == "instrumentation_incomplete"
            else process.returncode
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                # Outer pool EXIT trap still cleans this job's exact container label.


if __name__ == "__main__":
    raise SystemExit(main())
