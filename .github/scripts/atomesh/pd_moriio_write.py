"""Build the pinned PP backport and run the native srtctl MoRIIO workload."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    repo, root = Path(os.environ["REPO_ROOT"]), Path(os.environ["RUN_DIR"])
    contract = json.loads(
        (repo / os.environ["ATOMESH_ENV_ATOMESH_PR5090_CONTRACT"]).read_text()
    )
    assert contract["kv_transport"] == "moriio" and contract["router_arm"] == "atomesh"
    context = repo / contract["moriio_worker"]["context"]
    manifest = json.loads((context / "manifest.json").read_text())
    assert manifest["base_image"] == contract["image"]
    for name, expected in manifest["context_files"].items():
        assert hashlib.sha256((context / name).read_bytes()).hexdigest() == expected, (
            name
        )
    build = root / "worker-build"
    build.mkdir()
    (root / "logs/moriio-probe").mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(repo / ".github/scripts/atomesh/pd_gpool_guard.py"),
            "moriio-worker-build",
        ],
        check=True,
    )
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
                "--network=none",
                "--label",
                "srtctl.job=" + os.environ["SRTCTL_JOB_ID"],
                "--entrypoint",
                "python3",
                image,
                "/opt/atomesh-mori/verify_worker.py",
            ],
            text=True,
            timeout=120,
        )
        .strip()
        .splitlines()[-1]
    )
    runtime.update(image=image, manifest=manifest)
    (build / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    for role in ("prefill", "decode"):
        contract["roles"][role]["container"] = image
    relative = ".sa-srt-runtime/moriio-write-contract.json"
    (repo / relative).write_text(json.dumps(contract, indent=2) + "\n")
    env = dict(os.environ, ATOMESH_ENV_ATOMESH_PR5090_CONTRACT=relative)
    process = subprocess.Popen(
        ["bash", str(repo / ".github/scripts/atomesh/pd_srt_functional.sh")], env=env
    )
    signal.signal(signal.SIGTERM, lambda *_: process.terminate())
    signal.signal(signal.SIGINT, lambda *_: process.terminate())
    try:
        pattern = re.compile(
            r"MoRIIO batch wait failed|RDMA transfer timed out|RDMA transfer failed:"
        )
        while process.poll() is None:
            failures = []
            for path in (root / "logs").glob("*_w0.out"):
                for line in path.read_text(errors="replace").splitlines():
                    if pattern.search(line):
                        failures.append({"file": path.name, "line": line[:1500]})
                        break
            if failures:
                (root / "moriio-result.json").write_text(
                    json.dumps(
                        {"status": "native_failure_stopping", "failures": failures},
                        indent=2,
                    )
                )
                process.terminate()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return 86
            time.sleep(3)
        (root / "moriio-result.json").write_text(
            json.dumps(
                {"status": "workload_ended", "workload_rc": process.returncode},
                indent=2,
            )
        )
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
