"""Fresh task allocation and storage admission; bounded execution and export."""

import gzip
import hashlib
import json
import os
import re
import select
import shutil
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOB = 253987
SPOOL = Path(f"/tmp/dsv41-attention-stride-r1-{JOB}")
HEAD = "96405d8609abdb0e9be594b47dcd4a492e81c6d4"
IMAGE = "sha256:32c81e56e4680065b6ea3e0ca8363a00cd3ccb932d8990e83d799b8bbb58a4b3"
POOL = 1023410176
OUTPUT_LIMIT = 2000000
REMOTE_CHARGES = 1639763968 + 1900000000 + 1000000000
OLD_SPOOLS = 544768


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def output(command):
    return subprocess.check_output(command, text=True)


def du(path):
    return int(output(["du", "-s", "-B1", str(path)]).split()[0])


def source_bindings():
    # New source + exact reused helpers. The fixed runtime and dependency
    # receipts stand for unchanged code; do not rebuild an old 900-file review.
    paths = list(HERE.glob("*.py")) + [HERE / "LOCAL_PREPARATION.json"]
    paths += [
        HERE.parent / p
        for p in (
            "r8/common.py",
            "FP8_DEPENDENCY.json",
            "owner_window_cost_r4/fixture.py",
            "owner_window_cost_r4/device_clock.py",
            "owner_window_cost_r4/SOURCE.json.gz",
            "owner_window_cost_r3/output/window_clock.so",
        )
    ]
    paths += [
        ROOT / "ATOM" / p
        for p in (
            "tools/dsv41/run_dashboard.py",
            "tools/dsv41/dashboard_profile.json",
            "atom/models/deepseek_v41/owner_service/owner_cost_attention_chain.py",
            "atom/models/deepseek_v41/owner_service/owner_head_attention.py",
        )
    ]
    dependency = json.loads((HERE.parent / "FP8_DEPENDENCY.json").read_text())
    for name, digest in dependency["source_files"].items():
        assert sha(HERE.parent / "aiter_fp8" / name) == digest, name
    paths.append(ROOT / "validation/structure_ceiling_20261024/serialize_comgr.so")
    return {str(path): sha(path) for path in paths}


def main():
    job = output(["scontrol", "show", "job", str(JOB)])
    fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", job))
    assert fields["UserId"] == "yifehuan" and fields["JobState"] == "RUNNING"
    assert fields["Exclusive"] == "1" and fields["ReqGPUs"] == "8"
    assert fields["NodeList"] == socket.gethostname()
    assert not SPOOL.exists() and not (HERE / "SOURCE.json.gz").exists()
    assert not output(["docker", "ps", "-q"]).strip()
    pids = output(["rocm-smi", "--showpids"])
    assert "No KFD PIDs currently running" in pids
    gpu = output(["rocm-smi", "--showuse", "--showmemuse", "--csv"])
    cards = [line.split(",") for line in gpu.splitlines() if line.startswith("card")]
    assert len(cards) == 8 and all(row[1:3] == ["0", "0"] for row in cards)
    assert (
        output(["git", "-C", str(ROOT / "ATOM"), "rev-parse", "HEAD"]).strip() == HEAD
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT / "ATOM"),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "atom",
            "tools",
        ],
        check=True,
    )
    assert (
        output(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]).strip()
        == IMAGE
    )
    bindings = source_bindings()
    reserve = 2 * OUTPUT_LIMIT + 256000
    local = du(ROOT / "validation")
    aggregate = (
        du(ROOT / "validation/whole_join_20261028")
        + du(HERE.parent)
        + OLD_SPOOLS
        + reserve
    )
    total = local + REMOTE_CHARGES + OLD_SPOOLS + POOL + reserve
    assert aggregate < 148000000 and total < 10000000000, (aggregate, total)
    path = SPOOL / "output"
    path.mkdir(parents=True)
    name = f"dsv41-attention-stride-r1-{JOB}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--read-only",
        "--network=host",
        "--ipc=none",
        "--tmpfs",
        f"/tmp:rw,exec,size={POOL}",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add=video",
        "--group-add=render",
        "--security-opt=seccomp=unconfined",
        "--ulimit=memlock=-1:-1",
        "--ulimit=core=0",
        "--cpuset-cpus=0-63",
        "--memory=64g",
        "--mount",
        "type=bind,src=/home/yifehuan,dst=/home/yifehuan,readonly",
        "--mount",
        f"type=bind,src={path},dst={path}",
        "--mount",
        "type=bind,src=/shared_nfs/models,dst=/models,readonly",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONNOUSERSITE=1",
        "-e",
        f"WINDOW_OUTPUT={path}",
        IMAGE,
        "/opt/venv/bin/python",
        str(HERE / "driver.py"),
    ]
    save(
        HERE / "SOURCE.json.gz",
        {
            "source_sha256": bindings,
            "head": HEAD,
            "image": IMAGE,
            "job": job,
            "local_validation_bytes": local,
            "remote_charges": REMOTE_CHARGES,
            "old_spools": OLD_SPOOLS,
            "aggregate_bound": aggregate,
            "total_bound": total,
            "pool_bytes": POOL,
            "idle_gpu": gpu,
            "idle_kfd": pids,
            "scope": "A0 full layer8 attention with real projections/mHC; layer9 REUSE qualification/cost; no service/GSM",
        },
    )
    receipt = {"start": time.time(), "command": command, "retired": False}
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        with gzip.open(path / "container.log.gz", "wb") as log:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 5)
                if ready:
                    data = os.read(proc.stdout.fileno(), 65536)
                    if not data:
                        break
                    log.write(data)
                    log.flush()
                assert time.time() - receipt["start"] < 2500, "time bound"
                assert du(path) <= OUTPUT_LIMIT, "output bound"
            receipt["returncode"] = proc.wait(timeout=30)
    except BaseException as error:
        receipt["exception"] = repr(error)
        raise
    finally:
        subprocess.run(
            ["docker", "stop", "-t", "5", name], capture_output=True, check=False
        )
        removed = subprocess.run(
            ["docker", "rm", name], capture_output=True, check=False
        )
        receipt.update(retired=removed.returncode == 0, end=time.time())
        save(SPOOL / "ADMISSION.json", receipt)
        assert receipt["retired"], receipt
        # Only make this container's root-created compiler receipts readable.
        fix = "from pathlib import Path\nfor p in Path('/task').glob('comgr_guard_pid*.json'):\n p.chmod(p.stat().st_mode | 0o444)\n"
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--network=none",
                "--ipc=none",
                "--mount",
                f"type=bind,src={path},dst=/task",
                IMAGE,
                "/opt/venv/bin/python",
                "-c",
                fix,
            ],
            check=True,
        )
        destination = HERE / "output"
        shutil.copytree(path, destination)
        receipt["export_sha256"] = {
            str(p.relative_to(path)): sha(p) for p in path.rglob("*") if p.is_file()
        }
        assert all(
            sha(destination / p) == digest
            for p, digest in receipt["export_sha256"].items()
        )
        shutil.rmtree(path)
        receipt["final_kfd"] = output(["rocm-smi", "--showpids"])
        receipt["final_gpu"] = output(
            ["rocm-smi", "--showuse", "--showmemuse", "--csv"]
        )
        save(HERE / "ADMISSION.json", receipt)
        print(json.dumps(receipt), flush=True)
    assert all(sha(path) == digest for path, digest in bindings.items())
    assert receipt.get("returncode") == 0, receipt


if __name__ == "__main__":
    main()
