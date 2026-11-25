"""Bounded debugger sidecar; never install debugger packages in a model worker."""

# Pool-host Python 3.10 has no datetime.UTC.
# ruff: noqa: UP017

import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def capture_stacks(job_id, pid, worker_image, debugger_image, output):
    if type(pid) is not int or pid <= 0 or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("Expected a task identity and positive container PID")
    output = Path(output)
    output.mkdir(parents=True)
    ids = subprocess.check_output(
        ["docker", "ps", "-q", "--filter", f"label=srtctl.job={job_id}"], text=True
    ).split()
    if not ids:
        raise RuntimeError("The owned worker is no longer running")
    containers = json.loads(
        subprocess.check_output(["docker", "inspect", *ids], text=True)
    )
    prefills = [
        c
        for c in containers
        if c["Config"]["Labels"].get("srtctl.step", "").startswith("prefill_")
    ]
    if len(prefills) != 1 or prefills[0]["Config"]["Image"] != worker_image:
        raise RuntimeError("Cannot identify the exact diagnostic prefill container")
    target = prefills[0]["Id"]
    # The PID came from this job's native/connector records in this namespace.
    subprocess.run(
        ["docker", "exec", target, "test", "-d", f"/proc/{pid}"], check=True, timeout=10
    )
    token = uuid.uuid4().hex
    capture_label = f"atomesh.debug.capture={token}"
    result = {
        "target_container": target,
        "container_pid": pid,
        "worker_image": worker_image,
        "debugger_image": debugger_image,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        "atomesh-gdb-" + token,
        "--label",
        f"srtctl.job={job_id}",
        "--label",
        "srtctl.step=timeout_debugger",
        "--label",
        capture_label,
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--pid",
        "container:" + target,
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "--memory=2g",
        "--pids-limit=64",
        "-v",
        str(Path(__file__).with_name("pd_gdb_snapshot.py").resolve())
        + ":/capture-queue.py:ro",
        "-e",
        "DEBUGINFOD_URLS=",
        "--entrypoint",
        "/usr/bin/gdb",
        debugger_image,
        "-nx",
        "-batch",
        "-iex",
        "maintenance set worker-threads 1",
        "-ex",
        "set auto-load off",
        "-ex",
        "set auto-solib-add off",
        "-ex",
        "set debuginfod enabled off",
        "-ex",
        "set pagination off",
        "-ex",
        "set print frame-arguments none",
        "-ex",
        f"set sysroot /proc/{pid}/root",
        "-ex",
        f"attach {pid}",
        "-ex",
        "sharedlibrary /engine[.]so",
        "-ex",
        "sharedlibrary libstdc",
        "-ex",
        "sharedlibrary libc[.]so",
        "-ex",
        "info sharedlibrary",
        "-ex",
        f"python print(''.join(x for x in open('/proc/{pid}/maps') if '/engine.so' in x or 'libc.so' in x or 'libstdc' in x))",
        "-ex",
        "info threads",
        "-ex",
        "thread apply all bt 24",
        "-ex",
        "thread apply all info registers rip rsp rax rbx rcx rdx rsi rdi rbp r12 r13 r14 r15",
        "-ex",
        "source /capture-queue.py",
        "-ex",
        "detach",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    try:
        with (output / "stacks.txt").open("wb") as log:
            try:
                proc = subprocess.run(
                    command,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=25,
                )
                result["gdb_rc"] = proc.returncode
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
    finally:
        # Killing a Docker client does not kill its container. Scope cleanup to
        # this one capture token, even if --rm has already removed the sidecar.
        owned = subprocess.check_output(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=srtctl.job={job_id}",
                "--filter",
                "label=" + capture_label,
            ],
            text=True,
            timeout=10,
        ).split()
        for container in owned:
            if target.startswith(container) or not re.fullmatch(
                r"[0-9a-f]{12,64}", container
            ):
                raise RuntimeError("Unexpected debugger cleanup target")
            subprocess.run(
                ["docker", "rm", "-f", container],
                check=True,
                timeout=15,
                stdout=subprocess.DEVNULL,
            )
        result["sidecars_remaining"] = subprocess.check_output(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=srtctl.job={job_id}",
                "--filter",
                "label=" + capture_label,
            ],
            text=True,
            timeout=10,
        ).split()
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        if (output / "stacks.txt").exists():
            data = (output / "stacks.txt").read_bytes()
            result["stack_bytes"] = len(data)
            result["stack_sha256"] = hashlib.sha256(data).hexdigest()
            result["frames_present"] = b"#0 " in data or b"#0\t" in data
            result["mooncake_symbols_present"] = b"mooncake::WorkerPool" in data
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
