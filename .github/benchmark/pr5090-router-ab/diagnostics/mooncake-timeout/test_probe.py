"""CPU-only ABI/injection test; does not reproduce the live GPU timeout."""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def child():
    # RTLD_LOCAL mirrors a Python extension import, unlike a linked executable.
    lib = ctypes.CDLL(str(Path(__file__).with_name("fixture.so")))
    lib.run_case.argtypes = [ctypes.c_int]
    assert lib.run_case(1) == 0  # A real unfinished slice.
    assert lib.run_case(2) == 0  # All slices done, batch counter disagrees.
    with ThreadPoolExecutor(max_workers=10) as workers:
        assert list(workers.map(lib.run_case, [0] * 100)) == [0] * 100
    assert lib.calls() == lib.callbacks() == 102
    paths = list(Path(os.environ["ATOMESH_MOONCAKE_PROBE_DIR"]).glob("*.jsonl"))
    assert len(paths) == 1
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    active = events[0]
    if sys.argv[2] == "production":
        assert active["event"] == "hook_active" and not active["layout_compatible"]
        assert len(events) == 1  # Foreign ELF layout is not inspected.
    else:
        assert active["layout_compatible"] is True
        assert len(events) == 3  # Successes produce no snapshots.
        waiting, stale = events[1:]
        assert waiting["pending_tasks"] == 1
        assert waiting["all_slices_done_tasks"] == 1
        assert waiting["tasks"][1]["slice_sample"][0]["task_matches"]
        assert waiting["tasks"][1]["slice_sample"][0]["status"] == 1
        assert stale["pending_tasks"] == 0
        assert stale["all_slices_done_tasks"] == 2
        assert stale["finished_task_count"] == 0
        assert all(t["batch_id_matches"] for t in stale["tasks"])
        assert stale["tasks"][0]["request"]["length"] == 1024
    print(
        json.dumps(
            {
                "mode": sys.argv[2],
                "original_calls": lib.calls(),
                "callbacks": lib.callbacks(),
                "events": len(events),
                "passed": True,
            }
        )
    )


def main():
    root = Path(__file__).resolve().parent
    flags = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
    ]
    subprocess.run(
        flags + [str(root / "fixture.cpp"), "-o", str(root / "fixture.so")], check=True
    )
    for mode in ("fixture", "production"):
        target = root / f"probe-{mode}.so"
        subprocess.run(
            flags
            + (["-DMOONCAKE_PROBE_FIXTURE"] if mode == "fixture" else [])
            + [
                str(root / "mooncake_timeout_probe.cpp"),
                "-ldl",
                "-pthread",
                "-o",
                str(target),
            ],
            check=True,
        )
        with tempfile.TemporaryDirectory(prefix="mooncake-probe-test-") as output:
            env = dict(
                os.environ, LD_PRELOAD=str(target), ATOMESH_MOONCAKE_PROBE_DIR=output
            )
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", mode],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    child() if len(sys.argv) > 1 and sys.argv[1] == "--child" else main()
