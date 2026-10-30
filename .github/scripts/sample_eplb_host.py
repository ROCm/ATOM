"""Low-rate host/process pressure samples for the bounded EPLB comparison."""
import json
from pathlib import Path
import signal
import sys
import time

running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


with Path(sys.argv[1]).open("w", buffering=1) as out:
    while running:
        row = {"time_ns": time.time_ns(), "system": {}, "processes": []}
        for path in (
            "/proc/loadavg", "/proc/meminfo", "/proc/vmstat", "/proc/stat",
            "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io",
            "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.max",
            "/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpuset.cpus.effective",
        ):
            row["system"][path] = read(path)
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            comm = read(entry / "comm")
            if comm and any(k in comm.lower() for k in ("python", "atom")):
                row["processes"].append({
                    "pid": entry.name, "comm": comm.strip(),
                    "stat": read(entry / "stat"), "status": read(entry / "status"),
                    "io": read(entry / "io"), "schedstat": read(entry / "schedstat"),
                    "cgroup": read(entry / "cgroup"),
                })
        out.write(json.dumps(row) + "\n")
        for _ in range(5):
            if not running:
                break
            time.sleep(1)
