"""Reproduce Store registration without loading a model or compiling kernels."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path


def worker(args):
    import torch
    from mooncake.engine import TransferEngine
    from mooncake.store import MooncakeDistributedStore

    output = Path(args.output)
    result = {"rank": args.rank, "device": args.device, "segment": args.segment}
    direct = None
    store = None
    try:
        torch.cuda.set_device(args.rank)
        torch.cuda.init()
        host = socket.gethostbyname(socket.gethostname())
        direct = TransferEngine()
        result["direct_init"] = direct.initialize(host, "P2PHANDSHAKE", "rdma", "")
        store = MooncakeDistributedStore()
        start = time.monotonic()
        result["setup_rc"] = store.setup(
            host,
            "P2PHANDSHAKE",
            args.segment,
            1024**3,
            "rdma",
            args.device,
            f"127.0.0.1:{args.port}",
        )
        result["setup_seconds"] = time.monotonic() - start
        if result["setup_rc"] == 0:
            key = f"store-smoke-rank-{args.rank}"
            payload = b"mooncake-cpu-store-registration" * 4096
            result["put_rc"] = store.put(key, payload)
            result["roundtrip"] = store.get(key) == payload
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        traceback.print_exc()
        result["error"] = repr(exc)
    (output / f"rank-{args.rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    # Keep successful registrations live until all peers have reported.
    deadline = time.monotonic() + 210
    while not (output / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    if store is not None:
        store.close()


def run_case(root, name, ranks, devices, segment, relaxed=None):
    import mooncake

    output = root / name
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    if relaxed is not None:
        env["MC_IB_PCI_RELAXED_ORDERING"] = str(relaxed)
    processes = []
    handles = []
    master_log = (output / "master.log").open("w")
    master = subprocess.Popen(
        [str(Path(mooncake.__file__).parent / "mooncake_master"), "--port", str(port)],
        stdout=master_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        for attempt in range(60):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                if master.poll() is not None:
                    raise RuntimeError("Master exited before listening")
                time.sleep(0.5)
        else:
            raise TimeoutError("Master did not listen")
        for rank in range(ranks):
            log = (output / f"rank-{rank}.log").open("w")
            handles.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--rank",
                        str(rank),
                        "--device",
                        devices[rank],
                        "--segment",
                        str(segment),
                        "--port",
                        str(port),
                        "--output",
                        str(output),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if all(
                (output / f"rank-{rank}.json").exists() or proc.poll() is not None
                for rank, proc in enumerate(processes)
            ):
                break
            time.sleep(1)
    finally:
        (output / "release").touch()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        master.terminate()
        try:
            master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait()
        master_log.close()
        for log in handles:
            log.close()
    reports = []
    for rank, proc in enumerate(processes):
        path = output / f"rank-{rank}.json"
        report = (
            json.loads(path.read_text())
            if path.exists()
            else {"rank": rank, "missing": True}
        )
        report["process_rc"] = proc.returncode
        reports.append(report)
    result = {
        "name": name,
        "ranks": reports,
        "relaxed_ordering": relaxed,
        "passed": len(reports) == ranks
        and all(
            r.get("direct_init") == 0
            and r.get("setup_rc") == 0
            and r.get("roundtrip") is True
            and r["process_rc"] == 0
            for r in reports
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--segment", type=int, default=64 * 1024**2)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    nics = sorted(path.name for path in Path("/sys/class/infiniband").iterdir())
    if len(nics) != 8:
        raise RuntimeError(f"Expected eight RDMA NICs, found {nics}")
    small = 64 * 1024**2
    results = [run_case(root, "single-all", 1, [""], small)]
    results.append(run_case(root, "eight-all", 8, [""] * 8, small))
    results.append(run_case(root, "eight-all-no-relaxed", 8, [""] * 8, small, 0))
    results.append(run_case(root, "eight-per-nic", 8, nics, small))
    if results[-1]["passed"]:
        results.append(
            run_case(root, "eight-per-nic-full", 8, nics, int(224.875 * 1024**3))
        )
    (root / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
