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


def wait_for(path, deadline):
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(0.1)


def select_host(root):
    # Match vLLM's IPv4 selection inside the actual Docker network namespace.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
    outer = os.environ.get("STORE_SMOKE_OUTER_HOST_IP", "")
    checks = {}
    for address in dict.fromkeys([host, outer, "127.0.0.1"]):
        if not address:
            continue
        check = {}
        try:
            with socket.socket() as sock:
                sock.bind((address, 0))
            check["local_bind"] = True
        except OSError as exc:
            check["local_bind"] = False
            check["bind_error"] = str(exc)
        try:
            with socket.socket() as listener:
                listener.bind(("0.0.0.0", 0))
                listener.listen(1)
                listener.settimeout(2)
                with socket.create_connection(
                    (address, listener.getsockname()[1]), timeout=2
                ):
                    conn, _ = listener.accept()
                    conn.close()
            check["tcp_self_connect"] = True
        except OSError as exc:
            check["tcp_self_connect"] = False
            check["connect_error"] = str(exc)
        checks[address] = check
    report = {"selected_host": host, "outer_host": outer, "checks": checks}
    (root / "network.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not all(checks[host].get(key) for key in ("local_bind", "tcp_self_connect")):
        raise RuntimeError(
            "Container-selected address failed local connectivity checks"
        )
    os.environ["STORE_SMOKE_HOST_IP"] = host


def worker(args):
    import torch
    from mooncake.engine import TransferEngine
    from mooncake.store import MooncakeDistributedStore

    output = Path(args.output)
    result = {"rank": args.rank, "device": args.device, "segment": args.segment}
    direct = None
    store = None
    deadline = time.monotonic() + args.timeout
    try:
        torch.cuda.set_device(args.rank)
        torch.cuda.init()
        host = os.environ["STORE_SMOKE_HOST_IP"]
        direct = TransferEngine()
        result["direct_init"] = direct.initialize(host, "P2PHANDSHAKE", "rdma", "")
        (output / f"direct-ready-{args.rank}").touch()
        wait_for(output / f"start-setup-{args.rank}", deadline)
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
        (output / f"setup-{args.rank}.json").write_text(json.dumps(result, indent=2))
        wait_for(output / "start-io", deadline)
        if result["setup_rc"] == 0:
            key = f"store-smoke-rank-{args.rank}"
            payload = b"mooncake-cpu-store-registration" * 4096
            result["put_rc"] = store.put(key, payload)
            received = store.get(key)
            result["get_type"] = type(received).__name__
            result["get_length"] = len(received) if received is not None else None
            result["roundtrip"] = received == payload
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        traceback.print_exc()
        result["error"] = repr(exc)
    (output / f"rank-{args.rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    # Keep successful registrations live until all peers have reported.
    deadline = time.monotonic() + args.timeout
    while not (output / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    if store is not None:
        store.close()


def run_case(root, name, ranks, devices, segment, serialized=False, timeout=180):
    import mooncake

    output = root / name
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
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
                        "--timeout",
                        str(timeout),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            )
        deadline = time.monotonic() + timeout
        # All direct engines coexist before changing Store setup concurrency.
        for rank, proc in enumerate(processes):
            while not (output / f"direct-ready-{rank}").exists():
                if proc.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Direct engine readiness timed out")
                time.sleep(0.1)
        for rank, proc in enumerate(processes):
            (output / f"start-setup-{rank}").touch()
            if serialized:
                while not (output / f"setup-{rank}.json").exists():
                    if (
                        proc.poll() is not None
                        or (output / f"rank-{rank}.json").exists()
                    ):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Serialized Store setup timed out")
                    time.sleep(0.1)
        while not all(
            (output / f"setup-{rank}.json").exists()
            or (output / f"rank-{rank}.json").exists()
            or proc.poll() is not None
            for rank, proc in enumerate(processes)
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError("Store setup timed out")
            time.sleep(0.1)
        (output / "start-io").touch()
        while time.monotonic() < deadline:
            if all(
                (output / f"rank-{rank}.json").exists() or proc.poll() is not None
                for rank, proc in enumerate(processes)
            ):
                break
            time.sleep(1)
    except TimeoutError as exc:
        (output / "timeout.txt").write_text(str(exc))
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
        setup_path = output / f"setup-{rank}.json"
        if not path.exists() and setup_path.exists():
            report.update(json.loads(setup_path.read_text()))
        report["process_rc"] = proc.returncode
        reports.append(report)
    result = {
        "name": name,
        "ranks": reports,
        "serialized_setup": serialized,
        "registration_passed": len(reports) == ranks
        and all(r.get("direct_init") == 0 and r.get("setup_rc") == 0 for r in reports),
        "passed": len(reports) == ranks
        and all(
            r.get("direct_init") == 0
            and r.get("setup_rc") == 0
            and r.get("put_rc") == 0
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
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    select_host(root)
    nics = sorted(path.name for path in Path("/sys/class/infiniband").iterdir())
    if len(nics) != 8:
        raise RuntimeError(f"Expected eight RDMA NICs, found {nics}")
    small = 64 * 1024**2
    results = [run_case(root, "single-all", 1, [""], small)]
    results.append(run_case(root, "eight-all", 8, [""] * 8, small))
    results.append(
        run_case(root, "eight-all-serialized", 8, [""] * 8, small, serialized=True)
    )
    results.append(run_case(root, "eight-per-nic", 8, nics, small))
    if results[-1]["passed"]:
        results.append(
            run_case(
                root,
                "eight-per-nic-full",
                8,
                nics,
                int(224.875 * 1024**3),
                timeout=420,
            )
        )
    if results[2]["passed"]:
        results.append(
            run_case(
                root,
                "eight-all-serialized-full",
                8,
                [""] * 8,
                int(224.875 * 1024**3),
                serialized=True,
                timeout=420,
            )
        )
    (root / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
