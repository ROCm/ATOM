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
    result = {
        "rank": args.rank,
        "device": args.device,
        "segment": args.segment,
        "local_buffer": args.local_buffer,
        "glibc_tunables": os.environ.get("GLIBC_TUNABLES", ""),
    }
    direct = None
    store = None
    gpu_buffer = None
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
            args.local_buffer,
            "rdma",
            args.device,
            f"127.0.0.1:{args.port}",
        )
        result["setup_seconds"] = time.monotonic() - start
        (output / f"smaps-{args.rank}.txt").write_text(
            Path("/proc/self/smaps_rollup").read_text()
        )
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
            if args.gpu_bytes:
                gpu_buffer = torch.empty(
                    args.gpu_bytes, dtype=torch.uint8, device="cuda"
                )
                result["gpu_bytes"] = args.gpu_bytes
                result["direct_gpu_register_rc"] = direct.register_memory(
                    gpu_buffer.data_ptr(), args.gpu_bytes
                )
                result["store_gpu_register_rc"] = store.register_buffer(
                    gpu_buffer.data_ptr(), args.gpu_bytes
                )
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


def run_case(
    root,
    name,
    ranks,
    devices,
    segment,
    serialized=False,
    timeout=180,
    local_buffer=1024**3,
    env_overrides=None,
    gpu_bytes=0,
):
    import mooncake

    output = root / name
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(env_overrides or {})
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
                        "--local-buffer",
                        str(local_buffer),
                        "--gpu-bytes",
                        str(gpu_bytes),
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
            and "error" not in r
            and (
                not gpu_bytes
                or (
                    r.get("gpu_bytes") == gpu_bytes
                    and r.get("direct_gpu_register_rc") == 0
                    and r.get("store_gpu_register_rc") == 0
                )
            )
            and r["process_rc"] == 0
            for r in reports
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def capacity_cases(root, nics):
    results = [run_case(root, "eight-all-original", 8, [""] * 8, 64 * 1024**2)]
    probe = Path(__file__).with_name("pd_mooncake_registration_probe.py")
    for backend in ("verbs", "mooncake"):
        for thp in (False, True):
            for size, count in ((1024**3, 8), (8 * 1024**3, 1)):
                name = f"{backend}-{'thp' if thp else 'base'}-{size}-{count}"
                command = [
                    sys.executable,
                    str(probe),
                    "--backend",
                    backend,
                    "--size",
                    str(size),
                    "--count",
                    str(count),
                    "--output",
                    str(root / f"{name}.json"),
                ]
                if thp:
                    command.append("--thp")
                with (root / f"{name}.log").open("w") as log:
                    try:
                        rc = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=60,
                            check=False,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        rc = 124
                (root / f"{name}.rc").write_text(str(rc))
    small = 64 * 1024**2
    results.append(
        run_case(root, "eight-all-small-buffer", 8, [""] * 8, small, local_buffer=small)
    )
    thp_env = {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}
    results.append(
        run_case(
            root,
            "single-all-full-thp",
            1,
            [""],
            int(224.875 * 1024**3),
            local_buffer=small,
            env_overrides=thp_env,
            timeout=180,
        )
    )
    if results[-1]["passed"]:
        results.append(
            run_case(
                root,
                "eight-all-full-thp",
                8,
                [""] * 8,
                int(224.875 * 1024**3),
                local_buffer=small,
                env_overrides=thp_env,
                timeout=420,
            )
        )
    (root / "result.json").write_text(json.dumps(results, indent=2))


def thp_store_cases(root):
    library = os.environ["ATOMESH_THP_ALLOCATOR_LIBRARY"]
    results = []
    for name, count, segment in (
        ("eight-all-small-thp", 8, 64 * 1024**2),
        ("single-all-full-thp", 1, int(224.875 * 1024**3)),
        ("eight-all-full-thp", 8, int(224.875 * 1024**3)),
    ):
        result = run_case(
            root,
            name,
            count,
            [""] * count,
            segment,
            env_overrides={
                "LD_PRELOAD": library,
                "ATOMESH_THP_LOCAL_BUFFER_BYTES": str(1024**3),
                "ATOMESH_THP_GLOBAL_SEGMENT_BYTES": str(segment),
                "ATOMESH_THP_NUMA_REGION_BYTES": str(segment // 2),
                "ATOMESH_THP_NUMA_FALLBACK": "1",
            },
            timeout=600 if count == 8 and segment > 1024**3 else 180,
            gpu_bytes=20 * 1024**3 if segment > 1024**3 else 0,
        )
        results.append(result)
        (root / "result.json").write_text(json.dumps(results, indent=2))
        if not result["passed"]:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--segment", type=int, default=64 * 1024**2)
    parser.add_argument("--local-buffer", type=int, default=1024**3)
    parser.add_argument("--gpu-bytes", type=int, default=0)
    parser.add_argument(
        "--plan",
        choices=["registration", "capacity", "thp-store"],
        default="registration",
    )
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
    if args.plan == "capacity":
        capacity_cases(root, nics)
        return
    if args.plan == "thp-store":
        thp_store_cases(root)
        return
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
