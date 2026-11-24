# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""RCCL/NCCL health-check for AMD MI-series GPUs.

Can be launched in two ways:

    # Recommended — torchrun handles process spawning:
    torchrun --nproc_per_node=8 tests/env/test_nccl.py
    torchrun --nproc_per_node=1 tests/env/test_nccl.py   # single-GPU smoke test

    # Standalone — script spawns worker processes itself (no torchrun needed):
    python tests/env/test_nccl.py              # uses all visible GPUs
    python tests/env/test_nccl.py --nproc 4   # use 4 GPUs only

    # Explicitly set the NCCL socket interface (if auto-detect fails):
    python tests/env/test_nccl.py --ifname eth0

    # With extra RCCL diagnostics:
    NCCL_DEBUG=INFO python tests/env/test_nccl.py

Exit code:
    0  — all tests passed
    1  — one or more tests failed

Troubleshooting "socketPollConnect / no POLLOUT events":
    This error means RCCL's bootstrap TCP sockets cannot connect.  The root
    cause is almost always a wrong network interface selection.  Run with
    NCCL_DEBUG=INFO, look for "NET" lines to see which interface RCCL chose,
    then re-run with --ifname <correct-interface>.
    Common values: lo, eth0, ens3, bond0, ib0 …
    List available interfaces:  ip -o link show | awk -F': ' '{print $2}'
"""

import argparse
import os
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not dist.is_initialized(),
        reason="NCCL collectives run under torchrun or this file's own worker",
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_rank0() -> bool:
    return _rank() == 0


def log(msg: str, *, error: bool = False) -> None:
    """Print only from rank-0 (or always for errors)."""
    if _is_rank0() or error:
        prefix = f"[rank{_rank()}]" if error else "[rank0]"
        tag = " ERROR" if error else ""
        print(f"{prefix}{tag} {msg}", flush=True)


def separator(title: str = "") -> None:
    if _is_rank0():
        line = "=" * 60
        print(f"\n{line}")
        if title:
            print(f"  {title}")
            print(line)


PASSED: list = []
FAILED: list = []


def record(name: str, ok: bool, detail: str = "") -> None:
    symbol = "PASS" if ok else "FAIL"
    msg = f"  [{symbol}] {name}"
    if detail:
        msg += f"  ({detail})"
    log(msg, error=(not ok))
    (PASSED if ok else FAILED).append(name)


# ---------------------------------------------------------------------------
# Network interface helpers
# ---------------------------------------------------------------------------


def _list_interfaces() -> list[tuple[str, str]]:
    """Return [(ifname, ip_addr)] for UP interfaces with an IPv4 address."""
    results = []
    try:
        # /proc/net/fib_trie approach — works inside containers without tools
        with open("/proc/net/if_inet6") as _:
            pass  # just a probe; we use a different file below
    except OSError:
        pass

    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            parts = line.split()
            # format: index  ifname  inet  ip/prefix  …
            if len(parts) >= 4 and parts[2] == "inet":
                ifname = parts[1]
                ip = parts[3].split("/")[0]
                results.append((ifname, ip))
    except (OSError, subprocess.SubprocessError):
        # Fallback: just try to get hostname IP
        try:
            hostname_ip = socket.gethostbyname(socket.gethostname())
            results.append(("(hostname)", hostname_ip))
        except OSError:
            pass
    return results


def _probe_loopback() -> bool:
    """Check whether a TCP loopback connection on 127.0.0.1 works."""
    import threading

    success = [False]

    def _server():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            # Signal port to client thread via shared list
            success.append(port)
            srv.settimeout(2.0)
            try:
                conn, _ = srv.accept()
                conn.close()
                success[0] = True
            except OSError:
                pass

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    t.join(0.2)  # wait until port is known
    if len(success) < 2:
        return False
    port = success[1]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0):
            pass
    except OSError:
        return False
    t.join(2.0)
    return success[0]


def _suggest_ifname(interfaces: list[tuple[str, str]]) -> str:
    """Pick the most suitable interface for NCCL_SOCKET_IFNAME."""
    # Prefer loopback if TCP loopback works (single-node case)
    if _probe_loopback():
        for name, _ in interfaces:
            if name.startswith("lo"):
                return name

    # Otherwise prefer non-virtual Ethernet interfaces
    prefer = []
    for name, ip in interfaces:
        if name.startswith(("eth", "ens", "enp", "em", "bond", "ib")):
            prefer.append(name)
    if prefer:
        return prefer[0]

    # Fallback: first available (skip loopback)
    for name, _ in interfaces:
        if not name.startswith("lo"):
            return name

    return "lo"


# ---------------------------------------------------------------------------
# 1. Environment checks (no distributed required)
# ---------------------------------------------------------------------------


def check_environment() -> None:
    separator("1. Environment / ROCm stack")

    log(f"  PyTorch  : {torch.__version__}")

    rocm_ver = getattr(torch.version, "hip", None)
    cuda_ver = torch.version.cuda
    if rocm_ver:
        log(f"  ROCm     : {rocm_ver}")
    else:
        log(f"  CUDA     : {cuda_ver}")
    record(
        "pytorch_build_has_gpu_support", rocm_ver is not None or cuda_ver is not None
    )

    n_gpus = torch.cuda.device_count()
    log(f"  GPUs     : {n_gpus}")
    record("at_least_one_gpu_visible", n_gpus >= 1)

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        log(f"    GPU {i}: {props.name}  mem={props.total_memory // (1024**3)} GiB")

    try:
        nccl_ver = torch.cuda.nccl.version()
        log(f"  NCCL/RCCL: {nccl_ver}")
        record("nccl_library_available", True, str(nccl_ver))
    except (OSError, RuntimeError) as exc:
        record("nccl_library_available", False, str(exc))

    # Network interfaces — critical for NCCL_SOCKET_IFNAME diagnosis
    log("  Network interfaces (IPv4):")
    ifaces = _list_interfaces()
    if ifaces:
        for name, ip in ifaces:
            log(f"    {name:<12} {ip}")
    else:
        log("    (could not enumerate interfaces)")
    loopback_ok = _probe_loopback()
    log(
        f"  Loopback TCP probe : {'OK' if loopback_ok else 'FAIL — lo may not work for NCCL'}"
    )
    record("loopback_tcp_works", loopback_ok)

    log("  Key env vars:")
    for var in [
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "HSA_ENABLE_SDMA",
    ]:
        log(f"    {var}={os.environ.get(var, '<unset>')}")


# ---------------------------------------------------------------------------
# 2. Distributed init
# ---------------------------------------------------------------------------


def init_distributed() -> bool:
    separator("2. Distributed init (NCCL/RCCL backend)")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Bind device BEFORE init_process_group so RCCL knows which GPU owns this
    # rank and avoids the "device currently unknown" warning / hang.
    torch.cuda.set_device(local_rank)
    log(f"  local_rank={local_rank}  device=cuda:{local_rank}")
    log(f"  NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '<unset>')}")

    try:
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        log(f"  init_process_group OK  world_size={_world()}")
        record("init_process_group", True, f"world={_world()}")
        return True
    except TypeError:
        # Older PyTorch builds may not accept the device_id kwarg.
        try:
            dist.init_process_group(backend="nccl")
            log(f"  init_process_group OK (no device_id)  world_size={_world()}")
            record("init_process_group", True, f"world={_world()}, no device_id kwarg")
            return True
        except RuntimeError as exc:
            record("init_process_group", False, str(exc))
            log(str(exc), error=True)
            _print_socket_hint()
            return False
    except RuntimeError as exc:
        record("init_process_group", False, str(exc))
        log(str(exc), error=True)
        _print_socket_hint()
        return False


def _print_socket_hint() -> None:
    """Print targeted hints when a socket error is detected."""
    if not _is_rank0():
        return
    ifaces = _list_interfaces()
    names = [n for n, _ in ifaces]
    log("", error=True)
    log("  ── Diagnosis: NCCL bootstrap socket failure ──────────────", error=True)
    log("  RCCL could not establish its internal TCP bootstrap.", error=True)
    log("  Most likely cause: wrong network interface selected.", error=True)
    log("", error=True)
    log(f"  Available interfaces: {names}", error=True)
    log("", error=True)
    log("  Try one of the following (re-run with the correct --ifname):", error=True)
    for name, ip in ifaces:
        log(f"    python tests/env/test_nccl.py --ifname {name}  # ip={ip}", error=True)
    log("", error=True)
    log("  Or get detailed RCCL logs:", error=True)
    log(
        "    NCCL_DEBUG=INFO python tests/env/test_nccl.py 2>&1 | grep -E 'NET|socket|IFNAME'",
        error=True,
    )


# ---------------------------------------------------------------------------
# 3. Collective operations
# ---------------------------------------------------------------------------


def _alloc(shape, dtype=torch.float32) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype, device=f"cuda:{torch.cuda.current_device()}")


def test_barrier() -> None:
    try:
        dist.barrier()
        record("barrier", True)
    except RuntimeError as exc:
        record("barrier", False, str(exc))


def test_allreduce(sizes=(1, 1024, 1024 * 1024)) -> None:
    for numel in sizes:
        try:
            t = _alloc(numel)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.barrier()
            expected = float(_world())
            ok = torch.allclose(t, torch.full_like(t, expected))
            record(
                f"all_reduce_{numel}",
                ok,
                "sum check failed" if not ok else f"{numel} floats",
            )
        except RuntimeError as exc:
            record(f"all_reduce_{numel}", False, str(exc))


def test_broadcast() -> None:
    try:
        t = _alloc(1024)
        if _rank() == 0:
            t.fill_(42.0)
        dist.broadcast(t, src=0)
        ok = torch.allclose(t, torch.full_like(t, 42.0))
        record("broadcast", ok)
    except RuntimeError as exc:
        record("broadcast", False, str(exc))


def test_allgather() -> None:
    try:
        numel = 512
        t = _alloc(numel).fill_(float(_rank()))
        out = [torch.zeros(numel, device=t.device) for _ in range(_world())]
        dist.all_gather(out, t)
        ok = all(
            torch.allclose(out[r], torch.full_like(out[r], float(r)))
            for r in range(_world())
        )
        record("all_gather", ok)
    except RuntimeError as exc:
        record("all_gather", False, str(exc))


def test_reduce_scatter() -> None:
    try:
        numel = _world() * 256
        t = _alloc(numel).fill_(1.0)
        out = torch.zeros(numel // _world(), device=t.device)
        dist.reduce_scatter_tensor(out, t, op=dist.ReduceOp.SUM)
        expected = float(_world())
        ok = torch.allclose(out, torch.full_like(out, expected))
        record("reduce_scatter_tensor", ok)
    except RuntimeError as exc:
        record("reduce_scatter_tensor", False, str(exc))


def test_allgather_into_tensor() -> None:
    try:
        chunk = 256
        t = _alloc(chunk).fill_(float(_rank()))
        out = torch.zeros(chunk * _world(), device=t.device)
        dist.all_gather_into_tensor(out, t)
        ok = True
        for r in range(_world()):
            expected = torch.full((chunk,), float(r), device=t.device)
            if not torch.allclose(out[r * chunk : (r + 1) * chunk], expected):
                ok = False
                break
        record("all_gather_into_tensor", ok)
    except RuntimeError as exc:
        record("all_gather_into_tensor", False, str(exc))


def test_p2p() -> None:
    """Ring send/recv: each rank sends to (rank+1) % world."""
    if _world() < 2:
        record("p2p_ring_send_recv", True, "skipped (world_size=1)")
        return
    try:
        rank = _rank()
        world = _world()
        send_t = _alloc(256).fill_(float(rank))
        recv_t = torch.zeros(256, device=send_t.device)

        dst = (rank + 1) % world
        src = (rank - 1) % world

        send_op = dist.P2POp(dist.isend, send_t, dst)
        recv_op = dist.P2POp(dist.irecv, recv_t, src)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for r in reqs:
            r.wait()

        ok = torch.allclose(recv_t, torch.full_like(recv_t, float(src)))
        record("p2p_ring_send_recv", ok)
    except RuntimeError as exc:
        record("p2p_ring_send_recv", False, str(exc))


# ---------------------------------------------------------------------------
# 4. Bandwidth benchmark (all-reduce)
# ---------------------------------------------------------------------------


def bench_allreduce_bandwidth() -> None:
    separator("4. All-reduce bandwidth (256 MiB tensor)")
    if _world() < 2:
        log("  Skipped (world_size=1)")
        return

    numel = 256 * 1024 * 1024 // 4  # 256 MiB of float32
    t = _alloc(numel)

    for _ in range(3):  # warm-up
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dist.barrier()

    N = 10
    start = time.perf_counter()
    for _ in range(N):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    # Standard bus-bandwidth formula for all-reduce
    bytes_transferred = numel * 4 * 2 * (_world() - 1) / _world()
    bw_gb_s = bytes_transferred * N / elapsed / 1e9
    log(f"  {N} iters in {elapsed:.3f}s  →  {bw_gb_s:.1f} GB/s (bus bandwidth)")
    record("allreduce_bandwidth_nonzero", bw_gb_s > 0.1, f"{bw_gb_s:.1f} GB/s")


# ---------------------------------------------------------------------------
# 5. Mixed-precision collectives (bf16 / fp16)
# ---------------------------------------------------------------------------


def test_dtype_collectives() -> None:
    separator("5. Mixed-precision collectives")
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
        try:
            t = torch.ones(
                1024, dtype=dtype, device=f"cuda:{torch.cuda.current_device()}"
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            # torch.full() defaults to CPU; pass device=t.device so both
            # sides of allclose are on the same GPU.
            expected = torch.full((1024,), float(_world()), device=t.device)
            ok = torch.allclose(t.float(), expected)
            record(f"all_reduce_{name}", ok)
        except RuntimeError as exc:
            record(f"all_reduce_{name}", False, str(exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary() -> None:
    if not _is_rank0():
        return
    separator("Summary")
    total = len(PASSED) + len(FAILED)
    log(f"  {len(PASSED)}/{total} tests passed")
    if FAILED:
        log("  Failed tests:")
        for name in FAILED:
            log(f"    - {name}", error=True)
    else:
        log("  All RCCL/NCCL components are healthy.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Worker entry-point (used by both torchrun and mp.spawn)
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    ifname: str = "lo",
    ib_disable: str = "1",
) -> None:
    """Run all test phases for a single rank.

    NCCL_SOCKET_IFNAME and NCCL_IB_DISABLE are passed as explicit function
    arguments instead of relying on os.environ inheritance, because
    torch.multiprocessing.start_processes(spawn) does NOT guarantee that
    changes made to os.environ in the parent are visible in spawned children.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    # These MUST be set before any NCCL call (especially init_process_group).
    os.environ["NCCL_SOCKET_IFNAME"] = ifname
    os.environ["NCCL_IB_DISABLE"] = ib_disable

    check_environment()

    ok = init_distributed()
    if not ok:
        _print_summary()
        sys.exit(1)

    separator("3. Collective operations")
    test_barrier()
    test_allreduce()
    test_broadcast()
    test_allgather()
    test_reduce_scatter()
    test_allgather_into_tensor()
    test_p2p()

    bench_allreduce_bandwidth()
    test_dtype_collectives()

    dist.barrier()
    dist.destroy_process_group()

    _print_summary()
    if FAILED:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main — detect launch mode and dispatch
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCCL/NCCL health-check")
    p.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of GPUs to use in standalone mode (default: all visible GPUs)",
    )
    p.add_argument(
        "--master-addr",
        default="127.0.0.1",
        help="Master address for standalone mode (default: 127.0.0.1)",
    )
    p.add_argument(
        "--master-port",
        default="29500",
        help="Master port for standalone mode (default: 29500)",
    )
    p.add_argument(
        "--ifname",
        default=None,
        help=(
            "Network interface for NCCL bootstrap sockets "
            "(sets NCCL_SOCKET_IFNAME).  Auto-detected when not specified."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    launched_by_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if launched_by_torchrun:
        # torchrun already set RANK / LOCAL_RANK / WORLD_SIZE / MASTER_*
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        # In torchrun mode the user controls NCCL_SOCKET_IFNAME externally;
        # default to "lo" only if not already set.
        ifname = os.environ.get("NCCL_SOCKET_IFNAME") or args.ifname or "lo"
        ib_disable = os.environ.get("NCCL_IB_DISABLE", "1")
        _worker(rank, world_size, master_addr, master_port, ifname, ib_disable)
        return 0 if not FAILED else 1

    # ---- Standalone mode: spawn worker processes ourselves -----------------
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("ERROR: No GPUs found. Cannot run distributed tests.", file=sys.stderr)
        return 1

    world_size = args.nproc if args.nproc is not None else n_gpus
    world_size = min(world_size, n_gpus)

    # Determine which interface RCCL should use for bootstrap TCP sockets.
    #
    # IMPORTANT: we do NOT rely on os.environ inheritance here.  With
    # torch.multiprocessing.start_processes(spawn), child processes start a
    # fresh Python interpreter and changes made to os.environ in the parent
    # are NOT reliably visible in workers.  Instead, we pass ifname as an
    # explicit function argument so it is pickled and received correctly.
    #
    # Default is "lo" (loopback) because:
    #   • standalone mode is always single-node (all GPUs on one host)
    #   • loopback always works for intra-host TCP
    #   • actual GPU data (all-reduce etc.) travels over PCIe/xGMI, not TCP
    #   • physical NICs (eno1, eno0, …) may be unreachable inside containers
    #     (e.g. Kubernetes pods with Cilium CNI)
    if args.ifname:
        ifname = args.ifname
    elif "NCCL_SOCKET_IFNAME" in os.environ:
        ifname = os.environ["NCCL_SOCKET_IFNAME"]
    else:
        ifname = "lo"

    ib_disable = os.environ.get("NCCL_IB_DISABLE", "1")

    print(f"[launcher] Standalone mode: spawning {world_size} worker process(es)")
    print(f"[launcher] master={args.master_addr}:{args.master_port}")
    print(f"[launcher] NCCL_SOCKET_IFNAME={ifname}  NCCL_IB_DISABLE={ib_disable}")
    print(
        "[launcher] tip: if init fails, retry with --ifname <interface>  "
        "(e.g. eth0, ens3, eno1) or NCCL_DEBUG=INFO\n"
    )

    mp.start_processes(
        _worker,
        args=(world_size, args.master_addr, args.master_port, ifname, ib_disable),
        nprocs=world_size,
        start_method="spawn",
    )
    # mp.start_processes raises if any child exits non-zero.
    return 0


if __name__ == "__main__":
    sys.exit(main())
