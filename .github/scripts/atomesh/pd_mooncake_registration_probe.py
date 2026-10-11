"""Compare base-page and THP host memory registration on one RDMA NIC."""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path


def libc_api():
    lib = ctypes.CDLL(None, use_errno=True)
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t
    lib.mmap.argtypes = [
        ptr,
        size,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    lib.mmap.restype = ptr
    lib.munmap.argtypes = [ptr, size]
    lib.madvise.argtypes = [ptr, size, ctypes.c_int]
    return lib


def allocate(lib, size, thp):
    alignment = 2 * 1024**2
    raw = lib.mmap(None, size + alignment, 3, 0x22, -1, 0)
    if raw == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap")
    address = (raw + alignment - 1) // alignment * alignment
    if address > raw:
        lib.munmap(raw, address - raw)
    tail = raw + size + alignment - address - size
    if tail:
        lib.munmap(address + size, tail)
    if lib.madvise(address, size, 14 if thp else 15) != 0:
        lib.munmap(address, size)
        raise OSError(ctypes.get_errno(), "madvise page policy")
    # Fault the entire mapping before registering, identical for both policies.
    ctypes.memset(address, 0, size)
    return address


def smaps(address):
    section = []
    selected = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        token = line.split()[0]
        if "-" in token and ":" not in token:
            start, end = (int(x, 16) for x in token.split("-"))
            selected = start <= address < end
        if selected:
            section.append(line)
    return section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="rdma5")
    parser.add_argument("--backend", choices=["verbs", "mooncake"], default="verbs")
    parser.add_argument("--thp", action="store_true")
    parser.add_argument("--size", type=int, default=1024**3)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lib = libc_api()
    result = vars(args).copy()
    result["registrations"] = []
    buffers = []
    registrations = []
    engine = None
    devices = context = pd = None
    verbs = None
    if args.backend == "mooncake":
        import torch
        from mooncake.engine import TransferEngine

        torch.cuda.init()
        engine = TransferEngine()
        rc = engine.initialize(
            os.environ["STORE_SMOKE_HOST_IP"], "P2PHANDSHAKE", "rdma", args.device
        )
        if rc:
            raise RuntimeError(f"TransferEngine init: {rc}")
    else:
        verbs = ctypes.CDLL("libibverbs.so.1", use_errno=True)
        ptr = ctypes.c_void_p
        for name, argtypes, restype in [
            (
                "ibv_get_device_list",
                [ctypes.POINTER(ctypes.c_int)],
                ctypes.POINTER(ptr),
            ),
            ("ibv_get_device_name", [ptr], ctypes.c_char_p),
            ("ibv_open_device", [ptr], ptr),
            ("ibv_alloc_pd", [ptr], ptr),
            ("ibv_reg_mr", [ptr, ptr, ctypes.c_size_t, ctypes.c_int], ptr),
            ("ibv_dereg_mr", [ptr], ctypes.c_int),
            ("ibv_dealloc_pd", [ptr], ctypes.c_int),
            ("ibv_close_device", [ptr], ctypes.c_int),
            ("ibv_free_device_list", [ctypes.POINTER(ptr)], None),
        ]:
            function = getattr(verbs, name)
            function.argtypes = argtypes
            function.restype = restype
        count = ctypes.c_int()
        devices = verbs.ibv_get_device_list(ctypes.byref(count))
        for index in range(count.value):
            if verbs.ibv_get_device_name(devices[index]).decode() == args.device:
                context = verbs.ibv_open_device(devices[index])
                break
        if not context:
            raise RuntimeError(f"Could not open {args.device}")
        pd = verbs.ibv_alloc_pd(context)
        if not pd:
            raise OSError(ctypes.get_errno(), "ibv_alloc_pd")
    try:
        for index in range(args.count):
            address = allocate(lib, args.size, args.thp)
            buffers.append(address)
            entry = {"index": index, "address": address, "smaps_before": smaps(address)}
            start = time.monotonic()
            ctypes.set_errno(0)
            if engine is not None:
                rc = engine.register_memory(address, args.size)
                handle = address if rc == 0 else None
            else:
                handle = verbs.ibv_reg_mr(pd, address, args.size, 7)
                rc = 0 if handle else -1
            entry.update(
                rc=rc, errno=ctypes.get_errno(), seconds=time.monotonic() - start
            )
            entry["smaps_after"] = smaps(address)
            result["registrations"].append(entry)
            print(json.dumps(entry), flush=True)
            if handle is None:
                break
            registrations.append(handle)
    finally:
        cleanup = []
        for handle in reversed(registrations):
            cleanup.append(
                engine.unregister_memory(handle)
                if engine is not None
                else verbs.ibv_dereg_mr(handle)
            )
        result["deregister_rcs"] = cleanup
        for address in buffers:
            lib.munmap(address, args.size)
        if verbs is not None:
            result["dealloc_pd_rc"] = verbs.ibv_dealloc_pd(pd)
            result["close_device_rc"] = verbs.ibv_close_device(context)
            verbs.ibv_free_device_list(devices)
        result["passed"] = len(registrations) == args.count and all(
            rc == 0 for rc in cleanup
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
