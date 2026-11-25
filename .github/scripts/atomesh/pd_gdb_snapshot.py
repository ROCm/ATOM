"""Read metadata at two instruction-verified PCs in the pinned native ELF."""

import hashlib
import json
import struct
from pathlib import Path

ENGINE_SHA = "540428ce937a5be29726195730246f053f88d3ddfabaa2af33ad79c7a92198a8"


def decode_frame(rva, registers, read):
    """Offsets are from exact ELF instructions, not inferred C++ object casts."""
    if not (0x3C41E0 <= rva <= 0x3C4210 or rva == 0x3C2B9C):
        return None

    def u64(address):
        return struct.unpack("<Q", read(address, 8))[0]

    def i32(address):
        return struct.unpack("<i", read(address, 4))[0]

    pool = registers["r14"]
    result = {
        "worker_pool": hex(pool),
        "rva": hex(rva),
        "workers": i32(pool + 12),
        "submitted": u64(pool + 0x150),
        "processed": u64(pool + 0x158),
    }
    if not 1 <= result["workers"] <= 256:
        raise ValueError("Unexpected WorkerPool layout")
    if rva == 0x3C2B9C:
        # performPostSend stores the worker queue mutex at rsp+0x28 before
        # pthread_mutex_lock. glibc 2.35 stores its owner TID at mutex+8.
        mutex = u64(registers["rsp"] + 0x28)
        result.update(
            kind="worker_queue_mutex",
            mutex=hex(mutex),
            owner_tid=i32(mutex + 8),
            lock_word=i32(mutex),
        )
    else:
        # This loop stamps endpoint/token metadata on every still-queued slice.
        # Its map-node pointer is rsp+8; vector begin/end are node+0x28/+0x30.
        entry = u64(registers["rsp"] + 8)
        begin, end = u64(entry + 0x28), u64(entry + 0x30)
        cursor = registers["rax"]
        if not (
            begin <= cursor <= end
            and (end - begin) % 8 == 0
            and end == registers["rsi"]
            and (end - begin) // 8 < 100_000_000
        ):
            raise ValueError("Unexpected queue iterator layout")
        result.update(
            kind="full_queue_metadata_sweep",
            queued_slices=(end - begin) // 8,
            remaining_slices=(end - cursor) // 8,
        )
    return result


def main():
    import gdb  # Provided by GDB, not imported during the CPU metadata tests.

    inferior = gdb.selected_inferior()
    rows = Path(f"/proc/{inferior.pid}/maps").read_text().splitlines()
    row = next(
        x
        for x in rows
        if x.endswith("/mooncake/engine.so") and x.split()[2] == "00000000"
    )
    base = int(row.split("-", 1)[0], 16)
    path = Path(f"/proc/{inferior.pid}/root") / row.split()[-1].lstrip("/")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA:
        raise ValueError("Metadata decoder only supports the pinned engine ELF")
    found, errors = [], []
    for thread in inferior.threads():
        try:
            thread.switch()
            frame = gdb.newest_frame()
            for _ in range(8):
                if frame is None:
                    break
                rva = frame.pc() - base
                if 0x3C41E0 <= rva <= 0x3C4210 or rva == 0x3C2B9C:
                    names = (
                        ("r14", "rsp")
                        if rva == 0x3C2B9C
                        else ("r14", "rsp", "rax", "rsi")
                    )
                    regs = {name: int(frame.read_register(name)) for name in names}
                    data = decode_frame(
                        rva, regs, lambda a, n: bytes(inferior.read_memory(a, n))
                    )
                    data["tid"] = thread.ptid[1]
                    found.append(data)
                    break
                frame = frame.older()
        except (gdb.error, ValueError, struct.error) as error:
            errors.append({"tid": thread.ptid[1], "error": str(error)})
    print(
        "[DEBUG-ppdep-queue] "
        + json.dumps({"base": hex(base), "frames": found, "errors": errors})
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, StopIteration) as error:
        print("[DEBUG-ppdep-queue] " + json.dumps({"unavailable": str(error)}))
