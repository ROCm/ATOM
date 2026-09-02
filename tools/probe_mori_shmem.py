"""Check whether two MoRI shmem groups can share a GPU.

Rapidserve puts two processes on every GPU (one prefill, one decode), and with
``--enable-dp-attention --enable-expert-parallel`` each of those is a member of
its own expert-parallel group. Both groups call
``mori.shmem.shmem_torch_process_group_init``, so every GPU ends up hosting two
symmetric heaps. MoRI's peer path is HIP IPC (``P2PMemoryRegion`` holds a
``hipIpcMemHandle_t``), which is also what rapidserve itself uses to alias
weights and KV cache.

This reproduces that layout with no ATOM code in the picture. Like ATOM, each
group is its own ``torch.distributed`` world on its own master port -- mori's
``shmem_torch_process_group_init`` broadcasts the heap UID with ``src=0``, which
torch interprets as a *global* rank, so a group that does not contain global
rank 0 cannot initialise at all. Each rank stamps its heap with a fingerprint
identifying ``(group, pe)``, barriers, then reads every peer through
``shmem_ptr_p2p`` and checks it got the fingerprint it asked for.

A rank reading its own group's data everywhere means the heaps are properly
isolated and the corruption seen under rapidserve+EP lies elsewhere -- in the
dispatch/combine kernels rather than the heap. A rank that reads the *other*
group's fingerprint means peer resolution is aliasing across groups, which is
exactly the failure that would corrupt only tokens routed off-rank (and so
would stay invisible to a repeated prompt, whose routing is degenerate).

Run two groups concurrently, one per GPU set::

    for g in 0 1; do
      torchrun --nproc-per-node=8 --master-port=$((29700 + g)) \
        tools/probe_mori_shmem.py --group-id $g &
    done; wait

A single group on its own is the control and should always pass.
"""

import argparse
import ctypes
import os
import sys

import torch
import torch.distributed as dist

# hipMemcpyKind
_H2D = 1
_D2H = 2

_WORDS = 64  # fingerprint length in int32s; long enough to spot partial writes


def _hip():
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMemcpy.restype = ctypes.c_int
    lib.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.hipDeviceSynchronize.restype = ctypes.c_int
    return lib


def _check(lib, err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with hipError {err}")


def _fingerprint(group: int, pe: int) -> int:
    """A value unique to one (group, pe) and readable at a glance."""
    return 1000 * (group + 1) + pe


def _decode_fingerprint(value: int):
    if value <= 0 or value % 1000 >= 1000:
        return None
    group = value // 1000 - 1
    pe = value % 1000
    if group < 0:
        return None
    return group, pe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group-id",
        type=int,
        default=0,
        help="which shmem group this launch is; run two concurrently to put "
        "two heaps on every GPU the way rapidserve does",
    )
    ap.add_argument(
        "--group-name",
        default="mori",
        help="process-group name to register; ATOM uses 'mori' in every process",
    )
    args = ap.parse_args()

    group = args.group_id
    pe = int(os.environ["RANK"])
    pes = int(os.environ["WORLD_SIZE"])
    device = int(os.environ.get("LOCAL_RANK", pe))

    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")

    from mori import shmem

    # Mirror ATOM: register this launch's whole world under the mori name.
    world_group = dist.distributed_c10d._get_default_group()
    torch._C._distributed_c10d._register_process_group(args.group_name, world_group)
    shmem.shmem_torch_process_group_init(args.group_name)

    if shmem.shmem_mype() != pe or shmem.shmem_npes() != pes:
        print(
            f"[group {group} pe {pe}] BAD IDENTITY: mori says "
            f"pe={shmem.shmem_mype()} "
            f"npes={shmem.shmem_npes()}, expected pe={pe} npes={pes}",
            flush=True,
        )

    lib = _hip()
    nbytes = _WORDS * 4
    base = shmem.shmem_malloc(nbytes)

    mine = _fingerprint(group, pe)
    host = (ctypes.c_int32 * _WORDS)(*([mine] * _WORDS))
    _check(lib, lib.hipMemcpy(base, host, nbytes, _H2D), "stamp")
    _check(lib, lib.hipDeviceSynchronize(), "sync after stamp")

    shmem.shmem_barrier_all()
    dist.barrier()

    problems = []
    out = (ctypes.c_int32 * _WORDS)()
    for peer in range(pes):
        ptr = shmem.shmem_ptr_p2p(base, pe, peer)
        if not ptr:
            problems.append(f"peer {peer}: shmem_ptr_p2p returned NULL")
            continue
        _check(lib, lib.hipMemcpy(out, ctypes.c_void_p(ptr), nbytes, _D2H), "read")
        _check(lib, lib.hipDeviceSynchronize(), "sync after read")
        vals = set(out)
        want = _fingerprint(group, peer)
        if vals == {want}:
            continue
        if len(vals) != 1:
            problems.append(f"peer {peer}: torn read, saw {sorted(vals)[:4]}")
            continue
        got = vals.pop()
        seen = _decode_fingerprint(got)
        if seen is None:
            problems.append(f"peer {peer}: want {want}, got unrecognised {got}")
        elif seen[0] != group:
            problems.append(
                f"peer {peer}: CROSS-GROUP ALIAS -- want group {group} pe {peer}, "
                f"got group {seen[0]} pe {seen[1]}"
            )
        else:
            problems.append(
                f"peer {peer}: want pe {peer}, got pe {seen[1]} (same group)"
            )

    tag = f"[group {group} pe {pe} gpu {device}]"
    if problems:
        print(f"{tag} FAIL", flush=True)
        for p in problems:
            print(f"{tag}   {p}", flush=True)
    else:
        print(f"{tag} ok -- all {pes} peers read back correctly", flush=True)

    failed = torch.tensor([1 if problems else 0])
    dist.all_reduce(failed)
    dist.barrier()

    shmem.shmem_free(base)
    if pe == 0:
        n = int(failed.item())
        if n:
            print(
                f"\n=== group {group}: {n}/{pes} ranks saw bad peer data. "
                "Any CROSS-GROUP ALIAS line means two heaps on one GPU collide "
                "and the fault is in MoRI's peer resolution, not the kernels. ===",
                flush=True,
            )
        else:
            print(
                f"\n=== group {group}: all {pes} ranks clean -- this group's "
                "heap is intact, so look at dispatch/combine instead. ===",
                flush=True,
            )
    dist.destroy_process_group()
    return 1 if int(failed.item()) else 0


if __name__ == "__main__":
    sys.exit(main())
