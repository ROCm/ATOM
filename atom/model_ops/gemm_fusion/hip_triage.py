# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public repro / bisect for the E2 failure ``HIP error: operation not supported`` raised by
the FIRST torch allocation after gemm_fusion init (``CpuGpuBuffer`` -> ``torch.zeros_like(
pinned_cpu, device=cuda)``, model_runner.py:755 / utils/__init__.py:726).

STATUS: written, NEVER RUN (no GPU/torch on the authoring host). It is a triage tool, not a
fix and not evidence of a cause. The hypothesis it tests (and its competitors):
  H1  creating the CCO communicator (FusionResources -> gf_comm_create_v1 -> ccoCommCreate,
      which reserves a per-rank VMM heap) changes device/allocator state so the next torch
      allocation fails;
  H2  an earlier async HIP error (CCO init, an unsupported memory op) is only *reported* at the
      next API call, so the allocation is the messenger, not the cause;
  H3  the failure is independent of gemm_fusion (the e2_04 logs have no OFF control: e2_01 and
      e2_03 died before ModelRunner init, so they do not discriminate).

Run on the GPU host, 4 ranks, ONCE PER ARM, with serialized kernels so errors are reported at
the call that caused them::

    AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
      torchrun --nproc_per_node 4 -m atom.model_ops.gemm_fusion.hip_triage --arm off
    AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
      torchrun --nproc_per_node 4 -m atom.model_ops.gemm_fusion.hip_triage --arm on

Reading the result (each rank prints one ``[triage rank=R] stage=... OK|FAIL ...`` line):
  * ``off`` FAILS at ``alloc_pinned_zeros_like``  -> not gemm_fusion (H3); look at the node/torch.
  * ``off`` passes, ``on`` fails at ``alloc_after_fusion_resources`` but ``alloc_before`` passes
    and ``fusion_resources_sync`` passes -> the CCO comm creation changes state (H1).
  * ``on`` fails already at ``fusion_resources_sync`` (or ``fusion_resources``) -> H2: the
    error originates in CCO init; ``rc``/the rocm message there is the real clue.
  * ``--alloc-first`` allocates BEFORE building the resources: if that makes ``on`` pass, a
    candidate mitigation is to create the CCO comm later (after the runner's first allocations),
    which is an ATOM-side ordering change -- do not adopt it from this tool alone.
Env knobs it prints: PYTORCH_HIP_ALLOC_CONF (expandable_segments uses the same VMM APIs the CCO
heap does and is a prime suspect), HSA_*/HIP_* variables.
"""

import argparse
import os
import sys
import traceback


def _log(rank, stage, ok, detail=""):
    print(f"[triage rank={rank}] stage={stage} {'OK' if ok else 'FAIL'} {detail}", flush=True)


def _step(rank, stage, fn, results):
    """Run one stage, force async errors to surface, never raise."""
    import torch

    try:
        out = fn()
        torch.cuda.synchronize()
        _log(rank, stage, True)
        results.append((stage, True))
        return out
    except BaseException as e:  # noqa: BLE001 - report everything, including HIP errors
        _log(rank, stage, False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        results.append((stage, False))
        if os.environ.get("TRIAGE_TRACEBACK") == "1":
            traceback.print_exc()
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("off", "on"), required=True)
    ap.add_argument("--profile", default="minimax_m3_sp4_gfx950_v1")
    ap.add_argument("--alloc-first", action="store_true",
                    help="run the failing allocation before building the CCO resources too")
    ap.add_argument("--default-device", action="store_true",
                    help="torch.set_default_device(cuda) first, as ModelRunner does before the failing line")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    a = ap.parse_args(argv)

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    results = []
    print(f"[triage rank={rank}] torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} "
          f"arm={a.arm} PYTORCH_HIP_ALLOC_CONF={os.environ.get('PYTORCH_HIP_ALLOC_CONF')!r} "
          f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL')!r}", flush=True)

    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    _step(rank, "set_device", lambda: torch.zeros(1, device=device), results)

    _step(rank, "dist_init", lambda: (dist.init_process_group("nccl", device_id=device)
                                      if not dist.is_initialized() else None), results)
    _step(rank, "dist_barrier", lambda: dist.barrier(), results)
    # The exact failing call, BEFORE any gemm_fusion state exists (the baseline).
    if a.default_device:
        torch.set_default_device(device)

    def pinned_zeros_like():
        cpu = torch.zeros(128, dtype=torch.int64, device="cpu", pin_memory=True)  # CpuGpuBuffer line 725
        return torch.zeros_like(cpu, device=device)                              # CpuGpuBuffer line 726

    _step(rank, "alloc_before", pinned_zeros_like, results)
    if not all(ok for _, ok in results) and not a.keep_going:
        return _finish(rank, results)

    if a.arm == "on":
        # What ATOM's init_gemm_fusion does, in the same order: load the profile, then build the
        # real FusionResources (compiles gf_comm, ccoCommCreate with the per-rank VMM heap).
        gf = None

        def build():
            nonlocal gf
            import importlib

            gf = importlib.import_module("aiter.gemm_fusion")
            gf.load_profile(a.profile)
            from atom.model_ops.gemm_fusion.resources import fusion_resources_cls

            caps = gf.detect_capabilities(None)
            print(f"[triage rank={rank}] caps={sorted(caps)} arch={gf.detect_arch()} cu={gf.detect_cu()}", flush=True)
            return fusion_resources_cls(gf)(rank=rank, world=world, group=None, device=local, caps=caps)

        res = _step(rank, "fusion_resources", build, results)
        _step(rank, "fusion_resources_sync", lambda: torch.cuda.synchronize(), results)
        if res is not None:
            _step(rank, "fusion_resources_barrier", lambda: dist.barrier(), results)
        if not all(ok for _, ok in results) and not a.keep_going:
            return _finish(rank, results)

    # The failing line, AFTER init: this is the one that raised in e2_04.
    _step(rank, "alloc_after_fusion_resources" if a.arm == "on" else "alloc_pinned_zeros_like",
          pinned_zeros_like, results)
    _step(rank, "plain_cuda_zeros", lambda: torch.zeros(1 << 20, dtype=torch.float32, device=device), results)
    _step(rank, "large_cuda_empty_256MiB", lambda: torch.empty(256 << 20, dtype=torch.uint8, device=device), results)
    return _finish(rank, results)


def _finish(rank, results):
    failed = [s for s, ok in results if not ok]
    print(f"[triage rank={rank}] SUMMARY {'ALL_OK' if not failed else 'FAILED_AT=' + failed[0]} "
          f"stages={[(s, 'ok' if ok else 'FAIL') for s, ok in results]}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
