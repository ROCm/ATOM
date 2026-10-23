"""A rank-local allocation failure in ``Glm5MonoKernel.__init__`` must end in a
UNIFORM refusal, not a hang.

``SymmetricPeerBuffer``'s constructor is a collective, so a rank that unwinds
out of an earlier rank-local allocation (an OOM, say) leaves its peers blocked
in a gloo collective the failing rank never joins. This runs
``Glm5MonoKernel._allocate_rank_local`` itself, on four gloo ranks on the CPU,
with one rank made to fail. It does not run the constructor: that needs a GPU
and FlyDSL, so the line in ``__init__`` that calls the method is not covered
here.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
import types
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
FAILER = 1
DEADLINE = 60


def _worker(rank, world, init, fail_at, out):
    # The kernel module compiles through FlyDSL at import and nothing here
    # launches a kernel, so a stand-in lets `op` import without FlyDSL.
    stub = types.ModuleType("atom.models.glm52.mono.kernels.kernel")
    stub.build_glm5_monokernel = None
    sys.modules[stub.__name__] = stub
    try:
        from atom.models.glm52.mono.kernels import op
        from atom.mono.runtime.consensus import MonoUnsupported

        dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
        names = (
            "w_index_k",
            "s_index_k",
            "w_index_w",
            "w_index_q",
            "s_index_q",
            "g_index_k",
            "b_index_k",
        )
        this = SimpleNamespace(
            W=SimpleNamespace(
                t={name: torch.zeros(8, dtype=torch.uint8) for name in names}
            ),
            packed={},
            with_indexer=True,
            kv_cache_layout=op.KvCacheLayout.ATOM,
            S=4,
            index_request_width=4,
            index_page_size=16,
            index_max_seq=64,
            index_block_table_stride=4,
            scr_layout={"_bytes": 64},
            _owns_runtime=True,
            npes=world,
        )
        op.pack_fp8 = op.pack_bf16 = lambda tensor: tensor
        if rank == FAILER and fail_at == "pack_fp8":

            def oom(_tensor):
                raise torch.OutOfMemoryError("HIP out of memory in pack_fp8")

            op.pack_fp8 = oom
        if rank == FAILER and fail_at == "scratch":
            real_zeros = torch.zeros

            def zeros(*args, **kwargs):
                # once: gloo's own all_gather_object allocates a uint8 zeros too
                if kwargs.get("dtype") is torch.uint8 and not fired:
                    fired.append(True)
                    raise torch.OutOfMemoryError("HIP out of memory in scratch")
                return real_zeros(*args, **kwargs)

            fired = []

            op.torch.zeros = zeros

        try:
            op.Glm5MonoKernel._allocate_rank_local(this, "cpu", dist.group.WORLD)
            out[rank] = "ALLOCATED"
        except MonoUnsupported as error:
            out[rank] = f"REFUSED: {error}"
    except Exception:  # noqa: BLE001 - reported to the parent, not swallowed
        out[rank] = "ERROR: " + traceback.format_exc(limit=3)
        # Production keeps this process alive and merely stops it joining the
        # next collective. Exiting here would let gloo see the disconnect and
        # make the peers raise, which hides the very hang this test is for.
        time.sleep(DEADLINE + 15)
    finally:
        # Teardown is not the behaviour under test, and destroy_process_group
        # can itself block once peers are gone.
        sys.stdout.flush()
        os._exit(0)


def _run(fail_at):
    with tempfile.TemporaryDirectory() as d:
        out = mp.Manager().dict()
        context = mp.start_processes(
            _worker,
            args=(WORLD, f"file://{d}/store", fail_at, out),
            nprocs=WORLD,
            join=False,
            start_method="spawn",
        )
        # ProcessContext.join returns False while ANY child still runs, so one
        # call is not a verdict: loop it to a real deadline.
        deadline = time.time() + DEADLINE
        while time.time() < deadline:
            if context.join(timeout=2):
                return dict(out)
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        pytest.fail(
            f"hung: ranks that reported before the {DEADLINE}s deadline: {dict(out)}"
        )


def test_with_no_failing_rank_every_rank_allocates():
    assert _run(None) == {rank: "ALLOCATED" for rank in range(WORLD)}


@pytest.mark.parametrize("fail_at", ["pack_fp8", "scratch"])
def test_one_ranks_allocation_failure_refuses_every_rank(fail_at):
    out = _run(fail_at)
    assert sorted(out) == list(range(WORLD))
    for rank, verdict in out.items():
        assert verdict.startswith("REFUSED: mono rank-local allocation failed"), (
            rank,
            verdict,
        )
        assert f"rank {FAILER}: OutOfMemoryError" in verdict, (rank, verdict)
