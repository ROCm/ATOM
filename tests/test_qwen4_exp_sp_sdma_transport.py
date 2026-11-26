# SPDX-License-Identifier: MIT
"""The real SDMA reduce-scatter / all-gather (mori + FlyDSL) on 2 and 4 GPUs.

Each rank builds every rank's inputs from one seed, so it checks its own
result against a reference: the all-gather bit for bit, the reduce-scatter
against the fp32 sum. Token counts not divisible by the TP size, several
chunk / push counts and a slow producer (`torch.cuda._sleep`) cover the
padding, the chunk offsets and the waits. Like a decoder, the steps
alternate all-gather and reduce-scatter so both receive slots get reused."""

import importlib
import itertools
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2560
CAPACITY = 16384
# (tokens, reduce-scatter chunks, all-gather pushes)
STEPS = [
    (12000, 4, 4),
    (11998, 2, 3),
    (4097, 3, 1),
    (CAPACITY, 1, 2),
    (1025, 2, 4),
    (3, 1, 1),
    (12000, 4, 2),
]
_LIBDRM = "/opt/rocm/lib/rocm_sysdeps/lib"


def _sdma_stack_missing() -> str | None:
    if not torch.cuda.is_available():
        return "no GPU"
    for name in ("mori.cco", "flydsl.compiler"):
        try:
            importlib.import_module(name)
        except ImportError as e:
            return f"no {name}: {e}"
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops.communication import SdmaComm  # noqa: F401
    except ImportError as e:
        return f"no aiter SdmaComm: {e}"
    if get_gfx_runtime() != "gfx942":
        return "SDMA collectives are gfx942 only"
    return None


def _rank(rank, world, port, failures):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    from atom.model_ops.qwen4_exp import sp_sdma

    group = dist.group.WORLD
    col = sp_sdma.get_sdma_collectives(CAPACITY, HIDDEN, torch.bfloat16, group)
    errors = []
    if col is None:
        errors.append("SDMA unavailable (see the warning)")
    else:
        for step, (tokens, chunks, pushes) in enumerate(STEPS):
            local = -(-tokens // world)
            chunks, pushes = min(chunks, local), min(pushes, local)
            gen = torch.Generator(device="cuda").manual_seed(step)
            # All-gather, written and pushed range by range.
            full = torch.randn(
                world * local, HIDDEN, generator=gen, device="cuda"
            ).bfloat16()
            writer = col.all_gather(local, tokens, pushes)
            cuts = [local * k // pushes for k in range(pushes + 1)]
            for a, b in itertools.pairwise(cuts):
                torch.cuda._sleep(1_000_000)
                writer.rows(a, b).copy_(full[rank * local + a : rank * local + b])
                writer.push(a, b)
            if not torch.equal(writer.finish(), full[:tokens]):
                errors.append(f"all-gather step {step} ({tokens} tokens)")
            # Reduce-scatter of every rank's partial sums.
            parts = torch.randn(
                world, tokens, HIDDEN, generator=gen, device="cuda"
            ).bfloat16()
            torch.cuda._sleep(3_000_000)
            got = col.reduce_scatter(parts[rank] * 1, local, chunks).result()
            ref = torch.zeros(world * local, HIDDEN, device="cuda")
            ref[:tokens] = parts.float().sum(0)
            ref = ref[rank * local : (rank + 1) * local]
            err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
            # One bf16 rounding of a sum taken in another order (NaN fails).
            if not err <= 8e-3:
                errors.append(f"reduce-scatter step {step}: rel err {err:.3g}")
        torch.cuda.synchronize()
        col.close()
    failures[rank] = errors
    dist.destroy_process_group()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world", [2, 4])
def test_sdma_collectives_match_reference(monkeypatch, world):
    missing = _sdma_stack_missing()
    if missing:
        pytest.skip(missing)
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setenv("MORI_SOCKET_IFNAME", "lo")
    # mori must load the libdrm copy HIP uses (see sp_sdma).
    drm = [
        f"{_LIBDRM}/librocm_sysdeps_drm.so.2",
        f"{_LIBDRM}/librocm_sysdeps_drm_amdgpu.so.1",
    ]
    if all(os.path.exists(p) for p in drm):
        preload = os.environ.get("LD_PRELOAD", "")
        monkeypatch.setenv("LD_PRELOAD", ":".join(drm + ([preload] if preload else [])))
    failures = mp.Manager().dict()
    ctx = mp.spawn(_rank, args=(world, _port(), failures), nprocs=world, join=False)
    for _ in range(600):  # a missed signal shows up as a timeout
        if ctx.join(timeout=1):
            break
    else:
        for p in ctx.processes:
            p.kill()
        pytest.fail("a rank is stuck waiting for an SDMA transfer")
    assert dict(failures) == {r: [] for r in range(world)}
