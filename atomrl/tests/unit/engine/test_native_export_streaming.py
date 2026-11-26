# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Multi-GPU parity: streaming Megatron-native export vs the original gather."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER = HERE / "_native_export_worker.py"


def _visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except (AssertionError, ImportError, RuntimeError):
        return 0


def _run(*, nproc: int, port: str, prefetch: int | str, **topo: int) -> None:
    env = dict(os.environ)
    env["ATOMRL_ROOT"] = str(REPO_ROOT)
    env["ATOMRL_SYNC_PREFETCH_MB"] = str(prefetch)
    env["HIP_VISIBLE_DEVICES"] = os.environ.get(
        "HIP_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(nproc)),
    )
    env["CUDA_VISIBLE_DEVICES"] = env["HIP_VISIBLE_DEVICES"]
    for key, val in topo.items():
        env[f"NATIVE_{key.upper()}"] = str(val)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
            str(WORKER),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=1800,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert "NATIVE_EXPORT_OK" in proc.stdout, (
        f"{WORKER.name} failed (rc={proc.returncode}) topo={topo} prefetch={prefetch}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for dense TP=2")
@pytest.mark.parametrize("prefetch", [0, 1024])
def test_dense_tp2(prefetch: int) -> None:
    _run(nproc=2, port="29701", prefetch=prefetch, tp=2, pp=1, ep=1, moe=0)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 4, reason="needs 4 GPUs for dense TP2 PP2")
@pytest.mark.parametrize("prefetch", [0, 1024, "mixed"])
def test_dense_tp2_pp2(prefetch: int | str) -> None:
    _run(nproc=4, port="29702", prefetch=prefetch, tp=2, pp=2, ep=1, moe=0)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 4, reason="needs 4 GPUs for MoE EP=4")
@pytest.mark.parametrize("prefetch", [0, 1024])
def test_moe_ep4(prefetch: int) -> None:
    _run(nproc=4, port="29703", prefetch=prefetch, tp=1, pp=1, ep=4, moe=1)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 4, reason="needs 4 GPUs for MoE EP2 TP2")
@pytest.mark.parametrize("prefetch", [0, 1024])
def test_moe_ep2_tp2(prefetch: int) -> None:
    _run(nproc=4, port="29704", prefetch=prefetch, tp=2, pp=1, ep=2, moe=1)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 8, reason="needs 8 GPUs for MoE EP4 PP2")
@pytest.mark.parametrize("prefetch", [0, 1024, "mixed"])
def test_moe_ep4_pp2(prefetch: int | str) -> None:
    _run(nproc=8, port="29705", prefetch=prefetch, tp=1, pp=2, ep=4, moe=1)
