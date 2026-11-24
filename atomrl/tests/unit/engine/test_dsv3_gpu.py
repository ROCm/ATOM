# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DSv3 checks that need real GPUs: model parallelism, and real-checkpoint parity.

Three properties the CPU tests structurally cannot reach:

* **TP/EP** -- the engine reconstructs full tensors from ``sharded_state_dict()``
  metadata, so what matters is that MLA describes itself correctly there. Wrong
  metadata does not raise; it hands the rollout a mis-assembled attention block.
* **PP** -- each stage must publish its layers under global numbers. DSv3 always
  carries a ``moe_layer_freq`` list, which is exactly the heterogeneous case
  where deriving the stage offset from checkpoint metadata silently returned 0.
* **Real weights** -- a synthetic config has no ``rope_scaling`` and no
  ``n_group``, so no fixture-driven test can catch a field the builder never
  reads. Only a logit comparison against the HF reference fails when the model
  is subtly wrong rather than unbuildable.

Each runs as a subprocess: they need real process groups, and initialising
Megatron's parallel state in-process would leak into the rest of the suite.
"""

import glob
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SHARD_WORKER = HERE / "_dsv3_shard_worker.py"
PARITY_WORKER = HERE / "_dsv3_parity_worker.py"

_HF_HOME = os.environ.get("HF_HOME", "/data/rl_data/hf")
_CKPT_GLOB = f"{_HF_HOME}/hub/models--bzantium--tiny-deepseek-v3/snapshots/*/"


def _visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except (AssertionError, ImportError, RuntimeError):
        return 0


def _checkpoint() -> str | None:
    """The real DSv3 checkpoint, from ``DSV3_CKPT`` or the HF cache."""
    explicit = os.environ.get("DSV3_CKPT")
    if explicit and os.path.isdir(explicit):
        return explicit
    found = sorted(glob.glob(_CKPT_GLOB))
    return found[0] if found else None


def _run(
    worker: pathlib.Path, sentinel: str, *, launcher: list[str], **env_extra
) -> None:
    """Run a worker and require its success sentinel on stdout."""
    env = dict(os.environ)
    env["ATOMRL_ROOT"] = str(REPO_ROOT)
    env.update({k: str(v) for k, v in env_extra.items()})

    proc = subprocess.run(
        [sys.executable, *launcher, str(worker)],
        capture_output=True,
        check=False,
        text=True,
        timeout=1800,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert sentinel in proc.stdout, (
        f"{worker.name} failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )


def _run_shard_worker(port: str, **topology: int) -> None:
    _run(
        SHARD_WORKER,
        "DSV3_SHARD_OK",
        launcher=[
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={port}",
        ],
        HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES", "0,1"),
        **{f"DSV3_{k.upper()}": v for k, v in topology.items()},
    )


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for TP=2/EP=2")
def test_mla_describes_itself_correctly_under_tp2_ep2():
    """MLA global shapes and shard axes must match what the gather assumes.

    Asserted inside the worker on every rank: q/kv down-projections replicated,
    up-projections column-parallel on dim 0, output row-parallel on dim 1, fused
    LoRA norms replicated, experts split by EP, router replicated.
    """
    _run_shard_worker("29607", tp=2, pp=1, ep=2, layers=3)


@pytest.mark.multigpu
@pytest.mark.skipif(_visible_gpus() < 2, reason="needs 2 GPUs for PP=2")
def test_pipeline_stages_publish_global_layer_numbers():
    """Each stage must publish its layers under global numbers, dense layer included.

    4 layers over 2 stages with ``first_k_dense_replace=1`` puts the dense layer
    on stage 0 and an all-MoE stage 1, so a collapsed offset shows up as a
    coverage gap, a collision, and a dense/MoE mismatch at once.
    """
    _run_shard_worker("29608", tp=1, pp=2, ep=1, layers=4)


@pytest.mark.gpu
@pytest.mark.skipif(_visible_gpus() < 1, reason="needs a GPU")
@pytest.mark.skipif(
    _checkpoint() is None, reason=f"no DSv3 checkpoint under {_CKPT_GLOB}"
)
def test_bridge_reproduces_the_reference_logits_in_fp32():
    """Real weights through the bridge must match the HF reference exactly in fp32.

    ``bzantium/tiny-deepseek-v3`` is production DeepSeek-V3 geometry -- hidden
    7168, 128 heads, q_lora 1536, kv_lora 512, qk 128+64, v 128, moe_ffn 2048,
    YaRN factor 40 -- trimmed to 6 layers and 8 experts, so it is 10.7 GB and
    fits on one GPU. It also ships ``first_k_dense_replace=3``, exercising a
    multi-layer dense prefix the other fixtures do not.

    This is the test that caught the RoPE and expert-group defaults: with them
    left at Megatron's values the same comparison gave 11.4% mean relative error
    and 71.9% top-1 agreement. Fetch the checkpoint with::

        HF_HOME=/data/rl_data/hf python -c "from huggingface_hub import \
            snapshot_download; snapshot_download('bzantium/tiny-deepseek-v3')"
    """
    _run(
        PARITY_WORKER,
        "DSV3_PARITY_OK",
        launcher=[],
        DSV3_CKPT=_checkpoint(),
        HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES", "0"),
    )
