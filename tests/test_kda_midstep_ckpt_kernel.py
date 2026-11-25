# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's midstep checkpoints through `write_state_checkpoints`.

One launch per layer writes every checkpoint of a step: an interior target's
state from its row of FlashKDA's `snapshot_out` (`h_rows`), an end target's
from the runtime slot, and each one's conv window from the step's conv input.
The pool's speculation-rollback columns are left as they were. (GDN's use of
the same kernel, from a chunk kernel's per-chunk states, is
test_gdn_midstep_state_gpu.py's.)

The writer is loaded from its file: importing it through `atom.model_ops.fla_ops`
would pull in AITER. On a GPU the checks run in-process; on the CPU, under
Triton's interpreter in a child process (the flag must precede the triton
import). Skipped without Triton.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton", reason="the checkpoint writer is a Triton kernel")

from atom.models.kimi_k3.midstep import kda_snapshots

ROOT = Path(__file__).resolve().parents[1]
CONV_LEN, ROLLBACK, GRID = 3, 2, 32
SLOTS, H, V, K, D = 9, 2, 4, 4, 6


def load_writer():
    """`write_state_checkpoints`, from `state_checkpoint.py` alone."""
    spec = importlib.util.spec_from_file_location(
        "state_checkpoint_under_test",
        ROOT / "atom" / "model_ops" / "fla_ops" / "state_checkpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_state_checkpoints


def _pools(conv_transposed: bool, device):
    g = torch.Generator().manual_seed(0)
    ssm = torch.randn(SLOTS, H, V, K, generator=g)
    if conv_transposed:
        # As the KDA layer sees it: a view of a [slots, width, dim] allocation.
        conv = torch.randn(SLOTS, CONV_LEN + ROLLBACK, D, generator=g).transpose(1, 2)
    else:
        conv = torch.randn(SLOTS, D, CONV_LEN + ROLLBACK, generator=g)
    return ssm.to(device), conv.to(device)


def check_interior_and_end_targets(write, device, conv_transposed):
    """Row 0: interior targets at 32 and 64 and the anchor at its end, 96.
    Row 1, 48 tokens after it: one interior target at 32."""
    ssm, conv = _pools(conv_transposed, device)
    snaps = kda_snapshots(
        [[(4, 32), (5, 64), (6, 96)], [(7, 32)]],
        state_rows=[0, 1],
        num_cached_tokens=[0, 0],
        num_scheduled_tokens=[96, 48],
        runtime_slots=[1, 2],
        num_slots=SLOTS,
        grid=GRID,
        conv_len=CONV_LEN,
        device=device,
    )
    assert snaps.targets == ((0, 32), (0, 64), (1, 32))
    g = torch.Generator().manual_seed(1)
    snap_out = torch.randn(len(snaps.targets), H, V, K, generator=g).to(device)
    mixed_qkv = torch.randn(144, D, generator=g).to(device)
    before_ssm, before_conv = ssm.clone(), conv.clone()

    write(**snaps.checkpoint_args(snap_out, ssm, mixed_qkv, conv))

    for i, slot in enumerate((4, 5, 7)):
        assert torch.equal(ssm[slot], snap_out[i])
    assert torch.equal(ssm[6], before_ssm[1]), "the anchor is the runtime slot's state"
    for slot, start in ((4, 32), (5, 64), (6, 96), (7, 96 + 32)):
        window = mixed_qkv[start - CONV_LEN : start].T
        assert torch.equal(conv[slot, :, :CONV_LEN], window), slot
        # Left as they were: a resumed prefill reads only the first CONV_LEN.
        assert torch.equal(conv[slot, :, CONV_LEN:], before_conv[slot, :, CONV_LEN:])
    for slot in (0, 1, 2, 3, 8):
        assert torch.equal(ssm[slot], before_ssm[slot])
        assert torch.equal(conv[slot], before_conv[slot])


def check_an_end_target_alone(write, device):
    ssm, conv = _pools(False, device)
    snaps = kda_snapshots(
        [[(4, 64)]],
        [0],
        [0],
        [64],
        [1],
        num_slots=SLOTS,
        grid=GRID,
        conv_len=CONV_LEN,
        device=device,
    )
    mixed_qkv = torch.randn(64, D).to(device)
    before = ssm.clone()
    write(**snaps.checkpoint_args(None, ssm, mixed_qkv, conv))
    assert torch.equal(ssm[4], before[1])
    assert torch.equal(conv[4, :, :CONV_LEN], mixed_qkv[61:64].T)


def run_all_checks(device):
    write = load_writer()
    for conv_transposed in (False, True):
        check_interior_and_end_targets(write, device, conv_transposed)
    check_an_end_target_alone(write, device)


def test_the_writer_passes_every_check_under_the_triton_interpreter():
    code = (
        "import importlib.util;"
        f"s = importlib.util.spec_from_file_location('kernel_checks', {__file__!r});"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m.run_all_checks('cpu')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={
            **os.environ,
            "TRITON_INTERPRET": "1",
            "HIP_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@GPU
@pytest.mark.parametrize("conv_transposed", [False, True])
def test_one_launch_writes_interior_and_end_targets(conv_transposed):
    check_interior_and_end_targets(load_writer(), "cuda", conv_transposed)


@GPU
def test_an_end_target_alone_needs_no_snapshot():
    check_an_end_target_alone(load_writer(), "cuda")
