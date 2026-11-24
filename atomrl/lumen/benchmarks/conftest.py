# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared pytest fixtures and markers for benchmarks."""

import os
import warnings

import pytest
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Suppress noisy third-party warnings (Megatron-LM, Apex, PyTorch dist)
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", message=".*Aiter backend is selected for fused RoPE.*"
)
warnings.filterwarnings("ignore", message=".*will be removed in.*megatron-core.*")
warnings.filterwarnings("ignore", message=".*megatron.core.transformer.custom_layers.*")
warnings.filterwarnings(
    "ignore", message=".*No device id is provided via.*init_process_group.*"
)
warnings.filterwarnings("ignore", message=".*destroy_process_group.*was not called.*")
warnings.filterwarnings(
    "ignore", message=".*an autograd kernel was not registered to the Autograd key.*"
)

try:
    import aiter  # noqa: F401

    _HAS_AITER = True
except ImportError:
    _HAS_AITER = False

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
AITER = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_AITER,
    reason="CUDA + AITER required for benchmarks",
)


def _dist_rank():
    rank = os.environ.get("RANK")
    if rank is None:
        return None
    return int(rank)


def pytest_configure(config):
    rank = _dist_rank()
    if rank is None:
        return

    config.option.no_header = True
    config.option.no_summary = True

    if rank != 0:
        config.option.verbose = 0
        config.option.capture = "fd"
        config.option.quiet = max(getattr(config.option, "quiet", 0), 2)


def pytest_report_teststatus(report, config):
    rank = _dist_rank()
    if rank is None or rank == 0 or getattr(report, "outcome", None) != "passed":
        return None
    return "", "", ""


@pytest.fixture(scope="session", autouse=True)
def _cleanup_dist():
    """Ensure NCCL process group is destroyed on exit to avoid SIGSEGV."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
