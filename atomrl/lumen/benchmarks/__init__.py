# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Lumen performance benchmarks.

Run individual benchmarks::

    python -m benchmarks.bench_kernel_launch
    python -m benchmarks.bench_rope_fusion
    python -m benchmarks.bench_fp8_param_allgather
    python -m benchmarks.bench_wgrad_delay
    torchrun --nproc_per_node=2 -m benchmarks.bench_comm_overlap

Or via pytest (collected as regular tests)::

    pytest benchmarks/ -v -s
"""
