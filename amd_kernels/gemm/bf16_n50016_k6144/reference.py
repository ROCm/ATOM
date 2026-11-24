# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract: y = x @ w^T for bf16 x [M, 6144] and the bf16 lm_head shard w [50016, 6144].

The product of the bf16 operands in fp64, rounded once to bf16. The kernel accumulates in fp32 in
a fixed order, so it differs from the reference (and from hipBLASLt) only in the sum order: per row
cosine >= 0.99999 and relative L2 <= 1e-3, deterministic bits. NaN / inf propagate as in an fp32
accumulation (a NaN in a row of x makes that output row NaN).
"""
import torch

N, K = 50016, 6144


def bf16_gemm_n50016_k6144_reference(x: torch.Tensor, w: torch.Tensor,
                                      row_indices: torch.Tensor | None = None) -> torch.Tensor:
    if row_indices is not None:
        x = x[row_indices]
    return (x.double() @ w.double().t()).bfloat16()
