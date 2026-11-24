# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Bitwise row exchange: each input [sum(counts), 50016] is sorted by owner.

Rank r receives rows [sum(counts[:r]):sum(counts[:r+1])] from each source,
concatenated along the vocabulary into bf16 [counts[r], 200064]. There is
no arithmetic, including on NaN payloads. Counts are common across ranks.
"""
import torch

N = 50016
WORLD = 4


def all_to_all_reference(inputs: list[torch.Tensor], counts: list[int], rank: int) -> torch.Tensor:
    lo = sum(counts[:rank])
    hi = lo + counts[rank]
    return torch.cat([x[lo:hi].to(inputs[rank].device) for x in inputs], dim=1)


def merge_received_reference(received: torch.Tensor) -> torch.Tensor:
    """[source, owned row, 50016] -> [owned row, source vocabulary]."""
    return received.permute(1, 0, 2).reshape(received.shape[1], WORLD * N).contiguous()
