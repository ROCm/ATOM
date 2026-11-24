# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four contiguous vocabulary shards, each [V,6144] bf16, to [M,6144].

Read shard token//V, local row token%V. Invalid IDs produce positive zero.
Bf16 subnormals and signed zeros become positive zero, as when AITER adds the
one owner to positive zeros with fp32 flush-to-zero. Other finite values
are bitwise equal to the embedding lookup. There is no payload reduction. Replays and eager calls must be bitwise equal on every rank. Weights
are immutable, have one owner and one layout, and are never copied by the op.
"""
import torch


def vocab_gather_reference(shards,ids):
    weights=torch.cat(shards,dim=0)
    valid=(ids>=0)&(ids<weights.shape[0])
    out=weights[ids.clamp(0,weights.shape[0]-1)].clone()
    out[~valid]=0
    out[out.abs()<torch.finfo(torch.float32).tiny]=0
    return out
