# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + entry function of the Markov step tail (packed embedding rows)."""

import functools

import torch

from .reference import NB, RANK, V


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def warmup_gumbel_combine_markov_embed(device: torch.device) -> None:
    """Compile the kernel; it serves every row count and step."""
    lm = torch.zeros(1, NB, dtype=torch.float32, device=device)
    la = torch.zeros(1, NB, dtype=torch.int64, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    w1 = torch.zeros(V, RANK, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(1, 7, dtype=torch.int64, device=device)
    prev = torch.zeros(1, dtype=torch.int64, device=device)
    packed=torch.empty(16,RANK,dtype=torch.bfloat16,device=device)
    gumbel_combine_markov_embed_packed(lm,la,rows,w1,ids,0,prev,packed)
    torch.cuda.synchronize(device)


@functools.cache
def _packed_kernel_exe():
    from .kernel_packed import compile_gumbel_combine_markov_embed_packed
    return compile_gumbel_combine_markov_embed_packed()


def gumbel_combine_markov_embed_packed(local_max, local_argmax, local_rows, markov_w1,
                                      draft_ids, step, prev, packed_embed):
    """One launch for the tail of Markov draft step ``step`` over the n <= 64 owned rows (``reference.py``): the
    Gumbel combine of ``local_max`` fp32 [n, 196] / ``local_argmax`` int64 [n, 196] -> ``draft_ids[:n, step]``,
    ``prev[local_rows]`` and the next step's embedding row ``markov_w1[token]`` into row r of ``packed_embed``.

    packed_embed has 16/32/64 padded rows and 256 columns. Its physical layout
    is [m16,k32,k8,row16,feature8], the input of bf16_n200064_k256. Only valid
    row bytes are written; padded rows remain unused by the GEMM. CUDA-graph safe.
    """
    n=local_max.shape[0]
    pad=16 if n<=16 else (32 if n<=32 else 64)
    assert n<=64 and local_max.shape==(n,NB) and local_max.dtype==torch.float32 and local_max.is_contiguous()
    assert local_argmax.shape==(n,NB) and local_argmax.dtype==torch.int64 and local_argmax.is_contiguous()
    assert local_rows.shape==(n,) and local_rows.dtype==torch.int64 and local_rows.is_contiguous()
    assert markov_w1.shape==(V,RANK) and markov_w1.dtype==torch.bfloat16 and markov_w1.is_contiguous()
    assert packed_embed.shape==(pad,RANK) and packed_embed.dtype==torch.bfloat16 and packed_embed.is_contiguous()
    assert draft_ids.dtype==torch.int64 and draft_ids.dim()==2 and draft_ids.stride(1)==1 and draft_ids.shape[0]>=n
    assert 0<=step<draft_ids.shape[1]
    assert prev.dtype==torch.int64 and prev.dim()==1 and prev.is_contiguous()
    if n:
        _run_compiled(_packed_kernel_exe(),local_max.data_ptr(),local_argmax.data_ptr(),local_rows.data_ptr(),
                      markov_w1.data_ptr(),draft_ids.data_ptr(),prev.data_ptr(),packed_embed.data_ptr(),
                      n,step,draft_ids.stride(0),torch.cuda.current_stream())
