# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Code-object hashes of kernels the step tail must leave alone, with
``ATOM_KIMI_MONO_STEP_TAIL`` off: the Kimi-K3 mono kernels' source digest (the
build key every one of them carries), a sample of them compiled here, and the
DSpark Triton kernels after one small run. Run it in the tree and in a copy of
the base commit (it imports ``atom`` from the tree it sits in) and compare.

    python tools/steptail_hashes.py
"""

import dataclasses
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch


def sha(data) -> str:
    data = data.encode() if isinstance(data, str) else bytes(data)
    return hashlib.sha256(data).hexdigest()[:16]


def mono_kernels():
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
    from atom.models.kimi_k3.mono.sources import SOURCES
    from atom.mono.runtime.compile import _code_object, compile_kernels

    print("mono sources digest", SOURCES)
    for rows in (8, 24, 64):
        built = ar.build_allreduce(8, rows)
        parts = (
            [getattr(built, f.name) for f in dataclasses.fields(built)]
            if dataclasses.is_dataclass(built)
            else list(built)
        )
        name, launcher = parts[0], parts[1]
        compile_kernels([(launcher, ar.ALLREDUCE_ABI)])
        for artifact in launcher._mem_cache.values():
            print("flydsl", name, sha(_code_object(artifact._ir_text)))


def dspark_kernels():
    from aiter import topk_select

    from atom.model_ops import dspark_markov_argmax as markov
    from atom.model_ops import rejection_sampler as rej

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(3)
    rows, vocab, rank, qlen = 8, 4096, 64, 8

    def bf16(*shape):
        return torch.randn(shape, generator=gen, device=dev).to(torch.bfloat16)

    base, w1, w2 = bf16(rows, vocab), bf16(vocab, rank), bf16(vocab, rank)
    prev = torch.randint(0, vocab, (rows,), device=dev, generator=gen).to(torch.int32)
    out = torch.empty(rows, dtype=torch.int32, device=dev)
    markov.dspark_markov_argmax(base, prev, w1, w2, out)
    ids = torch.randint(0, vocab, (rows,), device=dev, generator=gen).to(torch.int32)
    t_idx = torch.arange(qlen - 1, device=dev)
    b_idx = torch.tensor([qlen - 1], device=dev)
    cu = torch.tensor([qlen - 1], dtype=torch.int32, device=dev)
    bonus = topk_select(torch.index_select(base, 0, b_idx), 1, tie="low")[1].view(-1)
    rej.rejection_sample(
        ids, t_idx, qlen - 1, cu, None, torch.index_select(base, 0, t_idx), bonus, b_idx
    )
    torch.cuda.synchronize()
    for fn in (
        markov._markov_stage1,
        markov._dspark_markov_argmax_stage2,
        rej.rejection_greedy_sample_kernel,
    ):
        name = getattr(fn, "__name__", None) or getattr(
            getattr(fn, "fn", None), "__name__", "?"
        )
        found = 0
        for entry in getattr(fn, "device_caches", {}).values():
            cache = entry[0] if isinstance(entry, tuple) else entry
            for compiled in cache.values():
                asm = getattr(compiled, "asm", {})
                for kind in ("amdgcn", "hsaco"):
                    if kind in asm:
                        print("triton", name, kind, sha(asm[kind]))
                        found += 1
        if not found:
            print("triton", name, "no compiled kernel reachable")


def main():
    print("switch", os.environ.get("ATOM_KIMI_MONO_STEP_TAIL", "unset"))
    mono_kernels()
    dspark_kernels()


if __name__ == "__main__":
    main()
