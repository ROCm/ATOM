# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The layer kernel's hand-off contract, checked on a build.

Tracing a build (compile-only, no launch, no GPU needed with ``ARCH`` set) records
its mailbox accesses (``atom.mono.plan.trace``); ``check_traced`` holds them to
``layout.mailbox_regions``. A build served from FlyDSL's disk cache is not traced,
so the caller disables the cache (``FLYDSL_RUNTIME_ENABLE_CACHE=0``).

Usage: python -m atom.models.qwen3_5.mono.contract [S ...]
"""

import inspect
import sys

from atom.models.qwen3_5.mono.config import Q35Config
from atom.models.qwen3_5.mono.kernels.layer import build_qwen35_kernel
from atom.models.qwen3_5.mono.layout import mailbox_regions, shape_dims
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import check_traced

TP = 4


def build_layer(
    S: int,
    full: bool,
    weights: str = "fp8",
    T: int = 1,
    npes: int = TP,
    cap: int = 0,
    pa_flydsl: bool = False,
):
    """The layer kernel for ``S`` rows (an MTP verify of ``T`` tokens a request; ``cap`` > 0 its GDN on
    ReplaySSM records; ``pa_flydsl``: decode attention as ATOM_PA_FLYDSL's), full attention or GDN, and
    its launch ABI."""
    cfg = Q35Config(tp=TP, rank=0, weights=weights, pa_flydsl=pa_flydsl)
    fn = build_qwen35_kernel(
        S,
        npes,
        full,
        eps=cfg.eps,
        rope_dim=cfg.rope_dim,
        tokens_per_seq=T,
        replay_cap=cap,
        **shape_dims(cfg),
    )
    params = tuple(inspect.signature(getattr(fn, "func", None) or fn._func).parameters)
    return fn, KernelAbi(params[:-1])


def check_build(S: int, full: bool, weights: str = "fp8", T: int = 1, npes: int = TP):
    """Trace ``build_layer`` and check its hand-offs; raises ``ContractError``."""
    fn, abi = build_layer(S, full, weights, T, npes)
    check_traced(fn, abi, mailbox_regions(S, full, weights, npes))


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or list(range(1, 17))
    for s in sizes:
        for full in (False, True):
            for weights in ("fp8", "mxfp4"):
                check_build(s, full, weights)
                print(f"S={s} full={full} {weights}: contract holds")
