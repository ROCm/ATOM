# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + entry function of the Gumbel block partials of a DSpark Markov draft step."""

import functools

import torch

from . import kernel as _kernel
from .reference import NB, V

# Blocks per workgroup by owned rows (tiling only; the same bits): one block per workgroup keeps enough workgroups
# for <= 16 rows, two blocks halve the per-row scalar work above (graph replay).
def blocks_per_wg(n: int) -> int:
    return 1 if n <= 16 else 2


@functools.cache
def _kernel_exe(blocks_per_wg: int):
    return _kernel.compile_gumbel_partials(blocks_per_wg)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def gumbel_partials_bias(
    base: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    cache: torch.Tensor,
    step: int,
    bias: torch.Tensor,
    bias_rows: torch.Tensor,
    local_max: torch.Tensor,
    local_argmax: torch.Tensor,
    tiles: int | None = None,
    pos_offset: int = 0,
) -> None:
    """vLLM's ``_gumbel_sample_kernel`` launch of a Markov draft step (``gumbel_sample(..., is_drafting=True,
    apply_temperature=True, use_fp64=False, bias=bias, bias_rows=bias_rows, logits_cache=cache,
    logits_cache_col=step)``) into caller buffers (``reference.py``): ``base`` bf16 [n, V] (row stride free),
    ``idx_mapping`` int32 [n] (slot, < 0 invalid), ``temperature`` fp32 / ``seeds`` int64 [slots], ``pos`` int64 [n],
    ``cache`` bf16 [slots, steps, V] contiguous, ``bias`` bf16 [*, V] contiguous, ``bias_rows`` int64 [n];
    ``local_max`` fp32 [n, 196], ``local_argmax`` int64 [n, 196] (``gumbel_combine_markov_embed_packed``'s input).
    The draws use position ``pos + pos_offset`` (vLLM passes ``draw_pos = sample_pos - 1``: give ``sample_pos`` and
    ``pos_offset=-1`` to skip that subtraction launch). CUDA-graph safe; ``step`` is a launch constant (one launch per
    step), ``pos_offset`` a launch argument."""
    n = base.shape[0]
    assert base.dtype == torch.bfloat16 and base.dim() == 2 and base.shape[1] == V and base.stride(1) == 1
    assert idx_mapping.dtype == torch.int32 and idx_mapping.shape == (n,) and idx_mapping.is_contiguous()
    assert temperature.dtype == torch.float32 and temperature.is_contiguous()
    assert seeds.dtype == torch.int64 and seeds.is_contiguous() and seeds.shape == temperature.shape
    assert pos.dtype == torch.int64 and pos.shape == (n,) and pos.is_contiguous()
    assert cache.dtype == torch.bfloat16 and cache.dim() == 3 and cache.shape[2] == V and cache.is_contiguous()
    assert 0 <= step < cache.shape[1] and abs(pos_offset) < (1 << 29)
    assert bias.dtype == torch.bfloat16 and bias.dim() == 2 and bias.shape[1] == V and bias.is_contiguous()
    assert bias_rows.dtype == torch.int64 and bias_rows.shape == (n,) and bias_rows.is_contiguous()
    assert local_max.dtype == torch.float32 and local_max.shape == (n, NB) and local_max.is_contiguous()
    assert local_argmax.dtype == torch.int64 and local_argmax.shape == (n, NB) and local_argmax.is_contiguous()
    if n == 0:
        return
    _run_compiled(
        _kernel_exe(blocks_per_wg(n) if tiles is None else tiles),
        base.data_ptr(), base.stride(0), bias.data_ptr(), bias_rows.data_ptr(), idx_mapping.data_ptr(),
        temperature.data_ptr(), seeds.data_ptr(), pos.data_ptr(), ((1 << 30) + pos_offset) & 0xFFFFFFFF,
        cache.data_ptr(), cache.shape[1], step,
        local_max.data_ptr(), local_argmax.data_ptr(), n, torch.cuda.current_stream(),
    )


def warmup_gumbel_partials_bias(device: torch.device, tiles: tuple[int, ...] = (1, 2)) -> None:
    """Compile both tilings (each serves every step and temperature; ``blocks_per_wg`` picks one by row count)."""
    base = torch.zeros(1, V, dtype=torch.bfloat16, device=device)
    idx = torch.zeros(1, dtype=torch.int32, device=device)
    temp = torch.ones(1, dtype=torch.float32, device=device)
    seeds = torch.zeros(1, dtype=torch.int64, device=device)
    pos = torch.zeros(1, dtype=torch.int64, device=device)
    cache = torch.zeros(1, 7, V, dtype=torch.bfloat16, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    lm = torch.empty(1, NB, dtype=torch.float32, device=device)
    la = torch.empty(1, NB, dtype=torch.int64, device=device)
    for t in tiles:
        gumbel_partials_bias(base, idx, temp, seeds, pos, cache, 0, base, rows, lm, la, t)
    torch.cuda.synchronize(device)
