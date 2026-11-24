# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Support check, workspace and entry of the one-launch top-p mask."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .reference import SPLIT_MAX_BATCH, V, splits_for


@functools.cache
def _num_cu(index: int) -> int:
    """vLLM's ``num_compute_units``: the slice count follows it."""
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.cache
def _kernel_exe(slices: int):
    return _kernel.compile_top_p_mask(slices)


@functools.cache
def _workspace(device: torch.device) -> torch.Tensor:
    """Partials + row counters, one set per device (as vLLM's ``_TRITON_SPLIT_CACHE``: launches on one stream);
    zeroed once, every launch leaves its counters at zero."""
    words = _kernel.MAX_ROWS * (_kernel.ROW_WORDS + _kernel.CNT_WORDS) + _kernel.DBG_WORDS
    return torch.zeros(words, dtype=torch.int32, device=device)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def top_p_blocker(vocab_size: int, mask_value: float) -> str | None:
    """Why ``top_p_mask`` cannot stand in for a model's split top-p chain (None: it can)."""
    from . import available

    if not available():
        return "needs gfx950 with FlyDSL"
    if vocab_size != V:
        return f"vocab {vocab_size} (the kernel is built for {V})"
    if mask_value != float("-inf"):
        return f"mask value {mask_value} (the kernel writes -inf)"
    return None


def _top_p_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    n = logits.shape[0]
    assert logits.dtype == torch.float32 and logits.dim() == 2 and logits.shape[1] == V and logits.stride(1) == 1
    assert 1 <= n <= SPLIT_MAX_BATCH, n
    assert p.dtype == torch.float32 and p.shape == (n,) and p.is_contiguous()
    s = splits_for(n, V, _num_cu(logits.device.index))
    _run_compiled(_kernel_exe(s), logits.data_ptr(), logits.stride(0), p.data_ptr(),
                  _workspace(logits.device).data_ptr(), n, 8 * s * ((n + 7) // 8), torch.cuda.current_stream())
    return logits


def _mode(logits, p):
    """One mode: the variant (slice count) follows the row count (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    """The first row count of each slice count."""
    cu = _num_cu(torch.cuda.current_device())
    first = {}
    for n in range(1, SPLIT_MAX_BATCH + 1):
        first.setdefault(splits_for(n, V, cu), n)
    return tuple(first.values())


def _inputs(mode, n, logits, p):
    """Zero rows at p 0.95 (compile-only: nothing launches)."""
    x = torch.zeros(n, V, dtype=torch.float32, device=logits.device)
    return (x, torch.full((n,), 0.95, dtype=torch.float32, device=logits.device)), {}


top_p_mask = entry("sampling/top_p_mask_f32_v200064", mode=_mode, sizes=_sizes, inputs=_inputs)(_top_p_mask)
top_p_mask.__doc__ = """vLLM's ``apply_top_k_top_p_triton(logits, None, p)`` at <= 64 rows (its split top-p pipeline:
stats, five pivot rounds, mask) in one launch: masks fp32 ``logits`` [n, 200064] (unit column stride, any row stride)
in place with -inf, bit for bit (``reference.py``); ``p`` fp32 [n]; rows with p >= 1 stay as they are. One workspace
per device: launches on one stream (or graphs replayed one after another). CUDA-graph safe. Returns ``logits``."""


def warmup_top_p_mask(device: torch.device) -> None:
    """Compile every slice count (S 32 at <= 8 rows, S 16 above on 256 CUs) and allocate the workspace, before graph
    capture; nothing compiles after it."""
    for n in _sizes(()):
        x = torch.zeros(n, V, dtype=torch.float32, device=device)
        top_p_mask(x, torch.full((n,), 0.95, dtype=torch.float32, device=device))
    torch.cuda.synchronize(device)
