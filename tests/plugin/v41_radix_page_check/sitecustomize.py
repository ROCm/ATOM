# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU validation tool for the DeepSeek-V4.1 radix replay on the SGLang plugin.

Not product code and not a unit test: it checks, on a running server, the two
properties the replay below a prefix hit must keep (`deepseek_v41_radix`):

* the replay writes no shared PAGE: every byte a hit's pages own (main rows,
  each owner's data and scale source, index planes) has the same digest before
  and after the forward that carries the replay;
* every TP rank plans the same replay rows for every EXTEND step.

Use it by putting this directory first on ``PYTHONPATH`` of the server, e.g.::

    PYTHONPATH=/path/to/atom/tests/plugin/v41_radix_page_check:$PYTHONPATH \\
        python3 -m sglang.launch_server ...

Python imports ``sitecustomize`` in every process (the schedulers included);
this one only registers an import hook, which wraps the radix seam when
``atom.plugin.sglang.deepseek_v41_radix`` is first imported. Each fused step
logs one ``V4.1 radix page check`` line (ERROR when a page changed or the
ranks disagree). It costs a device sync and a pass over the hit pages per hit
step, and a CPU all-reduce per EXTEND step.
"""

import importlib.abc
import importlib.machinery
import logging
import sys

logger = logging.getLogger("atom")

TARGET = "atom.plugin.sglang.deepseek_v41_radix"
# Digests taken before a fused step's forward, keyed by its record.
_PENDING = {}


def page_digests(cache, pages, chunk: int = 2):
    """``[len(pages), 2]`` int64 digests of every byte a PAGE owns: its main
    rows (``page_regions``: one region for a BF16 pool, each owner's data and
    scale source for a quantized one) and each owner's index-plane rows.
    Equal before and after a forward iff (up to a checksum collision) that
    forward wrote none of them."""
    import torch

    num_pages = int(cache.num_pages)
    regions = [view.reshape(num_pages, -1) for _, view in cache.page_regions] + [
        plane.reshape(num_pages, -1)
        for planes in cache.index_planes.values()
        for plane in planes
    ]
    device = regions[0].device
    pages = pages.to(device=device, dtype=torch.int64)
    out = torch.zeros(int(pages.numel()), 2, dtype=torch.int64, device=device)
    for region in regions:
        width = int(region.shape[1])
        weights = torch.arange(width, device=device, dtype=torch.int64) % 65521 + 1
        for begin in range(0, int(pages.numel()), chunk):
            rows = region[pages[begin : begin + chunk]].to(torch.int64)
            out[begin : begin + chunk, 0] += rows.sum(dim=1)
            out[begin : begin + chunk, 1] += (rows * weights).sum(dim=1)
    return out


def hit_pages(req_to_token, fused, page_size: int):
    """The pages below every hit of a fused record (``[0, P)`` of each)."""
    import torch

    pages = [
        req_to_token[int(slot), : int(end) : page_size].long() // page_size
        for slot, end in zip(fused.slots, fused.ends)
    ]
    return torch.unique(torch.cat(pages))


def tp_agreement(count: int) -> str:
    """Whether every TP rank planned ``count`` replay rows for this step."""
    import torch
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if group.world_size <= 1:
        return "single"
    values = torch.tensor([count, -count], dtype=torch.int64)
    torch.distributed.all_reduce(
        values, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
    )
    high, low = int(values[0]), -int(values[1])
    return "yes" if high == low == count else f"NO(min={low},max={high})"


def install(radix) -> None:
    """Wrap ``radix.run_v41_prefix_replay`` (digests before the forward, the
    TP check on every EXTEND step) and ``radix.run_with_fused_replay`` (the
    comparison after it)."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    plan, run = radix.run_v41_prefix_replay, radix.run_with_fused_replay
    if getattr(plan, "_page_check", False):
        return

    def checked_plan(worker, batch):
        fused = plan(worker, batch)
        if radix.radix_replay_rows() is None:
            return fused
        if not batch.forward_mode.is_extend_without_speculative():
            return fused
        agreement = tp_agreement(0 if fused is None else fused.num_rows)
        if agreement not in ("yes", "single"):
            logger.error("V4.1 radix page check: TP ranks disagree: %s", agreement)
        if fused is not None:
            builder = bridge.get_deepseek_v41_builder()
            table = worker.model_runner.req_to_token_pool.req_to_token
            pages = hit_pages(table, fused, int(builder.block_size))
            before = page_digests(builder.cache, pages)
            _PENDING[id(fused)] = (builder, pages, before, agreement)
        return fused

    def checked_run(original, worker, batch, fused, args, kwargs):
        try:
            return run(original, worker, batch, fused, args, kwargs)
        finally:
            builder, pages, before, agreement = _PENDING.pop(id(fused))
            after = page_digests(builder.cache, pages)
            changed = int((after != before).any(dim=1).sum())
            log = logger.error if changed else logger.info
            log(
                "V4.1 radix page check: hits=%d rows=%d pages=%d "
                "pages_changed=%d tp_agree=%s",
                int(fused.rows.size),
                fused.num_rows,
                int(pages.numel()),
                changed,
                agreement,
            )

    checked_plan._page_check = True
    radix.run_v41_prefix_replay = checked_plan
    radix.run_with_fused_replay = checked_run


class _Hook(importlib.abc.MetaPathFinder):
    """Install the check right after the radix module is executed."""

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


if not any(isinstance(finder, _Hook) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
