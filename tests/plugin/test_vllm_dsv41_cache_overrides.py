# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin's V4.1 cache: checkpoint restores and when the cursor read waits.

`VllmBackedPagedAttentionCache` restores checkpoints before core's stale-state
contract judges a step, and defers the cursor read as native ATOM does only
while vLLM schedules synchronously. These run on CPU against a small geometry.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.plugin.vllm import req_id_passthrough_patch as passthrough
from atom.plugin.vllm.v41_core_overrides import VllmBackedPagedAttentionCache
from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step, geometry
from tests.models.deepseek_v41.conftest import small_config  # noqa: F401


def _serving_synchronously(monkeypatch, value=True):
    monkeypatch.setattr(passthrough, "_SERVING_SYNCHRONOUSLY", value)


def _plugin_cache(geo, pages, slots):
    sizes = []

    class Sizing(PagedAttentionCache):
        def _allocate_backing(self, size, device):
            sizes.append(size)
            return torch.zeros(size, dtype=torch.uint8, device=device)

    Sizing(geo, pages, slots, "cpu")
    backing = torch.zeros(sizes[0], dtype=torch.uint8)
    return VllmBackedPagedAttentionCache(geo, pages, slots, "cpu", backing=backing)


def test_a_restore_runs_before_the_contract_even_on_a_deferred_read(
    small_config, monkeypatch  # noqa: F811
):
    """A restore has to land before the reset and the Engram staging.

    The deferred read judges a step's cursors only on the next call, after
    both, so a step that may restore restores first and then blocks, even
    when the caller asks for the deferred read. The contract then judges
    what the restore left, not what it replaced.
    """
    _serving_synchronously(monkeypatch)
    geo = replace(geometry(small_config), window_size=32)
    resumed = PagedRequest(27, 7, 0, 1, 0, (0, 1))
    # Without a restore the read stays deferred.
    plain = _plugin_cache(geo, 8, 4)
    assert plain.prepare_state(begin_step(plain, [resumed]), histories=False) is None

    cache = _plugin_cache(geo, 8, 4)
    # A slot another request held last: its cursor matches no position.
    cache.cursor[0] = torch.tensor([-1, 19, -1, 27])
    offered = []

    def restore(rows):
        offered.extend(rows)
        for slot, position, _request in rows:
            cache.cursor[slot, 0] = position
        return len(rows)

    cache.checkpoint_restore = restore
    step = begin_step(cache, [resumed])
    assert cache.prepare_state(step, histories=False) is not None
    assert offered == [(0, 7, 27)]
    cache.advance_cursor(step, np.full((1, 3), -1))
    cache.prepare_state(begin_step(cache, [replace(resumed, position=8)]))


def test_a_restore_that_leaves_the_row_stale_is_refused(
    small_config, monkeypatch  # noqa: F811
):
    """Core re-reads the cursors after the restore, so a restore that wrote
    the wrong row is named by the contract rather than answered from."""
    _serving_synchronously(monkeypatch)
    cache = _plugin_cache(replace(geometry(small_config), window_size=32), 8, 4)
    cache.cursor[0] = torch.tensor([-1, 19, -1, 27])
    cache.checkpoint_restore = lambda rows: len(rows)
    resumed = PagedRequest(27, 7, 0, 1, 0, (0, 1))
    with pytest.raises(ValueError, match="Request 27 needs state at 7"):
        cache.prepare_state(begin_step(cache, [resumed]), histories=False)


def test_the_plugin_cache_blocks_unless_serving_synchronously(
    small_config, monkeypatch  # noqa: F811
):
    """Under async scheduling, and during vLLM's warmup, the deferred read
    was not measured safe on this frontend, so it blocks there."""
    cache = _plugin_cache(replace(geometry(small_config), window_size=32), 8, 4)
    fresh = PagedRequest(27, 0, 0, 1, 0, (0, 1))
    _serving_synchronously(monkeypatch, False)
    assert cache.prepare_state(begin_step(cache, [fresh]), histories=False) is not None
    _serving_synchronously(monkeypatch, True)
    nxt = replace(fresh, position=1)
    assert cache.prepare_state(begin_step(cache, [nxt]), histories=False) is None


def test_a_deferred_read_waits_for_the_last_probe_rather_than_drop_it(
    small_config, monkeypatch  # noqa: F811
):
    """Core's judge skips rows that have not landed and the next deferred read
    ships over them; the plugin waits, so a stale row is still named."""
    _serving_synchronously(monkeypatch)
    cache = _plugin_cache(replace(geometry(small_config), window_size=32), 8, 4)
    request = PagedRequest(27, 7, 0, 1, 0, (0, 1))
    cache.cursor[0] = torch.tensor([7, 19, -1, 27])
    assert cache.prepare_state(begin_step(cache, [request]), histories=False) is None

    class _Unlanded:
        landed = False

        def query(self):
            return self.landed

        def synchronize(self):
            self.landed = True

        def record(self):
            self.landed = False

    cache._probe_done = _Unlanded()
    # What the last step shipped, once it lands: a cursor behind its position.
    cache._probe[0, 0] = 3
    with pytest.raises(ValueError, match="Request 27 needs state at 7, found 3"):
        cache.prepare_state(
            begin_step(cache, [replace(request, position=8)]), histories=False
        )


def test_the_plugin_cache_defers_a_step_with_nothing_to_restore(
    small_config,  # noqa: F811
):
    """Only a row above its floor that did not run last step can need a
    restore; every other step keeps the deferred read."""
    cache = _plugin_cache(replace(geometry(small_config), window_size=32), 8, 4)
    resumed = PagedRequest(27, 7, 0, 1, 0, (0, 1))
    assert cache._may_restore(begin_step(cache, [resumed]))
    assert not cache._may_restore(begin_step(cache, [replace(resumed, position=8)]))
    assert not cache._may_restore(
        begin_step(cache, [PagedRequest(31, 0, 0, 1, 0, (0, 1))])
    )


def test_the_plugin_cache_leaves_vllm_warmup_rows_unjudged(
    small_config,  # noqa: F811
):
    """vLLM warms SSM groups with `_warmup_N_` through slots nothing wrote."""
    cache = _plugin_cache(replace(geometry(small_config), window_size=32), 8, 4)
    cache.cursor[0] = torch.tensor([3, 19, -1, 27])
    warmup = PagedRequest("_warmup_0_", 2, 0, 1, 0, (0, 1))
    cache.prepare_state(begin_step(cache, [warmup]))
    cache.cursor[0] = torch.tensor([3, 19, -1, 27])
    with pytest.raises(ValueError, match="Request 28 needs state at 2"):
        cache.prepare_state(begin_step(cache, [replace(warmup, request_id=28)]))
