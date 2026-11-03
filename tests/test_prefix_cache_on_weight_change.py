# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The prefix-cache index goes wherever the KV pool or the weights change."""

import queue

import pytest

from atom.model_engine.collective_rpc import COLLECTIVE_RPC_CMD
from atom.model_engine.engine_utility import EngineUtilityHandler


class _BlockManager:
    def __init__(self):
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1


class _Engine:
    _has_pending_utility = True
    _is_rl_weights_offloaded = False
    _weight_sync_fenced = False
    _rl_weights_inconsistent = False
    _generation_paused = False


def _cleared_after(cmd, args, reply):
    """How often the index is dropped when *cmd* answers *reply*."""
    block_manager = _BlockManager()
    handler = EngineUtilityHandler(
        None, queue.Queue(), label="test", scheduler=type("S", (), {})()
    )
    handler.scheduler.block_manager = block_manager
    handler._execute_utility_command = lambda cmd, args: reply
    handler._settle_weight_stream = lambda reply, engine: None
    pending = queue.Queue()
    pending.put_nowait((cmd, args))
    handler.process_queue(pending, _Engine())
    return block_manager.cleared


@pytest.mark.parametrize(
    "cmd, args, reply",
    [
        ("clear_kv_cache", {}, {}),
        ("release_memory", {"tags": ["weights", "kv_cache"]}, {}),
        ("resume_memory", {}, {}),
        ("update_weights", {"sync_id": None}, {}),
        ("update_weights_shm", {"sync_id": None, "is_last": True}, {}),
        ("update_weights_ipc", {"sync_id": None}, {"error": "boom"}),
        (COLLECTIVE_RPC_CMD, {}, {"method": "receive_weights_rdma"}),
    ],
)
def test_the_index_goes_when_the_pool_or_the_weights_change(cmd, args, reply):
    assert _cleared_after(cmd, args, reply) == 1


@pytest.mark.parametrize(
    "cmd, args, reply",
    [
        ("update_weights_shm", {"sync_id": None, "is_last": False}, {}),
        ("release_memory", {"tags": ["weights"]}, {}),
        (COLLECTIVE_RPC_CMD, {}, {"method": "get_worker_capabilities"}),
        ("start_profile", {}, {}),
    ],
)
def test_the_index_stays_when_nothing_changes(cmd, args, reply):
    assert _cleared_after(cmd, args, reply) == 0
