# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two hooks the vLLM connector needs from ATOM's MP worker."""

from types import SimpleNamespace

from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
)
from atom.plugin.vllm.kv_transfer.mp_worker_adapter import MPOffloadWorkerAdapter


def test_a_failed_load_reports_only_the_blocks_it_writes():
    """The HBM prefix below the load may be shared with a running request."""
    inner = SimpleNamespace(block_size=4, start_load_kv=lambda metadata: None)
    adapter = MPOffloadWorkerAdapter(inner)
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=7,
            token_ids=list(range(16)),
            block_ids=[100, 101, 102, 103],
            load_spec=LoadSpec(8, 14, can_load=True),
        )
    )
    adapter.start_load_kv(metadata)
    assert adapter._load_blocks == {"7": {102, 103}}


def test_the_preemption_fence_is_the_workers_own():
    """vLLM's fence waits on the worker's, which knows its drain."""
    fenced = []
    adapter = MPOffloadWorkerAdapter(SimpleNamespace(fence=fenced.append))
    adapter.wait_for_requests(["7", "8"])
    assert fenced == [["7", "8"]]
