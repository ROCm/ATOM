# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fakes and builders the KV-store offload tests share (``mooncake_store``,
``umbp_store``): the block GPU connector, AITER's dtype table, the engine
config, prompts, save and load requests and their completions.

Each backend's test module keeps its own store fakes and settings; the
geometry here (blocks, chunks, slots) is the same for both.
"""

from __future__ import annotations

import array
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from conftest import atom_config_double, with_min_load

from atom.config import Config, KVCacheTensor
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadOperationId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.kv_store import keys
from atom.kv_transfer.offload.kv_store import worker as store_worker_mod
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec

BLOCK = 4  # tokens per KV block
CHUNK = 8  # tokens per store chunk: two blocks
BYTES_PER_BLOCK = 16
CHUNK_BYTES = BYTES_PER_BLOCK * CHUNK // BLOCK
SLOT = 4096  # one slot's stride for CHUNK_BYTES


class FakeGPUConnector:
    """Packs each KV block as a byte pattern, as the block GPU connector would.

    ``kv`` is the GPU KV as block id -> bytes; a block never written holds a
    pattern derived from its id. A save packs tail-to-head and publishes one
    source-safe group per chunk inside ``track_save_source``, as the real
    connector does once a group's copy has fenced.
    """

    gpu_staging_chunk_bytes = CHUNK_BYTES
    gpu_staging_buffer_bytes = 4 * CHUNK_BYTES
    # The real token range -> KV block mapping, which in-place copies use.
    block_size = BLOCK
    _ranges_to_block_ids = BlockGPUConnector._ranges_to_block_ids
    _range_block_ids = BlockGPUConnector._range_block_ids

    def __init__(self, source_safe=None) -> None:
        self.kv: dict[int, bytes] = {}
        self.calls: list[tuple] = []
        self.source_safe = source_safe
        self.fail_from = False
        self.fail_to = False
        self.events: list = []
        self.closed = False
        self._operation = None

    @contextmanager
    def track_save_source(self, operation):
        previous, self._operation = self._operation, operation
        try:
            yield
        finally:
            self._operation = previous

    @staticmethod
    def pattern(block: int) -> bytes:
        return bytes((block * 7 + i) % 251 for i in range(BYTES_PER_BLOCK))

    def batched_from_gpu(self, slots, starts, ends, *, block_ids, producer_event=None):
        self.calls.append(("from", list(starts)))
        self.events.append(producer_event)
        if self.fail_from:
            raise RuntimeError("pack failed")
        for slot, start, end in sorted(
            zip(slots, starts, ends, strict=True), key=lambda item: -item[1]
        ):
            blocks = block_ids[start // BLOCK : end // BLOCK]
            data = b"".join(self.kv.get(b, self.pattern(b)) for b in blocks)
            slot.tensor.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            if self._operation is not None and self.source_safe is not None:
                self.source_safe(SaveSourceGroupId(self._operation, ((start, end),)))

    def batched_to_gpu(self, slots, starts, ends, *, block_ids):
        self.calls.append(("to", list(starts)))
        if self.fail_to:
            raise RuntimeError("unpack failed")
        for slot, start, end in zip(slots, starts, ends, strict=True):
            data = slot.tensor.numpy().tobytes()
            for i, block in enumerate(block_ids[start // BLOCK : end // BLOCK]):
                self.kv[block] = data[i * BYTES_PER_BLOCK : (i + 1) * BYTES_PER_BLOCK]

    def close(self):
        self.closed = True


_AITER_FP8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


class _FakeAiterDtypes:
    """``aiter.dtypes`` as the arch in ``state`` resolves it."""

    def __init__(self, state) -> None:
        self._state = state

    @property
    def d_dtypes(self):
        return {"fp8": _AITER_FP8[self._state.gfx], "bf16": torch.bfloat16}


def install_fake_aiter(monkeypatch) -> SimpleNamespace:
    """AITER's dtype table and arch detection, as on gfx950 to start.

    The namespace asks AITER what ``fp8`` is stored as; CPU runners have no
    AITER. Setting the returned state's ``gfx = "gfx942"`` makes it an MI300X.
    """
    state = SimpleNamespace(gfx="gfx950")
    package = types.ModuleType("aiter")
    package.dtypes = _FakeAiterDtypes(state)
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: state.gfx
    for name, module in (
        ("aiter", package),
        ("aiter.jit", types.ModuleType("aiter.jit")),
        ("aiter.jit.utils", types.ModuleType("aiter.jit.utils")),
        ("aiter.jit.utils.chip_info", chip_info),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return state


def use_one_rank(monkeypatch) -> None:
    """The worker's TP group as one rank of one."""
    monkeypatch.setattr(
        store_worker_mod,
        "_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )


class EngineConfig(SimpleNamespace):
    """The engine `Config`'s fields (`atom_config_double`, which refuses any
    other name) with its TP placement properties, as the connectors read them
    off the engine config."""

    tp_world_size = Config.tp_world_size
    tp_size_local = Config.tp_size_local
    tp_rank_start = Config.tp_rank_start
    is_multinode_tp = Config.is_multinode_tp
    _tp_node_slice = Config._tp_node_slice


def engine_config(
    connector: str,
    extra: dict,
    role: str = "offload",
    *,
    pp: int = 1,
    tp: int = 1,
    layers: int = 4,
    **fields,
) -> EngineConfig:
    """An engine config whose transfer config selects ``connector``."""
    values = {
        "kv_transfer_config": {
            "kv_connector": connector,
            "kv_role": role,
            "kv_connector_extra_config": extra,
        },
        "kv_cache_block_size": BLOCK,
        "decode_context_parallel_size": 1,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "hf_config": SimpleNamespace(
            num_hidden_layers=layers, model_type="glm_moe_dsa"
        ),
        "model": "test-model",
        "kv_cache_dtype": "fp8",
        "speculative_config": None,
    }
    double = atom_config_double(**{**values, **fields})
    return with_min_load(EngineConfig(**vars(double)))


def seq(req_id, num_prompt_tokens, *, offset=0, cached=0, **fields):
    """A waiting request's prompt as the scheduler sees it."""
    request = SimpleNamespace(
        id=req_id,
        token_ids=array.array("i", range(offset, offset + num_prompt_tokens)),
        num_prompt_tokens=num_prompt_tokens,
        num_cached_tokens=cached,
        block_table=list(range(-(-num_prompt_tokens // BLOCK))),
        offload_loaded=False,
        offload_load_failed=False,
    )
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def chunk_hashes(num_tokens, offset=0):
    return keys.chunk_hash_chain(list(range(offset, offset + num_tokens)), CHUNK)


def chunk_key(namespace, hashes, index, *, rank=0, world=1):
    [key] = keys.chunk_keys(namespace, rank, world, hashes, index, index + 1)
    return key


def save_req(req_id, num_tokens, *, skip=0, generation=0, offset=0, block_ids=None):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=(
            list(range(num_tokens // BLOCK)) if block_ids is None else block_ids
        ),
        save_spec=SaveSpec(skip_leading_tokens=skip),
        save_operation=SaveOperationId(req_id, generation),
        chunk_hashes=chunk_hashes(num_tokens, offset),
    )


def load_req(req_id, *, hbm, lmc, block_ids, end=None, generation=0, num_tokens=None):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=[],
        block_ids=block_ids,
        load_spec=LoadSpec(
            hbm_cached_tokens=hbm,
            lmcache_cached_tokens=lmc,
            can_load=True,
            transfer_end_tokens=end,
        ),
        load_operation=LoadOperationId(req_id, generation),
        chunk_hashes=chunk_hashes(num_tokens or (end or lmc)),
    )


def stored(operation, succeeded=True):
    return ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, succeeded)


def quiescent(operation):
    return ConnectorCompletion(DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL, operation, True)


def source_safe(operation, index):
    return ConnectorCompletion(
        DENSE_PAGE_SOURCE_SAFE_CHANNEL,
        SaveSourceGroupId(operation, ((index * CHUNK, (index + 1) * CHUNK),)),
        True,
    )


def kv_caches(num_blocks=8):
    # MLA-like: token-major rows, one plane per layer.
    return {
        f"layer_{i}": KVCacheTensor(
            layer_num=i,
            k_cache=torch.zeros((num_blocks * BLOCK, 1, 8), dtype=torch.uint8),
            index_cache=None,
        )
        for i in range(2)
    }


def all_slots_free(pool, region):
    """Whether every slot of ``region`` is back in the pool (none leased)."""
    return len(pool._free[pool._region(region)]) == pool.capacity(region)
