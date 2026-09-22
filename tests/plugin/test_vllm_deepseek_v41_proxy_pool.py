# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Proxy-pool arithmetic for DeepSeek-V4.1 on the vLLM plugin path.

The bridge buys V4.1's pool through a fake vLLM attention layer, so three
numbers have to agree that nothing in either project checks for us: the
``head_size`` ATOM declares, the bytes vLLM's own ``FullAttentionSpec``
charges per block, and the bytes ``V41PoolGeometry`` will address. These tests
pin that agreement -- and the batch snapshot the metadata builder hands the
step -- against a stand-in geometry, so they run without a GPU or a
checkpoint.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.plugin.vllm import deepseek_v41_bridge as bridge
from atom.plugin.vllm.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_BLOCK_SIZE,
    AtomDeepseekV41ProxyAttention,
    _v41_dummy_batch,
    _v41_scheduled_batch,
    is_deepseek_v41_vllm_config,
    snapshot_v41_batch,
    v41_kv_cache_dtype,
    v41_proxy_head_size,
    v41_proxy_page_size_bytes,
    v41_proxy_state_reserve_blocks,
)
from atom.plugin.vllm.state_slot_allocator import StateSlotAllocator

# Shaped like the shipped Flash geometry: `paged_bytes` a multiple of 512 (so
# the proxy `head_size` is exact rather than rounded up) and a STATE entry
# several PAGEs wide, which is the whole reason for the tail reserve.
PAGED_BYTES = 41_984
STATE_BYTES = 213_760


class _FakeGeometry:
    """The two currencies `v41_proxy_*` reads, and nothing else."""

    def __init__(self, paged_bytes=PAGED_BYTES, state_bytes=STATE_BYTES):
        self.paged_bytes = paged_bytes
        self.state_bytes = state_bytes

    def paged_extents(self, pages):
        return (0, pages * self.paged_bytes)


def _vllm_config(
    *,
    architectures=("DeepseekV41ForCausalLM",),
    model_type="deepseek_v41",
    cache_dtype="auto",
    max_num_seqs=256,
    num_gpu_blocks=0,
    max_model_len=8192,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/DeepSeek-V4.1-Flash",
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(
                architectures=list(architectures),
                model_type=model_type,
            ),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=8192,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            num_gpu_blocks=num_gpu_blocks,
        ),
    )


@pytest.fixture
def fake_geometry(monkeypatch):
    geometry = _FakeGeometry()
    monkeypatch.setattr(bridge, "v41_proxy_geometry", lambda _config: geometry)
    return geometry


class TestConfigDetection:

    def test_architecture_and_model_type_both_identify_v41(self):
        assert is_deepseek_v41_vllm_config(_vllm_config())
        assert is_deepseek_v41_vllm_config(
            _vllm_config(architectures=(), model_type="deepseek_v41")
        )

    def test_v4_is_not_mistaken_for_v41(self):
        # `DeepseekV41ForCausalLM` starts with `DeepseekV4`, so the two are
        # told apart by exact match in both directions.
        assert not is_deepseek_v41_vllm_config(
            _vllm_config(
                architectures=("DeepseekV4ForCausalLM",), model_type="deepseek_v4"
            )
        )


class TestKVCacheDtype:

    @pytest.mark.parametrize("spelling", ["auto", "bfloat16", "float16"])
    def test_unquantized_spellings_select_the_bf16_pool(self, spelling):
        assert v41_kv_cache_dtype(_vllm_config(cache_dtype=spelling)) == "bf16"

    def test_nvfp4_selects_the_packed_pool(self):
        assert v41_kv_cache_dtype(_vllm_config(cache_dtype="nvfp4")) == "fp4"

    @pytest.mark.parametrize("spelling", ["fp8", "fp8_e4m3", "fp8_e5m2"])
    def test_fp8_is_refused_rather_than_downgraded(self, spelling):
        # V4.1 has no fp8 pool. Silently serving bf16 instead would size the
        # proxy layer for one geometry and address another.
        with pytest.raises(ValueError, match="kv-cache-dtype"):
            v41_kv_cache_dtype(_vllm_config(cache_dtype=spelling))


class TestProxyBlockSizing:

    def test_head_size_makes_one_proxy_block_hold_one_page(self, fake_geometry):
        config = _vllm_config()
        page_size = v41_proxy_page_size_bytes(config)
        assert page_size >= fake_geometry.paged_bytes
        # The ceiling exists only so an unforeseen geometry over-allocates; a
        # page that is a multiple of 512 has to land exactly.
        assert page_size - fake_geometry.paged_bytes < 2 * ATOM_DEEPSEEK_V41_BLOCK_SIZE
        assert page_size == 2 * ATOM_DEEPSEEK_V41_BLOCK_SIZE * v41_proxy_head_size(
            config
        )

    def test_vllm_charges_per_block_exactly_what_atom_declared(self, fake_geometry):
        # The number that matters is vLLM's, not ours: `page_size_bytes` is
        # what sizes the tensor and what the worker divides by to recover a
        # block count. Ask the spec rather than restating its formula.
        config = _vllm_config()
        spec = AtomDeepseekV41ProxyAttention().get_kv_cache_spec(config)
        assert spec.block_size == ATOM_DEEPSEEK_V41_BLOCK_SIZE
        assert spec.dtype == torch.uint8
        assert spec.num_kv_heads == 1
        assert spec.page_size_bytes == v41_proxy_page_size_bytes(config)

    @pytest.mark.parametrize("max_num_seqs", [1, 17, 256, 512])
    def test_state_reserve_is_the_smallest_tail_that_fits(
        self, fake_geometry, max_num_seqs
    ):
        config = _vllm_config(max_num_seqs=max_num_seqs)
        reserve = v41_proxy_state_reserve_blocks(config)
        page_size = v41_proxy_page_size_bytes(config)
        needed = max_num_seqs * fake_geometry.state_bytes
        assert reserve * page_size >= needed
        assert (reserve - 1) * page_size < needed

    def test_state_reserve_is_zero_for_every_other_model(self, monkeypatch):
        # The reserve patch is installed process-wide, so this zero is what
        # keeps it a no-op for the rest of the model zoo.
        def _no_geometry(_config):
            raise AssertionError("geometry must not be built for a non-V4.1 model")

        monkeypatch.setattr(bridge, "v41_proxy_geometry", _no_geometry)
        assert (
            v41_proxy_state_reserve_blocks(
                _vllm_config(
                    architectures=("Qwen3MoeForCausalLM",), model_type="qwen3_moe"
                )
            )
            == 0
        )


def _common_attn_metadata(query_lens, num_computed, blocks_per_row=8):
    """vLLM's device-side batch description, without the pass-through patch.

    Exercising the fallback path on purpose: it is the one a unit test can
    reach, and it is what runs when `req_id_passthrough_patch` is absent.
    """
    query_lens = np.asarray(query_lens, dtype=np.int64)
    num_computed = np.asarray(num_computed, dtype=np.int64)
    num_reqs = len(query_lens)
    qsl = np.zeros(num_reqs + 1, dtype=np.int32)
    qsl[1:] = np.cumsum(query_lens)
    block_table = np.arange(
        100, 100 + num_reqs * blocks_per_row, dtype=np.int32
    ).reshape(num_reqs, blocks_per_row)
    return SimpleNamespace(
        num_reqs=num_reqs,
        query_start_loc_cpu=torch.from_numpy(qsl),
        block_table_tensor=torch.from_numpy(block_table),
        seq_lens=torch.from_numpy((num_computed + query_lens).astype(np.int32)),
    )


class TestBatchSnapshot:

    def test_snapshot_reconstructs_lengths_and_page_rows(self):
        query_lens = [600, 1, 257]
        num_computed = [0, 1024, 255]
        snapshot = snapshot_v41_batch(_common_attn_metadata(query_lens, num_computed))

        assert snapshot.num_reqs == 3
        assert list(snapshot.query_lens) == query_lens
        assert list(snapshot.num_computed) == num_computed
        # `context_lens` in ATOM's protocol is the end position *after* this
        # step, which is vLLM's `seq_lens`.
        assert list(snapshot.ends) == [600, 1025, 512]
        assert snapshot.total_tokens == sum(query_lens)
        # One PAGE per 256 tokens of history, rounded up: a row short by one
        # page would drop the tokens this step is about to write.
        assert [len(row) for row in snapshot.block_rows] == [3, 5, 2]
        assert snapshot.block_rows[0] == (100, 101, 102)

    def test_snapshot_keys_slots_on_a_lifetime_stable_id(self):
        # Without the pass-through patch there is no `req_id`, so the slot key
        # falls back to the request's first block -- which vLLM holds for the
        # request's lifetime, which is all the allocator needs.
        snapshot = snapshot_v41_batch(_common_attn_metadata([4, 4], [0, 0]))
        assert snapshot.req_ids == [row[0] for row in snapshot.block_rows]

    def test_scheduled_batch_fields_line_up_row_for_row(self):
        snapshot = snapshot_v41_batch(
            _common_attn_metadata([600, 1, 257], [0, 1024, 255])
        )
        batch = _v41_scheduled_batch(snapshot, StateSlotAllocator(8))

        assert batch.is_dummy_run is False
        assert batch.total_seqs_num == snapshot.num_reqs
        assert batch.total_tokens_num == snapshot.total_tokens
        assert list(batch.num_scheduled_tokens) == list(snapshot.query_lens)
        assert list(batch.context_lens) == list(snapshot.ends)
        assert batch.block_tables == snapshot.block_rows
        # Every zipped per-request array is indexed by the same `i`, including
        # the committed slots.
        assert len(batch.state_slots_committed) == snapshot.num_reqs
        assert len(set(batch.state_slots_committed)) == snapshot.num_reqs

    def test_a_resumed_request_keeps_its_state_slot(self):
        allocator = StateSlotAllocator(8)
        first = _v41_scheduled_batch(
            snapshot_v41_batch(_common_attn_metadata([600], [0])), allocator
        )
        second = _v41_scheduled_batch(
            snapshot_v41_batch(_common_attn_metadata([1], [600])), allocator
        )
        assert list(second.state_slots_committed) == list(first.state_slots_committed)


class TestDummyBatch:
    """The profiling forward is wider than a servable request.

    vLLM profiles at ``max_num_batched_tokens``, which defaults above
    ``max_model_len``; a ``block_tables`` row is only ``max_model_len`` worth
    of PAGEs wide. A dummy batch that packs the whole forward into one request
    therefore overflows the row before the model ever runs.
    """

    # One `max_model_len` of 8192 at the shipped 256-token PAGE.
    MAX_REQ_TOKENS = 32 * ATOM_DEEPSEEK_V41_BLOCK_SIZE

    def _split(self, running_tokens, max_reqs=64):
        return _v41_dummy_batch(
            running_tokens,
            max_req_tokens=self.MAX_REQ_TOKENS,
            max_reqs=max_reqs,
        )

    @pytest.mark.parametrize(
        "running_tokens", [1, 100, 8192, 8193, 16384, 16385, 40_000]
    )
    def test_no_request_owns_more_pages_than_a_block_table_row(self, running_tokens):
        batch = self._split(running_tokens)
        assert batch.num_scheduled_tokens
        assert max(batch.num_scheduled_tokens) <= self.MAX_REQ_TOKENS

    @pytest.mark.parametrize("running_tokens", [1, 100, 8192, 8193, 16384, 16385])
    def test_the_split_covers_every_token_of_the_forward(self, running_tokens):
        batch = self._split(running_tokens)
        assert batch.total_tokens_num == running_tokens
        assert sum(batch.num_scheduled_tokens) == running_tokens

    def test_a_forward_that_fits_stays_one_request(self):
        batch = self._split(4096)
        assert batch.total_seqs_num == 1
        assert batch.num_scheduled_tokens == (4096,)

    def test_the_default_profile_shape_splits_rather_than_overflows(self):
        # max_num_batched_tokens=16384 against max_model_len=8192: two rows of
        # 32 PAGEs each, not one row of 64.
        batch = self._split(16384)
        assert batch.num_scheduled_tokens == (8192, 8192)

    def test_rows_are_capped_at_max_num_seqs_and_the_rest_is_padding(self):
        # Beyond `max_reqs` full-length requests there is no row left to put a
        # token on, so the step pads the tail instead of widening a row.
        batch = self._split(40_000, max_reqs=2)
        assert batch.total_seqs_num == 2
        assert batch.num_scheduled_tokens == (self.MAX_REQ_TOKENS,) * 2
        assert batch.total_tokens_num == sum(batch.num_scheduled_tokens) < 40_000

    def test_every_row_carries_a_token(self):
        # `_prepare` skips zero-token rows, so an empty row would silently
        # shrink the batch it published a width for.
        for running_tokens in (1, 3, 17, 16385):
            batch = self._split(running_tokens)
            assert min(batch.num_scheduled_tokens) >= 1

    def test_the_dummy_batch_declares_itself(self):
        batch = self._split(16384)
        # `_prepare` routes on this: private scratch cache, fabricated PAGE
        # ids, no live STATE slot touched.
        assert batch.is_dummy_run is True
        assert batch.state_slots_committed == ()
        assert batch.block_tables == ()
        assert batch.total_seqs_num == len(batch.req_ids)
