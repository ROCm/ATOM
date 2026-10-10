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
        # Two halves, not one: the scheduler's slots, and the same number
        # again for the rows vLLM stages during warmup and capture. Those rows
        # never finish, so lending them slots out of the scheduler's share
        # lends them permanently, and the first full batch of real requests
        # then evicted a live one. Asserted against
        # `v41_num_state_slots` rather than `2 * max_num_seqs` so the two
        # cannot drift, and the second half is asserted separately below --
        # sizing the tail for one half is exactly the regression this guards.
        slots = bridge.v41_num_state_slots(config)
        assert slots == 2 * bridge.v41_scheduler_state_slots(config)
        needed = slots * fake_geometry.state_bytes
        assert reserve * page_size >= needed
        assert (reserve - 1) * page_size < needed

    @pytest.mark.parametrize("num_reqs", [1, 8, 64])
    def test_capture_slots_cannot_collide_with_a_request(self, fake_geometry, num_reqs):
        """The capture half is disjoint from everything the allocator owns.

        Both halves of the refusal this replaces: a synthetic batch whose rows
        shared one slot (`begin_step`: "Each request needs its own valid STATE
        slot"), and one that took slots the allocator had already given out
        (a decode at position 912 finding a cursor at 1).
        """
        config = _vllm_config(max_num_seqs=64)
        slots = bridge.v41_capture_state_slots(num_reqs, config)
        assert len(set(slots.tolist())) == num_reqs
        assert min(slots.tolist()) >= bridge.v41_scheduler_state_slots(config)
        assert max(slots.tolist()) < bridge.v41_num_state_slots(config)

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


def test_the_v41_proxy_layer_is_recognised_for_non_immediate_block_reuse():
    """V4.1 has V4's global-arena property and must get V4's reuse patch.

    The markers are matched as substrings, and `".atom_deepseek_v4_proxy"` is
    NOT a substring of `"...atom_deepseek_v41_proxy"` -- so V4.1 silently went
    without it. Its PAGE and STATE share one address space (a slot's ring is
    an offset past the absolute end of the paged region), which is exactly the
    layout the patch exists to stop vLLM from recycling out from under.
    """
    from atom.plugin.vllm.deepseek_v4_prefix_patch import _V4_PROXY_LAYER_MARKERS
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )

    assert any(
        marker in ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME
        for marker in _V4_PROXY_LAYER_MARKERS
    )


def test_the_pool_is_sized_from_the_speculation_that_is_admitted():
    """The pool width and the gate have to agree, in both directions.

    The slack a verify step needs is real pool bytes -- `ring_slots` and
    `compress_ring_slots` both carry `speculative_tokens` -- so sizing from a
    hardcoded zero was only safe while every speculative config was refused.
    The predecessor of this test said exactly that and is what caught the
    refusal being lifted without the width following it.

    Now both move: the geometry reads vLLM's `num_speculative_tokens`, and the
    gate admits DSpark alone. Asserted on the pair rather than on either half,
    because a gate that admits a method the pool is not sized for is the
    failure neither one shows on its own.
    """
    config = _vllm_config()
    # The gate reads `model_config.architectures`; the fixture only carries
    # them on `hf_config`, which is where everything else in this file looks.
    config.model_config.architectures = list(
        config.model_config.hf_config.architectures
    )
    assert bridge.v41_speculative_tokens(config) == 0

    spec = SimpleNamespace(num_speculative_tokens=5, method="dspark")
    config.speculative_config = spec
    assert bridge.v41_speculative_tokens(config) == 5

    config.speculative_config = SimpleNamespace(num_speculative_tokens=3, method="mtp")
    with pytest.raises(ValueError, match="DSpark speculation only"):
        atom_platform_module().enforce_deepseek_v41_constraints(config)

    config.speculative_config = spec
    # Admitted, and nothing else about the config refused along the way.
    atom_platform_module().enforce_deepseek_v41_constraints(config)


def atom_platform_module():
    import importlib

    return importlib.import_module("atom.plugin.vllm.platform")


def test_the_proxy_backend_answers_block_sizes_with_or_without_a_spec():
    """vLLM 0.31 passes `kv_cache_spec` positionally; 0.28 passes nothing.

    Both call sites in 0.31 (`attention.py`, `composite.py`) hand it one
    argument, so a zero-arg definition raises TypeError there -- a break that
    is invisible on 0.28 and fatal on 0.31. Accepting an optional argument
    answers both, and the answer does not depend on it: V4.1's PAGE is 256
    tokens whatever spec is asking.
    """
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_BLOCK_SIZE,
        AtomDeepseekV41ProxyBackend,
    )

    expected = [ATOM_DEEPSEEK_V41_BLOCK_SIZE]
    assert AtomDeepseekV41ProxyBackend.get_supported_kernel_block_sizes() == expected
    assert (
        AtomDeepseekV41ProxyBackend.get_supported_kernel_block_sizes(object())
        == expected
    )


def test_the_proxy_layer_is_markable_for_the_memory_profile():
    """The bind must be able to tell a profiling pool from the serving one.

    `_mark_v4_proxy_cache_mode` only marks layers carrying
    `_atom_v4_proxy_layer`, and the bind reads
    `_atom_v4_profiling_kv_cache`. V4.1's proxy had neither, so the flag never
    left its default and the bind ran against the profile's placeholder pool
    -- 64 blocks under cudagraph capture, where `num_gpu_blocks` is already
    non-zero and so clears the `pages <= 0` check. The result was a bind-time
    "pool is too small", true of the placeholder and silent about the real one.
    """
    from atom.plugin.vllm.deepseek_v4_prefix_patch import _mark_v4_proxy_cache_mode
    from atom.plugin.vllm.deepseek_v41_bridge import AtomDeepseekV41ProxyAttention

    proxy = AtomDeepseekV41ProxyAttention()
    assert proxy._atom_v4_proxy_layer is True
    assert proxy._atom_v4_profiling_kv_cache is False

    # The marker reaches it, both ways, through the shared V4 helper.
    _mark_v4_proxy_cache_mode({"layer": proxy}, True)
    assert proxy._atom_v4_profiling_kv_cache is True
    _mark_v4_proxy_cache_mode({"layer": proxy}, False)
    assert proxy._atom_v4_profiling_kv_cache is False


def test_the_profile_cache_patch_installs_without_a_v4_model():
    """Installed by `register_model`, not by V4's proxy-layer registration.

    The patch wraps `initialize_kv_cache` so `_mark_v4_proxy_cache_mode` can
    flip the profiling flag V4.1's bind reads. It used to be installed from
    `register_deepseek_v4_proxy_layer`, so a V4.1 run -- which registers its
    own layer and never that one -- left the flag at its default and bound
    against the profile's placeholder pool. Checked on the wiring: calling
    `register_model` here would reach far more than this question.
    """
    import importlib
    import inspect

    register = importlib.import_module("atom.plugin.vllm.register")
    source = inspect.getsource(register.register_model)
    assert "apply_vllm_v4_profile_cache_patch()" in source, (
        "register_model must install the profile-cache patch; installing it "
        "from a model's own registration is what skipped V4.1"
    )
    # `register_platform` installs it too, but it is not checked here: that
    # hook can be swallowed whole (see `deepseek_v41_state_reserve_patch`), so
    # it is the belt and this is the braces.


def test_the_profile_cache_patch_reaches_the_runner_the_worker_builds():
    """vLLM has two unrelated `GPUModelRunner` classes; patch both.

    `GPUWorker` picks between `vllm.v1.worker.gpu_model_runner` and the V2
    rewrite in `vllm.v1.worker.gpu.model_runner` on `use_v2_model_runner`.
    They share no base class and no method objects, so the patch -- written
    against the first name alone -- was inert on every V2 deployment: the
    marker never ran, `_atom_v4_profiling_kv_cache` stayed False through the
    profiling capture, and the bind could not tell the 64-block throwaway pool
    from the serving one. Inert and applied look identical in the logs, which
    is why this is asserted on the classes rather than on a log line.
    """
    from atom.plugin.vllm.deepseek_v4_prefix_patch import (
        apply_vllm_v4_profile_cache_patch,
    )
    from atom.plugin.vllm.gpu_model_runner_targets import gpu_model_runner_classes

    classes = gpu_model_runner_classes()
    assert classes, "no vLLM GPUModelRunner class resolved"
    apply_vllm_v4_profile_cache_patch()
    for runner_cls in classes:
        assert getattr(
            runner_cls.initialize_kv_cache, "_atom_v4_profile_cache_patched", False
        ), f"{runner_cls.__module__}.{runner_cls.__qualname__} left unpatched"
    # Idempotent: a second install must not stack a wrapper on a wrapper.
    wrapped = [c.initialize_kv_cache for c in classes]
    apply_vllm_v4_profile_cache_patch()
    assert [c.initialize_kv_cache for c in classes] == wrapped


def test_the_bind_does_not_read_whether_capturing_is_merely_permitted():
    """`cudagraph_capturing_enabled` is a permission, not a phase.

    It is declared `True` in `vllm/compilation/monitor.py` and set False only
    when a capture phase ends, so in an eager run -- where no capture ever
    happens -- it stays True for the life of the process. A bind guard keyed on
    it therefore stood down on every forward, and the model served from the
    private scratch cache: a healthy server emitting noise. Keyed the other
    way it is just as wrong, since capture against the *serving* pool is
    exactly when the bind must happen, or the captured graphs replay a scratch
    cache.

    Asserted on the source because the symptom is an absence: there is no
    value this predicate could return that would make reading it correct.
    """
    import inspect

    import atom.plugin.vllm.deepseek_v41_bridge as bridge_mod

    source = inspect.getsource(bridge_mod)
    assert "cudagraph_capturing_enabled" not in source, (
        "the V4.1 bind must not gate on vLLM's capture permission flag; "
        "the phase it needs to exclude is the profiling pool, which "
        "_atom_v4_profiling_kv_cache names"
    )
