# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The image layer on vLLM's real scheduler, block pool and KV-config code.

A fake adapter (block 256, k=3) over a tiny local Llama config, so nothing here
needs a checkpoint or a GPU.
"""

import json
import time
from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("vllm", reason="the image layer patches vLLM")

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.plugin.vllm import paged_state_image as G
from atom.plugin.vllm.paged_state_image_scheduler import apply_scheduler_patches

BLOCK = 256
K = 3
PAGE = 1024


class _Adapter(G.PagedStateImageAdapter):
    name = "FakeSched"
    layer_suffix = "fake_sched_image"
    block_size = BLOCK
    proxy_layer_name = "model.layers.0.fake_sched_proxy"

    def matches(self, vllm_config):
        return True

    def sizing(self, vllm_config):
        return NS(
            spec=PagedStateCheckpointSpec(1000, 2600, "fake", image_bytes=2600),
            page_bytes=PAGE,
            num_slots=4,
            slot_bytes=2600,
            usable_blocks=lambda n: n - 11,
        )

    def checkpoint_defaults(self):
        return (4096, True)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS", "512")
    a = _Adapter()
    G._adapters.append(a)
    apply_scheduler_patches()
    init_none_hash(sha256)
    yield a
    G._adapters.remove(a)


def _model_dir(tmp_path, max_pos=65536):
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 1000,
        "max_position_embeddings": max_pos,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def _vllm_config(tmp_path, *, max_model_len=4096, num_blocks=64):
    mc = ModelConfig(
        model=_model_dir(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
        dtype="float16",
    )
    sc = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        max_model_len=mc.max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        async_scheduling=True,
    )
    cc = CacheConfig(
        block_size=BLOCK, enable_prefix_caching=True, mamba_cache_mode="align"
    )
    cc.num_gpu_blocks = num_blocks
    cc.kv_cache_layout = "LBHNC"  # resolved by EngineCore in a server
    return VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc)


def _specs(adapter):
    proxy = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=2, dtype=torch.uint8
    )
    assert proxy.page_size_bytes == PAGE
    specs = {adapter.proxy_layer_name: proxy}
    for name in adapter.image_layer_names(K):
        specs[name] = MambaSpec(
            block_size=BLOCK,
            shapes=((PAGE,),),
            dtypes=(torch.uint8,),
            mamba_cache_mode="align",
        )
    return specs


def _scheduler(adapter, tmp_path, *, num_blocks=64, hash_block=BLOCK, **kw):
    vc = _vllm_config(tmp_path, num_blocks=num_blocks, **kw)
    groups = [KVCacheGroupSpec([n], s) for n, s in _specs(adapter).items()]
    kvc = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    s = AsyncScheduler(
        vllm_config=vc,
        kv_cache_config=kvc,
        block_size=BLOCK,
        hash_block_size=hash_block,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vc),
    )
    assert s._atom_state_images is not None
    s._test_hasher = get_request_block_hasher(hash_block, sha256)
    return s


def _request(s, rid, prompt, max_tokens):
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, 1)
    req = Request(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=sp,
        pooling_params=None,
        mm_features=None,
        block_hasher=s._test_hasher,
    )
    s.add_request(req)
    return req


class _Engine:
    """vLLM's async loop: step N+1 is scheduled before step N's output lands."""

    def __init__(self, s):
        self.s = s
        self.q = deque()
        self.tok = 0

    def schedule(self):
        out = self.s.schedule()
        sampling = [
            rid
            for rid in out.num_scheduled_tokens
            if self.s.requests[rid].num_computed_tokens
            >= self.s.requests[rid].num_prompt_tokens
        ]
        self.q.append((out, sampling))
        return out

    def land(self):
        out, sampling = self.q.popleft()
        ids = list(out.num_scheduled_tokens)
        toks = []
        for rid in ids:
            self.tok += 1
            toks.append([self.tok % 997 + 1] if rid in sampling else [])
        self.s.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=toks,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    def step(self):
        self.schedule()
        if len(self.q) >= 2:
            self.land()

    def drain(self, max_steps=20000):
        for _ in range(max_steps):
            if not self.s.requests and not self.q:
                return
            if self.s.has_requests():
                self.schedule()
            if self.q:
                self.land()
        raise AssertionError("requests did not finish")


def _until_pending(eng, rid, max_steps=2000):
    """Step until the decode keep is deferred, stopping right after that step is
    scheduled (its output not landed yet)."""
    st = eng.s._atom_state_images
    for _ in range(max_steps):
        if len(eng.q) >= 2:
            eng.land()
        eng.schedule()
        if rid in st.pending_hash:
            return st.pending_hash[rid]
    raise AssertionError("no decode keep was deferred")


def _boundary_cached(s, req, idx):
    st = s._atom_state_images
    return st._image_cached(req, idx)


def _upstream_configs(adapter, tmp_path, physical_blocks, *, max_model_len=4096):
    """vLLM's real get_kv_cache_configs (auto-fit, null-block reserve, memory
    check, tensor placement) under the image hook, for ``physical_blocks``."""
    from vllm.v1.core import kv_cache_utils as U

    vc = _vllm_config(tmp_path, max_model_len=max_model_len)
    original = getattr(U.get_kv_cache_configs, "__wrapped__", U.get_kv_cache_configs)
    (cfg,) = G._kv_cache_configs_hook(
        original, vc, [_specs(adapter)], [physical_blocks * PAGE]
    )
    return vc, cfg


# max_model_len 4096 = 16 blocks; a running request also holds 2 blocks in each
# of the K image groups, and vLLM's block pool keeps one null block for good.
MIN_SCHEDULABLE = 16 + 2 * K + 1


def test_min_pool_check_counts_the_null_block(adapter, tmp_path):
    # The fake sizing withholds 11 blocks for the slots.
    _vc, cfg = _upstream_configs(adapter, tmp_path, MIN_SCHEDULABLE + 11)
    assert cfg.num_blocks == MIN_SCHEDULABLE
    with pytest.raises(ValueError, match="pool too small"):
        _upstream_configs(adapter, tmp_path, MIN_SCHEDULABLE + 10)


def _serve_max_len(s, *, preempt=False, max_steps=400):
    eng = _Engine(s)
    req = _request(s, "m", range(1, 4096), max_tokens=1)
    preempted = False
    for _ in range(max_steps):
        if not s.requests and not eng.q:
            return True
        if preempt and not preempted and req.num_computed_tokens >= 3840 and not eng.q:
            # The prompt-end anchor (3840) is stored and hashed; resume hits it.
            s.running.remove(req)
            s._preempt_request(req, time.time())
            preempted = True
        if s.has_requests():
            eng.schedule()
        if eng.q:
            eng.land()
    return False


@pytest.mark.parametrize("preempt", [False, True])
def test_smallest_accepted_pool_serves_a_max_length_request(adapter, tmp_path, preempt):
    ok = _scheduler(adapter, tmp_path, num_blocks=MIN_SCHEDULABLE)
    assert _serve_max_len(ok, preempt=preempt)
    short = _scheduler(adapter, tmp_path, num_blocks=MIN_SCHEDULABLE - 1)
    assert not _serve_max_len(short, preempt=preempt)


def test_interval_zero_still_counts_working_blocks_and_never_hits(
    adapter, tmp_path, monkeypatch
):
    monkeypatch.setenv("ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS", "0")
    monkeypatch.setenv("ATOM_STATE_IMAGE_CHECK_BLOCKS", "1")
    # Interval 0 publishes no image, but the image groups still allocate.
    with pytest.raises(ValueError, match="pool too small"):
        _upstream_configs(adapter, tmp_path, 16 + 1 + 11)
    s = _scheduler(adapter, tmp_path)
    st = s._atom_state_images
    eng = _Engine(s)
    prompt = list(range(1, 1001))
    _request(s, "z1", prompt, max_tokens=8)
    eng.drain()
    assert 0 < st.counters["max_image_blocks_per_request"] <= 2 * K
    assert st.counters["images_hashed"] == 0
    # No image, so no prefix hit: never a proxy-only (zero-state) resume.
    again = _request(s, "z2", prompt, max_tokens=1)
    eng.schedule()
    assert again.num_computed_tokens == len(prompt)  # scheduled from 0 in one go
    assert eng.q[0][0].scheduled_new_reqs[0].num_computed_tokens == 0
    eng.drain()


def test_auto_fit_max_model_len_leaves_room_for_the_slot_tail(adapter, tmp_path):
    """--max-model-len -1 under memory pressure: vLLM fits the length to all
    100 physical blocks (23,808 tokens = 99 blocks with the images and the null
    block); the slots then take 11, so the length has to fit 89."""
    vc, cfg = _upstream_configs(adapter, tmp_path, 100, max_model_len=-1)
    assert vc.model_config.original_max_model_len == -1
    fitted = vc.model_config.max_model_len
    assert fitted == (89 - 1 - 2 * K) * BLOCK < 23808
    assert cfg.num_blocks == 89
    # The slot bytes stay allocated after the schedulable blocks.
    assert {t.size for t in cfg.kv_cache_tensors} == {100 * PAGE}
    s = _scheduler(adapter, tmp_path, num_blocks=cfg.num_blocks, max_model_len=fitted)
    eng = _Engine(s)
    _request(s, "long", range(1, fitted), max_tokens=1)
    eng.drain()


def _native_keeps(n_answer, *, abort_after=None):
    """Native BlockManager on the same request: a 1-token prompt, ``n_answer``
    generated tokens (or an abort after ``abort_after`` were delivered), block 256,
    interval 512, a PAGE (copy) checkpoint. Postprocess as native's scheduler with
    deferred output (one step late, what vLLM's async scheduling compares to):
    ``hash_decode_blocks(seq, num_tokens, next_forward_tokens=0 on the last step)``,
    and an aborted seq skips it."""
    from conftest import MockConfig  # ATOM tests/conftest.py

    from atom.model_engine.block_manager import BlockManager
    from atom.model_engine.page_unit_checkpoint import (
        PagedStateCheckpointCoordinator,
        PagedStateCheckpointSpec,
    )
    from atom.model_engine.sequence import Sequence
    from atom.model_engine.state_runtime import StateRuntime, StateTransfer

    spec = PagedStateCheckpointSpec(1000, 2600, "fake", image_bytes=2600)
    cfg = MockConfig(
        kv_cache_block_size=BLOCK,
        num_kvcache_blocks=64,
        enable_prefix_caching=True,
        max_num_batched_tokens=2048,
        max_model_len=4096,
        pool_entries={"state": 4},
        state_checkpoint_interval_tokens=512,
        state_checkpoint_demand=True,
    )
    bm = BlockManager(
        cfg,
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )
    coord = bm.paged_state_checkpoints
    assert isinstance(coord, PagedStateCheckpointCoordinator)
    keeps = []

    def checkpoint(seq, boundary_blocks, h):
        keeps.append(int(boundary_blocks) * BLOCK)
        return PagedStateCheckpointCoordinator.checkpoint(
            coord, seq, boundary_blocks, h
        )

    coord.checkpoint = checkpoint
    seq = Sequence([7], BLOCK, has_per_req_cache=True)
    bm.allocate(seq, bm.can_allocate(seq))
    bm.hash_blocks(seq, 1, start_tokens=0)
    seq.num_cached_tokens = 1
    for j in range(n_answer):
        if j == abort_after:
            break
        seq.append_token(100 + j)
        bm.may_append(seq)
        last = j == n_answer - 1
        bm.hash_decode_blocks(seq, len(seq), next_forward_tokens=0 if last else 1)
        seq.num_cached_tokens = len(seq) - 1
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
    bm.deallocate(seq)
    return keeps


def _plugin_keeps_512(s, n_answer, *, abort_after=None, max_steps=3000):
    """Drive vLLM's async loop; an abort is handled before the oldest in-flight
    step's output lands, as EngineCore does. Whether 512's image ends up cached."""
    from vllm.v1.request import RequestStatus

    req = _request(s, "e", [7], max_tokens=n_answer)
    eng = _Engine(s)
    for _ in range(max_steps):
        if "e" not in s.requests:
            break
        eng.schedule()
        if len(eng.q) >= 2:
            if abort_after is not None and len(req.output_token_ids) == abort_after:
                s.finish_requests("e", RequestStatus.FINISHED_ABORTED)
                break
            eng.land()
    while eng.q:
        eng.land()
    assert "e" not in s.requests
    return _boundary_cached(s, req, 1)


@pytest.mark.parametrize(
    ("n_answer", "abort_after", "native", "plugin"),
    [
        # Natural end on the step whose forward reaches 512, or after it.
        (512, None, True, True),
        (520, None, True, True),
        # Abort after k delivered tokens: the keep exists once the token closing
        # its block (position 511) has landed, on both sides.
        (2000, 509, False, False),
        (2000, 510, False, False),
        (2000, 511, True, True),
        (2000, 512, True, True),
        # Engine scheduling, not placement: max_tokens ends one token short of
        # the forward that reaches 512. Native's deferred output forwards the
        # last token once more and keeps 512 there; vLLM schedules no forward
        # past max_tokens, so that state is never computed for the plugin.
        (511, None, True, False),
    ],
)
def test_request_end_around_the_decode_keep_matches_native(
    adapter, tmp_path, n_answer, abort_after, native, plugin
):
    """Native keeps a decode checkpoint on a request's last step as well: a PAGE
    (copy) checkpoint's successor_room is 0, and the scheduler passes
    next_forward_tokens=0 there. An aborted seq skips the postprocess that would
    hash and keep. The plugin publishes a pending decode keep exactly when its
    boundary hash exists, which gives the same decisions."""
    assert (512 in _native_keeps(n_answer, abort_after=abort_after)) is native
    s = _scheduler(adapter, tmp_path)
    assert _plugin_keeps_512(s, n_answer, abort_after=abort_after) is plugin


@pytest.mark.parametrize("hash_block", [BLOCK, BLOCK // 2])
def test_a_deferred_decode_keep_is_published_while_the_request_runs(
    adapter, tmp_path, hash_block
):
    """Async decode defers the keep at 512 (its closing token is in flight). It
    is published as soon as that hash exists, while the request keeps running,
    and a later keep (1024) does not replace it; a second request over the same
    prefix hits the image while the first one is still generating.

    vLLM caps both ``cache_blocks`` call sites at the finalized tokens, so the
    image groups' cached-block count passes the keep's block only once its hash
    exists: the deferred branch fires on the very next schedule."""
    s = _scheduler(adapter, tmp_path, hash_block=hash_block)
    st = s._atom_state_images
    eng = _Engine(s)
    req = _request(s, "long", [7], max_tokens=3000)
    assert _until_pending(eng, "long") == 1

    def run_to(computed):
        for _ in range(4 * computed):
            if req.num_computed_tokens >= computed:
                return
            if len(eng.q) >= 2:
                eng.land()
            eng.schedule()
        raise AssertionError(f"did not reach {computed}")

    run_to(512 + 2 * BLOCK)
    assert "long" in s.requests and _boundary_cached(s, req, 1)
    run_to(1024 + 2 * BLOCK)
    assert "long" in s.requests
    assert _boundary_cached(s, req, 1) and _boundary_cached(s, req, 3)
    assert st.counters["decode_keeps"] >= 2
    # A follow-up over the first request's prompt and part of its output.
    seen = [7] + list(req.output_token_ids[:700])
    _request(s, "next", seen, max_tokens=1)
    for _ in range(50):
        if len(eng.q) >= 2:
            eng.land()
        out = eng.schedule()
        new = [r for r in out.scheduled_new_reqs if r.req_id == "next"]
        if new:
            break
    assert new and new[0].num_computed_tokens == 512
    assert "long" in s.requests
    while eng.q:
        eng.land()
    eng.drain()


class _NoAdd(set):
    def add(self, item):
        pass


@pytest.mark.parametrize("guard", [True, False])
def test_a_co_scheduled_reader_restores_only_after_the_writers_store(
    adapter, tmp_path, guard
):
    """Two requests with the same 600-token prompt arrive together. The writer
    cuts its prefill at the 512 anchor, and its image there is hashed when that
    chunk is scheduled, before the forward that produces it. A reader admitted
    in the same schedule() would restore in that step's metadata build, before
    the writer's forward and store. vLLM's Mamba manager refuses a hit whose
    last block was cached in the current step (``cached_blocks_this_step``,
    which the image groups fill), so the reader is admitted on a later step,
    still with the 512 hit. Without that set (``guard=False``) the reader is
    co-scheduled with the writer, which is the hazard."""
    s = _scheduler(adapter, tmp_path)
    st = s._atom_state_images
    if not guard:
        for gid in st.image_gids:
            st.coordinator.single_type_managers[gid].cached_blocks_this_step = _NoAdd()
    eng = _Engine(s)
    prompt = list(range(1, 601))
    _request(s, "w", prompt, max_tokens=2)
    _request(s, "r", prompt, max_tokens=2)
    store_step = first_r = None
    for step in range(20):
        if len(eng.q) >= 2:
            eng.land()
        out = eng.schedule()
        if store_step is None and out.atom_image_stores.get("w") == 512:
            store_step = step
        for new in out.scheduled_new_reqs:
            if new.req_id == "r":
                first_r = (step, new.num_computed_tokens)
        if first_r is not None:
            break
    assert store_step is not None and first_r is not None
    step_r, hit_r = first_r
    assert hit_r == 512
    if guard:
        assert step_r > store_step
    else:
        assert step_r == store_step
    while eng.q:
        eng.land()
    eng.drain()


@pytest.mark.parametrize("hash_block", [BLOCK, BLOCK // 2])
def test_abort_with_a_deferred_decode_keep_frees_the_request(
    adapter, tmp_path, hash_block
):
    """Async decode of a 1-token prompt reaches the decode keep at 512 while the
    CPU holds 511 tokens; the client cancels. vLLM handles the abort before the
    step's output (EngineCore), so the boundary hash does not exist yet."""
    s = _scheduler(adapter, tmp_path, hash_block=hash_block)
    eng = _Engine(s)
    req = _request(s, "a", [7], max_tokens=2000)
    idx = _until_pending(eng, "a")
    assert idx == 1 and req.num_computed_tokens == 512 and req.num_tokens == 511
    free_before = s.kv_cache_manager.block_pool.get_num_free_blocks()
    s.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert "a" not in s.requests
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() > free_before
    assert "a" not in s._atom_state_images.pending_hash
    assert not _boundary_cached(s, req, idx)
    # The engine keeps serving.
    _request(s, "b", list(range(1, 300)), max_tokens=4)
    while eng.q:
        eng.land()
    eng.drain()


def test_natural_completion_publishes_the_deferred_decode_keep(adapter, tmp_path):
    s = _scheduler(adapter, tmp_path)
    eng = _Engine(s)
    req = _request(s, "n", [7], max_tokens=512)
    idx = _until_pending(eng, "n")
    eng.drain()
    assert "n" not in s.requests
    assert _boundary_cached(s, req, idx)


def test_preemption_drops_the_deferred_decode_keep_and_resumes(adapter, tmp_path):
    s = _scheduler(adapter, tmp_path)
    eng = _Engine(s)
    req = _request(s, "p", [7], max_tokens=600)
    idx = _until_pending(eng, "p")
    s.running.remove(req)
    s._preempt_request(req, time.time())
    assert "p" not in s._atom_state_images.pending_hash
    assert not _boundary_cached(s, req, idx)
    eng.drain()
    assert "p" not in s.requests
