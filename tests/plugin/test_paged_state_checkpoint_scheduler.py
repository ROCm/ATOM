# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The state checkpoint layer on vLLM's real scheduler, block pool and KV-config code.

A fake adapter (block 256, k=3) over a tiny local Llama config, so nothing here
needs a checkpoint or a GPU.
"""

import json
import time
from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("vllm", reason="the state checkpoint layer patches vLLM")

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
from atom.plugin.vllm import paged_state_checkpoint as G
from atom.plugin.vllm.paged_state_checkpoint_scheduler import apply_scheduler_patches

BLOCK = 256
K = 3
PAGE = 1024


class _Adapter(G.PagedStateCheckpointAdapter):
    name = "FakeSched"
    layer_suffix = "fake_sched_state_checkpoint"
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


def _mtp_spec(k):
    """vLLM's own SpeculativeConfig as an EAGLE-class (MTP) drafter would have
    it, without a draft checkpoint: built as ngram, then marked ``mtp``."""
    from vllm.config import SpeculativeConfig

    spec = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=k,
        prompt_lookup_max=2,
        prompt_lookup_min=1,
    )
    object.__setattr__(spec, "method", "mtp")
    assert spec.use_eagle() and spec.use_eagle_block_drop()
    return spec


def _vllm_config(tmp_path, *, max_model_len=4096, num_blocks=64, spec_k=0):
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
    vc = VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc)
    if spec_k:
        vc.speculative_config = _mtp_spec(spec_k)
    return vc


def _specs(adapter):
    proxy = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=2, dtype=torch.uint8
    )
    assert proxy.page_size_bytes == PAGE
    specs = {adapter.proxy_layer_name: proxy}
    for name in adapter.state_checkpoint_layer_names(K):
        specs[name] = MambaSpec(
            block_size=BLOCK,
            shapes=((PAGE,),),
            dtypes=(torch.uint8,),
            mamba_cache_mode="align",
        )
    return specs


def _scheduler(
    adapter, tmp_path, *, num_blocks=64, hash_block=BLOCK, native_hits=False, **kw
):
    vc = _vllm_config(tmp_path, num_blocks=num_blocks, **kw)
    if native_hits:
        assert G.follow_native_hit_rule(vc)
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
    assert s._atom_state_checkpoints is not None
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
    st = eng.s._atom_state_checkpoints
    for _ in range(max_steps):
        if len(eng.q) >= 2:
            eng.land()
        eng.schedule()
        if rid in st.pending_hash:
            return st.pending_hash[rid]
    raise AssertionError("no decode keep was deferred")


def _boundary_cached(s, req, idx):
    st = s._atom_state_checkpoints
    return st._state_checkpoint_cached(req, idx)


def _upstream_configs(adapter, tmp_path, physical_blocks, *, max_model_len=4096):
    """vLLM's real get_kv_cache_configs (auto-fit, null-block reserve, memory
    check, tensor placement) under the state checkpoint hook, for ``physical_blocks``.
    """
    from vllm.v1.core import kv_cache_utils as U

    vc = _vllm_config(tmp_path, max_model_len=max_model_len)
    original = getattr(U.get_kv_cache_configs, "__wrapped__", U.get_kv_cache_configs)
    (cfg,) = G._kv_cache_configs_hook(
        original, vc, [_specs(adapter)], [physical_blocks * PAGE]
    )
    return vc, cfg


# max_model_len 4096 = 16 blocks; a running request also holds 2 blocks in each
# of the K state checkpoint groups, and vLLM's block pool keeps one null block for good.
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
    monkeypatch.setenv("ATOM_STATE_CHECKPOINT_CHECK_BLOCKS", "1")
    # Interval 0 publishes no state checkpoint, but the state checkpoint groups still
    # allocate.
    with pytest.raises(ValueError, match="pool too small"):
        _upstream_configs(adapter, tmp_path, 16 + 1 + 11)
    s = _scheduler(adapter, tmp_path)
    st = s._atom_state_checkpoints
    eng = _Engine(s)
    prompt = list(range(1, 1001))
    _request(s, "z1", prompt, max_tokens=8)
    eng.drain()
    assert 0 < st.counters["max_state_checkpoint_blocks_per_request"] <= 2 * K
    assert st.counters["state_checkpoints_hashed"] == 0
    # No state checkpoint, so no prefix hit: never a proxy-only (zero-state) resume.
    again = _request(s, "z2", prompt, max_tokens=1)
    eng.schedule()
    assert again.num_computed_tokens == len(prompt)  # scheduled from 0 in one go
    assert eng.q[0][0].scheduled_new_reqs[0].num_computed_tokens == 0
    eng.drain()


def test_auto_fit_max_model_len_leaves_room_for_the_slot_tail(adapter, tmp_path):
    """--max-model-len -1 under memory pressure: vLLM fits the length to all
    100 physical blocks (23,808 tokens = 99 blocks with the state checkpoints and the
    null block); the slots then take 11, so the length has to fit 89."""
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
    step's output lands, as EngineCore does. Whether 512's state checkpoint ends up
    cached.
    """
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
    prefix hits the state checkpoint while the first one is still generating.

    vLLM caps both ``cache_blocks`` call sites at the finalized tokens, so the
    state checkpoint groups' cached-block count passes the keep's block only once its
    hash exists: the deferred branch fires on the very next schedule."""
    s = _scheduler(adapter, tmp_path, hash_block=hash_block)
    st = s._atom_state_checkpoints
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
    cuts its prefill at the 512 anchor, and its state checkpoint there is hashed when
    that chunk is scheduled, before the forward that produces it. A reader admitted
    in the same schedule() would restore in that step's metadata build, before
    the writer's forward and store. vLLM's Mamba manager refuses a hit whose
    last block was cached in the current step (``cached_blocks_this_step``,
    which the state checkpoint groups fill), so the reader is admitted on a later step,
    still with the 512 hit. Without that set (``guard=False``) the reader is
    co-scheduled with the writer, which is the hazard."""
    s = _scheduler(adapter, tmp_path)
    st = s._atom_state_checkpoints
    if not guard:
        for gid in st.state_checkpoint_gids:
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
        if store_step is None and out.atom_state_checkpoint_stores.get("w") == 512:
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
    assert "a" not in s._atom_state_checkpoints.pending_hash
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
    assert "p" not in s._atom_state_checkpoints.pending_hash
    assert not _boundary_cached(s, req, idx)
    eng.drain()
    assert "p" not in s.requests


def _hit(s, rid, prompt):
    """vLLM's prefix hit for a new request, and each group's hit blocks."""
    probe = Request(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=s._test_hasher,
    )
    blocks, hit, _spb = s.kv_cache_manager.get_computed_blocks(probe)
    return hit, [list(g) for g in blocks.blocks]


@pytest.mark.parametrize("native_hits", [False, True])
@pytest.mark.parametrize("extra", [0, 300])
def test_with_mtp_the_hit_lands_on_the_anchor_as_native(
    adapter, tmp_path, native_hits, extra
):
    """A 1,000-token prompt keeps state checkpoints at the 512 rung and the 768 anchor.
    An EAGLE-class drafter makes vLLM drop one more matched block, so a re-send (or
    an extension) of the prompt hits 512 and recomputes 488+ tokens. Native's
    can_allocate only withholds the last block (always recomputed for logits),
    so it resumes at 768; with the draft's state in the state checkpoint the plugin
    follows it. The proxy group and every state checkpoint group report the same hit."""
    s = _scheduler(adapter, tmp_path, spec_k=1, native_hits=native_hits)
    assert s.use_eagle and s.use_eagle_block_drop is not native_hits
    st = s._atom_state_checkpoints
    eng = _Engine(s)
    prompt = list(range(1, 1001))
    _request(s, "w", prompt, max_tokens=1)
    eng.drain()
    req = Request(
        request_id="w2",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=s._test_hasher,
    )
    assert _boundary_cached(s, req, 1) and _boundary_cached(s, req, 2)
    hit, groups = _hit(s, "r", prompt + list(range(2001, 2001 + extra)))
    assert hit == (768 if native_hits else 512)
    assert len(groups[st.proxy_gid]) * BLOCK == hit
    for gid in st.state_checkpoint_gids:
        assert len(groups[gid]) * BLOCK == hit
        assert not groups[gid][-1].is_null and groups[gid][-1].block_hash is not None


class _SpecWorld:
    """vLLM's async loop under MTP k, with a model of the worker that runs each
    step right after it is scheduled (as the worker does, in step order).

    The worker uses the plugin's own decode-keep functions. A slot's content is
    the committed position its last step left it at; a state checkpoint block's content
    is the position of the slot it was copied from. A step commits
    ``1 + accept(rid, j)`` tokens on its ``j``-th decode step. Every state checkpoint
    block vLLM publishes must hold exactly its boundary's state by the next build,
    before any reader can restore it."""

    def __init__(self, s, accept, monkeypatch):
        self.s, self.accept = s, accept
        self.st = s._atom_state_checkpoints
        self.q = deque()
        self.tok = 0
        self.pos, self.slot_of, self.slot_pos = {}, {}, {}
        self.decode_steps = {}
        self.content = {}
        self.published = []
        self.pending_prefill = []
        self.copier = NS(execute_paged_state_copies=self._copies, checkpoint_spec=NS())
        self.worker = G.StateCheckpointWorker(
            None, self.copier, self.st.state_checkpoint_gids
        )
        self.req_ids = []
        self.builds = 0
        self.scheduling = False
        self.checked = 0
        self.restores = []
        self.prefill_end = {}
        monkeypatch.setattr(G, "state_checkpoint_unit_ids", self._unit_ids)
        monkeypatch.setattr(
            G, "_ops", lambda c, slots, units, store: list(zip(slots, units))
        )
        pool = s.kv_cache_manager.block_pool
        orig = pool.cache_full_blocks

        def cache_full_blocks(
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask=None,
        ):
            if kv_cache_group_id in self.st.state_checkpoint_gids:
                # The first build that can restore it: the next step's (one
                # published by schedule() is read no earlier than the build
                # after the one that step runs).
                first_reader = self.builds + (2 if self.scheduling else 1)
                for j, i in enumerate(range(num_cached_blocks, num_full_blocks)):
                    if block_mask is None or block_mask[j]:
                        self.published.append(
                            (first_reader, blocks[i].block_id, (i + 1) * block_size)
                        )
            return orig(
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
                block_mask,
            )

        pool.cache_full_blocks = cache_full_blocks

    def _unit_ids(self, group_ids, rows, cols, name="t"):
        out = []
        for r, c in zip(rows, cols):
            rid = self.req_ids[r]
            units = []
            for gid in group_ids:
                blk = self.st.coordinator.single_type_managers[gid].req_to_blocks[rid][
                    c
                ]
                assert not blk.is_null, (rid, c)
                units.append(blk.block_id)
            out.append(tuple(units))
        return out

    def _copies(self, stores, restores, descriptor_slot=0):
        for slot, units in stores:
            for u in units:
                self.content[u] = self.slot_pos[slot]

    def _verify(self, build):
        """Every state checkpoint a reader could restore in ``build`` holds its
        boundary's state once the build's own stores are issued."""
        left = []
        for first_reader, block_id, boundary in self.published:
            if first_reader <= build:
                assert self.content.get(block_id) == boundary, (block_id, boundary)
                self.checked += 1
            else:
                left.append((first_reader, block_id, boundary))
        self.published = left

    def _build_and_forward(self, out):
        self.builds += 1
        sched = out.num_scheduled_tokens
        self.req_ids = ids = list(sched)
        new = {r.req_id: r.num_computed_tokens for r in out.scheduled_new_reqs}
        cached = out.scheduled_cached_reqs
        resumed = {
            rid: cached.num_computed_tokens[i]
            for i, rid in enumerate(cached.req_ids)
            if rid in cached.resumed_req_ids
        }
        fresh = [i for i, rid in enumerate(ids) if rid in new or rid in resumed]
        starts = [new.get(r, resumed.get(r, self.pos.get(r))) for r in ids]
        for i in fresh:
            # A fresh slot; a hit restores the state checkpoint at its start. Up to
            # vLLM's prefill end the request replays as prefill.
            self.slot_of[ids[i]] = len(self.slot_of) + 100 * self.builds
            req = self.s.requests[ids[i]]
            self.prefill_end[ids[i]] = max(req.num_prompt_tokens, req.num_tokens - 1)
        for rid in ids:
            self.slot_of.setdefault(rid, len(self.slot_of))
        slots = [self.slot_of[r] for r in ids]
        # Native's site with a drafter: last step's stores, first in the build.
        for slot, units in self.pending_prefill:
            self._copies([(slot, units)], [])
        self.pending_prefill = []
        G._decode_plan.clear()
        G._decode_plan.update(out.atom_state_checkpoint_decode_candidates)
        G.store_decode_keeps(self.worker, ids, starts, fresh)
        lens = [sched[r] for r in ids]
        self.last_rows = {r: (starts[i], lens[i]) for i, r in enumerate(ids)}
        G.record_decode_candidates(self.worker, ids, slots, starts, lens, BLOCK, "t")
        for i in fresh:
            if starts[i] > 0:
                (units,) = self._unit_ids(
                    self.st.state_checkpoint_gids, [i], [starts[i] // BLOCK - 1]
                )
                assert [self.content.get(u) for u in units] == [starts[i]] * K, (
                    ids[i],
                    starts[i],
                    [self.content.get(u) for u in units],
                )
                self.restores.append((ids[i], starts[i]))
            self.slot_pos[slots[i]] = starts[i]
        for rid, end in out.atom_state_checkpoint_stores.items():
            i = ids.index(rid)
            assert starts[i] + lens[i] == end, (rid, starts[i], lens[i], end)
            (units,) = self._unit_ids(
                self.st.state_checkpoint_gids, [i], [end // BLOCK - 1]
            )
            self.pending_prefill.append((slots[i], units))
        self._verify(self.builds)
        # The forwards.
        toks = []
        for i, rid in enumerate(ids):
            req = self.s.requests[rid]
            spec = out.scheduled_spec_decode_tokens.get(rid, ())
            if starts[i] >= self.prefill_end[rid] and i not in fresh:
                j = self.decode_steps.get(rid, 0)
                self.decode_steps[rid] = j + 1
                gen = 1 + min(len(spec), self.accept(rid, j))
            else:
                gen = 1 if starts[i] + lens[i] >= req.num_tokens else 0
                if gen:
                    lens_done = starts[i] + lens[i]
                    self.pos[rid] = lens_done
                    self.slot_pos[slots[i]] = lens_done
                    toks.append([self._t()])
                    continue
                self.pos[rid] = starts[i] + lens[i]
                self.slot_pos[slots[i]] = self.pos[rid]
                toks.append([])
                continue
            self.pos[rid] = starts[i] + gen
            self.slot_pos[slots[i]] = self.pos[rid]
            toks.append([self._t() for _ in range(gen)])
        return toks

    def _t(self):
        self.tok += 1
        return self.tok % 997 + 1

    def schedule(self):
        self.scheduling = True
        try:
            out = self.s.schedule()
        finally:
            self.scheduling = False
        self.q.append((out, self._build_and_forward(out)))
        return out

    def land(self):
        out, toks = self.q.popleft()
        ids = list(out.num_scheduled_tokens)
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

    def drain(self, max_steps=20000):
        for _ in range(max_steps):
            if not self.s.requests and not self.q:
                # A later reader's build issues what is left first.
                for slot, units in self.pending_prefill:
                    self._copies([(slot, units)], [])
                self._verify(float("inf"))
                return
            if self.s.has_requests():
                self.schedule()
            if len(self.q) >= 2 or (self.q and not self.s.has_requests()):
                self.land()
        raise AssertionError("requests did not finish")


def _native_mtp_keeps(prompt, gens):
    """Native BlockManager on ``prompt``, whose decode steps commit
    ``gens[j]`` tokens each (block 256, interval 512, a PAGE checkpoint):
    postprocess as native's scheduler, ``hash_decode_blocks(seq, committed,
    next_forward_tokens=1)`` with ``committed = num_tokens - 1`` (the KV the
    step computed; undeferred output)."""
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
    keeps = []

    def checkpoint(seq, boundary_blocks, h):
        keeps.append(int(boundary_blocks) * BLOCK)
        return PagedStateCheckpointCoordinator.checkpoint(
            coord, seq, boundary_blocks, h
        )

    coord.checkpoint = checkpoint
    seq = Sequence(list(prompt), BLOCK, has_per_req_cache=True)
    bm.allocate(seq, bm.can_allocate(seq))
    bm.hash_blocks(seq, len(prompt), start_tokens=0)
    tok = 100
    seq.append_token(tok)  # sampled by the prefill
    bm.may_append(seq)
    for gen in gens:
        for _ in range(gen):
            tok += 1
            seq.append_token(tok)
            bm.may_append(seq)
        bm.hash_decode_blocks(seq, len(seq) - 1, next_forward_tokens=1)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
    bm.deallocate(seq)
    return keeps


_ACCEPT = {
    "all_accepted": lambda j: 1,
    "all_rejected": lambda j: 0,
    "alternating": lambda j: j % 2,
    "parity_lock_odd": lambda j: 0 if j == 0 else 1,
    "parity_lock_even": lambda j: 0 if j in (0, 1) else 1,
    "every_third_rejected": lambda j: 0 if j % 3 == 2 else 1,
}


# Native keeps per (prompt length, pattern); the parity lock leaves some empty.
_KEPT = {}


@pytest.mark.parametrize("prompt_len", [1, 2])
@pytest.mark.parametrize("pattern", sorted(_ACCEPT))
def test_mtp_decode_keeps_land_on_the_committed_end_as_native(
    adapter, tmp_path, monkeypatch, pattern, prompt_len
):
    """Under MTP k=1 on vLLM's AsyncScheduler a step commits 1 + accepted, known
    one step late. The plugin keeps exactly where native's hash_decode_blocks
    does (committed end on a block boundary, one interval past the last keep;
    a step that steps over a boundary keeps nothing), every published state checkpoint
    holds its boundary's state when the next build is done, and no step trips
    the worker's ``hashed at … but this step ends at …`` check."""
    accept = _ACCEPT[pattern]
    s = _scheduler(adapter, tmp_path, spec_k=1, native_hits=True, num_blocks=96)
    st = s._atom_state_checkpoints
    st.events = []
    world = _SpecWorld(s, lambda rid, j: accept(j), monkeypatch)
    n_answer = 2600
    prompt = list(range(7, 7 + prompt_len))
    _request(s, "d", prompt, max_tokens=n_answer)
    world.drain()
    gens = []
    j = 0
    total = 1  # the prefill's sampled token
    while total < n_answer:
        g = min(1 + accept(j), n_answer - total)
        gens.append(g)
        total += g
        j += 1
    native = [p for p in _native_mtp_keeps(prompt, gens) if p > prompt_len]
    plugin = [end for kind, rid, end in st.events if kind == "hash"]
    # The request's last step is never rescheduled, so it cannot keep there
    # (DESIGN §3); none of these sequences ends on a keep.
    assert native == plugin, (native, plugin)
    assert st.counters["decode_keeps_truncated"] == 0
    assert world.checked >= K * len(plugin) and not world.published
    _KEPT[(prompt_len, pattern)] = plugin


def test_mtp_decode_keep_scenarios_cover_both_parities():
    """Some sequences keep (e.g. all accepted from an even end), some are
    locked out by parity (all accepted from an odd end), as natively."""
    if len(_KEPT) < 2 * len(_ACCEPT):
        pytest.skip("run with the scenarios above")
    assert _KEPT[(2, "all_accepted")] and not _KEPT[(1, "all_accepted")]
    every_512 = [512, 1024, 1536, 2048, 2560]
    assert _KEPT[(1, "all_rejected")] == _KEPT[(2, "all_rejected")] == every_512


def test_mtp_follow_up_hits_a_decode_keep_and_a_preempted_request_resumes(
    adapter, tmp_path, monkeypatch
):
    """The EngineCore crash the earlier experiment hit (``hashed at 8448 but
    this step ends at 8447`` after a resume) at this test's scale: a request
    resumed from a cached state checkpoint under MTP keeps decoding, with acceptance
    varying, past its next decode keep. A follow-up over the first request's
    output resumes at its decode keep (restored state = that boundary), and a
    preempted request resumes from its own state checkpoints."""
    s = _scheduler(adapter, tmp_path, spec_k=1, native_hits=True, num_blocks=96)
    st = s._atom_state_checkpoints
    st.events = []
    world = _SpecWorld(s, lambda rid, j: (j * 7 + len(rid)) % 3 != 0, monkeypatch)
    prompt = list(range(1, 601))
    a = _request(s, "a", prompt, max_tokens=1500)
    world.drain()
    kept_a = [end for kind, rid, end in st.events if kind == "hash" and rid == "a"]
    decode_a = [p for p in kept_a if p > len(prompt)]
    assert decode_a, kept_a
    seen = prompt + list(a.output_token_ids[: decode_a[0] - len(prompt) + 37])
    _request(s, "b", seen, max_tokens=1300)
    for _ in range(40):
        if s.has_requests():
            world.schedule()
        if len(world.q) >= 2:
            world.land()
        if world.restores:
            break
    assert world.restores[0] == ("b", decode_a[0])
    # Preempt b mid-decode; it resumes from a state checkpoint and keeps going.
    for _ in range(400):
        breq = s.requests.get("b")
        if breq is not None and breq.num_computed_tokens > decode_a[0] + 700:
            break
        world.schedule()
        if len(world.q) >= 2:
            world.land()
    while world.q:
        world.land()
    breq = s.requests["b"]
    s.running.remove(breq)
    s._preempt_request(breq, time.time())
    world.drain()
    assert [r for r in world.restores if r[0] == "b"][1:], world.restores
    assert world.checked > 0 and not world.published
    kept_b = [end for kind, rid, end in st.events if kind == "hash" and rid == "b"]
    assert [p for p in kept_b if p > decode_a[0] + 512], kept_b


def test_align_under_spec_decode_keeps_the_running_block_at_end_over_block_minus_one(
    adapter, tmp_path, monkeypatch
):
    """vLLM's Mamba align bookkeeping for the state checkpoint groups under MTP k=1: no
    speculative blocks (a state checkpoint is not a running state), no lookahead
    allocation, the running block at column ``end // block - 1`` of the
    scheduled (optimistic) end, which also holds every boundary a step can
    commit to; vLLM's accepted-column copy moves nothing."""
    layer_spec = G.PagedStateCheckpointLayer("x", 0, PAGE, adapter).get_kv_cache_spec(
        None
    )
    assert layer_spec.num_speculative_blocks == 0
    assert layer_spec.mamba_cache_mode == "align"
    s = _scheduler(adapter, tmp_path, spec_k=1, native_hits=True, num_blocks=96)
    st = s._atom_state_checkpoints
    mgrs = [st.coordinator.single_type_managers[g] for g in st.state_checkpoint_gids]
    assert all(m.num_speculative_blocks == 0 for m in mgrs)
    assert all(m.mamba_cache_mode == "align" for m in mgrs)
    world = _SpecWorld(s, lambda rid, j: j % 3 != 1, monkeypatch)
    req = _request(s, "c", [7, 8], max_tokens=1400)
    checked = 0
    boundaries = set()
    for _ in range(5000):
        if "c" not in s.requests:
            break
        world.schedule()
        out = world.q[-1][0]
        threshold = out.atom_state_checkpoint_decode_candidates.get("c")
        if threshold is not None:
            end = int(req.num_computed_tokens)  # scheduled end, all drafts accepted
            start, n = world.last_rows["c"]  # the worker's exact start
            for m in mgrs:
                blocks = m.req_to_blocks["c"]
                # No lookahead blocks; the running block is the last one,
                # which is column end // block - 1 whenever end is aligned.
                assert len(blocks) == -(-end // BLOCK)
                assert not blocks[-1].is_null
                # Every boundary this step can commit to keeps its block.
                for b in range((start // BLOCK + 1) * BLOCK, start + n + 1, BLOCK):
                    if b >= threshold:
                        assert not blocks[b // BLOCK - 1].is_null, (b, start, n, end)
                        boundaries.add(b)
            checked += 1
        if len(world.q) >= 2:
            world.land()
    world.drain()
    assert checked > 500 and len(boundaries) >= 2, (checked, boundaries)
    # vLLM's pre/post copy of the accepted column (preprocess_mamba /
    # postprocess_mamba) goes through the state checkpoint layers' copy function.
    from vllm.model_executor.layers.mamba.mamba_utils import MambaCopySpec

    state = torch.zeros(8, PAGE, dtype=torch.uint8)
    (copy,) = G.state_checkpoint_copy_funcs()
    for accepted in (1, 2):
        spec = copy(state, [3, 5], 1, accepted)
        assert isinstance(spec, MambaCopySpec) and spec.num_elements == 0
        assert spec.start_addr == state[5].data_ptr()
