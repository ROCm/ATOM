# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contract of DeepSeek-V4.1 DSpark on the SGLang plugin path.

SGLang's ``DSparkWorkerV2`` (static verify, eager) drives three forwards a
round: the draft block as a TARGET_VERIFY of five rows on the draft runner,
the target's TARGET_VERIFY of six rows, and the accept. These tests pin what
the plugin translates: the capability predicate and its negatives, the
speculative config, the verify batch and its tentative staging, the commit of
the accepted prefix (once, from SGLang's ``commit_lens``), the draft's role
of the proxy pool, the draft block's inputs and forward context, the Markov
head adapter and the target's aux taps. No GPU.
"""

import json
import os
import weakref
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    DEEPSEEK_V41_ARCH,
    build_deepseek_v41_forward_metadata,
    build_v41_scheduled_batch,
    commit_deepseek_v41_dspark_verify,
    dspark_commit_anchors,
    is_v41_dspark_atom_config,
    v41_dspark_capability_error,
)

try:
    import sglang.srt.mem_cache.base_swa_memory_pool  # noqa: F401
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    HAVE_SGLANG = True
except Exception:  # noqa: BLE001 - any import failure means "no real SGLang"
    HAVE_SGLANG = False
    ForwardMode = None

needs_sglang = pytest.mark.skipif(not HAVE_SGLANG, reason="SGLang is not importable")

try:
    # Native DSpark imports the V4.1 attention, which needs an aiter with
    # the V4.1 sparse-prefill backend.
    import atom.models.deepseek_v41.dspark  # noqa: F401

    HAVE_NATIVE_DSPARK = True
except ImportError:
    HAVE_NATIVE_DSPARK = False

needs_native_dspark = pytest.mark.skipif(
    not HAVE_NATIVE_DSPARK, reason="native DeepSeek-V4.1 DSpark is not importable"
)

PAGE = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE
# The builder geometry's Engram history width.
HISTORY = 3
W = 6  # verify rows a request: bonus + 5 drafts
FLASH_CHECKPOINT = "/data/models/ginsongsong/DeepSeek-V4.1-Flash-FP4PTPC"
HAVE_CHECKPOINT = os.path.exists(os.path.join(FLASH_CHECKPOINT, "config.json"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(bridge, "_V41_BUILDER", None)
    monkeypatch.delenv("SGLANG_RAGGED_VERIFY_MODE", raising=False)
    monkeypatch.delenv("SGLANG_SIMULATE_ACC_LEN", raising=False)
    monkeypatch.delenv("SGLANG_DSPARK_EMBED_IN_GRAPH", raising=False)


def _forward_batch(**fields):
    """A ForwardBatch stand-in: every field the bridge reads, at SGLang's
    defaults, then ``fields``."""
    batch = SimpleNamespace(
        req_pool_indices_cpu=None,
        extend_seq_lens=None,
        extend_prefix_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
        is_prefill_only=False,
        capture_hidden_mode=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
        spec_info=None,
    )
    batch.__dict__.update(fields)
    return batch


def _v41(**overrides):
    """SGLang's normalized V4.1 config (flat, V4's architecture name)."""
    fields = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v41",
        "num_hidden_layers": 40,
        "num_nextn_predict_layers": 3,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_noise_token_id": 128799,
        "dspark_markov_rank": 256,
        "hidden_size": 5120,
        "vocab_size": 129280,
        "engram_layer_ids": [2, 14],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _atom_text_v41(**overrides):
    config = _v41(**overrides)
    config.model_type = "deepseek_v41_text"
    config.architectures = [DEEPSEEK_V41_ARCH]
    return config


# ---------------------------------------------------------------------------
# Capability routing
# ---------------------------------------------------------------------------


class TestCapability:
    @pytest.mark.parametrize("config", [_v41(), _atom_text_v41()])
    def test_positive(self, config):
        assert v41_dspark_capability_error(config) is None

    @pytest.mark.parametrize(
        ("config", "match"),
        [
            (
                SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"],
                    model_type="deepseek_v4",
                    dspark_block_size=5,
                ),
                "not a DeepSeek-V4.1",
            ),
            (_v41(dspark_block_size=None), "bundles no DSpark head"),
            (_v41(dspark_block_size=4), "dspark_block_size 4"),
            (_v41(dspark_target_layer_ids=[]), "dspark_target_layer_ids"),
            (_v41(dspark_noise_token_id=None), "noise"),
            (_v41(dspark_markov_rank=0), "markov_rank"),
            (_v41(num_nextn_predict_layers=1), "num_nextn_predict_layers 1"),
            (
                _v41(compress_ratios=[0] * 40 + [0, 4, 0]),
                "not window-only",
            ),
            (_v41(compress_ratios=[0] * 40), "not window-only"),
        ],
    )
    def test_negative(self, config, match):
        assert match in v41_dspark_capability_error(config)

    def test_atom_config_predicate(self):
        spec = SimpleNamespace(method="dspark")
        assert is_v41_dspark_atom_config(
            SimpleNamespace(hf_config=_atom_text_v41(), speculative_config=spec)
        )
        # V4.1 without speculation keeps the no-spec path.
        assert not is_v41_dspark_atom_config(
            SimpleNamespace(hf_config=_atom_text_v41(), speculative_config=None)
        )
        # A V4 target with a dspark config is not this path.
        assert not is_v41_dspark_atom_config(
            SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"]
                ),
                speculative_config=spec,
            )
        )


# ---------------------------------------------------------------------------
# Speculative config
# ---------------------------------------------------------------------------


# The server args `_server_args` built last: SGLang resolves the speculative
# settings into its spec bag, which `spec_config_env` serves from here.
_LAST_SERVER_ARGS = []


# ---------------------------------------------------------------------------
# Launch refusals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TARGET_VERIFY -> tentative ATOM step
# ---------------------------------------------------------------------------


def _req_to_token(rows, context_len, pages_by_row):
    table = torch.zeros(rows, context_len, dtype=torch.int32)
    for row, pages in pages_by_row.items():
        for i, page in enumerate(pages):
            start = i * PAGE
            width = min(PAGE, context_len - start)
            table[row, start : start + width] = page * PAGE + torch.arange(width)
    return table


@pytest.fixture
def req_to_token(monkeypatch):
    table = _req_to_token(
        rows=5,
        context_len=4 * PAGE,
        pages_by_row={1: [7, 3, 9], 2: [5], 3: [2, 11, 4, 8]},
    )
    monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
    return table


def _verify_batch(prefix, slots, *, width=W, live=True, seq_lens=None):
    bs = len(prefix)
    prefix = torch.tensor(prefix, dtype=torch.int64)
    ends = prefix + width if seq_lens is None else torch.tensor(seq_lens)
    return _forward_batch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        rids=[f"r{i}" for i in range(bs)],
        batch_size=bs,
        # ForwardBatch.seq_lens is the committed prefix; the verify caller
        # widens only the CPU mirror.
        seq_lens=prefix.clone(),
        seq_lens_cpu=ends,
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=None,
        spec_info=SimpleNamespace(
            draft_token_num=width, live_seq_lens_cpu=prefix.clone() if live else None
        ),
        input_ids=torch.zeros(bs * width, dtype=torch.int64),
    )


@needs_sglang
class TestVerifyBatch:
    def test_verify_rows_and_pages(self, req_to_token):
        # p = 254 crosses a page inside the verify: 254..259.
        fb = _verify_batch([254, 10, 300], [1, 2, 3])
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert batch.num_scheduled_tokens.tolist() == [W] * 3
        assert batch.context_lens.tolist() == [260, 16, 306]
        assert batch.total_tokens_num == 3 * W
        assert batch.state_slots_committed == [1, 2, 3]
        assert [row.tolist() for row in batch.block_tables] == [[7, 3], [5], [2, 11]]

    def test_a_verify_records_its_host_rows(self, monkeypatch, req_to_token):
        # Over-allocated pages stay with the request until its KV is
        # released, so a verify's rows are copies a later replay may use.
        monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
        fb = _verify_batch([254, 10], [1, 2])
        build_v41_scheduled_batch(fb, PAGE)
        rows = bridge._HOST_ROWS.lookup([1, 2], ["r0", "r1"], [2, 1])
        assert [row.tolist() for row in rows] == [[7, 3], [5]]

    def test_verify_without_live_lengths_is_refused(self, req_to_token):
        # seq_lens may be SGLang's over-allocated bound: no silent fallback.
        fb = _verify_batch([254, 10], [1, 2], live=False)
        with pytest.raises(ValueError, match="live_seq_lens_cpu"):
            build_v41_scheduled_batch(fb, PAGE)

    def test_lengths_must_agree(self, req_to_token):
        fb = _verify_batch([254, 10], [1, 2], seq_lens=[255, 16])
        with pytest.raises(ValueError, match="lengths disagree"):
            build_v41_scheduled_batch(fb, PAGE)


class _Pending:
    def __init__(self, step):
        self.step = step


class _VerifyBuilder:
    """Records the staging calls; `_prepare` stages a tentative step the way
    the real builder's `prepare_model_inputs` leaves `cache.pending`."""

    _decoder_replays = False

    def __init__(self, positions, speculative_tokens=5):
        self.block_size = PAGE
        self.max_bs = 8
        self.max_num_batched_tokens = 1024
        self.geometry = SimpleNamespace(
            speculative_tokens=speculative_tokens, history_size=HISTORY
        )
        self.cache = SimpleNamespace(pending=None)
        self.positions = positions
        self.calls = []

    def _metadata(self, bs, tentative):
        step = SimpleNamespace(scheduled_bs=bs, tentative=tentative)
        return SimpleNamespace(cache=self.cache, step=step, engram_embeddings=None)

    def _prepare(self, batch, running_bs, running_tokens, **kwargs):
        self.calls.append(("prepare", running_bs, running_tokens, kwargs))
        self.batch = batch
        return self._metadata(running_bs, True), self.positions

    def prepare_prefill(self, batch, running_bs, running_tokens):
        self.calls.append(("prefill", running_bs, running_tokens))
        return self._metadata(running_bs, False), self.positions

    def prepare_decode(self, *args):
        raise AssertionError("DSpark never stages a plain decode")

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", int(input_ids.shape[0])))
        if metadata.step.tentative:
            self.cache.pending = _Pending(metadata.step)


@needs_sglang
class TestVerifyMetadata:
    def _positions(self, prefix):
        return torch.cat([torch.arange(p, p + W) for p in prefix])

    def test_verify_is_staged_tentatively(self, monkeypatch, req_to_token):
        prefix = [254, 10]
        positions = self._positions(prefix)
        builder = _VerifyBuilder(positions.clone())
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb = _verify_batch(prefix, [1, 2])
        metadata = build_deepseek_v41_forward_metadata(fb)
        assert metadata.staged_outside_forward
        assert builder.calls[0] == (
            "prepare",
            2,
            2 * W,
            {"max_q_len": W, "tentative": True},
        )
        assert builder.calls[1] == ("inputs", 2 * W)
        assert builder.cache.pending.step is metadata.step

    def test_extend_stays_non_tentative(self, monkeypatch, req_to_token):
        positions = torch.arange(5)
        builder = _VerifyBuilder(positions.clone())
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb = _forward_batch(
            forward_mode=ForwardMode.EXTEND,
            rids=["r0"],
            batch_size=1,
            seq_lens=torch.tensor([5]),
            seq_lens_cpu=torch.tensor([5]),
            req_pool_indices=torch.tensor([1]),
            req_pool_indices_cpu=None,
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            input_ids=torch.zeros(5, dtype=torch.int64),
        )
        build_deepseek_v41_forward_metadata(fb)
        assert builder.calls[0] == ("prefill", 1, 5)
        assert builder.cache.pending is None


# ---------------------------------------------------------------------------
# Commit of the accepted prefix
# ---------------------------------------------------------------------------


def test_anchors_are_last_accepted_rows():
    lens = torch.tensor([1, 6, 3])
    assert dspark_commit_anchors(lens, W).tolist() == [0, 11, 14]
    assert dspark_commit_anchors(lens, W).dtype == torch.int32


def _tentative_state(slots, prefix, histories=4):
    pytest.importorskip("triton")
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
    from atom.model_ops.attentions.deepseek_v41.speculative import TentativeState

    bs = len(slots)
    spans = [
        RequestSpan(f"r{i}", int(prefix[i]), i * W, W, int(slot))
        for i, slot in enumerate(slots)
    ]
    step = SimpleNamespace(
        requests=tuple(spans),
        scheduled_bs=bs,
        max_q_len=W,
        cu_seqlens_q=torch.arange(0, (bs + 1) * W, W, dtype=torch.int32),
        slots=torch.tensor(slots, dtype=torch.int32),
    )
    staging = torch.arange(bs * W * (1 + histories), dtype=torch.int64).view(
        bs, W, 1 + histories
    )
    cache = SimpleNamespace(
        tentative_staging=SimpleNamespace(gpu=staging),
        cursor=torch.full((8, 1 + histories), -7, dtype=torch.int64),
        pending=None,
        # Set by the verify's staging (`prepare_model_inputs`).
        _sglang_h2d_owner=_Owner(),
    )
    state = TentativeState(cache, step)
    state.staged_on_device = True
    cache.pending = state

    def commit_tentative(s, anchors):
        if cache.pending is None or cache.pending.step is not s:
            raise RuntimeError("Tentative state was not prepared for this step")
        cache.pending.commit(anchors)
        cache.pending = None

    cache.commit_tentative = commit_tentative
    return cache, step, staging


def test_tentative_commit_picks_each_requests_accepted_prefix():
    cache, step, staging = _tentative_state([3, 1, 5], [100, 7, 250])
    lens = torch.tensor([1, 6, 3])
    cache.commit_tentative(step, dspark_commit_anchors(lens, W))
    for i, slot in enumerate([3, 1, 5]):
        assert torch.equal(cache.cursor[slot], staging[i, int(lens[i]) - 1])
    assert cache.pending is None


def _target_pool(cache):
    return SimpleNamespace(atom_cache=cache)


class _Owner:
    def __init__(self):
        self.calls = []

    def resume(self):
        self.calls.append("resume")

    def finish(self):
        self.calls.append("finish")

    def fail(self):
        self.calls.append("fail")


class TestCommit:
    def test_commits_once_in_the_resumed_epoch(self):
        cache, _step, staging = _tentative_state([3, 1], [100, 7])
        owner = cache._sglang_h2d_owner = _Owner()
        pool = _target_pool(cache)
        lens = torch.tensor([2, 4])
        commit_deepseek_v41_dspark_verify(
            pool, req_slots=[3, 1], commit_lens=lens, verify_width=W
        )
        assert owner.calls == ["resume", "finish"]
        assert torch.equal(cache.cursor[3], staging[0, 1])
        assert torch.equal(cache.cursor[1], staging[1, 3])
        assert cache.pending is None

    def test_slot_mismatch_is_refused(self):
        cache, _, _ = _tentative_state([3, 1], [100, 7])
        with pytest.raises(RuntimeError, match="slots"):
            commit_deepseek_v41_dspark_verify(
                _target_pool(cache),
                req_slots=[1, 3],
                commit_lens=torch.tensor([1, 1]),
                verify_width=W,
            )
        assert cache.pending is not None


@needs_sglang
class TestCommitHook:
    def _worker(self, pool):
        return SimpleNamespace(
            _need_mamba_verify_commit=False,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(token_to_kv_pool=pool)
            ),
            verify_num_draft_tokens=W,
        )

    def _commit(self, worker, slots, lens):
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        DSparkWorkerV2._commit_target_mamba_states_after_verify(
            worker,
            batch=SimpleNamespace(
                req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64)
            ),
            seq_lens_pre_verify=None,
            seq_lens_post_verify=None,
            commit_lens=lens,
        )

    def test_install_is_idempotent_and_commits_v41(self):
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        bridge.install_deepseek_v41_dspark_commit_patch()
        patched = DSparkWorkerV2._commit_target_mamba_states_after_verify
        bridge.install_deepseek_v41_dspark_commit_patch()
        assert DSparkWorkerV2._commit_target_mamba_states_after_verify is patched
        cache, _, staging = _tentative_state([2, 4], [50, 60])
        self._commit(self._worker(_target_pool(cache)), [2, 4], torch.tensor([3, 1]))
        assert cache.pending is None
        assert torch.equal(cache.cursor[2], staging[0, 2])


# ---------------------------------------------------------------------------
# Proxy pool: the draft's role
# ---------------------------------------------------------------------------


def _small_geometry(speculative_tokens):
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    return V41PoolGeometry(
        4 + (3 if speculative_tokens else 0),
        ((0, 2), (2, 1)),
        PAGE,
        4,
        512,
        32,
        kv_dtype="bf16",
        speculative_tokens=speculative_tokens,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )


def _proxy(**overrides):
    kwargs = {
        "max_num_reqs": 3,
        "swa_size": 0,
        "full_size": 4 * PAGE,
        "page_size": PAGE,
        "swa_page_size": PAGE,
        "dtype": torch.bfloat16,
        "layer_num": 3,
        "device": "cpu",
        "num_req_slots": 4,
        "is_draft_worker": True,
    }
    kwargs.update(overrides)
    return bridge._make_proxy_pool_cls()(**kwargs)


# ---------------------------------------------------------------------------
# Geometry and builder capacity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_CHECKPOINT, reason="V4.1-Flash checkpoint absent")
def test_flash_dspark_geometry():
    from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry

    with open(os.path.join(FLASH_CHECKPOINT, "config.json")) as f:
        text = SimpleNamespace(**json.load(f)["text_config"])
    kwargs = {"kv_dtype": "bf16", "index_fp4": False, "index_layout": None}
    spec = build_v41_pool_geometry(text, PAGE, speculative_tokens=5, **kwargs)
    plain = build_v41_pool_geometry(text, PAGE, speculative_tokens=0, **kwargs)
    assert spec.layers == 43 and plain.layers == 40
    assert spec.ring_slots == 133 and plain.ring_slots == 128
    assert spec.layout_id.endswith(":spec=5")
    assert spec.state_bytes > plain.state_bytes
    # The PAGE side is the same; only STATE carries the draft rings and slack.
    assert spec.paged_bytes == plain.paged_bytes


def test_builder_must_hold_a_full_verify():
    hf = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=3)
    builder = SimpleNamespace(
        geometry=_small_geometry(5), max_bs=64, max_num_batched_tokens=64 * W
    )
    bridge._check_dspark_geometry(builder, SimpleNamespace(hf_config=hf))
    builder.max_num_batched_tokens = 64 * W - 1
    with pytest.raises(RuntimeError, match="max_num_batched_tokens"):
        bridge._check_dspark_geometry(builder, SimpleNamespace(hf_config=hf))
    builder = SimpleNamespace(
        geometry=_small_geometry(0), max_bs=1, max_num_batched_tokens=64
    )
    with pytest.raises(RuntimeError, match="speculative tokens"):
        bridge._check_dspark_geometry(builder, SimpleNamespace(hf_config=hf))


# ---------------------------------------------------------------------------
# Draft block inputs and forward context
# ---------------------------------------------------------------------------

NOISE = 99


# ---------------------------------------------------------------------------
# Markov head adapter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Draft wrapper contract (no model construction)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Target surface: lm_head view, aux taps, in-forward context write
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, residual):
        self.residual = residual

    def settle(self):
        return self


class _ToyBlock(torch.nn.Module):
    def forward(self, state):
        return _State(state.residual + 1)


class _ToyBackbone(torch.nn.Module):
    def __init__(self, layers=5, vocab=12, hidden=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(_ToyBlock() for _ in range(layers))
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Module()
        self.head.weight = torch.nn.Parameter(torch.randn(vocab, hidden))
        self.head.tp_size, self.head.num_embeddings = 1, vocab
        self.head.num_embeddings_per_partition = vocab

    @property
    def model(self):
        return self

    def forward(self, state):
        for layer in self.layers:
            state = layer(state)
        return state


def _draft(engram=(0,)):
    """The ATOM draft as the target surface uses it: native
    ``target_aux_capture_spec`` over its config's Engram layers, and the
    context write (recorded)."""
    from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

    class _Draft:
        _target_layer_input = staticmethod(DeepseekV41DSpark._target_layer_input)
        target_aux_capture_spec = DeepseekV41DSpark.target_aux_capture_spec

        def __init__(self):
            self.config = SimpleNamespace(engram_layer_ids=list(engram))
            self.writes = []

        def write_context_kv(self, taps, positions):
            # ATOM takes the taps as a sequence; the target hands one block.
            (aux,) = taps
            self.writes.append((aux, positions))

    return _Draft()


class _ToyWrapper(torch.nn.Module):
    def __init__(self, ids=(1, 2, 3)):
        super().__init__()
        self.model = _ToyBackbone()
        self.atom_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                dspark_target_layer_ids=list(ids),
                engram_layer_ids=[0],
                hidden_size=4,
            )
        )
        self.capture_aux_hidden_states = False
        self._v41_dspark = None


@pytest.fixture
def target_forward(monkeypatch):
    """ATOM's forward context as a target forward sees it."""
    from atom.spec_decode import drafter

    context = SimpleNamespace(
        is_draft=False, ubatch_token_order=None, ubatch_token_offset=0
    )
    monkeypatch.setattr(
        drafter, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    return context


@needs_sglang
@needs_native_dspark
class TestTargetSurface:
    def _surface(self, monkeypatch, wrapper=None, draft=True):
        from atom.plugin.sglang import deepseek_v41_dspark as dspark

        monkeypatch.setattr(
            dspark, "install_deepseek_v41_dspark_commit_patch", lambda: None
        )
        wrapper = wrapper or _ToyWrapper()
        surface = dspark.install_v41_dspark_target_surface(wrapper)
        if draft:
            # SGLang's DSpark worker attaches the draft before the taps.
            surface.attach_draft(_draft())
        return wrapper, surface

    def test_lm_head_view_shares_atom_weights(self, monkeypatch):
        wrapper, surface = self._surface(monkeypatch)
        view = wrapper.lm_head
        assert not isinstance(view, torch.nn.Module)
        assert view.weight is wrapper.model.head.weight
        assert view.atom_backbone is wrapper.model
        assert view.atom_wrapper is wrapper
        assert wrapper.get_input_embeddings() is wrapper.model.embed
        # Not registered twice: the wrapper's parameters are the model's.
        assert len(list(wrapper.parameters())) == len(list(wrapper.model.parameters()))
        # The view does not keep the wrapper alive.
        ref = weakref.ref(wrapper)
        del wrapper, surface, view
        assert ref() is None

    def test_no_spec_wrapper_keeps_its_surface(self):
        from atom.plugin.sglang.models.base_model_wrapper import (
            _AtomCausalLMBaseForSglang,
        )

        cls = _AtomCausalLMBaseForSglang
        for name in (
            "lm_head",
            "get_input_embeddings",
            "set_dspark_layers_to_capture",
            "attach_dspark_draft",
        ):
            assert not hasattr(cls, name), name

    def test_taps_capture_layer_inputs(self, monkeypatch, target_forward):
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        surface.set_layers_to_capture((1, 2, 3))  # idempotent
        assert wrapper.capture_aux_hidden_states
        residual = torch.randn(1, 6, 2, 4)  # [1, N, hc, H]
        surface.begin_forward()
        wrapper.model(_State(residual))
        aux = surface.take_aux(6)
        # layer k's input is the residual after k blocks (+1 each), hc-mean.
        expected = torch.cat(
            [(residual + k).mean(dim=-2).reshape(6, 4) for k in (1, 2, 3)], dim=-1
        )
        torch.testing.assert_close(aux, expected)

    def test_draft_forwards_do_not_tap(self, monkeypatch, target_forward):
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        target_forward.is_draft = True
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 6, 2, 4)))
        with pytest.raises(RuntimeError, match="did not fire"):
            surface.take_aux(6)

    @pytest.mark.parametrize(
        ("ids", "match"),
        [([1, 2], "taps layers"), ([0, 1, 2], "taps layers")],
    )
    def test_non_checkpoint_taps_are_refused(self, monkeypatch, ids, match):
        _, surface = self._surface(monkeypatch)
        with pytest.raises(ValueError, match=match):
            surface.set_layers_to_capture(ids)

    def test_engram_taps_are_refused(self, monkeypatch):
        _, surface = self._surface(monkeypatch, _ToyWrapper(ids=(0, 1, 2)))
        with pytest.raises(ValueError, match="Engram"):
            surface.set_layers_to_capture([0, 1, 2])

    def test_taps_need_the_draft(self, monkeypatch):
        _, surface = self._surface(monkeypatch, draft=False)
        with pytest.raises(RuntimeError, match="before the draft was attached"):
            surface.set_layers_to_capture([1, 2, 3])

    def test_forward_before_capture_layers_is_refused(self, monkeypatch):
        _, surface = self._surface(monkeypatch)
        with pytest.raises(RuntimeError, match="capture layers"):
            surface.begin_forward()

    def _finish(self, monkeypatch, surface, *, rows, mode, bs, width=W, step=None):
        import atom.utils.forward_context as fc

        step = step or SimpleNamespace(running_tokens=rows, replay=None, decoder=None)
        metadata = SimpleNamespace(step=step, served_by_round=0)
        monkeypatch.setattr(
            fc, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        fb = SimpleNamespace(
            forward_mode=mode,
            batch_size=bs,
            spec_info=SimpleNamespace(draft_token_num=width),
        )
        return surface.finish_forward(torch.zeros(rows, 4), torch.arange(rows), fb)

    def test_finish_writes_the_draft_context(self, monkeypatch, target_forward):
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        writes = surface.draft.writes
        rows = 2 * W
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, rows, 2, 4)))
        aux = self._finish(
            monkeypatch, surface, rows=rows, mode=ForwardMode.TARGET_VERIFY, bs=2
        )
        assert aux.shape == (rows, 12)
        assert len(writes) == 1 and writes[0][0] is aux
        assert writes[0][1].tolist() == list(range(rows))

    def test_finish_refusals(self, monkeypatch, target_forward):
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 7, 2, 4)))
        with pytest.raises(RuntimeError, match="7 rows for 2 requests"):
            self._finish(
                monkeypatch, surface, rows=7, mode=ForwardMode.TARGET_VERIFY, bs=2
            )
        with pytest.raises(RuntimeError, match="already attached"):
            surface.attach_draft(SimpleNamespace())
        with pytest.raises(RuntimeError, match="bounded replay"):
            self._finish(
                monkeypatch,
                surface,
                rows=6,
                mode=ForwardMode.EXTEND,
                bs=1,
                step=SimpleNamespace(running_tokens=6, replay=object(), decoder=None),
            )
        with pytest.raises(RuntimeError, match="bounded replay"):
            # A decoder SWA tail (`_publish_decoder`) leaves rows uncomputed.
            self._finish(
                monkeypatch,
                surface,
                rows=6,
                mode=ForwardMode.EXTEND,
                bs=1,
                step=SimpleNamespace(running_tokens=6, replay=None, decoder=object()),
            )
        with pytest.raises(RuntimeError, match="no plain DECODE"):
            self._finish(monkeypatch, surface, rows=1, mode=ForwardMode.DECODE, bs=1)
