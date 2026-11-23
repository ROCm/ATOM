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

import copy
import json
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    DEEPSEEK_V41_ARCH,
    V41_DSPARK_DRAFT_ARCH,
    build_deepseek_v41_forward_metadata,
    build_v41_scheduled_batch,
    commit_deepseek_v41_dspark_verify,
    dspark_commit_anchors,
    is_v41_dspark_atom_config,
    refuse_unsupported_sglang_features,
    v41_dspark_capability_error,
)
from atom.plugin.sglang.runtime.model_arch import (
    MODEL_ADAPTER_SPECS,
    resolve_model_arch_spec,
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
    monkeypatch.delenv("ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC", raising=False)
    monkeypatch.delenv("SGLANG_DSPARK_EMBED_IN_GRAPH", raising=False)


def test_embed_in_graph_off_is_refused(monkeypatch):
    monkeypatch.setenv("SGLANG_DSPARK_EMBED_IN_GRAPH", "0")
    spec = SimpleNamespace(speculative_use_block_verification=False)
    features = SimpleNamespace(
        enable_return_hidden_states=False, return_hidden_states_mode=None
    )
    refused = bridge._dspark_launch_refusals(
        spec,
        SimpleNamespace(graph=_graph(), features=features),
        SimpleNamespace(max_running_requests=64),
    )
    assert any("SGLANG_DSPARK_EMBED_IN_GRAPH" in r for r in refused)


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


class TestAdapterRouting:
    def test_draft_config_routes_to_the_draft_spec(self):
        from atom.config import SpeculativeConfig

        target = _atom_text_v41()
        draft = copy.deepcopy(target)
        SpeculativeConfig.hf_config_override(draft)
        assert draft.model_type == "deepseek_v41_dspark"
        assert draft.architectures == [V41_DSPARK_DRAFT_ARCH]
        arch, spec = resolve_model_arch_spec(draft)
        assert arch == V41_DSPARK_DRAFT_ARCH
        assert spec is not MODEL_ADAPTER_SPECS[DEEPSEEK_V41_ARCH]
        with pytest.raises(RuntimeError, match="build their metadata"):
            spec.build_forward_metadata(None, SimpleNamespace(), None)
        # The target's config is a different object and stays the target.
        assert resolve_model_arch_spec(target)[0] == DEEPSEEK_V41_ARCH
        assert target.model_type == "deepseek_v41_text"

    def test_target_and_v4_keep_their_specs(self):
        assert resolve_model_arch_spec(_v41())[0] == DEEPSEEK_V41_ARCH
        v4 = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4"
        )
        arch, spec = resolve_model_arch_spec(v4)
        assert arch == "DeepseekV4ForCausalLM"
        assert (
            spec.build_forward_metadata.__name__
            == "_build_deepseek_v4_forward_metadata"
        )
        nextn = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLMNextN"], model_type="deepseek_v4_mtp"
        )
        assert resolve_model_arch_spec(nextn)[0] == "DeepseekV4ForCausalLMNextN"

    @pytest.mark.skipif(not HAVE_CHECKPOINT, reason="V4.1-Flash checkpoint absent")
    def test_real_checkpoint_configs(self):
        from atom.config import SpeculativeConfig, get_hf_config

        target = get_hf_config(FLASH_CHECKPOINT, trust_remote_code=True)
        assert v41_dspark_capability_error(target) is None
        draft = copy.deepcopy(target)
        SpeculativeConfig.hf_config_override(draft, FLASH_CHECKPOINT)
        assert resolve_model_arch_spec(draft)[0] == V41_DSPARK_DRAFT_ARCH
        assert resolve_model_arch_spec(target)[0] == DEEPSEEK_V41_ARCH


# ---------------------------------------------------------------------------
# Speculative config
# ---------------------------------------------------------------------------


# The server args `_server_args` built last: SGLang resolves the speculative
# settings into its spec bag, which `spec_config_env` serves from here.
_LAST_SERVER_ARGS = []


@pytest.fixture
def spec_config_env(monkeypatch):
    """The builder's inputs without a checkpoint: ATOM's config loader returns
    a V4.1 text config, SGLang's spec bag holds the launch's settings."""
    from sglang.srt import runtime_context

    import atom.config as atom_config

    monkeypatch.setattr(atom_config, "get_hf_config", lambda *a, **k: _atom_text_v41())
    monkeypatch.setattr(runtime_context, "get_spec", lambda: _LAST_SERVER_ARGS[-1])
    _LAST_SERVER_ARGS.clear()


def _server_args(**overrides):
    args = {
        "speculative_algorithm": "DSPARK",
        "speculative_num_draft_tokens": 6,
        # SGLang's DSpark resolution: a bundled draft is the target checkpoint.
        "speculative_draft_model_path": "/models/v41",
        "model_path": "/models/v41",
    }
    args.update(overrides)
    _LAST_SERVER_ARGS.append(SimpleNamespace(**args))
    return _LAST_SERVER_ARGS[-1]


class TestSpeculativeConfig:
    @needs_native_dspark
    def test_dspark_on_v41(self, spec_config_env):
        from atom.plugin.config import _build_atom_speculative_config_from_sglang

        hf = _v41()
        before = copy.deepcopy(vars(hf))
        spec = _build_atom_speculative_config_from_sglang(_server_args(), hf)
        assert spec.method == "dspark"
        assert spec.num_speculative_tokens == 5
        assert spec.model == "/models/v41"
        assert spec.use_dspark()
        assert spec.draft_model_hf_config.model_type == "deepseek_v41_dspark"
        # SGLang's config is never what the override mutates.
        assert vars(hf) == before

    @needs_native_dspark
    def test_same_draft_path_is_accepted(self, spec_config_env):
        from atom.plugin.config import _build_atom_speculative_config_from_sglang

        args = _server_args(speculative_draft_model_path="/models/v41")
        assert _build_atom_speculative_config_from_sglang(args, _v41()) is not None

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"speculative_num_draft_tokens": 8}, "num-draft-tokens 6"),
            ({"speculative_draft_model_path": "/other"}, "draft from the target"),
        ],
    )
    def test_refusals(self, spec_config_env, overrides, match):
        from atom.plugin.config import _build_atom_speculative_config_from_sglang

        with pytest.raises(ValueError, match=match):
            _build_atom_speculative_config_from_sglang(
                _server_args(**overrides), _v41()
            )

    def test_checkpoint_without_dspark_is_refused(self, spec_config_env):
        from atom.plugin.config import _build_atom_speculative_config_from_sglang

        with pytest.raises(ValueError, match="not servable"):
            _build_atom_speculative_config_from_sglang(
                _server_args(), _v41(dspark_block_size=None)
            )

    def test_no_speculation_and_other_models_unchanged(self, spec_config_env):
        from atom.plugin.config import _build_atom_speculative_config_from_sglang

        args = _server_args(speculative_algorithm=None)
        assert _build_atom_speculative_config_from_sglang(args, _v41()) is None
        qwen3 = SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
        assert _build_atom_speculative_config_from_sglang(_server_args(), qwen3) is None


# ---------------------------------------------------------------------------
# Launch refusals
# ---------------------------------------------------------------------------


def _graph(decode="disabled", prefill="disabled"):
    return SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=decode, bs=None),
            prefill=SimpleNamespace(backend=prefill),
        ),
        disable_cuda_graph_padding=False,
    )


@pytest.fixture
def dspark_launch(monkeypatch):
    """SGLang's resolved config bags for a DSpark launch."""
    from sglang.srt import runtime_context

    state = SimpleNamespace(
        bags={
            "memory": SimpleNamespace(
                disable_radix_cache=True,
                enable_hierarchical_cache=False,
                enable_hisparse=False,
                enable_session_radix_cache=False,
                radix_cache_backend=None,
                enable_unified_memory=False,
                enable_lmcache=False,
                enable_flexkv=False,
                enable_unified_cache_external_linker=False,
            ),
            "exec": SimpleNamespace(
                features=SimpleNamespace(
                    enable_encoder_swa_bounded_replay=False,
                    enable_decoder_swa_bounded_replay=False,
                    enable_memory_saver=False,
                    enable_watermark=False,
                    enable_return_hidden_states=False,
                    return_hidden_states_mode=None,
                ),
                moe=SimpleNamespace(moe_a2a_backend="none"),
                graph=_graph(),
            ),
            "spec": SimpleNamespace(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                speculative_draft_model_path="/models/v41",
                speculative_use_block_verification=False,
            ),
            "disagg": SimpleNamespace(
                disaggregation_mode="null", language_model_only=True
            ),
            "parallel": SimpleNamespace(
                enable_dp_attention=False,
                pp_size=1,
                dcp_size=1,
                moe_dense_tp_size=None,
            ),
            "schedule": SimpleNamespace(
                disable_hybrid_swa_memory=False,
                page_size=PAGE,
                chunked_prefill_size=8192,
                max_running_requests=64,
                disable_overlap_schedule=True,
                enable_mixed_chunk=False,
            ),
            "model": SimpleNamespace(model_path="/models/v41"),
        },
    )
    for name in ("memory", "exec", "spec", "disagg", "parallel", "schedule", "model"):
        monkeypatch.setattr(
            runtime_context, f"get_{name}", lambda name=name: state.bags[name]
        )
    return state


class TestLaunchRefusals:
    def test_supported_dspark_launch_passes(self, dspark_launch, monkeypatch):
        refuse_unsupported_sglang_features()
        monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
        monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "1.0")
        refuse_unsupported_sglang_features()
        monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "-1")
        refuse_unsupported_sglang_features()
        # SGLang reads an unparsable value as its default, -1 (off).
        monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "off")
        with pytest.warns(UserWarning, match="SGLANG_SIMULATE_ACC_LEN"):
            refuse_unsupported_sglang_features()
        # A fixed mean accept length is a perf-run opt-in.
        monkeypatch.setenv("SGLANG_SIMULATE_ACC_LEN", "3.47")
        with pytest.raises(ValueError, match="SIMULATE_ACC_LEN=3.47"):
            refuse_unsupported_sglang_features()
        monkeypatch.setenv("ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC", "true")
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("speculative_use_block_verification", True, "block verification"),
        ],
    )
    def test_spec_refusals(self, dspark_launch, field, value, match):
        setattr(dspark_launch.bags["spec"], field, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("env", "value", "match"),
        [
            ("SGLANG_RAGGED_VERIFY_MODE", "compact", "must be static"),
            ("SGLANG_RAGGED_VERIFY_MODE", "cap_accept", "must be static"),
            ("SGLANG_SIMULATE_ACC_LEN", "2.5", "SIMULATE_ACC_LEN=2.5"),
        ],
    )
    def test_env_refusals(self, dspark_launch, monkeypatch, env, value, match):
        monkeypatch.setenv(env, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    def test_prefill_graph_is_refused(self, dspark_launch):
        dspark_launch.bags["exec"].graph = _graph(prefill="full")
        with pytest.raises(ValueError, match="prefill CUDA graph"):
            refuse_unsupported_sglang_features()

    def test_full_decode_graphs_pass(self, dspark_launch, monkeypatch):
        monkeypatch.delenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", raising=False)
        graph = _graph(decode="full")
        graph.cuda_graph_config.decode.bs = [1, 2, 4, 8, 16, 32]
        dspark_launch.bags["exec"].graph = graph
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("breakable", "breakable decode CUDA graph backend"),
            ("padding", "disable-cuda-graph-padding"),
            ("glue", "METADATA_GLUE_GRAPH with decode CUDA graphs"),
        ],
    )
    def test_decode_graph_refusals(self, dspark_launch, monkeypatch, case, match):
        monkeypatch.delenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", raising=False)
        graph = _graph(decode="breakable" if case == "breakable" else "full")
        graph.cuda_graph_config.decode.bs = [1, 2, 64]
        graph.disable_cuda_graph_padding = case == "padding"
        dspark_launch.bags["exec"].graph = graph
        if case == "glue":
            monkeypatch.setenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", "1")
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    def test_decode_graphs_below_the_running_batch_warn(
        self, dspark_launch, monkeypatch, caplog
    ):
        # Not refused: rounds above the largest bucket run eagerly.
        monkeypatch.delenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", raising=False)
        graph = _graph(decode="full")
        dspark_launch.bags["exec"].graph = graph
        for sizes, warned in (([1, 2, 4, 8, 16, 32], True), ([1, 2, 64], False)):
            graph.cuda_graph_config.decode.bs = sizes
            caplog.clear()
            with caplog.at_level("WARNING", logger="atom"):
                refuse_unsupported_sglang_features()
            lines = [
                r.getMessage()
                for r in caplog.records
                if "decode graphs stop at bs" in r.getMessage()
            ]
            if warned:
                (line,) = lines
                assert "bs 32 below --max-running-requests 64" in line
            else:
                assert not lines

    @pytest.mark.parametrize(
        ("enable", "mode", "refused"),
        [
            (True, None, True),
            (True, "full", True),
            (False, "full", True),
            # SGLang's "last" (enable is set with it) returns the last row,
            # which the replay computes.
            (True, "last", False),
        ],
    )
    def test_full_hidden_state_returns_are_refused_under_replay(
        self, dspark_launch, monkeypatch, enable, mode, refused
    ):
        features = dspark_launch.bags["exec"].features
        features.enable_return_hidden_states = enable
        features.return_hidden_states_mode = mode
        if refused:
            with pytest.raises(ValueError, match="full hidden-state returns"):
                refuse_unsupported_sglang_features()
        else:
            refuse_unsupported_sglang_features()
        # Without the replay every row is computed.
        monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", "0")
        refuse_unsupported_sglang_features()

    def test_decoder_replay_flag_is_served(self, dspark_launch, monkeypatch):
        # DSpark runs with decoder replay, as natively: the flag restates the
        # default and is refused only with ATOM_SGLANG_V41_DECODER_REPLAY off.
        dspark_launch.bags["exec"].features.enable_decoder_swa_bounded_replay = True
        refuse_unsupported_sglang_features()
        monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", "0")
        with pytest.raises(ValueError, match="DECODER_REPLAY off"):
            refuse_unsupported_sglang_features()

    def test_overlap_schedule_is_served(self, dspark_launch):
        # A DSpark round (draft, verify, accept, commit) is one worker call
        # under SGLang 0.5.22's overlap loop: nothing to refuse.
        dspark_launch.bags["schedule"].disable_overlap_schedule = False
        refuse_unsupported_sglang_features()

    def test_radix_is_served_and_dp_stays_refused(self, dspark_launch, monkeypatch):
        # DSpark runs under the radix replay (`deepseek_v41_radix`, "DSpark").
        monkeypatch.delenv("ATOM_SGLANG_V41_RADIX", raising=False)
        dspark_launch.bags["memory"].disable_radix_cache = False
        refuse_unsupported_sglang_features()
        dspark_launch.bags["parallel"].enable_dp_attention = True
        with pytest.raises(ValueError, match="DP attention"):
            refuse_unsupported_sglang_features()

    def test_launch_policy_keeps_radix_under_dspark(self, monkeypatch):
        from atom.plugin.sglang import deepseek_v41_radix as radix

        monkeypatch.setattr(radix, "_POLICY_WARNED", False)
        monkeypatch.setattr(radix, "_resolving_view", lambda server_args: server_args)
        monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "atom.plugin.sglang.models")
        monkeypatch.delenv("ATOM_SGLANG_V41_RADIX", raising=False)
        args = SimpleNamespace(
            speculative_algorithm="DSPARK", disable_radix_cache=False
        )
        config = SimpleNamespace(model_type="deepseek_v41")
        assert radix.v41_radix_launch_override(args, config) == {}
        # ATOM_SGLANG_V41_RADIX=0 turns it off, with or without speculation.
        monkeypatch.setenv("ATOM_SGLANG_V41_RADIX", "0")
        assert radix.v41_radix_launch_override(args, config) == {
            "disable_radix_cache": True
        }


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
        "start_layer": 0,
        "end_layer": 3,
        "num_req_slots": 4,
        "is_draft_worker": True,
    }
    kwargs.update(overrides)
    return bridge._make_proxy_pool_cls()(**kwargs)


@needs_sglang
class TestDraftPool:
    def _builder(self, monkeypatch, speculative_tokens=5):
        cache = SimpleNamespace(num_pages=5, num_slots=4, pending=None)
        builder = SimpleNamespace(
            geometry=_small_geometry(speculative_tokens), cache=cache
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        return builder

    def test_binds_the_target_arena_and_allocates_nothing(self, monkeypatch):
        builder = self._builder(monkeypatch)
        pool = _proxy()
        assert pool.is_atom_v41_proxy_pool and pool.is_atom_v41_draft_proxy
        assert pool.atom_cache is builder.cache
        assert pool.get_kv_size_bytes() == 0
        assert pool.mem_usage == 0
        assert (pool.num_pages, pool.num_slots) == (5, 4)
        assert pool.needs_paged_swa_allocator is False
        assert pool.request_window is None
        # SGLang's injector must hand the hidden to the draft model.
        assert not hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope")
        pool.register_mapping(torch.zeros(3))
        assert pool.full_to_swa_index_mapping is None

    def test_backend_roles_must_match_pools(self, monkeypatch):
        from atom.plugin.sglang.attention_backend.deepseek_v41_backend import (
            ATOMDeepseekV41BackendForSgl,
        )

        self._builder(monkeypatch)
        draft_pool = _proxy()
        runner = SimpleNamespace(
            device="cpu",
            token_to_kv_pool=draft_pool,
            req_to_token_pool=object(),
            is_draft_worker=True,
        )
        assert ATOMDeepseekV41BackendForSgl(runner).role == "draft"
        runner.is_draft_worker = False
        with pytest.raises(RuntimeError, match="draft role"):
            ATOMDeepseekV41BackendForSgl(runner)
        target_pool = SimpleNamespace(
            is_atom_v41_proxy_pool=True, is_atom_v41_draft_proxy=False
        )
        runner = SimpleNamespace(
            device="cpu",
            token_to_kv_pool=target_pool,
            req_to_token_pool=object(),
            is_draft_worker=True,
        )
        with pytest.raises(RuntimeError, match="target role"):
            ATOMDeepseekV41BackendForSgl(runner)


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


def _draft_batch(prefix, slots, width=5, mode=None, noise=NOISE):
    bs = len(prefix)
    seq = torch.tensor(prefix, dtype=torch.int64)
    positions = (seq[:, None] + torch.arange(width)).reshape(-1)
    ids = torch.full((bs, width), noise, dtype=torch.int64)
    ids[:, 0] = torch.arange(10, 10 + bs)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY if mode is None else mode,
        batch_size=bs,
        seq_lens=seq,
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=None,
        spec_info=SimpleNamespace(draft_token_num=width),
    )
    return fb, ids.reshape(-1), positions


@needs_sglang
class TestDraftBlock:
    def test_anchor_is_the_bonus_one_row_back(self):
        from atom.plugin.sglang.deepseek_v41_dspark import draft_block_inputs

        fb, ids, positions = _draft_batch([10, 300], [2, 5])
        block = draft_block_inputs(fb, ids, positions, noise_token_id=NOISE)
        assert block.anchor_ids.tolist() == [10, 11]
        # native: anchors locate the last processed target row, the block
        # runs at anchor + 1 .. anchor + 5 == p .. p + 4.
        assert block.anchor_positions.tolist() == [9, 299]
        assert block.slots.tolist() == [2, 5] and block.slots.dtype == torch.int32
        assert (block.bs, block.width) == (2, 5)

    def test_only_the_block_runs(self):
        from atom.plugin.sglang.deepseek_v41_dspark import draft_block_inputs

        fb, ids, positions = _draft_batch([10, 300], [2, 5])
        fb.forward_mode = ForwardMode.EXTEND
        with pytest.raises(NotImplementedError, match="only its block"):
            draft_block_inputs(fb, ids, positions, noise_token_id=NOISE)

    def test_forward_context_is_a_draft_pass(self):
        from atom.plugin.sglang.deepseek_v41_dspark import (
            draft_block_inputs,
            v41_dspark_draft_forward_context,
        )
        from atom.utils.forward_context import AttnState, get_forward_context

        fb, ids, positions = _draft_batch([10, 300], [2, 5])
        block = draft_block_inputs(fb, ids, positions, noise_token_id=NOISE)
        cache = object()
        atom_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with v41_dspark_draft_forward_context(atom_config, cache=cache, block=block):
            forward = get_forward_context()
            metadata, context = forward.attn_metadata, forward.context
            assert metadata.cache is cache and metadata.step is None
            assert metadata.state_slot_out is block.slots
            assert metadata.max_seqlen_q == 5 and metadata.state == AttnState.DECODE
            assert not metadata.served_by_round
            assert context.is_draft and not context.is_prefill
            assert (context.scheduled_bs, context.running_tokens) == (2, 10)
        # Reset on exit: no draft metadata outlives the block.
        assert get_forward_context().context is None


# ---------------------------------------------------------------------------
# Markov head adapter
# ---------------------------------------------------------------------------


@needs_sglang
def test_markov_adapter_matches_atom_head(monkeypatch):
    from atom.models.dspark_draft import DSparkMarkovHead
    from atom.plugin.sglang.models.deepseek_v41_dspark_wrapper import (
        ATOMV41MarkovHeadAdapter,
    )

    torch.manual_seed(0)
    head = DSparkMarkovHead(50, 8)
    head.fused_sample = False
    with torch.no_grad():
        head.markov_w1.weight.normal_()
        head.markov_w2.weight.normal_()
    adapter = ATOMV41MarkovHeadAdapter(head)
    # ATOM's weights are not re-registered on the adapter.
    assert list(adapter.parameters()) == []
    assert adapter.vocab_size == 50 and adapter.vocab_shard is None
    assert adapter.greedy_w2_weight() is None
    assert adapter.step_logits_dtype(torch.bfloat16) == torch.float32
    adapter.refresh_w2_cache()  # as `load_weights` does
    prev = torch.tensor([3, 17, 42])
    bias, state = adapter.step_bias(prev, hidden_states=None, state="s")
    assert state == "s" and bias.dtype == torch.float32
    assert torch.equal(bias, head(prev)[0])
    assert torch.equal(adapter.get_prev_embeddings(prev), head.markov_w1(prev))
    base = torch.randn(3, 50)
    out = torch.empty(3, dtype=torch.int64)
    head.sample_next(prev, base, out)
    assert torch.equal((base + bias).argmax(-1), out)


def test_markov_w2_cache_is_the_cast_and_refreshes_in_place():
    from atom.models.dspark_draft import DSparkMarkovHead
    from atom.plugin.sglang.models.deepseek_v41_dspark_wrapper import (
        ATOMV41MarkovHeadAdapter,
    )

    torch.manual_seed(0)
    head = DSparkMarkovHead(50, 8)
    with torch.no_grad():
        head.markov_w1.weight.normal_()
        head.markov_w2.weight.copy_(torch.randn(50, 8).bfloat16().float())
    head.to(torch.bfloat16)
    adapter = ATOMV41MarkovHeadAdapter(head)
    cache = adapter.refresh_w2_cache()
    assert cache.dtype == torch.float32 and cache.is_contiguous()
    assert torch.equal(cache, head.markov_w2.weight.float())
    prev = torch.tensor([3, 17, 42])
    bias, _ = adapter.step_bias(prev, hidden_states=None, state=None)
    # Bit-identical to ATOM's per-step cast.
    assert torch.equal(bias, head(prev)[0])
    # A reload in place (the loader copies into the parameter) refreshes the
    # same storage: a captured draft graph keeps reading valid values.
    with torch.no_grad():
        head.markov_w2.weight.mul_(2)
    again = adapter.refresh_w2_cache()
    assert again is cache and torch.equal(again, head.markov_w2.weight.float())
    # A replaced parameter is noticed on the next eager step.
    head.markov_w2.weight = torch.nn.Parameter(torch.randn(50, 8).bfloat16())
    bias, _ = adapter.step_bias(prev, hidden_states=None, state=None)
    assert torch.equal(bias, head(prev)[0])
    assert adapter.__dict__["_w2_src"] == head.markov_w2.weight.data_ptr()
    assert list(adapter.parameters()) == []


# ---------------------------------------------------------------------------
# Draft wrapper contract (no model construction)
# ---------------------------------------------------------------------------


def _bare_wrapper():
    from atom.plugin.sglang.models.deepseek_v41_dspark_wrapper import (
        DeepseekV4ForCausalLMDSpark,
    )

    wrapper = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
    torch.nn.Module.__init__(wrapper)
    wrapper.num_context_features = 3
    wrapper.hidden_size = 4
    wrapper.vocab_size = 12
    wrapper.logits_mup_width_multiplier = None
    return wrapper


@needs_sglang
class TestDraftWrapper:
    def test_v4_dspark_is_refused(self):
        from atom.plugin.sglang.models.deepseek_v41_dspark_wrapper import (
            DeepseekV4ForCausalLMDSpark,
        )

        v4 = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLMDSpark"],
            model_type="deepseek_v4",
            vocab_size=10,
        )
        with pytest.raises(NotImplementedError, match="V4.1 only"):
            DeepseekV4ForCausalLMDSpark(v4)

    def test_sglang_contract_flags(self):
        from atom.plugin.sglang.models.deepseek_v41_dspark_wrapper import (
            DeepseekV4ForCausalLMDSpark,
            EntryClass,
        )

        cls = DeepseekV4ForCausalLMDSpark
        assert EntryClass == [cls]
        assert cls.sglang_skip_quant_config and cls.sample_from_anchor
        assert not cls.uses_own_vocab_modules
        assert not cls.supports_pre_gather_target_hidden_projection
        assert cls.confidence_head is None
        # SGLang embeds nothing when the draft defines forward_embed.
        assert hasattr(cls, "forward_embed")

    def test_injection_is_a_no_op(self):
        # The target forward wrote the rows already; SGLang's injection call
        # keeps its keyword contract.
        wrapper = _bare_wrapper()
        pool = SimpleNamespace(is_atom_v41_draft_proxy=True)
        hidden = torch.zeros(7, 12)
        positions = torch.arange(7)
        wrapper.write_target_hidden_kv(
            target_hidden=hidden, pool=pool, positions=positions
        )
        with pytest.raises(TypeError):
            wrapper.write_target_hidden_kv(
                main_hidden=hidden, swa_loc=positions, positions=positions, pool=pool
            )

    def test_forward_takes_token_ids_only(self):
        wrapper = _bare_wrapper()
        with pytest.raises(NotImplementedError, match="token ids only"):
            wrapper.forward(
                torch.zeros(5, dtype=torch.int64),
                torch.arange(5),
                SimpleNamespace(),
                input_embeds=torch.zeros(5, 4),
            )
        with pytest.raises(NotImplementedError, match="inside block_backbone"):
            wrapper.forward_embed(torch.zeros(5, dtype=torch.int64))
        for method in (
            "set_dspark_layers_to_capture",
            "prune_to_ctx_kv_injection",
            "project_target_hidden",
        ):
            with pytest.raises(NotImplementedError):
                getattr(wrapper, method)(
                    *([None] if method != "prune_to_ctx_kv_injection" else [])
                )

    def test_base_logits_are_the_shared_head_full_vocab(self):
        wrapper = _bare_wrapper()
        head = SimpleNamespace(get_logits=lambda x: x @ torch.ones(4, 12))
        wrapper.__dict__["model"] = SimpleNamespace(head=head)
        wrapper.target_atom_config = None
        x = torch.randn(10, 4)
        logits, features = wrapper.compute_base_logits(x)
        assert logits.shape == (10, 12) and features is x

    def test_attach_needs_the_atom_target_view(self):
        wrapper = _bare_wrapper()
        with pytest.raises(RuntimeError, match="lm_head view"):
            wrapper.attach_shared_modules(embed_tokens=object(), lm_head=object())


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
    # The runtime model's decoder SWA bounded replay switch (off here).
    replay = False

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


def _window_rows(lengths, window):
    """Each request's last ``window`` forward rows, in request order: what a
    replay's tail holds and the draft's first block reads."""
    ends = np.cumsum(lengths)
    return np.concatenate(
        [np.arange(end - min(n, window), end) for n, end in zip(lengths, ends)]
    )


def _plan(lengths, *, window, running_bs):
    """A `ReplayPlan`'s row fields, laid out by native's own
    `tail_spans` / `lay_out_tail` for prompts of ``lengths`` rows (every
    request finishing, as the plugin stages them)."""
    from atom.models.deepseek_v41.bounded_replay import (
        lay_out_tail,
        tail_spans,
        tail_width,
    )

    offsets = np.cumsum([0, *lengths[:-1]])
    step = SimpleNamespace(
        requests=[
            SimpleNamespace(offset=int(o), length=int(n), end=int(n))
            for o, n in zip(offsets, lengths)
        ],
        request_finals=[True] * len(lengths),
    )
    picks = tail_spans(step, window)
    width = tail_width(running_bs, window, 1 << 20)
    layout = lay_out_tail(picks, running_bs, width)
    return SimpleNamespace(
        tail=SimpleNamespace(running_tokens=width, requests=picks),
        tail_rows=torch.as_tensor(layout.rows),
        filled=layout.filled,
    )


class TestReplayRowSelection:
    """Which forward rows a replayed DSpark prefill's draft context comes
    from: each request's last window rows, at their own forward rows."""

    @pytest.mark.parametrize(
        ("lengths", "window", "running_bs"),
        [
            ([300], 128, 1),
            ([5, 300, 128, 129], 128, 4),
            ([5, 300], 128, 4),  # padding requests in the tail
            ([1], 128, 1),
        ],
    )
    def test_tail_is_each_requests_last_window(self, lengths, window, running_bs):
        plan = _plan(lengths, window=window, running_bs=running_bs)
        expected = _window_rows(lengths, window)
        assert plan.tail_rows[: plan.filled].tolist() == expected.tolist()
        assert plan.filled == sum(min(n, window) for n in lengths)

    def test_tail_aux_lands_on_its_forward_rows(self):
        from atom.plugin.sglang.deepseek_v41_dspark import tail_aux_to_forward_rows

        lengths, window = [5, 300, 129], 128
        plan = _plan(lengths, window=window, running_bs=4)
        rows = sum(lengths)
        # A replay-off forward computes every row; the tail computes the
        # same rows (here: the same values) at tail positions.
        full = torch.randn(rows, 6)
        tail_aux = torch.zeros(plan.tail.running_tokens, 6)
        tail_aux[: plan.filled] = full[plan.tail_rows[: plan.filled]]
        aux = tail_aux_to_forward_rows(tail_aux, plan, rows)
        wanted = torch.as_tensor(_window_rows(lengths, window))
        # The rows the draft reads match replay-off; nothing else is written.
        assert torch.equal(aux[wanted], full[wanted])
        others = torch.ones(rows, dtype=torch.bool)
        others[wanted] = False
        assert not aux[others].any()
        assert torch.equal(aux[plan.tail_rows][: plan.filled], tail_aux[: plan.filled])

    def test_tail_aux_shape_is_checked(self):
        from atom.plugin.sglang.deepseek_v41_dspark import tail_aux_to_forward_rows

        plan = _plan([300], window=128, running_bs=1)
        with pytest.raises(RuntimeError, match="tail's 128 rows"):
            tail_aux_to_forward_rows(torch.zeros(300, 6), plan, 300)
        empty = SimpleNamespace(
            tail=SimpleNamespace(running_tokens=4),
            tail_rows=torch.zeros(4, dtype=torch.int64),
            filled=0,
        )
        aux = tail_aux_to_forward_rows(torch.ones(4, 6), empty, 3)
        assert aux.shape == (3, 6) and not aux.any()


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

    def _finish_fused(self, monkeypatch, surface, fb, rows, positions):
        import atom.utils.forward_context as fc

        metadata = SimpleNamespace(
            step=SimpleNamespace(running_tokens=rows, replay=None),
            served_by_round=0,
        )
        monkeypatch.setattr(
            fc, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        return surface.finish_forward(torch.zeros(rows, 4), positions, fb)

    @staticmethod
    def _hit_forward():
        """A hit at P = 8 replaying [4, 8) (R = 4) with three own rows, beside
        a cold request of two: SGLang's five rows, ATOM's nine."""
        from atom.plugin.sglang.deepseek_v41_radix import (
            FUSED_ATTR,
            FusedReplay,
            expand_v41_forward_inputs,
        )

        fused = FusedReplay([0], [5], [4], [8], [40, 50, 60, 70], [[1, 2, 3]])
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_seq_lens_cpu=[3, 2],
            spec_info=None,
        )
        setattr(fb, FUSED_ATTR, fused)
        ids = torch.tensor([11, 12, 13, 21, 22])
        positions = torch.tensor([8, 9, 10, 0, 1])
        _, wide_positions, keep = expand_v41_forward_inputs(fb, ids, positions)
        return fb, fused, wide_positions, keep

    def test_finish_writes_a_radix_hit_replay_rows(self, monkeypatch, target_forward):
        """The draft's windows below a hit are written by the forward that
        replays it, as native ``compute_draft_kv`` writes every row of a
        forward starting a hash block below the hit; SGLang's injection then
        sees its own rows only."""
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        writes = surface.draft.writes
        fb, _, wide_positions, keep = self._hit_forward()
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 9, 2, 4)))
        aux = self._finish_fused(monkeypatch, surface, fb, 9, wide_positions)
        assert aux.shape == (9, 12)
        ((written, positions),) = writes
        assert written is aux
        assert positions.tolist() == [4, 5, 6, 7, 8, 9, 10, 0, 1]
        # The wrapper's trim (`SGLangPluginRuntime.trim_output`): SGLang's rows.
        assert keep.tolist() == [4, 5, 6, 7, 8]
        assert positions.index_select(0, keep).tolist() == [8, 9, 10, 0, 1]

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
        # A replay's taps ran on the tail: a forward-width tap is refused.
        plan = _plan([6], window=4, running_bs=1)
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 6, 2, 4)))
        with pytest.raises(RuntimeError, match="did not fire|has shape"):
            self._finish(
                monkeypatch,
                surface,
                rows=6,
                mode=ForwardMode.EXTEND,
                bs=1,
                step=SimpleNamespace(running_tokens=6, replay=plan),
            )
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 2 * W, 2, 4)))
        with pytest.raises(RuntimeError, match="verify ran a bounded replay"):
            self._finish(
                monkeypatch,
                surface,
                rows=2 * W,
                mode=ForwardMode.TARGET_VERIFY,
                bs=2,
                step=SimpleNamespace(running_tokens=2 * W, replay=plan),
            )
        with pytest.raises(RuntimeError, match="no plain DECODE"):
            self._finish(monkeypatch, surface, rows=1, mode=ForwardMode.DECODE, bs=1)

    def test_published_but_unreplayed_decoder_writes_every_row(
        self, monkeypatch, target_forward
    ):
        # `step.decoder` without `step.replay`: the model ran every row.
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        writes = surface.draft.writes
        surface.begin_forward()
        wrapper.model(_State(torch.randn(1, 6, 2, 4)))
        aux = self._finish(
            monkeypatch,
            surface,
            rows=6,
            mode=ForwardMode.EXTEND,
            bs=1,
            step=SimpleNamespace(running_tokens=6, replay=None, decoder=object()),
        )
        assert aux.shape == (6, 12) and writes[0][0] is aux

    def test_replay_writes_the_tail_rows_at_their_forward_rows(
        self, monkeypatch, target_forward
    ):
        # Two prompts of 6 and 3 rows, window 4: the tail is rows 2..5 and
        # 6..8, laid out for 2 requests (8 rows, the last one padding).
        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        writes = surface.draft.writes
        plan = _plan([6, 3], window=4, running_bs=2)
        assert plan.tail.running_tokens == 8 and plan.filled == 7
        residual = torch.randn(1, 8, 2, 4)  # the late layers ran on the tail
        surface.begin_forward()
        wrapper.model(_State(residual))
        aux = self._finish(
            monkeypatch,
            surface,
            rows=9,
            mode=ForwardMode.EXTEND,
            bs=2,
            step=SimpleNamespace(running_tokens=9, replay=plan),
        )
        tail = torch.cat(
            [(residual + k).mean(dim=-2).reshape(8, 4) for k in (1, 2, 3)], dim=-1
        )
        assert aux.shape == (9, 12)
        assert plan.tail_rows.tolist() == [2, 3, 4, 5, 6, 7, 8, 8]
        torch.testing.assert_close(aux[2:9], tail[:7])
        assert not aux[:2].any()
        # What native's write_context_kv reads under the plan: the tail, in
        # tail order, every real row its own (padding repeats the last).
        torch.testing.assert_close(aux[plan.tail_rows][:7], tail[:7])
        assert len(writes) == 1 and writes[0][0] is aux
        assert writes[0][1].tolist() == list(range(9))

    def test_radix_hit_under_decoder_replay_writes_its_widened_tail(
        self, monkeypatch, target_forward
    ):
        """DSpark + fused radix hit + decoder SWA bounded replay: the hit's
        widened span [P - R, L) is >= the window, so the step replays; the
        draft context is each request's last window rows of the widened
        forward -- for the hit [L - window, L), reaching into the replay rows
        [P - R, P) when L - P < window -- at their forward rows."""
        from atom.plugin.sglang import deepseek_v41_bridge as bridge

        wrapper, surface = self._surface(monkeypatch)
        surface.set_layers_to_capture([1, 2, 3])
        writes = surface.draft.writes
        fb, fused, wide_positions, keep = self._hit_forward()
        # The bridge's verdict on this step (window 4 = R): the hit's span of
        # 7 rows starts at P - R = 4 > 0 but reaches the window; FULL capture
        # (SGLang's DSpark prefill) is allowed for a DSpark target.
        fb.capture_hidden_mode = SimpleNamespace(is_full=lambda: True)
        fb.is_prefill_only = False
        fb.return_logprob = False
        widened = np.array([7, 2])
        batch = SimpleNamespace(
            num_scheduled_tokens=widened,
            context_lens=np.array([11, 2]),
            replay_rows=fused.replay_rows(2),
        )
        assert bridge.v41_extend_replays(fb, batch, 4, dspark_context_in_forward=True)
        assert not bridge.v41_extend_replays(fb, batch, 4)
        plan = _plan(widened.tolist(), window=4, running_bs=2)
        # Hit: forward rows 3..6 (positions 7..10, P - 1 = 7 a replay row);
        # cold: rows 7, 8.
        assert plan.tail_rows[: plan.filled].tolist() == [3, 4, 5, 6, 7, 8]
        assert wide_positions[plan.tail_rows[: plan.filled]].tolist() == [
            7,
            8,
            9,
            10,
            0,
            1,
        ]
        residual = torch.randn(1, int(plan.tail.running_tokens), 2, 4)
        surface.begin_forward()
        wrapper.model(_State(residual))
        import atom.utils.forward_context as fc

        metadata = SimpleNamespace(
            step=SimpleNamespace(running_tokens=9, replay=plan), served_by_round=0
        )
        monkeypatch.setattr(
            fc, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
        )
        aux = surface.finish_forward(torch.zeros(9, 4), wide_positions, fb)
        tail = torch.cat(
            [(residual + k).mean(dim=-2).reshape(-1, 4) for k in (1, 2, 3)], dim=-1
        )
        assert aux.shape == (9, 12)
        torch.testing.assert_close(aux[3:9], tail[:6])
        assert not aux[:3].any()
        ((written, positions),) = writes
        assert written is aux and positions.tolist() == wide_positions.tolist()
        # SGLang's injection sees its own rows; their last is in the tail.
        assert aux.index_select(0, keep)[[2, 4]].abs().sum(-1).gt(0).all()

    def test_replay_taps_must_be_late_layers(self, monkeypatch):
        wrapper = _ToyWrapper()
        wrapper.model.replay = True
        wrapper.model.late = SimpleNamespace(first=2)
        _, surface = self._surface(monkeypatch, wrapper)
        with pytest.raises(ValueError, match="V41_DECODER_REPLAY=0"):
            surface.set_layers_to_capture([1, 2, 3])


# ---------------------------------------------------------------------------
# DSpark CUDA graphs: draft block (x5), target verify (x6)
# ---------------------------------------------------------------------------


@pytest.fixture
def spec_capture_builder():
    """The SGLang builder class on a DSpark cache stub, recording what a
    verify capture asks of core `_prepare` / `prepare_model_inputs`."""
    pytest.importorskip("aiter", reason="the V4.1 builder module reaches AITER")
    from atom.utils.forward_context import RoundService

    cls = bridge._make_builder_cls()
    builder = cls.__new__(cls)
    builder.block_size = PAGE
    builder.max_bs = 8
    builder.max_num_batched_tokens = 64
    builder.cache = SimpleNamespace(
        geometry=SimpleNamespace(window_size=128, speculative_tokens=5), pending=None
    )
    calls = SimpleNamespace(prepared=[], inputs=[])

    def prepare(batch, running_bs, running_tokens, **kwargs):
        calls.prepared.append((batch, running_bs, running_tokens, kwargs))
        metadata = SimpleNamespace(
            cache=builder.cache,
            served_by_round=RoundService.NONE,
            dummy=False,
            step=SimpleNamespace(running_tokens=running_tokens),
        )
        return metadata, torch.zeros(running_tokens, dtype=torch.int64)

    builder._prepare = prepare
    builder.prepare_model_inputs = lambda ids, md: calls.inputs.append((ids, md.dummy))
    return builder, calls


class TestVerifyCaptureStep:
    def test_tentative_step_on_capture_slots(self, spec_capture_builder):
        builder, calls = spec_capture_builder
        ids = torch.zeros(3 * W, dtype=torch.int64)
        metadata = builder.build_for_sglang_capture(
            3, input_ids=ids, slots=(65, 66, 67)
        )
        ((batch, *args),) = calls.prepared
        # Native `build_for_cudagraph_capture(bs, max_q_len=K+1)`: tentative,
        # Engram's live count 0, the bucket's (bs, max_q_len).
        assert args == [
            3,
            3 * W,
            {"max_q_len": W, "tentative": True, "engram_live": False},
        ]
        assert batch.num_scheduled_tokens == (W,) * 3
        # A full window in (a tentative step needs a prefix), page 0 only.
        assert batch.context_lens == (128 + W,) * 3
        assert batch.block_tables == ((0,),) * 3
        assert batch.state_slots_committed == (65, 66, 67)
        assert batch.total_seqs_num == 3 and batch.total_tokens_num == 3 * W
        # SGLang's graph input_ids, dummy: no prepare_state, so nothing pending.
        assert calls.inputs == [(ids, True)]
        assert metadata.dummy and builder.cache.pending is None

    def test_a_plain_pool_captures_one_row_a_request(self, spec_capture_builder):
        builder, calls = spec_capture_builder
        builder.cache.geometry.speculative_tokens = 0
        builder.build_for_sglang_capture(
            2, input_ids=torch.zeros(2, dtype=torch.int64), slots=(65, 66)
        )
        ((batch, *args),) = calls.prepared
        assert args == [
            2,
            2,
            {"max_q_len": 1, "tentative": False, "engram_live": False},
        ]
        assert batch.num_scheduled_tokens == (1, 1)


@needs_sglang
class TestVerifyCaptureStaging:
    def _builder(self, monkeypatch, speculative=5):
        calls = []
        builder = SimpleNamespace(
            geometry=SimpleNamespace(speculative_tokens=speculative),
            cache=object(),
            calls=calls,
            build_for_sglang_capture=lambda bs, *, input_ids, slots, **kw: calls.append(
                (bs, input_ids, tuple(slots), kw)
            )
            or "captured-md",
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        return builder

    def _fb(self, bs=2, width=W, rows=None, mode=None):
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY if mode is None else mode,
            batch_size=bs,
            input_ids=torch.zeros(bs * width if rows is None else rows),
            spec_info=SimpleNamespace(draft_token_num=width),
        )

    def test_verify_bucket_is_staged_on_the_capture_slots(self, monkeypatch):
        # The builder takes the verify width off the pool's geometry.
        builder = self._builder(monkeypatch)
        pool = SimpleNamespace(capture_slots=(65, 66, 67))
        fb = self._fb()
        assert bridge.stage_deepseek_v41_graph_capture(fb, pool) == "captured-md"
        ((bs, ids, slots, kw),) = builder.calls
        assert (bs, slots, kw) == (2, (65, 66), {})
        assert ids is fb.input_ids
        assert fb.atom_v41_graph_metadata == "captured-md"


class _VerifyReplayBuilder:
    """Records the verify replay staging; `prepare_model_inputs` leaves the
    device-staged tentative state the real builder leaves."""

    def __init__(self):
        self.block_size = PAGE
        self.max_bs = 8
        self.max_num_batched_tokens = 64
        self.geometry = SimpleNamespace(speculative_tokens=5)
        self.cache = SimpleNamespace(pending=None)
        self.calls = []

    def _prepare(self, batch, running_bs, running_tokens, **kwargs):
        from atom.utils.forward_context import RoundService

        self.calls.append(("prepare", running_bs, running_tokens, kwargs))
        self.batch = batch
        step = SimpleNamespace(
            tentative=True,
            max_q_len=kwargs["max_q_len"],
            running_tokens=running_tokens,
            scheduled_bs=batch.total_seqs_num,
            replay=None,
            decoder=None,
        )
        metadata = SimpleNamespace(
            cache=self.cache, step=step, served_by_round=RoundService.NONE
        )
        return metadata, torch.zeros(running_tokens, dtype=torch.int64)

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", input_ids))
        self.cache.pending = SimpleNamespace(step=metadata.step, staged_on_device=True)


def _verify_view(slots, prefix, bs, *, live=True, ends=None, rows=None):
    pad = bs - len(slots)
    prefix_t = torch.tensor(prefix, dtype=torch.int64)
    widened = prefix_t + W if ends is None else torch.tensor(ends, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        num_padding=pad,
        forward_mode=ForwardMode.TARGET_VERIFY,
        actual_forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=torch.zeros(bs * W if rows is None else rows, dtype=torch.int64),
        positions=torch.zeros(bs * W, dtype=torch.int64),
        req_pool_indices=torch.tensor(list(slots) + [0] * pad, dtype=torch.int64),
        # The device seq_lens stay the committed prefix; SGLang widens only
        # the host mirror (no make_forward_metadata_from_raw_verify).
        seq_lens=torch.cat([prefix_t, torch.ones(pad, dtype=torch.int64)]),
        seq_lens_cpu=torch.cat([widened, torch.ones(pad, dtype=torch.int64)]),
        spec_info=SimpleNamespace(
            draft_token_num=W, live_seq_lens_cpu=prefix_t.clone() if live else None
        ),
        # The host rows the verify ForwardBatch carries (`init_new`).
        atom_v41_req_pool_indices_cpu=torch.tensor(slots, dtype=torch.int64),
        atom_v41_rids=[f"r{i}" for i in range(len(slots))],
    )


@pytest.fixture
def verify_table(monkeypatch):
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
    return _req_to_token(
        rows=8,
        context_len=4 * PAGE,
        pages_by_row={1: [7, 3, 9], 2: [5], 3: [2, 11, 4, 8]},
    )


@needs_sglang
class TestVerifyReplayStaging:
    def test_padded_replay_stages_live_rows_at_bucket_width(
        self, monkeypatch, verify_table
    ):
        builder = _VerifyReplayBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        # raw 3 < bucket 4; p = 254 crosses a page inside the verify (254..259)
        # and 2 * PAGE - 3 crosses into a third page (509..514).
        view = _verify_view([3, 1, 2], [254, 2 * PAGE - 3, 10], 4)
        metadata = bridge.stage_deepseek_v41_verify_replay(view, verify_table)
        assert builder.calls[0] == (
            "prepare",
            4,
            4 * W,
            {"max_q_len": W, "tentative": True},
        )
        assert builder.calls[1][0] == "inputs" and builder.calls[1][1] is view.input_ids
        batch = builder.batch
        assert batch.total_seqs_num == 3 and batch.total_tokens_num == 3 * W
        assert batch.num_scheduled_tokens.tolist() == [W] * 3
        assert batch.context_lens.tolist() == [260, 2 * PAGE + 3, 16]
        assert batch.state_slots_committed == [3, 1, 2]
        assert [row.tolist() for row in batch.block_tables] == [
            [2, 11],
            [7, 3, 9],
            [5],
        ]
        assert builder.cache.pending.step is metadata.step
        # Nothing staged outside the graph's own Engram step.
        assert not getattr(metadata, "staged_outside_forward", False)

    @pytest.mark.parametrize(
        ("view_kwargs", "error", "match"),
        [
            ({"ends": [261, 16]}, ValueError, "lengths disagree"),
            ({"live": False}, ValueError, "live_seq_lens_cpu"),
        ],
    )
    def test_refusals(self, monkeypatch, verify_table, view_kwargs, error, match):
        builder = _VerifyReplayBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(error, match=match):
            bridge.stage_deepseek_v41_verify_replay(
                _verify_view([3, 1], [254, 10], 4, **view_kwargs), verify_table
            )
        assert builder.calls == []

    def _host_rows_replay(self, monkeypatch, verify_table, view):
        reads = []
        original = bridge._block_rows

        def counted(*args, **kwargs):
            reads.append(len(args[1]))
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge, "_block_rows", counted)
        builder = _VerifyReplayBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        bridge.stage_deepseek_v41_verify_replay(view, verify_table)
        return reads, [row.tolist() for row in builder.batch.block_tables]

    def _mirrored_view(self, slots, prefix, bs, rids):
        view = _verify_view(slots, prefix, bs)
        view.atom_v41_req_pool_indices_cpu = torch.tensor(slots)
        view.atom_v41_rids = list(rids)
        return view

    def test_verify_replay_stages_off_the_host_rows(self, monkeypatch, verify_table):
        # The overlap path: a padded verify replay whose copies hold every
        # page it needs reads nothing back. The copies are the host's word
        # (they differ from the table here on purpose).
        monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
        copies = [np.asarray([42, 43, 44], np.int32), np.asarray([45], np.int32)]
        bridge._HOST_ROWS.record([3, 1], ["a", "b"], copies)
        view = self._mirrored_view([3, 1], [254, 10], 4, ["a", "b"])
        reads, tables = self._host_rows_replay(monkeypatch, verify_table, view)
        assert reads == []
        # 254..259 spans two pages; 10..15 one.
        assert tables == [[42, 43], [45]]

    def test_verify_replay_reads_back_a_page_its_copy_lacks(
        self, monkeypatch, verify_table
    ):
        # A verify entering a page past its copy (p = 254: 254..259), another
        # tenant of the slot, or no copy at all: one read refreshes them.
        monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
        copies = [np.asarray([2], np.int32), np.asarray([7], np.int32)]
        bridge._HOST_ROWS.record([3, 1], ["a", "old"], copies)
        view = self._mirrored_view([3, 1], [254, 10], 2, ["a", "b"])
        reads, tables = self._host_rows_replay(monkeypatch, verify_table, view)
        assert reads == [2]
        assert tables == [[2, 11], [7]]
        rows = bridge._HOST_ROWS.lookup([3, 1], ["a", "b"], [2, 1])
        assert [row.tolist() for row in rows] == [[2, 11], [7]]
        # The next round finds them on the host.
        reads, _ = self._host_rows_replay(monkeypatch, verify_table, view)
        assert reads == []


def _graph_backend_for(
    monkeypatch, role, *, speculative=5, model=None, max_bs=8, pending=None
):
    from atom.plugin.sglang.attention_backend.deepseek_v41_backend import (
        ATOMDeepseekV41BackendForSgl,
    )

    def require_committed():
        if cache.pending is not None:
            raise RuntimeError("Commit the accepted prefix before reusing state")

    cache = SimpleNamespace(
        geometry=SimpleNamespace(speculative_tokens=speculative),
        pending=pending,
        require_committed=require_committed,
    )
    builder = SimpleNamespace(
        cache=cache, geometry=cache.geometry, max_bs=max_bs, _decoder_replays=False
    )
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    capacity = []
    monkeypatch.setattr(
        bridge,
        "check_deepseek_v41_graph_capacity",
        lambda bs, tokens: capacity.append((bs, tokens)) or builder,
    )
    monkeypatch.delenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", raising=False)
    pool = SimpleNamespace(
        is_atom_v41_proxy_pool=True,
        is_atom_v41_draft_proxy=role == "draft",
        atom_cache=cache,
        raw_arena=torch.zeros(8, dtype=torch.uint8),
        capture_slot_base=65,
        num_slots=129,
    )
    table = torch.zeros(4, PAGE, dtype=torch.int32)
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        is_draft_worker=role == "draft",
        model=model,
    )
    backend = ATOMDeepseekV41BackendForSgl(runner)
    return SimpleNamespace(
        backend=backend,
        builder=builder,
        cache=cache,
        pool=pool,
        capacity=capacity,
        table=table,
    )


def _target_model(calls):
    draft = SimpleNamespace(
        _context_kv_fusion=lambda: calls.append("kv_fusion"),
        _draft_kv_tail_fusion=lambda: calls.append("tail_fusion"),
    )
    return SimpleNamespace(_v41_dspark=SimpleNamespace(draft=draft))


@needs_sglang
class TestVerifyGraphBackend:
    def test_verify_state(self, monkeypatch):
        calls = []
        env = _graph_backend_for(monkeypatch, "target", model=_target_model(calls))
        env.backend.init_cuda_graph_state(4, 4 * W)
        assert env.backend._v41_graph_kind == "verify"
        assert env.capacity == [(4, 4 * W)]
        assert calls == ["kv_fusion", "tail_fusion"]
        # The verify replay's staging, bound once here.
        assert env.backend._stage_replay is bridge.stage_deepseek_v41_verify_replay

    def test_verify_state_under_decoder_replay(self, monkeypatch):
        # A verify publishes no decoder tail: it runs (and its graph records)
        # every row, replay or not.
        env = _graph_backend_for(monkeypatch, "target", model=_target_model([]))
        env.builder._decoder_replays = True
        env.backend.init_cuda_graph_state(4, 4 * W)
        assert env.backend._v41_graph_kind == "verify"

    def test_capture_and_replay(self, monkeypatch):
        env = _graph_backend_for(monkeypatch, "target", model=_target_model([]))
        captured, staged = [], []
        monkeypatch.setattr(
            bridge,
            "stage_deepseek_v41_graph_capture",
            lambda fb, pool: captured.append((fb, pool)),
        )
        monkeypatch.setattr(
            bridge,
            "stage_deepseek_v41_verify_replay",
            lambda view, table: staged.append((view, table)) or "replay-md",
        )
        backend = env.backend
        backend.init_cuda_graph_state(4, 4 * W)
        ids = torch.zeros(2 * W, dtype=torch.int64)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, batch_size=2, input_ids=ids
        )
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        assert captured == [(fb, env.pool)]
        view = SimpleNamespace(
            batch_size=2,
            num_padding=1,
            forward_mode=ForwardMode.TARGET_VERIFY,
            actual_forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=ids,
        )
        backend.init_forward_metadata_out_graph(view)
        backend.init_forward_metadata_out_graph(view)
        assert staged == [(view, env.table)] * 2
        assert backend._v41_replayed_buckets == {2}
        with pytest.raises(NotImplementedError, match="verify graphs .* DECODE"):
            backend.init_forward_metadata_out_graph(
                SimpleNamespace(
                    batch_size=2,
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    actual_forward_mode=ForwardMode.DECODE,
                    input_ids=ids,
                )
            )
        assert len(staged) == 2


def _draft_model(calls):
    return SimpleNamespace(
        prepare_v41_draft_graph=lambda: calls.append("prepare"),
        noise_token_id=NOISE,
    )


def _draft_graph_fb(prefix, slots, bs, *, noise=NOISE):
    raw, pad = len(slots), bs - len(slots)
    seq = torch.tensor(list(prefix) + [1] * pad, dtype=torch.int64)
    positions = torch.zeros(bs, 5, dtype=torch.int64)
    positions[:raw] = seq[:raw, None] + torch.arange(5)
    ids = torch.full((bs, 5), noise, dtype=torch.int64)
    ids[:, 0] = torch.arange(10, 10 + bs)
    return SimpleNamespace(
        batch_size=bs,
        num_padding=pad,
        forward_mode=ForwardMode.TARGET_VERIFY,
        actual_forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=ids.reshape(-1),
        positions=positions.reshape(-1),
        req_pool_indices=torch.tensor(list(slots) + [0] * pad, dtype=torch.int64),
        seq_lens=seq,
        spec_info=SimpleNamespace(draft_token_num=5),
    )


@needs_sglang
class TestDraftGraphBackend:
    def test_draft_state(self, monkeypatch):
        calls = []
        env = _graph_backend_for(monkeypatch, "draft", model=_draft_model(calls))
        env.backend.init_cuda_graph_state(4, 20)
        assert env.backend._v41_graph_kind == "draft"
        assert calls == ["prepare"]
        # Within the builder's bounds, as every decode-graph runner.
        assert env.capacity == [(4, 20)]

    def _captured(self, monkeypatch, bs=4, **kwargs):
        env = _graph_backend_for(monkeypatch, "draft", model=_draft_model([]), **kwargs)
        env.backend.init_cuda_graph_state(bs, bs * 5)
        fb = _draft_graph_fb([], [], bs)
        env.backend.init_forward_metadata_out_graph(fb, in_capture=True)
        return env, fb

    def _refill(self, fb, prefix, slots):
        live = _draft_graph_fb(prefix, slots, fb.batch_size)
        for name in ("input_ids", "positions", "req_pool_indices", "seq_lens"):
            getattr(fb, name).copy_(getattr(live, name))
        fb.num_padding = live.num_padding
        return fb

    def test_capture_and_padded_replay(self, monkeypatch):
        env, fb = self._captured(monkeypatch)
        self._refill(fb, [10, 300, 7], [2, 5, 3])
        env.backend.init_forward_metadata_out_graph(fb)
        assert env.backend._v41_replayed_buckets == {4}
        self._refill(fb, [10, 300, 7, 9], [2, 5, 3, 4])
        env.backend.init_forward_metadata_out_graph(fb)
        assert env.backend._v41_replayed_buckets == {4}

    def test_replay_needs_committed_state(self, monkeypatch):
        env, fb = self._captured(monkeypatch)
        self._refill(fb, [10], [2])
        env.cache.pending = object()
        with pytest.raises(RuntimeError, match="Commit the accepted prefix"):
            env.backend.init_forward_metadata_out_graph(fb)

    def test_a_replay_that_is_not_a_verify_is_refused(self, monkeypatch):
        env, _ = self._captured(monkeypatch)
        bad = _draft_graph_fb([10], [2], 4)
        bad.actual_forward_mode = ForwardMode.DECODE
        with pytest.raises(NotImplementedError, match="draft graphs .* DECODE"):
            env.backend.init_forward_metadata_out_graph(bad)


@needs_sglang
class TestDraftBlockPadding:
    def test_padding_rows_draft_from_slot_zero(self):
        from atom.plugin.sglang import deepseek_v41_dspark as dspark

        # SGLang's capture batch: zero positions and slots, any ids.
        fb, ids, positions = _draft_batch([1, 1], [0, 0])
        positions = torch.zeros_like(positions)
        ids = torch.zeros_like(ids)
        block = dspark.draft_block_inputs(fb, ids, positions, noise_token_id=NOISE)
        # Padding rows draft from anchor 0 in slot 0, as native pads.
        assert block.anchor_positions.tolist() == [0, 0]
        assert block.slots.tolist() == [0, 0]

    def test_anchor_clamp_is_a_no_op_on_live_rows(self):
        from atom.plugin.sglang.deepseek_v41_dspark import draft_block_inputs

        fb, ids, positions = _draft_batch([1, 300], [2, 5])
        block = draft_block_inputs(fb, ids, positions, noise_token_id=NOISE)
        assert block.anchor_positions.tolist() == [0, 299]


@needs_sglang
def test_verify_capture_is_not_a_dummy_run():
    # The draft's context write is left out of a forward whose ATOM context
    # says dummy; a TARGET_VERIFY capture batch must not.
    from atom.plugin.sglang.runtime.forward_context import _is_dummy_forward

    assert not _is_dummy_forward(
        SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    )
    assert _is_dummy_forward(SimpleNamespace(forward_mode=ForwardMode.IDLE))


@needs_sglang
class TestDraftWrapperGraphs:
    def test_attach_exposes_the_head_view_for_the_folded_sampler(self):
        wrapper = _bare_wrapper()
        attached, shared = [], []
        backbone = SimpleNamespace(
            embed=object(), head=SimpleNamespace(num_embeddings=12, tp_size=1)
        )
        wrapper.__dict__["model"] = SimpleNamespace(share_with_target=shared.append)
        lm_head = SimpleNamespace(
            atom_backbone=backbone,
            atom_wrapper=SimpleNamespace(attach_dspark_draft=attached.append),
        )
        wrapper.attach_shared_modules(embed_tokens=backbone.embed, lm_head=lm_head)
        assert wrapper.lm_head is lm_head
        assert shared == [backbone] and attached == [wrapper.model]
        assert "lm_head" not in dict(wrapper.named_modules())

    def test_memory_profile_compacts_the_draft_after_the_target(self, monkeypatch):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.model_executor import pool_configurator

        calls = []
        monkeypatch.setattr(
            bridge,
            "compact_v41_moe_buffers",
            lambda model, role="target": calls.append((model, role)),
        )
        monkeypatch.setattr(bridge, "init_deepseek_v41_builder", lambda cfg: None)
        monkeypatch.setattr(bridge, "_V41_BUILDER", None)
        monkeypatch.setattr(bridge, "_is_atom_v41_target", lambda kvc: True)
        monkeypatch.setattr(
            KVCacheConfigurator, "_profile_available_bytes", lambda self, m: 7
        )
        monkeypatch.setattr(
            pool_configurator, "_atom_v41_sizing_patched", False, raising=False
        )
        monkeypatch.setattr(
            pool_configurator,
            "create_memory_pool_configurator",
            pool_configurator.create_memory_pool_configurator,
        )
        bridge._install_pool_sizing_patch()
        draft = object()
        target = SimpleNamespace(
            atom_config=None, _v41_dspark=SimpleNamespace(draft=draft)
        )
        kvc = SimpleNamespace(model=target)
        assert KVCacheConfigurator._profile_available_bytes(kvc, 0) == 7
        assert calls == [(target, "target"), (draft, "draft")]
        # No DSpark draft: the target alone.
        calls.clear()
        target._v41_dspark = None
        KVCacheConfigurator._profile_available_bytes(kvc, 0)
        assert calls == [(target, "target")]

    def test_prepare_builds_the_lazy_caches(self, monkeypatch):
        wrapper = _bare_wrapper()
        calls = []
        weight = torch.zeros(2, 2)
        wrapper.__dict__["model"] = SimpleNamespace(
            embed=SimpleNamespace(weight=weight),
            head=object(),
            _context_layer_ids=lambda device: calls.append(("ids", device)),
            _noise_embedding=lambda: calls.append("noise"),
        )
        wrapper.target_atom_config = None
        wrapper.prepare_v41_draft_graph()
        assert calls == [("ids", weight.device), "noise"]


@needs_sglang
def test_v4_verify_graph_knob_does_not_govern_v41(monkeypatch):
    # SGLang names V4.1 `DeepseekV4ForCausalLM`; ATOM's V4 spec-graph patch
    # refuses every V4 TARGET_VERIFY replay unless its V4 knob is set. V4.1's
    # verify and draft-block graphs are staged by its own backend shim.
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    from atom.plugin import register

    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, fb: True)
    monkeypatch.setattr(
        DecodeCudaGraphRunner, "_atom_dsv4_spec_can_run_patched", False, raising=False
    )
    for cls, flag in [
        (EAGLEDraftCudaGraphRunner, "_atom_dsv4_replay_patched"),
        (EAGLEDraftExtendCudaGraphRunner, "_atom_dsv4_replay_patched"),
        (EagleDraftWorker, "_atom_qwen4_draft_mrope_patched"),
        (EagleDraftWorker, "_atom_dsv4_draft_extend_accept_patched"),
        (EagleDraftWorker, "_atom_dsv4_init_cuda_graphs_patched"),
    ]:
        monkeypatch.setattr(cls, flag, True, raising=False)
    monkeypatch.delenv("ATOM_SGLANG_V4_ENABLE_TARGET_VERIFY_CG", raising=False)
    register._patch_sglang_dsv4_spec_cuda_graph()

    def runner(hf_config):
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=hf_config)
            )
        )

    verify = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    v4 = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4"
    )
    glm = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="x")
    assert DecodeCudaGraphRunner.can_run_graph(runner(v4), verify) is False
    assert DecodeCudaGraphRunner.can_run_graph(runner(glm), verify) is False
    v41 = runner(_v41())
    assert DecodeCudaGraphRunner.can_run_graph(v41, verify) is True
    draft = _v41(architectures=["DeepseekV4ForCausalLMDSpark"])
    assert DecodeCudaGraphRunner.can_run_graph(runner(draft), verify) is True
    # Decided once per graph runner: later steps do not look at the model.
    v41.model_runner = None
    assert DecodeCudaGraphRunner.can_run_graph(v41, verify) is True


# ---------------------------------------------------------------------------
# Deferred host seq_lens (overlap): the draft launches before the host waits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Greedy draft: ATOM's sharded fused Markov argmax in SGLang's folded tail
# ---------------------------------------------------------------------------
