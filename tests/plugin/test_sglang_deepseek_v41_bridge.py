# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contract of DeepSeek-V4.1 on the SGLang plugin path.

SGLang 0.5.22 renames V4.1 to ``DeepseekV4ForCausalLM`` and keeps
``model_type=deepseek_v41``, so detection and adapter routing have to key on
the model type. Each ``ForwardBatch`` is translated into ATOM's
scheduled-batch protocol (block tables read off ``req_to_token`` at
256-token pages, STATE slot == ``req_pool_idx``), and SGLang's KV pool is
replaced by a proxy whose bytes are ATOM's CSA2 arena and whose size is priced
from the V4.1 geometry. All of it runs here on CPU against a real
``V41PoolGeometry`` and SGLang 0.5.22's own classes; the module skips without
SGLang (the bridge imports it). No checkpoint is read.
"""

import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sglang", reason="the bridge runs against SGLang 0.5.22")

import sglang.srt.layers.attention.base_attn_backend
import sglang.srt.mem_cache.base_swa_memory_pool
import sglang.srt.model_executor.forward_context
import sglang.srt.model_executor.pool_configurator
import sglang.srt.runtime_context  # noqa: F401
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang.attention_backend.deepseek_v41_backend import (
    ATOMDeepseekV41BackendForSgl,
)
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    DEEPSEEK_V41_ARCH,
    align_atom_config_with_sglang,
    build_deepseek_v41_forward_metadata,
    build_v41_scheduled_batch,
    is_deepseek_v41_config,
    refuse_unsupported_sglang_features,
    sglang_v41_kv_cache_dtype,
)
from atom.plugin.sglang.patches import sglang_version_guard
from atom.plugin.sglang.runtime import model_arch as model_arch_module
from atom.plugin.sglang.runtime.model_arch import (
    MODEL_ADAPTER_SPECS,
    resolve_model_arch_spec,
)

# The real SGLang modules, captured at collection so that a sibling test
# module that stubs ``sglang`` in ``sys.modules`` cannot swap them out; the
# bridge imports them lazily, so tests reinstall them (`real_sglang`).
_SGLANG_MODULE_NAMES = (
    "sglang.srt.runtime_context",
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.model_executor.forward_context",
    "sglang.srt.model_executor.pool_configurator",
    "sglang.srt.mem_cache.base_swa_memory_pool",
    "sglang.srt.layers.attention.base_attn_backend",
)
_SGLANG_MODULES = {name: sys.modules[name] for name in _SGLANG_MODULE_NAMES}

PAGE = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE


@pytest.fixture
def real_sglang(monkeypatch):
    for name, module in _SGLANG_MODULES.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _SGLANG_MODULES


@pytest.fixture(autouse=True)
def _no_global_builder(monkeypatch):
    # The bridge keeps one builder per process; no test may leak one.
    monkeypatch.setattr(bridge, "_V41_BUILDER", None)
    # No SGLang forward context here: a staged EXTEND notes its decoder rows
    # in a scratch table unless a test hands it a pool's.
    monkeypatch.setattr(bridge, "_decoder_rows_from", lambda: np.zeros(64, np.int64))


def _small_geometry(kv_dtype="bf16"):
    """A real CSA2 geometry at SGLang's 256-token PAGE, small enough for CPU."""
    return V41PoolGeometry(
        4,
        ((0, 2), (2, 1)),
        PAGE,
        4,
        512,
        32,
        kv_dtype=kv_dtype,
        layer_ratios=(0, 1, 2),
        index_topk=4,
    )


# ---------------------------------------------------------------------------
# Detection and adapter routing
# ---------------------------------------------------------------------------


def _sglang_normalized_v41():
    # What SGLang's config normalization leaves: V4's architecture name, V4.1's
    # model type.
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
    )


def _deepseek_v4():
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4"
    )


class TestDetection:
    def test_sglang_normalized_config_is_v41(self):
        assert is_deepseek_v41_config(_sglang_normalized_v41())

    def test_atom_text_config_is_v41(self):
        assert is_deepseek_v41_config(
            SimpleNamespace(
                architectures=[DEEPSEEK_V41_ARCH], model_type="deepseek_v41_text"
            )
        )

    @pytest.mark.parametrize(
        "config",
        [
            _deepseek_v4(),
            # V4 HF configs also report deepseek_v3 / the MTP model type.
            SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v3"
            ),
            SimpleNamespace(
                architectures=["DeepseekV4ForCausalLMNextN"],
                model_type="deepseek_v4_mtp",
            ),
            SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3"),
            # The shared resolver's own tests pass architecture-only configs.
            SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        ],
    )
    def test_other_models_are_not_v41(self, config):
        assert not is_deepseek_v41_config(config)


class TestAdapterRouting:
    def test_sglang_normalized_v41_resolves_to_v41_adapter(self):
        arch, spec = resolve_model_arch_spec(_sglang_normalized_v41())
        assert arch == DEEPSEEK_V41_ARCH
        assert spec is MODEL_ADAPTER_SPECS[DEEPSEEK_V41_ARCH]
        assert (
            spec.build_forward_metadata.__name__
            == "_build_deepseek_v41_forward_metadata"
        )
        assert spec.uses_context_only_forward

    def test_text_config_resolves_to_v41_adapter(self):
        arch, _ = resolve_model_arch_spec(
            SimpleNamespace(architectures=[], model_type="deepseek_v41_text")
        )
        assert arch == DEEPSEEK_V41_ARCH

    def test_v4_still_resolves_to_v4_adapter(self):
        arch, spec = resolve_model_arch_spec(_deepseek_v4())
        assert arch == "DeepseekV4ForCausalLM"
        assert (
            spec.build_forward_metadata.__name__
            == "_build_deepseek_v4_forward_metadata"
        )


class TestKVCacheDtype:
    # SGLang's DSV4 hook admits auto, bfloat16 and fp8_e4m3 only, and serves
    # auto as fp8_e4m3 (`_deepseek_v4_kv_cache_dtype`).
    def test_bfloat16_selects_the_bf16_pool(self):
        assert sglang_v41_kv_cache_dtype("bfloat16") == "bf16"

    def test_fp8_e4m3_selects_fp8(self):
        assert sglang_v41_kv_cache_dtype("fp8_e4m3") == "fp8"

    def test_auto_is_refused_as_sglangs_fp8(self):
        with pytest.raises(
            ValueError,
            match=r"--kv-cache-dtype bfloat16 or fp8_e4m3, got 'auto' \(SGLang "
            r"resolves auto to fp8_e4m3",
        ):
            sglang_v41_kv_cache_dtype("auto")


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------


def _supported_bags(**overrides):
    """SGLang's resolved config bags for a launch the bridge serves."""
    bags = {
        "memory": SimpleNamespace(
            disable_radix_cache=True,
            enable_hierarchical_cache=False,
            enable_unified_memory=False,
            enable_session_radix_cache=False,
            radix_cache_backend=None,
            enable_lmcache=False,
            enable_flexkv=False,
            enable_unified_cache_external_linker=False,
        ),
        "exec": SimpleNamespace(
            features=SimpleNamespace(
                enable_encoder_swa_bounded_replay=False,
                enable_decoder_swa_bounded_replay=False,
                enable_memory_saver=False,
            ),
            moe=SimpleNamespace(moe_a2a_backend="none"),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full"),
                    prefill=SimpleNamespace(backend="disabled"),
                )
            ),
        ),
        "spec": SimpleNamespace(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            speculative_draft_model_path=None,
            speculative_use_block_verification=False,
        ),
        "model": SimpleNamespace(model_path="/models/v41"),
        "disagg": SimpleNamespace(
            disaggregation_mode="null",
            language_model_only=True,
        ),
        "parallel": SimpleNamespace(
            enable_dp_attention=False, dcp_size=1, moe_dense_tp_size=None
        ),
        "schedule": SimpleNamespace(
            disable_hybrid_swa_memory=False,
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
            max_running_requests=256,
            mem_fraction_static=0.83,
            enable_mixed_chunk=False,
            disable_overlap_schedule=False,
        ),
    }
    bags.update(overrides)
    return bags


@pytest.fixture
def sglang_launch(real_sglang, monkeypatch):
    """Point SGLang's config-bag getters at fake bags (SGLang publishes them
    per scheduler process before the model is built)."""
    runtime_context = real_sglang["sglang.srt.runtime_context"]
    state = SimpleNamespace(bags=_supported_bags())
    for name in ("memory", "exec", "spec", "disagg", "parallel", "schedule", "model"):
        monkeypatch.setattr(
            runtime_context, f"get_{name}", lambda name=name: state.bags[name]
        )
    for name in (
        "SGLANG_ENABLE_METADATA_GLUE_GRAPH",
        "ATOM_ENGRAM_UVA",
        "ATOM_ENGRAM_OVERLAP",
    ):
        monkeypatch.delenv(name, raising=False)
    return state


class TestLaunchContract:
    def test_supported_launch_passes(self, sglang_launch):
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("bag", "field", "value", "match"),
        [
            ("memory", "enable_hierarchical_cache", True, "hierarchical cache"),
            ("disagg", "disaggregation_mode", "prefill", "PD disaggregation"),
            ("disagg", "language_model_only", False, "language-model-only"),
            ("schedule", "disable_hybrid_swa_memory", True, "hybrid-swa-memory"),
        ],
    )
    def test_bag_features_are_refused(self, sglang_launch, bag, field, value, match):
        setattr(sglang_launch.bags[bag], field, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("field", "match"),
        [
            ("enable_encoder_swa_bounded_replay", "encoder SWA bounded replay"),
            ("enable_memory_saver", "memory-saver"),
        ],
    )
    def test_exec_features_are_refused(self, sglang_launch, field, match):
        setattr(sglang_launch.bags["exec"].features, field, True)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("env", "refused"),
        # An empty value is off too (`_flag_env`).
        [(None, False), ("1", False), ("0", True), ("off", True), ("", True)],
    )
    def test_sglang_decoder_replay_flag_restates_the_default(
        self, sglang_launch, monkeypatch, env, refused
    ):
        # Decoder replay is on unless ATOM_SGLANG_V41_DECODER_REPLAY turns it
        # off; SGLang's flag may restate that, never contradict it.
        if env is None:
            monkeypatch.delenv("ATOM_SGLANG_V41_DECODER_REPLAY", raising=False)
        else:
            monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", env)
        sglang_launch.bags["exec"].features.enable_decoder_swa_bounded_replay = True
        if refused:
            with pytest.raises(ValueError, match="DECODER_REPLAY off"):
                refuse_unsupported_sglang_features()
        else:
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("enable_dp_attention", True, "DP attention"),
            ("dcp_size", 2, "decode context parallelism"),
            # SGLang would gather attention inputs the model does not read,
            # and align its graph buckets past the builder.
            ("moe_dense_tp_size", 1, "--moe-dense-tp-size"),
        ],
    )
    def test_parallel_features_are_refused(self, sglang_launch, field, value, match):
        setattr(sglang_launch.bags["parallel"], field, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize("backend", ["deepep", "mori", "megamoe"])
    def test_moe_all_to_all_is_refused(self, sglang_launch, backend):
        sglang_launch.bags["exec"].moe.moe_a2a_backend = backend
        with pytest.raises(
            ValueError,
            match=f"MoE all-to-all backend \\(resolved moe_a2a_backend={backend}",
        ):
            refuse_unsupported_sglang_features()

    def test_another_sglang_is_refused_before_any_rewiring(self, monkeypatch):
        # The bridge's rewiring of SGLang internals has no fallback: another
        # SGLang is refused before any of it is installed.
        def rewired():
            pytest.fail("SGLang rewired before the version refusal")

        for name in (
            "_install_pool_sizing_patch",
            "_install_graph_variants_patch",
            "_install_host_mirror_patch",
        ):
            monkeypatch.setattr(bridge, name, rewired)
        from atom.plugin.sglang.patches import (
            v41_extend_rows_at_alloc_patch,
            v41_greedy_sampler_patch,
            v41_prefill_token_graph_patch,
        )

        for module, name in (
            (v41_greedy_sampler_patch, "install_v41_greedy_sampler_patch"),
            (v41_extend_rows_at_alloc_patch, "install_v41_extend_rows_at_alloc_patch"),
            (v41_prefill_token_graph_patch, "install_v41_prefill_token_graph_patch"),
        ):
            monkeypatch.setattr(module, name, rewired)
        monkeypatch.setattr(sglang_version_guard, "sglang_version", lambda: "0.5.22")
        with pytest.raises(ValueError, match=r"this SGLang \(sglang 0\.5\.22"):
            bridge.install_deepseek_v41_proxy_pool_patch()
        monkeypatch.setattr(sglang_version_guard, "sglang_version", lambda: "0.5.23")
        sglang_version_guard.refuse_unsupported_sglang()

    def test_moe_fields_are_where_sglang_publishes_them(self, real_sglang):
        # The bags above are fakes; the fields the refusals read are SGLang
        # 0.5.22's (`get_exec().moe.moe_a2a_backend`,
        # `get_parallel().moe_dense_tp_size`).
        from sglang.srt.arg_groups.fields.exec_ import ExecMoe
        from sglang.srt.arg_groups.fields.parallel import Parallel

        assert "moe_a2a_backend" in ExecMoe.__struct_fields__
        assert "moe_dense_tp_size" in Parallel.__struct_fields__

    @pytest.mark.parametrize("backend", ["full", "breakable"])
    def test_prefill_graphs_are_refused(self, sglang_launch, backend):
        graphs = sglang_launch.bags["exec"].graph.cuda_graph_config
        graphs.prefill.backend = backend
        with pytest.raises(ValueError, match="prefill CUDA graphs"):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("env", "value", "match"),
        [
            ("SGLANG_ENABLE_METADATA_GLUE_GRAPH", "1", "METADATA_GLUE_GRAPH"),
            ("ATOM_ENGRAM_UVA", "0", "ATOM_ENGRAM_UVA=0 or ATOM_ENGRAM_OVERLAP=0"),
            ("ATOM_ENGRAM_OVERLAP", "0", "ATOM_ENGRAM_UVA=0 or ATOM_ENGRAM_OVERLAP=0"),
        ],
    )
    def test_host_staging_is_refused_with_decode_graphs(
        self, sglang_launch, monkeypatch, env, value, match
    ):
        # Refused before the weights load, and only with decode graphs on.
        monkeypatch.setenv(env, value)
        with pytest.raises(ValueError, match=f"{match} with decode CUDA graphs"):
            refuse_unsupported_sglang_features()
        graphs = sglang_launch.bags["exec"].graph.cuda_graph_config
        graphs.decode.backend = "disabled"
        refuse_unsupported_sglang_features()

    def test_radix_cache_is_served_by_default(self, sglang_launch, monkeypatch):
        # On by default, radix cache itself is not refused; what it cannot
        # run beside is (`test_sglang_deepseek_v41_radix.py`). Overlap
        # scheduling runs beside it (the radix seam drops a slot's host rows
        # on every checkpoint).
        monkeypatch.delenv("ATOM_SGLANG_V41_RADIX", raising=False)
        sglang_launch.bags["memory"].disable_radix_cache = False
        refuse_unsupported_sglang_features()
        sglang_launch.bags["schedule"].enable_mixed_chunk = True
        with pytest.raises(ValueError, match="mixed chunk with radix cache"):
            refuse_unsupported_sglang_features()

    def test_unpublished_bags_raise(self, real_sglang):
        # Outside a scheduler process SGLang has published no bags; the
        # bridge does not fall back to the raw ServerArgs.
        runtime_context = real_sglang["sglang.srt.runtime_context"]
        if runtime_context.get_context().is_config_namespace_published("memory"):
            pytest.skip("this test process published SGLang's config")
        with pytest.raises(ValueError, match="not published"):
            refuse_unsupported_sglang_features()
        with pytest.raises(ValueError, match="not published"):
            align_atom_config_with_sglang(SimpleNamespace(max_num_batched_tokens=1))

    def test_every_refusal_is_reported_at_once(self, sglang_launch):
        sglang_launch.bags["memory"].enable_hierarchical_cache = True
        sglang_launch.bags["parallel"].dcp_size = 2
        with pytest.raises(
            ValueError, match="hierarchical cache.*decode context parallelism"
        ):
            refuse_unsupported_sglang_features()

    def test_atom_config_takes_sglang_resolved_schedule(self, sglang_launch):
        config = SimpleNamespace(
            max_num_batched_tokens=4096, max_num_seqs=None, gpu_memory_utilization=None
        )
        align_atom_config_with_sglang(config)
        assert config.max_num_batched_tokens == 16384
        assert config.max_num_seqs == 256
        assert config.gpu_memory_utilization == 0.83

    @pytest.mark.parametrize("chunk", [-1, 0, None])
    def test_disabled_chunking_is_refused(self, sglang_launch, chunk):
        # `--chunked-prefill-size -1` stays -1 in the resolved bag; SGLang's
        # scheduler turns it into None, and admits a first prompt whole.
        sglang_launch.bags["schedule"].chunked_prefill_size = chunk
        with pytest.raises(ValueError, match="disabled chunked prefill"):
            refuse_unsupported_sglang_features()

    def test_the_builder_holds_the_widest_step(self, sglang_launch):
        # A mixed step (chunk plus decode rows) fits too: SGLang takes the
        # mixed decode rows out of the chunk budget (`PrefillAdder`); a
        # decode step has one row per running request.
        schedule = sglang_launch.bags["schedule"]
        for chunk, prefill, reqs in (
            (8192, 16384, 256),
            (32768, 16384, 256),
            (8192, 16384, 20000),
        ):
            schedule.chunked_prefill_size, schedule.max_prefill_tokens = chunk, prefill
            schedule.max_running_requests = reqs
            config = SimpleNamespace(max_num_batched_tokens=16384)
            align_atom_config_with_sglang(config)
            assert config.max_num_batched_tokens == max(16384, chunk, prefill, reqs)


class TestPrepareConfig:
    @pytest.fixture
    def prepared(self, sglang_launch, monkeypatch):
        from atom.plugin import register
        from atom.plugin.sglang.runtime.model_arch import (
            _prepare_deepseek_v41_config,
        )

        # The quant remap only reads the model class's mappings; the class
        # itself (`DeepseekV41RuntimeModel`) needs a GPU build of AITER.
        monkeypatch.setattr(
            register, "_ATOM_SUPPORTED_MODELS", {DEEPSEEK_V41_ARCH: object}
        )

        def prepare(
            env=None, fp4ptpc=False, online_quant_config=None, kv_cache_dtype="bf16"
        ):
            if env is None:
                monkeypatch.delenv("ATOM_SGLANG_V41_DECODER_REPLAY", raising=False)
            else:
                monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", env)
            config = SimpleNamespace(
                max_num_batched_tokens=16384,
                max_num_seqs=64,
                gpu_memory_utilization=None,
                kv_cache_block_size=128,
                kv_cache_dtype=kv_cache_dtype,
                index_cache_dtype="fp8",
                hf_config=SimpleNamespace(fp4ptpc=fp4ptpc),
                online_quant_config=online_quant_config,
                quant_config=None,
                speculative_config=None,
                enable_decoder_swa_bounded_replay=True,
                enable_encoder_swa_bounded_replay=True,
            )
            _prepare_deepseek_v41_config(config, DEEPSEEK_V41_ARCH)
            return config

        return prepare

    def test_fp4ptpc_needs_the_routed_expert_conversion(self, prepared):
        # As native refuses it (`FP4PTPC_RECIPE`), on every arch, in the
        # spelling an SGLang launch passes.
        with pytest.raises(ValueError, match="--model-loader-extra-config"):
            prepared(fp4ptpc=True)
        conversion = {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}
        prepared(fp4ptpc=True, online_quant_config=conversion)
        prepared(fp4ptpc=False)

    def test_the_recipe_is_the_json_the_plugin_reads(self):
        # `--model-loader-extra-config` carries `online_quant_config`, which
        # `_generate_atom_config_from_sglang_config` pops into ATOM's Config.
        import json
        import shlex

        flag = bridge.FP4PTPC_SGLANG_RECIPE.split(": ", 1)[1]
        name, value = shlex.split(flag)
        assert name == "--model-loader-extra-config"
        assert json.loads(value) == {
            "online_quant_config": {
                "layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}
            }
        }

    def test_fp8_kv_is_refused_where_aiter_has_no_backend(self, prepared, monkeypatch):
        # MI308X: the FP8 pool would build, then the first decode-graph capture
        # forward fails in `mla_sparse_prefill_v41`, as natively.
        monkeypatch.setattr(bridge, "fp8_kv_attention_backend", lambda config: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda index: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
        )
        with pytest.raises(
            ValueError,
            match=r"backend, which this device \(gfx942\) lacks; launch with "
            r"--kv-cache-dtype bfloat16$",
        ):
            prepared(kv_cache_dtype="fp8")

    def test_fp8_kv_passes_where_aiter_has_a_backend(self, prepared, monkeypatch):
        monkeypatch.setattr(bridge, "fp8_kv_attention_backend", lambda config: "opus")
        assert prepared(kv_cache_dtype="fp8").kv_cache_dtype == "fp8"

    def test_bf16_kv_is_never_probed(self, prepared, monkeypatch):
        # `bfloat16` reaches the bridge as "bf16" (`auto` is refused first).
        def probe(config):
            raise AssertionError("BF16 KV probed AITER's FP8 attention")

        monkeypatch.setattr(bridge, "fp8_kv_attention_backend", probe)
        prepared(kv_cache_dtype="bf16")

    def test_the_probe_asks_for_the_fp8_pools_attention_call(self, monkeypatch):
        # The probe must describe the call the FP8 pool's attention makes
        # (`sparse_attn_v41_paged_2src` over `PagedAttentionCache.query_fp8`
        # and MXFP8 planes), dtype for dtype, on one TP4 rank's heads.
        import types

        fp8_attention = pytest.importorskip(
            "atom.model_ops.attentions.deepseek_v41.fp8_attention"
        )
        from atom.model_ops.attentions.deepseek_v41.cache import (
            PagedAttentionCache,
        )

        heads, dim = 16, 512
        attended = {}
        monkeypatch.setattr(
            fp8_attention,
            "mla_sparse_prefill_v41",
            lambda *args, **kwargs: attended.update(args=args),
        )
        query = PagedAttentionCache.query_fp8(
            SimpleNamespace(kv_fp8=True), torch.zeros(1, heads, dim)
        )

        def plane(rows):
            return (
                torch.zeros(rows, dim, dtype=torch.uint8),
                torch.zeros(rows, dim // 32, dtype=torch.uint8),
            )

        index = torch.zeros(1, dtype=torch.int32)
        fp8_attention.sparse_attn_v41_paged_2src(
            query, plane(2), index, index, plane(3), index, index, None, 1.0
        )
        q, q_scale, csa, csa_scale, _, _, swa, swa_scale = attended["args"][:8]

        probed = []
        aiter_module = types.ModuleType("aiter.ops.mla_sparse_prefill")
        aiter_module.get_mla_sparse_prefill_v41_backend = (
            lambda *args, **kwargs: probed.append((args, kwargs)) or "opus"
        )
        monkeypatch.setitem(sys.modules, aiter_module.__name__, aiter_module)
        config = SimpleNamespace(
            tensor_parallel_size=4,
            hf_config=SimpleNamespace(num_attention_heads=4 * heads, head_dim=dim),
        )
        assert bridge.fp8_kv_attention_backend(config) == "opus"
        (device, shape), kwargs = probed[0]
        assert (device, shape) == (torch.device("cuda"), (1, heads, dim))
        assert kwargs == {
            "q_dtype": q.dtype,
            "q_scale_dtype": q_scale.dtype,
            "unified_kv_dtype": csa.dtype,
            "unified_kv_scale_dtype": csa_scale.dtype,
            "kv_dtype": swa.dtype,
            "kv_scale_dtype": swa_scale.dtype,
            "softmax_scale": dim**-0.5,
            "layout": "contiguous",
        }

    @pytest.mark.parametrize(
        ("env", "replay"),
        [(None, True), ("1", True), ("0", False), ("off", False), ("", False)],
    )
    def test_decoder_replay_is_on_unless_turned_off(self, prepared, env, replay):
        config = prepared(env)
        assert config.enable_decoder_swa_bounded_replay is replay
        assert config.enable_encoder_swa_bounded_replay is False
        assert config.kv_cache_block_size == PAGE
        assert (config.max_num_seqs, config.max_num_batched_tokens) == (256, 16384)


# ---------------------------------------------------------------------------
# ForwardBatch -> ATOM scheduled batch
# ---------------------------------------------------------------------------


def _req_to_token(rows, context_len, pages_by_row):
    """``req_to_token`` whose rows map position i to token slot
    ``page * 256 + i % 256`` over each row's own pages."""
    table = torch.zeros(rows, context_len, dtype=torch.int32)
    for row, pages in pages_by_row.items():
        for i, page in enumerate(pages):
            start = i * PAGE
            width = min(PAGE, context_len - start)
            table[row, start : start + width] = page * PAGE + torch.arange(width)
    return table


@pytest.fixture
def req_to_token(monkeypatch):
    """Rows 1..3 hold requests over scattered pages; row 0 is padding."""
    table = _req_to_token(
        rows=5,
        context_len=4 * PAGE,
        pages_by_row={1: [7, 3, 9], 2: [5], 3: [2, 11, 4, 8]},
    )
    monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
    return table


def _forward_batch(
    mode,
    *,
    req_pool_indices,
    seq_lens,
    extend_seq_lens=None,
    extend_prefix_lens=None,
    rids=None,
    cpu_lengths=True,
):
    bs = len(req_pool_indices)
    # Every ForwardBatch field the bridge reads, at SGLang's defaults.
    batch = SimpleNamespace(
        forward_mode=mode,
        rids=list(rids) if rids is not None else [f"r{i}" for i in range(bs)],
        batch_size=bs,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64) if cpu_lengths else None,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        # SGLang mirrors the rows on the host for EXTEND and MIXED only.
        req_pool_indices_cpu=(
            torch.tensor(req_pool_indices, dtype=torch.int64)
            if mode.is_extend_without_speculative()
            else None
        ),
        extend_seq_lens=None,
        extend_prefix_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens_cpu=None,
        is_prefill_only=False,
        capture_hidden_mode=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    if extend_seq_lens is not None:
        if cpu_lengths:
            batch.extend_seq_lens_cpu = list(extend_seq_lens)
            batch.extend_prefix_lens_cpu = list(extend_prefix_lens)
        else:
            batch.extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int32)
            batch.extend_prefix_lens = torch.tensor(
                extend_prefix_lens, dtype=torch.int32
            )
    return batch


def test_fake_forward_batches_carry_only_real_fields(real_sglang):
    # The bridge reads these ForwardBatch fields directly; the fake batches
    # here must not invent one SGLang's dataclass does not have.
    import dataclasses

    info = real_sglang["sglang.srt.model_executor.forward_batch_info"]
    real = {field.name for field in dataclasses.fields(info.ForwardBatch)}
    fake = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[2])
    assert set(vars(fake)) <= real


def _tables(batch):
    return [row.tolist() for row in batch.block_tables]


class TestScheduledBatch:
    def test_fresh_extend(self, req_to_token):
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[1, 2],
                seq_lens=[600, 10],
                extend_seq_lens=[600, 10],
                extend_prefix_lens=[0, 0],
            ),
            PAGE,
        )
        assert batch.req_ids == ["r0", "r1"]
        assert batch.num_scheduled_tokens.tolist() == [600, 10]
        assert batch.num_scheduled_tokens.dtype == np.int32
        assert batch.context_lens.tolist() == [600, 10]
        # ceil(600 / 256) = 3 pages, ceil(10 / 256) = 1.
        assert _tables(batch) == [[7, 3, 9], [5]]
        assert all(row.dtype == np.int32 for row in batch.block_tables)
        assert batch.state_slots_committed == [1, 2]
        assert batch.total_seqs_num == 2
        assert batch.total_tokens_num == 610
        assert batch.is_dummy_run is False
        assert batch.is_final_chunk is None
        assert batch.replaying == {}

    def test_chunked_extend_with_prefix(self, req_to_token):
        # Second chunk of a long prompt plus a short fresh one. The block
        # table covers the whole context, prefix pages included.
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[3, 2],
                seq_lens=[3 * PAGE + 1, 40],
                extend_seq_lens=[PAGE + 1, 40],
                extend_prefix_lens=[2 * PAGE, 0],
            ),
            PAGE,
        )
        assert batch.num_scheduled_tokens.tolist() == [PAGE + 1, 40]
        assert batch.context_lens.tolist() == [3 * PAGE + 1, 40]
        assert _tables(batch) == [[2, 11, 4, 8], [5]]
        assert batch.state_slots_committed == [3, 2]
        assert batch.total_tokens_num == PAGE + 41

    def test_extend_ending_on_a_page_boundary_takes_no_extra_page(self, req_to_token):
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[1],
                seq_lens=[2 * PAGE],
                extend_seq_lens=[PAGE],
                extend_prefix_lens=[PAGE],
            ),
            PAGE,
        )
        assert _tables(batch) == [[7, 3]]

    def test_extend_lengths_from_device_tensors(self, req_to_token):
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[2, 1],
                seq_lens=[30, PAGE + 5],
                extend_seq_lens=[20, 5],
                extend_prefix_lens=[10, PAGE],
                cpu_lengths=False,
            ),
            PAGE,
        )
        assert batch.num_scheduled_tokens.tolist() == [20, 5]
        assert batch.context_lens.tolist() == [30, PAGE + 5]
        assert _tables(batch) == [[5], [7, 3]]

    def test_decode(self, req_to_token):
        # seq_lens already counts the token being decoded.
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.DECODE,
                req_pool_indices=[3, 1, 2],
                seq_lens=[4 * PAGE, 2 * PAGE + 1, 1],
            ),
            PAGE,
        )
        assert batch.num_scheduled_tokens.tolist() == [1, 1, 1]
        assert batch.context_lens.tolist() == [4 * PAGE, 2 * PAGE + 1, 1]
        assert _tables(batch) == [[2, 11, 4, 8], [7, 3, 9], [5]]
        assert batch.state_slots_committed == [3, 1, 2]
        assert batch.total_tokens_num == 3

    def test_mixed_is_translated_as_extend(self, req_to_token):
        # A decode row rides a MIXED batch as a 1-token extend.
        batch = build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.MIXED,
                req_pool_indices=[1, 2],
                seq_lens=[PAGE + 1, 64],
                extend_seq_lens=[1, 64],
                extend_prefix_lens=[PAGE, 0],
            ),
            PAGE,
        )
        assert batch.num_scheduled_tokens.tolist() == [1, 64]
        assert _tables(batch) == [[7, 3], [5]]

    def test_only_the_live_batch_rows_are_read(self, req_to_token):
        # SGLang buffers can be longer than batch_size; the bridge slices.
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2, 0], seq_lens=[5, 6, 1]
        )
        fb.batch_size = 2
        batch = build_v41_scheduled_batch(fb, PAGE)
        assert batch.state_slots_committed == [1, 2]
        assert batch.total_seqs_num == 2

    @pytest.mark.parametrize(
        "mode",
        [ForwardMode.DRAFT_EXTEND_V2, ForwardMode.IDLE],
    )
    def test_unsupported_modes_are_refused(self, req_to_token, mode):
        fb = _forward_batch(mode, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(NotImplementedError, match="does not translate"):
            build_v41_scheduled_batch(fb, PAGE)

    def test_length_mismatch_is_refused(self, req_to_token):
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1],
            seq_lens=[100],
            extend_seq_lens=[50],
            extend_prefix_lens=[40],
        )
        with pytest.raises(ValueError, match="lengths disagree"):
            build_v41_scheduled_batch(fb, PAGE)

    @pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
    def test_unwritten_page_is_refused(self, req_to_token, mode):
        # Row 2 holds one PAGE; a length past it reads a req_to_token entry
        # SGLang never wrote (page 0, its padding page), as the replay path
        # already refuses.
        lengths = {}
        if mode == ForwardMode.EXTEND:
            lengths = {"extend_seq_lens": [5, PAGE + 1], "extend_prefix_lens": [0, 0]}
        fb = _forward_batch(
            mode, req_pool_indices=[1, 2], seq_lens=[5, PAGE + 1], **lengths
        )
        with pytest.raises(ValueError, match="padding page 0"):
            build_v41_scheduled_batch(fb, PAGE)

    @pytest.mark.parametrize("slots", [[1, 1], [0, 2]])
    def test_slots_must_be_live_and_unique(self, req_to_token, slots):
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=slots, seq_lens=[3, 4])
        with pytest.raises(ValueError, match="req_pool_idx"):
            build_v41_scheduled_batch(fb, PAGE)


class _SyncedEvent:
    def __init__(self):
        self.synced = 0

    def record(self, stream=None):
        pass

    def synchronize(self):
        self.synced += 1


class TestExtendRowsAtAlloc:
    """An EXTEND's PAGE rows, copied where `alloc_for_extend` wrote them and
    taken by the staging of the same batch (`ExtendRowRead`)."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch):
        monkeypatch.setattr(bridge, "_EXTEND_ROWS", bridge.ExtendRowRead())
        monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
        monkeypatch.setattr(
            bridge, "_device_module", lambda device: SimpleNamespace(Event=_SyncedEvent)
        )

    def _alloc(self, monkeypatch, table, slots, rids, ends):
        monkeypatch.setattr(
            bridge, "_V41_BUILDER", SimpleNamespace(cache=object(), block_size=PAGE)
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in rids],
            seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
        )
        slots_cpu = torch.tensor(slots, dtype=torch.int64)
        bridge.read_v41_extend_rows_at_alloc(batch, slots_cpu, slots_cpu)

    def _extend(self, slots, ends, prefix=None, rids=("r0", "r1")):
        prefix = prefix or [0] * len(slots)
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=slots,
            seq_lens=ends,
            extend_seq_lens=[e - p for e, p in zip(ends, prefix)],
            extend_prefix_lens=prefix,
            rids=list(rids[: len(slots)]),
        )
        fb.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
        return fb

    def _no_read_back(self, monkeypatch):
        reads = []
        original = bridge._block_rows

        def counted(*args, **kwargs):
            reads.append(len(args[1]))
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge, "_block_rows", counted)
        return reads

    def test_extend_takes_the_rows_its_allocation_copied(
        self, monkeypatch, req_to_token
    ):
        self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        reads = self._no_read_back(monkeypatch)
        batch = build_v41_scheduled_batch(self._extend([1, 2], [600, 10]), PAGE)
        assert reads == []
        assert _tables(batch) == [[7, 3, 9], [5]]
        assert all(row.dtype == np.int32 for row in batch.block_tables)
        assert batch.state_slots_committed == [1, 2]
        assert bridge._EXTEND_ROWS.pending is None
        # Graph replays stage off the same copies afterwards.
        rows = bridge._HOST_ROWS.lookup([1, 2], ["r0", "r1"], [3, 1])
        assert [row.tolist() for row in rows] == [[7, 3, 9], [5]]

    def test_chunk_with_a_prefix_takes_the_whole_row(self, monkeypatch, req_to_token):
        self._alloc(monkeypatch, req_to_token, [3], ["r0"], [3 * PAGE + 1])
        reads = self._no_read_back(monkeypatch)
        batch = build_v41_scheduled_batch(
            self._extend([3], [3 * PAGE + 1], prefix=[2 * PAGE]), PAGE
        )
        assert reads == []
        assert _tables(batch) == [[2, 11, 4, 8]]

    @pytest.mark.parametrize(
        "other",
        [
            # another batch (a MIXED one carries more requests)
            {"slots": [1, 2, 3], "ends": [600, 10, 30], "rids": ("r0", "r1", "r2")},
            # same slots, another request
            {"slots": [1, 2], "ends": [600, 10], "rids": ("r0", "x")},
            # same requests, other order
            {"slots": [2, 1], "ends": [10, 600], "rids": ("r1", "r0")},
            # wider than the copy
            {"slots": [1, 2], "ends": [700, 300], "rids": ("r0", "r1")},
        ],
    )
    def test_any_other_batch_reads_back(self, monkeypatch, req_to_token, other):
        req_to_token[2, PAGE : 2 * PAGE] = 6 * PAGE + torch.arange(PAGE)
        req_to_token[3, :PAGE] = 12 * PAGE + torch.arange(PAGE)
        self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        reads = self._no_read_back(monkeypatch)
        build_v41_scheduled_batch(
            self._extend(other["slots"], other["ends"], rids=other["rids"]),
            PAGE,
        )
        assert reads == [len(other["slots"])]
        assert bridge._EXTEND_ROWS.pending is None

    def test_a_copy_is_taken_once(self, monkeypatch, req_to_token):
        self._alloc(monkeypatch, req_to_token, [1], ["r0"], [600])
        reads = self._no_read_back(monkeypatch)
        build_v41_scheduled_batch(self._extend([1], [600]), PAGE)
        build_v41_scheduled_batch(self._extend([1], [600]), PAGE)
        assert reads == [1]

    def test_decode_never_takes_it(self, monkeypatch, req_to_token):
        self._alloc(monkeypatch, req_to_token, [1], ["r0"], [600])
        reads = self._no_read_back(monkeypatch)
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[600])
        build_v41_scheduled_batch(fb, PAGE)
        assert reads == [1]
        assert bridge._EXTEND_ROWS.pending is not None

    def test_a_padding_page_in_the_copy_is_refused(self, monkeypatch, req_to_token):
        # Row 2 holds one page: a length past it names the padding page 0.
        self._alloc(monkeypatch, req_to_token, [2], ["r0"], [PAGE + 1])
        with pytest.raises(ValueError, match="padding page 0"):
            build_v41_scheduled_batch(self._extend([2], [PAGE + 1]), PAGE)


class TestExtendRowsAtAllocPatch:
    def test_wraps_alloc_for_extend_once_and_copies_after_it(self, monkeypatch):
        from sglang.srt.managers import schedule_batch

        from atom.plugin.sglang.patches import v41_extend_rows_at_alloc_patch as patch

        assert patch._upstream_shape_error(schedule_batch) is None
        calls = []
        original = schedule_batch.alloc_for_extend

        def fake_alloc(batch):
            calls.append("alloc")
            return "loc", "dev", "cpu"

        fake_alloc.__wrapped__ = original
        monkeypatch.setattr(schedule_batch, "alloc_for_extend", fake_alloc)
        monkeypatch.setattr(
            bridge,
            "read_v41_extend_rows_at_alloc",
            lambda batch, dev, cpu: calls.append(("read", batch, dev, cpu)),
        )
        assert patch.install_v41_extend_rows_at_alloc_patch()
        wrapped = schedule_batch.alloc_for_extend
        assert patch.install_v41_extend_rows_at_alloc_patch()
        assert schedule_batch.alloc_for_extend is wrapped
        assert wrapped("b") == ("loc", "dev", "cpu")
        assert calls == ["alloc", ("read", "b", "dev", "cpu")]

    def test_kill_switch_and_shape_guard(self, monkeypatch):
        from sglang.srt.managers import schedule_batch

        from atom.plugin.sglang.patches import v41_extend_rows_at_alloc_patch as patch

        original = schedule_batch.alloc_for_extend
        monkeypatch.setenv("ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC", "0")
        assert not patch.install_v41_extend_rows_at_alloc_patch()
        assert schedule_batch.alloc_for_extend is original
        monkeypatch.delenv("ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC")

        def returns_two(batch):
            return 1, 2

        fake = SimpleNamespace(alloc_for_extend=returns_two)
        assert "no longer returns" in patch._upstream_shape_error(fake)


def test_every_batch_the_bridge_stages_reaches_the_plugin_prepare(
    monkeypatch, req_to_token
):
    """Each scheduled batch the bridge builds for the builder -- an EXTEND,
    a DECODE, a decode-graph capture and a decode-graph replay (the prefill
    capture step and warm-up: `test_sglang_v41_prefill_warmup.py`) -- goes
    through the plugin builder's own ``_prepare``, which reads
    ``decoder_replays`` off it. Only native's ``_prepare`` underneath is
    recorded."""
    pytest.importorskip("aiter", reason="the V4.1 builder module reaches AITER")
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    cls = bridge._make_builder_cls()
    builder = cls.__new__(cls)
    builder.block_size, builder.max_bs = PAGE, 8
    builder.max_num_batched_tokens = 1024
    owner = SimpleNamespace(begin=lambda: None, finish=lambda: None, fail=lambda: None)
    builder.geometry = SimpleNamespace(window_size=128, speculative_tokens=0)
    builder.cache = SimpleNamespace(geometry=builder.geometry, _sglang_h2d_owner=owner)
    builder.model_runner = SimpleNamespace(h2d_owner=owner, h2d_groups={})
    builder.publish_query_metadata = lambda *args: None
    decoders = []

    def native_prepare(self, batch, running_bs, running_tokens, **kwargs):
        decoders.append(self._sglang_decoder_replays)
        return SimpleNamespace(cache=self.cache), torch.arange(running_tokens)

    monkeypatch.setattr(DeepseekV41MetadataBuilder, "_prepare", native_prepare)
    monkeypatch.setattr(
        DeepseekV41MetadataBuilder, "prepare_model_inputs", lambda *args: None
    )
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
    extend = build_v41_scheduled_batch(
        _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1, 2],
            seq_lens=[600, 10],
            extend_seq_lens=[600, 10],
            extend_prefix_lens=[0, 0],
        ),
        PAGE,
    )
    builder.prepare_prefill(extend, 2, 610)
    decode = build_v41_scheduled_batch(
        _forward_batch(ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[5, 6]),
        PAGE,
    )
    builder.prepare_decode(decode, 2, 2, 1)
    builder.build_for_sglang_capture(
        2, input_ids=torch.zeros(2, dtype=torch.int64), slots=(4, 5)
    )
    view = SimpleNamespace(
        batch_size=3,
        num_padding=1,
        seq_lens_cpu=torch.tensor([5, 6, 1]),
        atom_v41_req_pool_indices_cpu=torch.tensor([1, 2]),
        atom_v41_rids=["a", "b"],
        req_pool_indices=torch.tensor([1, 2, 0]),
    )
    replay = bridge.build_v41_graph_replay_batch(view, req_to_token, PAGE)
    builder.prepare_decode(replay, 3, 3, 1)
    assert decoders == [True] * 4


class _FakeAttnBackend:
    def __init__(self, table):
        self.req_to_token_pool = SimpleNamespace(req_to_token=table)


def test_req_to_token_comes_from_sglang_forward_context(real_sglang):
    forward_context = real_sglang["sglang.srt.model_executor.forward_context"]
    table = torch.arange(8, dtype=torch.int32).view(2, 4)
    with forward_context.forward_context(
        forward_context.ForwardContext(attn_backend=_FakeAttnBackend(table))
    ):
        assert bridge._req_to_token() is table


# ---------------------------------------------------------------------------
# One step staged outside the forward
# ---------------------------------------------------------------------------


def _staged_rows(calls):
    """Engram rows whose stage and join the bridge runs outside the forward."""
    from atom.model_ops.engram.device.staging import EngramStagedRows

    class Rows(EngramStagedRows):
        def __init__(self):
            dict.__init__(self)

        def stage(self, *, tbo=False):
            calls.append("stage")

        def join(self):
            calls.append("join")

    return Rows()


class _RecordingBuilder:
    _decoder_replays = False

    def __init__(self, max_bs=4, max_num_batched_tokens=1024, positions=None):
        self.block_size = PAGE
        self.geometry = SimpleNamespace(window_size=128)
        self.max_bs = max_bs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.calls = []
        self.positions = positions
        self.rows = _staged_rows(self.calls)

    def _metadata(self):
        return SimpleNamespace(
            engram_embeddings=self.rows,
            # A prefill step: no request floor of its own, positions on CPU.
            step=SimpleNamespace(
                request_floors=np.zeros(0, np.int32), positions=torch.zeros(0)
            ),
        )

    def prepare_prefill(self, batch, running_bs, running_tokens):
        self.calls.append(("prefill", running_bs, running_tokens))
        self.batch = batch
        return self._metadata(), self.positions

    def prepare_decode(self, batch, running_bs, running_tokens, num_tokens):
        self.calls.append(("decode", running_bs, running_tokens, num_tokens))
        self.batch = batch
        return self._metadata(), self.positions

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", int(input_ids.shape[0])))


def _staged(monkeypatch, builder, fb, positions):
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    fb.input_ids = torch.zeros(positions.shape[0], dtype=torch.int64)
    return build_deepseek_v41_forward_metadata(fb)


class TestForwardMetadata:
    def test_extend_is_staged_once(self, monkeypatch, req_to_token):
        positions = torch.cat([torch.arange(PAGE, PAGE + 3), torch.arange(2)])
        builder = _RecordingBuilder(positions=positions.clone())
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1, 2],
            seq_lens=[PAGE + 3, 2],
            extend_seq_lens=[3, 2],
            extend_prefix_lens=[PAGE, 0],
        )
        metadata = _staged(monkeypatch, builder, fb, positions)
        assert metadata.staged_outside_forward is True
        assert builder.calls == [
            ("prefill", 2, 5),
            ("inputs", 5),
            "stage",
            "join",
        ]
        assert builder.batch.state_slots_committed == [1, 2]

    def test_decode_is_staged_once(self, monkeypatch, req_to_token):
        positions = torch.tensor([PAGE + 4, 9])
        builder = _RecordingBuilder(positions=positions.clone())
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[PAGE + 5, 10]
        )
        metadata = _staged(monkeypatch, builder, fb, positions)
        assert metadata.staged_outside_forward is True
        assert builder.calls[0] == ("decode", 2, 2, 1)

    def test_extend_carries_the_replay_verdict(self, monkeypatch, req_to_token):
        builder = _RecordingBuilder(positions=torch.arange(5))
        builder._decoder_replays = True
        builder.geometry = SimpleNamespace(window_size=128, speculative_tokens=0)
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1, 2],
            seq_lens=[PAGE + 3, 2],
            extend_seq_lens=[3, 2],
            extend_prefix_lens=[PAGE, 0],
        )
        _staged(monkeypatch, builder, fb, torch.arange(5))
        # A resumed request of 3 < 128 rows: every row runs.
        assert builder.batch.decoder_replays is False
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1],
            seq_lens=[PAGE + 200],
            extend_seq_lens=[200],
            extend_prefix_lens=[PAGE],
        )
        _staged(monkeypatch, builder, fb, torch.arange(200))
        assert builder.batch.decoder_replays is True
        assert builder.batch.is_final_chunk is None  # no per-request rest

    def test_extend_marks_the_chunk_sglang_continues(self, monkeypatch, req_to_token):
        """A middle chunk with a window or more of its prompt left finishes
        nothing (`v41_chunk_finals`); one with less keeps its tail. A step
        none of whose requests finishes says so on its step (the decoder
        gate's one read)."""
        builder = _RecordingBuilder(positions=torch.arange(300))
        builder._decoder_replays = True
        builder.geometry = SimpleNamespace(window_size=128, speculative_tokens=0)

        def extend(rest):
            fb = _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[1, 2],
                seq_lens=[PAGE + 200, 100],
                extend_seq_lens=[200, 100],
                extend_prefix_lens=[PAGE, 0],
            )
            setattr(fb, bridge.PROMPT_REST_ATTR, rest)
            metadata = _staged(monkeypatch, builder, fb, torch.arange(300))
            return (
                builder.batch.is_final_chunk,
                getattr(metadata.step, bridge.NO_PROMPT_ENDS_ATTR),
            )

        assert extend((0, 5000)) == ((True, False), False)
        assert extend((0, 127)) == (None, False)
        assert extend((0, 0)) == (None, False)
        assert extend((9000, 5000)) == ((False, False), True)

    def test_default_verdict_without_replay(self, monkeypatch, req_to_token):
        builder = _RecordingBuilder(positions=torch.arange(2))
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1],
            seq_lens=[2],
            extend_seq_lens=[2],
            extend_prefix_lens=[0],
        )
        _staged(monkeypatch, builder, fb, torch.arange(2))
        # The builder reads it for every batch; without replay nothing decides.
        assert builder.batch.decoder_replays is True

    def test_uninitialized_builder_is_refused(self):
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(RuntimeError, match="not initialized"):
            build_deepseek_v41_forward_metadata(fb)


class _PaddingBuilder(_RecordingBuilder):
    """Publishes positions out to the rows it is asked to run, as the V4.1
    builder does for a padded prefill."""

    def prepare_prefill(self, batch, running_bs, running_tokens):
        self.calls.append(("prefill", running_bs, running_tokens))
        self.batch = batch
        return self._metadata(), torch.arange(100, 100 + running_tokens)

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", int(input_ids.shape[0])))
        self.ids = input_ids.clone()


class TestPrefillTokenGraphs:
    """An EXTEND padded to a recorded token-graph bucket
    (`prefill_token_graph_bucket`, `PaddedPrefillInputs`)."""

    @pytest.fixture(autouse=True)
    def _graphs(self, monkeypatch):
        from atom.utils import region_graph

        graphs = bridge._PrefillTokenGraphs()
        graphs.sizes = (8, 16)
        monkeypatch.setattr(bridge, "_PREFILL_GRAPHS", graphs)
        monkeypatch.setattr(
            region_graph.TokenGraph,
            "bucket",
            staticmethod(
                lambda tokens, phase, multiple_of=1: next(
                    (n for n in (8, 16, 32) if n >= tokens), None
                )
            ),
        )
        return graphs

    def test_bucket_is_a_recorded_one(self, _graphs):
        assert bridge.prefill_token_graph_bucket(5) == 8
        assert bridge.prefill_token_graph_bucket(16) == 16
        # 32 exists in the class registry but was not recorded here.
        assert bridge.prefill_token_graph_bucket(17) is None
        _graphs.sizes = ()
        assert bridge.prefill_token_graph_bucket(5) is None

    def test_extend_is_padded_to_its_bucket(self, monkeypatch, req_to_token):
        builder = _PaddingBuilder()
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1, 2],
            seq_lens=[PAGE + 3, 2],
            extend_seq_lens=[3, 2],
            extend_prefix_lens=[PAGE, 0],
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb.input_ids = torch.tensor([11, 12, 13, 14, 15])
        bridge._PREFILL_GRAPHS.ids_buffer(16, "cpu").fill_(-1)
        metadata = build_deepseek_v41_forward_metadata(fb)
        # Laid out at the bucket; ids and positions at its rows.
        assert builder.calls[:2] == [("prefill", 2, 8), ("inputs", 8)]
        padded = metadata.sglang_padded_inputs
        assert padded.input_ids.tolist() == [11, 12, 13, 14, 15, 0, 0, 0]
        assert builder.ids.tolist() == [11, 12, 13, 14, 15, 0, 0, 0]
        assert padded.positions.tolist() == list(range(100, 108))
        assert padded.scheduled_tokens == 5
        assert padded.scheduled_tokens_gpu.tolist() == [5]
        assert padded.scheduled_tokens_gpu.dtype == torch.int32

    def test_no_bucket_runs_the_scheduled_rows(self, monkeypatch, req_to_token):
        builder = _PaddingBuilder()
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[3],
            seq_lens=[20],
            extend_seq_lens=[20],
            extend_prefix_lens=[0],
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb.input_ids = torch.arange(20)
        metadata = build_deepseek_v41_forward_metadata(fb)
        assert builder.calls[:2] == [("prefill", 1, 20), ("inputs", 20)]
        assert getattr(metadata, "sglang_padded_inputs", None) is None

    def test_decode_is_never_padded(self, monkeypatch, req_to_token):
        builder = _PaddingBuilder(positions=torch.tensor([4]))
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb.input_ids = torch.tensor([7])
        metadata = build_deepseek_v41_forward_metadata(fb)
        assert builder.calls[0] == ("decode", 1, 1, 1)
        assert getattr(metadata, "sglang_padded_inputs", None) is None

    def test_capture_step_is_natives_on_the_capture_slots(self):
        # Native `build_for_token_graph_capture` runs as is over the plugin's
        # `_build_for_prefill_capture`: native's request layout (one window
        # each, the last the remainder), the pool's capture slots and the
        # padded-prefill ids, `total_tokens_num_prefill` as native sets it.
        pytest.importorskip("aiter", reason="the V4.1 builder module reaches AITER")
        cls = bridge._make_builder_cls()
        builder = cls.__new__(cls)
        builder.block_size, builder.max_bs = PAGE, 8
        builder.max_num_batched_tokens = 1024
        builder.cache = object()
        builder.geometry = SimpleNamespace(window_size=128)
        prepared, inputs = [], []

        def prepare(batch, count, tokens, **kwargs):
            prepared.append((batch, count, tokens, kwargs))
            return SimpleNamespace(dummy=False), torch.arange(1024)

        builder._prepare = prepare
        builder.prepare_model_inputs = lambda ids, md: inputs.append((ids, md.dummy))
        ids = torch.zeros(512, dtype=torch.int64)
        builder._sglang_capture_inputs = ((65, 66, 67, 68, 69), ids)
        metadata, context = builder.build_for_token_graph_capture(300)
        ((batch, count, _tokens, kwargs),) = prepared
        assert batch.num_scheduled_tokens == (128, 128, 44) and count == 3
        assert batch.state_slots_committed == (65, 66, 67)
        assert batch.total_tokens_num == batch.total_tokens_num_prefill == 300
        assert batch.is_final_chunk is None and batch.block_tables == ((0,),) * 3
        assert kwargs == {"is_prefill": True, "engram_live": False}
        assert metadata.dummy and inputs[0][1] and inputs[0][0].shape == (300,)
        assert context.positions.shape == (300,) and context.running_tokens == 300

    @pytest.mark.parametrize(
        ("chunk", "budget"), [(512, 512), (4096, 1024), (-1, 1024), (None, 1024)]
    )
    def test_chunk_budget_reads_the_resolved_schedule(self, monkeypatch, chunk, budget):
        import sglang.srt.runtime_context as rc

        monkeypatch.setattr(
            rc, "get_schedule", lambda: SimpleNamespace(chunked_prefill_size=chunk)
        )
        builder = SimpleNamespace(max_num_batched_tokens=1024)
        assert bridge._prefill_chunk_budget(builder) == budget

    def _runner(self, *, dspark, replay):
        wrapper = SimpleNamespace(
            atom_config=SimpleNamespace(),
            model=SimpleNamespace(replay=replay),
            _v41_dspark=SimpleNamespace(_rows={}) if dspark else None,
        )
        return SimpleNamespace(model=wrapper, token_to_kv_pool=object())

    def test_dspark_without_decoder_replay_records_nothing(self, monkeypatch):
        # Without replay the recorded unit is `_Backbone`, the DSpark taps'
        # layers included: a replay would never fire them.
        monkeypatch.setattr(bridge, "get_deepseek_v41_builder", SimpleNamespace)
        runner = self._runner(dspark=True, replay=False)
        assert bridge.record_deepseek_v41_prefill_token_graphs(runner) == ()

    @pytest.mark.parametrize("dspark,replay", [(True, True), (False, False)])
    def test_records_past_the_dspark_guard(self, monkeypatch, dspark, replay):
        # DSpark under replay, or no DSpark: the guard lets the recorder on to
        # the pool (here a stub without the builder's limits, which stops it).
        monkeypatch.setattr(bridge, "get_deepseek_v41_builder", SimpleNamespace)
        runner = self._runner(dspark=dspark, replay=replay)
        with pytest.raises(AttributeError, match="max_num_batched_tokens"):
            bridge.record_deepseek_v41_prefill_token_graphs(runner)


class TestPrefillTokenGraphPatch:
    def _scheduler(self, *, draft=False, graphs=True):
        return SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    is_draft_worker=draft,
                    token_to_kv_pool=SimpleNamespace(
                        is_atom_v41_proxy_pool=True, is_atom_v41_draft_proxy=False
                    ),
                    decode_cuda_graph_runner=object() if graphs else None,
                )
            )
        )

    def test_records_after_sglang_captured_everything(self, monkeypatch):
        from sglang.srt.managers import scheduler as scheduler_module

        from atom.plugin.sglang.patches import v41_prefill_token_graph_patch as patch

        cls = scheduler_module.Scheduler
        original = cls.init_all_cuda_graphs
        assert patch._upstream_shape_error(cls) is None
        calls = []

        def fake_init(self):
            calls.append("sglang graphs")

        fake_init.__wrapped__ = original
        monkeypatch.setattr(cls, "init_all_cuda_graphs", fake_init)
        monkeypatch.setattr(
            bridge, "_V41_BUILDER", SimpleNamespace(cache=object(), block_size=PAGE)
        )
        monkeypatch.setattr(
            bridge,
            "record_deepseek_v41_prefill_token_graphs",
            lambda runner: calls.append(("record", runner)),
        )
        monkeypatch.setattr(
            bridge,
            "warm_deepseek_v41_eager_prefill",
            lambda runner: calls.append(("warm", runner)),
        )
        assert patch.install_v41_prefill_token_graph_patch()
        wrapped = cls.init_all_cuda_graphs
        assert patch.install_v41_prefill_token_graph_patch()
        assert cls.init_all_cuda_graphs is wrapped
        scheduler = self._scheduler()
        wrapped(scheduler)
        # The eager prefill warm-up runs after the token graphs are recorded.
        assert calls == [
            "sglang graphs",
            ("record", scheduler.tp_worker.model_runner),
            ("warm", scheduler.tp_worker.model_runner),
        ]
        # Graphs off (--disable-cuda-graph): no recording, and a log line
        # says so; the warm-up still runs.
        logs = []
        monkeypatch.setattr(patch.logger, "info", lambda msg, *args: logs.append(msg))
        calls.clear()
        scheduler = self._scheduler(graphs=False)
        wrapped(scheduler)
        assert calls == ["sglang graphs", ("warm", scheduler.tp_worker.model_runner)]
        assert len(logs) == 1 and "no decode CUDA graphs" in logs[0]
        # Each switch on its own, read when SGLang's graphs are done.
        for env, kept in (
            ("ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS", "warm"),
            ("ATOM_SGLANG_V41_PREFILL_WARMUP", "record"),
        ):
            monkeypatch.setenv(env, "0")
            calls.clear()
            scheduler = self._scheduler()
            wrapped(scheduler)
            assert calls == ["sglang graphs", (kept, scheduler.tp_worker.model_runner)]
            monkeypatch.delenv(env)

    def test_shape_guard(self):
        from sglang.srt.managers import scheduler as scheduler_module

        from atom.plugin.sglang.patches import v41_prefill_token_graph_patch as patch

        assert patch._upstream_shape_error(scheduler_module.Scheduler) is None

        class Other:
            def init_all_cuda_graphs(self, runner):
                pass

        assert "no longer takes (self)" in patch._upstream_shape_error(Other)
        assert "is missing" in patch._upstream_shape_error(type("Bare", (), {}))


class TestPaddedPrefillRuntime:
    """The plugin forward context runs a padded prefill's rows and hands
    SGLang the scheduled ones."""

    def test_forward_context_replays_and_trims(self, monkeypatch):
        from atom.plugin.sglang.runtime import forward_context as plugin_fc
        from atom.utils.forward_context import get_forward_context

        padded = bridge.PaddedPrefillInputs(
            input_ids=torch.arange(8),
            positions=torch.arange(100, 108),
            scheduled_tokens=5,
            scheduled_tokens_gpu=torch.tensor([5], dtype=torch.int32),
        )
        spec = SimpleNamespace(
            build_forward_metadata=lambda *a: SimpleNamespace(
                sglang_padded_inputs=padded, max_seqlen_q=0
            ),
            expand_forward_inputs=None,
        )
        monkeypatch.setattr(
            "atom.plugin.sglang.runtime.model_arch.resolve_model_arch_spec",
            lambda config: (DEEPSEEK_V41_ARCH, spec),
        )
        atom_config = SimpleNamespace(
            hf_config=None,
            enable_dp_attention=False,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        monkeypatch.setattr(
            plugin_fc, "_resolve_running_tokens_are_unified", lambda *a: True
        )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(5),
            positions=torch.arange(5),
        )
        with plugin_fc.SGLangPluginRuntime(
            atom_config=atom_config,
            forward_batch=fb,
            positions=torch.arange(5),
            input_ids=torch.arange(5),
        ) as runtime:
            forward = get_forward_context()
            assert forward.tr_graph == "replay"
            assert forward.context.running_tokens == 8
            assert forward.context.scheduled_tokens == 5
            assert forward.context.scheduled_tokens_gpu is padded.scheduled_tokens_gpu
            assert runtime.input_ids is padded.input_ids
            assert runtime.positions is padded.positions
            assert runtime.trim_output(torch.zeros(8, 3)).shape == (5, 3)


# ---------------------------------------------------------------------------
# KV pool: the proxy's arena and SGLang's sizing of it
# ---------------------------------------------------------------------------


# Decode-graph capture slots the small test builder reserves (its max_bs).
CAPTURE_SLOTS = 2


def _install_builder(monkeypatch, geometry, max_num_batched_tokens=32):
    builder = SimpleNamespace(
        geometry=geometry,
        device=torch.device("cpu"),
        block_size=PAGE,
        max_bs=CAPTURE_SLOTS,
        max_num_batched_tokens=max_num_batched_tokens,
        engram=None,
        score_workspace=SimpleNamespace(packed=False),
        _prefill_window_staging=None,
        num_blocks=None,
        cache=None,
    )
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    return builder


def _proxy(**overrides):
    kwargs = {
        "max_num_reqs": 3,
        "swa_size": 0,
        "full_size": 4 * PAGE,
        "page_size": PAGE,
        "swa_page_size": PAGE,
        "dtype": torch.bfloat16,
        "layer_num": 4,
        "device": "cpu",
        "start_layer": 0,
        "end_layer": 4,
        "num_req_slots": 4,
        "is_draft_worker": False,
    }
    kwargs.update(overrides)
    return bridge._make_proxy_pool_cls()(**kwargs)


class TestProxyPool:
    @pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
    def test_arena_is_pages_plus_padding_page_plus_state_slots(
        self, real_sglang, monkeypatch, kv_dtype
    ):
        geometry = _small_geometry(kv_dtype)
        builder = _install_builder(monkeypatch, geometry)
        pool = _proxy()

        # 4 usable PAGEs and SGLang's padding page 0; one slot per req row,
        # then one reserved decode-graph capture slot per builder request.
        assert pool.num_pages == 5
        assert pool.num_req_slots == 4
        assert pool.num_slots == 4 + CAPTURE_SLOTS
        slots = 4 + CAPTURE_SLOTS
        expected = geometry.paged_extents(5)[1] + slots * geometry.state_bytes
        if kv_dtype == "fp8":
            # A quantized pool is separate source buffers (auxiliary, one
            # main per owner, one window per layer), each started on its own
            # 256-byte boundary.
            expected += 256 * (1 + len(geometry.owners) + geometry.layers)
        assert pool.raw_arena.numel() == expected
        assert pool.get_kv_size_bytes() == expected
        assert pool.get_contiguous_buf_infos() == (
            [pool.raw_arena.data_ptr()],
            [expected],
            [1],
        )
        assert pool.mem_usage == pytest.approx(expected / (1 << 30))
        # ATOM's cache is handed the very bytes SGLang accounts for.
        cache = builder.cache
        assert builder.num_blocks == 5
        assert (cache.num_pages, cache.num_slots) == (5, slots)
        assert cache.backing.data_ptr() == pool.raw_arena.data_ptr()
        base, end = pool.raw_arena.data_ptr(), pool.raw_arena.data_ptr() + expected
        spans = sorted(
            (buf.data_ptr(), buf.data_ptr() + buf.numel()) for buf in cache.allocations
        )
        # Every buffer the cache allocated is the arena's, aligned, disjoint.
        assert len(spans) == (
            1 + len(geometry.owners) + geometry.layers if kv_dtype == "fp8" else 1
        )
        assert all(
            base <= lo and hi <= end and (lo - base) % 256 == 0 for lo, hi in spans
        )
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))
        if kv_dtype == "bf16":
            assert cache.backing.numel() == expected
        # Its publication owner is made at the first staged step.
        assert cache._sglang_h2d_owner is None

    @pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
    def test_cache_takes_the_builders_prefill_window_staging(
        self, real_sglang, monkeypatch, kv_dtype
    ):
        builder = _install_builder(monkeypatch, _small_geometry(kv_dtype))
        staging = torch.empty(64, dtype=torch.uint8)
        builder._prefill_window_staging = staging
        _proxy()
        assert builder.cache.prefill_window_staging is staging

    def test_sglang_facing_contract(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        pool = _proxy(swa_size=PAGE)
        base = real_sglang["sglang.srt.mem_cache.base_swa_memory_pool"]
        assert isinstance(pool, base.BaseSWAKVPool)
        assert pool.is_atom_v41_proxy_pool
        # The window lives in ATOM's STATE ring: no SWA allocator, no SWA
        # tree cache, no unified-kv ring.
        assert pool.needs_paged_swa_allocator is False
        assert pool._unified_kv is False
        assert pool.request_window is None
        assert (pool.size, pool.full_size, pool.swa_size) == (4 * PAGE, 4 * PAGE, PAGE)
        assert pool.store_dtype == torch.uint8
        assert pool.dtype == torch.bfloat16
        assert pool.get_state_buf_infos() == ([], [], [])
        with pytest.raises(NotImplementedError, match="relocation"):
            pool.move_kv_cache(torch.zeros(1), torch.zeros(1))
        with pytest.raises(NotImplementedError):
            pool.get_kv_buffer(0)
        with pytest.raises(NotImplementedError):
            pool.set_kv_buffer(None, None, None, None)
        with pytest.raises(NotImplementedError):
            pool.translate_loc_from_full_to_swa(torch.zeros(1))

    def test_refusals(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        # Built without the sizing hook's builder: its buffers would be
        # outside the profiled budget.
        monkeypatch.setattr(bridge, "_V41_BUILDER", None)
        with pytest.raises(RuntimeError, match="before the metadata builder"):
            _proxy()


class TestBackedCache:
    @pytest.fixture
    def cache_cls(self):
        pytest.importorskip("aiter", reason="the paged cache reaches AITER")
        return bridge._make_backed_cache_cls()

    def test_arena_is_a_zeroed_prefix_of_the_backing(self, cache_cls):
        backing = torch.full((100,), 7, dtype=torch.uint8)
        cache = cache_cls.__new__(cache_cls)
        cache._external_backing = backing
        arena = cache._allocate_backing(64, "cpu")
        assert arena.data_ptr() == backing.data_ptr()
        assert arena.numel() == 64
        assert not arena.any()
        assert (backing[64:] == 7).all()

    @pytest.mark.parametrize(
        ("backing", "match"),
        [
            (torch.zeros(63, dtype=torch.uint8), "short"),
        ],
    )
    def test_short_backings_are_refused(self, cache_cls, backing, match):
        cache = cache_cls.__new__(cache_cls)
        cache._external_backing = backing
        with pytest.raises(ValueError, match=match):
            cache._allocate_backing(64, "cpu")


class _FakeEvent:
    """``torch.cuda.Event`` for a CPU publication owner."""

    def record(self, *args):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def cpu_event(monkeypatch):
    # The bridge's owner always records a CUDA event; the CPU tests fake it.
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)


def _host_buffer(group=None):
    from atom.utils import CpuGpuBuffer

    return CpuGpuBuffer(
        4, dtype=torch.int32, device="cpu", pin_memory=False, publication_group=group
    )


@pytest.mark.usefixtures("cpu_event")
class TestPublication:
    def test_owner_binds_buffers_and_their_groups(self):
        a, b, c = _host_buffer("meta"), _host_buffer("meta"), _host_buffer()
        owner, groups = bridge._v41_publication_owner(
            {"a": a, "b": b, "c": c, "not_a_buffer": torch.zeros(1)}, "cpu"
        )
        assert set(groups) == {"meta"}
        assert [m.name for m in groups["meta"].members] == ["a", "b"]
        assert c._publication is not None and c._publication.owner is owner

    def test_epoch_finishes_or_fails_the_owner(self):
        from atom.utils.h2d import PublicationError

        owner, _ = bridge._v41_publication_owner({"a": _host_buffer()}, "cpu")
        with bridge._v41_publication_epoch(owner):
            assert owner._state == "active"
        assert (owner._state, owner.epoch) == ("sealed", 1)
        with pytest.raises(KeyError), bridge._v41_publication_epoch(owner):
            raise KeyError("staging failed")
        assert owner._state == "failed"
        with pytest.raises(PublicationError), bridge._v41_publication_epoch(owner):
            pass

    def test_runner_registry_has_the_builders_phase_groups(self):
        runner = SimpleNamespace(
            forward_vars={
                "x": _host_buffer(),
                "y": _host_buffer(),
                "z": _host_buffer("pos"),
            }
        )
        _, groups = bridge._init_v41_publication(
            runner.forward_vars, {"decode_metadata": ("x", "y")}, "cpu"
        )
        assert set(groups) == {"pos", "decode_metadata"}
        assert [m.name for m in groups["decode_metadata"].members] == ["x", "y"]


def _kvc(context_len=4096, hf_config=None):
    return SimpleNamespace(
        page_size=PAGE,
        is_draft_worker=False,
        model_config=SimpleNamespace(
            context_len=context_len,
            hf_config=hf_config or _sglang_normalized_v41(),
        ),
        model=SimpleNamespace(atom_config=object()),
    )


@pytest.fixture
def sizing(sglang_launch, monkeypatch):
    """The real configurator over a small geometry and fake schedule bags."""
    geometry = _small_geometry()
    _install_builder(monkeypatch, geometry)
    bags = sglang_launch.bags
    bags["schedule"].max_running_requests = 4
    return SimpleNamespace(
        geometry=geometry, bags=bags, cls=bridge._make_pool_configurator_cls()
    )


def _pool_cost(geometry, pages, slots, context_len):
    return (
        geometry.paged_extents(pages + 1)[1]
        + (slots + CAPTURE_SLOTS) * geometry.state_bytes
        + slots * context_len * 4
    )


class TestPoolSizing:
    @pytest.mark.parametrize("extra_pages", [1, 2, 37])
    def test_budget_buys_the_most_pages_that_fit(self, sizing, extra_pages):
        geometry = sizing.geometry
        slots = 4 + 1  # requested requests plus the padding row
        base = _pool_cost(geometry, 0, slots, 4096)
        available = base + extra_pages * geometry.paged_bytes + 1000
        configurator = sizing.cls(_kvc())
        config = configurator.calculate_pool_sizes(available, PAGE)

        pages = config.max_total_num_tokens // PAGE
        assert config.max_total_num_tokens % PAGE == 0
        assert config.full_max_total_num_tokens == config.max_total_num_tokens
        assert config.swa_max_total_num_tokens == 0
        assert _pool_cost(geometry, pages, slots, 4096) <= available
        assert _pool_cost(geometry, pages + 1, slots, 4096) > available
        # Extent alignment may cost a PAGE, never more.
        assert extra_pages - 1 <= pages <= extra_pages

    def test_sized_pool_fits_what_the_proxy_allocates(self, sizing, monkeypatch):
        # The proxy SGLang then builds from the configurator's token count
        # plus the req_to_token rows must stay inside the profiled budget.
        available = 50 * sizing.geometry.paged_bytes
        config = sizing.cls(_kvc()).calculate_pool_sizes(available, PAGE)
        pool = _proxy(full_size=config.max_total_num_tokens, num_req_slots=5)
        assert pool.get_kv_size_bytes() + 5 * 4096 * 4 <= available

    def test_too_small_budget_is_sglangs_out_of_memory(self, sizing):
        with pytest.raises(RuntimeError, match="Not enough memory"):
            sizing.cls(_kvc()).calculate_pool_sizes(1024, PAGE)

    def test_token_cap_rounds_down_to_pages(self, sizing):
        config = sizing.cls(_kvc()).calculate_pool_sizes_from_max_tokens(
            3 * PAGE + 17, PAGE
        )
        assert config.max_total_num_tokens == 3 * PAGE
        assert config.swa_max_total_num_tokens == 0

    def test_target_detection(self, sizing):
        assert bridge._is_atom_v41_target(_kvc())
        assert not bridge._is_atom_v41_target(_kvc(hf_config=_deepseek_v4()))
        draft = _kvc()
        draft.is_draft_worker = True
        assert not bridge._is_atom_v41_target(draft)


def test_sizing_patch_routes_only_v41(real_sglang, sglang_launch, monkeypatch):
    pool_configurator = real_sglang["sglang.srt.model_executor.pool_configurator"]
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    _install_builder(monkeypatch, _small_geometry())
    calls = []

    def factory(kvc):
        calls.append("sglang")
        return "sglang-configurator"

    def profile(self, pre_model_load_memory):
        calls.append(("profile", pre_model_load_memory))
        return 123

    # Registered with monkeypatch so the patch is undone after the test.
    monkeypatch.setattr(pool_configurator, "create_memory_pool_configurator", factory)
    monkeypatch.setattr(KVCacheConfigurator, "_profile_available_bytes", profile)
    monkeypatch.setattr(
        pool_configurator, "_atom_v41_sizing_patched", False, raising=False
    )
    inits = []
    monkeypatch.setattr(
        bridge, "init_deepseek_v41_builder", lambda config: inits.append(config)
    )

    bridge._install_pool_sizing_patch()
    bridge._install_pool_sizing_patch()  # idempotent

    v41, v4 = _kvc(), _kvc(hf_config=_deepseek_v4())
    assert type(pool_configurator.create_memory_pool_configurator(v41)).__name__ == (
        "ATOMDeepSeekV41PoolConfigurator"
    )
    assert pool_configurator.create_memory_pool_configurator(v4) == (
        "sglang-configurator"
    )
    # The builder is created right before SGLang profiles, for V4.1 only.
    assert KVCacheConfigurator._profile_available_bytes(v41, 5) == 123
    assert inits == [v41.model.atom_config]
    assert KVCacheConfigurator._profile_available_bytes(v4, 6) == 123
    assert len(inits) == 1
    assert calls == ["sglang", ("profile", 5), ("profile", 6)]


# The text_config fields of DeepSeek-V4.1-Flash's config.json the pool
# geometry reads (`build_v41_pool_geometry`).
FLASH_TEXT_CONFIG = {
    "num_hidden_layers": 40,
    "num_nextn_predict_layers": 3,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "index_topk": 512,
    "sliding_window": 128,
    "head_dim": 512,
    "index_head_dim": 128,
    "engram_max_ngram_size": 4,
}


@pytest.mark.parametrize(
    ("kv_dtype", "paged_bytes", "state_bytes"),
    [("bf16", 739_840, 5_276_672), ("fp8", 422_400, 2_745_600)],
)
def test_flash_geometry_prices(kv_dtype, paged_bytes, state_bytes):
    """What the shipped Flash checkpoint costs per 256-token PAGE and per
    request at SGLang's page size: 2890 B/token in bf16 (1650 in fp8), far
    above the ~1630 B/token SGLang's own DSV4 configurator would charge."""
    from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry

    geometry = build_v41_pool_geometry(
        SimpleNamespace(**FLASH_TEXT_CONFIG),
        PAGE,
        kv_dtype=kv_dtype,
        speculative_tokens=0,
        index_fp4=False,
        index_layout=None,
    )
    assert geometry.block_size == PAGE
    assert geometry.paged_bytes == paged_bytes
    assert geometry.state_bytes == state_bytes


# ---------------------------------------------------------------------------
# Attention backend shim
# ---------------------------------------------------------------------------


class TestMoeCompaction:
    def _layer(self, rebuilt):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.arange(12.0).view(3, 4), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(torch.ones(3, 2), requires_grad=False)
        layer.w13_weight.is_shuffled = True
        layer.quant_method = SimpleNamespace(
            moe_quant_config="built during the load",
            get_fused_moe_quant_config=lambda module: rebuilt.append(module)
            or ("current", module),
        )
        return layer

    def test_layers_get_fresh_storage_same_values_and_parameters(self, monkeypatch):
        emptied, rebuilt = [], []
        monkeypatch.setattr(torch.cuda, "memory_stats", dict)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([self._layer(rebuilt) for _ in range(2)])
        model.norm = torch.nn.LayerNorm(4)
        before = [
            (p, p.data.data_ptr(), p.data.clone())
            for layer in model.layers
            for p in (layer.w13_weight, layer.w2_weight)
        ]
        norm_ptr = model.norm.weight.data_ptr()

        assert bridge.compact_v41_moe_buffers(model) == 2
        for (param, ptr, values), now in zip(
            before,
            [p for layer in model.layers for p in (layer.w13_weight, layer.w2_weight)],
        ):
            assert now is param and now.data_ptr() != ptr
            assert torch.equal(now.data, values)
        assert model.layers[0].w13_weight.is_shuffled
        # Only MoE layers are copied.
        assert model.norm.weight.data_ptr() == norm_ptr
        for layer in model.layers:
            assert layer.quant_method.moe_quant_config == ("current", layer)
        assert rebuilt == list(model.layers)
        # Once before, once per layer, once at the end.
        assert len(emptied) == 4

    def test_memory_profile_compacts_once_then_builds(self, monkeypatch):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.model_executor import pool_configurator

        calls = []
        monkeypatch.setattr(
            bridge,
            "compact_v41_moe_buffers",
            lambda model, role="target": calls.append(("compact", model, role)),
        )
        monkeypatch.setattr(
            bridge,
            "init_deepseek_v41_builder",
            lambda config: calls.append(("builder", config))
            or monkeypatch.setattr(bridge, "_V41_BUILDER", object()),
        )
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
        config = object()
        model = SimpleNamespace(atom_config=config, _v41_dspark=None)
        kvc = SimpleNamespace(model=model)
        assert KVCacheConfigurator._profile_available_bytes(kvc, 0) == 7
        assert calls == [("compact", model, "target"), ("builder", config)]
        # A second profile builds nothing new and compacts nothing.
        calls.clear()
        KVCacheConfigurator._profile_available_bytes(kvc, 0)
        assert calls == [("builder", config)]


class TestBackendShim:
    def _runner(self, pool):
        return SimpleNamespace(
            device="cpu",
            token_to_kv_pool=pool,
            req_to_token_pool=object(),
            is_draft_worker=False,
        )

    def _backend(self):
        pool = SimpleNamespace(
            is_atom_v41_proxy_pool=True, is_atom_v41_draft_proxy=False
        )
        return ATOMDeepseekV41BackendForSgl(self._runner(pool))

    def test_publishes_pools_and_keeps_no_step_state(self):
        backend = self._backend()
        assert backend.get_name() == "dsv4"
        # SGLang's default, which publishes the host seq_lens the bridge reads.
        assert backend.needs_cpu_seq_lens
        backend.init_forward_metadata(object())
        assert backend.forward_metadata is None


# ---------------------------------------------------------------------------
# Decode CUDA graphs
# ---------------------------------------------------------------------------


class _Step(SimpleNamespace):
    pass


def _graph_metadata(builder, running_tokens, *, decode=True, tentative=False):
    from atom.utils.forward_context import RoundService

    return SimpleNamespace(
        cache=builder.cache,
        served_by_round=RoundService.NONE,
        dummy=False,
        step=_Step(running_tokens=running_tokens, decode=decode, tentative=tentative),
    )


class _CoreCaptureRecorder:
    """Stands in for `_prepare` / `prepare_model_inputs` on a real builder
    class, recording what a capture asks of them."""

    def __init__(self, builder):
        self.builder = builder
        self.prepared = []
        self.inputs = []

    def prepare(self, batch, running_bs, running_tokens, **kwargs):
        self.prepared.append((batch, running_bs, running_tokens, kwargs))
        return _graph_metadata(self.builder, running_tokens), torch.zeros(
            running_tokens, dtype=torch.int64
        )

    def prepare_model_inputs(self, input_ids, metadata):
        self.inputs.append((input_ids, metadata.dummy, metadata))


@pytest.fixture
def capture_builder():
    """The SGLang builder class over a stub cache: everything a capture reads
    off it, without a GPU."""
    pytest.importorskip("aiter", reason="the V4.1 builder module reaches AITER")
    cls = bridge._make_builder_cls()
    builder = cls.__new__(cls)
    builder.block_size = PAGE
    builder.max_bs = 8
    builder.max_num_batched_tokens = 64
    builder.cache = SimpleNamespace(
        geometry=SimpleNamespace(window_size=128, speculative_tokens=0)
    )
    builder.model_runner = SimpleNamespace(
        forward_vars={"input_ids": SimpleNamespace(gpu=torch.arange(64))}
    )
    recorder = _CoreCaptureRecorder(builder)
    builder._prepare = recorder.prepare
    builder.prepare_model_inputs = recorder.prepare_model_inputs
    return builder, recorder


class TestGraphCaptureStep:
    def test_matches_core_capture_but_slots_and_input_ids(self, capture_builder):
        builder, recorder = capture_builder
        bs, slots = 3, (10, 11, 12)
        input_ids = torch.zeros(bs, dtype=torch.int64)
        metadata = builder.build_for_sglang_capture(
            bs, input_ids=input_ids, slots=slots
        )
        builder.build_for_cudagraph_capture(bs)
        (ours, *ours_args), (core, *core_args) = recorder.prepared
        # Same widths and options: a decode step, Engram live count 0.
        assert ours_args == core_args
        assert ours_args == [
            bs,
            bs,
            {"max_q_len": 1, "tentative": False, "engram_live": False},
        ]
        for name in (
            "is_dummy_run",
            "num_scheduled_tokens",
            "context_lens",
            "block_tables",
            "total_seqs_num",
            "total_tokens_num",
            "is_final_chunk",
            "replaying",
        ):
            assert getattr(ours, name) == getattr(core, name), name
        # A full window in, every entry SGLang's padding page 0.
        assert ours.context_lens == (129,) * bs
        assert ours.block_tables == ((0,),) * bs
        # The caller's capture slots instead of core's 0..bs-1.
        assert ours.state_slots_committed == slots
        assert ours.req_ids == slots
        assert core.state_slots_committed == (0, 1, 2)
        # SGLang's input_ids buffer itself, not the shim's forward_vars.
        (ours_ids, ours_dummy, ours_md), (core_ids, core_dummy, _) = recorder.inputs
        assert ours_ids is input_ids
        assert (
            core_ids.data_ptr()
            == builder.model_runner.forward_vars["input_ids"].gpu.data_ptr()
        )
        assert ours_dummy and core_dummy
        assert ours_md is metadata
        # The forward stages Engram itself: in the recorded graph.
        assert not getattr(metadata, "staged_outside_forward", False)

    def test_a_speculative_geometry_captures_a_tentative_step(self, capture_builder):
        # As native's capture: `tentative` makes a multi-token bucket a decode.
        builder, recorder = capture_builder
        builder.cache.geometry.speculative_tokens = 2
        builder.build_for_sglang_capture(
            1, input_ids=torch.zeros(1, dtype=torch.int64), slots=(5,)
        )
        builder.build_for_cudagraph_capture(1)
        (_, *ours), (_, *core) = recorder.prepared
        assert ours[2]["tentative"] is core[2]["tentative"] is True

    def test_window_spanning_pages(self, capture_builder):
        builder, recorder = capture_builder
        builder.cache.geometry.window_size = PAGE
        builder.build_for_sglang_capture(
            1, input_ids=torch.zeros(1, dtype=torch.int64), slots=(5,)
        )
        assert recorder.prepared[0][0].block_tables == ((0, 0),)


class _GraphBuilder:
    """A builder that records the graph staging calls the bridge makes."""

    def __init__(self, max_bs=8, cache=None):
        self.block_size = PAGE
        self.max_bs = max_bs
        self.max_num_batched_tokens = 64
        self.cache = cache if cache is not None else SimpleNamespace()
        self.engram = None
        self.calls = []

    def build_for_sglang_capture(self, bs, *, input_ids, slots):
        self.calls.append(("capture", bs, input_ids, tuple(slots)))
        return _graph_metadata(self, bs)

    def prepare_decode(self, batch, running_bs, running_tokens, max_seqlen_q):
        self.calls.append(("decode", running_bs, running_tokens, max_seqlen_q))
        self.batch = batch
        metadata = _graph_metadata(self, running_tokens)
        # Graph steps never stage Engram outside the forward.
        metadata.engram_embeddings = SimpleNamespace(
            stage=lambda: self.calls.append("stage"),
            join=lambda: self.calls.append("join"),
        )
        return metadata, None

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", input_ids))


@pytest.fixture
def replay_table():
    # Rows 1..6 hold live requests; row 0 is padding. Row 6 names page 0.
    return _req_to_token(
        rows=8,
        context_len=4 * PAGE,
        pages_by_row={
            1: [7, 3, 9],
            2: [5],
            3: [2, 11, 4, 8],
            4: [6],
            5: [10, 12],
            6: [13, 0],
        },
    )


def _fb_view(req_pool_indices, seq_lens, bs, rids=None):
    """A replay view (`build_replay_fb_view`) with the host request rows every
    V4.1 decode batch carries (`_install_host_mirror_patch`)."""
    raw = len(req_pool_indices)
    pad = bs - raw
    lengths = torch.tensor(list(seq_lens) + [1] * pad, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        num_padding=pad,
        forward_mode=ForwardMode.DECODE,
        actual_forward_mode=ForwardMode.DECODE,
        input_ids=torch.zeros(bs, dtype=torch.int64),
        req_pool_indices=torch.tensor(
            list(req_pool_indices) + [0] * pad, dtype=torch.int64
        ),
        seq_lens=lengths.clone(),
        seq_lens_cpu=lengths,
        atom_v41_req_pool_indices_cpu=torch.tensor(req_pool_indices, dtype=torch.int64),
        atom_v41_rids=(
            list(rids) if rids is not None else [f"r{i}" for i in req_pool_indices]
        ),
    )


class TestGraphReplayStep:
    def test_padded_replay_stages_live_rows_at_bucket_width(
        self, monkeypatch, replay_table
    ):
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb = _fb_view([3, 1, 5, 2, 4], [3 * PAGE + 9, 2 * PAGE + 1, PAGE + 1, 7, 1], 8)
        metadata = bridge.stage_deepseek_v41_graph_replay(fb, replay_table)

        assert builder.calls == [("decode", 8, 8, 1), ("inputs", fb.input_ids)]
        assert builder.calls[1][1] is fb.input_ids
        batch = builder.batch
        assert batch.total_seqs_num == batch.total_tokens_num == 5
        assert batch.state_slots_committed == [3, 1, 5, 2, 4]
        assert batch.req_ids == [3, 1, 5, 2, 4]
        assert batch.context_lens.tolist() == [
            3 * PAGE + 9,
            2 * PAGE + 1,
            PAGE + 1,
            7,
            1,
        ]
        assert batch.num_scheduled_tokens.tolist() == [1] * 5
        assert _tables(batch) == [[2, 11, 4, 8], [7, 3, 9], [10, 12], [5], [6]]
        assert batch.is_dummy_run is False and batch.replaying == {}
        # No Engram stage/join and no outside-forward mark: the graph runs it.
        assert not getattr(metadata, "staged_outside_forward", False)

    def test_unpadded_replay(self, monkeypatch, replay_table):
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb = _fb_view([2], [PAGE], 1)
        bridge.stage_deepseek_v41_graph_replay(fb, replay_table)
        assert builder.calls[0] == ("decode", 1, 1, 1)
        assert _tables(builder.batch) == [[5]]

    @pytest.mark.parametrize(
        ("indices", "lengths", "match"),
        [
            ([1, 1], [5, 6], "req_pool_idx"),
            ([0, 2], [5, 6], "req_pool_idx"),
            ([6, 2], [PAGE + 1, 6], "padding page 0"),
        ],
    )
    def test_bad_rows_are_refused(
        self, monkeypatch, replay_table, indices, lengths, match
    ):
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(ValueError, match=match):
            bridge.stage_deepseek_v41_graph_replay(
                _fb_view(indices, lengths, 4), replay_table
            )
        assert builder.calls == []

    def test_live_first_page_of_a_page0_row_is_fine(self, monkeypatch, replay_table):
        # Row 6's second page is 0, but a request of PAGE tokens never reads it.
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        bridge.stage_deepseek_v41_graph_replay(_fb_view([6], [PAGE], 2), replay_table)
        assert _tables(builder.batch) == [[13]]


@pytest.fixture(autouse=True)
def _fresh_host_rows(monkeypatch):
    # The bridge keeps one host PAGE-row cache per process; no test may leak one.
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())


@pytest.fixture
def device_reads(monkeypatch):
    # Rows read back from req_to_token (`_block_rows`), per read.
    reads = []
    original = bridge._block_rows

    def counted(req_to_token, req_pool_indices, *args, **kwargs):
        reads.append(len(req_pool_indices))
        return original(req_to_token, req_pool_indices, *args, **kwargs)

    monkeypatch.setattr(bridge, "_block_rows", counted)
    return reads


class TestHostPageRows:
    def _replay(self, monkeypatch, table, view):
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        bridge.stage_deepseek_v41_graph_replay(view, table)
        return builder.batch

    def test_steady_decode_reads_the_device_once(
        self, monkeypatch, replay_table, device_reads
    ):
        rids = ["a", "b", "c"]
        first = self._replay(
            monkeypatch, replay_table, _fb_view([3, 1, 2], [700, 300, 9], 4, rids)
        )
        again = self._replay(
            monkeypatch,
            replay_table,
            _fb_view([3, 1, 2], [701, 301, 10], 4, rids),
        )
        assert device_reads == [3]
        assert _tables(first) == _tables(again) == [[2, 11, 4], [7, 3], [5]]
        assert again.state_slots_committed == [3, 1, 2]

    def test_an_extend_seeds_the_replay(self, monkeypatch, req_to_token, device_reads):
        build_v41_scheduled_batch(
            _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[1, 3],
                seq_lens=[PAGE + 5, 2 * PAGE + 1],
                extend_seq_lens=[PAGE + 5, 2 * PAGE + 1],
                extend_prefix_lens=[0, 0],
                rids=["x", "y"],
            ),
            PAGE,
        )
        assert device_reads == [2]  # The EXTEND's own read.
        device_reads.clear()
        batch = self._replay(
            monkeypatch,
            req_to_token,
            _fb_view([3, 1], [2 * PAGE + 2, PAGE + 6], 2, ["y", "x"]),
        )
        assert device_reads == []
        assert _tables(batch) == [[2, 11, 4], [7, 3]]

    def test_a_new_page_or_tenant_reads_the_device(
        self, monkeypatch, replay_table, device_reads
    ):
        self._replay(monkeypatch, replay_table, _fb_view([1], [PAGE], 1, ["a"]))
        # Entering its second page.
        batch = self._replay(
            monkeypatch, replay_table, _fb_view([1], [PAGE + 1], 1, ["a"])
        )
        assert _tables(batch) == [[7, 3]]
        # Another request in the same slot.
        self._replay(monkeypatch, replay_table, _fb_view([1], [5], 1, ["b"]))
        assert device_reads == [1, 1, 1]


def test_host_mirror_patch_carries_v41_decode_and_verify_rows(real_sglang, monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    runner_module = pytest.importorskip(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
    )

    def fake_init_new(cls, batch, model_runner, **kwargs):
        return SimpleNamespace(forward_mode=batch.mode, rids=batch.rids)

    def fake_view(*, forward_batch, **kwargs):
        return SimpleNamespace(batch_size=kwargs.get("bs"))

    monkeypatch.setattr(ForwardBatch, "init_new", classmethod(fake_init_new))
    monkeypatch.setattr(
        ForwardBatch, "_atom_v41_host_mirror_patched", False, raising=False
    )
    monkeypatch.setattr(runner_module, "build_replay_fb_view", fake_view)
    bridge._install_host_mirror_patch()
    bridge._install_host_mirror_patch()  # idempotent

    v41 = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_sglang_normalized_v41())
    )
    v4 = SimpleNamespace(model_config=SimpleNamespace(hf_config=_deepseek_v4()))
    slots = torch.tensor([4, 2])

    def batch(mode):
        return SimpleNamespace(
            mode=mode,
            rids=["a", "b"],
            req_pool_indices_cpu=slots,
            reqs=[],
            chunked_req=None,
        )

    decode = ForwardBatch.init_new(batch(ForwardMode.DECODE), v41)
    assert decode.atom_v41_req_pool_indices_cpu is slots
    # DSpark's verify is built from the ScheduleBatch too (prepare_for_verify).
    verify = ForwardBatch.init_new(batch(ForwardMode.TARGET_VERIFY), v41)
    assert verify.atom_v41_req_pool_indices_cpu is slots
    extend = ForwardBatch.init_new(batch(ForwardMode.EXTEND), v41)
    assert not hasattr(extend, "atom_v41_req_pool_indices_cpu")
    # An EXTEND carries each request's prompt rows left (`v41_chunk_finals`).
    cont, done = _req(20000, 8192), _req(300, 300)
    chunked = batch(ForwardMode.EXTEND)
    chunked.reqs, chunked.chunked_req = [done, cont], cont
    rest = ForwardBatch.init_new(chunked, v41)
    assert getattr(rest, bridge.PROMPT_REST_ATTR) == (0, 11808)
    other = ForwardBatch.init_new(batch(ForwardMode.DECODE), v4)
    assert not hasattr(other, "atom_v41_req_pool_indices_cpu")
    assert not hasattr(ForwardBatch.init_new(chunked, v4), bridge.PROMPT_REST_ATTR)

    view = runner_module.build_replay_fb_view(forward_batch=decode, bs=2)
    assert view.atom_v41_req_pool_indices_cpu is slots
    assert view.atom_v41_rids == ["a", "b"]
    assert not hasattr(
        runner_module.build_replay_fb_view(forward_batch=other, bs=2),
        "atom_v41_rids",
    )


class TestGraphAdapter:
    # The adapter calls the bridge's staging through model_arch's import.
    def _adapter(self):
        return MODEL_ADAPTER_SPECS[DEEPSEEK_V41_ARCH].build_forward_metadata

    def test_capture_stash_is_returned_untouched(self, monkeypatch):
        monkeypatch.setattr(
            model_arch_module,
            "build_deepseek_v41_forward_metadata",
            lambda *a: pytest.fail("eager staging ran for a captured step"),
        )
        stash = object()
        fb = SimpleNamespace(atom_v41_graph_metadata=stash)
        assert self._adapter()(None, fb, torch.zeros(1)) is stash

    def test_eager_still_stages_in_the_forward(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            model_arch_module,
            "build_deepseek_v41_forward_metadata",
            lambda fb: calls.append(fb) or "eager",
        )
        fb = SimpleNamespace()
        assert self._adapter()(None, fb, torch.zeros(1)) == "eager"
        assert calls == [fb]


@pytest.fixture
def graph_backend(real_sglang, monkeypatch):
    """The real backend over a real proxy pool (2 capture slots, 4 req rows)."""
    builder = _install_builder(monkeypatch, _small_geometry())
    pool = _proxy()
    table = torch.zeros(4, 4 * PAGE, dtype=torch.int32)
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        is_draft_worker=False,
    )
    return SimpleNamespace(
        backend=ATOMDeepseekV41BackendForSgl(runner),
        builder=builder,
        pool=pool,
        table=table,
    )


def _capture_fb(bs):
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=torch.zeros(bs, dtype=torch.int64),
    )


class TestGraphBackend:
    def test_capture_slots_follow_the_request_rows(self, graph_backend):
        pool = graph_backend.pool
        assert pool.capture_slot_base == pool.num_req_slots == 4
        assert pool.num_capture_slots == CAPTURE_SLOTS
        assert pool.capture_slots == (4, 5)
        assert pool.num_slots == 6

    def test_graph_state_is_validated(self, graph_backend, monkeypatch):
        # SGLang aligns its largest bucket up past the request rows it holds.
        backend = graph_backend.backend
        with pytest.raises(ValueError, match="exceed the builder"):
            backend.init_cuda_graph_state(CAPTURE_SLOTS + 1, CAPTURE_SLOTS + 1)
        with pytest.raises(ValueError, match="exceed the builder"):
            backend.init_cuda_graph_state(1, 64)
        backend.init_cuda_graph_state(2, 2)
        # The staging is bound once, here, not looked up per replay.
        assert backend._stage_capture is bridge.stage_deepseek_v41_graph_capture
        assert backend._stage_replay is bridge.stage_deepseek_v41_graph_replay

    def test_engram_host_fallback_is_refused(self, graph_backend):
        # Engram's tables could not be page-locked at load (the variables
        # that force the host path are refused at launch).
        graph_backend.builder.engram = SimpleNamespace(
            host=SimpleNamespace(overlap=None)
        )
        with pytest.raises(NotImplementedError, match="device staging"):
            graph_backend.backend.init_cuda_graph_state(2, 2)

    def _capture(self, graph_backend, bs):
        builder = graph_backend.builder
        builder.calls = []
        builder.build_for_sglang_capture = (
            lambda bs, *, input_ids, slots: builder.calls.append(
                ("capture", bs, input_ids, tuple(slots))
            )
            or "captured-md"
        )
        fb = _capture_fb(bs)
        graph_backend.backend.init_forward_metadata_out_graph(fb, in_capture=True)
        return fb

    def test_capture_stages_on_reserved_slots(self, graph_backend):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        fb = self._capture(graph_backend, 2)
        ((kind, bs, input_ids, slots),) = graph_backend.builder.calls
        assert (kind, bs, slots) == ("capture", 2, (4, 5))
        assert input_ids is fb.input_ids
        assert fb.atom_v41_graph_metadata == "captured-md"
        assert backend.forward_metadata is None

    def test_capture_is_decode_only(self, graph_backend):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        fb = _capture_fb(1)
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        with pytest.raises(NotImplementedError, match="decode graphs .* TARGET_VERIFY"):
            backend.init_forward_metadata_out_graph(fb, in_capture=True)

    def test_replay_stages_through_the_bridge(self, graph_backend, monkeypatch):
        staged = []
        monkeypatch.setattr(
            bridge,
            "stage_deepseek_v41_graph_replay",
            lambda view, table: staged.append((view, table)) or "replay-md",
        )
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        fb = self._capture(graph_backend, 2)

        def replay_view(bs, input_ids, actual=ForwardMode.DECODE):
            # `build_replay_fb_view`: the capture's mode and the batch's own.
            return SimpleNamespace(
                batch_size=bs,
                num_padding=1,
                input_ids=input_ids,
                forward_mode=ForwardMode.DECODE,
                actual_forward_mode=actual,
            )

        view = replay_view(2, fb.input_ids)
        backend.init_forward_metadata_out_graph(view)
        assert staged == [(view, graph_backend.table)]
        assert backend._v41_replayed_buckets == {2}

        # A batch that is not a decode (an IDLE rank's) is refused.
        with pytest.raises(NotImplementedError, match="IDLE"):
            backend.init_forward_metadata_out_graph(
                replay_view(2, fb.input_ids, ForwardMode.IDLE)
            )
        assert len(staged) == 1

    def test_eager_metadata_is_untouched(self, graph_backend):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        backend.init_forward_metadata(object())
        assert backend.forward_metadata is None


def test_candidate_graph_variants_are_off_for_v41(real_sglang, monkeypatch):
    runner_module = pytest.importorskip(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
    )
    calls = []

    def original(model_runner, *args, **kwargs):
        calls.append(model_runner)
        return "variants"

    monkeypatch.setattr(
        runner_module, "create_dsv41_candidate_graph_variants", original
    )
    bridge._install_graph_variants_patch()
    bridge._install_graph_variants_patch()  # idempotent
    patched = runner_module.create_dsv41_candidate_graph_variants
    v41 = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_sglang_normalized_v41())
    )
    v4 = SimpleNamespace(model_config=SimpleNamespace(hf_config=_deepseek_v4()))
    assert patched(v41, "DECODE", 0) is None
    assert patched(v4, "DECODE", 0) == "variants"
    assert calls == [v4]


# ---------------------------------------------------------------------------
# Decoder SWA bounded replay per EXTEND
# ---------------------------------------------------------------------------


def _replays(lengths, prefix, window=128, **fields):
    fb = SimpleNamespace(
        is_prefill_only=False,
        capture_hidden_mode=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    fb.__dict__.update(fields)
    batch = SimpleNamespace(
        num_scheduled_tokens=np.asarray(lengths, np.int32),
        context_lens=np.asarray(prefix, np.int64) + np.asarray(lengths, np.int64),
        replay_rows=np.zeros(len(lengths), np.int64),
    )
    return bridge.v41_extend_replays(fb, batch, window)


def _req(prompt, end):
    return SimpleNamespace(
        full_untruncated_fill_ids=[0] * prompt,
        extend_end=end,
        return_logprob=False,
        logprob_start_len=-1,
    )


class TestChunkFinals:
    """A chunk SGLang continues with at least a window of its prompt left runs
    none of the decoder, as native's middle chunk (`v41_chunk_finals`)."""

    def test_the_eager_runners_view_carries_the_host_mirror(self, monkeypatch):
        import dataclasses
        import types

        @dataclasses.dataclass
        class ForwardBatch:
            forward_mode: object
            input_ids: object = None

        class EagerRunner:
            def load_batch(self, forward_batch, pp_proxy_tensors=None, **kwargs):
                # SGLang's view over its static buffers: fields only.
                return dataclasses.replace(forward_batch, input_ids="static")

        module = types.ModuleType("sglang.srt.model_executor.runner.eager_runner")
        module.EagerRunner = EagerRunner
        monkeypatch.setitem(
            sys.modules, "sglang.srt.model_executor.runner.eager_runner", module
        )
        bridge._install_host_mirror_view_carrier()
        bridge._install_host_mirror_view_carrier()  # idempotent
        fb = ForwardBatch("EXTEND")
        setattr(fb, bridge.PROMPT_REST_ATTR, (0, 9000))
        view = EagerRunner().load_batch(fb)
        assert view is not fb and view.input_ids == "static"
        assert getattr(view, bridge.PROMPT_REST_ATTR) == (0, 9000)
        assert getattr(view, "atom_v41_req_pool_indices_cpu", None) is None
        # A verify's (or decode's) host request rows ride on its view too.
        verify = ForwardBatch("TARGET_VERIFY")
        verify.atom_v41_req_pool_indices_cpu = torch.tensor([3, 5])
        view = EagerRunner().load_batch(verify)
        assert (
            view.atom_v41_req_pool_indices_cpu is verify.atom_v41_req_pool_indices_cpu
        )
        assert getattr(view, bridge.PROMPT_REST_ATTR, None) is None
        other = EagerRunner().load_batch(ForwardBatch("DECODE"))
        assert getattr(other, bridge.PROMPT_REST_ATTR, None) is None

    def test_prompt_rest_is_the_continued_request_rows_left(self):
        a, b = _req(300, 300), _req(29000, 8192)
        batch = SimpleNamespace(reqs=[a, b], chunked_req=b)
        assert bridge.v41_prompt_rest(batch) == (0, 20808)
        # No continued request: every chunk finishes its prompt.
        assert bridge.v41_prompt_rest(
            SimpleNamespace(reqs=[a, b], chunked_req=None)
        ) == (
            0,
            0,
        )
        # A parked chunked request (not in this batch) marks nobody.
        parked = _req(29000, 8192)
        assert bridge.v41_prompt_rest(
            SimpleNamespace(reqs=[a, b], chunked_req=parked)
        ) == (0, 0)

    def _finals(self, rest, count=None, window=128):
        fb = SimpleNamespace()
        if rest is not None:
            setattr(fb, bridge.PROMPT_REST_ATTR, rest)
        n = len(rest) if count is None else count
        batch = SimpleNamespace(num_scheduled_tokens=np.ones(n, np.int32))
        return bridge.v41_chunk_finals(fb, batch, window)

    def test_middle_chunks_with_a_window_left_finish_nothing(self, monkeypatch):
        assert self._finals((0, 20808)) == (True, False)
        assert self._finals((8192,)) == (False,)
        assert self._finals((128,)) == (False,)  # exactly a window left

    def test_a_chunk_short_of_a_window_left_keeps_its_tail(self, monkeypatch):
        """Its next chunk is shorter than the window and reads the decoder
        rings this chunk's tail writes (`v41_extend_replays`)."""
        assert self._finals((127,)) is None
        assert self._finals((0, 1)) is None
        assert self._finals((0, 0)) is None

    def test_unknown_rest_finishes_every_request(self):
        assert self._finals(None, count=2) is None
        assert self._finals((8192,), count=2) is None  # misaligned

    @staticmethod
    def _forward(no_prompt_ends):
        step = SimpleNamespace(atom_v41_finishes_no_prompt=no_prompt_ends)
        return SimpleNamespace(attn_metadata=SimpleNamespace(step=step))

    def test_decoder_gate_reads_the_steps_flag(self, monkeypatch):
        runtime = pytest.importorskip("atom.models.deepseek_v41.runtime")
        from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep

        calls = []

        def core(forward):
            calls.append(forward)
            return True

        monkeypatch.setattr(runtime, "decoder_runs", core)
        monkeypatch.delattr(BatchStep, bridge.NO_PROMPT_ENDS_ATTR, raising=False)
        bridge.install_v41_chunk_decoder_gate()
        gate = runtime.decoder_runs
        bridge.install_v41_chunk_decoder_gate()  # idempotent
        assert runtime.decoder_runs is gate
        # Every step the plugin does not stage as a prefill reads False.
        assert getattr(BatchStep, bridge.NO_PROMPT_ENDS_ATTR) is False
        assert gate(self._forward(True)) is False
        assert calls == []
        final = self._forward(False)
        assert gate(final) is True and calls == [final]


class TestExtendReplays:
    def test_fresh_and_long_resumed_prefills_replay(self):
        assert _replays([8192], [0])
        assert _replays([5, 8192], [0, 8192])  # a short fresh prompt is whole
        assert _replays([128], [8192])  # exactly a window

    def test_short_resumed_chunk_runs_every_row(self):
        assert not _replays([127], [8192])
        assert not _replays([4096, 1], [0, 300])  # MIXED: a decode row

    def test_rows_sglang_reads_beyond_the_last_refuse(self):
        full = SimpleNamespace(is_full=lambda: True)
        last = SimpleNamespace(is_full=lambda: False)
        assert not _replays([4096], [0], capture_hidden_mode=full)
        assert _replays([4096], [0], capture_hidden_mode=last)
        # DSpark's FULL capture feeds only the no-op injector: the draft
        # context is written in the forward, from the tail.
        fb = SimpleNamespace(
            capture_hidden_mode=full,
            is_prefill_only=False,
            return_logprob=False,
            extend_logprob_start_lens_cpu=None,
        )
        batch = SimpleNamespace(
            num_scheduled_tokens=np.asarray([4096], np.int32),
            context_lens=np.asarray([4096], np.int64),
        )
        assert bridge.v41_extend_replays(fb, batch, 128, dspark_context_in_forward=True)
        fb.is_prefill_only = True
        assert not bridge.v41_extend_replays(
            fb,
            batch,
            128,
            dspark_context_in_forward=True,
        )
        assert not _replays([4096], [0], is_prefill_only=True)
        # Input logprobs on any row before the last.
        assert not _replays(
            [4096], [0], return_logprob=True, extend_logprob_start_lens_cpu=[0]
        )
        # Output logprobs only (start == length): the last row alone.
        assert _replays(
            [4096], [0], return_logprob=True, extend_logprob_start_lens_cpu=[4096]
        )
        assert not _replays([4096], [0], return_logprob=True)


def test_runtime_log_names_the_constructed_attention(caplog):
    class V41Attention:
        pass

    model = SimpleNamespace(
        layers=[SimpleNamespace(attn=V41Attention()) for _ in range(2)],
        replay=False,
    )
    with caplog.at_level("INFO", logger="atom"):
        bridge.log_deepseek_v41_runtime(model)
    assert "replay=False attention=['V41Attention']" in caplog.text


# ---------------------------------------------------------------------------
# The adapter an eager forward uses, resolved once per hf_config
# ---------------------------------------------------------------------------


class TestCachedAdapter:
    def test_second_forward_does_not_resolve(self, monkeypatch):
        from atom.plugin.sglang.runtime import forward_context, model_arch

        calls = []

        def refuse(atom_config, forward_batch, positions):
            raise LookupError("adapter reached")

        spec = model_arch.SGLangModelAdapterSpec(build_forward_metadata=refuse)

        def counted(config):
            calls.append(config)
            return "Counted", spec

        monkeypatch.setattr(model_arch, "resolve_model_arch_spec", counted)
        atom_config = SimpleNamespace(hf_config=_sglang_normalized_v41())
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False, is_draft_extend=lambda **_: False
            )
        )
        for _ in range(3):
            with pytest.raises(RuntimeError, match="Counted") as raised:
                forward_context._set_atom_forward_context(
                    atom_config, forward_batch, torch.zeros(1)
                )
            assert isinstance(raised.value.__cause__, LookupError)
        assert calls == [atom_config.hf_config]

    def test_a_replaced_hf_config_is_resolved_again(self):
        from atom.plugin.sglang.runtime.model_arch import cached_model_arch_spec

        atom_config = SimpleNamespace(hf_config=_sglang_normalized_v41())
        assert cached_model_arch_spec(atom_config) == (
            DEEPSEEK_V41_ARCH,
            MODEL_ADAPTER_SPECS[DEEPSEEK_V41_ARCH],
        )
        atom_config.hf_config = _deepseek_v4()
        assert cached_model_arch_spec(atom_config)[0] == "DeepseekV4ForCausalLM"
        atom_config.hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa"
        )
        assert cached_model_arch_spec(atom_config)[0] == "GlmMoeDsaForCausalLM"


def _extend_batch(spans, *, replays, finals=None, replaying=None, replay_rows=None):
    """An EXTEND ScheduledBatch over ``spans`` = [(slot, prefix, length)]."""
    slots = [s for s, _, _ in spans]
    lengths = np.asarray([n for _, _, n in spans], np.int64)
    ends = np.asarray([p + n for _, p, n in spans], np.int64)
    rows = np.zeros(len(spans), np.int64) if replay_rows is None else replay_rows
    return SimpleNamespace(
        state_slots_committed=slots,
        num_scheduled_tokens=(lengths + rows).astype(np.int32),
        context_lens=ends,
        replay_rows=np.asarray(rows, np.int64),
        decoder_replays=replays,
        is_final_chunk=finals,
        replaying=replaying or {},
    )


class TestDecoderFloors:
    """A step computing every row after a chunk that skipped the decoder
    (`v41_chunk_finals`) starts that request's decoder window where its
    rings hold rows (`v41_decoder_floors`), never below."""

    W = 128

    def _step(self, rows_from, batch):
        floors = bridge.v41_decoder_floors(batch, rows_from, self.W)
        bridge.v41_note_decoder_rows(batch, rows_from, self.W)
        return None if floors is None else floors.tolist()

    def test_skipped_chunk_then_a_logprob_step(self):
        """X (slot 1) skips its decoder on chunk [0, 8192); the next step
        carries another request's input logprobs, so it computes every row:
        X's decoder window starts at 8192, Z (fresh) at 0."""
        rows_from = np.zeros(8, np.int64)
        assert (
            self._step(
                rows_from, _extend_batch([(1, 0, 8192)], replays=True, finals=(False,))
            )
            is None
        )
        assert rows_from[1] == 8192
        step2 = _extend_batch([(1, 8192, 300), (2, 0, 200)], replays=False)
        assert self._step(rows_from, step2) == [8192, 0]
        # Rows written from the floor on stay where they were.
        assert rows_from[1] == 8192 and rows_from[2] == 0
        # A window above the floor, its rings hold the window again; a
        # replaying finishing chunk writes its tail.
        assert (
            self._step(rows_from, _extend_batch([(1, 8492, 50)], replays=False)) is None
        )
        assert (
            self._step(rows_from, _extend_batch([(1, 8542, 300)], replays=True)) is None
        )
        assert rows_from[1] == 8842 - self.W

    def test_short_continuations_after_a_skip_stay_floored(self):
        """SGLang may cut a continuation below the window (its memory
        budget): such a chunk computes every row, from the floor, and so does
        the next one while it still reaches below the floor."""
        rows_from = np.zeros(4, np.int64)
        self._step(
            rows_from, _extend_batch([(1, 0, 8192)], replays=True, finals=(False,))
        )
        assert self._step(rows_from, _extend_batch([(1, 8192, 50)], replays=False)) == [
            8192
        ]
        assert self._step(rows_from, _extend_batch([(1, 8242, 50)], replays=False)) == [
            8192
        ]
        assert self._step(
            rows_from, _extend_batch([(1, 8292, 200)], replays=False)
        ) == [8192]
        assert (
            self._step(rows_from, _extend_batch([(1, 8492, 20)], replays=False)) is None
        )

    def test_rings_a_tail_wrote_are_read_as_before(self):
        """A finishing chunk's (or one short of a window's) tail writes
        ``[e - W, e)``: the next step reads it, unfloored."""
        rows_from = np.zeros(8, np.int64)
        self._step(rows_from, _extend_batch([(1, 0, 8192)], replays=True))
        assert rows_from[1] == 8192 - self.W
        assert (
            self._step(rows_from, _extend_batch([(1, 8192, 60)], replays=False)) is None
        )
        # A step that computed every row leaves every row.
        rows_from[3] = 0
        self._step(rows_from, _extend_batch([(3, 0, 4096)], replays=False))
        assert (
            self._step(rows_from, _extend_batch([(3, 4096, 10)], replays=False)) is None
        )
        # A replaying step that finishes no prompt leaves nothing below its end.
        self._step(
            rows_from, _extend_batch([(3, 4106, 8192)], replays=True, finals=(False,))
        )
        assert rows_from[3] == 4106 + 8192

    def test_skipped_chunk_and_radix_hit_floors(self):
        """A request that skipped its last chunk is floored in a step that
        computes every row; a fused radix hit brings its own floor; a fresh
        request reads nothing below 0."""
        from atom.model_engine.sequence import Replay

        P, R = 2048, 128
        rows_from = np.full(8, 99999, np.int64)  # a previous tenant's
        rows_from[1] = 0
        self._step(
            rows_from, _extend_batch([(1, 0, 8192)], replays=True, finals=(False,))
        )
        main = _extend_batch([(1, 8192, 300)], replays=False)
        assert self._step(rows_from, main) == [8192]
        # Fused: the hit's span [P - R, end) in this forward, its own floor.
        rows_from[5] = 99999
        fused = _extend_batch(
            [(5, P, 50)],
            replays=False,
            replay_rows=np.asarray([R]),
            replaying={0: Replay(until=P, rebuilds=True, history_ids=(), start=P - R)},
        )
        assert self._step(rows_from, fused) is None
        assert rows_from[5] == P - R
        # A fresh request reads nothing below 0, whatever the slot held.
        rows_from[6] = 99999
        assert self._step(rows_from, _extend_batch([(6, 0, 10)], replays=False)) is None
        assert rows_from[6] == 0

    def test_replaying_steps_floor_nothing(self):
        rows_from = np.full(4, 8192, np.int64)
        replay = _extend_batch([(1, 8192, 300)], replays=True)
        assert bridge.v41_decoder_floors(replay, rows_from, self.W) is None

    def test_staged_logprob_step_after_a_skipped_chunk(self, monkeypatch, req_to_token):
        """Through the EXTEND staging: the chunk is skipped, the next step's
        logprob request turns replay off, and the continued request carries
        its decoder floor, eager (no prefill token graph)."""
        pool = SimpleNamespace(decoder_rows_from=np.zeros(5, np.int64))
        monkeypatch.setattr(
            bridge, "_decoder_rows_from", lambda: pool.decoder_rows_from
        )
        monkeypatch.setattr(bridge, "prefill_token_graph_bucket", lambda tokens: 1024)
        monkeypatch.setattr(bridge, "_PREFILL_GRAPHS", bridge._PrefillTokenGraphs())

        def run():
            builder = _RecordingBuilder(positions=torch.arange(1024))
            builder._decoder_replays = True
            builder.geometry = SimpleNamespace(window_size=128, speculative_tokens=0)
            first = _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[3],
                seq_lens=[512],
                extend_seq_lens=[512],
                extend_prefix_lens=[0],
            )
            setattr(first, bridge.PROMPT_REST_ATTR, (300,))
            _staged(monkeypatch, builder, first, torch.arange(512))
            first_finals = builder.batch.is_final_chunk
            second = _forward_batch(
                ForwardMode.EXTEND,
                req_pool_indices=[3, 2],
                seq_lens=[812, 100],
                extend_seq_lens=[300, 100],
                extend_prefix_lens=[512, 0],
            )
            second.return_logprob = True
            second.extend_logprob_start_lens_cpu = [300, 0]
            setattr(second, bridge.PROMPT_REST_ATTR, (0, 0))
            metadata = _staged(monkeypatch, builder, second, torch.arange(400))
            return first_finals, builder, metadata

        finals, builder, metadata = run()
        assert finals == (False,)
        assert builder.batch.decoder_replays is False
        floors = getattr(metadata, bridge.DECODER_FLOORS_ATTR)
        assert floors.tolist() == [512, 0] and floors.dtype == torch.int32
        assert builder.calls[-4] == ("prefill", 2, 400)  # eager, unpadded
        assert pool.decoder_rows_from[3] == 512 and pool.decoder_rows_from[2] == 0

    def test_logprob_request_keeps_its_chunk_tail(self):
        """A continued request asking input logprobs of a later prompt row
        counts as finishing each chunk: the chunk holding that row computes
        every row and reads the tail this one writes."""

        def req(start, asks=True):
            r = _req(9000, 8192)
            r.return_logprob, r.logprob_start_len = asks, start
            return SimpleNamespace(reqs=[r], chunked_req=r)

        assert bridge.v41_prompt_rest(req(8950)) == (0,)
        assert bridge.v41_prompt_rest(req(0)) == (0,)
        # Output logprobs only (-1), past the prompt, or none asked.
        assert bridge.v41_prompt_rest(req(-1)) == (808,)
        assert bridge.v41_prompt_rest(req(9000)) == (808,)
        assert bridge.v41_prompt_rest(req(0, asks=False)) == (808,)

    def test_floor_gate_runs_the_decoder_on_the_floors(self, monkeypatch):
        runtime = pytest.importorskip("atom.models.deepseek_v41.runtime")
        seen = []
        zeros = torch.zeros(2, dtype=torch.int32)

        def core(forward, stages, first, middle, last):
            step = forward.attn_metadata.step
            seen.append((step.swa_replay_start, dict(step.indptrs)))
            step.indptrs = {"ratio": "decoder's"}
            return "hidden"

        monkeypatch.setattr(runtime, "run_decoder_stages", core)
        bridge.install_v41_decoder_floor_gate()
        gate = runtime.run_decoder_stages
        bridge.install_v41_decoder_floor_gate()  # idempotent
        assert runtime.run_decoder_stages is gate

        def forward(floors):
            step = SimpleNamespace(
                swa_replay_start=zeros, indptrs={"ratio": "encoder's"}
            )
            metadata = SimpleNamespace(step=step)
            if floors is not None:
                setattr(metadata, bridge.DECODER_FLOORS_ATTR, floors)
            return SimpleNamespace(attn_metadata=metadata), step

        floors = torch.tensor([512, 0], dtype=torch.int32)
        fwd, step = forward(floors)
        assert gate(fwd, None, None, None, None) == "hidden"
        assert seen[-1][0] is floors and seen[-1][1] == {}
        assert step.swa_replay_start is zeros and step.indptrs == {}
        for fwd, step in (forward(None),):
            gate(fwd, None, None, None, None)
            assert seen[-1] == (zeros, {"ratio": "encoder's"})
            assert step.indptrs == {"ratio": "decoder's"}

    def test_publish_takes_the_step_floor_too(self):
        step = SimpleNamespace(
            request_floors=np.asarray([0, 1920, 0], np.int32),
            positions=torch.zeros(3),
        )
        metadata = SimpleNamespace(step=step)
        bridge.publish_v41_decoder_floors(metadata, np.asarray([512, 0, 0]))
        floors = getattr(metadata, bridge.DECODER_FLOORS_ATTR)
        assert floors.tolist() == [512, 1920, 0] and floors.dtype == torch.int32


def _publication_vars(device, pin, seed=0):
    from atom.utils import CpuGpuBuffer

    g = torch.Generator().manual_seed(seed)

    def buf(group, n=8, dtype=torch.int32):
        b = CpuGpuBuffer(
            n, dtype=dtype, device=device, pin_memory=pin, publication_group=group
        )
        b.np[:] = torch.randint(0, 1 << 20, (n,), generator=g).numpy()
        return b

    return {
        "seq_a": buf("seq_metadata"),
        "pos": buf("positions", dtype=torch.int64),
        "plan": buf("v4_plans"),
        "step": buf("v41_step", n=16),
        "block_tables": buf(None, n=12),
    }


_STEP_MEMBERS = {"step_metadata": ("plan", "step", "block_tables")}


def test_v41_shim_runner_has_the_builders_phase_groups():
    """The shim runner's publication owner gets native's phase groups: a
    builder-declared phase joins its members (shared destinations allowed,
    each once); those names are the packed transport's boundaries, and the
    producer groups stay addressable for their producers."""
    forward_vars = _publication_vars("cpu", False)
    _, groups = bridge._init_v41_publication(
        forward_vars,
        {"step_metadata": ("plan", "step", "block_tables", "plan")},
        "cpu",
    )
    assert [b.name for b in groups["step_metadata"].members] == [
        "plan",
        "step",
        "block_tables",
    ]
    assert [b.name for b in groups["v4_plans"].members] == ["plan"]
    # Off the GPU the transport stays direct (the packed arena needs a device).
    assert all(g.transport == "direct" for g in groups.values())


def test_gpu_packed_publication_writes_the_direct_bytes():
    """The plugin's packed publication vs a direct one built here (one copy a
    member) over the same sources: every destination identical, step after
    step (counts change, a member skipped, a member zeroed)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    dev = torch.device("cuda", torch.cuda.current_device())
    arms = {}
    fv = _publication_vars(dev, True, seed=7)
    arms[True] = (fv, *bridge._init_v41_publication(fv, _STEP_MEMBERS, dev))
    fv = _publication_vars(dev, True, seed=7)
    owner, groups = bridge._v41_publication_owner(fv, dev)
    for name, names in _STEP_MEMBERS.items():
        groups[name] = owner.group(name, [fv[item]._publication for item in names])
    arms[False] = (fv, owner, groups)
    assert arms[True][2]["step_metadata"].transport == "packed"
    assert all(g.transport == "direct" for g in arms[False][2].values())
    steps = [
        {"plan": 8, "step": 16, "block_tables": 12},
        {"plan": 3, "step": 5, "block_tables": 0},
        {"plan": 8, "step": 9},
    ]
    for i, counts in enumerate(steps):
        for fv, owner, groups in arms.values():
            owner.begin()
            g = torch.Generator().manual_seed(100 + i)
            for name in ("plan", "step", "block_tables", "seq_a", "pos"):
                fv[name].np[:] = torch.randint(
                    0, 1 << 20, (fv[name].np.shape[0],), generator=g
                ).numpy()
            groups["step_metadata"].set_counts(counts)
            groups["step_metadata"].publish()
            # Producer groups a packed phase covers stay direct when published
            # on their own, as the plugin's core staging publishes them.
            groups["seq_metadata"].set_counts({"seq_a": 8 - i})
            groups["seq_metadata"].publish()
            groups["positions"].set_counts({"pos": 8})
            groups["positions"].publish()
            owner.finish()
        torch.cuda.synchronize()
        for name in arms[True][0]:
            assert torch.equal(arms[True][0][name].gpu, arms[False][0][name].gpu), (
                i,
                name,
            )
