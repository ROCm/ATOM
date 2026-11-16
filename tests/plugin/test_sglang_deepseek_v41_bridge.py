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
``V41PoolGeometry``; the SGLang classes are the real ones when SGLang is
importable and the tests that need them skip otherwise.
"""

import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    DEEPSEEK_V41_ARCH,
    align_atom_config_with_sglang,
    build_deepseek_v41_forward_metadata,
    build_v41_scheduled_batch,
    is_deepseek_v41_config,
    refuse_unsupported_sglang_features,
    sglang_v41_kv_cache_dtype,
    v41_checkpoint_is_fp4ptpc,
    v41_default_online_quant_config,
)
from atom.plugin.sglang.runtime.model_arch import (
    MODEL_ADAPTER_SPECS,
    resolve_model_arch_spec,
)

# Real SGLang pieces, captured at collection so that a sibling test module
# that stubs ``sglang`` in ``sys.modules`` later cannot swap them out; the
# bridge imports them lazily, so tests reinstall them (`real_sglang`).
_SGLANG_MODULE_NAMES = (
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.model_executor.forward_context",
    "sglang.srt.model_executor.pool_configurator",
    "sglang.srt.mem_cache.base_swa_memory_pool",
    "sglang.srt.layers.attention.base_attn_backend",
)
try:
    import sglang.srt.layers.attention.base_attn_backend
    import sglang.srt.mem_cache.base_swa_memory_pool
    import sglang.srt.model_executor.forward_context
    import sglang.srt.model_executor.pool_configurator  # noqa: F401
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from atom.plugin.sglang.attention_backend.deepseek_v41_backend import (
        ATOMDeepseekV41BackendForSgl,
    )

    _SGLANG_MODULES = {name: sys.modules[name] for name in _SGLANG_MODULE_NAMES}
    HAVE_SGLANG = True
except Exception:  # noqa: BLE001 - any import failure means "no real SGLang"
    import enum

    class ForwardMode(enum.IntEnum):
        """The ForwardMode predicates the bridge calls, as SGLang 0.5.22 has them."""

        EXTEND = enum.auto()
        DECODE = enum.auto()
        MIXED = enum.auto()
        IDLE = enum.auto()
        TARGET_VERIFY = enum.auto()
        DRAFT_EXTEND_V2 = enum.auto()

        def is_decode(self):
            return self == ForwardMode.DECODE

        def is_idle(self):
            return self == ForwardMode.IDLE

        def is_context_parallel_extend(self, include_draft_extend_v2=False):
            return self in (ForwardMode.EXTEND, ForwardMode.MIXED) or (
                include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2
            )

    ATOMDeepseekV41BackendForSgl = None
    _SGLANG_MODULES = {}
    HAVE_SGLANG = False

needs_sglang = pytest.mark.skipif(not HAVE_SGLANG, reason="SGLang is not importable")

PAGE = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE
FLASH_CHECKPOINT = "/data/models/ginsongsong/DeepSeek-V4.1-Flash-FP4PTPC"


@pytest.fixture
def real_sglang(monkeypatch):
    if not HAVE_SGLANG:
        pytest.skip("SGLang is not importable")
    for name, module in _SGLANG_MODULES.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _SGLANG_MODULES


@pytest.fixture(autouse=True)
def _no_global_builder(monkeypatch):
    # The bridge keeps one builder per process; no test may leak one.
    monkeypatch.setattr(bridge, "_V41_BUILDER", None)


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
            SimpleNamespace(architectures=None, model_type="deepseek_v41_text")
        )

    def test_v41_found_through_hf_config_and_text_config(self):
        # SGLang's ModelConfig wraps the HF config; a multimodal HF config
        # nests the text config.
        assert is_deepseek_v41_config(
            SimpleNamespace(hf_config=_sglang_normalized_v41())
        )
        assert is_deepseek_v41_config(
            SimpleNamespace(
                architectures=["SomethingForConditionalGeneration"],
                model_type="some_vl",
                text_config=SimpleNamespace(model_type="deepseek_v41_text"),
            )
        )

    def test_v41_architecture_alone_is_v41(self):
        assert is_deepseek_v41_config(
            SimpleNamespace(architectures=[DEEPSEEK_V41_ARCH], model_type="")
        )

    @pytest.mark.parametrize(
        "config",
        [
            None,
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
            SimpleNamespace(),
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
    @pytest.mark.parametrize("spelling", [None, "auto", "bf16", "bfloat16", "AUTO"])
    def test_unquantized_spellings_select_bf16(self, spelling):
        assert sglang_v41_kv_cache_dtype(spelling) == "bf16"

    @pytest.mark.parametrize("spelling", ["fp8_e4m3", "FP8_E4M3"])
    def test_e4m3_selects_fp8(self, spelling):
        assert sglang_v41_kv_cache_dtype(spelling) == "fp8"

    @pytest.mark.parametrize("spelling", ["fp8_e5m2", "fp4_e2m1", "fp4", "float16"])
    def test_other_formats_are_refused(self, spelling):
        with pytest.raises(ValueError, match="kv-cache-dtype"):
            sglang_v41_kv_cache_dtype(spelling)


def _v41_with_quant(method):
    """SGLang's normalized V4.1 config: the checkpoint's quantization_config
    is kept on the root."""
    config = _sglang_normalized_v41()
    config.quantization_config = {"quant_method": method, "expert_dtype": "fp4"}
    return config


# The routed-expert conversion the plugin defaults to on gfx942.
GFX942_PTPC = {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}


class TestOnlineQuantDefault:
    def test_gfx942_converts_the_fp4ptpc_routed_experts(self):
        config = v41_default_online_quant_config(
            None, "gfx942", _v41_with_quant("w8a8_fp8")
        )
        assert config == GFX942_PTPC
        # Every routed expert the backbone and the DSpark draft build.
        from atom.config import QuantizationConfig

        patterns = list(config["layer_quant_config"].items())
        for prefix in ("layers.0.ffn.experts", "mtp.2.ffn.experts"):
            assert QuantizationConfig._pattern_spec(prefix, patterns) == "ptpc_fp8"

    def test_a_native_checkpoint_gets_none(self):
        # V4.1's runtime gate refuses online quantization of it.
        assert (
            v41_default_online_quant_config(None, "gfx942", _v41_with_quant("fp8"))
            is None
        )
        assert (
            v41_default_online_quant_config(None, "gfx942", _sglang_normalized_v41())
            is None
        )

    def test_the_operators_choice_and_other_archs_are_kept(self):
        mine = {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}
        fp4ptpc = _v41_with_quant("w8a8_fp8")
        assert v41_default_online_quant_config(mine, "gfx942", fp4ptpc) is mine
        assert v41_default_online_quant_config(None, "gfx950", fp4ptpc) is None

    def test_fp4ptpc_detection(self):
        assert v41_checkpoint_is_fp4ptpc(_v41_with_quant("w8a8_fp8"))
        assert not v41_checkpoint_is_fp4ptpc(_v41_with_quant("fp8"))
        assert not v41_checkpoint_is_fp4ptpc(_sglang_normalized_v41())
        # An ATOM-normalized config says so itself; its quant_method reads fp8.
        atom_normalized = _v41_with_quant("fp8")
        atom_normalized.fp4ptpc = True
        assert v41_checkpoint_is_fp4ptpc(atom_normalized)
        nested = SimpleNamespace(text_config=_v41_with_quant("w8a8_fp8"))
        assert v41_checkpoint_is_fp4ptpc(nested)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(FLASH_CHECKPOINT, "config.json")),
    reason="DeepSeek-V4.1-Flash checkpoint not present",
)
def test_flash_fp4ptpc_default_is_what_the_model_asks(monkeypatch):
    """The shipped FP4PTPC checkpoint under the gfx942 default: every routed
    expert, the draft's included, runs per-channel FP8, and every backbone
    layer's per-channel FP8 shared expert takes their fused slot -- the two
    things V4.1's MoE and DSpark draft refuse to build without."""
    from copy import copy

    from aiter import QuantType, dtypes

    if not dtypes.fp8.is_floating_point:
        pytest.skip("aiter resolves its FP8 dtype only on a GPU (or AOT replay)")

    import atom.model_ops.topK as topk
    from atom.models.deepseek_v4 import make_v4_quant_config
    from atom.models.deepseek_v41.config import normalize_hf_config
    from atom.models.deepseek_v41.layers import (
        converted_linear_spec,
        routed_experts_are_ptpc,
    )

    with open(os.path.join(FLASH_CHECKPOINT, "config.json")) as f:
        raw = json.load(f)
    sglang_view = _sglang_normalized_v41()
    sglang_view.quantization_config = raw["quantization_config"]
    online = v41_default_online_quant_config(None, "gfx942", sglang_view)
    assert online == GFX942_PTPC

    args = normalize_hf_config(raw)
    assert args.fp4ptpc
    backbone = make_v4_quant_config(
        args,
        online_quant_config=online,
        shared_expert_spec=converted_linear_spec(args),
        online_routed_experts=True,
    )
    # The single-rank TP plugin launch: no DP, no EP, no MoRI.
    monkeypatch.setattr(
        topk,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            quant_config=backbone,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            moe_ep_flatten_tp_across_dp=False,
            moe_all2all_backend="auto",
            enable_dp_attention=False,
            enable_expert_parallel=False,
            eplb_enable=False,
        ),
    )
    shared = converted_linear_spec(args)
    assert shared.quant_type == QuantType.per_Token and shared.quant_dtype == dtypes.fp8
    for layer in range(args.num_hidden_layers):
        prefix = f"layers.{layer}.ffn"
        assert routed_experts_are_ptpc(backbone, f"{prefix}.experts"), prefix
        assert topk.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
            backbone,
            shared_expert_prefix=f"{prefix}.shared_experts",
            routed_expert_prefix=f"{prefix}.experts",
        ), prefix

    # As `DeepseekV41DSpark` builds its MoE quant config.
    draft = copy(args)
    draft.n_routed_experts = args.dspark_n_routed_experts
    draft.num_experts_per_tok = args.dspark_num_experts_per_tok
    draft.fp4ptpc = False
    draft_quant = make_v4_quant_config(
        draft,
        online_quant_config=online,
        shared_expert_spec=converted_linear_spec(draft),
        online_routed_experts=True,
    )
    for stage in range(args.num_nextn_predict_layers):
        assert routed_experts_are_ptpc(draft_quant, f"mtp.{stage}.ffn.experts")


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------


def _supported_bags(**overrides):
    bags = {
        "memory": SimpleNamespace(
            disable_radix_cache=True,
            enable_hierarchical_cache=False,
            enable_hisparse=False,
        ),
        "exec": SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
        ),
        "spec": SimpleNamespace(speculative_algorithm=None),
        "disagg": SimpleNamespace(disaggregation_mode="null", language_model_only=True),
        "schedule": None,
    }
    bags.update(overrides)
    return bags


@pytest.fixture
def sglang_launch(monkeypatch):
    """Point the bridge at fake SGLang config bags and raw server args."""
    import atom.plugin.config as plugin_config

    state = SimpleNamespace(
        bags=_supported_bags(),
        args=SimpleNamespace(
            enable_dp_attention=False,
            pp_size=1,
            enable_two_batch_overlap=False,
            page_size=PAGE,
        ),
    )
    monkeypatch.setattr(bridge, "_sglang_bag", lambda name: state.bags.get(name))
    monkeypatch.setattr(plugin_config, "get_sglang_server_args", lambda: state.args)
    return state


class TestLaunchContract:
    def test_supported_launch_passes(self, sglang_launch):
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("bag", "field", "value", "match"),
        [
            ("memory", "enable_hierarchical_cache", True, "hierarchical cache"),
            ("memory", "enable_hisparse", True, "HiSparse"),
            ("spec", "speculative_algorithm", "EAGLE", "speculative algorithm EAGLE"),
            ("disagg", "disaggregation_mode", "prefill", "PD disaggregation"),
            ("disagg", "language_model_only", False, "language-model-only"),
        ],
    )
    def test_bag_features_are_refused(self, sglang_launch, bag, field, value, match):
        setattr(sglang_launch.bags[bag], field, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    def test_encoder_replay_is_refused(self, sglang_launch):
        sglang_launch.bags["exec"].features.enable_encoder_swa_bounded_replay = True
        with pytest.raises(ValueError, match="encoder SWA bounded replay"):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("enable_dp_attention", True, "DP attention"),
            ("pp_size", 2, "pipeline parallelism"),
            ("enable_two_batch_overlap", True, "two-batch overlap"),
        ],
    )
    def test_server_arg_features_are_refused(self, sglang_launch, field, value, match):
        setattr(sglang_launch.args, field, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    def test_raw_args_are_the_fallback_without_bags(self, sglang_launch):
        sglang_launch.bags = {}
        sglang_launch.args.disable_radix_cache = True
        sglang_launch.args.language_model_only = True
        refuse_unsupported_sglang_features()
        sglang_launch.args.enable_hierarchical_cache = True
        with pytest.raises(ValueError, match="hierarchical cache"):
            refuse_unsupported_sglang_features()

    def test_radix_kill_switch_refuses_a_launch_left_on(
        self, sglang_launch, monkeypatch
    ):
        # The launch policy turns SGLang's radix cache off for
        # ATOM_V41_RADIX=0; a launch where it did not run is refused.
        monkeypatch.setenv("ATOM_V41_RADIX", "0")
        sglang_launch.bags["memory"].disable_radix_cache = False
        with pytest.raises(ValueError, match="ATOM_V41_RADIX=0.*--disable-radix"):
            bridge.refuse_unsupported_sglang_features()

    def test_radix_cache_is_served_by_default(self, sglang_launch, monkeypatch):
        # On by default, radix cache itself is not refused; what it cannot
        # run beside is (`test_sglang_deepseek_v41_radix.py`).
        monkeypatch.delenv("ATOM_V41_RADIX", raising=False)
        sglang_launch.bags["memory"].disable_radix_cache = False
        sglang_launch.args.disable_overlap_schedule = True
        refuse_unsupported_sglang_features()
        # Overlap scheduling runs beside it (the radix seam drops a slot's
        # host rows on every checkpoint).
        sglang_launch.args.disable_overlap_schedule = False
        refuse_unsupported_sglang_features()
        sglang_launch.args.enable_mixed_chunk = True
        with pytest.raises(ValueError, match="mixed chunk with radix cache"):
            refuse_unsupported_sglang_features()

    def test_every_refusal_is_reported_at_once(self, sglang_launch):
        sglang_launch.bags["memory"].enable_hierarchical_cache = True
        sglang_launch.args.pp_size = 2
        with pytest.raises(
            ValueError, match="hierarchical cache.*pipeline parallelism"
        ):
            refuse_unsupported_sglang_features()

    def test_atom_config_takes_sglang_resolved_schedule(self, sglang_launch):
        sglang_launch.bags["schedule"] = SimpleNamespace(
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
            max_running_requests=256,
            mem_fraction_static=0.83,
        )
        config = SimpleNamespace(
            max_num_batched_tokens=4096, max_num_seqs=None, gpu_memory_utilization=None
        )
        align_atom_config_with_sglang(config)
        assert config.max_num_batched_tokens == 16384
        assert config.max_num_seqs == 256
        assert config.gpu_memory_utilization == 0.83

    def test_disabled_chunking_does_not_shrink_the_builder(self, sglang_launch):
        sglang_launch.bags["schedule"] = SimpleNamespace(
            chunked_prefill_size=-1,
            max_prefill_tokens=None,
            max_running_requests=None,
        )
        config = SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=64, gpu_memory_utilization=0.9
        )
        align_atom_config_with_sglang(config)
        assert (config.max_num_batched_tokens, config.max_num_seqs) == (16384, 64)
        assert config.gpu_memory_utilization == 0.9

    def test_unresolvable_mem_fraction_is_refused(self, sglang_launch):
        sglang_launch.bags["schedule"] = SimpleNamespace(mem_fraction_static=None)
        config = SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=64, gpu_memory_utilization=None
        )
        with pytest.raises(ValueError, match="mem-fraction-static"):
            align_atom_config_with_sglang(config)

    def test_no_schedule_bag_leaves_config_alone(self, sglang_launch):
        config = SimpleNamespace(max_num_batched_tokens=1, max_num_seqs=2)
        align_atom_config_with_sglang(config)
        assert (config.max_num_batched_tokens, config.max_num_seqs) == (1, 2)

    def test_page_size_must_be_one_v41_page(self, sglang_launch):
        assert bridge._sglang_page_size() == PAGE
        sglang_launch.args.page_size = None
        assert bridge._sglang_page_size() == PAGE
        sglang_launch.args.page_size = 64
        with pytest.raises(ValueError, match="page_size 256"):
            bridge._sglang_page_size()


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
    rids="default",
    cpu_lengths=True,
):
    bs = len(req_pool_indices)
    batch = SimpleNamespace(
        forward_mode=mode,
        rids=[f"r{i}" for i in range(bs)] if rids == "default" else rids,
        batch_size=bs,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64) if cpu_lengths else None,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        req_pool_indices_cpu=None,
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
        [
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.IDLE,
        ],
    )
    def test_unsupported_modes_are_refused(self, req_to_token, mode):
        fb = _forward_batch(mode, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(NotImplementedError, match="does not translate"):
            build_v41_scheduled_batch(fb, PAGE)

    def test_dummy_batches_are_refused(self, req_to_token):
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[0, 0], seq_lens=[1, 1], rids=None
        )
        with pytest.raises(NotImplementedError, match="dummy"):
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

    def test_missing_length_field_is_refused(self, req_to_token):
        fb = _forward_batch(ForwardMode.EXTEND, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(ValueError, match="missing a length field"):
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
        monkeypatch.setattr(bridge, "_CHECK_HOST_ROWS", False)
        monkeypatch.setattr(
            bridge, "V41_DSPARK_COUNTERS", dict.fromkeys(bridge.V41_DSPARK_COUNTERS, 0)
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
        return bridge.read_v41_extend_rows_at_alloc(batch, slots_cpu, slots_cpu)

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
        original = bridge._read_replay_rows

        def counted(*args, **kwargs):
            reads.append(args[2])
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge, "_read_replay_rows", counted)
        return reads

    def test_extend_takes_the_rows_its_allocation_copied(
        self, monkeypatch, req_to_token
    ):
        assert self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        reads = self._no_read_back(monkeypatch)
        batch = build_v41_scheduled_batch(self._extend([1, 2], [600, 10]), PAGE)
        assert reads == []
        assert _tables(batch) == [[7, 3, 9], [5]]
        assert all(row.dtype == np.int32 for row in batch.block_tables)
        assert batch.state_slots_committed == [1, 2]
        assert bridge._EXTEND_ROWS.pending is None
        assert bridge.V41_DSPARK_COUNTERS["extend_rows_early"] == 1
        assert bridge.V41_DSPARK_COUNTERS["extend_rows_read"] == 0
        # Graph replays stage off the same copies afterwards.
        rows = bridge._HOST_ROWS.lookup([1, 2], ["r0", "r1"], [3, 1])
        assert [row.tolist() for row in rows] == [[7, 3, 9], [5]]

    def test_chunk_with_a_prefix_takes_the_whole_row(self, monkeypatch, req_to_token):
        assert self._alloc(monkeypatch, req_to_token, [3], ["r0"], [3 * PAGE + 1])
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
        assert self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        reads = self._no_read_back(monkeypatch)
        build_v41_scheduled_batch(
            self._extend(other["slots"], other["ends"], rids=other["rids"]), PAGE
        )
        assert reads == [len(other["slots"])]
        assert bridge._EXTEND_ROWS.pending is None
        assert bridge.V41_DSPARK_COUNTERS["extend_rows_read"] == 1

    def test_a_copy_is_taken_once(self, monkeypatch, req_to_token):
        assert self._alloc(monkeypatch, req_to_token, [1], ["r0"], [600])
        reads = self._no_read_back(monkeypatch)
        build_v41_scheduled_batch(self._extend([1], [600]), PAGE)
        build_v41_scheduled_batch(self._extend([1], [600]), PAGE)
        assert reads == [1]

    def test_decode_never_takes_it(self, monkeypatch, req_to_token):
        assert self._alloc(monkeypatch, req_to_token, [1], ["r0"], [600])
        reads = self._no_read_back(monkeypatch)
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[600])
        build_v41_scheduled_batch(fb, PAGE)
        assert reads == [1]
        assert bridge._EXTEND_ROWS.pending is not None

    def test_check_mode_compares_with_the_device(self, monkeypatch, req_to_token):
        monkeypatch.setattr(bridge, "_CHECK_HOST_ROWS", True)
        assert self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        reads = self._no_read_back(monkeypatch)
        batch = build_v41_scheduled_batch(self._extend([1, 2], [600, 10]), PAGE)
        assert reads == [2]
        assert _tables(batch) == [[7, 3, 9], [5]]
        # A copy that is not what the device holds is refused.
        assert self._alloc(monkeypatch, req_to_token, [1, 2], ["r0", "r1"], [600, 10])
        bridge._EXTEND_ROWS.pending[3][0, 1] = 99
        with pytest.raises(RuntimeError, match="read at allocation disagree"):
            build_v41_scheduled_batch(self._extend([1, 2], [600, 10]), PAGE)

    def test_a_padding_page_in_the_copy_is_refused(self, monkeypatch, req_to_token):
        # Row 2 holds one page: a length past it names the padding page 0.
        assert self._alloc(monkeypatch, req_to_token, [2], ["r0"], [PAGE + 1])
        with pytest.raises(ValueError, match="padding page 0"):
            build_v41_scheduled_batch(self._extend([2], [PAGE + 1]), PAGE)

    def test_no_builder_issues_nothing(self, monkeypatch, req_to_token):
        monkeypatch.setattr(bridge, "_V41_BUILDER", None)
        bridge._EXTEND_ROWS.pending = ("stale",)
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="r0")],
            seq_lens_cpu=torch.tensor([600]),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        assert not bridge.read_v41_extend_rows_at_alloc(
            batch, torch.tensor([1]), torch.tensor([1])
        )
        assert bridge._EXTEND_ROWS.pending is None


@needs_sglang
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
        monkeypatch.setattr(bridge, "_HOST_ROWS_SUPPORTED", True)
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
        # Radix cache: no copy.
        monkeypatch.setattr(bridge, "_HOST_ROWS_SUPPORTED", False)
        calls.clear()
        wrapped("b")
        assert calls == ["alloc"]

    def test_kill_switch_and_version_guard(self, monkeypatch):
        from sglang.srt.managers import schedule_batch

        from atom.plugin.sglang.patches import v41_extend_rows_at_alloc_patch as patch

        original = schedule_batch.alloc_for_extend
        monkeypatch.setenv(bridge.V41_EXTEND_ROWS_AT_ALLOC_ENV, "0")
        assert not patch.install_v41_extend_rows_at_alloc_patch()
        assert schedule_batch.alloc_for_extend is original
        monkeypatch.delenv(bridge.V41_EXTEND_ROWS_AT_ALLOC_ENV)
        monkeypatch.setattr(patch, "_sglang_version", lambda: "0.5.99")
        assert not patch.install_v41_extend_rows_at_alloc_patch()
        assert schedule_batch.alloc_for_extend is original

        def returns_two(batch):
            return 1, 2

        monkeypatch.setattr(patch, "_sglang_version", lambda: "0.5.22")
        fake = SimpleNamespace(alloc_for_extend=returns_two)
        assert "no longer returns" in patch._upstream_shape_error(fake)


class _FakeAttnBackend:
    def __init__(self, table):
        self.req_to_token_pool = SimpleNamespace(req_to_token=table)


@needs_sglang
def test_req_to_token_comes_from_sglang_forward_context(real_sglang):
    forward_context = real_sglang["sglang.srt.model_executor.forward_context"]
    table = torch.arange(8, dtype=torch.int32).view(2, 4)
    with pytest.raises(RuntimeError, match="outside SGLang's forward context"):
        bridge._req_to_token()
    with forward_context.forward_context(
        forward_context.ForwardContext(attn_backend=_FakeAttnBackend(table))
    ):
        assert bridge._req_to_token() is table


# ---------------------------------------------------------------------------
# One step staged outside the forward
# ---------------------------------------------------------------------------


class _RecordingBuilder:
    def __init__(self, max_bs=4, max_num_batched_tokens=1024, positions=None):
        self.block_size = PAGE
        self.geometry = SimpleNamespace(speculative_tokens=0)
        self.max_bs = max_bs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.cache = object()
        self.calls = []
        self.positions = positions
        self.engram = SimpleNamespace(
            stage=lambda: self.calls.append("stage"),
            join=lambda: self.calls.append("join"),
        )

    def _metadata(self):
        return SimpleNamespace(engram_embeddings=self.engram)

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
    return build_deepseek_v41_forward_metadata(fb, positions)


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

    def test_positions_must_match_sglang(self, monkeypatch, req_to_token):
        monkeypatch.setattr(bridge, "_CHECK_POSITIONS", True)
        builder = _RecordingBuilder(positions=torch.tensor([4, 9]))
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[5, 11]
        )
        with pytest.raises(ValueError, match="positions disagree"):
            _staged(monkeypatch, builder, fb, torch.tensor([4, 10]))

    def test_positions_check_is_debug_only(self, monkeypatch, req_to_token):
        # The comparison is a device sync on every eager step, so it runs
        # only under ATOM_SGLANG_V41_CHECK_POSITIONS=1.
        assert bridge._CHECK_POSITIONS is (
            os.environ.get("ATOM_SGLANG_V41_CHECK_POSITIONS", "0") == "1"
        )
        monkeypatch.setattr(bridge, "_CHECK_POSITIONS", False)

        builder = _RecordingBuilder(positions=torch.tensor([4, 9]))
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[5, 11]
        )
        monkeypatch.setattr(
            bridge.torch,
            "equal",
            lambda *a: (_ for _ in ()).throw(AssertionError("synced")),
        )
        metadata = _staged(monkeypatch, builder, fb, torch.tensor([4, 10]))
        assert metadata.staged_outside_forward is True

    def test_step_over_builder_capacity_is_refused(self, monkeypatch, req_to_token):
        builder = _RecordingBuilder(max_bs=1)
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[5, 6]
        )
        with pytest.raises(ValueError, match="exceeds the builder"):
            _staged(monkeypatch, builder, fb, torch.tensor([4, 5]))
        builder = _RecordingBuilder(max_num_batched_tokens=8)
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1],
            seq_lens=[9],
            extend_seq_lens=[9],
            extend_prefix_lens=[0],
        )
        with pytest.raises(ValueError, match="exceeds the builder"):
            _staged(monkeypatch, builder, fb, torch.arange(9))
        assert builder.calls == []

    def test_input_ids_and_positions_must_agree(self, monkeypatch, req_to_token):
        monkeypatch.setattr(bridge, "_V41_BUILDER", _RecordingBuilder())
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        fb.input_ids = torch.zeros(2, dtype=torch.int64)
        with pytest.raises(ValueError, match="input_ids rows"):
            build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))

    def test_idle_is_refused(self, monkeypatch):
        monkeypatch.setattr(bridge, "_V41_BUILDER", _RecordingBuilder())
        fb = SimpleNamespace(forward_mode=ForwardMode.IDLE)
        with pytest.raises(NotImplementedError, match="IDLE"):
            build_deepseek_v41_forward_metadata(fb, torch.zeros(0))

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

    def test_no_verdict_without_replay(self, monkeypatch, req_to_token):
        builder = _RecordingBuilder(positions=torch.arange(2))
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1],
            seq_lens=[2],
            extend_seq_lens=[2],
            extend_prefix_lens=[0],
        )
        _staged(monkeypatch, builder, fb, torch.arange(2))
        assert not hasattr(builder.batch, "decoder_replays")

    def test_uninitialized_builder_or_pool_is_refused(self, monkeypatch):
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(RuntimeError, match="not initialized"):
            build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))
        builder = _RecordingBuilder()
        builder.cache = None
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(RuntimeError, match="proxy KV pool is not initialized"):
            build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))


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
        monkeypatch.setattr(
            bridge, "V41_DSPARK_COUNTERS", dict.fromkeys(bridge.V41_DSPARK_COUNTERS, 0)
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
        metadata = build_deepseek_v41_forward_metadata(
            fb, torch.tensor([PAGE, PAGE + 1, PAGE + 2, 0, 1])
        )
        # Laid out at the bucket; ids and positions at its rows.
        assert builder.calls[:2] == [("prefill", 2, 8), ("inputs", 8)]
        padded = metadata.sglang_padded_inputs
        assert padded.input_ids.tolist() == [11, 12, 13, 14, 15, 0, 0, 0]
        assert builder.ids.tolist() == [11, 12, 13, 14, 15, 0, 0, 0]
        assert padded.positions.tolist() == list(range(100, 108))
        assert padded.scheduled_tokens == 5
        assert padded.scheduled_tokens_gpu.tolist() == [5]
        assert padded.scheduled_tokens_gpu.dtype == torch.int32
        assert bridge.V41_DSPARK_COUNTERS["prefill_token_graph"] == 1

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
        metadata = build_deepseek_v41_forward_metadata(fb, torch.arange(20))
        assert builder.calls[:2] == [("prefill", 1, 20), ("inputs", 20)]
        assert getattr(metadata, "sglang_padded_inputs", None) is None

    def test_decode_is_never_padded(self, monkeypatch, req_to_token):
        builder = _PaddingBuilder(positions=torch.tensor([4]))
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        fb.input_ids = torch.tensor([7])
        metadata = build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))
        assert builder.calls[0] == ("decode", 1, 1, 1)
        assert getattr(metadata, "sglang_padded_inputs", None) is None

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv(bridge.V41_PREFILL_TOKEN_GRAPHS_ENV, "0")
        assert not bridge.v41_prefill_token_graphs_enabled()
        assert bridge.record_deepseek_v41_prefill_token_graphs(object()) == ()

    def _runner(self, *, dspark, replay):
        wrapper = SimpleNamespace(
            atom_config=SimpleNamespace(enable_dp_attention=False, enable_tbo=False),
            model=SimpleNamespace(replay=replay),
        )
        if dspark:
            wrapper._v41_dspark = SimpleNamespace(_rows={})
        return SimpleNamespace(model=wrapper, token_to_kv_pool=object())

    def test_dspark_without_decoder_replay_records_nothing(self, monkeypatch):
        # Without replay the recorded unit is `_Backbone`, the DSpark taps'
        # layers included: a replay would never fire them.
        monkeypatch.delenv(bridge.V41_PREFILL_TOKEN_GRAPHS_ENV, raising=False)
        monkeypatch.setattr(
            bridge, "_graph_builder", lambda: SimpleNamespace(supports_token_graph=True)
        )
        runner = self._runner(dspark=True, replay=False)
        assert bridge.record_deepseek_v41_prefill_token_graphs(runner) == ()
        # The runtime model without a `replay` attribute: refused the same way.
        runner.model.model = SimpleNamespace()
        assert bridge.record_deepseek_v41_prefill_token_graphs(runner) == ()

    @pytest.mark.parametrize("dspark,replay", [(True, True), (False, False)])
    def test_records_past_the_dspark_guard(self, monkeypatch, dspark, replay):
        # DSpark under replay, or no DSpark: the guard lets the recorder on to
        # the pool (here a stub without the builder's limits, which stops it).
        monkeypatch.delenv(bridge.V41_PREFILL_TOKEN_GRAPHS_ENV, raising=False)
        monkeypatch.setattr(
            bridge,
            "_graph_builder",
            lambda: SimpleNamespace(supports_token_graph=True),
        )
        runner = self._runner(dspark=dspark, replay=replay)
        with pytest.raises(AttributeError, match="max_num_batched_tokens"):
            bridge.record_deepseek_v41_prefill_token_graphs(runner)


@needs_sglang
class TestPrefillTokenGraphPatch:
    def _scheduler(self, *, draft=False, graphs=True, pool=True):
        return SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    is_draft_worker=draft,
                    token_to_kv_pool=SimpleNamespace(
                        is_atom_v41_proxy_pool=pool, is_atom_v41_draft_proxy=False
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
        assert patch.install_v41_prefill_token_graph_patch()
        wrapped = cls.init_all_cuda_graphs
        assert patch.install_v41_prefill_token_graph_patch()
        assert cls.init_all_cuda_graphs is wrapped
        scheduler = self._scheduler()
        wrapped(scheduler)
        assert calls == [
            "sglang graphs",
            ("record", scheduler.tp_worker.model_runner),
        ]
        # A draft runner, graphs off, or another pool: SGLang's graphs only.
        for other in (
            self._scheduler(draft=True),
            self._scheduler(graphs=False),
            self._scheduler(pool=False),
        ):
            calls.clear()
            wrapped(other)
            assert calls == ["sglang graphs"]

    def test_kill_switch_and_version_guard(self, monkeypatch):
        from sglang.srt.managers import scheduler as scheduler_module

        from atom.plugin.sglang.patches import v41_prefill_token_graph_patch as patch

        cls = scheduler_module.Scheduler
        original = cls.init_all_cuda_graphs
        monkeypatch.setenv(bridge.V41_PREFILL_TOKEN_GRAPHS_ENV, "0")
        assert not patch.install_v41_prefill_token_graph_patch()
        assert cls.init_all_cuda_graphs is original
        monkeypatch.delenv(bridge.V41_PREFILL_TOKEN_GRAPHS_ENV)
        monkeypatch.setattr(patch, "_sglang_version", lambda: "0.5.99")
        assert not patch.install_v41_prefill_token_graph_patch()
        assert cls.init_all_cuda_graphs is original


@needs_sglang
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
        score_workspace=None,
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
        "num_req_slots": 4,
    }
    kwargs.update(overrides)
    return bridge._make_proxy_pool_cls()(**kwargs)


@needs_sglang
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

    def test_default_slots_cover_padding_row(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        pool = _proxy(num_req_slots=None)
        assert pool.num_req_slots == 3 + 1
        assert pool.num_slots == 3 + 1 + CAPTURE_SLOTS

    def test_full_size_falls_back_to_swa_size(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        pool = _proxy(full_size=None, swa_size=2 * PAGE)
        assert (pool.size, pool.num_pages) == (2 * PAGE, 3)

    def test_refusals(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        # The DSpark draft role binds the target's arena, so it needs one.
        with pytest.raises(RuntimeError, match="draft pool built before"):
            _proxy(is_draft_worker=True)
        with pytest.raises(ValueError, match="page_size 64"):
            _proxy(page_size=64)
        with pytest.raises(ValueError, match="full token count"):
            _proxy(full_size=None, swa_size=0)


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
            (torch.zeros(100, dtype=torch.int8), "flat uint8"),
            (torch.zeros(10, 10, dtype=torch.uint8), "flat uint8"),
            (torch.zeros(200, dtype=torch.uint8)[::2], "contiguous"),
            (torch.zeros(63, dtype=torch.uint8), "short"),
        ],
    )
    def test_bad_backings_are_refused(self, cache_cls, backing, match):
        cache = cache_cls.__new__(cache_cls)
        cache._external_backing = backing
        with pytest.raises(ValueError, match=match):
            cache._allocate_backing(64, "cpu")


def _kvc(context_len=4096, attn_dp_size=1, hf_config=None, atom_config=True):
    return SimpleNamespace(
        page_size=PAGE,
        attn_dp_size=attn_dp_size,
        is_draft_worker=False,
        model_config=SimpleNamespace(
            context_len=context_len,
            hf_config=hf_config or _sglang_normalized_v41(),
        ),
        model=SimpleNamespace(atom_config=object() if atom_config else None),
    )


@pytest.fixture
def sizing(real_sglang, monkeypatch):
    """The real configurator over a small geometry and fake schedule bags."""
    geometry = _small_geometry()
    _install_builder(monkeypatch, geometry)
    bags = {"schedule": SimpleNamespace(max_running_requests=4), "disagg": None}
    monkeypatch.setattr(bridge, "_sglang_bag", lambda name: bags.get(name))
    return SimpleNamespace(
        geometry=geometry, bags=bags, cls=bridge._make_pool_configurator_cls()
    )


def _pool_cost(geometry, pages, slots, context_len):
    return (
        geometry.paged_extents(pages + 1)[1]
        + (slots + CAPTURE_SLOTS) * geometry.state_bytes
        + slots * context_len * 4
    )


@needs_sglang
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

    def test_requests_are_split_over_attention_dp(self, sizing):
        sizing.bags["schedule"].max_running_requests = 16
        configurator = sizing.cls(_kvc(attn_dp_size=4))
        assert configurator.requested_reqs == 4

    def test_pd_decode_reserves_extra_slots(self, sizing):
        sizing.bags["disagg"] = SimpleNamespace(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=3
        )
        assert sizing.cls(_kvc()).extra_slots == 3
        sizing.bags["disagg"].disaggregation_mode = "prefill"
        assert sizing.cls(_kvc()).extra_slots == 0

    def test_unset_request_count_follows_sglang_estimate(self, sizing):
        sizing.bags["schedule"].max_running_requests = None
        configurator = sizing.cls(_kvc(context_len=4096))
        available = 10_000 * sizing.geometry.paged_bytes
        tokens = 10_000 * PAGE
        assert configurator._reqs(available) == min(
            max(min(int(tokens / 4096 * 512), 4096), 2048), tokens // 2
        )

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
        assert not bridge._is_atom_v41_target(_kvc(atom_config=False))
        draft = _kvc()
        draft.is_draft_worker = True
        assert not bridge._is_atom_v41_target(draft)


@needs_sglang
def test_sizing_patch_routes_only_v41(real_sglang, monkeypatch):
    pool_configurator = real_sglang["sglang.srt.model_executor.pool_configurator"]
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    _install_builder(monkeypatch, _small_geometry())
    monkeypatch.setattr(bridge, "_sglang_bag", lambda name: None)
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


@pytest.mark.skipif(
    not os.path.exists(os.path.join(FLASH_CHECKPOINT, "config.json")),
    reason="DeepSeek-V4.1-Flash checkpoint not present",
)
@pytest.mark.parametrize(
    ("kv_dtype", "paged_bytes", "state_bytes"),
    [("bf16", 739_840, 5_276_672), ("fp8", 422_400, 2_745_600)],
)
def test_flash_geometry_prices(kv_dtype, paged_bytes, state_bytes):
    """What the shipped Flash checkpoint costs per 256-token PAGE and per
    request at SGLang's page size: 2890 B/token in bf16 (1650 in fp8), far
    above the ~1630 B/token SGLang's own DSV4 configurator would charge."""
    from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry

    with open(os.path.join(FLASH_CHECKPOINT, "config.json")) as f:
        text_config = SimpleNamespace(**json.load(f)["text_config"])
    geometry = build_v41_pool_geometry(
        text_config,
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
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
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

    def test_refused_under_a_capture(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="under a capture"):
            bridge.compact_v41_moe_buffers(torch.nn.Module())

    @needs_sglang
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
        model = SimpleNamespace(atom_config=config)
        kvc = SimpleNamespace(model=model)
        assert KVCacheConfigurator._profile_available_bytes(kvc, 0) == 7
        assert calls == [("compact", model, "target"), ("builder", config)]
        # A second profile builds nothing new and compacts nothing.
        calls.clear()
        KVCacheConfigurator._profile_available_bytes(kvc, 0)
        assert calls == [("builder", config)]


@needs_sglang
class TestBackendShim:
    def _runner(self, pool):
        return SimpleNamespace(
            device="cpu", token_to_kv_pool=pool, req_to_token_pool=object()
        )

    def _backend(self):
        pool = SimpleNamespace(is_atom_v41_proxy_pool=True)
        return ATOMDeepseekV41BackendForSgl(self._runner(pool))

    def test_needs_the_proxy_pool(self):
        with pytest.raises(RuntimeError, match="proxy KV pool"):
            ATOMDeepseekV41BackendForSgl(self._runner(SimpleNamespace()))

    def test_publishes_pools_and_keeps_no_step_state(self):
        backend = self._backend()
        assert backend.get_name() == "dsv4"
        assert backend.needs_cpu_seq_lens
        backend.init_forward_metadata(object())
        assert backend.forward_metadata is None

    @pytest.mark.parametrize("method", ["forward_decode", "forward_extend"])
    def test_attention_never_runs_in_the_shim(self, method):
        with pytest.raises(RuntimeError, match="executes inside the ATOM model"):
            getattr(self._backend(), method)(None, None, None, None, None)

    def test_graph_staging_needs_graph_state(self):
        backend = self._backend()
        with pytest.raises(RuntimeError, match="before init_cuda_graph_state"):
            backend.init_forward_metadata_out_graph(
                SimpleNamespace(forward_mode=ForwardMode.DECODE), in_capture=True
            )


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

    def test_window_spanning_pages(self, capture_builder):
        builder, recorder = capture_builder
        builder.cache.geometry.window_size = PAGE
        builder.build_for_sglang_capture(
            1, input_ids=torch.zeros(1, dtype=torch.int64), slots=(5,)
        )
        assert recorder.prepared[0][0].block_tables == ((0, 0),)

    @pytest.mark.parametrize(
        ("bs", "slots", "rows", "match"),
        [
            (2, (5, 5), 2, "distinct STATE slots"),
            (2, (5,), 2, "distinct STATE slots"),
            (2, (5, 6), 3, "input_ids rows"),
            (9, tuple(range(9)), 9, "exceeds"),
            (0, (), 0, "exceeds"),
        ],
    )
    def test_refusals(self, capture_builder, bs, slots, rows, match):
        builder, recorder = capture_builder
        with pytest.raises(ValueError, match=match):
            builder.build_for_sglang_capture(
                bs, input_ids=torch.zeros(rows, dtype=torch.int64), slots=slots
            )
        assert recorder.prepared == []

    def test_speculative_and_unallocated_are_refused(self, capture_builder):
        builder, _ = capture_builder
        ids = torch.zeros(1, dtype=torch.int64)
        builder.cache.geometry.speculative_tokens = 2
        with pytest.raises(NotImplementedError, match="speculative"):
            builder.build_for_sglang_capture(1, input_ids=ids, slots=(5,))
        builder.cache = None
        with pytest.raises(RuntimeError, match="serving cache"):
            builder.build_for_sglang_capture(1, input_ids=ids, slots=(5,))

    def test_a_decode_round_is_refused(self, capture_builder):
        from atom.utils.forward_context import RoundService

        builder, recorder = capture_builder

        def served(*args, **kwargs):
            metadata, positions = recorder.prepare(*args, **kwargs)
            metadata.served_by_round = RoundService.FORWARD
            return metadata, positions

        builder._prepare = served
        with pytest.raises(NotImplementedError, match="decode round"):
            builder.build_for_sglang_capture(
                1, input_ids=torch.zeros(1, dtype=torch.int64), slots=(5,)
            )


class _GraphBuilder:
    """A builder that records the graph staging calls the bridge makes."""

    def __init__(self, max_bs=8, cache=None):
        self.block_size = PAGE
        self.max_bs = max_bs
        self.max_num_batched_tokens = 64
        self.cache = cache if cache is not None else SimpleNamespace()
        self.engram = None
        self.calls = []
        self.step_width = None

    def build_for_sglang_capture(self, bs, *, input_ids, slots):
        self.calls.append(("capture", bs, input_ids, tuple(slots)))
        return _graph_metadata(self, bs)

    def prepare_decode(self, batch, running_bs, running_tokens, max_seqlen_q):
        self.calls.append(("decode", running_bs, running_tokens, max_seqlen_q))
        self.batch = batch
        width = running_tokens if self.step_width is None else self.step_width
        metadata = _graph_metadata(self, width)
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


def _fb_view(
    req_pool_indices,
    seq_lens,
    bs,
    mode=ForwardMode.DECODE,
    cpu_lengths=True,
    input_ids=None,
):
    raw = len(req_pool_indices)
    pad = bs - raw
    lengths = torch.tensor(list(seq_lens) + [1] * pad, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        num_padding=pad,
        forward_mode=ForwardMode.DECODE,
        actual_forward_mode=mode,
        input_ids=(
            input_ids if input_ids is not None else torch.zeros(bs, dtype=torch.int64)
        ),
        req_pool_indices=torch.tensor(
            list(req_pool_indices) + [0] * pad, dtype=torch.int64
        ),
        seq_lens=lengths.clone(),
        seq_lens_cpu=lengths if cpu_lengths else None,
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

    def test_non_decode_and_missing_mirror_are_refused(self, monkeypatch, replay_table):
        builder = _GraphBuilder()
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(NotImplementedError, match="IDLE"):
            bridge.stage_deepseek_v41_graph_replay(
                _fb_view([1], [5], 2, mode=ForwardMode.IDLE), replay_table
            )
        with pytest.raises(ValueError, match="host seq_lens mirror"):
            bridge.stage_deepseek_v41_graph_replay(
                _fb_view([1], [5], 2, cpu_lengths=False), replay_table
            )
        with pytest.raises(ValueError, match="input rows"):
            bridge.stage_deepseek_v41_graph_replay(
                _fb_view([1], [5], 2, input_ids=torch.zeros(3)), replay_table
            )
        assert builder.calls == []

    def test_a_step_wider_than_the_graph_is_refused(self, monkeypatch, replay_table):
        builder = _GraphBuilder()
        builder.step_width = 4
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(RuntimeError, match="did not record"):
            bridge.stage_deepseek_v41_graph_replay(_fb_view([1], [5], 2), replay_table)


@pytest.fixture(autouse=True)
def _fresh_host_rows(monkeypatch):
    # The bridge keeps one host PAGE-row cache per process; no test may leak one.
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
    monkeypatch.setattr(bridge, "_CHECK_HOST_ROWS", False)
    monkeypatch.setattr(bridge, "_HOST_ROWS_SUPPORTED", True)


def _mirrored_view(req_pool_indices, seq_lens, bs, rids):
    """A replay view carrying the host request rows (`_install_host_mirror_patch`)."""
    fb = _fb_view(req_pool_indices, seq_lens, bs)
    fb.atom_v41_req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    fb.atom_v41_rids = list(rids)
    return fb


@pytest.fixture
def device_reads(monkeypatch):
    reads = []
    original = bridge._read_replay_rows

    def counted(*args, **kwargs):
        reads.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "_read_replay_rows", counted)
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
            monkeypatch, replay_table, _mirrored_view([3, 1, 2], [700, 300, 9], 4, rids)
        )
        again = self._replay(
            monkeypatch,
            replay_table,
            _mirrored_view([3, 1, 2], [701, 301, 10], 4, rids),
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
        # The extend itself reads its rows (and slots) from the device once.
        assert device_reads == [2]
        device_reads.clear()
        batch = self._replay(
            monkeypatch,
            req_to_token,
            _mirrored_view([3, 1], [2 * PAGE + 2, PAGE + 6], 2, ["y", "x"]),
        )
        assert device_reads == []
        assert _tables(batch) == [[2, 11, 4], [7, 3]]

    def test_eager_host_slots_must_match_the_device(self, req_to_token):
        fb = _forward_batch(
            ForwardMode.EXTEND,
            req_pool_indices=[1, 3],
            seq_lens=[5, 6],
            extend_seq_lens=[5, 6],
            extend_prefix_lens=[0, 0],
            rids=["x", "y"],
        )
        fb.req_pool_indices_cpu = torch.tensor([3, 1])
        with pytest.raises(RuntimeError, match="disagree with the device"):
            build_v41_scheduled_batch(fb, PAGE)

    def test_a_new_page_or_tenant_reads_the_device(
        self, monkeypatch, replay_table, device_reads
    ):
        self._replay(monkeypatch, replay_table, _mirrored_view([1], [PAGE], 1, ["a"]))
        # Entering its second page.
        batch = self._replay(
            monkeypatch, replay_table, _mirrored_view([1], [PAGE + 1], 1, ["a"])
        )
        assert _tables(batch) == [[7, 3]]
        # Another request in the same slot.
        self._replay(monkeypatch, replay_table, _mirrored_view([1], [5], 1, ["b"]))
        assert device_reads == [1, 1, 1]

    def test_without_host_rows_every_replay_reads(
        self, monkeypatch, replay_table, device_reads
    ):
        for _ in range(2):
            self._replay(monkeypatch, replay_table, _fb_view([1], [5], 1))
        assert device_reads == [1, 1]

    def test_check_mode_catches_a_stale_copy(
        self, monkeypatch, replay_table, device_reads
    ):
        bridge._HOST_ROWS.record([1], ["a"], [np.asarray([7, 99], np.int32)])
        monkeypatch.setattr(bridge, "_CHECK_HOST_ROWS", True)
        with pytest.raises(RuntimeError, match="disagree with the device"):
            self._replay(
                monkeypatch, replay_table, _mirrored_view([1], [PAGE + 1], 1, ["a"])
            )

    def test_host_slots_must_match_the_device(
        self, monkeypatch, replay_table, device_reads
    ):
        view = _mirrored_view([1, 2], [5, 6], 2, ["a", "b"])
        view.atom_v41_req_pool_indices_cpu = torch.tensor([2, 1])
        with pytest.raises(RuntimeError, match="disagree with the device"):
            self._replay(monkeypatch, replay_table, view)

    def test_host_slots_without_rids_read_the_device(
        self, monkeypatch, replay_table, device_reads
    ):
        view = _mirrored_view([1], [5], 1, ["a"])
        view.atom_v41_rids = None
        self._replay(monkeypatch, replay_table, view)
        assert device_reads == [1]

    def test_radix_cache_disables_the_host_copies(
        self, monkeypatch, replay_table, device_reads
    ):
        monkeypatch.setattr(bridge, "_HOST_ROWS_SUPPORTED", False)
        for _ in range(2):
            self._replay(monkeypatch, replay_table, _mirrored_view([1], [5], 1, ["a"]))
        assert device_reads == [1, 1]

    def test_mirror_width_must_be_the_live_rows(self, monkeypatch, replay_table):
        view = _mirrored_view([1, 2], [5, 6], 4, ["a", "b"])
        view.atom_v41_rids = ["a"]
        with pytest.raises(ValueError, match="request ids"):
            self._replay(monkeypatch, replay_table, view)


@needs_sglang
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
        return SimpleNamespace(mode=mode, rids=["a", "b"], req_pool_indices_cpu=slots)

    decode = ForwardBatch.init_new(batch(ForwardMode.DECODE), v41)
    assert decode.atom_v41_req_pool_indices_cpu is slots
    # DSpark's verify is built from the ScheduleBatch too (prepare_for_verify).
    verify = ForwardBatch.init_new(batch(ForwardMode.TARGET_VERIFY), v41)
    assert verify.atom_v41_req_pool_indices_cpu is slots
    extend = ForwardBatch.init_new(batch(ForwardMode.EXTEND), v41)
    assert not hasattr(extend, "atom_v41_req_pool_indices_cpu")
    other = ForwardBatch.init_new(batch(ForwardMode.DECODE), v4)
    assert not hasattr(other, "atom_v41_req_pool_indices_cpu")

    view = runner_module.build_replay_fb_view(forward_batch=decode, bs=2)
    assert view.atom_v41_req_pool_indices_cpu is slots
    assert view.atom_v41_rids == ["a", "b"]
    assert not hasattr(
        runner_module.build_replay_fb_view(forward_batch=other, bs=2),
        "atom_v41_rids",
    )


class TestGraphAdapter:
    def _adapter(self):
        return MODEL_ADAPTER_SPECS[DEEPSEEK_V41_ARCH].build_forward_metadata

    def test_capture_stash_is_returned_untouched(self, monkeypatch):
        monkeypatch.setattr(
            bridge,
            "build_deepseek_v41_forward_metadata",
            lambda *a: pytest.fail("eager staging ran for a captured step"),
        )
        stash = object()
        fb = SimpleNamespace(atom_v41_graph_metadata=stash)
        assert self._adapter()(None, fb, torch.zeros(1)) is stash

    def test_capture_without_a_stash_is_refused(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        monkeypatch.setattr(
            bridge,
            "build_deepseek_v41_forward_metadata",
            lambda *a: pytest.fail("staged under capture"),
        )
        with pytest.raises(RuntimeError, match="not staged before capture"):
            self._adapter()(None, SimpleNamespace(), torch.zeros(1))

    def test_eager_still_stages_in_the_forward(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            bridge,
            "build_deepseek_v41_forward_metadata",
            lambda fb, positions: calls.append((fb, positions)) or "eager",
        )
        fb, positions = SimpleNamespace(), torch.zeros(1)
        assert self._adapter()(None, fb, positions) == "eager"
        assert calls == [(fb, positions)]


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
    )
    monkeypatch.delenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", raising=False)
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


@needs_sglang
class TestGraphBackend:
    def test_capture_slots_follow_the_request_rows(self, graph_backend):
        pool = graph_backend.pool
        assert pool.capture_slot_base == pool.num_req_slots == 4
        assert pool.num_capture_slots == CAPTURE_SLOTS
        assert pool.capture_slots == (4, 5)
        assert pool.num_slots == 6

    def test_graph_state_is_validated(self, graph_backend, monkeypatch):
        backend = graph_backend.backend
        with pytest.raises(ValueError, match="exceed the builder"):
            backend.init_cuda_graph_state(CAPTURE_SLOTS + 1, CAPTURE_SLOTS + 1)
        with pytest.raises(ValueError, match="exceed the builder"):
            backend.init_cuda_graph_state(1, 64)
        graph_backend.builder.max_bs = 8
        with pytest.raises(ValueError, match="capture slots"):
            backend.init_cuda_graph_state(4, 4)
        graph_backend.builder.max_bs = CAPTURE_SLOTS
        # A req_to_token row past the base would alias a capture slot.
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros(5, PAGE, dtype=torch.int32)
        )
        with pytest.raises(ValueError, match="inside the 5 req_to_token rows"):
            backend.init_cuda_graph_state(2, 2)
        backend.req_to_token_pool = SimpleNamespace(req_to_token=graph_backend.table)
        backend.init_cuda_graph_state(2, 2)
        assert backend._v41_graph_buckets == {}

    def test_glue_graph_is_refused(self, graph_backend, monkeypatch):
        monkeypatch.setenv("SGLANG_ENABLE_METADATA_GLUE_GRAPH", "1")
        with pytest.raises(NotImplementedError, match="GLUE_GRAPH"):
            graph_backend.backend.init_cuda_graph_state(2, 2)

    def test_missing_or_foreign_cache_is_refused(self, graph_backend):
        builder = graph_backend.builder
        cache = builder.cache
        builder.cache = SimpleNamespace(backing=torch.zeros(4, dtype=torch.uint8))
        with pytest.raises(RuntimeError, match="not backed by the SGLang proxy"):
            graph_backend.backend.init_cuda_graph_state(2, 2)
        builder.cache = None
        with pytest.raises(RuntimeError, match="not initialized"):
            graph_backend.backend.init_cuda_graph_state(2, 2)
        builder.cache = cache

    def test_host_engram_lookup_is_refused(self, graph_backend):
        graph_backend.builder.engram = SimpleNamespace(
            host=SimpleNamespace(overlap=None)
        )
        with pytest.raises(NotImplementedError, match="device Engram"):
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
        assert backend._v41_graph_buckets == {
            2: (fb.input_ids.data_ptr(), graph_backend.pool.raw_arena.data_ptr())
        }
        assert backend.forward_metadata is None

    def test_capture_is_decode_only(self, graph_backend):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        fb = _capture_fb(1)
        fb.forward_mode = ForwardMode.TARGET_VERIFY
        with pytest.raises(NotImplementedError, match="plain decode only"):
            backend.init_forward_metadata_out_graph(fb, in_capture=True)

    def test_replay_stages_on_the_recorded_addresses(self, graph_backend, monkeypatch):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        fb = self._capture(graph_backend, 2)
        staged = []
        monkeypatch.setattr(
            bridge,
            "stage_deepseek_v41_graph_replay",
            lambda view, table: staged.append((view, table)) or "replay-md",
        )
        decode = ForwardMode.DECODE
        view = SimpleNamespace(
            batch_size=2, num_padding=1, input_ids=fb.input_ids, forward_mode=decode
        )
        backend.init_forward_metadata_out_graph(view)
        assert staged == [(view, graph_backend.table)]
        assert backend._v41_last_replay_md == "replay-md"
        assert backend._v41_replayed_buckets == {2}

        # A bucket never captured, or one replaying elsewhere, is refused.
        with pytest.raises(RuntimeError, match="was not captured"):
            backend.init_forward_metadata_out_graph(
                SimpleNamespace(
                    batch_size=1, input_ids=fb.input_ids[:1], forward_mode=decode
                )
            )
        with pytest.raises(RuntimeError, match="did not record"):
            backend.init_forward_metadata_out_graph(
                SimpleNamespace(
                    batch_size=2, input_ids=torch.zeros(2), forward_mode=decode
                )
            )
        assert len(staged) == 1

    def test_eager_metadata_is_untouched(self, graph_backend):
        backend = graph_backend.backend
        backend.init_cuda_graph_state(2, 2)
        backend.init_forward_metadata(object())
        assert backend.forward_metadata is None
        assert backend._v41_graph_buckets == {}


@needs_sglang
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
    fb = SimpleNamespace(**fields)
    batch = SimpleNamespace(
        num_scheduled_tokens=np.asarray(lengths, np.int32),
        context_lens=np.asarray(prefix, np.int64) + np.asarray(lengths, np.int64),
    )
    return bridge.v41_extend_replays(fb, batch, window)


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
        fb = SimpleNamespace(capture_hidden_mode=full)
        batch = SimpleNamespace(
            num_scheduled_tokens=np.asarray([4096], np.int32),
            context_lens=np.asarray([4096], np.int64),
        )
        assert bridge.v41_extend_replays(fb, batch, 128, dspark_context_in_forward=True)
        assert not bridge.v41_extend_replays(
            SimpleNamespace(capture_hidden_mode=full, is_prefill_only=True),
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
