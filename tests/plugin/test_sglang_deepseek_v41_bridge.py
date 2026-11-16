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
            ("memory", "disable_radix_cache", False, "radix cache"),
            ("memory", "enable_hierarchical_cache", True, "hierarchical cache"),
            ("memory", "enable_hisparse", True, "HiSparse"),
            ("spec", "speculative_algorithm", "EAGLE", "speculative decoding"),
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
        sglang_launch.args.disable_radix_cache = False
        with pytest.raises(ValueError, match="radix cache"):
            refuse_unsupported_sglang_features()

    def test_every_refusal_is_reported_at_once(self, sglang_launch):
        sglang_launch.bags["memory"].disable_radix_cache = False
        sglang_launch.args.pp_size = 2
        with pytest.raises(ValueError, match="radix cache.*pipeline parallelism"):
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

    @pytest.mark.parametrize("slots", [[1, 1], [0, 2]])
    def test_slots_must_be_live_and_unique(self, req_to_token, slots):
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=slots, seq_lens=[3, 4])
        with pytest.raises(ValueError, match="req_pool_idx"):
            build_v41_scheduled_batch(fb, PAGE)


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
        builder = _RecordingBuilder(positions=torch.tensor([4, 9]))
        fb = _forward_batch(
            ForwardMode.DECODE, req_pool_indices=[1, 2], seq_lens=[5, 11]
        )
        with pytest.raises(ValueError, match="positions disagree"):
            _staged(monkeypatch, builder, fb, torch.tensor([4, 10]))

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

    def test_uninitialized_builder_or_pool_is_refused(self, monkeypatch):
        fb = _forward_batch(ForwardMode.DECODE, req_pool_indices=[1], seq_lens=[5])
        with pytest.raises(RuntimeError, match="not initialized"):
            build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))
        builder = _RecordingBuilder()
        builder.cache = None
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        with pytest.raises(RuntimeError, match="proxy KV pool is not initialized"):
            build_deepseek_v41_forward_metadata(fb, torch.tensor([4]))


# ---------------------------------------------------------------------------
# KV pool: the proxy's arena and SGLang's sizing of it
# ---------------------------------------------------------------------------


def _install_builder(monkeypatch, geometry, max_num_batched_tokens=32):
    builder = SimpleNamespace(
        geometry=geometry,
        device=torch.device("cpu"),
        max_num_batched_tokens=max_num_batched_tokens,
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

        # 4 usable PAGEs and SGLang's padding page 0; one slot per req row.
        assert pool.num_pages == 5
        assert pool.num_slots == 4
        expected = geometry.paged_extents(5)[1] + 4 * geometry.state_bytes
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
        assert (cache.num_pages, cache.num_slots) == (5, 4)
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
        with pytest.raises(NotImplementedError, match="radix-cache"):
            pool.move_kv_cache(torch.zeros(1), torch.zeros(1))
        with pytest.raises(NotImplementedError):
            pool.get_kv_buffer(0)
        with pytest.raises(NotImplementedError):
            pool.set_kv_buffer(None, None, None, None)
        with pytest.raises(NotImplementedError):
            pool.translate_loc_from_full_to_swa(torch.zeros(1))

    def test_default_slots_cover_padding_row(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        assert _proxy(num_req_slots=None).num_slots == 3 + 1

    def test_full_size_falls_back_to_swa_size(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        pool = _proxy(full_size=None, swa_size=2 * PAGE)
        assert (pool.size, pool.num_pages) == (2 * PAGE, 3)

    def test_refusals(self, real_sglang, monkeypatch):
        _install_builder(monkeypatch, _small_geometry())
        with pytest.raises(NotImplementedError, match="draft"):
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
        + slots * geometry.state_bytes
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
        backend.init_forward_metadata_out_graph(object())
        assert backend.forward_metadata is None

    @pytest.mark.parametrize("method", ["forward_decode", "forward_extend"])
    def test_attention_never_runs_in_the_shim(self, method):
        with pytest.raises(RuntimeError, match="executes inside the ATOM model"):
            getattr(self._backend(), method)(None, None, None, None, None)

    def test_cuda_graphs_are_refused(self):
        backend = self._backend()
        with pytest.raises(NotImplementedError, match="CUDA graphs"):
            backend.init_cuda_graph_state(8, 8)
        with pytest.raises(NotImplementedError, match="CUDA graphs"):
            backend.init_forward_metadata_out_graph(object(), in_capture=True)
