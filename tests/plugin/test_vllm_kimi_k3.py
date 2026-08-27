import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_without_test_stubs(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kimi_k3_plugin_registries_are_synchronized():
    from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
    from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES

    arch = "KimiK3ForConditionalGeneration"
    assert (
        _VLLM_MODEL_REGISTRY_OVERRIDES[arch]
        == "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLMVllm"
    )
    assert (
        _ATOM_MODEL_CLASSES[arch]
        == "atom.plugin.vllm.models.kimi_k3:KimiK3ForConditionalGeneration_"
    )


def test_importing_vllm_plugin_does_not_require_vllm():
    _run_without_test_stubs("""
        import builtins

        original_import = builtins.__import__

        def import_without_vllm(name, *args, **kwargs):
            if name == "vllm" or name.startswith("vllm."):
                raise ModuleNotFoundError("No module named 'vllm'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_vllm
        try:
            import atom.plugin.vllm.register
        finally:
            builtins.__import__ = original_import
        """)


def test_atom_patch_preserves_rocm_dcp_full_decode_cuda_graph_mode():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        from vllm.config.compilation import CUDAGraphMode
        from vllm.platforms.rocm import RocmPlatform

        from atom.plugin.vllm.rocm_dcp_full_graph_patch import (
            apply_vllm_rocm_dcp_full_graph_patch,
        )

        apply_vllm_rocm_dcp_full_graph_patch()

        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=8,
                prefill_context_parallel_size=1,
                worker_cls="auto",
            ),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
            ),
        )
        RocmPlatform.check_and_update_config(config)
        assert (
            config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
        )

        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.prefill_context_parallel_size = 2
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        RocmPlatform.check_and_update_config(config)
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        """)


def test_kimi_k3_temporal_state_uses_fp32():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.models.kimi_k3 import _get_k3_state_dtype

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(
                mamba_cache_dtype="auto",
                mamba_ssm_cache_dtype="auto",
            ),
        )
        conv_dtype, temporal_dtype = _get_k3_state_dtype(vllm_config)
        assert conv_dtype == torch.bfloat16
        assert temporal_dtype == torch.float32
        """)


def test_kimi_k3_outer_is_multimodal_and_hybrid():
    _run_without_test_stubs("""
        from vllm.model_executor.models.interfaces import IsHybrid

        from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
        from atom.plugin.vllm.models.kimi_k3 import KimiK3ForCausalLMVllm

        mro = KimiK3ForCausalLMVllm.__mro__
        # Multimodal via ATOMForConditionalGeneration; hybrid state retained.
        assert ATOMForConditionalGeneration in mro
        assert IsHybrid in mro

        # Image placeholder matches the checkpoint token.
        assert (
            KimiK3ForCausalLMVllm.get_placeholder_str("image", 0)
            == "<|kimi_image_placeholder|>"
        )

        # Hybrid mamba-state entry points survive the base-class change.
        for name in (
            "get_mamba_state_dtype_from_config",
            "get_mamba_state_shape_from_config",
            "get_mamba_state_copy_func",
        ):
            assert hasattr(KimiK3ForCausalLMVllm, name)
        """)


def test_kimi_k3_is_plugin_supported_multimodal():
    from atom.config import (
        _MULTIMODAL_MODEL_TYPES,
        _PLUGIN_SUPPORTED_MULTIMODAL_MODELS,
    )

    # kimi_k3 must be a known multimodal type AND opted into plugin-mode
    # pass-through, so get_hf_config keeps vision_config instead of stripping
    # to text_config.
    assert "kimi_k3" in _MULTIMODAL_MODEL_TYPES
    assert "kimi_k3" in _PLUGIN_SUPPORTED_MULTIMODAL_MODELS


def test_kimi_k3_inner_conditional_generation_class():
    _run_without_test_stubs("""
        from vllm.models.kimi_k3 import (
            KimiK3ForConditionalGeneration as vLLMKimiK3,
        )
        from atom.plugin.vllm.models.kimi_k3 import (
            KimiK3ForCausalLM,
            KimiK3ForConditionalGeneration_,
        )

        # Inner class subclasses the upstream multimodal model so it inherits
        # embed_multimodal / media parsing / vision-tower forward.
        assert issubclass(KimiK3ForConditionalGeneration_, vLLMKimiK3)

        # ATOM plugin loading + expert routing entry points exist.
        assert hasattr(KimiK3ForConditionalGeneration_, "load_weights")
        assert hasattr(KimiK3ForConditionalGeneration_, "get_expert_mapping")

        # Weight mapper collapses the double language_model nesting and renames
        # the projector layers.
        mapper = KimiK3ForConditionalGeneration_.hf_to_atom_mapper
        assert mapper.orig_to_new_prefix["language_model."] == (
            "language_model.language_model."
        )
        assert mapper.orig_to_new_prefix["mm_projector.proj.0"] == (
            "mm_projector.linear_1"
        )
        assert mapper.orig_to_new_prefix["mm_projector.proj.2"] == (
            "mm_projector.linear_2"
        )

        # Language-model wrapper exposes embed_input_ids for vLLM multimodal
        # discovery.
        assert hasattr(KimiK3ForCausalLM, "embed_input_ids")

        # Quant-remap attributes must be present (ATOMModelBase reads them off
        # the inner class); missing them silently corrupts a quantized ckpt.
        packed = KimiK3ForConditionalGeneration_.packed_modules_mapping
        assert packed, "packed_modules_mapping must be non-empty"

        excl = KimiK3ForConditionalGeneration_.quant_exclude_name_mapping
        assert excl, "quant_exclude_name_mapping must be non-empty"
        # Values carry the doubled language_model. prefix for the extra nesting
        # (inner.language_model = ATOM KimiK3ForCausalLM -> KimiLinearForCausalLM).
        for value in excl.values():
            assert value.startswith("language_model.language_model."), value
        """)


def test_kimi_k3_post_load_accepts_vllm_dtype():
    _run_without_test_stubs("""
        from inspect import Parameter, signature

        from atom.plugin.vllm.models.kimi_k3 import KimiKDAAttentionVllm

        parameters = signature(
            KimiKDAAttentionVllm.process_weights_after_loading
        ).parameters
        assert parameters["args"].kind is Parameter.VAR_POSITIONAL
        assert parameters["kwargs"].kind is Parameter.VAR_KEYWORD
        """)


def test_kimi_k3_vllm_metadata_adds_state_read_indices():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.models.kimi_k3 import (
            _adapt_kda_metadata_for_atom,
        )

        state_indices = torch.tensor([3, 7], dtype=torch.int32)
        metadata = SimpleNamespace(
            non_spec_state_indices_tensor=state_indices,
        )

        _adapt_kda_metadata_for_atom(metadata)
        assert metadata.non_spec_state_indices_in_tensor is state_indices
        """)


def test_kimi_k3_uses_dedicated_kda_metadata_backend():
    _run_without_test_stubs("""
        from vllm.models.kimi_k3.nvidia.kda_metadata import (
            KimiK3KDAMetadata,
            KimiK3KDAMetadataBuilder,
        )
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        from atom.plugin.vllm.gdn_backend import AtomGDNAttentionMetadataBuilder
        from atom.plugin.vllm.kda_backend import (
            AtomKimiK3KDAAttentionBackend,
            AtomKimiK3KDAMetadataBuilder,
        )
        from atom.plugin.vllm.models.kimi_k3 import KimiKDAAttentionVllm

        assert (
            KimiKDAAttentionVllm.get_attn_backend(None)
            is AtomKimiK3KDAAttentionBackend
        )
        assert issubclass(AtomKimiK3KDAMetadataBuilder, KimiK3KDAMetadataBuilder)
        assert issubclass(KimiK3KDAMetadata, GDNAttentionMetadata)
        assert not hasattr(
            AtomGDNAttentionMetadataBuilder,
            "_compact_full_graph_decode_metadata",
        )
        """)


def test_kda_metadata_adapter_compacts_full_graph_padding():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.kda_backend import AtomKimiK3KDAMetadataBuilder

        builder = SimpleNamespace(
            use_full_cuda_graph=True,
            decode_cudagraph_max_bs=4,
            non_spec_state_indices_tensor=torch.full((4,), -1, dtype=torch.int32),
            non_spec_query_start_loc=torch.zeros(5, dtype=torch.int32),
            kv_cache_spec=SimpleNamespace(),
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(mamba_cache_mode="all")
            ),
        )
        common = SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            num_reqs=4,
            block_table_tensor=torch.tensor([[5], [7], [0], [0]], dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_spec_decodes=0,
            num_decodes=4,
            num_decode_tokens=4,
            non_spec_state_indices_tensor=None,
            non_spec_query_start_loc=None,
        )

        AtomKimiK3KDAMetadataBuilder._adapt_full_graph_decode_metadata(
            builder, common, metadata
        )

        assert metadata.num_decodes == 2
        assert metadata.num_decode_tokens == 2
        assert metadata.non_spec_state_indices_tensor.tolist() == [5, 7, -1, -1]
        assert metadata.non_spec_query_start_loc.tolist() == [0, 1, 2, 2, 2]
        """)


def test_gdn_metadata_builder_does_not_compact_full_graph_padding():
    _run_without_test_stubs("""
        from atom.plugin.vllm.gdn_backend import AtomGDNAttentionMetadataBuilder

        # vLLM 0.27+ pads FULL-graph decode metadata by num_reqs; a prior
        # post-build compaction pass corrupted ssm_state on Qwen3.5 replay.
        assert not hasattr(
            AtomGDNAttentionMetadataBuilder,
            "_compact_full_graph_decode_metadata",
        )
        assert "build" not in AtomGDNAttentionMetadataBuilder.__dict__
        """)


def test_aiter_tp_group_must_match_vllm_dcp_order():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import pytest

        from atom.plugin.vllm.attention.layer_mla import (
            _validate_aiter_tp_matches_vllm_dcp,
        )

        aiter_group = SimpleNamespace(
            world_size=8,
            ranks=list(range(8)),
            rank_in_group=3,
        )
        vllm_group = SimpleNamespace(
            world_size=8,
            ranks=list(range(8)),
            rank_in_group=3,
        )
        _validate_aiter_tp_matches_vllm_dcp(aiter_group, vllm_group)

        vllm_group.ranks = [0, 2, 4, 6, 1, 3, 5, 7]
        with pytest.raises(RuntimeError, match="rank membership/order"):
            _validate_aiter_tp_matches_vllm_dcp(aiter_group, vllm_group)
        """)


def test_dense_mla_decode_pads_small_head_count():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.attention import layer_mla

        seen = {}

        def fake_mla_decode_fwd(q, _kv, output, *_args, **_kwargs):
            seen["num_heads"] = q.shape[1]
            output.fill_(1)
            return output, None

        layer_mla.mla_decode_fwd = fake_mla_decode_fwd
        attention = SimpleNamespace(
            head_repeat_factor=1,
            head_pad=4,
            kv_lora_rank=8,
            dcp_world_size=1,
            scale=1.0,
            _q_scale=None,
            _k_scale=None,
            _pad_decode_query_heads=lambda q: torch.nn.functional.pad(
                q, (0, 0, 0, 4)
            ),
            _restore_decode_query_heads=lambda output, num_heads: output[
                :, :num_heads
            ],
        )
        decode = SimpleNamespace(
            attn_out_dtype=torch.bfloat16,
            use_persistent_metadata=False,
            paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_indices=torch.tensor([0], dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
            fold_factor=None,
            max_qo_len=1,
        )
        output, lse = layer_mla.AttentionForVllmMLA._forward_decode(
            attention,
            torch.zeros(1, 12, 8, dtype=torch.bfloat16),
            torch.zeros(1, 8, dtype=torch.bfloat16),
            SimpleNamespace(decode=decode),
        )
        assert seen["num_heads"] == 16
        assert output.shape == (1, 12, 8)
        assert lse is None
        """)


def test_dense_mla_decode_pads_gathered_dcp_heads():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.model_ops.attention_mla import MLAAttention
        from atom.plugin.vllm.attention import layer_mla

        seen = {}

        def fake_mla_decode_fwd(q, _kv, output, *_args, **_kwargs):
            seen["num_heads"] = q.shape[1]
            output.fill_(1)
            lse = torch.ones(
                q.shape[0], q.shape[1], dtype=torch.float32, device=q.device
            )
            return output, lse

        layer_mla.mla_decode_fwd = fake_mla_decode_fwd
        attention = SimpleNamespace(
            num_heads=12,
            min_query_heads=16,
            kv_lora_rank=8,
            dcp_world_size=8,
            kv_cache_dtype="fp8",
            is_sparse_mla=False,
            dcp_persistent_supported=True,
            scale=1.0,
            _q_scale=None,
            _k_scale=None,
        )
        MLAAttention._configure_dcp_decode_head_padding(attention, 8)
        attention._pad_decode_query_heads = (
            lambda q: MLAAttention._pad_decode_query_heads(attention, q)
        )
        attention._restore_decode_query_heads = (
            lambda output, num_heads: MLAAttention._restore_decode_query_heads(
                attention, output, num_heads
            )
        )
        decode = SimpleNamespace(
            attn_out_dtype=torch.bfloat16,
            use_persistent_metadata=False,
            paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_indices=torch.tensor([0], dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
            fold_factor=None,
            max_qo_len=1,
        )
        output, lse = layer_mla.AttentionForVllmMLA._forward_decode(
            attention,
            torch.zeros(1, 96, 8, dtype=torch.bfloat16),
            torch.zeros(1, 8, dtype=torch.bfloat16),
            SimpleNamespace(decode=decode),
        )
        assert attention.dcp_kernel_num_heads == 128
        assert attention.dcp_head_pad == 32
        assert seen["num_heads"] == 128
        assert output.shape == (1, 96, 8)
        assert lse.shape == (1, 96)
        """)
