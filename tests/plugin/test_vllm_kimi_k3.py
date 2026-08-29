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
        _ATOM_MODEL_CLASSES[arch] == "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLM"
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
            causal=True,
            g_kv_indptr=None,
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
            causal=True,
            g_kv_indptr=None,
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


def test_dcp_local_slots_match_the_unsharded_layout_at_cp1():
    import torch

    from atom.plugin.vllm.dspark_dcp_patch import _dcp_local_slots

    block_size = 4
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    positions = torch.tensor([0, 3, 4, 7, 8], dtype=torch.int64)
    token_req = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int64)

    slots = _dcp_local_slots(
        positions.clone(),
        block_table,
        token_req,
        block_size,
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
        pad_slot_id=-1,
    )
    # block_id * block_size + position % block_size, with no rank filtering.
    assert slots.tolist() == [40, 43, 84, 87, 48]


def test_dcp_local_slots_keep_only_this_ranks_round_robin_share():
    import torch

    from atom.plugin.vllm.dspark_dcp_patch import _dcp_local_slots

    block_size, cp_size = 4, 2
    # One block-table entry now spans block_size * cp_size = 8 global tokens.
    block_table = torch.tensor([[10, 11]], dtype=torch.int32)
    positions = torch.arange(16, dtype=torch.int64)
    token_req = torch.zeros(16, dtype=torch.int64)

    per_rank = [
        _dcp_local_slots(
            positions.clone(),
            block_table,
            token_req,
            block_size,
            cp_size=cp_size,
            cp_rank=rank,
            cp_interleave=1,
            pad_slot_id=-1,
        ).tolist()
        for rank in range(cp_size)
    ]

    # Rank r owns exactly the positions with p % cp_size == r, packed densely
    # into its own physical block, and drops the rest.
    assert per_rank[0] == [40, -1, 41, -1, 42, -1, 43, -1] + [
        44,
        -1,
        45,
        -1,
        46,
        -1,
        47,
        -1,
    ]
    assert per_rank[1] == [-1, 40, -1, 41, -1, 42, -1, 43] + [
        -1,
        44,
        -1,
        45,
        -1,
        46,
        -1,
        47,
    ]
    # Every global position is stored by exactly one rank.
    for p in range(16):
        assert sum(per_rank[r][p] != -1 for r in range(cp_size)) == 1


def test_atom_patch_hides_dcp_from_speculative_config_validation():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        from vllm.engine.arg_utils import EngineArgs

        from atom.plugin.vllm.dspark_dcp_patch import (
            apply_vllm_dspark_dcp_config_patch,
        )

        seen = {}

        def stub(self, target_model_config, target_parallel_config):
            # Stands in for the real constructor, whose only DCP-dependent
            # behaviour is the raise this patch exists to skip.
            seen["dcp"] = target_parallel_config.decode_context_parallel_size
            return "spec-config"

        EngineArgs.create_speculative_config = stub
        apply_vllm_dspark_dcp_config_patch()

        args = SimpleNamespace(speculative_config={"method": "dspark"})
        parallel_config = SimpleNamespace(decode_context_parallel_size=8)
        result = EngineArgs.create_speculative_config(args, None, parallel_config)

        assert result == "spec-config"
        # Masked for the duration of the call, restored the moment it returns.
        assert seen["dcp"] == 1
        assert parallel_config.decode_context_parallel_size == 8

        # Without a speculative config, or without DCP, nothing is touched.
        seen.clear()
        no_dcp = SimpleNamespace(decode_context_parallel_size=1)
        EngineArgs.create_speculative_config(args, None, no_dcp)
        assert seen["dcp"] == 1
        """)


def test_dcp_multi_token_decode_selects_the_round_robin_kernel():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.model_ops.attention_mla import MLAAttention
        from atom.plugin.vllm.attention import layer_mla

        seen = {}

        def fake_mla_decode_fwd(q, _kv, output, *_args, **kwargs):
            seen.update(kwargs)
            output.fill_(1)
            lse = torch.ones(
                q.shape[0], q.shape[1], dtype=torch.float32, device=q.device
            )
            return output, lse

        layer_mla.mla_decode_fwd = fake_mla_decode_fwd

        def run(max_qo_len, causal, g_kv_indptr):
            attention = SimpleNamespace(
                num_heads=12,
                min_query_heads=16,
                kv_lora_rank=8,
                dcp_world_size=8,
                dcp_rank=3,
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
                lambda output, num_heads: (
                    MLAAttention._restore_decode_query_heads(
                        attention, output, num_heads
                    )
                )
            )
            decode = SimpleNamespace(
                attn_out_dtype=torch.bfloat16,
                use_persistent_metadata=False,
                paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
                paged_kv_indices=torch.tensor([0], dtype=torch.int32),
                qo_indptr=torch.tensor([0, max_qo_len], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
                fold_factor=None,
                max_qo_len=max_qo_len,
                causal=causal,
                g_kv_indptr=g_kv_indptr,
            )
            seen.clear()
            layer_mla.AttentionForVllmMLA._forward_decode(
                attention,
                torch.zeros(max_qo_len, 96, 8, dtype=torch.bfloat16),
                torch.zeros(1, 8, dtype=torch.bfloat16),
                SimpleNamespace(decode=decode),
            )
            return dict(seen)

        indptr = torch.tensor([0, 40], dtype=torch.int32)

        # Causal verify block: the mask has to be placed on global positions,
        # so the kernel gets the cprr parameters.
        cprr = run(max_qo_len=8, causal=True, g_kv_indptr=indptr)
        assert cprr["cp_world_size"] == 8
        assert cprr["cp_rank"] == 3
        assert cprr["g_kv_indptr"] is indptr
        assert cprr["causal"] is True

        # Bidirectional draft block: nothing to mask, plain kernel.
        plain = run(max_qo_len=8, causal=False, g_kv_indptr=None)
        assert plain["cp_world_size"] == 1
        assert plain["cp_rank"] == 0
        assert plain["g_kv_indptr"] is None
        assert plain["causal"] is False

        # Single-token decode sees all of its local KV; no mask either.
        single = run(max_qo_len=1, causal=True, g_kv_indptr=None)
        assert single["cp_world_size"] == 1
        assert single["g_kv_indptr"] is None
        """)
