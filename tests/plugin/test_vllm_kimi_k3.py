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
        assert hasattr(
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


def test_gdn_metadata_adapter_compacts_full_graph_padding():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.gdn_backend import AtomGDNAttentionMetadataBuilder

        builder = SimpleNamespace(
            use_full_cuda_graph=True,
            decode_cudagraph_max_bs=4,
            non_spec_state_indices_tensor=torch.zeros(4, dtype=torch.int32),
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

        AtomGDNAttentionMetadataBuilder._compact_full_graph_decode_metadata(
            builder, common, metadata
        )

        assert metadata.num_decodes == 2
        assert metadata.num_decode_tokens == 2
        assert metadata.non_spec_state_indices_tensor.tolist() == [5, 7, -1, -1]
        assert metadata.non_spec_query_start_loc.tolist() == [0, 1, 2, 2, 2]
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
            _pad_query_heads=lambda q: torch.nn.functional.pad(
                q, (0, 0, 0, 4)
            ),
            _restore_query_heads=lambda output, num_heads: output[:, :num_heads],
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
