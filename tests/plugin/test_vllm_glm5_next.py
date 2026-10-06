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


def test_glm5_next_plugin_registries_are_synchronized():
    from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
    from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES

    arch = "Glm5NextForConditionalGeneration"
    assert (
        _VLLM_MODEL_REGISTRY_OVERRIDES[arch]
        == "atom.plugin.vllm.models.glm5_next:Glm5NextForConditionalGenerationVllm"
    )
    assert (
        _ATOM_MODEL_CLASSES[arch]
        == "atom.plugin.vllm.models.glm5_next:Glm5NextForConditionalGeneration"
    )
    assert (
        _ATOM_MODEL_CLASSES["Glm5NextMTPModel"]
        == "atom.plugin.vllm.models.glm5_next:Glm5NextMTP"
    )
    assert "Glm5NextMTPModel" in _VLLM_MODEL_REGISTRY_OVERRIDES


def test_vllm_sizes_glm5_next_as_mla_with_the_padded_latent():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import vllm.config
        from vllm.transformers_utils import model_arch_config_convertor as conv

        from atom.plugin.vllm.register import _register_glm5_next_arch_config

        _register_glm5_next_arch_config()
        text = SimpleNamespace(model_type="glm5_next_text", kv_lora_rank=512,
                               qk_rope_head_dim=0, head_dim=0)
        outer = SimpleNamespace(model_type="glm5_next", text_config=text)
        c = conv.MODEL_ARCH_CONFIG_CONVERTORS["glm5_next"](outer, text)
        assert c.is_deepseek_mla()
        assert c.get_head_size() == 576
        text.num_nextn_predict_layers = 1
        draft = conv.MODEL_ARCH_CONFIG_CONVERTORS["glm5_next_mtp"](outer, text)
        assert draft.get_num_hidden_layers() == 1 and draft.get_head_size() == 576
        """)


def test_mtp_method_resolves_a_glm5_next_draft():
    _run_without_test_stubs("""
        import typing

        import vllm.config
        from transformers import PretrainedConfig
        from vllm.config import speculative
        from vllm.config.speculative import SpeculativeConfig

        from atom.plugin.vllm.register import _register_glm5_next_mtp_draft_config

        for _ in range(2):
            _register_glm5_next_mtp_draft_config()
        assert "glm5_next_mtp" in typing.get_args(speculative.MTPModelTypes)

        cfg = PretrainedConfig(
            model_type="glm5_next",
            architectures=["Glm5NextForConditionalGeneration"],
            text_config={"num_nextn_predict_layers": 1},
        )
        cfg.model_type = "glm5_next"
        cfg.text_config = PretrainedConfig(num_nextn_predict_layers=1)
        out = SpeculativeConfig.hf_config_override(cfg)
        assert out.model_type == "glm5_next_mtp"
        assert out.architectures == ["Glm5NextMTPModel"] and out.n_predict == 1

        other = PretrainedConfig(architectures=["DeepseekV3ForCausalLM"])
        other.model_type = "deepseek_v3"
        other.num_nextn_predict_layers = 1
        assert SpeculativeConfig.hf_config_override(other).architectures == [
            "DeepSeekMTPModel"
        ]
        """)


def test_glm5_next_kda_state_follows_vllm_kda_with_speculative_rows():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch
        from transformers import PretrainedConfig

        from atom.plugin.vllm.models.glm5_next import (
            Glm5NextForConditionalGenerationVllm as M,
        )

        layer_types = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
        text = PretrainedConfig(
            model_type="glm5_next_text", num_hidden_layers=4,
            num_attention_heads=64, layer_types=layer_types,
            linear_attn_config={"num_heads": 64, "head_dim": 128,
                                "short_conv_kernel_size": 4},
            index_head_dim=128, index_kpool=4, index_topk=2048,
            qk_rope_head_dim=0, qk_nope_head_dim=256, kv_lora_rank=512,
        )

        def config(num_spec):
            spec = SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
            return SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=text, dtype=torch.bfloat16),
                cache_config=SimpleNamespace(mamba_cache_dtype="auto",
                                             mamba_ssm_cache_dtype="auto"),
                parallel_config=SimpleNamespace(tensor_parallel_size=8),
                speculative_config=spec,
            )

        plain = M.get_mamba_state_shape_from_config(config(0))
        mtp = M.get_mamba_state_shape_from_config(config(3))
        assert len(plain) == len(M.get_mamba_state_dtype_from_config(config(0))) == 2
        assert len(M.get_mamba_state_copy_func()) == 2
        conv_rows = lambda shape: min(shape)
        assert conv_rows(mtp[0]) == conv_rows(plain[0]) + 3
        assert mtp[1] == plain[1]
        """)


def test_kpool_index_proxy_carves_one_sub_block_per_indexer_layer():
    _run_without_test_stubs("""
        from types import SimpleNamespace

        import torch

        from atom.plugin.vllm.glm5_kpool import Glm5KpoolIndexProxy

        class Indexer:
            index_kpool = 4
            config = SimpleNamespace(index_head_dim=128)

            def bind_kpool_index_cache(self, cache, slot, subs):
                self.cache, self.slot, self.subs = cache, slot, subs

        store = SimpleNamespace(tensor=torch.empty(12), ring=4)
        indexers = [Indexer() for _ in range(11)]
        proxy = Glm5KpoolIndexProxy(
            "model.layers.46.glm5_kpool_index", None, indexers[:10], store
        )
        proxy.add_indexers(indexers[10:])
        pages = torch.arange(3 * 64 * 576, dtype=torch.int64).to(torch.uint8)
        pages = pages.view(3, 64, 576)
        proxy.bind_kv_cache(pages)

        rows, row_bytes = 16, 144
        sub_bytes = rows * row_bytes
        for j, idx in enumerate(indexers):
            assert idx.slot == j and idx.subs == 16
            assert idx.cache.shape == (3 * 16, rows, row_bytes)
            assert idx.cache.is_contiguous()
            for b in range(3):
                got = idx.cache[b * idx.subs + j].flatten()
                want = pages[b].flatten()[j * sub_bytes : (j + 1) * sub_bytes]
                assert torch.equal(got, want), (j, b)

        too_many = Glm5KpoolIndexProxy(
            "x.1.y", None, [Indexer() for _ in range(17)], store
        )
        try:
            too_many.bind_kv_cache(pages)
        except RuntimeError:
            pass
        else:
            raise AssertionError("17 indexer layers cannot fit 16 sub-blocks")
        """)


def test_sparse_selection_counts_match_the_pooled_expansion():
    _run_without_test_stubs("""
        import torch

        from atom.plugin.vllm.attention.metadata import sparse_selection_counts

        lens = torch.tensor([1, 3, 4, 2047, 2048, 2049, 2051, 2052, 4000, 4003])
        pooled = sparse_selection_counts(lens, 2048, 4)
        assert pooled.tolist() == [1, 3, 4, 2047, 2048, 2049, 2051, 2048, 2048, 2051]
        dense = sparse_selection_counts(lens, 2048, 1)
        assert dense.tolist() == [min(int(x), 2048) for x in lens]
        """)


def test_the_sparse_seqlen_kernel_matches_its_host_twin():
    _run_without_test_stubs("""
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(0)
        from atom.plugin.vllm.attention.layer_sparse_mla import (
            generate_sparse_seqlen_triton,
        )
        from atom.plugin.vllm.attention.metadata import sparse_selection_counts

        lens = torch.tensor(
            [1, 3, 4, 2047, 2048, 2049, 2051, 2052, 4000, 4003], dtype=torch.int32
        )
        ones = torch.ones_like(lens)
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(ones, 0)
        for pool in (1, 4):
            gpu = generate_sparse_seqlen_triton(
                ones.cuda(), lens.cuda(), cu.cuda(), 2048, len(lens), 1, index_kpool=pool
            )
            host = sparse_selection_counts(lens.long(), 2048, pool)
            assert gpu.cpu().tolist() == host.tolist(), (pool, gpu, host)
        """)
