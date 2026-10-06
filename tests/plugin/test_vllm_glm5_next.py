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
            "model.layers.46.glm5_kpool_index", None, indexers, store
        )
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
