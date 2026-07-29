"""CPU characterization tests for DeepSeek-V4 KV tensor binding."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parent.parent / "atom/model_ops/attentions/dsv4/kv_bind.py"
)
_SPEC = importlib.util.spec_from_file_location("_dsv4_kv_bind_under_test", _PATH)
kv_bind = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(kv_bind)


class Attention:
    def __init__(self, layer_id: int, *, draft: bool = False):
        self.layer_id = layer_id
        self.dspark_draft = draft


class Indexer:
    def __init__(self, layer_id: int):
        self.prefix = f"layers.{layer_id}.attn.indexer"


class Compressor:
    def __init__(self, layer_id: int, ratio: int, *, indexer: bool = False):
        suffix = "indexer.compressor" if indexer else "compressor"
        self.prefix = f"layers.{layer_id}.attn.{suffix}"
        self.compress_ratio = ratio


class Unknown:
    pass


def _install_model_types(monkeypatch):
    module = types.ModuleType("atom.models.deepseek_v4")
    module.DeepseekV4Attention = Attention
    module.Indexer = Indexer
    module.Compressor = Compressor
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v4", module)


def _fixture(*, arena: bool = False, fp8: bool = False):
    # index_dim includes room for each row's inline fp32 scale.
    nb, k1, head_dim, index_head_dim, aligned_index_dim, rope_dim = (
        3,
        2,
        4,
        4,
        8,
        2,
    )
    swa_blocks, block_size = 2, 4
    swa_rows = swa_blocks * block_size

    runner = types.SimpleNamespace(
        num_swa_blocks=swa_blocks,
        num_physical_kvcache_blocks=nb,
        device="cpu",
        config=types.SimpleNamespace(
            v4_arena_group_specs=[
                {
                    "name": "c4",
                    "num_chunks": 3,
                    "bytes_per_chunk": 32,
                    "chunk_rows": 8,
                    "owners": {"swa": 32, "compress": 8},
                },
                {
                    "name": "c128",
                    "num_chunks": 3,
                    "bytes_per_chunk": 16,
                    "chunk_rows": 4,
                    "owners": {"swa": 16, "compress": 4},
                },
                {
                    "name": "dense",
                    "num_chunks": 3,
                    "bytes_per_chunk": 16,
                    "chunk_rows": 4,
                    "owners": {"swa": 16},
                },
            ]
        ),
    )
    rows = [24, 12, 8] if arena else [swa_rows + nb * k1, swa_rows + nb, swa_rows]
    runner.v4_unified_kv = [torch.zeros((n, head_dim)) for n in rows]
    runner.v4_unified_kv_rope = [torch.zeros((n, rope_dim)) for n in rows]
    # Keep a layer-major base so the second slice has a non-zero storage offset.
    runner.v4_csa_idx_kv = torch.zeros(
        (2, nb, k1, aligned_index_dim), dtype=torch.uint8
    )
    state = torch.zeros((2, 1, 1))
    for name in (
        "v4_csa_idx_kv_state",
        "v4_csa_idx_score_state",
        "v4_csa_main_kv_state",
        "v4_csa_main_score_state",
        "v4_hca_main_kv_state",
        "v4_hca_main_score_state",
        "v4_csa_idx_boundary_kv",
        "v4_csa_idx_boundary_score",
        "v4_csa_main_boundary_kv",
        "v4_csa_main_boundary_score",
    ):
        setattr(runner, name, state)

    builder = types.SimpleNamespace(
        model_runner=runner,
        block_size=block_size,
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_dim,
        k1_csa=k1,
        k2_hca=1,
        compress_ratios=[4, 128, 0],
        layer_id_to_csa_pos={0: 1},
        layer_id_to_hca_pos={1: 0},
        enable_csa_prefix_state_cache=True,
        _kv_fp8=fp8,
        _arena_on=arena,
    )
    return builder


def test_attention_bind_fixed_and_dspark(monkeypatch):
    _install_model_types(monkeypatch)
    builder = _fixture()
    attn = Attention(0)
    assert kv_bind.bind_kv_cache_tensor(builder, 0, attn)
    assert attn.swa_kv.shape == (8, 4)
    assert attn.swa_kv.stride() == (4, 1)
    assert attn.swa_kv.storage_offset() == 0
    assert attn.swa_row_stride == 4
    assert attn.swa_kv.dtype == torch.float32

    draft = Attention(2, draft=True)
    assert kv_bind.bind_kv_cache_tensor(builder, 2, draft)
    assert draft.swa_kv.shape == (8, 4)
    assert draft.swa_kv.dtype == torch.bfloat16
    assert draft.unified_kv is None


def test_arena_bind_persists_group_rows(monkeypatch):
    _install_model_types(monkeypatch)
    builder = _fixture(arena=True)
    attn = Attention(0)
    kv_bind.bind_kv_cache_tensor(builder, 0, attn)
    assert builder._arena_group_rows == {"c4": 8, "c128": 4, "dense": 4}
    assert attn.swa_kv is attn.unified_kv
    assert attn.swa_row_stride == 8


def test_indexer_and_inner_compressor_views(monkeypatch):
    _install_model_types(monkeypatch)
    builder = _fixture()
    indexer = Indexer(0)
    inner = Compressor(0, 4, indexer=True)
    kv_bind.bind_kv_cache_tensor(builder, 0, indexer)
    kv_bind.bind_kv_cache_tensor(builder, 0, inner)

    assert indexer.kv_cache.shape == (3, 2, 8)
    assert inner.kv_cache.data_ptr() == indexer.kv_cache.data_ptr()
    assert inner.cache_scale.shape == (3, 2)
    assert inner.cache_scale.stride() == (4, 1)
    idx_f32 = indexer.kv_cache.view(torch.float32)
    assert inner.cache_scale.storage_offset() == idx_f32.storage_offset() + 2
    assert inner.write_mode == "indexer_fp8"


def test_c4_c128_and_fp8_rope_bind_separately(monkeypatch):
    _install_model_types(monkeypatch)
    builder = _fixture(fp8=True)
    c4 = Compressor(0, 4)
    c128 = Compressor(1, 128)
    kv_bind.bind_kv_cache_tensor(builder, 0, c4)
    kv_bind.bind_kv_cache_tensor(builder, 1, c128)
    assert c4.kv_cache.shape == (3, 2, 4)
    assert c4.kv_cache.storage_offset() == 8 * 4
    assert c4.kv_cache_rope.shape == (3, 2, 2)
    assert c4._csa_owner == "main"
    assert c128.kv_cache.shape == (3, 1, 4)
    assert c128.kv_cache_rope.shape == (3, 1, 2)
    assert not hasattr(c128, "_csa_owner")


def test_unknown_module_is_not_handled(monkeypatch):
    _install_model_types(monkeypatch)
    assert not kv_bind.bind_kv_cache_tensor(_fixture(), 0, Unknown())
