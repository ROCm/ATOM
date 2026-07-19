import torch
from types import SimpleNamespace

from atom.plugin.sglang.deepseek_v4_bridge import (
    ATOMDeepSeekV4ProxyKVPool,
    bind_deepseek_v4_proxy_cache_views,
)


def test_deepseek_v4_proxy_fp8_views_use_packed_main_kv_rows():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        swa_size=256,
        c4_size=4,
        c128_size=4,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=128,
        swa_page_size=128,
        dtype="fp8",
        state_dtype="fp8",
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=3,
        device="cpu",
        compression_ratios=[0, 4, 128],
        num_req_slots=2,
    )

    for unified, swa in zip(pool.views["unified"], pool.views["swa"]):
        assert unified.shape[-1] == 512
        assert swa.shape[-1] == 512

    for unified_rope, swa_rope in zip(
        pool.views["unified_rope"], pool.views["swa_rope"]
    ):
        assert unified_rope is not None
        assert unified_rope.shape[-1] == 64
        assert swa_rope is not None
        assert swa_rope.shape[-1] == 64

    assert pool.views["csa_main"][0].shape[-1] == 512
    assert pool.views["hca_main"][0].shape[-1] == 512


def test_deepseek_v4_proxy_bf16_views_keep_inline_kv_rows():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        swa_size=256,
        c4_size=4,
        c128_size=4,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=128,
        swa_page_size=128,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=3,
        device="cpu",
        compression_ratios=[0, 4, 128],
        num_req_slots=2,
    )

    for unified, swa in zip(pool.views["unified"], pool.views["swa"]):
        assert unified.shape[-1] == 512
        assert swa.shape[-1] == 512

    assert pool.views["unified_rope"] == [None, None, None]
    assert pool.views["swa_rope"] == [None, None, None]


def test_deepseek_v4_proxy_binding_exposes_packed_fp8_decode_buffers():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        swa_size=256,
        c4_size=4,
        c128_size=4,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype="fp8_e4m3",
        state_dtype="fp8_e4m3",
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=3,
        device="cpu",
        compression_ratios=[0, 4, 128],
        num_req_slots=2,
    )

    def compressor():
        return SimpleNamespace(
            kv_state=torch.empty((1, 1), dtype=torch.float32),
            score_state=torch.empty((1, 1), dtype=torch.float32),
        )

    dense_attn = SimpleNamespace(compress_ratio=0)
    csa_attn = SimpleNamespace(
        compress_ratio=4,
        compressor=compressor(),
        indexer=SimpleNamespace(
            index_topk=1024,
            compressor=compressor(),
        ),
    )
    hca_attn = SimpleNamespace(compress_ratio=128, compressor=compressor())
    model = SimpleNamespace(
        args=SimpleNamespace(index_topk=1024),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(attn=dense_attn),
                SimpleNamespace(attn=csa_attn),
                SimpleNamespace(attn=hca_attn),
            ]
        ),
    )

    assert bind_deepseek_v4_proxy_cache_views(model, pool)

    for attn in (dense_attn, csa_attn, hca_attn):
        assert attn.unified_kv.shape[-1] == 512
        assert attn.unified_kv_rope is not None
        assert attn.unified_kv_rope.shape[-1] == 64
        assert attn.swa_kv.shape[-1] == 512
        assert attn.swa_kv_rope is not None
        assert attn.swa_kv_rope.shape[-1] == 64
        assert attn.swa_block_size == pool.swa_cache_size

    assert csa_attn.compressor.kv_cache.shape[-1] == 512
    assert csa_attn.compressor.kv_cache_rope.shape[-1] == 64
    assert csa_attn.compressor.write_mode == "main_2buff_fp8"
    assert hca_attn.compressor.kv_cache.shape[-1] == 512
    assert hca_attn.compressor.kv_cache_rope.shape[-1] == 64
    assert hca_attn.compressor.write_mode == "main_2buff_fp8"
