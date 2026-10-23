import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Bridge imports sglang at module load; keep this file runnable without SGLang.
for _name in (
    "sglang",
    "sglang.srt",
    "sglang.srt.model_executor",
    "sglang.srt.mem_cache",
    "sglang.srt.configs",
    "sglang.srt.speculative",
):
    if _name not in sys.modules:
        _mod = ModuleType(_name)
        _mod.__path__ = []
        sys.modules[_name] = _mod

if "sglang.srt.model_executor.forward_batch_info" not in sys.modules:
    _fb = ModuleType("sglang.srt.model_executor.forward_batch_info")
    _fb.ForwardBatch = object
    _fb.PPProxyTensors = object
    sys.modules[_fb.__name__] = _fb
if "sglang.srt.mem_cache.base_swa_memory_pool" not in sys.modules:
    _swa = ModuleType("sglang.srt.mem_cache.base_swa_memory_pool")
    _swa.BaseSWAKVPool = object
    sys.modules[_swa.__name__] = _swa
if "sglang.srt.configs.load_config" not in sys.modules:
    _lc = ModuleType("sglang.srt.configs.load_config")

    @dataclass
    class _LoadConfig:
        def __post_init__(self):
            return None

    _lc.LoadConfig = _LoadConfig
    sys.modules[_lc.__name__] = _lc

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
)
from atom.plugin.sglang.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    ATOMDeepSeekV4ProxyKVPool,
    _attach_v4_decode_split_plans,
    _geometry_serves_ratio,
    _geometry_supports_shared_prefill_writer,
    _proxy_pool_geometry,
    _resolve_v4_pool_geometry,
    _retarget_v4_decode_split_plans,
    _write_dense_only_prefill_indices,
)

SGLANG_BRIDGE = (
    Path(__file__).parents[2] / "atom/plugin/sglang/deepseek_v4_bridge.py"
).read_text()


def test_proxy_geometry_matches_per_layer_cache_views():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=64,
        c128_size=3,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=3,
        compression_ratios=[0, 4, 128],
        device="cpu",
    )
    geometry = _proxy_pool_geometry(pool)

    assert geometry.classes == (DENSE_RATIO, CSA_RATIO, HCA_RATIO)
    assert geometry.window_params(0).ring_start == 0
    for layer, ratio, compressed in (
        (1, 4, pool.views["csa_main"][0]),
        (2, 128, pool.views["hca_main"][0]),
    ):
        unified = pool.views["unified"][layer]
        window = pool.views["swa"][layer]
        ring_start = pool.num_blocks * (ATOM_DEEPSEEK_V4_BLOCK_SIZE // ratio)

        assert geometry.window_params(ratio).ring_start == ring_start
        assert compressed.data_ptr() == unified.data_ptr()
        assert window.data_ptr() == unified[ring_start].data_ptr()


def test_proxy_metadata_uses_per_layer_csa_block_stride():
    pool = SimpleNamespace(
        num_blocks=3,
        swa_cache_size=128,
        stage_ratios=[DENSE_RATIO, CSA_RATIO, HCA_RATIO],
        _atom_v4_geometry=None,
    )
    metadata = SimpleNamespace()

    geometry = _resolve_v4_pool_geometry(metadata, pool)

    assert metadata.pool_geometry is geometry
    assert metadata.envelope_rows == geometry.block_rows(CSA_RATIO)
    assert geometry.envelope_rows == geometry.block_rows(HCA_RATIO)
    assert metadata.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // CSA_RATIO
    assert geometry.envelope_rows == ATOM_DEEPSEEK_V4_BLOCK_SIZE // HCA_RATIO


def test_proxy_geometry_omits_absent_stage_ratios():
    pool = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=0,
        c128_size=3,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=2,
        compression_ratios=[DENSE_RATIO, HCA_RATIO],
        device="cpu",
    )
    geometry = _proxy_pool_geometry(pool)

    assert geometry.classes == (DENSE_RATIO, HCA_RATIO)
    assert _geometry_serves_ratio(geometry, DENSE_RATIO)
    assert _geometry_serves_ratio(geometry, HCA_RATIO)
    assert not _geometry_serves_ratio(geometry, CSA_RATIO)
    with pytest.raises(KeyError):
        geometry.window_params(CSA_RATIO)


def test_sglang_decode_graph_pads_csa_visibility_to_t_pad():
    assert "visible_np = np.zeros(t_pad, dtype=np.int32)" in SGLANG_BRIDGE
    assert "visible_np[:total] = visible_csa(pos_np).astype(np.int32)" in SGLANG_BRIDGE


def test_sglang_graph_buffers_keep_distinct_state_slot_addresses():
    assert "self.state_slot_in = i32(s)" in SGLANG_BRIDGE
    assert "self.state_slot_out = i32(s)" in SGLANG_BRIDGE
    assert (
        "md.state_slot_in = out.clone() if state_slot_in is None else state_slot_in"
        in SGLANG_BRIDGE
    )
    assert (
        "md.state_slot_out = bufs.stage(bufs.state_slot_out, slot_arr, n)"
        in SGLANG_BRIDGE
    )
    assert (
        "md.state_slot_in = bufs.stage(bufs.state_slot_in, slot_arr, n)"
        in SGLANG_BRIDGE
    )


def test_shared_prefill_writer_requires_csa_and_hca():
    full = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=64,
        c128_size=3,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=3,
        compression_ratios=[DENSE_RATIO, CSA_RATIO, HCA_RATIO],
        device="cpu",
    )
    draft = ATOMDeepSeekV4ProxyKVPool(
        max_num_reqs=2,
        num_req_slots=2,
        swa_size=256,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        dtype=torch.bfloat16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        indexer_head_dim=8,
        layer_num=1,
        compression_ratios=[DENSE_RATIO],
        device="cpu",
    )
    assert _geometry_supports_shared_prefill_writer(_proxy_pool_geometry(full))
    assert not _geometry_supports_shared_prefill_writer(_proxy_pool_geometry(draft))


def test_dense_only_prefill_indices_write_swa_without_calling_shared_writer():
    geometry = _proxy_pool_geometry(
        ATOMDeepSeekV4ProxyKVPool(
            max_num_reqs=2,
            num_req_slots=2,
            swa_size=8,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.bfloat16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            indexer_head_dim=8,
            layer_num=1,
            compression_ratios=[DENSE_RATIO],
            device="cpu",
        )
    )
    win = 4
    positions = torch.tensor([0, 1, 16, 17], dtype=torch.int32)
    bid = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    chunk_start = torch.tensor([0, 16], dtype=torch.int32)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    # Counts match the shared prefill formulas for these positions.
    # extend: [1, 2, 1, 2]; prefix_swa: [0, 0, 3, 2]
    extend_indptr = torch.tensor([0, 1, 3, 4, 6], dtype=torch.int32)
    swa_indptr = torch.tensor([0, 0, 0, 3, 5], dtype=torch.int32)
    sentinel = -9
    extend_indices = torch.full((6,), sentinel, dtype=torch.int32)
    prefix_swa = torch.full((5,), sentinel, dtype=torch.int32)

    _write_dense_only_prefill_indices(
        positions=positions,
        bid_per_token=bid,
        chunk_start_per_seq=chunk_start,
        cu_seqlens_q_per_seq=cu_q,
        state_slot_per_seq=slots,
        extend_indptr=extend_indptr,
        prefix_swa_indptr=swa_indptr,
        extend_indices=extend_indices,
        prefix_swa_indices=prefix_swa,
        T=4,
        win=win,
        geometry=geometry,
    )

    assert (extend_indices != sentinel).all()
    assert (prefix_swa != sentinel).all()
    dense = geometry.window_params(DENSE_RATIO)
    assert int(prefix_swa[0]) == dense.index(1, 13)
    assert "_write_dense_only_prefill_indices" in SGLANG_BRIDGE
    assert "_geometry_supports_shared_prefill_writer" in SGLANG_BRIDGE


def _split_plan_pool(*, fp8: bool, num_blocks: int = 7):
    return SimpleNamespace(
        use_fp8_kv=fp8,
        window_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        num_slots=4,
        num_blocks=num_blocks,
        device=torch.device("cpu"),
    )


def test_decode_split_plans_are_absent_without_fp8_kv():
    md = SimpleNamespace()
    _attach_v4_decode_split_plans(
        md,
        rows=3,
        proxy_pool=_split_plan_pool(fp8=False),
        model=SimpleNamespace(n_local_heads=16),
        index_topk=32,
        max_committed_hca=7,
    )
    assert md.split_plan_swa is None
    assert md.split_plan_csa is None
    assert md.split_plan_hca is None
    assert md._atom_v4_split_plan_spec is None


def test_decode_split_plans_follow_native_kv_lengths(monkeypatch):
    from atom.model_ops.v4_kernels import paged_decode

    calls = []

    def _fake_plan(rows, heads, kv_len, table):
        calls.append((rows, heads, kv_len, table))
        return 2, table[1]

    monkeypatch.setattr(paged_decode, "v4_decode_split_plan", _fake_plan)
    pool = _split_plan_pool(fp8=True, num_blocks=9)
    md = SimpleNamespace()
    _attach_v4_decode_split_plans(
        md,
        rows=3,
        proxy_pool=pool,
        model=SimpleNamespace(n_local_heads=16),
        index_topk=32,
        max_committed_hca=9,
    )

    win = ATOM_DEEPSEEK_V4_BLOCK_SIZE
    assert [call[2] for call in calls] == [win, win + 32, win + 9]
    assert all(call[0] == 3 and call[1] == 16 for call in calls)
    table = calls[0][3]
    assert all(call[3] is table for call in calls)
    assert md.split_plan_swa == (2, table[1])
    assert md.split_plan_csa[0] == 2
    assert md.split_plan_hca[0] == 2

    again = SimpleNamespace()
    _attach_v4_decode_split_plans(
        again,
        rows=2,
        proxy_pool=pool,
        model=SimpleNamespace(n_local_heads=16),
        index_topk=32,
        max_committed_hca=9,
    )
    assert again._atom_v4_split_plan_spec[0] is table

    _retarget_v4_decode_split_plans(md, 1)
    assert calls[-1][0] == 1
    assert calls[-3][0] == 1
    assert md.split_plan_swa[1].data_ptr() == table[1].data_ptr()
