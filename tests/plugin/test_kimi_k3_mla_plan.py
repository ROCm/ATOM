"""K3 decode must plan MLA at ATOM's per-token page, not SGLang's 128."""

import sys
import types

import torch

from atom.plugin.sglang.kimi_k3_bridge import retarget_kimi_k3_decode_mla_plan


def test_decode_plan_uses_one_token_pages(monkeypatch):
    captured = {}

    def fake_get(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    try:
        import aiter
    except ImportError:
        aiter = types.ModuleType("aiter")
        monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setattr(aiter, "get_mla_metadata_v1", fake_get, raising=False)

    bs = 3
    device = "cpu"
    # A page-128 remainder cumsum is not a q_len=1 indptr.
    qo = torch.tensor([0, 40, 90, 128, 0], dtype=torch.int32)
    last_page = torch.tensor([40, 50, 128, 7], dtype=torch.int32)
    metadata = types.SimpleNamespace(
        kv_indptr=torch.tensor([0, 10, 30, 40], dtype=torch.int32),
        kv_last_page_len=last_page,
        qo_indptr=qo,
        work_meta_data=torch.empty(4, device=device),
        work_info_set=torch.empty(4, device=device),
        work_indptr=torch.empty(4, device=device),
        reduce_indptr=torch.empty(4, device=device),
        reduce_final_map=torch.empty(4, device=device),
        reduce_partial_map=torch.empty(4, device=device),
    )

    assert retarget_kimi_k3_decode_mla_plan(
        metadata,
        batch_size=bs,
        num_heads=12,
        dtype_q=torch.float8_e4m3fn,
        dtype_kv=torch.float8_e4m3fn,
        fast_mode=False,
        intra_batch_mode=True,
        max_split_per_batch=64,
    )
    assert captured["kwargs"]["page_size"] == 1
    assert captured["kwargs"]["kv_granularity"] == 16
    assert captured["kwargs"]["fast_mode"] is False
    assert captured["kwargs"]["intra_batch_mode"] is True
    assert captured["kwargs"]["max_split_per_batch"] == 64
    assert captured["args"][3] == 16  # 12 heads padded to the kernel width
    assert torch.equal(captured["args"][0], torch.arange(bs + 1, dtype=torch.int32))
    assert torch.equal(captured["args"][2], torch.ones(bs, dtype=torch.int32))
    # The live buffers the graph already captured stay the ones that were filled.
    assert captured["args"][6] is metadata.work_meta_data
    assert torch.equal(last_page[:bs], torch.ones(bs, dtype=torch.int32))


def test_decode_plan_reads_sglang_work_metadata_field(monkeypatch):
    """SGLang stores the plan on work_metadata, not ATOM's work_meta_data."""

    captured = {}

    def fake_get(*args, **kwargs):
        captured["args"] = args

    try:
        import aiter
    except ImportError:
        aiter = types.ModuleType("aiter")
        monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setattr(aiter, "get_mla_metadata_v1", fake_get, raising=False)

    work = torch.empty(4)
    metadata = types.SimpleNamespace(
        kv_indptr=torch.tensor([0, 4], dtype=torch.int32),
        kv_last_page_len=torch.tensor([128], dtype=torch.int32),
        qo_indptr=torch.tensor([0, 128], dtype=torch.int32),
        work_metadata=work,
        work_info_set=torch.empty(4),
        work_indptr=torch.empty(4),
        reduce_indptr=torch.empty(4),
        reduce_final_map=torch.empty(4),
        reduce_partial_map=torch.empty(4),
    )

    assert retarget_kimi_k3_decode_mla_plan(
        metadata,
        batch_size=1,
        num_heads=16,
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
        fast_mode=True,
        intra_batch_mode=False,
        max_split_per_batch=None,
    )
    assert captured["args"][6] is work
    assert metadata.work_meta_data is work
    assert metadata.work_metadata is work
    assert int(metadata.kv_last_page_len[0]) == 1


def test_decode_plan_skips_when_sglang_did_not_allocate_persistent_buffers():
    metadata = types.SimpleNamespace(
        kv_indptr=torch.zeros(2, dtype=torch.int32),
        kv_last_page_len=torch.ones(1, dtype=torch.int32),
        work_meta_data=None,
    )
    assert not retarget_kimi_k3_decode_mla_plan(
        metadata,
        batch_size=1,
        num_heads=16,
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
        fast_mode=False,
        intra_batch_mode=True,
        max_split_per_batch=None,
    )
