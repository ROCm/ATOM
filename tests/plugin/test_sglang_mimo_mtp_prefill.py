"""MiMo MTP prefill and graph replay on unmodified SGLang dependencies."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("Requires ROCm and AITER", allow_module_level=True)

pytest.importorskip("sglang")
pytest.importorskip("aiter")

from aiter import dtypes
from sglang.kernels.ops.attention.utils import launch_reshape_and_cache_flash
from sglang.srt.layers.attention.aiter_backend import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)
from atom.plugin.sglang.attention_backend.full_attention.mimo_mtp import (
    init_mimo_mtp_prefill_metadata,
)


def make_case(query_lengths, lengths, window, hybrid_pool=False):
    torch.manual_seed(3124)
    bs, page, dim, heads = len(lengths), 64, 192, 16
    pages_per_req = (max(lengths) + page - 1) // page + 1
    capacity = pages_per_req * page
    blocks = bs * pages_per_req + 1
    tables = (torch.randperm(blocks - 1, device="cuda") + 1).view(bs, -1)
    slots = (tables[..., None] * page + torch.arange(page, device="cuda")).flatten(1)
    page_mapping = torch.cat(
        (
            torch.zeros(1, device="cuda", dtype=torch.int64),
            torch.randperm(blocks - 1, device="cuda") + 1,
        )
    )
    mapping = (
        page_mapping[:, None] * page + torch.arange(page, device="cuda")
    ).flatten()
    cache_slots = mapping[slots] if hybrid_pool and window > 0 else slots
    raw_k = (
        torch.randn(bs * capacity, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    )
    raw_v = torch.randn_like(raw_k) * 0.5
    raw_v[..., 128:] = 0
    key = torch.zeros(blocks * page, 1, dim, device="cuda", dtype=dtypes.fp8)
    value = torch.zeros_like(key)
    ks = torch.tensor([0.25], device="cuda")
    vs = torch.tensor([0.5], device="cuda")
    launch_reshape_and_cache_flash(
        raw_k,
        raw_v,
        key.view(-1, page, 1, dim),
        value.view(-1, page, 1, dim),
        cache_slots.flatten(),
        k_scale=ks,
        v_scale=vs,
    )
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    qlens = torch.tensor(query_lengths, device="cuda", dtype=torch.int32)
    cu_q = torch.nn.functional.pad(qlens.cumsum(0, dtype=torch.int32), (1, 0))
    selected = torch.cat(
        [
            torch.arange(
                i * capacity + length - count, i * capacity + length, device="cuda"
            )
            for i, (count, length) in enumerate(zip(query_lengths, lengths))
        ]
    )
    out_loc = torch.cat(
        [
            slots[i, length - count : length]
            for i, (count, length) in enumerate(zip(query_lengths, lengths))
        ]
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        seq_lens=seq,
        seq_lens_cpu=seq.cpu(),
        req_pool_indices=torch.arange(bs, device="cuda"),
        out_cache_loc=out_loc,
    )
    layer = SimpleNamespace(
        layer_id=0,
        head_dim=dim,
        qk_head_dim=dim,
        v_head_dim=dim,
        tp_q_head_num=heads,
        tp_k_head_num=1,
        tp_v_head_num=1,
        scaling=dim**-0.5,
        sliding_window_size=window,
        logit_cap=0.0,
        is_cross_attention=False,
        k_scale=ks,
        v_scale=vs,
    )
    backend = object.__new__(ATOMAttnBackendForSgl)
    backend._is_mimo_mtp = True
    backend._is_mimo_v2_family = True
    backend._mimo_legacy_unified_prefill = True
    backend.device = "cuda"
    backend.use_sliding_window_kv_pool = hybrid_pool
    backend.use_triton_unified_attention = True
    backend.use_mla = False
    backend.kv_cache_is_vectorized_5d = False
    backend.kv_cache_dtype = dtypes.fp8
    backend.k_scale, backend.v_scale = ks, vs
    backend.page_size = page
    backend.input_dtype = torch.bfloat16
    backend.req_to_token = slots
    backend.qo_indptr = cu_q
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda _: (key, value), full_to_swa_index_mapping=mapping
    )
    backend.cuda_graph_page_table = torch.zeros(
        bs, pages_per_req, device="cuda", dtype=torch.int32
    )
    backend.cuda_graph_swa_page_table = torch.zeros_like(backend.cuda_graph_page_table)
    backend.forward_metadata = ForwardMetadata(
        None, None, None, None, max(query_lengths), max(lengths)
    )
    q = (
        torch.randn(sum(query_lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
        * 0.5
    )
    sinks = torch.linspace(3, 5, heads, device="cuda") if window > 0 else None

    def forward(save_kv_cache=True):
        return backend.forward_extend(
            q,
            raw_k[selected],
            raw_v[selected],
            layer,
            batch,
            save_kv_cache=save_kv_cache,
            sinks=sinks,
        )

    def check_reference(output):
        offset = 0
        for req, (count, length) in enumerate(
            zip(query_lengths, batch.seq_lens.tolist())
        ):
            indices = sorted({0, count // 2, count - 1})
            qi = q[offset : offset + count][indices].float()
            k = key.float()[cache_slots[req, :length]].squeeze(1) * ks
            v = value.float()[cache_slots[req, :length]].squeeze(1) * vs
            logits = qi @ k.T * layer.scaling
            positions = torch.tensor(indices, device="cuda") + length - count
            key_pos = torch.arange(length, device="cuda")
            mask = key_pos[None, :] <= positions[:, None]
            if window > 0:
                mask &= key_pos[None, :] >= positions[:, None] - window + 1
            logits = logits.masked_fill(~mask[:, None], -torch.inf)
            if sinks is not None:
                logits = torch.cat(
                    [logits, sinks[None, :, None].expand(len(indices), -1, -1)], dim=-1
                )
            expected = logits.softmax(dim=-1)[..., :length] @ v
            actual = output.view(-1, heads, dim)[offset : offset + count][
                indices
            ].float()
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
            offset += count
        assert torch.count_nonzero(output.view(-1, heads, dim)[..., 128:]) == 0

    return SimpleNamespace(
        backend=backend,
        batch=batch,
        forward=forward,
        check=check_reference,
        q=q,
        lengths=lengths,
        key=key,
        value=value,
        raw_k=raw_k,
        raw_v=raw_v,
    )


@pytest.mark.parametrize("hybrid_pool", [False, True])
@pytest.mark.parametrize("window", [-1, 128])
@pytest.mark.parametrize(
    "query_lengths,lengths",
    [([1, 1], [1, 2]), ([97, 129], [130, 195]), ([4096, 4096], [4096, 5003])],
)
def test_mimo_mtp_prefill_and_prefix_cache(window, query_lengths, lengths, hybrid_pool):
    c = make_case(query_lengths, lengths, window, hybrid_pool)
    init_mimo_mtp_prefill_metadata(c.backend, c.batch)
    c.check(c.forward())


@pytest.mark.parametrize("hybrid_pool", [False, True])
@pytest.mark.parametrize("window", [-1, 128])
def test_mimo_mtp_draft_extend_graph_replays(window, hybrid_pool):
    c = make_case([2, 2], [130, 195], window, hybrid_pool)
    c.batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    c.backend._resolve_v2_num_draft_tokens = lambda: 2
    init_mimo_mtp_prefill_metadata(c.backend, c.batch, for_cuda_graph=True)
    pointer = c.backend._mimo_prefill_page_table.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        c.forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = c.forward()
    c.q.mul_(0.8)
    c.batch.seq_lens.sub_(1)
    init_mimo_mtp_prefill_metadata(c.backend, c.batch, for_cuda_graph=True)
    assert c.backend._mimo_prefill_page_table.data_ptr() == pointer
    graph.replay()
    c.check(output)


@pytest.mark.parametrize("hybrid_pool", [False, True])
def test_mimo_mtp_prefill_preserves_cache_when_writes_are_disabled(hybrid_pool):
    c = make_case([2, 2], [130, 195], 128, hybrid_pool)
    init_mimo_mtp_prefill_metadata(c.backend, c.batch)
    key_before, value_before = c.key.clone(), c.value.clone()
    c.raw_k.zero_()
    c.raw_v.zero_()
    c.check(c.forward(save_kv_cache=False))
    torch.testing.assert_close(c.key.float(), key_before.float(), atol=0, rtol=0)
    torch.testing.assert_close(c.value.float(), value_before.float(), atol=0, rtol=0)
