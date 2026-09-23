"""MiMo must address its independently recycled SWA pool, including graph replay."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("Requires ROCm and AITER", allow_module_level=True)

pytest.importorskip("sglang")
from aiter import dtypes
from sglang.srt.layers.attention.aiter_backend import ForwardMetadata as AiterMetadata
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)
from atom.plugin.sglang.attention_backend.full_attention.kv_cache import (
    set_kv_buffer_with_layout_shuffle,
)
from atom.plugin.sglang.attention_backend.full_attention.metadata import ForwardMetadata


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", "1")
    torch.manual_seed(12345)
    device, page, dim, heads = "cuda", 64, 192, 16
    full_blocks, swa_blocks = 16, 8
    full_k = torch.zeros((full_blocks * page, 1, dim), device=device, dtype=dtypes.fp8)
    full_v = torch.zeros_like(full_k)
    # A backing guard makes a regression observable without corrupting other allocations.
    swa_k_guard = torch.zeros_like(full_k)
    swa_v_guard = torch.zeros_like(full_v)
    swa_k, swa_v = swa_k_guard[: swa_blocks * page], swa_v_guard[: swa_blocks * page]
    mapping = torch.zeros(full_blocks * page + 1, device=device, dtype=torch.int64)
    mapping[-1] = -1
    pool = SWAKVPool.__new__(SWAKVPool)
    pool.layer_transfer_counter = None
    pool.layers_mapping = {0: (0, False), 1: (0, True)}
    pool.full_to_swa_index_mapping = mapping
    pool.full_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (full_k, full_v))
    pool.swa_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (swa_k, swa_v))
    backend = ATOMAttnBackendForSgl.__new__(ATOMAttnBackendForSgl)
    backend._is_mimo_v2_family = True
    backend._is_mimo_mtp = False
    backend._mimo_target_graph_capture = False
    backend.use_sliding_window_kv_pool = True
    backend.use_mla = False
    backend.token_to_kv_pool = pool
    backend.page_size = page
    backend.kv_cache_dtype = dtypes.fp8
    backend.input_dtype = torch.bfloat16
    backend.device = device
    backend.topk = 1
    backend.scale = dim**-0.5
    backend.max_context_len = 256
    backend.qo_indptr = torch.zeros(3, device=device, dtype=torch.int32)
    backend.req_to_token = torch.zeros((2, 256), device=device, dtype=torch.int64)
    backend.cuda_graph_swa_page_table = torch.zeros(
        (2, 4), device=device, dtype=torch.int32
    )
    backend.cuda_graph_swa_out_cache_loc = torch.zeros(
        4, device=device, dtype=torch.int64
    )
    layers = []
    scale_guards = []
    for layer_id, blocks in ((0, full_blocks), (1, swa_blocks)):
        ks = torch.ones((full_blocks, 1, page), device=device)
        vs = torch.ones_like(ks)
        scale_guards.append((ks, vs))
        layers.append(
            SimpleNamespace(
                layer_id=layer_id,
                is_cross_attention=False,
                sliding_window_size=128 if layer_id else -1,
                head_dim=dim,
                qk_head_dim=dim,
                v_head_dim=dim,
                tp_q_head_num=heads,
                tp_k_head_num=1,
                tp_v_head_num=1,
                scaling=dim**-0.5,
                sinks=None,
                k_scale=torch.nn.Parameter(ks[:blocks], requires_grad=False),
                v_scale=torch.nn.Parameter(vs[:blocks], requires_grad=False),
            )
        )

    lengths = [130, 150]
    full_slots = []
    for req, length in enumerate(lengths):
        slots = torch.arange(
            (10 + req * 3) * page,
            (10 + req * 3) * page + length,
            device=device,
            dtype=torch.int64,
        )
        backend.req_to_token[req, :length] = slots
        full_slots.append(slots)
    full_slots_flat = torch.cat(full_slots)
    k = torch.randn((sum(lengths), 1, dim), device=device, dtype=torch.bfloat16) * 0.125
    v = torch.cat(
        [
            torch.full((length, 1, dim), val, device=device, dtype=torch.bfloat16)
            for length, val in zip(lengths, (0.25, -0.5))
        ]
    )
    new_rows = torch.tensor([128, 129, 278, 279], device=device)
    batch = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
        req_pool_indices=torch.arange(2, device=device, dtype=torch.int64),
        out_cache_loc=full_slots_flat[new_rows],
        extend_prefix_lens_cpu=[length - 2 for length in lengths],
    )

    def populate(swap=False):
        for full_page in range(10, 16):
            relative = full_page - 10
            swa_page = 1 + ((relative + 3) % 6 if swap else relative)
            mapping[full_page * page : (full_page + 1) * page] = torch.arange(
                swa_page * page, (swa_page + 1) * page, device=device
            )
        for layer in layers:
            slots = mapping[full_slots_flat] if layer.layer_id else full_slots_flat
            kb, vb = pool.get_kv_buffer(layer.layer_id)
            set_kv_buffer_with_layout_shuffle(
                slots, k, v, kb, vb, layer.k_scale, layer.v_scale, page
            )

    def eager_metadata():
        backend.forward_metadata = ForwardMetadata(
            kv_indptr=torch.tensor([0, 130, 280], device=device, dtype=torch.int32),
            kv_indices=full_slots_flat,
            qo_indptr=torch.tensor([0, 2, 4], device=device, dtype=torch.int32),
            kv_last_page_len=None,
            max_q_len=2,
            max_kv_len=150,
            page_table=torch.tensor(
                [[10, 11, 12, 0], [13, 14, 15, 0]], device=device, dtype=torch.int32
            ),
            kv_lens=batch.seq_lens,
        )
        backend._init_mimo_swa_metadata(batch)

    populate()
    eager_metadata()
    return SimpleNamespace(
        backend=backend,
        layers=layers,
        batch=batch,
        pool=pool,
        populate=populate,
        eager_metadata=eager_metadata,
        q=torch.randn((4, heads * dim), device=device, dtype=torch.bfloat16) * 0.125,
        k=k[new_rows],
        v=v[new_rows],
        swa_guard=swa_k_guard[swa_blocks * page :],
        scale_guard=scale_guards[1][0][swa_blocks:],
    )


@pytest.mark.parametrize("mode", ["extend", "decode"])
@pytest.mark.parametrize("swa", [False, True])
def test_mimo_cache_write_stays_in_selected_pool(case, monkeypatch, mode, swa):
    backend, batch, layer = case.backend, case.batch, case.layers[int(swa)]
    monkeypatch.setattr(backend, "_forward_extend_mha_mimo", lambda q, *args: q)
    monkeypatch.setattr(backend, "_forward_decode_native_dense_mha", lambda q, *args: q)
    batch.forward_mode = ForwardMode.EXTEND if mode == "extend" else ForwardMode.DECODE
    slots = (
        case.pool.translate_loc_from_full_to_swa(batch.out_cache_loc)
        if swa
        else batch.out_cache_loc
    )
    layer.k_scale.fill_(7)
    getattr(backend, "forward_" + mode)(case.q, case.k, case.v, layer, batch)
    torch.cuda.synchronize()
    assert torch.all(layer.k_scale.flatten()[slots] != 7)
    assert torch.count_nonzero(case.swa_guard.float()) == 0
    assert torch.all(case.scale_guard == 1)


@pytest.mark.parametrize("swa", [False, True])
@pytest.mark.parametrize("path", ["prefix", "decode"])
def test_mimo_attention_reads_selected_pool(case, swa, path):
    backend, layer = case.backend, case.layers[int(swa)]
    if path == "prefix":
        output = backend._forward_extend_mha_mimo(
            case.q, case.k, case.v, layer, case.batch
        )
        expected = torch.tensor([0.25, 0.25, -0.5, -0.5], device="cuda")
    else:
        output = backend._forward_decode_native_dense_mha(
            case.q[::2], layer, case.batch
        )
        expected = torch.tensor([0.25, -0.5], device="cuda")
    torch.testing.assert_close(
        output.float(), expected[:, None].expand_as(output), atol=0.005, rtol=0
    )


def test_mimo_decode_metadata_reuses_graph_buffers_after_recycling(case):
    backend, batch = case.backend, case.batch
    original_slots = batch.out_cache_loc.clone()
    original_pages = backend.forward_metadata.page_table.clone()
    backend._init_mimo_swa_metadata(batch, for_cuda_graph=True)
    write_ptr = backend.forward_metadata.swa_out_cache_loc.data_ptr()
    page_ptr = backend.forward_metadata.swa_page_table.data_ptr()
    old_swa_pages = backend.forward_metadata.swa_page_table.clone()
    case.populate(swap=True)
    case.eager_metadata()
    backend._init_mimo_swa_metadata(batch, for_cuda_graph=True)
    md = backend.forward_metadata
    assert md.swa_out_cache_loc.data_ptr() == write_ptr
    assert md.swa_page_table.data_ptr() == page_ptr
    assert not torch.equal(old_swa_pages, md.swa_page_table)
    torch.testing.assert_close(batch.out_cache_loc, original_slots)
    torch.testing.assert_close(md.page_table, original_pages)
    torch.testing.assert_close(
        md.swa_out_cache_loc, case.pool.translate_loc_from_full_to_swa(original_slots)
    )


def test_mimo_prefix_read_after_old_swa_pages_are_evicted(case):
    backend, layer, batch = case.backend, case.layers[1], case.batch
    slots = torch.arange(6 * 64, 6 * 64 + 280, device="cuda", dtype=torch.int64)
    mapping = case.pool.full_to_swa_index_mapping
    mapping[slots[:128]] = 0
    mapping[slots[128:]] = torch.arange(64, 64 + 152, device="cuda")
    k = torch.full((152, 1, 192), 0.125, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, 0.75)
    kb, vb = case.pool.get_kv_buffer(1)
    set_kv_buffer_with_layout_shuffle(
        mapping[slots[128:]], k, v, kb, vb, layer.k_scale, layer.v_scale, 64
    )
    batch.batch_size = 1
    batch.out_cache_loc = slots[-2:]
    batch.extend_prefix_lens_cpu = [278]
    backend.forward_metadata = ForwardMetadata(
        kv_indptr=torch.tensor([0, 280], device="cuda", dtype=torch.int32),
        kv_indices=slots,
        qo_indptr=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        kv_last_page_len=None,
        max_q_len=2,
        max_kv_len=280,
        page_table=None,
        kv_lens=None,
    )
    backend._init_mimo_swa_metadata(batch)
    output = backend._forward_extend_mha_mimo(case.q[:2], k[-2:], v[-2:], layer, batch)
    torch.testing.assert_close(
        output.float(), torch.full_like(output.float(), 0.75), atol=0.005, rtol=0
    )


@pytest.mark.parametrize("mimo,hybrid", [(False, False), (False, True), (True, False)])
def test_swa_translation_is_scoped_to_mimo_hybrid_cache(case, mimo, hybrid):
    backend = case.backend
    backend._is_mimo_v2_family = mimo
    backend.use_sliding_window_kv_pool = hybrid
    md = SimpleNamespace()
    backend.forward_metadata = md
    backend._init_mimo_swa_metadata(case.batch)
    assert vars(md) == {}
    assert not backend._is_mimo_swa_layer(case.layers[1])


def test_mimo_target_verify_graph_replays_with_recycled_swa_pages(case):
    backend, batch, layer = case.backend, case.batch, case.layers[1]
    backend._mimo_target_graph_capture = True
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.seq_lens.sub_(2)
    full_page_buf = torch.zeros((2, 4), device="cuda", dtype=torch.int32)

    def update_metadata():
        pages, qo_indptr, max_q_len, swa_pages = backend._build_verify_unified_metadata(
            2,
            batch.seq_lens,
            batch.req_pool_indices,
            2,
            page_table_dest=full_page_buf,
            swa_page_table_dest=backend.cuda_graph_swa_page_table,
        )
        backend.cuda_graph_swa_out_cache_loc.copy_(
            case.pool.translate_loc_from_full_to_swa(batch.out_cache_loc)
        )
        backend.forward_metadata = AiterMetadata(
            None,
            pages,
            qo_indptr,
            None,
            max_q_len,
            256,
            swa_page_table=swa_pages,
            swa_out_cache_loc=backend.cuda_graph_swa_out_cache_loc,
        )

    update_metadata()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            backend.forward_extend(case.q, case.k, case.v, layer, batch)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = backend.forward_extend(case.q, case.k, case.v, layer, batch)
    expected = torch.tensor([0.25, 0.25, -0.5, -0.5], device="cuda")
    for swap in (False, True):
        case.populate(swap=swap)
        update_metadata()
        graph.replay()
        torch.cuda.synchronize()
        eager = backend.forward_extend(case.q, case.k, case.v, layer, batch)
        # Graph replay must match eager execution after the allocator remaps pages.
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        # FP8 PA also quantizes attention probabilities; its accumulation can
        # differ from the real-valued constant-V reference by roughly 1%.
        torch.testing.assert_close(
            output.float(), expected[:, None].expand_as(output), atol=0.01, rtol=0
        )
        assert torch.count_nonzero(case.swa_guard.float()) == 0
