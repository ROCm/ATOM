"""CUDA-graph pad rows must not keep a finished request's QSA page table."""

from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend.backend_resolver import real_batch_size
from atom.plugin.sglang.qwen4_exp_bridge import (
    _NO_WRITE,
    _ple_state_pool_slots,
    _query_start_loc,
    _seq_lens,
    build_qsa_metadata,
)


class _DecodeMode:
    @staticmethod
    def is_decode_or_idle():
        return True

    @staticmethod
    def is_extend():
        return False


class _Pool:
    def __init__(self):
        # Row 3 is a just-finished request whose pages were freed / poisoned.
        self.req_to_token = torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128)
        self.req_to_token[3] = 9_000_000


def test_real_batch_size_prefers_num_padding():
    fb = SimpleNamespace(batch_size=4, num_padding=1, _original_batch_size=4)
    assert real_batch_size(fb) == 3


def test_real_batch_size_uses_original_when_no_padding_field():
    fb = SimpleNamespace(batch_size=4, _original_batch_size=2)
    assert real_batch_size(fb) == 2


def test_seq_lens_and_query_start_loc_zero_cuda_graph_pad_rows():
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=4,
        num_padding=1,
        seq_lens=torch.tensor([1024, 1024, 1024, 1], dtype=torch.int32),
    )
    seq = _seq_lens(fb, device)
    loc = _query_start_loc(fb, num_tokens=4, device=device)
    assert torch.equal(seq, torch.tensor([1024, 1024, 1024, 0], dtype=torch.int32))
    assert torch.equal(loc, torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32))


def test_build_qsa_metadata_drops_stale_pad_page_table():
    pool = _Pool()
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=4,
        num_padding=1,
        device=device,
        req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([64, 64, 64, 1], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        page_size=64,
    )
    positions = torch.arange(4, dtype=torch.int64)
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )

    qsa = build_qsa_metadata(atom_config, fb, positions)

    assert qsa is not None
    assert int(qsa.seq_lens[-1]) == 0
    assert int(qsa.slot_mapping[-1]) == _NO_WRITE
    assert int(qsa.logical_positions[-1]) == _NO_WRITE
    assert int(qsa.block_tables[-1].max()) == 0
    assert int(qsa.block_tables[0, 0]) == 0


def test_compressed_slots_native_matches_floor_div():
    from atom.plugin.sglang.qwen4_exp_bridge import _compressed_slots_native

    slot = torch.tensor([64, 65, 66, 67, -1], dtype=torch.int64)
    logical = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    out = torch.empty(5, dtype=torch.int64)
    _compressed_slots_native(slot, logical, 4, out)
    # Only position 3 closes a group of 4; Native uses slot // ratio.
    assert out.tolist() == [-1, -1, -1, 16, -1]


def test_decode_graph_fill_drops_stale_pad_page_table():
    """Native-style persistent fill (``_DECODE_GRAPH.active``) must keep the pad contract."""
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    pool = _Pool()
    device = torch.device("cpu")
    _DECODE_GRAPH.ensure(max_bs=4, max_tokens=4, max_pages=8, device=device)
    _DECODE_GRAPH.active = True
    try:
        fb = SimpleNamespace(
            forward_mode=_DecodeMode(),
            batch_size=4,
            num_padding=1,
            device=device,
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([64, 64, 64, 1], dtype=torch.int32),
            req_to_token_pool=pool,
            out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            page_size=64,
        )
        positions = torch.arange(4, dtype=torch.int64)
        atom_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen4_exp",
                indexer_compress_ratio=4,
                indexer_budget=2048,
                page_size=64,
            )
        )
        qsa = build_qsa_metadata(atom_config, fb, positions)
        assert qsa is not None
        assert int(qsa.seq_lens[-1]) == 0
        assert int(qsa.slot_mapping[-1]) == _NO_WRITE
        assert int(qsa.logical_positions[-1]) == _NO_WRITE
        assert int(qsa.block_tables[-1].max()) == 0
        assert qsa.token_to_req.tolist() == [0, 1, 2, _NO_WRITE]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_ple_state_pool_slots_uses_mamba_pool_size():
    fb = SimpleNamespace(
        batch_size=4,
        attn_backend=None,
        req_to_token_pool=SimpleNamespace(size=64, req_to_token=torch.zeros(64, 8)),
        token_to_kv_pool=None,
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    assert _ple_state_pool_slots(fb, idx) >= 64


def test_plugin_ple_metadata_is_native():
    """Plugin PLE metadata is Native Qwen4ExpPLEMetadata, including num_accepted_tokens."""
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpPLEMetadata
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    assert bridge.Qwen4ExpPLEMetadata is Qwen4ExpPLEMetadata
    dummy = torch.zeros(2, dtype=torch.int32)
    md = Qwen4ExpPLEMetadata(
        query_start_loc=dummy,
        ngram_state=dummy,
        state_indices_in=dummy,
        state_indices_out=dummy,
        has_initial_state=dummy.bool(),
        conv_state=dummy,
    )
    assert md.num_accepted_tokens is None
    assert hasattr(bridge.Qwen4ExpQSAMetadata, "for_tokens")
