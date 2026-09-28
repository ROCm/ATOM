"""CUDA-graph pad rows must not keep a finished request's QSA page table."""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.attention_backend.backend_resolver import real_batch_size
from atom.plugin.sglang.qwen4_exp_bridge import (
    _DECODE_GRAPH,
    _DRAFT_QSA,
    _NO_WRITE,
    _eager_qsa_max_seq_len,
    _fill_block_tables_into,
    _lift_qsa_seq_lens,
    _ple_state_pool_slots,
    _query_start_loc,
    _seq_lens,
    _use_decode_graph_buffers,
    bind_qsa_replay_batch,
    build_qsa_metadata,
)


@pytest.fixture(autouse=True)
def _reset_decode_graph():
    _DECODE_GRAPH.reset()
    _DRAFT_QSA.reset()
    yield
    _DECODE_GRAPH.reset()
    _DRAFT_QSA.reset()


class _DecodeMode:
    @staticmethod
    def is_decode_or_idle():
        return True

    @staticmethod
    def is_extend():
        return False

    @staticmethod
    def is_target_verify():
        return False


class _VerifyMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_extend():
        return True

    @staticmethod
    def is_target_verify():
        return True


class _Pool:
    def __init__(self):
        # Row 3 is a just-finished request whose pages were freed / poisoned.
        # SGLang allocates live pages starting at 1; page 0 is dummy storage.
        self.req_to_token = torch.arange(64, 64 + 4 * 128, dtype=torch.int32).reshape(
            4, 128
        )
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
    assert fb.seq_lens.tolist() == [1024, 1024, 1024, 1]


def test_page_table_masks_dummy_page_in_any_column_and_unallocated_tail():
    # The final column contains stale, nonzero mappings beyond each context.
    pool = SimpleNamespace(
        req_to_token=(
            torch.tensor([[0, 3, 9], [4, 0, 8], [7, 6, 5]], dtype=torch.int32)
            .repeat_interleave(64, dim=1)
            .mul(64)
        )
    )
    out = torch.full((3, 3), 999, dtype=torch.int32)
    _fill_block_tables_into(
        out,
        pool=pool,
        req_pool_indices=torch.arange(3),
        table_tokens=192,
        block_size=64,
        live_bs=2,
        seq_lens=torch.tensor([65, 65, 1]),
    )
    assert out.tolist() == [[-1, 3, -1], [4, -1, -1], [-1, -1, -1]]


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
    assert int(qsa.block_tables[-1].max()) == _NO_WRITE
    assert int(qsa.block_tables[0, 0]) == 1


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
    _DECODE_GRAPH.allocate_once(max_bs=4, max_tokens=4, max_pages=8, device=device)
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
        assert int(qsa.block_tables[-1].max()) == _NO_WRITE
        assert qsa.token_to_req.tolist() == [0, 1, 2, _NO_WRITE]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_lift_qsa_seq_lens_eagle_tree_but_not_decode():
    """EAGLE tree queries sit at seq_lens; greedy decode (pos = seq_lens - 1) is a no-op."""
    decode_seq = torch.tensor([64], dtype=torch.int32)
    _lift_qsa_seq_lens(
        decode_seq,
        torch.tensor([63], dtype=torch.int64),
        live_bs=1,
        tokens_per_req=1,
    )
    assert decode_seq.tolist() == [64]

    draft_seq = torch.tensor([64], dtype=torch.int32)
    _lift_qsa_seq_lens(
        draft_seq,
        torch.tensor([64], dtype=torch.int64),
        live_bs=1,
        tokens_per_req=1,
    )
    assert draft_seq.tolist() == [65]

    verify_seq = torch.tensor([64, 80, 0], dtype=torch.int32)
    _lift_qsa_seq_lens(
        verify_seq,
        torch.tensor([64, 65, 66, 80, 81, 82, -1, -1, -1], dtype=torch.int64),
        live_bs=2,
        tokens_per_req=3,
    )
    assert verify_seq.tolist() == [67, 83, 0]


def test_lift_qsa_seq_lens_does_not_broadcast_global_max_when_ragged():
    """A short leftover tail must not lift every request to the batch max pos."""
    seq = torch.tensor([10, 80], dtype=torch.int32)
    _lift_qsa_seq_lens(
        seq,
        torch.tensor([10, 11, 12, 80], dtype=torch.int64),
        live_bs=2,
        tokens_per_req=3,
    )
    assert seq.tolist() == [13, 80]


def test_mixed_length_verify_page_tables_do_not_alias_page_zero():
    """Short request tails in a mixed batch must not score physical page 0."""
    pool = _Pool()
    fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=2,
        num_padding=0,
        device=torch.device("cpu"),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([10, 80], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.tensor([10, 11, 12, 80, 81, 82], dtype=torch.int64),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )
    qsa = build_qsa_metadata(
        atom_config,
        fb,
        torch.tensor([10, 11, 12, 80, 81, 82], dtype=torch.int64),
    )
    assert qsa is not None
    assert qsa.seq_lens.tolist() == [13, 83]
    assert qsa.token_to_req.tolist() == [0, 0, 0, 1, 1, 1]
    assert int(qsa.block_tables[0, 0]) == 1
    assert int(qsa.block_tables[0, 1]) == _NO_WRITE
    assert int(qsa.block_tables[1, 0]) == 3
    assert int(qsa.block_tables[1, 1]) == 4


def test_verify_query_start_loc_ignores_stale_extend_starts():
    """TARGET_VERIFY is_extend(); leftover prefill cu_seqlens must not win."""
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=2,
        num_padding=0,
        extend_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        extend_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    loc = _query_start_loc(fb, num_tokens=6, device=device)
    assert loc.tolist() == [0, 3, 6]


class _PrefillMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_extend():
        return True

    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_draft_extend_v2():
        return False


def test_prefill_query_start_loc_ignores_leftover_draft_token_num():
    """True prefill keeps cu_seqlens; leftover EAGLE tokens_per_req must not re-pack it."""
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_PrefillMode(),
        batch_size=2,
        num_padding=0,
        extend_start_loc=torch.tensor([0, 128], dtype=torch.int32),
        extend_seq_lens=torch.tensor([128, 896], dtype=torch.int32),
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    loc = _query_start_loc(fb, num_tokens=1024, device=device)
    assert loc.tolist() == [0, 128, 1024]

    missing_cu = SimpleNamespace(
        forward_mode=_PrefillMode(),
        batch_size=2,
        num_padding=0,
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    loc = _query_start_loc(missing_cu, num_tokens=1024, device=device)
    assert loc.tolist() == [0, 0, 1024]


class _DraftExtendMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_extend():
        return False

    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_draft_extend_v2():
        return True


def test_draft_extend_query_start_loc_is_packed():
    """DRAFT_EXTEND_V2 uses packed offsets, independent of is_extend()."""
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_DraftExtendMode(),
        batch_size=2,
        num_padding=0,
        extend_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        extend_seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    loc = _query_start_loc(fb, num_tokens=6, device=device)
    assert loc.tolist() == [0, 3, 6]


def test_verify_qsa_seq_lens_cover_tree_root_at_prefix():
    pool = _Pool()
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=1,
        num_padding=0,
        device=device,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([64], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.tensor([64, 65, 66], dtype=torch.int64),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )
    qsa = build_qsa_metadata(
        atom_config, fb, torch.tensor([64, 65, 66], dtype=torch.int64)
    )
    assert qsa is not None
    assert qsa.seq_lens.tolist() == [67]
    assert qsa.logical_positions.tolist() == [64, 65, 66]


def test_draft_decode_qsa_lifts_one_token_step(monkeypatch):
    """Draft decode is 1 token/req but sits at seq_lens, so QSA must include it."""
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    monkeypatch.setattr(bridge, "_is_draft_forward", lambda: True)
    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(
            context_length=8192,
            max_model_len=8192,
            max_running_requests=8,
            page_size=64,
            speculative_num_draft_tokens=3,
        ),
    )
    pool = _Pool()
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=1,
        num_padding=0,
        device=device,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([64], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.tensor([64], dtype=torch.int64),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )
    qsa = build_qsa_metadata(atom_config, fb, torch.tensor([64], dtype=torch.int64))
    assert qsa is not None
    assert qsa.seq_lens.tolist() == [65]
    assert qsa.logical_positions.tolist() == [64]


def test_qsa_tokens_per_req_is_one_on_decode_three_on_verify():
    from atom.plugin.sglang.qwen4_exp_bridge import _get_qsa_tokens_per_req

    leftover = SimpleNamespace(draft_token_num=3, num_tokens_per_req=3)
    decode = SimpleNamespace(forward_mode=_DecodeMode(), spec_info=leftover)
    verify = SimpleNamespace(forward_mode=_VerifyMode(), spec_info=leftover)
    assert _get_qsa_tokens_per_req(decode) == 1
    assert _get_qsa_tokens_per_req(verify) == 3


def test_graph_max_tokens_per_req_follows_cli(monkeypatch):
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    monkeypatch.setattr(bridge, "_server_args", lambda: None)
    assert bridge._get_qsa_graph_max_tokens_per_req() == 1
    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(speculative_num_draft_tokens=3),
    )
    assert bridge._get_qsa_graph_max_tokens_per_req() == 3
    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(speculative_num_draft_tokens=0),
    )
    assert bridge._get_qsa_graph_max_tokens_per_req() == 1


def test_verify_graph_fill_drops_stale_pad_page_table():
    """TARGET_VERIFY graphs are width-3; pad-row tokens must stay _NO_WRITE."""
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    pool = _Pool()
    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=4, max_tokens=12, max_pages=8, device=device)
    _DECODE_GRAPH.active = True
    try:
        fb = SimpleNamespace(
            forward_mode=_VerifyMode(),
            batch_size=4,
            num_padding=1,
            device=device,
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([64, 64, 64, 1], dtype=torch.int32),
            req_to_token_pool=pool,
            out_cache_loc=torch.arange(12, dtype=torch.int64),
            page_size=64,
            spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
        )
        positions = torch.arange(12, dtype=torch.int64)
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
        assert qsa.token_to_req.tolist() == [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            _NO_WRITE,
            _NO_WRITE,
            _NO_WRITE,
        ]
        assert qsa.slot_mapping[:9].tolist() == list(range(9))
        assert qsa.slot_mapping[9:].tolist() == [_NO_WRITE] * 3
        assert qsa.logical_positions[9:].tolist() == [_NO_WRITE] * 3
        assert int(qsa.block_tables[-1].max()) == _NO_WRITE
        # Prefix seq_lens=64 with dummy arange positions 0..8 stays 64.
        assert qsa.seq_lens[:3].tolist() == [64, 64, 64]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_verify_graph_fill_covers_tree_positions_past_prefix():
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    pool = _Pool()
    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=2, max_tokens=6, max_pages=8, device=device)
    _DECODE_GRAPH.active = True
    try:
        fb = SimpleNamespace(
            forward_mode=_VerifyMode(),
            batch_size=1,
            num_padding=0,
            device=device,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([64], dtype=torch.int32),
            req_to_token_pool=pool,
            out_cache_loc=torch.tensor([64, 65, 66], dtype=torch.int64),
            page_size=64,
            spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
        )
        qsa = build_qsa_metadata(
            atom_config := SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="qwen4_exp",
                    indexer_compress_ratio=4,
                    indexer_budget=2048,
                    page_size=64,
                )
            ),
            fb,
            torch.tensor([64, 65, 66], dtype=torch.int64),
        )
        del atom_config
        assert qsa is not None
        assert qsa.seq_lens.tolist() == [67]
        assert qsa.logical_positions.tolist() == [64, 65, 66]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_verify_graph_fill_masks_current_bucket_after_larger_batch():
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    pool = _Pool()
    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=4, max_tokens=12, max_pages=8, device=device)
    _DECODE_GRAPH.active = True
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )
    try:
        wide = SimpleNamespace(
            forward_mode=_VerifyMode(),
            batch_size=4,
            num_padding=0,
            device=device,
            req_pool_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([64, 64, 64, 64], dtype=torch.int32),
            req_to_token_pool=pool,
            out_cache_loc=torch.arange(12, dtype=torch.int64),
            page_size=64,
            spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
        )
        build_qsa_metadata(atom_config, wide, torch.arange(12, dtype=torch.int64))
        narrow = SimpleNamespace(
            forward_mode=_VerifyMode(),
            batch_size=2,
            num_padding=1,
            device=device,
            req_pool_indices=torch.tensor([0, 3], dtype=torch.int32),
            seq_lens=torch.tensor([64, 1], dtype=torch.int32),
            req_to_token_pool=pool,
            out_cache_loc=torch.tensor([7, 8, 9, 999, 999, 999], dtype=torch.int64),
            page_size=64,
            spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
        )
        qsa = build_qsa_metadata(
            atom_config, narrow, torch.tensor([10, 11, 12, 999, 999, 999])
        )
        assert qsa is not None
        assert qsa.slot_mapping.tolist() == [7, 8, 9, -1, -1, -1]
        assert qsa.token_to_req.tolist() == [0, 0, 0, -1, -1, -1]
        assert qsa.logical_positions.tolist() == [10, 11, 12, -1, -1, -1]
        assert qsa.compressed_slot_mapping[3:].tolist() == [-1, -1, -1]
        assert qsa.seq_lens.tolist() == [64, 0]
        assert int(qsa.block_tables[1].max()) == _NO_WRITE
        # Source metadata belongs to SGLang and must remain untouched.
        assert narrow.req_pool_indices.tolist() == [0, 3]
        assert narrow.seq_lens.tolist() == [64, 1]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_prepare_verify_graph_metadata_pins_width_three_buffers(monkeypatch):
    from atom.plugin.sglang import qwen4_exp_bridge as bridge
    from atom.plugin.sglang.qwen4_exp_bridge import (
        _DECODE_GRAPH,
        prepare_qwen4_exp_decode_graph_metadata,
    )

    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
        ),
    )
    pool = _Pool()
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=2,
        num_padding=0,
        device=device,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.arange(6, dtype=torch.int64),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3),
        positions=torch.arange(6, dtype=torch.int64),
        cuda_graph_max_bs_decode=4,
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )
    try:
        qsa = prepare_qwen4_exp_decode_graph_metadata(
            fb, in_capture=True, atom_config=atom_config
        )
        assert qsa is not None
        assert _DECODE_GRAPH.active
        assert _DECODE_GRAPH.max_tokens >= 12
        assert qsa.token_to_req.tolist() == [0, 0, 0, 1, 1, 1]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_token_to_req_from_packed_writes_into_out():
    from atom.plugin.sglang.qwen4_exp_bridge import _token_to_req_from_packed

    buf = torch.full((6,), 7, dtype=torch.int32)
    ptr = buf.data_ptr()
    out = _token_to_req_from_packed(
        live_bs=2,
        tokens_per_req=2,
        num_tokens=6,
        device=buf.device,
        out=buf,
    )
    assert buf.data_ptr() == ptr
    assert out.data_ptr() == ptr
    assert buf.tolist() == [0, 0, 1, 1, _NO_WRITE, _NO_WRITE]


def test_token_to_req_gpu_fill_keeps_graph_gpu_address():
    from atom.plugin.sglang.qwen4_exp_bridge import (
        _DECODE_GRAPH,
        _token_to_req_from_packed,
    )

    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=2, max_tokens=8, max_pages=4, device=device)
    gpu_ptr = _DECODE_GRAPH.token_to_req.data_ptr()

    def fill(*, live_bs: int, tokens_per_req: int, num_tokens: int) -> None:
        _token_to_req_from_packed(
            live_bs=live_bs,
            tokens_per_req=tokens_per_req,
            num_tokens=num_tokens,
            device=device,
            out=_DECODE_GRAPH.token_to_req,
        )

    fill(live_bs=1, tokens_per_req=3, num_tokens=4)
    assert _DECODE_GRAPH.token_to_req.data_ptr() == gpu_ptr
    assert _DECODE_GRAPH.token_to_req[:4].tolist() == [0, 0, 0, _NO_WRITE]
    fill(live_bs=2, tokens_per_req=2, num_tokens=4)
    assert _DECODE_GRAPH.token_to_req[:4].tolist() == [0, 0, 1, 1]
    assert _DECODE_GRAPH.token_to_req.data_ptr() == gpu_ptr
    fill(live_bs=0, tokens_per_req=3, num_tokens=4)
    assert _DECODE_GRAPH.token_to_req[:4].tolist() == [_NO_WRITE] * 4
    assert _DECODE_GRAPH.token_to_req.data_ptr() == gpu_ptr


def test_eager_qsa_max_seq_len_uses_host_prefix_not_gpu_tensor():
    fb = SimpleNamespace(seq_lens_cpu=[40, 100, 8])
    gpu_lens = torch.zeros(3, dtype=torch.int32)
    if torch.cuda.is_available():
        gpu_lens = gpu_lens.cuda()
    assert _eager_qsa_max_seq_len(fb, gpu_lens, tokens_per_req=1, ctx_len=65536) == 100
    assert _eager_qsa_max_seq_len(fb, gpu_lens, tokens_per_req=3, ctx_len=65536) == 103


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


def test_bind_qsa_replay_batch_attaches_omitted_pool():
    """CUDA-graph replay views drop req_to_token_pool; bind it back."""
    pool = _Pool()
    fb = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([64], dtype=torch.int32),
        positions=torch.tensor([64, 65, 66], dtype=torch.int64),
    )
    backend = SimpleNamespace(req_to_token_pool=pool, page_size=64, device="cpu")
    bind_qsa_replay_batch(fb, backend)
    assert fb.req_to_token_pool is pool


def test_verify_replay_view_without_pool_fills_after_bind():
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    pool = _Pool()
    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=2, max_tokens=6, max_pages=8, device=device)
    _DECODE_GRAPH.active = True
    try:
        fb = SimpleNamespace(
            forward_mode=_VerifyMode(),
            batch_size=1,
            num_padding=0,
            positions=torch.tensor([64, 65, 66], dtype=torch.int64),
            seq_lens=torch.tensor([64], dtype=torch.int32),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            out_cache_loc=torch.tensor([64, 65, 66], dtype=torch.int64),
            spec_info=SimpleNamespace(draft_token_num=3, num_tokens_per_req=3),
        )
        backend = SimpleNamespace(req_to_token_pool=pool, page_size=64, device=device)
        bind_qsa_replay_batch(fb, backend)
        atom_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen4_exp",
                indexer_compress_ratio=4,
                indexer_budget=2048,
                page_size=64,
            )
        )
        qsa = build_qsa_metadata(
            atom_config, fb, torch.tensor([64, 65, 66], dtype=torch.int64)
        )
        assert qsa is not None
        assert qsa.seq_lens.tolist() == [67]
        assert qsa.logical_positions.tolist() == [64, 65, 66]
    finally:
        _DECODE_GRAPH.active = False
        _DECODE_GRAPH.last_qsa = None


def test_use_decode_graph_buffers_on_target_decode_and_verify():
    from atom.plugin.sglang.qwen4_exp_bridge import _DECODE_GRAPH

    fb = SimpleNamespace(forward_mode=_DecodeMode(), batch_size=1)
    _DECODE_GRAPH.allocate_once(
        max_bs=2, max_tokens=6, max_pages=4, device=torch.device("cpu")
    )
    try:
        assert _use_decode_graph_buffers(fb) is True
        assert (
            _use_decode_graph_buffers(
                SimpleNamespace(forward_mode=_VerifyMode(), batch_size=1)
            )
            is True
        )
    finally:
        _DECODE_GRAPH.active = False


def test_use_decode_graph_buffers_false_when_is_draft(monkeypatch):
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    monkeypatch.setattr(bridge, "_is_draft_forward", lambda: True)
    bridge._DECODE_GRAPH.active = True
    try:
        fb = SimpleNamespace(forward_mode=_DecodeMode(), batch_size=1)
        assert bridge._use_decode_graph_buffers(fb) is False
    finally:
        bridge._DECODE_GRAPH.active = False


def _flash_atom_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp",
            indexer_compress_ratio=4,
            indexer_budget=2048,
            page_size=64,
        )
    )


@pytest.mark.parametrize(
    "max_bs,max_tokens,max_pages,device",
    [
        (8, 24, 128, torch.device("cpu")),
        (2, 2, 8, torch.device("cpu")),
        (2, 2, 4, torch.device("meta")),
    ],
)
def test_ensure_refuses_grow_after_first_alloc(max_bs, max_tokens, max_pages, device):
    _DECODE_GRAPH.allocate_once(
        max_bs=2, max_tokens=2, max_pages=4, device=torch.device("cpu")
    )
    ptr = _DECODE_GRAPH.block_tables.data_ptr()
    tok_ptr = _DECODE_GRAPH.token_to_req.data_ptr()
    with pytest.raises(RuntimeError, match="capacity/device mismatch"):
        _DECODE_GRAPH.allocate_once(
            max_bs=max_bs, max_tokens=max_tokens, max_pages=max_pages, device=device
        )
    assert _DECODE_GRAPH.block_tables.data_ptr() == ptr
    assert _DECODE_GRAPH.token_to_req.data_ptr() == tok_ptr
    assert _DECODE_GRAPH.max_bs == 2
    assert _DECODE_GRAPH.max_tokens == 2
    assert _DECODE_GRAPH.max_pages == 4


def test_decode_capture_preallocates_verify_width_and_full_pages(monkeypatch):
    """Decode capture must already own MTP verify width + full context pages."""
    from atom.plugin.sglang import qwen4_exp_bridge as bridge
    from atom.plugin.sglang.qwen4_exp_bridge import (
        prepare_qwen4_exp_decode_graph_metadata,
    )

    args = SimpleNamespace(
        speculative_algorithm="EAGLE",
        speculative_num_steps=2,
        speculative_num_draft_tokens=3,
        cuda_graph_max_bs_decode=8,
        cuda_graph_max_bs=None,
        cuda_graph_config=None,
        max_running_requests=8,
        context_length=8192,
        max_model_len=8192,
        page_size=64,
    )
    monkeypatch.setattr(bridge, "_server_args", lambda: args)
    pool = _Pool()
    device = torch.device("cpu")
    decode_fb = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=1,
        num_padding=0,
        device=device,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([16], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.tensor([0], dtype=torch.int64),
        page_size=64,
        spec_info=None,
        positions=torch.tensor([15], dtype=torch.int64),
        cuda_graph_max_bs_decode=8,
    )
    qsa = prepare_qwen4_exp_decode_graph_metadata(
        decode_fb, in_capture=True, atom_config=_flash_atom_config()
    )
    assert qsa is not None
    assert _DECODE_GRAPH.max_tokens >= 24
    assert _DECODE_GRAPH.max_pages >= 128
    ptr = _DECODE_GRAPH.token_to_req.data_ptr()
    pages_ptr = _DECODE_GRAPH.block_tables.data_ptr()

    verify_fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=2,
        num_padding=0,
        device=device,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([64, 64], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.arange(6, dtype=torch.int64),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3),
        positions=torch.arange(6, dtype=torch.int64),
        cuda_graph_max_bs_decode=8,
    )
    qsa2 = prepare_qwen4_exp_decode_graph_metadata(
        verify_fb, in_capture=True, atom_config=_flash_atom_config()
    )
    assert qsa2 is not None
    assert _DECODE_GRAPH.token_to_req.data_ptr() == ptr
    assert _DECODE_GRAPH.block_tables.data_ptr() == pages_ptr
    assert _DECODE_GRAPH.max_tokens == 24
    assert _DECODE_GRAPH.max_pages == 128


@pytest.mark.parametrize(
    "decode,running,expected",
    [
        (SimpleNamespace(bs=[1, 4, 12], max_bs=32), 16, 12),
        (SimpleNamespace(bs=[1, 4, 12], max_bs=32), 8, 8),
        (SimpleNamespace(bs=None, max_bs=12), 16, 12),
        (None, 16, 16),
    ],
)
def test_graph_capacity_uses_resolved_config_before_legacy_flags(
    monkeypatch, decode, running, expected
):
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=decode),
            cuda_graph_max_bs_decode=32,
            max_running_requests=running,
            speculative_num_draft_tokens=3,
            context_length=8192,
        ),
    )
    # The first capture must reserve the larger future buckets, regardless of
    # stale convenience fields attached to a batch by an older caller.
    fb = SimpleNamespace(
        batch_size=1, device=torch.device("cpu"), cuda_graph_max_bs_decode=64
    )
    bs, tokens, pages, device = bridge._decode_graph_capacity(_flash_atom_config(), fb)
    assert bs == expected
    assert tokens == expected * 3
    assert pages == 128
    assert device == torch.device("cpu")


def test_verify_above_graph_capacity_uses_separate_eager_metadata():
    device = torch.device("cpu")
    _DECODE_GRAPH.allocate_once(max_bs=2, max_tokens=6, max_pages=2, device=device)
    _DECODE_GRAPH.block_tables.fill_(-7)
    pointer = _DECODE_GRAPH.block_tables.data_ptr()
    fb = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=3,
        num_padding=0,
        req_pool_indices=torch.arange(3, dtype=torch.int32),
        seq_lens=torch.full((3,), 64, dtype=torch.int32),
        req_to_token_pool=_Pool(),
        out_cache_loc=torch.arange(64, 73),
        page_size=64,
        spec_info=SimpleNamespace(draft_token_num=3),
    )
    positions = torch.tensor([64, 65, 66] * 3)
    md = build_qsa_metadata(_flash_atom_config(), fb, positions)
    assert md.seq_lens.tolist() == [67, 67, 67]
    assert md.token_to_req.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert md.block_tables.data_ptr() != pointer
    assert _DECODE_GRAPH.block_tables.data_ptr() == pointer
    assert torch.all(_DECODE_GRAPH.block_tables == -7)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("has_gdn", [False, True])
def test_eager_ple_preserves_state_indices_and_acceptance(
    monkeypatch, prefill, has_gdn
):
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    idx = torch.tensor([1, 2], dtype=torch.int32)
    idx_in = torch.tensor([0, 3], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    gdn = (
        SimpleNamespace(
            non_spec_state_indices_tensor=idx,
            non_spec_state_indices_in_tensor=idx_in,
            num_accepted_tokens=accepted,
        )
        if has_gdn
        else None
    )
    monkeypatch.setattr(
        bridge, "_linear_static_ple_slots", lambda *_: (torch.tensor([0, 3, 6]), idx)
    )
    monkeypatch.setattr(bridge, "_ple_state_pool_slots", lambda *_: 4)
    monkeypatch.setattr(
        bridge,
        "_ensure_ple_states",
        lambda *_: (torch.zeros(4, 8, 3), torch.zeros(4, 2, dtype=torch.int64)),
    )
    fb = SimpleNamespace(
        forward_mode=_PrefillMode() if prefill else _VerifyMode(),
        batch_size=2,
        num_padding=1,
        spec_info=SimpleNamespace(draft_token_num=3),
        extend_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        extend_seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0, 5], dtype=torch.int32),
    )
    config = SimpleNamespace(hf_config=SimpleNamespace(ple_layer_ids=[0]))
    md = bridge.build_ple_metadata(
        config, fb, torch.arange(6), model=object(), gdn_metadata=gdn
    )
    if prefill and not has_gdn:
        assert md is None
        return
    assert md.query_start_loc.tolist() == [0, 3, 3]
    assert md.state_indices_out.tolist() == [1, -1]
    assert md.state_indices_in.tolist() == ([0, -1] if has_gdn else [1, -1])
    assert md.has_initial_state.tolist() == [not prefill, False]
    if has_gdn:
        assert md.num_accepted_tokens is accepted
    else:
        assert md.num_accepted_tokens.tolist() == [1, 1]
    # Padding must not modify scheduler-owned state indices.
    assert idx.tolist() == [1, 2]
    assert idx_in.tolist() == [0, 3]


def test_eager_prefill_max_seq_len_ignores_stale_decode_graph_flag(monkeypatch):
    """Leftover ``_DECODE_GRAPH.active`` must not score a prefill at context_length."""
    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    class _Prefill:
        @staticmethod
        def is_decode_or_idle():
            return False

        @staticmethod
        def is_extend():
            return True

        @staticmethod
        def is_target_verify():
            return False

        @staticmethod
        def is_draft_extend_v2():
            return False

    monkeypatch.setattr(
        bridge,
        "_server_args",
        lambda: SimpleNamespace(
            context_length=131072, max_model_len=131072, page_size=64
        ),
    )
    pool = _Pool()
    device = torch.device("cpu")
    bridge._DECODE_GRAPH.allocate_once(
        max_bs=1, max_tokens=8, max_pages=8, device=device
    )
    fb = SimpleNamespace(
        forward_mode=_Prefill(),
        batch_size=1,
        device=device,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([4096], dtype=torch.int32),
        extend_start_loc=torch.tensor([0], dtype=torch.int32),
        extend_seq_lens=torch.tensor([8], dtype=torch.int32),
        req_to_token_pool=pool,
        out_cache_loc=torch.arange(8, dtype=torch.int64),
        page_size=64,
    )
    qsa = build_qsa_metadata(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen4_exp",
                indexer_compress_ratio=4,
                indexer_budget=2048,
                page_size=64,
            )
        ),
        fb,
        torch.arange(8, dtype=torch.int64),
    )
    assert qsa is not None
    assert qsa.max_seq_len == 4096
