"""Unit tests for MTP draft index_share_for_mtp_iteration helpers.

Avoid importing ``atom.models.deepseek_mtp`` here: that module pulls in
``atom.config`` (torch / aiter / transformers) and is brittle in lightweight
test collection.  The helper below mirrors
``DeepSeekMultiTokenPredictor.set_skip_topk`` -- keep it in sync when editing
production code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

try:
    from atom.model_ops.attentions import aiter_mla
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.spec_decode import eagle_proposer
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the MLA builder and the drafter import AITER")


class _FakeMlaAttn:
    def __init__(self, buf: torch.Tensor):
        self.sparse_kv_indices_buffer = buf


class _FakeSelfAttn:
    def __init__(self, *, has_indexer: bool, buf: torch.Tensor):
        self.skip_topk = False
        self.indexer = object() if has_indexer else None
        self.mla_attn = _FakeMlaAttn(buf)


class _FakeMtpBlock:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.self_attn = self_attn


class _FakeLayer:
    def __init__(self, self_attn: _FakeSelfAttn):
        self.mtp_block = _FakeMtpBlock(self_attn)


def _set_skip_topk(layers: dict[str, Any], skip: bool) -> None:
    """Mirror of ``DeepSeekMultiTokenPredictor.set_skip_topk``."""
    for layer in layers.values():
        mtp_block = getattr(layer, "mtp_block", None)
        if mtp_block is None:
            continue
        self_attn = getattr(mtp_block, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "skip_topk"):
            continue
        if getattr(self_attn, "indexer", None) is not None:
            self_attn.skip_topk = skip


def test_set_skip_topk_only_layers_with_indexer():
    buf0 = torch.zeros(4, 8, dtype=torch.int32)
    buf1 = torch.zeros(4, 8, dtype=torch.int32)
    layers = {
        "80": _FakeLayer(_FakeSelfAttn(has_indexer=True, buf=buf0)),
        "81": _FakeLayer(_FakeSelfAttn(has_indexer=False, buf=buf1)),
    }

    _set_skip_topk(layers, True)

    assert layers["80"].mtp_block.self_attn.skip_topk is True
    assert layers["81"].mtp_block.self_attn.skip_topk is False


def test_compact_topk_indices_packs_the_rows_to_the_front(monkeypatch):
    """A sparse step 0 hands its last-token rows to `pack_row_selections`
    (tested on GPU in test_selection_pack) over the step's indptr and the
    shared buffer, and copies what it packed back to the front of both."""
    monkeypatch.setattr(
        aiter_mla,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_prefill=False), attn_metadata=None
        ),
    )
    calls = []

    def pack(rows, indptr, indices, out_indptr, out_indices, width):
        calls.append((rows, indptr, indices, width))
        out_indptr[:3] = torch.tensor([0, 5, 9])
        out_indices[: 2 * width] = torch.arange(100, 100 + 2 * width)

    monkeypatch.setattr(aiter_mla, "pack_row_selections", pack)
    indptr = torch.zeros(8, dtype=torch.int32)
    buf = torch.zeros(100, dtype=torch.int32)
    builder = AiterMLAMetadataBuilder.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        forward_vars={"sparse_kv_indptr": SimpleNamespace(gpu=indptr)}
    )
    builder._sparse_kv_indices_gpu = buf
    builder._packed_indptr = torch.zeros(5, dtype=torch.int32)
    builder._packed_indices = torch.zeros(32, dtype=torch.int32)
    builder.index_topk_out = 8
    rows = torch.tensor([1, 3])

    builder.compact_topk_indices(rows)

    ((rows_in, indptr_in, indices_in, width),) = calls
    assert rows_in is rows and indptr_in is indptr and indices_in is buf
    assert width == 8
    assert indptr[:3].tolist() == [0, 5, 9]
    assert buf[:16].tolist() == list(range(100, 116))


def test_compact_topk_indices_after_a_dense_prefill_selects_every_key(monkeypatch):
    """A prefill run dense selected nothing, because top-k would take every
    key: the selection is each sequence's whole context (a row past
    `scheduled_bs` none),
    laid out from the block tables at the indptr it is packed by."""
    block_tables = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    attn_metadata = SimpleNamespace(
        run_sparse=False,
        context_lens=torch.tensor([5, 3, 0], dtype=torch.int32),
        block_tables=block_tables,
        max_seqlen_k=5,
    )
    monkeypatch.setattr(
        aiter_mla,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_prefill=True), attn_metadata=attn_metadata
        ),
    )
    laid_out = []
    monkeypatch.setattr(
        aiter_mla,
        "kv_indices_generate_triton",
        lambda *args: laid_out.append(args),
    )
    indptr = torch.full((8,), 99, dtype=torch.int32)
    buf = torch.empty(64, dtype=torch.int32)
    builder = AiterMLAMetadataBuilder.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        forward_vars={"sparse_kv_indptr": SimpleNamespace(gpu=indptr)}
    )
    builder._sparse_kv_indices_gpu = buf
    builder.index_topk = 4
    builder.block_ratio = 1

    builder.compact_topk_indices(torch.tensor([4, 7, 0]))

    # Clamped at top-k, as the prefill's own sparse counts are.
    assert indptr[:4].tolist() == [0, 4, 7, 7]
    ((tables, out, packed_indptr, ratio, max_k),) = laid_out
    assert tables.data_ptr() == block_tables.data_ptr() and out is buf
    assert packed_indptr.tolist() == [0, 4, 7, 7] and (ratio, max_k) == (1, 5)


@pytest.mark.parametrize(
    "method,index_share,index_topk,has_api,dcp,pcp,expected",
    [
        ("mtp", True, 2048, True, 1, 1, True),
        ("mtp", False, 2048, True, 1, 1, False),
        ("eagle3", True, 2048, True, 1, 1, False),
        ("mtp", True, None, True, 1, 1, False),
        ("mtp", True, 2048, False, 1, 1, False),
        ("mtp", True, 2048, True, 2, 1, False),
        ("mtp", True, 2048, True, 1, 2, False),
    ],
)
def test_share_mtp_indices_gate(
    monkeypatch, method, index_share, index_topk, has_api, dcp, pcp, expected
):
    draft_hf = SimpleNamespace(index_share_for_mtp_iteration=index_share)
    if index_topk is not None:
        draft_hf.index_topk = index_topk
    mtp_inner = (
        SimpleNamespace(set_skip_topk=lambda _: None) if has_api else SimpleNamespace()
    )

    def drafter_init(self, atom_config, device, runner):
        self.speculative_config = SimpleNamespace(
            method=method, draft_model_hf_config=draft_hf
        )
        self.model = SimpleNamespace(model=mtp_inner)

    monkeypatch.setattr(eagle_proposer.Drafter, "__init__", drafter_init)
    monkeypatch.setattr(eagle_proposer, "get_dcp_world_size", lambda: dcp)
    monkeypatch.setattr(eagle_proposer, "get_pcp_world_size", lambda: pcp)
    proposer = eagle_proposer.EagleProposer(None, None, None)
    assert proposer.share_mtp_indices is expected
