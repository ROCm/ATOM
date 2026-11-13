# SPDX-License-Identifier: MIT
"""A sparse capture's layout defines every selected key its warmup MLA reads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    import aiter  # noqa: F401

    from atom.model_ops.attentions import backends
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"aiter MLA backend unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize("q", [1, 4])
def test_a_sparse_capture_defines_every_selected_key(monkeypatch, q):
    """The warmup MLA reads `sparse_kv_indptr[tokens]` keys, one a row; an
    indexer writes only a row's context, which leaves a verify's early rows
    without one (DSA indexer mono on a replicated cache)."""
    monkeypatch.setattr(
        backends.CommonAttentionBuilder,
        "publish_capture_layout",
        lambda self, bs, max_seqlen_q: self._capture_layout_counts(bs, max_seqlen_q),
    )
    rows = (
        "context_lens",
        "kv_indptr",
        "slot_mapping",
        "kv_last_page_lens",
        "sparse_kv_indptr",
    )
    var = {name: SimpleNamespace(np=np.zeros(64, dtype=np.int32)) for name in rows}
    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    builder.is_sparse = True
    builder._publishes_dcp_local_lens = False
    builder.dcp_world_size = 1
    builder._sparse_kv_indices_gpu = torch.full((64,), -1, dtype=torch.int32)

    builder.publish_capture_layout(4, q)

    span = int(var["sparse_kv_indptr"].np[4 * q])
    assert span == 4 * q
    assert builder._sparse_kv_indices_gpu[:span].tolist() == [0] * span
