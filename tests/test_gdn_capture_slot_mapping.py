# SPDX-License-Identifier: MIT
"""The GDN decode capture gives every row of its batch a slot.

A captured decode runs bs * max_qlen rows: k + 1 per sequence under MTP-k. A
fused cache write that bounds its rows by the slot table bakes that bound into
the graph, so a table of bs slots drops the K/V of every MTP row past it on
each replay. `prepare_decode` already gives every row a slot
(`assert_shape_contract`); this is the capture's half, as the aiter builder
does it.

No device: the builder runs on CPU tensors, with its GDN state and work plan
stubbed. Like the other attention tests, this file skips where triton or aiter
does not import.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="gdn_attn defines @triton.jit kernels")
pytest.importorskip("aiter", reason="gdn_attn imports the AITER runtime")

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

ROWS = 64


def _builder(max_qlen):
    """The part of a GDN builder its capture reads, over CPU buffers."""
    var = {
        name: SimpleNamespace(gpu=torch.zeros(ROWS, dtype=torch.int32))
        for name in (
            "slot_mapping",
            "context_lens",
            "block_tables",
            "cu_seqlens_q",
            "kv_indptr",
            "kv_indices",
        )
    }
    var["max_qlen"] = max_qlen
    var["positions"] = SimpleNamespace(
        copy_to_gpu=lambda n: torch.zeros(n, dtype=torch.int64)
    )
    builder = GDNAttentionMetadataBuilder.__new__(GDNAttentionMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        forward_vars=var, config=SimpleNamespace(max_model_len=128)
    )
    builder.block_size = 16
    builder._build_gdn_capture_metadata = lambda bs: None
    builder.refresh_flydsl_plan = lambda context_lens, create: None
    return builder


@pytest.mark.parametrize("max_qlen", [1, 4])
def test_capture_gives_every_row_a_slot(max_qlen):
    bs = 8
    builder = _builder(max_qlen)
    attn_metadata, context = builder.build_for_cudagraph_capture(bs)

    assert context.running_tokens == bs * max_qlen
    assert attn_metadata.slot_mapping.shape[0] == context.running_tokens
    assert attn_metadata.context_lens.shape[0] == bs
    # The graph reads the buffer that `prepare_decode` fills before a replay.
    slots = builder.model_runner.forward_vars["slot_mapping"].gpu
    assert attn_metadata.slot_mapping.data_ptr() == slots.data_ptr()
