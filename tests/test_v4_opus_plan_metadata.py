import numpy as np
import pytest
import torch

attn = pytest.importorskip(
    "atom.model_ops.attentions.deepseek_v4_attn",
    reason="the V4 builder's module imports aiter at load",
    exc_type=ImportError,
)
DeepseekV4AttentionMetadataBuilder = attn.DeepseekV4AttentionMetadataBuilder
_chunk_cu_seqlens = attn._chunk_cu_seqlens


def test_chunk_cu_seqlens_cuts_sequences_and_drops_empty_segments():
    cu = np.asarray([0, 3, 8, 8, 12], dtype=np.int32)

    assert _chunk_cu_seqlens(cu, 2, 10).tolist() == [0, 1, 6, 8]


@pytest.mark.parametrize("layout", ["fp4-gfx950-opus", "fp4-gfx1250-natural"])
def test_decode_plan_selects_qlen_variant_and_reuses_its_buffers(monkeypatch, layout):
    opus = pytest.importorskip(
        "aiter.ops.opus.pa_mqa_logits_mxfp4",
        reason="the OPUS FP4 MQA extension is not installed",
        exc_type=ImportError,
    )

    calls = []

    def fake_plan(cu_seq_q, local_ends, **kwargs):
        calls.append((cu_seq_q, local_ends, kwargs))
        return kwargs["buffers"]

    monkeypatch.setattr(opus, "pa_mqa_logits_mxfp4_plan", fake_plan)

    builder = object.__new__(DeepseekV4AttentionMetadataBuilder)
    builder.indexer_layout = layout
    q1_buffers = object()
    q4_buffers = object()
    builder._v4_fp4_opus_plan_buffers = {
        "": {"qlen1_kv64": q1_buffers, "qlen4_kv64": q4_buffers}
    }
    if layout == "fp4-gfx950-opus":
        del builder._v4_fp4_opus_plan_buffers[""]["qlen4_kv64"]
    assert tuple(
        builder._v4_fp4_opus_plan_buffers[""]
    ) == attn._fp4_opus_decode_variants(layout)
    metadata = type(
        "Metadata",
        (),
        {
            "cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32),
            "csa_n_committed_per_token": torch.tensor([7, 8, 999], dtype=torch.int32),
            "batch_id_per_q_token": torch.tensor([0, 0, -1], dtype=torch.int32),
            "max_seqlen_q": 1,
        },
    )()
    positions = torch.empty(2)

    meta = {}
    builder._refresh_fp4_opus_decode_plan(metadata, positions, meta)
    assert meta["fp4_opus_plan"] is q1_buffers
    assert calls[-1][2]["buffers"] is q1_buffers
    assert calls[-1][2]["total_q"] == 2
    assert calls[-1][1].tolist() == [7, 8]
    assert calls[-1][2]["row_to_batch"].tolist() == [0, 0]

    metadata.max_seqlen_q = 4
    meta = {}
    builder._refresh_fp4_opus_decode_plan(metadata, positions, meta)
    expected = q1_buffers if layout == "fp4-gfx950-opus" else q4_buffers
    assert meta["fp4_opus_plan"] is expected
    assert calls[-1][2]["buffers"] is expected
