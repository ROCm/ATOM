# SPDX-License-Identifier: MIT
"""Default DSpark preparation and mixed attention buffer ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter")


@pytest.mark.parametrize(
    "width,explicit,expected",
    [(1, False, [20, 24]), (4, False, [19, 22]), (1, True, [19, 22])],
)
def test_decode_rollback_and_positions(width, explicit, expected):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    class Prepared(Exception):
        pass

    def stop_after_positions(*args):
        raise Prepared

    variables = {
        "positions": SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
        "context_lens": SimpleNamespace(np=np.zeros(2, dtype=np.int32)),
    }
    lens = np.full(2, width, dtype=np.int32)
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(
            forward_vars=variables,
            arange_np=np.arange(8, dtype=np.int32),
            tokenID_processor=SimpleNamespace(num_rejected=np.array([1, 2])),
        ),
        decode_spans=lambda batch: (2, lens, np.array([0, width, 2 * width])),
        prepare_block_tables=stop_after_positions,
    )
    batch = SimpleNamespace(
        context_lens=[20, 24],
        is_dummy_run=False,
        num_spec_step=3,
        total_tokens_num_decode=2 * width,
    )
    counts = np.array([1, 2]) if explicit else None
    with pytest.raises(Prepared):
        DeepseekV4AttentionMetadataBuilder.prepare_decode(
            builder, batch, 2, 2 * width, width, num_rejected=counts
        )
    assert variables["context_lens"].np.tolist() == expected
    positions = np.concatenate(
        [np.arange(ctx - 4, ctx - 4 + width) for ctx in expected]
    )
    np.testing.assert_array_equal(variables["positions"].np[: 2 * width], positions)
    with pytest.raises(ValueError, match="aligned"):
        DeepseekV4AttentionMetadataBuilder.prepare_decode(
            builder, batch, 2, 2 * width, width, num_rejected=np.array([1, 2, 3])
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_default_first_decode_keeps_ragged_shrink(mixed):
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        config=SimpleNamespace(
            enable_mixed_prefill_decode=mixed,
            dspark=SimpleNamespace(ragged_graph_sizes="4"),
        ),
        tokenID_processor=SimpleNamespace(
            prev_batch=SimpleNamespace(
                req_ids=[3, 4],
                total_seqs_num_prefill=2,
                is_dummy_run=False,
            )
        ),
        enforce_eager=True,
    )
    batch = SimpleNamespace(
        req_ids=[3, 4],
        total_seqs_num_prefill=0,
        is_mixed=False,
        num_scheduled_tokens=np.array([4, 4]),
        num_bonus=np.array([0, 1]),
        scheduled_tokens=np.arange(8),
        total_tokens_num_decode=8,
        total_tokens_num_prefill=0,
        total_tokens_num=8,
    )
    ModelRunner._dspark_apply_ragged(runner, batch, 2, 4, {3: 0, 4: 2})
    assert batch.num_scheduled_tokens.tolist() == ([4, 4] if mixed else [1, 3])


@pytest.mark.parametrize("bf16", [True, False])
def test_attention_output_reuse_keeps_decode_input_intact(monkeypatch, bf16):
    import atom.models.deepseek_v4 as model
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut

    original = torch.arange(24, dtype=torch.bfloat16).reshape(6, 2, 2)
    query = original.clone()
    qkn = QKNormRopeOut(q_sa=query) if bf16 else QKNormRopeOut(q_packed=query)
    metadata = SimpleNamespace(
        num_prefill_tokens=4,
        num_decode_tokens=2,
        prefill_attn_metadata=object(),
        decode_attn_metadata=object(),
    )
    ctx = SimpleNamespace(is_prefill=True, input_ids=torch.arange(6))
    fc = SimpleNamespace(context=ctx, attn_metadata=metadata)
    monkeypatch.setattr(model, "get_forward_context", lambda: fc)
    calls = []

    def attend(segment, positions, *, out, **kwargs):
        sl = slice(0, 4) if ctx.is_prefill else slice(4, 6)
        incoming = segment.q_sa if bf16 else segment.q_packed
        torch.testing.assert_close(incoming, original[sl])
        calls.append(ctx.is_prefill)
        out.copy_(incoming + 10)

    attention = SimpleNamespace(
        n_local_heads=2,
        head_dim=2,
        _qk_norm_rope=lambda *args: qkn,
        _launch_compressors=lambda x: None,
        _sparse_attention=attend,
    )
    out = model.DeepseekV4Attention._attn_mixed(
        attention, metadata, query, query, query, None, query, torch.arange(6)
    )
    assert calls == [True, False]
    assert (out.data_ptr() == query.data_ptr()) == bf16
    torch.testing.assert_close(out, (original + 10).reshape(6, 4))
    assert fc.attn_metadata is metadata and ctx.is_prefill


@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_aux_capture_uses_live_offsets(monkeypatch, compiled):
    from atom.spec_decode.drafter import Drafter
    from atom.utils import forward_context as fc

    context = SimpleNamespace(is_draft=False, ubatch_token_offset=0)
    monkeypatch.setattr(
        fc._forward_context_local,
        "ctx",
        SimpleNamespace(context=context),
        raising=False,
    )
    buffer = torch.full((6, 2), -1.0, device="cuda")
    drafter = SimpleNamespace(
        config=SimpleNamespace(enable_mixed_prefill_decode=True),
        _aux_buffers=[buffer],
    )
    hook = Drafter._make_aux_hook(drafter, 0, lambda output, module: output)

    def run(x):
        hook(None, (), x)
        return x + 1

    if compiled:
        run = torch.compile(run, fullgraph=True, backend="eager")
    run(torch.ones(2, 2, device="cuda"))
    context.ubatch_token_offset = 3
    run(torch.full((2, 2), 2.0, device="cuda"))
    context.is_draft = True
    run(torch.full((2, 2), 99.0, device="cuda"))
    assert buffer[:, 0].tolist() == [1, 1, -1, 2, 2, -1]
