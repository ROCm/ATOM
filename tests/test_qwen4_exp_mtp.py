# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen MTP history recovery must equal execution of accepted tokens only."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.qwen4_exp.ops.ple import advance_ngram_state, dilated_causal_conv1d
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_draft_decode_metadata,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_decode_mrope_storage_matches_padded_graph_stride(monkeypatch, k):
    import numpy as np

    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    builder = object.__new__(Qwen4ExpMetadataBuilder)
    cpu = torch.full((3, 32), -999, dtype=torch.int64)
    gpu = torch.full_like(cpu, -777)
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(max_model_len=8192),
        forward_vars={
            "mrope_positions": SimpleNamespace(cpu=cpu, np=cpu.numpy(), gpu=gpu)
        },
        _mrope_positions_view=lambda n: gpu.as_strided((3, n), (n, 1)),
    )
    builder._build_qsa_metadata = lambda *args, **kwargs: None
    builder._build_ple_metadata = lambda *args, **kwargs: None
    width = k + 1

    def prepare(self, batch, running_bs, running_tokens, max_seqlen_q):
        real = batch.total_tokens_num_decode
        self._mrope_cpu_view(real)[:] = batch.positions
        return (
            SimpleNamespace(max_seqlen_q=max_seqlen_q),
            self._copy_mrope_to_gpu(real),
        )

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "prepare_decode", prepare)
    # Shrinking requests must not expose the previous batch's axis/tail data.
    for scheduled in (4, 3, 1):
        tokens = scheduled * width
        expected = np.arange(3 * tokens).reshape(3, tokens) + 100
        batch = SimpleNamespace(total_tokens_num_decode=tokens, positions=expected)
        _, positions = builder.prepare_decode(batch, 4, 4 * width, width)
        reference = torch.from_numpy(expected)
        torch.testing.assert_close(positions, reference)
        assert positions.stride(0) == 4 * width
        graph_positions = builder.model_runner._mrope_positions_view(4 * width)
        torch.testing.assert_close(graph_positions[:, :tokens], reference)
        assert not graph_positions[:, tokens:].count_nonzero()


def test_draft_qsa_view_drops_only_graph_padding():
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpQSAMetadata

    token_buffer = torch.arange(16)
    metadata = Qwen4ExpQSAMetadata(
        torch.zeros(4, 2, dtype=torch.int32),
        token_buffer,
        token_buffer.clone(),
        token_buffer.to(torch.int32),
        token_buffer.clone(),
        torch.full((4,), 100, dtype=torch.int32),
        128,
    )
    draft = metadata.for_tokens(12)
    assert metadata.slot_mapping.numel() == 16
    for name in (
        "slot_mapping",
        "compressed_slot_mapping",
        "token_to_req",
        "logical_positions",
    ):
        assert getattr(draft, name).numel() == 12
        assert getattr(draft, name).data_ptr() == getattr(metadata, name).data_ptr()
    assert draft.block_tables is metadata.block_tables
    assert draft.seq_lens is metadata.seq_lens


def test_draft_allocation_profile_does_not_require_qsa_pools():
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    slots = torch.zeros(16, dtype=torch.int64)
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        forward_vars={"slot_mapping": SimpleNamespace(gpu=slots)}
    )
    for positions in (torch.zeros(4), torch.zeros(3, 4)):
        metadata = builder.prepare_mtp_decode(4, 1, 16, positions)
        assert metadata["qsa_metadata"] is None
        assert metadata["slot_mapping"].numel() == 4
        assert metadata["slot_mapping"].data_ptr() == slots.data_ptr()


def test_qwen_replay_uses_shared_record_dtype_and_preserves_state_dtype(monkeypatch):
    from atom.model_ops.attentions.gdn_attn import STATE_SLOT_CLASS
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        hf_config=SimpleNamespace(mamba_ssm_dtype="float32"),
    )
    qwen = object.__new__(Qwen4ExpMetadataBuilder)
    qwen.model_runner = SimpleNamespace(config=config)
    qwen._state_shape_for_runner = lambda: ((3, 32), (2, 4, 4))
    qwen._replayssm_buffer_shapes = lambda: ((2, 16, 4), (2, 16, 4), (2, 16))
    qwen.num_state_layers = lambda: 2

    # Exercise the real allocator and byte accounting without requiring a GPU.
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        assert kwargs.pop("device") == "cuda"
        return zeros(*args, device="cpu", **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    slots = 3
    caches = qwen.allocate_per_req_cache({STATE_SLOT_CLASS: slots})
    assert caches["mamba_k_cache"].dtype == torch.bfloat16
    assert caches["mamba_v_cache"].dtype == torch.float32
    assert caches["replayssm_buf_k"].dtype == torch.bfloat16
    assert caches["replayssm_buf_u"].dtype == torch.bfloat16
    assert caches["replayssm_buf_g"].dtype == torch.float32
    allocated = sum(
        caches[name].numel() * caches[name].element_size()
        for name in ("replayssm_buf_k", "replayssm_buf_u", "replayssm_buf_g")
    )
    assert qwen._replayssm_bytes_per_slot() * slots == allocated


def conv_reference(x, past, weight, dilation):
    combined = torch.cat((past.T, x))
    history = past.shape[1]
    result = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(weight.shape[1]):
        start = history - (weight.shape[1] - 1 - tap) * dilation
        result += combined[start : start + len(x)].float() * weight[:, tap].float()
    result = result.to(x.dtype).float()
    return (result * torch.sigmoid(result)).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_mtp_shared_projection_uses_full_width_hidden_norm():
    from atom.model_ops.qwen4_exp.hyperconnection import Qwen4ExpGroupedRMSNorm
    from atom.models.qwen4_exp_mtp import Qwen4ExpMultiTokenPredictor

    class IdentityDecoder(torch.nn.Module):
        def forward(self, positions, hidden, input_ids):
            return hidden

    torch.manual_seed(17)
    n, hc, width, eps = 5, 4, 64, 1e-6
    model = Qwen4ExpMultiTokenPredictor.__new__(Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count, model.hidden_size = hc, width
    model.pre_fc_norm_hidden = Qwen4ExpGroupedRMSNorm(hc * width, hc * width, eps)
    model.pre_fc_norm_embedding = Qwen4ExpGroupedRMSNorm(width, width, eps)
    model.fc_hidden = torch.nn.Linear(width, width, bias=False)
    model.fc_embedding = torch.nn.Linear(width, width, bias=False)
    model.layers = torch.nn.ModuleList([IdentityDecoder()])
    model = model.to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(n, hc, width, device="cuda", dtype=torch.bfloat16)
    hidden *= torch.tensor([0.1, 1, 2, 5], device="cuda")[None, :, None]
    embedding = torch.randn(n, width, device="cuda", dtype=torch.bfloat16)

    def norm(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)).to(
            x.dtype
        )

    with torch.inference_mode():
        expected = model.fc_hidden(norm(hidden.flatten(1)).view(n, hc, width))
        expected += model.fc_embedding(norm(embedding))[:, None, :]
        actual = model(None, torch.arange(n, device="cuda"), hidden, embedding)
    assert actual.shape == (n, hc, width)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 3, 4])
@pytest.mark.parametrize("width", [2, 3])
def test_ple_recovers_every_accepted_prefix(k, dilation, width):
    torch.manual_seed(42)
    device, channels, taps, eos = "cuda", 37, 4, 99
    history = (taps - 1) * dilation
    weight = torch.randn(channels, taps, device=device, dtype=torch.bfloat16) * 0.1
    conv_state = torch.randn(
        7, channels, history + k, device=device, dtype=torch.bfloat16
    )
    token_state = torch.full((7, width + k), -77, device=device, dtype=torch.int64)
    slots = torch.tensor([[4, 6], [1, 5], [-1, 3]], device=device, dtype=torch.int32)[
        :, 0
    ]
    has = torch.tensor([False, False, False], device=device)
    untouched_conv, untouched_tokens = conv_state[0].clone(), token_state[0].clone()
    past = [
        torch.zeros(channels, history, device=device, dtype=torch.bfloat16)
        for _ in range(2)
    ]
    past_ids = [
        torch.full((width,), eos, device=device, dtype=torch.int64) for _ in range(2)
    ]
    accepted = torch.ones(3, device=device, dtype=torch.int32)
    # Start with ordinary prefill; follow with repeated partial/full rejection.
    for iteration in range(k + 4):
        lengths = [11, 4, 0] if iteration == 0 else [k + 1, k + 1, k + 1]
        starts = torch.tensor(
            [0, lengths[0], sum(lengths[:2]), sum(lengths)],
            device=device,
            dtype=torch.int32,
        )
        x = torch.randn(sum(lengths), channels, device=device, dtype=torch.bfloat16)
        ids = torch.randint(0, 120, (sum(lengths),), device=device, dtype=torch.int64)
        spec = accepted if iteration else None
        context = advance_ngram_state(
            ids,
            starts,
            token_state,
            slots,
            slots,
            has,
            eos,
            history_width=width,
            num_accepted_tokens=spec,
        )
        actual = dilated_causal_conv1d(
            x,
            weight,
            conv_state,
            starts,
            slots,
            slots,
            has,
            dilation,
            spec,
        )
        for req in range(2):
            start, end = int(starts[req]), int(starts[req + 1])
            expected = conv_reference(x[start:end], past[req], weight, dilation)
            torch.testing.assert_close(
                actual[start:end], expected, rtol=0.01, atol=0.001
            )
            torch.testing.assert_close(context[req], past_ids[req])
            count = lengths[req] if not iteration else 1 + (iteration + req) % (k + 1)
            past[req] = torch.cat((past[req], x[start : start + count].T), dim=1)[
                :, -history:
            ]
            past_ids[req] = torch.cat((past_ids[req], ids[start : start + count]))[
                -width:
            ]
            accepted[req] = count if iteration else 1
        if iteration:
            assert torch.count_nonzero(actual[sum(lengths[:2]) :]) == 0
        has[:2] = True
        torch.testing.assert_close(conv_state[0], untouched_conv)
        torch.testing.assert_close(token_state[0], untouched_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("k", [1, 2, 3])
def test_qsa_draft_slots_follow_accepted_tail_and_page_boundary(k):
    device, block, ratio = "cuda", 16, 4
    tables = torch.tensor(
        [[7, 3, 9], [2, 8, 6], [1, 4, 5]], device=device, dtype=torch.int32
    )
    lengths = torch.tensor([17 + k, 32 + k, 42], device=device, dtype=torch.int32)
    rejects = torch.tensor([k, k - 1], device=device, dtype=torch.int32)
    slots = torch.empty(3, device=device, dtype=torch.int64)
    logical, compressed = torch.empty_like(slots), torch.empty_like(slots)
    reqs = torch.empty(3, device=device, dtype=torch.int32)
    qsa_draft_decode_metadata(
        lengths, tables, rejects, slots, logical, reqs, compressed, 2, block, ratio
    )
    assert lengths.tolist() == [17, 33, 0]
    assert logical.tolist() == [16, 32, -1]
    assert slots.tolist() == [3 * block, 6 * block, -1]
    assert reqs.tolist() == [0, 1, -1]
    assert compressed.tolist() == [-1, -1, -1]
    lengths[:2] += 3
    qsa_draft_decode_metadata(
        lengths, tables, None, slots, logical, reqs, compressed, 2, block, ratio
    )
    assert compressed.tolist() == [
        (3 * block + 3) // ratio,
        (6 * block + 3) // ratio,
        -1,
    ]
    lengths.zero_()
    qsa_draft_decode_metadata(
        lengths, tables, None, slots, logical, reqs, compressed, 2, block, ratio
    )
    assert slots.tolist() == [-1, -1, -1]
    assert compressed.tolist() == [-1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("start", [15, 2047, 2063])
def test_qsa_verification_is_causal_and_matches_individual_queries(k, start):
    """Future draft keys must not enter an earlier query's selected context."""
    torch.manual_seed(93)
    device, block, ratio, budget = "cuda", 16, 4, 2048
    rows, pages = k + 1, 132
    positions = torch.arange(start, start + rows, device=device)
    # A permuted physical layout exercises page translation, not only masking.
    tables = torch.randperm(pages, device=device).to(torch.int32)[None]
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    lengths = torch.tensor([start + rows], dtype=torch.int32, device=device)
    index_q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    compressed = torch.randn(
        pages, block // ratio, 1, 128, dtype=torch.bfloat16, device=device
    )
    selected = qsa_select_paged_tokens(
        index_q, compressed, tables, requests, positions, lengths, budget, ratio
    )
    query = torch.randn(rows, 24, 256, dtype=torch.bfloat16, device=device)
    keys = torch.randn(pages, block, 2, 256, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    batched = qsa_sparse_paged_gqa(
        query, keys, values, selected, tables, requests, num_decode_requests=1
    )
    for row in range(rows):
        visible = selected[row][selected[row] >= 0]
        assert (
            visible.numel()
            == min((start + row + 1) // ratio * ratio, budget)
            + (start + row + 1) % ratio
        )
        assert visible.unique().numel() == visible.numel()
        assert int(visible.max()) <= start + row
        individual = qsa_select_paged_tokens(
            index_q[row : row + 1],
            compressed,
            tables,
            requests[row : row + 1],
            positions[row : row + 1],
            lengths.new_tensor([start + row + 1]),
            budget,
            ratio,
        )
        torch.testing.assert_close(individual, selected[row : row + 1])
        reference = qsa_sparse_paged_gqa(
            query[row : row + 1],
            keys,
            values,
            individual,
            tables,
            requests[row : row + 1],
            num_decode_requests=1,
        )
        torch.testing.assert_close(batched[row : row + 1], reference, rtol=0, atol=0)
