# SPDX-License-Identifier: MIT
"""Draft loading and target-feature contracts, without full model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
from torch import nn

from atom.config import SpeculativeConfig, get_hf_config
from atom.spec_decode.drafter import AuxCaptureSpec
from atom.spec_decode.dspark_proposer import DSparkProposer

from .reference import FIXTURES


def test_native_draft_config_preserves_v41_architecture():
    config = get_hf_config(str(FIXTURES))
    SpeculativeConfig.hf_config_override(config, model_path=None)
    assert config.architectures == ["DeepseekV41DSparkModel"]
    assert config.model_type == "deepseek_v41_dspark"
    assert config.num_nextn_predict_layers == 3
    assert config.head_dim == 512 and config.qk_rope_head_dim == 64
    assert config.dspark_n_routed_experts == 128
    assert config.n_routed_experts == 384


def test_v41_draft_declines_capture_of_live_request_metadata():
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(proposer.model)
    (block,) = proposer._declare_draft_graphs()
    assert not block.capture_supported


@pytest.mark.parametrize("width", [1, 6])
def test_capture_builder_uses_full_query_width_and_private_state(width):
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(
        1, ((0, 2),), 16, 128, 128, 32, speculative_tokens=5
    )
    builder.cache = PagedAttentionCache(builder.geometry, 4, 2, "cpu")
    builder.cache.backing.fill_(17)
    before = builder.cache.backing.clone()
    builder.block_size, builder.device = 16, "cpu"
    builder.max_num_batched_tokens = 12
    builder.model_runner = SimpleNamespace(
        forward_vars={
            "positions": SimpleNamespace(gpu=torch.empty(12, dtype=torch.int64)),
            "cu_seqlens_q": SimpleNamespace(gpu=torch.empty(3, dtype=torch.int32)),
            "input_ids": SimpleNamespace(gpu=torch.zeros(12, dtype=torch.int32)),
        }
    )
    prepared = []
    builder.prepare_model_inputs = lambda tokens, metadata: prepared.append(
        tokens.numel()
    )
    metadata, context = builder.build_for_cudagraph_capture(2, width)
    assert prepared == [2 * width]
    assert context.running_tokens == context.scheduled_tokens == 2 * width
    assert context.is_dummy_run
    assert metadata.step.length == 2 * width
    assert metadata.step.positions.tolist() == list(range(width)) * 2
    assert metadata.cu_seqlens_q.tolist() == [0, width, 2 * width]
    assert metadata.cache is not builder.cache
    assert metadata.cache.backing.data_ptr() != builder.cache.backing.data_ptr()
    assert torch.equal(builder.cache.backing, before)
    with pytest.raises(ValueError, match="capture shape"):
        builder.build_for_cudagraph_capture(3, 6)


class ShiftBlock(nn.Module):
    def forward(self, state):
        return SinglePassHCState(state.residual + 100, state.pre_mix)


def capture_fixture(monkeypatch):
    draft = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(engram_layer_ids=())
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = draft
    proposer.speculative_config = SimpleNamespace(
        draft_model_hf_config=SimpleNamespace(dspark_target_layer_ids=(0,))
    )
    proposer.config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=8))
    proposer.max_num_tokens, proposer.device, proposer.dtype = 9, "cpu", torch.float32
    target = nn.Module()
    target.layers = nn.ModuleList([ShiftBlock()])
    context = SimpleNamespace(is_draft=False, ubatch_token_offset=2)
    monkeypatch.setattr(
        "atom.spec_decode.drafter.get_forward_context",
        lambda: SimpleNamespace(context=context),
    )
    return proposer, target, context


def test_input_capture_uses_stream_mean_and_respects_offset_and_draft(monkeypatch):
    proposer, target, context = capture_fixture(monkeypatch)
    proposer.arm_aux_capture(target)
    residual = torch.arange(1 * 3 * 4 * 8).reshape(1, 3, 4, 8).float()
    pre = torch.zeros(1, 3, 4)
    pre[..., 0] = 1
    state = SinglePassHCState(residual, pre)
    output = target.layers[0](state)
    captured = proposer.aux_for(torch.empty(9, 8))[0]
    assert torch.equal(captured[2:5], residual.mean(-2).squeeze(0))
    assert not torch.equal(captured[2:5], state.collapse().squeeze(0))
    assert not torch.equal(captured[2:5], output.residual.mean(-2).squeeze(0))
    assert captured[:2].count_nonzero() == captured[5:].count_nonzero() == 0
    saved = captured.clone()
    context.is_draft = True
    target.layers[0](output)
    assert torch.equal(captured, saved)


def test_output_capture_remains_default(monkeypatch):
    proposer, target, _ = capture_fixture(monkeypatch)
    proposer._aux_capture_spec = lambda _: AuxCaptureSpec(
        (0,), 8, lambda output, block: output.residual.mean(-2).squeeze(0)
    )
    proposer.arm_aux_capture(target)
    output = target.layers[0](
        SinglePassHCState.from_embeddings(torch.zeros(1, 3, 8), 4)
    )
    assert torch.equal(
        proposer.aux_for(torch.empty(9, 8))[0][2:5], output.residual.mean(-2).squeeze(0)
    )


def test_input_tap_cannot_silently_skip_engram(monkeypatch):
    proposer, target, _ = capture_fixture(monkeypatch)
    proposer.model.config.engram_layer_ids = (0,)
    with pytest.raises(ValueError, match="Engram injection"):
        proposer.arm_aux_capture(target)


def test_decode_positions_use_accepted_prefix_and_full_reservation():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(
        1, ((0, 2),), 16, 128, 128, 32, speculative_tokens=5
    )
    builder.cache = PagedAttentionCache(builder.geometry, 20, 5, "cpu")
    builder.block_size, builder.device = 16, "cpu"
    positions = torch.empty(4, dtype=torch.int32)
    builder.model_runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(num_rejected=np.array([0, 4])),
        forward_vars={
            "positions": SimpleNamespace(gpu=positions),
            "cu_seqlens_q": SimpleNamespace(gpu=torch.tensor([0, 1, 4])),
        },
    )
    batch = SimpleNamespace(
        is_dummy_run=False,
        req_ids=(11, 22),
        total_seqs_num=2,
        total_tokens_num=4,
        state_slots_committed=(4, 1),
        num_spec_step=5,
        context_lens=np.array([135, 150]),
        num_scheduled_tokens=(1, 3),
        block_tables=(tuple(range(10)), tuple(range(10, 20))),
    )
    metadata, actual = builder.prepare_decode(batch, 2, 4, 3)
    assert [span.position for span in metadata.step.requests] == [129, 140]
    assert actual.tolist() == [129, 140, 141, 142]
    assert metadata.step.tentative
