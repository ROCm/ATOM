# SPDX-License-Identifier: MIT
"""All-to-all consumers need the same resolved counts as DPMetadata."""

from types import SimpleNamespace

import pytest
import torch

from atom import config as config_module
from atom.utils import forward_context as fc
from atom.utils.tbo.ubatch_splitting import UBatchSlice
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper


@pytest.fixture
def config(monkeypatch):
    value = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=0),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(config_module, "get_current_atom_config", lambda: value)
    monkeypatch.setattr(fc, "_CUDA_AVAILABLE", False)
    monkeypatch.setattr(fc, "_forward_context", fc._forward_context)
    monkeypatch.setattr(fc._forward_context_local, "ctx", None, raising=False)
    return value


def context(tokens=7):
    return fc.Context(
        positions=torch.arange(tokens),
        is_prefill=False,
        scheduled_bs=1,
        running_bs=1,
        scheduled_tokens=tokens,
        running_tokens=tokens,
    )


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_forward_publishes_resolved_counts(config, monkeypatch, supplied, draft):
    counts = torch.tensor([7, 19], dtype=torch.int32)
    calls = []

    def reduce_counts(tokens, size, rank):
        calls.append((tokens, size, rank))
        return counts

    monkeypatch.setattr(
        fc.DPMetadata, "num_tokens_across_dp", staticmethod(reduce_counts)
    )
    ctx = context()
    ctx.is_draft = draft
    fc.set_forward_context(
        None,
        config,
        ctx,
        num_tokens=7,
        num_tokens_across_dp=counts if supplied else None,
    )
    forward = fc.get_forward_context()
    assert forward.context.running_tokens_across_dp == (7, 19)
    assert tuple(forward.dp_metadata.get_sizes_across_dp()) == (7, 19)
    assert calls == ([] if supplied else [(7, 2, 0)])


def test_reused_context_clears_previous_dp_counts(config):
    ctx = context()
    fc.set_forward_context(
        None, config, ctx, num_tokens=7, num_tokens_across_dp=torch.tensor([7, 19])
    )
    config.parallel_config.data_parallel_size = 1
    fc.set_forward_context(None, config, ctx, num_tokens=7)
    assert ctx.running_tokens_across_dp is None
    assert fc.get_forward_context().dp_metadata is None


@pytest.mark.parametrize("precomputed", [False, True])
def test_tbo_children_publish_their_own_resolved_counts(
    config, monkeypatch, precomputed
):
    tables = ((3, 9), (4, 5))
    calls = []

    def reduce_counts(tokens, size, rank):
        calls.append(tokens)
        return torch.tensor(tables[len(calls) - 1], dtype=torch.int32)

    monkeypatch.setattr(
        fc.DPMetadata, "num_tokens_across_dp", staticmethod(reduce_counts)
    )
    parent_dp = fc.DPMetadata.make(config.parallel_config, 7, torch.tensor([7, 14]))
    ctx = context()
    ctx.is_prefill = True
    ctx.running_tokens_are_unified = False
    parent = fc.ForwardContext(
        attn_metadata=SimpleNamespace(),
        no_compile_layers={},
        kv_cache_data={},
        context=ctx,
        dp_metadata=parent_dp,
        ubatch_slices=[
            UBatchSlice(slice(0, 1), slice(0, 3)),
            UBatchSlice(slice(0, 1), slice(3, 7)),
        ],
        ub_tokens_across_dp=tables if precomputed else None,
    )
    builder = SimpleNamespace(
        build_ubatch_prefill_metadata=lambda *args, **kwargs: SimpleNamespace()
    )
    wrapper = UBatchWrapper(torch.nn.Identity(), builder)
    child_metas = wrapper._make_ubatch_dp_metadata(parent, 2)
    for index, part in enumerate(parent.ubatch_slices):
        child = wrapper._make_ubatch_context(
            parent,
            part,
            1,
            index,
            dp_metadata=child_metas[index],
            running_tokens_across_dp=wrapper._ub_tokens_across_dp(parent, 2, index),
        )
        assert child.context.running_tokens_across_dp == tables[index]
        assert child.context.running_tokens_across_dp != (7, 14)
        assert tuple(child.dp_metadata.get_sizes_across_dp()) == tables[index]
    assert calls == ([] if precomputed else [3, 4])
