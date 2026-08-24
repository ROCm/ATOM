# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for sampling helpers."""

import sys
import types
from unittest.mock import MagicMock

import torch


def _import_sampler_module():
    try:
        import atom.model_ops.sampler as sampler_module

        return sampler_module
    except ImportError:
        # Either AITER is missing outright, or — as on the non-GPU CI runner —
        # an empty ``aiter`` namespace package shadows it and the kernel symbols
        # cannot be bound. Both land here.
        pass

    # ``sampler`` requires AITER in production. Stub the modules it imports;
    # these tests exercise the pure PyTorch masking helper. Every stub is
    # restored afterwards so the fake does not leak into later test modules.
    stub_names = (
        "aiter",
        "aiter.ops",
        "aiter.ops.triton",
        "aiter.ops.triton.softmax",
        "aiter.ops.triton.topk",
        "atom.model_ops.sampler",
    )
    saved = {name: sys.modules.get(name) for name in stub_names}
    for name in stub_names[:-1]:
        module = types.ModuleType(name)
        module.__path__ = []
        # Any symbol sampler.py binds from AITER resolves to a mock.
        module.__getattr__ = lambda _attr: MagicMock()
        sys.modules[name] = module
    sys.modules["aiter.ops.triton.softmax"].softmax = torch.softmax
    sys.modules["aiter.ops.triton.topk"].topk = torch.topk
    sys.modules.pop("atom.model_ops.sampler", None)
    try:
        import atom.model_ops.sampler as sampler_module

        return sampler_module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_sampler = _import_sampler_module()
apply_min_tokens_mask = _sampler.apply_min_tokens_mask
apply_min_tokens_mask_with_spec_decode = _sampler.apply_min_tokens_mask_with_spec_decode


def test_min_tokens_masks_first_token_terminal_logits():
    logits = torch.arange(6, dtype=torch.float32).unsqueeze(0)

    result = apply_min_tokens_mask(
        logits,
        min_tokens=[1],
        num_completion_tokens=[0],
        eos_token_id=1,
        stop_token_ids=[2],
        single_token_stops=[{3}],
    )

    assert torch.isneginf(result[0, 1])
    assert torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])
    torch.testing.assert_close(result[0, [0, 4, 5]], torch.tensor([0.0, 4.0, 5.0]))


def test_min_tokens_does_not_mask_at_threshold():
    logits = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    original = logits.clone()

    result = apply_min_tokens_mask(
        logits,
        min_tokens=[1],
        num_completion_tokens=[1],
        eos_token_id=1,
        stop_token_ids=[2],
        single_token_stops=[{3}],
    )

    torch.testing.assert_close(result, original)


def test_spec_decode_masks_only_the_draft_rows_below_the_floor():
    """A floor of 2 with one token already emitted reaches one draft row.

    The second and third draft positions are only reachable after the first
    one has been accepted, which puts the request at its floor.
    """
    logits = torch.zeros(3, 6)

    result = apply_min_tokens_mask_with_spec_decode(
        logits,
        min_tokens=[2],
        num_completion_tokens=[1],
        num_draft_tokens=[3],
        eos_token_id=1,
        stop_token_ids=[2],
        single_token_stops=[{3}],
    )

    assert torch.isneginf(result[0, [1, 2, 3]]).all()
    assert torch.isfinite(result[1]).all()
    assert torch.isfinite(result[2]).all()


def test_spec_decode_ragged_batch_maps_each_row_to_its_own_request():
    """Uneven draft counts: the row->request map is what this pins down.

    req0 owns rows 0-1 and needs two more tokens; req1 owns row 2 and is at
    its floor; req2 owns rows 3-5 and needs one more. So rows 0, 1 and 3 are
    masked, each with *its own* single-token stop and nobody else's.
    """
    logits = torch.zeros(6, 8)

    result = apply_min_tokens_mask_with_spec_decode(
        logits,
        min_tokens=[5, 1, 4],
        num_completion_tokens=[3, 1, 3],
        num_draft_tokens=[2, 1, 3],
        eos_token_id=1,
        stop_token_ids=(),
        single_token_stops=[{4}, {5}, {6}],
    )

    # req0's two rows: EOS plus req0's stop, and not req1's or req2's.
    for row in (0, 1):
        assert torch.isneginf(result[row, 1])
        assert torch.isneginf(result[row, 4])
        assert torch.isfinite(result[row, 5])
        assert torch.isfinite(result[row, 6])
    # req1 is at its floor.
    assert torch.isfinite(result[2]).all()
    # req2's first row only, carrying req2's stop rather than req0's.
    assert torch.isneginf(result[3, 1])
    assert torch.isneginf(result[3, 6])
    assert torch.isfinite(result[3, 4])
    assert torch.isfinite(result[4]).all()
    assert torch.isfinite(result[5]).all()


def test_spec_decode_is_a_no_op_once_every_request_has_met_its_floor():
    logits = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    original = logits.clone()

    result = apply_min_tokens_mask_with_spec_decode(
        logits,
        min_tokens=[0, 2],
        num_completion_tokens=[5, 2],
        num_draft_tokens=[2, 2],
        eos_token_id=1,
    )

    torch.testing.assert_close(result, original)


def test_spec_decode_reads_only_the_decode_prefix_of_the_batch_arrays():
    """The batch carries sampling arrays for every scheduled seq; the spec
    metadata covers only the decode prefix of that same ordering. A trailing
    prefill seq must not shift the row math."""
    logits = torch.zeros(2, 5)

    result = apply_min_tokens_mask_with_spec_decode(
        logits,
        min_tokens=[3, 9],
        num_completion_tokens=[0, 0],
        num_draft_tokens=[2],
        eos_token_id=1,
    )

    assert torch.isneginf(result[:, 1]).all()
    assert torch.isfinite(result[:, [0, 2, 3, 4]]).all()
