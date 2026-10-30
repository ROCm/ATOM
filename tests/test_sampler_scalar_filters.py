# SPDX-License-Identifier: MIT
"""CPU-known filters keep AITER scalar dispatch without device readback."""

import pytest
import torch

sampler = pytest.importorskip("atom.model_ops.sampler", exc_type=ImportError)


@pytest.mark.parametrize("top_k,top_p", [(3, None), (None, 0.75), (3, 0.75)])
def test_uniform_verification_filters_preserve_scalars(monkeypatch, top_k, top_p):
    def capture(self, logits, temperatures, top_ks, top_ps, **kwargs):
        assert temperatures.tolist() == [1, 1, 2, 2, 2]
        assert top_ks == top_k and top_ps == top_p
        assert kwargs["needs_independent_noise"]
        return torch.zeros(5, dtype=torch.int32)

    monkeypatch.setattr(sampler.Sampler, "forward", capture)
    sampler.Sampler().sample_verification_tokens(
        torch.zeros(5, 8),
        torch.tensor([0, 2, 2, 5]),
        torch.tensor([0.5, 1.0, 1.5, 2.0]),
        top_k,
        top_p,
    )


@pytest.mark.parametrize("rows,k_rows", [(1, 1), (4, 1), (4, 4)])
def test_kept_ties_take_a_per_row_k_without_readback(monkeypatch, rows, k_rows):
    """A one-element k (a bs=1 batch, or a verify's one k for all its rows)
    reaches the joint op per row, not read back to a scalar first."""
    from types import SimpleNamespace

    seen = {}

    def joint(probs, indices, k_tensor, k_scalar, p_tensor, p_scalar, **_):
        seen.update(k_tensor=k_tensor, k_scalar=k_scalar, p_scalar=p_scalar)
        return torch.zeros(probs.shape[0], 1, dtype=torch.int64)

    monkeypatch.setattr(
        sampler, "aiter_ops", SimpleNamespace(top_k_top_p_sampling_from_probs=joint)
    )
    monkeypatch.setattr(sampler, "_apply_greedy_tokens", lambda p, t, tokens: tokens)
    instance = sampler.Sampler()
    instance.keep_top_k_ties = True
    probs, temperatures = torch.full((rows, 8), 1 / 8), torch.ones(rows)
    k = torch.full((k_rows,), 4, dtype=torch.int32)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("filter readback"))
        patch.setattr(torch, "full", lambda *_, **__: pytest.fail("k allocated"))
        instance._aiter_sample(probs, k, 0.9, True, True, temperatures)
    assert seen["k_tensor"].tolist() == [4] * rows
    assert seen["k_scalar"] == 0 and seen["p_scalar"] == 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("path", ["aiter", "native"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("top_k,top_p", [(3, None), (None, 0.75), (3, 0.75), (-1, 1.0)])
def test_scalar_filters_match_legacy_tensor_sampling(
    monkeypatch, path, rows, top_k, top_p
):
    monkeypatch.setattr(sampler, "AITER_TOPK_TOPP_AVAILABLE", path == "aiter")
    instance = sampler.Sampler()
    logits = torch.randn(rows, 4096, device="cuda")
    temperatures = torch.linspace(0.5, 1.5, rows, device="cuda")
    k = (
        None
        if top_k is None
        else torch.tensor([top_k], dtype=torch.int32, device="cuda")
    )
    p = None if top_p is None else torch.tensor([top_p], device="cuda")
    torch.manual_seed(71)
    expected = instance(logits, temperatures, k, p)
    torch.manual_seed(71)
    # The real sampler must accept CPU filters without a Python tensor readback.
    # Its RNG may use internal C++ scalar operations independently of filters.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("filter readback"))
        actual = instance(logits, temperatures, top_k, top_p)
    assert actual.dtype == torch.int32 and actual.shape == (rows,)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("keep_ties", [False, True])
def test_runner_uniform_filters_reach_sampler_without_readback(
    monkeypatch, transport, bs, keep_ties
):
    from types import SimpleNamespace

    import numpy as np

    from tests.test_h2d_runner_publication import runner_with_buffers

    runner = runner_with_buffers(monkeypatch, transport)
    # Keeping ties hands even a uniform k per row (`prepare_sample`).
    runner.sampler.keep_top_k_ties = keep_ties
    batch = SimpleNamespace(
        total_seqs_num=bs,
        temperatures=np.ones(bs, dtype=np.float32),
        top_ks=np.full(bs, 3, dtype=np.int32),
        top_ps=np.full(bs, 0.75, dtype=np.float32),
    )
    runner._gate_staging_reuse()
    group = runner.h2d_groups.get("token_inputs")
    temperatures, k, p, greedy, noise = runner.prepare_sample(
        batch, publication_group=group
    )
    if group is not None:
        group.publish(group.counts)
    runner._mark_staging_h2d_enqueued()
    logits = torch.full((bs, 4096), -1000.0, device="cuda")
    logits[:, 7] = 0
    instance = sampler.Sampler()
    instance.keep_top_k_ties = keep_ties
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("filter readback"))
        actual = instance(logits, temperatures, k, p, greedy, noise)
    assert actual.tolist() == [7] * bs
