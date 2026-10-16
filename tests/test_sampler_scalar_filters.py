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
def test_runner_uniform_filters_reach_sampler_without_readback(monkeypatch, transport):
    from types import SimpleNamespace

    import numpy as np

    from tests.test_h2d_runner_publication import runner_with_buffers

    runner = runner_with_buffers(monkeypatch, transport)
    batch = SimpleNamespace(
        total_seqs_num=3,
        temperatures=np.ones(3, dtype=np.float32),
        top_ks=np.full(3, 3, dtype=np.int32),
        top_ps=np.full(3, 0.75, dtype=np.float32),
    )
    runner._gate_staging_reuse()
    group = runner.h2d_groups.get("token_inputs")
    temperatures, k, p, greedy, noise = runner.prepare_sample(
        batch, publication_group=group
    )
    if group is not None:
        group.publish(group.counts)
    runner._mark_staging_h2d_enqueued()
    logits = torch.full((3, 4096), -1000.0, device="cuda")
    logits[:, 7] = 0
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("filter readback"))
        actual = sampler.Sampler()(logits, temperatures, k, p, greedy, noise)
    assert actual.tolist() == [7, 7, 7]
