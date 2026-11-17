# SPDX-License-Identifier: MIT
"""Context-only producers load draft weights and KV state without token heads."""

from types import SimpleNamespace

import pytest

drafter_module = pytest.importorskip("atom.spec_decode.drafter", exc_type=ImportError)

from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.spec_decode.eagle_proposer import EagleProposer


@pytest.mark.parametrize("pp_size", [1, 4])
@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("drafter_class", [DSparkProposer, EagleProposer])
def test_weight_sharing_follows_propose_but_keeps_kv_initialization(
    monkeypatch, pp_size, producer, drafter_class
):
    """PP alone must not skip sharing: Kimi-K3 can propose under PP, and a
    serial producer still uses its first draft step to write context K/V."""
    drafter = object.__new__(drafter_class)
    drafter.runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(is_pd_producer=producer)
    )
    drafter.config = SimpleNamespace(
        model="target-checkpoint",
        load_dummy=False,
        max_num_seqs=3,
        pipeline_parallel_size=pp_size,
    )
    block = drafter_class is DSparkProposer
    drafter.speculative_config = SimpleNamespace(
        method="dspark" if block else "mtp",
        model="draft-checkpoint",
        draft_model_hf_config=object(),
        use_dspark_with_draft=lambda: block,
    )
    drafter.device, drafter.dtype = "cpu", "bf16"
    monkeypatch.setattr(
        drafter_module,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=pp_size, is_last_rank=True),
    )
    events = []
    loaded = {"context_projection.weight"}
    target = object()

    def load_weights(model, checkpoint, hf_config, dummy, is_mtp):
        assert model is drafter.model
        assert checkpoint == ("draft-checkpoint" if block else "target-checkpoint")
        assert hf_config is drafter.speculative_config.draft_model_hf_config
        assert dummy is False
        assert is_mtp is not block
        events.append("load")
        return loaded

    def share(target_base, loaded_weights):
        assert target_base is target
        assert loaded_weights is loaded
        events.append("share")

    def reset_kv(max_num_seqs, device, dtype):
        assert (max_num_seqs, device, dtype) == (3, "cpu", "bf16")
        events.append("reset_kv")

    drafter.model = SimpleNamespace(share_with_target=share, reset_kv_cache=reset_kv)
    monkeypatch.setattr(drafter_module, "load_model", load_weights)

    # The multimodal wrapper is unwrapped before sharing.
    drafter.load_model(SimpleNamespace(language_model=target))

    context_only = producer and block
    assert drafter.context_kv_only is context_only
    expected = ["load"]
    if not context_only:
        expected.append("share")
    if block:
        expected.append("reset_kv")
    assert events == expected


@pytest.mark.parametrize("producer", [False, True])
def test_minimax_sharing_only_requires_embedding_when_proposing(monkeypatch, producer):
    from torch import nn

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark
    from atom.models.utils import PPMissingLayer

    draft_model = MiniMaxM3DSpark.__new__(MiniMaxM3DSpark)
    nn.Module.__init__(draft_model)
    draft_model.vocab_size = 8
    draft_model.embed_tokens = draft_model.lm_head = None
    embed = PPMissingLayer() if producer else nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    head.num_embeddings = 8  # ParallelLMHead's logical vocabulary size.
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)

    drafter = object.__new__(DSparkProposer)
    drafter.runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(is_pd_producer=producer)
    )
    drafter.config = SimpleNamespace(model="target-checkpoint", load_dummy=False)
    drafter.speculative_config = SimpleNamespace(
        method="dspark",
        model="draft-checkpoint",
        draft_model_hf_config=object(),
        use_dspark_with_draft=lambda: True,
    )
    drafter.model = draft_model
    monkeypatch.setattr(drafter_module, "load_model", lambda *args: set())

    drafter.load_model(target)

    assert draft_model.embed_tokens is (None if producer else embed)
    assert draft_model.lm_head is (None if producer else head)
