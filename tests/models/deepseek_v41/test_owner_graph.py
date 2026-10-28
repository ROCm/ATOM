"""Owner graph inputs follow live publications while recordings keep addresses."""

import pytest
import torch

from atom.models.deepseek_v41.owner_service.owner_graph import (
    GraphCoverage,
    OwnerGraphInputs,
    graph_key,
)


def test_reordered_publications_update_tokens_and_engram_at_recorded_addresses():
    source = torch.arange(96, dtype=torch.int64)
    embeddings = {1: torch.arange(96 * 8, dtype=torch.float32).view(1, 96, 8)}
    inputs = OwnerGraphInputs(source, embeddings)
    packet = torch.arange(64) % 96
    local = packet[16:32]
    tokens, owned = inputs.stage(source, packet, local, embeddings, 16)
    key = graph_key(16, tokens, 6, {"bands": ((0, 16),)}, owned)
    recorded_token_ptr = tokens.data_ptr()
    recorded_engram_ptr = owned[1].data_ptr()

    # Another native buffer, with another width, request order and values.
    source2 = torch.arange(200, 328, dtype=torch.int64)
    embeddings2 = {1: torch.arange(128 * 8, dtype=torch.float32).view(1, 128, 8) + 1000}
    packet2 = (torch.arange(64).flip(0) * 2) % 128
    local2 = packet2[16:32]
    tokens2, owned2 = inputs.stage(source2, packet2, local2, embeddings2, 16)
    assert source.data_ptr() != source2.data_ptr()
    assert tokens2.data_ptr() == recorded_token_ptr
    assert owned2[1].data_ptr() == recorded_engram_ptr
    assert graph_key(16, tokens2, 6, {"bands": ((0, 16),)}, owned2) == key
    assert torch.equal(tokens, source2[packet2])
    assert torch.equal(owned[1], embeddings2[1][:, local2])


def test_staging_consumes_native_engram_readiness_before_copy():
    class Embeddings(dict):
        def get(self, layer):
            # Models the producer becoming ready at the native get seam.
            value = super().get(layer)
            value.fill_(11)
            return value

    source = torch.arange(64)
    embeddings = Embeddings({14: torch.zeros(1, 64, 8)})
    inputs = OwnerGraphInputs(source, embeddings)
    _, owned = inputs.stage(source, torch.arange(64), torch.arange(16), embeddings, 16)
    assert torch.equal(owned[14], torch.full_like(owned[14], 11))


def test_key_retains_capacity_q_band_and_captured_binding_changes():
    source, embeddings = torch.arange(128), {1: torch.zeros(1, 128, 8)}
    inputs = OwnerGraphInputs(source, embeddings)
    tokens, owned = inputs.views(16)
    key = graph_key(16, tokens, 6, {"bands": ((0, 16),)}, owned)
    assert graph_key(16, tokens, 5, {"bands": ((0, 16),)}, owned) != key
    assert graph_key(16, tokens, 6, {"bands": ((0, 8), (8, 16))}, owned) != key
    assert graph_key(16, tokens.clone(), 6, {"bands": ((0, 16),)}, owned) != key
    assert graph_key(16, tokens, 6, {"bands": ((0, 16),)}, {1: owned[1].clone()}) != key
    wider, wider_owned = inputs.views(32)
    assert graph_key(32, wider, 6, {"bands": ((0, 32),)}, wider_owned) != key


def test_no_stale_layer_or_dtype_contract_is_reused():
    source, embeddings = torch.arange(64), {1: torch.zeros(1, 64, 8)}
    inputs = OwnerGraphInputs(source, embeddings)
    with pytest.raises(ValueError, match="input contract"):
        inputs.stage(source, torch.arange(64), torch.arange(16), {}, 16)
    with pytest.raises(ValueError, match="Engram contract"):
        inputs.stage(
            source, torch.arange(64), torch.arange(16), {1: torch.zeros(1, 64, 9)}, 16
        )
    with pytest.raises(ValueError, match="input contract"):
        inputs.stage(source.int(), torch.arange(64), torch.arange(16), embeddings, 16)


def test_any_rank_can_require_a_collective_capture_in_warmup():
    policies = [GraphCoverage() for _ in range(4)]
    requests = [p.request(i == 2, "warmup") for i, p in enumerate(policies)]
    for policy in policies:
        assert policy.require_capture(requests)
        policy.captured()
    assert all(p.captures == 1 for p in policies)


def test_one_rank_sealing_prevents_every_rank_from_capturing_a_miss():
    policies = [GraphCoverage() for _ in range(4)]
    requests = [
        p.request(i == 2, "sealed" if i == 1 else "warmup")
        for i, p in enumerate(policies)
    ]
    for policy in policies:
        with pytest.raises(RuntimeError, match="Unwarmed owner graph"):
            policy.require_capture(requests)
        assert policy.sealed and policy.misses_after_seal == 1
        assert policy.captures == 0
        with pytest.raises(RuntimeError, match="Cannot record"):
            policy.captured()


def test_sealed_hits_replay_and_cannot_reopen_learning():
    policy = GraphCoverage()
    request = policy.request(False, "sealed")
    assert not policy.require_capture([request] * 4)
    request = policy.request(True, "warmup")
    with pytest.raises(RuntimeError, match="Unwarmed"):
        policy.require_capture([request] * 4)


@pytest.mark.parametrize(
    "requests", [[], [(False, False)] * 3, [False] * 4, [(False, 0)] * 4]
)
def test_incomplete_rank_agreement_is_refused(requests):
    with pytest.raises(ValueError, match="four-rank"):
        GraphCoverage().require_capture(requests)
