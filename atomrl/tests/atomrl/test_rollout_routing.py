# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""R3 packing alignment and replay semantics.

The alignment here is the one thing in R3 that fails silently: feed a token the
wrong row and training still runs, it just optimizes against experts that were
never used. CPU-only. Run: python -m atomrl.tests.test_rollout_routing
"""

from types import SimpleNamespace

import numpy as np
import torch
from atomrl.core.protocol import DataProto
from atomrl.engine.training.megatron_engine import MegatronEngine
from atomrl.moe.rollout_routing import (
    RoutingReplayContext,
    apply_replay,
    layer_index_of,
    pack_rollout_routing,
)
from atomrl.trainer.rl_trainer import RLTrainer

N_LAYERS, TOP_K = 3, 2


def _rows(lens):
    """Row i gets routing whose every entry encodes (row, position)."""
    out = []
    for i, sl in enumerate(lens):
        a = np.zeros((sl - 1, N_LAYERS, TOP_K), dtype=np.uint8)
        for t in range(sl - 1):
            a[t, :, :] = i * 10 + t
        out.append(a)
    return out


def test_packed_layout_matches_pack_sequences_order():
    """pack_sequences concatenates each row's real tokens in row order."""
    lens = [4, 2, 5]
    idx, valid = pack_rollout_routing(
        _rows(lens), torch.tensor(lens), N_LAYERS, TOP_K, "cpu"
    )
    assert idx.shape == (sum(lens), N_LAYERS, TOP_K)

    off = 0
    for i, sl in enumerate(lens):
        for t in range(sl - 1):
            assert idx[off + t, 0, 0].item() == i * 10 + t, (i, t)
            assert bool(valid[off + t])
        # the engine never ran a forward at the final position, so no routing
        assert not bool(valid[off + sl - 1])
        off += sl
    assert int(valid.sum()) == sum(lens) - len(lens)


def test_missing_and_short_rows_are_tolerated():
    lens = [3, 3]
    rows = _rows(lens)
    rows[0] = None  # engine returned nothing for this row
    rows[1] = rows[1][:1]  # shorter than L-1
    _idx, valid = pack_rollout_routing(rows, torch.tensor(lens), N_LAYERS, TOP_K, "cpu")
    assert not valid[0:3].any()
    assert bool(valid[3]) and not bool(valid[4]) and not bool(valid[5])


def test_shape_mismatch_is_loud():
    bad = [np.zeros((2, N_LAYERS + 1, TOP_K), dtype=np.uint8)]
    try:
        pack_rollout_routing(bad, torch.tensor([3]), N_LAYERS, TOP_K, "cpu")
    except RuntimeError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("wrong-shaped routing was accepted")


def test_megatron_accepts_ragged_routing_contract():
    routes = _rows([3, 4])
    batch = DataProto(
        tensors={"input_ids": torch.zeros(2, 4, dtype=torch.long)},
        ragged={"rollout_routing": routes},
    )
    engine = object.__new__(MegatronEngine)
    engine._r3_enabled = True
    assert engine._r3_routes(batch) is routes


def test_ray_logprob_handoff_keeps_routing_ragged():
    class FakeWorkerGroup:
        request = None

        def dispatch_and_call(self, _method, data, **_kwargs):
            self.request = data
            bsz, seq_len = data["input_ids"].shape
            return DataProto(
                tensors={
                    "log_probs": torch.zeros(bsz, seq_len - 1),
                    "input_ids": data["input_ids"],
                }
            )

    trainer = object.__new__(RLTrainer)
    trainer._device = torch.device("cpu")
    role_cfg = SimpleNamespace(dispatch_mode="dp_compute_proto", lazy_dispatch_key=None)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            generation=SimpleNamespace(vllm_cfg=SimpleNamespace(temperature=1.0)),
            max_token_len_per_gpu=128,
        ),
        controller=SimpleNamespace(
            ray=SimpleNamespace(actor=role_cfg, ref=role_cfg),
        ),
    )
    sequences = torch.zeros(2, 4, dtype=torch.long)
    routes = _rows([3, 4])
    wg = FakeWorkerGroup()
    trainer._compute_log_probs_with_worker_group(
        wg,
        sequences,
        role="actor",
        attention_mask=torch.ones_like(sequences),
        rollout_routing=routes,
    )
    assert wg.request.ragged["rollout_routing"] == routes
    chunks = wg.request.split(2)
    assert chunks[0].ragged["rollout_routing"][0] is routes[0]
    assert chunks[1].ragged["rollout_routing"][0] is routes[1]


def test_replay_swaps_selection_and_regathers_weights():
    T, E = 5, 8
    probs = torch.rand(T, E, requires_grad=True)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    injected = torch.zeros(T, TOP_K, dtype=torch.long)
    injected[:, 0], injected[:, 1] = 7, 6
    valid = torch.tensor([True, True, False, True, False])

    idx3 = injected.unsqueeze(1).expand(T, N_LAYERS, TOP_K).contiguous()
    with RoutingReplayContext(idx3, valid):
        out = apply_replay(probs, computed, layer_idx=1)
    assert out is not None
    top_value, merged = out

    # valid rows take the injected experts, invalid rows keep the computed ones
    assert torch.equal(merged[valid], injected[valid])
    assert torch.equal(merged[~valid], computed[~valid])
    # weights come from the live probabilities at the replayed positions
    assert torch.allclose(top_value, probs.gather(1, merged))

    # and the router stays differentiable -- overwriting logits would break this
    top_value.sum().backward()
    assert probs.grad is not None and probs.grad.abs().sum() > 0


def test_no_context_is_a_noop():
    probs = torch.rand(4, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    assert apply_replay(probs, computed, layer_idx=0) is None


def test_token_count_mismatch_raises():
    idx = torch.zeros(5, N_LAYERS, TOP_K, dtype=torch.long)
    valid = torch.ones(5, dtype=torch.bool)
    probs = torch.rand(6, 8)  # router saw 6 tokens, routing has 5 rows
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        try:
            apply_replay(probs, computed, layer_idx=0)
        except RuntimeError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("misaligned routing was silently accepted")


def test_context_restores_previous_state():
    idx = torch.zeros(2, N_LAYERS, TOP_K, dtype=torch.long)
    valid = torch.ones(2, dtype=torch.bool)
    probs = torch.rand(2, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        with RoutingReplayContext(None, None):
            assert apply_replay(probs, computed, 0) is None
        assert apply_replay(probs, computed, 0) is not None
    assert apply_replay(probs, computed, 0) is None


def test_layer_index_parsing():
    assert layer_index_of("model.layers.7.mlp.gate") == 7
    assert layer_index_of("_fsdp_wrapped.model.layers.41.mlp.gate") == 41
    assert layer_index_of("model.mlp.gate") is None


def test_layer_lookup_is_order_independent():
    """Backward recompute visits layers in reverse; lookup must not care."""
    T = 3
    idx = torch.zeros(T, N_LAYERS, TOP_K, dtype=torch.long)
    for layer in range(N_LAYERS):
        idx[:, layer, :] = layer + 1
    valid = torch.ones(T, dtype=torch.bool)
    probs = torch.rand(T, 8)
    computed = torch.topk(probs, TOP_K, dim=-1)[1]
    with RoutingReplayContext(idx, valid):
        forward = [
            apply_replay(probs, computed, l)[1][0, 0].item() for l in range(N_LAYERS)
        ]
        backward = [
            apply_replay(probs, computed, l)[1][0, 0].item()
            for l in reversed(range(N_LAYERS))
        ]
    assert forward == [1, 2, 3] and backward == [3, 2, 1]


def test_megatron_checkpoint_uses_dp_zero_optimizer_state():
    """CPU-offloaded optimizer state must use the HDO-compatible format."""
    model_state = {"model": object()}
    optimizer_state = {"optimizer": object()}

    class ModuleStub:
        def sharded_state_dict(self):
            return model_state

    class OptimizerStub:
        def sharded_state_dict(self, state, *, is_loading, metadata):
            assert state is model_state
            assert is_loading is False
            assert metadata == {"distrib_optim_sharding_type": "dp_zero_gather_scatter"}
            return optimizer_state

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.module = ModuleStub()
    engine.optimizer = OptimizerStub()

    assert engine._dist_sharded_state_dict(False) == {
        "model": model_state,
        "optimizer": optimizer_state,
    }


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all R3 routing tests passed")
