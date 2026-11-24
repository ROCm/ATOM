# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Compare DSV4 hash-MoE replay weights with the vLLM reference router."""

from __future__ import annotations

import torch
from megatron.core.transformer.moe.moe_utils import (
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router_replay import (
    RouterReplay,
    RouterReplayAction,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)


def main() -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(7)
    num_tokens, num_experts, topk = 32, 256, 6
    hidden = torch.randn(num_tokens, 16, device="cuda")
    logits = torch.randn(
        num_tokens,
        num_experts,
        device="cuda",
        dtype=torch.float32,
    )
    input_tokens = torch.arange(
        num_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    tid2eid = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="cuda",
        dtype=torch.int32,
    )

    rollout_weights, rollout_ids = fused_topk_bias(
        hidden,
        logits,
        "sqrtsoftplus",
        None,
        topk,
        True,
        input_tokens=input_tokens,
        hash_indices_table=tid2eid,
        routed_scaling_factor=1.5,
    )

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    replay.set_target_indices(tid2eid.long())
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    training_probs, _ = topk_routing_with_score_function(
        logits,
        topk,
        scaling_factor=1.5,
        score_function="sqrtsoftplus",
        router_replay=replay,
    )
    training_weights = training_probs.gather(1, rollout_ids.long())

    torch.testing.assert_close(rollout_ids, tid2eid)
    torch.testing.assert_close(
        training_weights,
        rollout_weights,
        rtol=2e-6,
        atol=2e-6,
    )
    print(
        "HASH_TID2EID_WEIGHT_PARITY_OK "
        f"max_diff={float((training_weights - rollout_weights).abs().max())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
