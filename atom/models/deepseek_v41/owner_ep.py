# SPDX-License-Identifier: MIT
"""Validate the native EP4 expert layout used by the TP4 owner deployment."""

import hashlib


def validate_mapping(mapping, mask, rank, *, expert_ids=None):
    """Validate native local-slot order, membership and non-local sentinel."""
    if rank not in range(4):
        raise ValueError("EP4 requires ranks 0..3")
    ids = list(range(rank * 96, (rank + 1) * 96)) if expert_ids is None else expert_ids
    if (
        len(ids) != 96
        or any(type(e) is not int for e in ids)
        or ids != sorted(set(ids))
        or any(e not in range(384) for e in ids)
    ):
        raise ValueError("Expected 96 ascending unique global expert IDs")
    expected = [-1] * 385
    for local, global_id in enumerate(ids):
        expected[global_id] = local
    if mapping != expected or mask != [int(e >= 0) for e in expected]:
        raise ValueError("EP4 owner mapping does not match its fixed placement")


def validate_owner_ep4(model, config, *, loaded=False):
    """Check actual modules at construction and again before graph warmup.

    No forward or capture uses this function. Attention, shared experts and
    request-owner reductions retain the global TP4 group; routed experts own
    96 complete checkpoint experts each, with no DP or all-to-all transport.
    """
    import torch
    from aiter.dist.parallel_state import get_dp_group, get_tp_group

    from atom.model_ops.moe import Mxfp4MoEMethod
    from atom.models.deepseek_v4 import DeepseekV4Args

    group = get_tp_group()
    args = DeepseekV4Args.from_hf_config(config.hf_config)
    if not (
        config.enable_expert_parallel
        and config.tensor_parallel_size == 4
        and config.parallel_config.data_parallel_size == 1
        and group.world_size == 4
        and get_dp_group().world_size == 1
    ):
        raise ValueError("Native EP4 owner requires one TP4 group and DP1")
    if (
        args.n_layers,
        args.dim,
        args.n_routed_experts,
        args.moe_inter_dim,
        args.n_activated_experts,
        args.n_hash_layers,
    ) != (40, 5120, 384, 2304, 6, 0):
        raise ValueError("EP4 owner requires the V4.1 Flash layer and routing geometry")
    if len(model.layers) != args.n_layers:
        raise ValueError("Incomplete owner backbone")
    placement = getattr(model, "owner_ep4_placement", None)
    if placement is not None:
        from .owner_placement import PLAN_ID, expected_experts

        if placement != PLAN_ID:
            raise ValueError("Unrecognized fixed expert placement")
    records = []
    for i, layer in enumerate(model.layers):
        moe = layer.ffn
        experts, shared = moe.experts, moe.shared_experts
        method = experts.quant_method
        parallel = experts.moe_parallel_config
        if not (
            moe.layer_id == i
            and not moe.is_hash_layer
            and not moe._use_comm_fused_dispatch
            and type(method) is Mxfp4MoEMethod
            and (experts.tp_size, experts.ep_size, experts.ep_rank, experts.dp_size)
            == (1, 4, group.rank_in_group, 1)
            and (
                experts.global_num_experts,
                experts.local_num_experts,
                experts.hidden_size,
                method.intermediate_size,
            )
            == (384, 96, 5120, 2304)
            and not parallel.use_all2all_kernels
            and parallel.selected_all2all_backend is None
            and parallel.dp_logical_ratio == 1
            and not experts.expert_layout.uses_dispatch_remap
            and experts.num_fused_shared_experts == 0
            and experts._comm_fused_moe is None
            and experts.supports_prequantized_input()
        ):
            raise ValueError(f"Unsupported EP4 expert layout at layer {i}")
        route = experts.custom_routing_function
        if not (
            getattr(route, "__self__", None) is moe
            and getattr(route, "__func__", None) is type(moe)._topk
            and experts.top_k == 6
            and experts.renormalize
            and moe.routed_scaling_factor == 1.5
            and tuple(moe.gate.weight.shape) == (384, 5120)
        ):
            raise ValueError(f"Unsupported EP4 routing at layer {i}")
        if not (
            shared is not None
            and shared.gate_up_proj.tp_size == 4
            and shared.w2.tp_size == 4
            and not shared.w2.reduce_results
            and shared.gate_up_proj.native_a8_group_rows == 32
        ):
            raise ValueError(
                f"EP4 owner requires native TP4 shared expert at layer {i}"
            )
        if (tuple(experts.w13_weight.shape), tuple(experts.w2_weight.shape)) != (
            (96, 4608, 2560),
            (96, 5120, 1152),
        ):
            raise ValueError(f"Unexpected full-expert weights at layer {i}")
        if (
            experts.expert_map.dtype != torch.int32
            or experts.expert_mask.dtype != torch.int32
        ):
            raise ValueError(f"Invalid expert index dtype at layer {i}")
        mapping = experts.expert_map.detach().cpu().tolist()
        mask = experts.expert_mask.detach().cpu().tolist()
        validate_mapping(
            mapping,
            mask,
            group.rank_in_group,
            expert_ids=(
                expected_experts(i, group.rank_in_group)
                if placement is not None
                else None
            ),
        )
        if loaded and not (
            experts.w13_weight.is_shuffled and experts.w2_weight.is_shuffled
        ):
            raise ValueError(f"Native weight processing is incomplete at layer {i}")
        records.append(
            {
                "layer": i,
                "expert_TP": 1,
                "EP": 4,
                "DP": 1,
                "local_experts": 96,
                "shared_TP": 4,
                "hash_routing": False,
                "router": route.__func__.__qualname__,
                "prequantized_consumer": True,
                "mapping_sha256": hashlib.sha256(str(mapping).encode()).hexdigest(),
                "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(),
            }
        )
    return {
        "protocol": "v41-native-ep4-owner-v1",
        **({"fixed_placement": placement} if placement is not None else {}),
        "rank": group.rank_in_group,
        "loaded": loaded,
        "layers": records,
        "all2all": None,
        "owner_TP": 4,
        "hash_layers": [],
    }
