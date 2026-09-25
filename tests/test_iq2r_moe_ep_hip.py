# SPDX-License-Identifier: MIT

"""Multi-GPU correctness tests for IQ2R expert-parallel route remapping.

Run explicitly with::

    HIP_VISIBLE_DEVICES=0,1,2,3 RUN_IQ2R_EP_TEST=1 \
      torchrun --standalone --nproc-per-node=4 -m pytest -q \
      tests/test_iq2r_moe_ep_hip.py

or::

    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_IQ2R_EP_TEST=1 \
      torchrun --standalone --nproc-per-node=8 -m pytest -q \
      tests/test_iq2r_moe_ep_hip.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.ops.iq2r import iq2r_encode_device
from aiter.ops.iq2r_encoder import iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2RMetadata

from atom.model_ops import moe as moe_mod

_RUN_EP_TEST = (
    os.environ.get("RUN_IQ2R_EP_TEST") == "1"
    or os.environ.get("RUN_IQ2R_EP4_TEST") == "1"
)

pytestmark = pytest.mark.skipif(
    not _RUN_EP_TEST,
    reason="set RUN_IQ2R_EP_TEST=1 and launch with 4 or 8 torchrun workers",
)


@pytest.fixture(scope="module", autouse=True)
def _nccl_process_group():
    """Keep one process group alive across all tests in a torchrun worker."""
    if not _RUN_EP_TEST or int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        yield
        return

    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        yield
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def _method(*, local_experts: int, global_experts: int, use_ep: bool):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        max_num_tokens=4,
    )
    method.num_experts = local_experts
    method.global_num_experts = global_experts
    method.expert_start = (
        dist.get_rank() * local_experts if use_ep and dist.is_initialized() else 0
    )
    method.hidden_size = 128
    method.intermediate_size = 128
    method._workspaces = {}
    return method


def _layer(gate_data, gate_auxiliary, down_data, down_auxiliary):
    return SimpleNamespace(
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        w13_weight=gate_data,
        w13_weight_scale=gate_auxiliary,
        w2_weight=down_data,
        w2_weight_scale=down_auxiliary,
        w13_bias=None,
        w2_bias=None,
        iq2r_gate_up_metadata=IQ2RMetadata(logical_n=256, logical_k=128),
        iq2r_down_metadata=IQ2RMetadata(logical_n=128, logical_k=128),
        iq2r_gate_up_tile_n=128,
        iq2r_down_tile_n=128,
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )


def test_iq2r_ep_sum_matches_full_expert_execution():
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("requires exactly four or eight torchrun workers")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    global_experts = 8
    local_experts = global_experts // dist.get_world_size()
    expert_start = rank * local_experts
    codebook = iq2r_initial_codebook(device)
    importance = torch.ones(128, dtype=torch.float32, device=device)

    gate_data = []
    gate_auxiliary = []
    down_data = []
    down_auxiliary = []
    for expert in range(global_experts):
        generator = torch.Generator(device=device).manual_seed(0x5300 + expert)
        gate = (
            torch.randn((256, 128), generator=generator, device=device) * 0.02
        ).contiguous()
        down = (
            torch.randn((128, 128), generator=generator, device=device) * 0.02
        ).contiguous()
        data, auxiliary = iq2r_encode_device(gate, importance, codebook)
        gate_data.append(data)
        gate_auxiliary.append(auxiliary)
        data, auxiliary = iq2r_encode_device(down, importance, codebook)
        down_data.append(data)
        down_auxiliary.append(auxiliary)

    gate_data = torch.stack(gate_data)
    gate_auxiliary = torch.stack(gate_auxiliary)
    down_data = torch.stack(down_data)
    down_auxiliary = torch.stack(down_auxiliary)

    generator = torch.Generator(device=device).manual_seed(0x53E4)
    hidden = torch.randn((3, 128), generator=generator, device=device).to(
        torch.bfloat16
    )
    logits = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 3.0, 5.0, 7.0, 8.0, 6.0, 4.0, 2.0],
            [2.0, 8.0, 4.0, 6.0, 1.0, 7.0, 3.0, 5.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    correction_bias = torch.zeros(global_experts, dtype=torch.float32, device=device)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[expert_start : expert_start + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )

    local_method = _method(
        local_experts=local_experts,
        global_experts=global_experts,
        use_ep=True,
    )
    local = local_method.apply(
        layer=_layer(
            gate_data[expert_start : expert_start + local_experts].contiguous(),
            gate_auxiliary[expert_start : expert_start + local_experts].contiguous(),
            down_data[expert_start : expert_start + local_experts].contiguous(),
            down_auxiliary[expert_start : expert_start + local_experts].contiguous(),
        ),
        x=hidden,
        router_logits=logits,
        top_k=2,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        global_num_experts=global_experts,
        expert_map=expert_map,
        scoring_func="sigmoid",
        e_score_correction_bias=correction_bias,
        activation=moe_mod.ActivationType.Swiglu,
    )
    dist.all_reduce(local)

    reference = _method(
        local_experts=global_experts,
        global_experts=global_experts,
        use_ep=False,
    ).apply(
        layer=_layer(gate_data, gate_auxiliary, down_data, down_auxiliary),
        x=hidden,
        router_logits=logits,
        top_k=2,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        global_num_experts=global_experts,
        expert_map=None,
        scoring_func="sigmoid",
        e_score_correction_bias=correction_bias,
        activation=moe_mod.ActivationType.Swiglu,
    )

    torch.testing.assert_close(local, reference, rtol=0.01, atol=0.02)
    assert torch.isfinite(local).all()

    # The original global-to-local remap used index_select and could survive a
    # single layer while faulting after dozens of asynchronous launches in a
    # production-sized stack. Queue more than GLM-5.3's 75 routed layers before
    # synchronizing so this test retains coverage for that failure mode.
    pending = []
    for step in range(80):
        pending.append(
            local_method.apply(
                layer=_layer(
                    gate_data[expert_start : expert_start + local_experts].contiguous(),
                    gate_auxiliary[
                        expert_start : expert_start + local_experts
                    ].contiguous(),
                    down_data[expert_start : expert_start + local_experts].contiguous(),
                    down_auxiliary[
                        expert_start : expert_start + local_experts
                    ].contiguous(),
                ),
                x=hidden,
                router_logits=torch.roll(logits, shifts=step, dims=1),
                top_k=2,
                renormalize=True,
                use_grouped_topk=True,
                topk_group=1,
                num_expert_group=1,
                global_num_experts=global_experts,
                expert_map=expert_map,
                scoring_func="sigmoid",
                e_score_correction_bias=correction_bias,
                activation=moe_mod.ActivationType.Swiglu,
            )
        )
    torch.cuda.synchronize(device)
    assert all(torch.isfinite(output).all() for output in pending)


def test_plain_glm53_real_layer_ep_matches_full_execution():
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("requires exactly four or eight torchrun workers")
    checkpoint_dir = os.environ.get("GLM53_IQ2R_DIAGNOSTIC_DIR")
    if not checkpoint_dir:
        pytest.skip("set GLM53_IQ2R_DIAGNOSTIC_DIR to a compiled plain-GLM layer")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    global_experts = 256
    local_experts = global_experts // dist.get_world_size()
    expert_start = rank * local_experts
    checkpoint = load_iq2r_layer_checkpoint(
        checkpoint_dir,
        3,
        expert_start=expert_start,
        expert_count=local_experts,
        device=device,
    )
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    expert_map[expert_start : expert_start + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=device
    )
    generator = torch.Generator(device=device).manual_seed(0x53D5)
    hidden = torch.randn((1, 6144), generator=generator, device=device).to(
        torch.bfloat16
    )
    logits = torch.full((1, global_experts), -20.0, dtype=torch.float32, device=device)
    selected = torch.tensor(
        [0, 63, 64, 127, 128, 191, 192, 255], dtype=torch.int64, device=device
    )
    logits[0, selected] = torch.arange(8, 0, -1, dtype=torch.float32, device=device)
    correction_bias = torch.zeros(global_experts, dtype=torch.float32, device=device)

    local = _method(
        local_experts=local_experts,
        global_experts=global_experts,
        use_ep=True,
    )
    local.hidden_size = 6144
    local.intermediate_size = 2048
    local_output = local.apply(
        layer=SimpleNamespace(
            num_fused_shared_experts=0,
            routed_scaling_factor=2.5,
            w13_weight=checkpoint.gate_up_data,
            w13_weight_scale=checkpoint.gate_up_auxiliary,
            w2_weight=checkpoint.down_data,
            w2_weight_scale=checkpoint.down_auxiliary,
            w13_bias=None,
            w2_bias=None,
            iq2r_gate_up_metadata=checkpoint.gate_up_metadata,
            iq2r_down_metadata=checkpoint.down_metadata,
            iq2r_gate_up_tile_n=checkpoint.gate_up_tile_n,
            iq2r_down_tile_n=checkpoint.down_tile_n,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_up_offset=0.0,
        ),
        x=hidden,
        router_logits=logits,
        top_k=8,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        global_num_experts=global_experts,
        expert_map=expert_map,
        scoring_func="sigmoid",
        e_score_correction_bias=correction_bias,
        activation=moe_mod.ActivationType.Swiglu,
    )
    dist.all_reduce(local_output)

    full_checkpoint = load_iq2r_layer_checkpoint(
        checkpoint_dir,
        3,
        device=device,
    )
    full = _method(
        local_experts=global_experts,
        global_experts=global_experts,
        use_ep=False,
    )
    full.hidden_size = 6144
    full.intermediate_size = 2048
    reference = full.apply(
        layer=SimpleNamespace(
            num_fused_shared_experts=0,
            routed_scaling_factor=2.5,
            w13_weight=full_checkpoint.gate_up_data,
            w13_weight_scale=full_checkpoint.gate_up_auxiliary,
            w2_weight=full_checkpoint.down_data,
            w2_weight_scale=full_checkpoint.down_auxiliary,
            w13_bias=None,
            w2_bias=None,
            iq2r_gate_up_metadata=full_checkpoint.gate_up_metadata,
            iq2r_down_metadata=full_checkpoint.down_metadata,
            iq2r_gate_up_tile_n=full_checkpoint.gate_up_tile_n,
            iq2r_down_tile_n=full_checkpoint.down_tile_n,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_up_offset=0.0,
        ),
        x=hidden,
        router_logits=logits,
        top_k=8,
        renormalize=True,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        global_num_experts=global_experts,
        expert_map=None,
        scoring_func="sigmoid",
        e_score_correction_bias=correction_bias,
        activation=moe_mod.ActivationType.Swiglu,
    )

    torch.testing.assert_close(local_output, reference, rtol=0.01, atol=0.05)
    assert torch.isfinite(local_output).all()
