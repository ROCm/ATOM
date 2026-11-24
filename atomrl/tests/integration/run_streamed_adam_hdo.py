# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Exercise patched Megatron streamed Adam through HybridDeviceOptimizer."""

from __future__ import annotations

import torch

_CHUNK_MIB = 1
_CHUNK_NUMEL = _CHUNK_MIB * 1024 * 1024 // torch.float32.itemsize
_PARAM_NUMEL = 2 * _CHUNK_NUMEL + 137
_STEPS = 3


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-5)


def main() -> None:
    if not torch.cuda.is_available():
        print("STREAMED_ADAM_HDO_SKIPPED_NO_GPU", flush=True)
        raise SystemExit(0)

    from megatron.core.optimizer.cpu_offloading import hybrid_optimizer

    assert (
        getattr(
            hybrid_optimizer,
            "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM",
            False,
        )
        is True
    ), "patched Megatron streamed Adam capability is required"

    torch.cuda.set_device(0)
    torch.manual_seed(41)
    initial = torch.linspace(-0.75, 0.75, _PARAM_NUMEL, dtype=torch.float32)
    gpu_parameter = torch.nn.Parameter(initial.cuda())
    reference_parameter = torch.nn.Parameter(initial.clone())
    options = {
        "lr": 2e-3,
        "betas": (0.9, 0.98),
        "eps": 1e-8,
        "weight_decay": 0.07,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
    }
    optimizer = hybrid_optimizer.HybridDeviceOptimizer(
        [gpu_parameter],
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=None,
        param_update_in_fp32=True,
        pin_cpu_grads=True,
        pin_cpu_params=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        **options,
    )
    optimizer._lumen_streamed_optimizer_mode = "adam"
    optimizer._lumen_streamed_adam_chunk_numel = _CHUNK_NUMEL
    optimizer._lumen_streamed_adam_moment_dtype = torch.bfloat16
    optimizer._lumen_streamed_adam_rss_announced = {
        "after state allocation",
        "after first step",
    }

    assert optimizer.offload_fraction == 1.0
    assert optimizer.gpu_optimizer is None
    assert len(optimizer.cpu_optimizers) == 1
    cpu_optimizer = optimizer.cpu_optimizers[0]
    assert isinstance(cpu_optimizer, torch.optim.AdamW)
    assert cpu_optimizer.param_groups[0]["fused"] is True
    cpu_parameter = cpu_optimizer.param_groups[0]["params"][0]
    assert cpu_parameter.device.type == "cpu"
    assert cpu_parameter.dtype == torch.float32
    assert cpu_parameter.is_pinned()
    assert optimizer.cpu_copys_map_gpu_param[cpu_parameter] is gpu_parameter

    reference = torch.optim.AdamW(
        [reference_parameter],
        lr=options["lr"],
        betas=options["betas"],
        eps=options["eps"],
        weight_decay=options["weight_decay"],
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )

    for step_index in range(_STEPS):
        gradient = torch.linspace(
            -0.2 + 0.03 * step_index,
            0.3 - 0.02 * step_index,
            _PARAM_NUMEL,
            dtype=torch.float32,
        )
        gpu_parameter.decoupled_grad = gradient.cuda()
        reference_parameter.grad = gradient.clone()

        optimizer.step()
        reference.step()
        reference.zero_grad(set_to_none=True)

        assert gpu_parameter.decoupled_grad is None
        assert gpu_parameter.grad is None
        _assert_close(cpu_parameter, reference_parameter)
        _assert_close(gpu_parameter.cpu(), reference_parameter)

        streamed_state = cpu_optimizer.state[cpu_parameter]
        reference_state = reference.state[reference_parameter]
        assert int(streamed_state["step"].item()) == step_index + 1
        assert int(reference_state["step"].item()) == step_index + 1
        _assert_close(streamed_state["exp_avg"], reference_state["exp_avg"])
        _assert_close(
            streamed_state["exp_avg_sq"],
            reference_state["exp_avg_sq"],
        )

    state = cpu_optimizer.state[cpu_parameter]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert set(optimizer.state) == {gpu_parameter}
    assert cpu_parameter not in optimizer.state
    public_state = optimizer.state[gpu_parameter]
    assert set(public_state) == {
        "step",
        "exp_avg",
        "exp_avg_sq",
        "master_param",
    }
    assert public_state["step"] is streamed_state["step"]
    assert public_state["exp_avg"] is streamed_state["exp_avg"]
    assert public_state["exp_avg_sq"] is streamed_state["exp_avg_sq"]
    assert public_state["master_param"] is cpu_parameter
    assert streamed_state["master_param"] is cpu_parameter
    _assert_close(public_state["step"], reference_state["step"])
    _assert_close(public_state["exp_avg"], reference_state["exp_avg"])
    _assert_close(public_state["exp_avg_sq"], reference_state["exp_avg_sq"])
    _assert_close(public_state["master_param"], reference_parameter)

    assert hasattr(optimizer, "_lumen_streamed_adam_last_h2d_event")
    last_h2d_event = optimizer._lumen_streamed_adam_last_h2d_event
    assert last_h2d_event.query()
    staging = optimizer._lumen_streamed_adam_buffers
    assert staging.shape == (4, _CHUNK_NUMEL)
    assert staging.dtype == torch.float32
    assert staging.is_pinned()
    assert not optimizer.cpu_copy_map_grad
    print(
        "STREAMED_ADAM_HDO_OK buffers=4 moments=bf16 chunk_mib=1 "
        "h2d_synchronized=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
