# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Offline gate for HDO checkpoint step canonicalization."""

from __future__ import annotations

import logging

import torch
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class _MinimalHDO(HybridDeviceOptimizer):
    def state_dict(self):
        return {
            "state": self.sub_optimizers[0].state,
            "param_groups": [{"params": [0, 1, 2]}],
        }


def main() -> None:
    parameters = [torch.nn.Parameter(torch.tensor(float(index))) for index in range(3)]
    adam = torch.optim.Adam(parameters)
    original_steps = (25.0, 49.0, 50.0)
    for parameter, step in zip(parameters, original_steps):
        adam.state[parameter] = {
            "step": torch.tensor(step),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }

    hdo = object.__new__(_MinimalHDO)
    hdo.cpu_optimizers = [adam]
    hdo.gpu_optimizer = None
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = hdo
    optimizer.grad_scaler = None

    logging.basicConfig(level=logging.WARNING)
    checkpoint_state = optimizer.state_dict()
    saved_step = checkpoint_state["optimizer"]["param_groups"][0]["step"]
    live_steps = tuple(adam.state[parameter]["step"].item() for parameter in parameters)
    assert saved_step == 50
    assert live_steps == original_steps
    print(
        "HDO_CHECKPOINT_STEP_GATE_OK "
        f"saved_step={saved_step} live_steps={live_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
