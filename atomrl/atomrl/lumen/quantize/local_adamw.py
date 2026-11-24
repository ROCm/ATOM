# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused AdamW on FSDP2's local shards, without DTensor dispatch on every step.

Under FSDP2 every parameter, gradient and optimizer state is a DTensor, and a
fused ``torch.optim.AdamW`` step pays for that in Python even though its kernel
is already the fused one: ``_init_group`` walks every parameter each step, and
each call on DTensors (the device/dtype grouping, the step increment, the
fused op) unwraps thousands of them through DTensor's ``__torch_dispatch__``.
Measured on the Qwen-Image DiT (~1900 parameters, 8x MI350X): the optimizer
phase costs 43.9 ms/step of host time for 13.6 ms of GPU work.

:func:`install_local_shard_adamw` replaces ``AdamW.step`` with one that caches,
per param group, the device/dtype buckets of the *local* shards of params and
state, refreshes only the gradient list each step, and makes the same two calls
``torch.optim.adam._fused_adam`` makes -- ``_foreach_add_`` on the step
counters, then ``_fused_adamw_`` -- with the same arguments, on plain tensors.
Same kernel, same inputs: the update is bit-identical (loss and grad-norm 10/10
steps equal to the stock optimizer), and the phase drops to 6.4 ms.

Falls back to the stock step for anything it does not reproduce exactly: a
closure, a group that is not fused / decoupled-weight-decay, amsgrad, maximize,
capturable, differentiable, a tensor learning rate, or a step whose state has
not been initialised yet (the first one).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _eligible(group):
    return (
        group.get("fused")
        and group.get("decoupled_weight_decay", True)
        and not group.get("amsgrad")
        and not group.get("maximize")
        and not group.get("capturable")
        and not group.get("differentiable")
        and not isinstance(group["lr"], torch.Tensor)
    )


def install_local_shard_adamw():
    """Rebind ``torch.optim.AdamW.step``. Idempotent; returns True once installed."""
    from torch.distributed.tensor import DTensor
    from torch.optim import AdamW
    from torch.optim.optimizer import Optimizer

    if getattr(AdamW.step, "_lumen_local_shard", False):
        return True
    orig_step = AdamW.step

    def local(t):
        return t._local_tensor if isinstance(t, DTensor) else t

    def build_cache(opt, params):
        tensors = [
            [local(p) for p in params],
            [local(p.grad) for p in params],
            [local(opt.state[p]["exp_avg"]) for p in params],
            [local(opt.state[p]["exp_avg_sq"]) for p in params],
            [],
            [opt.state[p]["step"] for p in params],
        ]
        grouped = Optimizer._group_tensors_by_device_and_dtype(
            tensors, with_indices=True
        )
        buckets = []
        for lists, indices in grouped.values():
            buckets.append(
                {
                    "params": lists[0],
                    "exp_avgs": lists[2],
                    "exp_avg_sqs": lists[3],
                    "steps": lists[5],
                    "indices": list(indices),
                }
            )
        return {
            "params": params,
            "exp_avg0": opt.state[params[0]]["exp_avg"],
            "nstate": len(opt.state),
            "buckets": buckets,
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or not all(_eligible(g) for g in self.param_groups):
            return orig_step(self, closure)
        plans = []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                plans.append((group, None, None))
                continue
            if any(
                p not in self.state or "exp_avg" not in self.state[p] for p in params
            ):
                return orig_step(
                    self, closure
                )  # first step initialises state the stock way
            caches = self.__dict__.setdefault("_lumen_local_cache", {})
            c = caches.get(id(group))
            if (
                c is None
                or len(c["params"]) != len(params)
                or c["exp_avg0"] is not self.state[params[0]]["exp_avg"]
                or c["nstate"] != len(self.state)
            ):
                c = caches[id(group)] = build_cache(self, params)
            plans.append((group, params, c))

        for group, params, c in plans:
            if params is None:
                continue
            grads = [local(p.grad) for p in params]
            beta1, beta2 = group["betas"]
            for b in c["buckets"]:
                torch._foreach_add_(b["steps"], 1)
                torch._fused_adamw_(
                    b["params"],
                    [grads[i] for i in b["indices"]],
                    b["exp_avgs"],
                    b["exp_avg_sqs"],
                    [],
                    b["steps"],
                    amsgrad=False,
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                    grad_scale=None,
                    found_inf=None,
                )
        return None

    step._lumen_local_shard = True
    step._lumen_orig = orig_step
    AdamW.step = step
    logger.info("local_adamw: AdamW.step runs fused on cached local shards")
    return True


def uninstall_local_shard_adamw():
    from torch.optim import AdamW

    orig = getattr(AdamW.step, "_lumen_orig", None)
    if orig is not None:
        AdamW.step = orig


__all__ = ["install_local_shard_adamw", "uninstall_local_shard_adamw"]
