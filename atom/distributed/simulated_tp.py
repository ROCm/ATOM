# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Run the first N devices of a wider TP deployment on a box that has N GPUs.

`-tp 8 --fake-eplb` on 4 GPUs launches 4 workers that each behave like rank 0..3
of a real TP8 run, so per-device shapes and memory match the 8-GPU deployment --
which is what a kernel benchmark measures.

Every shard-size computation in the tree bottoms out at
`get_tp_group().world_size`, so making that one number report the logical width
shards the whole model 8 ways without touching a layer. The same number also
sizes collectives, where the real membership is what counts: `all_reduce` only
reads it for a `== 1` shortcut, but `all_gather` / `gather` / `reduce_scatter`
shape their buffers from it and are replaced below with versions that talk to
the real group and pad the absent ranks with zeros.

Shapes stay right for every caller; values do not (an all-reduce over half the
shards is half a sum). Hence the `--fake-eplb` gate, which already means "this
run's output is garbage, I am measuring kernels".

This goes all the way down to one device, where every collective degenerates to
a local op -- one card then runs rank 0 of a TP8 deployment, holding 1/8 of each
weight and computing exactly what that rank would.
"""

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from atom.config import Config

logger = logging.getLogger("atom")


def apply_simulated_tp(config: "Config") -> None:
    """Make the TP group report `tensor_parallel_size` ranks when fewer exist.

    Call once per worker, right after the process group is built. A no-op
    unless `Config.tp_world_size` came out lower, which needs `--fake-eplb`.
    """
    logical = config.tensor_parallel_size
    physical = config.tp_world_size
    if logical == physical:
        return

    # Local import so the module stays importable (and testable) without aiter.
    from aiter.dist.parallel_state import get_tp_group

    _reject_unsupported(config, logical, physical)
    group = get_tp_group()
    assert group.world_size == physical, (
        f"TP group has {group.world_size} ranks, expected the physical "
        f"world size {physical}; was it built with tp_world_size?"
    )
    _patch_group(group, logical, physical)

    logger.warning(
        "Simulated TP: running ranks 0-%d of a TP%d deployment on %d device(s). "
        "Per-device shapes match TP%d, but collectives only cover the ranks that "
        "exist, so THE MODEL OUTPUT IS MEANINGLESS. Benchmarking only.",
        physical - 1,
        logical,
        physical,
        logical,
    )


def reject_simulated_tp(config: "Config", why: str) -> None:
    """Raise if simulated TP is active, for paths `apply_simulated_tp` misses.

    Those paths would otherwise hang on a group sized for absent ranks.
    """
    logical = config.tensor_parallel_size
    physical = config.tp_world_size
    if logical != physical:
        _raise(logical, physical, f"does not support {why}")


def _raise(logical: int, physical: int, why: str) -> None:
    raise ValueError(
        f"Simulated TP (-tp {logical} on {physical} device(s)) {why}. "
        f"Either free up {logical} devices or lower -tp."
    )


def _reject_unsupported(config: "Config", logical: int, physical: int) -> None:
    def _reject(why: str) -> None:
        _raise(logical, physical, why)

    if not config.fake_eplb:
        _reject("requires --fake-eplb")
    if physical < 1:
        _reject("needs at least one device")
    if config.pipeline_parallel_size > 1:
        # next_rank/prev_rank index `ranks` by (rank ± 1) % world_size.
        _reject("does not support pipeline parallel")
    if config.prefill_context_parallel_size > 1:
        _reject("does not support prefill context parallel")
    if config.decode_context_parallel_size > 1:
        _reject("does not support decode context parallel")
    if config.parallel_config.data_parallel_size > 1 or config.enable_dp_attention:
        # mori all2all is real peer-to-peer: absent ranks cannot be faked, and
        # it derives the destination as expert_id // (E // real peer count).
        _reject("does not support data parallel / DP-attention")
    if config.enable_tbo:
        _reject("has not been validated with TBO")
    if getattr(config, "eplb_enable", False):
        _reject("conflicts with EPLB, which rebalances across real ranks")
    if config.kv_transfer_config or config.enable_rapidserve:
        # Disagg connectors count ranks from tensor_parallel_size, so prefill
        # and decode would disagree about how many exist.
        _reject("does not support disaggregated prefill / KV transfer")


def _patch_group(group, logical: int, physical: int) -> None:
    """Report `logical` ranks; keep collectives on the `physical` ones."""
    device_group = group.device_group
    rank_in_group = group.rank_in_group

    def _all_gather(
        input_: torch.Tensor, use_custom: bool = False, dim: int = -1
    ) -> torch.Tensor:
        # Not delegating to the original: it sizes its buffer (and the custom
        # kernel's) from world_size, which now lies.
        dim = dim % input_.dim()
        input_ = input_.contiguous()
        rows, rest = input_.shape[0], tuple(input_.shape[1:])
        # Rank-major and zero-filled, so absent ranks read as 0. Concatenated
        # rather than stacked on a new leading axis: gloo's
        # all_gather_into_tensor only accepts the flat form.
        flat = torch.zeros(
            (logical * rows,) + rest, dtype=input_.dtype, device=input_.device
        )
        if physical == 1:
            # Nothing to talk to; the one shard that exists is this rank's.
            flat[:rows].copy_(input_)
        else:
            torch.distributed.all_gather_into_tensor(
                flat[: physical * rows], input_, group=device_group
            )
        shape = list(input_.shape)
        shape[dim] *= logical
        return flat.view((logical, rows) + rest).movedim(0, dim).reshape(shape)

    def _gather(
        input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        dim = dim % input_.dim()
        input_ = input_.contiguous()
        is_dst = rank_in_group == dst
        gather_list = (
            [torch.empty_like(input_) for _ in range(physical)] if is_dst else None
        )
        if physical == 1:
            gather_list[0].copy_(input_)
        else:
            torch.distributed.gather(
                input_, gather_list, dst=group.ranks[dst], group=device_group
            )
        if not is_dst:
            return None
        absent = [torch.zeros_like(input_) for _ in range(logical - physical)]
        return torch.cat(gather_list + absent, dim=dim)

    def _reduce_scatter_tensor(
        input_: torch.Tensor, use_custom: bool = True, dim: int = 0
    ) -> torch.Tensor:
        # `logical` shards in, only `physical` ranks to take one: not
        # expressible as a scatter, so reduce in full and keep our slice.
        dim = dim % input_.dim()
        assert input_.shape[dim] % logical == 0, (
            f"reduce_scatter input dim {dim} = {input_.shape[dim]} is not "
            f"divisible by the logical TP size {logical}"
        )
        reduced = group.all_reduce(input_)
        shard = reduced.shape[dim] // logical
        return reduced.narrow(dim, rank_in_group * shard, shard).contiguous()

    def _reduce_scatter(input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return _reduce_scatter_tensor(input_, dim=dim)

    group.all_gather = _all_gather
    group.gather = _gather
    group.reduce_scatter_tensor = _reduce_scatter_tensor
    group.reduce_scatter = _reduce_scatter
    if physical == 1:
        # A lone rank has no peers and, because GroupCoordinator only builds a
        # device communicator for world_size > 1, no communicator either -- and
        # the `world_size == 1` shortcut that used to cover that stops firing
        # once world_size reads as logical. Summing the one shard that exists
        # is the identity, so say so rather than dispatching.
        group.all_reduce = lambda input_, *a, **kw: input_
    # Last: the wrappers close over the real sizes.
    group.simulated_tp_physical_world_size = physical
    group.world_size = logical
