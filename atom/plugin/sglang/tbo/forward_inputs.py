"""Prepare ATOM TBO forward inputs from SGLang-owned child batches."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from atom.plugin.sglang.tbo.adapter import (
    adapt_sglang_tbo_ubatch_slices,
    normalize_child_forward_batches,
)
from atom.utils.tbo.ubatch_splitting import UBatchSlice


@dataclass(frozen=True)
class SGLangTBOForwardInputs:
    """Validated child batches and slices ready for the ATOM executor."""

    ubatch_slices: list[UBatchSlice]
    child_forward_batches: list[ForwardBatch]
    ub_max_tokens_across_dp: tuple[int, int] | None


def _sync_atom_tbo_adapter_gate(
    *,
    ubatch_slices: list[UBatchSlice] | None,
    enable_expert_parallel: bool,
) -> tuple[bool, tuple[int, int] | None]:
    """Require every model-parallel rank to choose the same execution mode.

    SGLang already owns TBO eligibility and splitting. This additional gate
    only protects ATOM's executor: every rank participating in child-forward
    collectives must enter the same number of non-empty ubatches.

    EP groups can be strict subgroups of the TP/model-parallel world. ATOM child
    forwards also contain non-MoE TP collectives, so gating only within the EP
    subgroup could let one subgroup run two children while another falls back
    to one parent forward. Synchronize across the complete TP group.
    """

    local_ready = ubatch_slices is not None
    ubatch_token_counts = (
        [s.token_slice.stop - s.token_slice.start for s in ubatch_slices]
        if local_ready
        else [0, 0]
    )
    local = torch.tensor(
        [int(local_ready), *ubatch_token_counts],
        dtype=torch.int32,
        device="cpu",
    )
    if not dist.is_available() or not dist.is_initialized():
        return local_ready, tuple(ubatch_token_counts) if local_ready else None

    from sglang.srt.distributed import get_tp_group

    # Keep the parameter in this adapter API because the caller's topology is
    # still useful context and older integrations pass it explicitly. The gate
    # itself must cover the full model-parallel group in either mode.
    _ = enable_expert_parallel
    sync_group = get_tp_group()
    world_size = sync_group.world_size
    if world_size <= 1:
        return local_ready, tuple(ubatch_token_counts) if local_ready else None

    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=sync_group.cpu_group)
    sync = torch.stack(gathered, dim=0)
    if not bool(sync[:, 0].all().item()):
        return False, None
    return True, (
        int(sync[:, 1].max().item()),
        int(sync[:, 2].max().item()),
    )


def prepare_sglang_tbo_forward_inputs(
    forward_batch: ForwardBatch,
    *,
    enable_expert_parallel: bool,
) -> SGLangTBOForwardInputs | None:
    """Validate SGLang children collectively and build child-local views."""

    ubatch_slices = adapt_sglang_tbo_ubatch_slices(forward_batch)
    child_forward_batches = forward_batch.tbo_children
    global_ready, ub_max_tokens = _sync_atom_tbo_adapter_gate(
        ubatch_slices=ubatch_slices,
        enable_expert_parallel=enable_expert_parallel,
    )
    if not global_ready:
        return None

    normalized_children = normalize_child_forward_batches(
        list(child_forward_batches), ubatch_slices
    )
    return SGLangTBOForwardInputs(
        ubatch_slices=ubatch_slices,
        child_forward_batches=normalized_children,
        ub_max_tokens_across_dp=ub_max_tokens,
    )
