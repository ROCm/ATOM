# SPDX-License-Identifier: MIT
# Pipeline-parallel communication and distributed-init helpers (CPP A·P1a).
#
# aiter's `init_dist_env` hardcodes pipeline_model_parallel_size=1 and does not
# expose it. Rather than patch aiter, this module replicates its body while
# threading `pipeline_model_parallel_size` through to the lower-level aiter
# primitives, so ATOM can build a real `_PP` GroupCoordinator. The pp=1 path is
# left to aiter's own `init_dist_env` (byte-identical to today).
#
# The aiter rank layout is DP x PP x PCP x TP (TP innermost/contiguous), i.e.
#   all_ranks.reshape(-1, dp, pp, pcp, tp)
# so a global rank decomposes as:
#   tp_idx  =  rank                       % tp
#   pcp_idx = (rank // tp)                % pcp
#   pp_idx  = (rank // (tp*pcp))          % pp
#   dp_idx  =  rank // (tp*pcp*pp)

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    ensure_model_parallel_initialized,
)
from aiter.ops.communication import set_custom_all_reduce
from atom.models.utils import IntermediateTensors

logger = logging.getLogger("atom")

# Keys carried between pipeline stages.
_PP_PROXY_KEYS = ("hidden_states", "residual")


@dataclass
class P2PWork:
    """Handle for an in-flight isend; keeps the payload tensor alive until
    the send completes so the underlying buffer is not GC'd or reused."""

    work: Optional[torch.distributed.Work]
    payload: Optional[torch.Tensor]


def init_pp_aware_dist_env(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    global_rank: int,
    world_size: int,
    distributed_init_method: str,
    backend: str = "nccl",
    data_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> None:
    """PP-aware distributed init for the "one EngineCore per stage" model.

    Unlike aiter's init_dist_env (which pins pp=1), this threads
    pipeline_model_parallel_size into group construction. Because each PP stage
    is a *separate* EngineCore process with its own CUDA_VISIBLE_DEVICES, the
    caller passes the already-resolved GLOBAL rank (in the DPxPPxPCPxTP layout)
    and the full world_size; we set data_parallel_size=1 for the environment
    init (no internal DP rank offset — the caller already folded dp/pp into
    global_rank) but pass the real data_parallel_size to the group builder so
    TP/PP/DP groups come out correct.
    """
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=global_rank,
        distributed_init_method=distributed_init_method,
        backend=backend,
        data_parallel_size=1,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        decode_context_model_parallel_size=decode_context_parallel_size,
        data_parallel_size=data_parallel_size,
        prefill_context_model_parallel_size=prefill_context_model_parallel_size,
    )

    if tensor_model_parallel_size > 1:
        tp_grp = get_tp_group()
        ca_comm = tp_grp.device_communicator.ca_comm
        signal = torch.zeros(
            tensor_model_parallel_size * 64,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        ca_comm.signal = signal
        ca_comm.register_input_buffer(signal)
        ca_comm.buffer = ca_comm._pool["input"].tensor

    logger.debug(
        "init_pp_aware_dist_env: global_rank=%d tp=%d pp=%d pcp=%d dp=%d world=%d",
        global_rank,
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        data_parallel_size,
        world_size,
    )


def send_intermediate_tensors(it: IntermediateTensors) -> None:
    """Send hidden_states/residual to the next pipeline stage (blocking).

    aiter's send_tensor_dict expects `dst` as the GROUP-relative rank (not the
    global rank that `next_rank` returns), so compute the next index in the PP
    group directly.
    """
    pp = get_pp_group()
    tensors = {k: it.tensors[k] for k in _PP_PROXY_KEYS if k in it.tensors}
    dst = (pp.rank_in_group + 1) % pp.world_size
    pp.send_tensor_dict(tensors, dst=dst)


def recv_intermediate_tensors() -> IntermediateTensors:
    """Receive hidden_states/residual from the previous pipeline stage.

    `src` is the GROUP-relative rank of the upstream stage.
    """
    pp = get_pp_group()
    src = (pp.rank_in_group - 1) % pp.world_size
    tensors = pp.recv_tensor_dict(src=src)
    return IntermediateTensors(tensors)


# ---------------------------------------------------------------------------
# Async send — bypasses aiter's synchronous send_tensor_dict, using
# torch.distributed.isend directly.  Wire protocol is identical to aiter's
# send_object + per-tensor send, so the receiver's recv_tensor_dict (which
# uses synchronous recv) works unchanged.
# ---------------------------------------------------------------------------

# Reuse aiter's internal helper that splits a dict into (metadata, tensors).
from aiter.dist.parallel_state import _split_tensor_dict  # noqa: E402


def _async_send_object(
    obj: object, dst_global: int, cpu_group: torch.distributed.ProcessGroup
) -> list[P2PWork]:
    """Async version of aiter GroupCoordinator.send_object."""
    import pickle

    object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
    size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")
    works: list[P2PWork] = []
    w = torch.distributed.isend(size_tensor, dst=dst_global, group=cpu_group)
    works.append(P2PWork(w, size_tensor))
    w = torch.distributed.isend(object_tensor, dst=dst_global, group=cpu_group)
    works.append(P2PWork(w, object_tensor))
    return works


def async_send_intermediate_tensors(
    it: IntermediateTensors,
) -> list[P2PWork]:
    """Non-blocking send of hidden_states/residual to the next PP stage.

    Each tensor is cloned before isend so the model can immediately reuse
    its forward buffers for the next micro-batch.  The returned P2PWork
    list holds references to the cloned tensors (preventing GC) and must
    be committed via ``commit_pp_send_work`` before the *next* async send.
    """
    pp = get_pp_group()
    dst_local = (pp.rank_in_group + 1) % pp.world_size
    dst_global = pp.ranks[dst_local]

    tensors = {k: it.tensors[k] for k in _PP_PROXY_KEYS if k in it.tensors}
    metadata_list, tensor_list = _split_tensor_dict(tensors)

    works = _async_send_object(metadata_list, dst_global, pp.cpu_group)

    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue
        buf = tensor.clone()
        if buf.is_cpu:
            w = torch.distributed.isend(buf, dst=dst_global, group=pp.cpu_group)
        else:
            w = torch.distributed.isend(buf, dst=dst_global, group=pp.device_group)
        works.append(P2PWork(w, buf))

    return works


def commit_pp_send_work(works: list[P2PWork]) -> None:
    """Block until all in-flight isend operations complete, then clear."""
    for p2p in works:
        if p2p.work is not None:
            p2p.work.wait()
    works.clear()
