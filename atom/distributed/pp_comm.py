# SPDX-License-Identifier: MIT
# Pipeline-parallel communication and distributed-init helpers.

import logging
import pickle
from dataclasses import dataclass
from typing import Optional

import torch

from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    ensure_model_parallel_initialized,
    _split_tensor_dict,
)
from aiter.ops.communication import set_custom_all_reduce
from atom.models.utils import IntermediateTensors
from atom.utils import envs

logger = logging.getLogger("atom")

# Keys carried between pipeline stages.
_PP_PROXY_KEYS = ("hidden_states", "residual")


def pp_send_allgather_group():
    """TP group for PP send-allgather, or None if disabled/tp=1."""
    if not envs.ATOM_PP_SEND_ALLGATHER:
        return None
    tp = get_tp_group()
    if tp.world_size <= 1:
        return None
    return tp


@dataclass
class P2PWork:
    """Handle for an in-flight isend; prevents payload GC until completion."""

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
    """PP-aware distributed init (aiter's init_dist_env pins pp=1)."""
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
    """Blocking send of hidden_states/residual to the next PP stage."""
    pp = get_pp_group()
    tensors = {k: it.tensors[k] for k in _PP_PROXY_KEYS if k in it.tensors}
    dst = (pp.rank_in_group + 1) % pp.world_size
    pp.send_tensor_dict(tensors, dst=dst, all_gather_group=pp_send_allgather_group())


def recv_intermediate_tensors() -> IntermediateTensors:
    """Receive hidden_states/residual from the previous PP stage."""
    pp = get_pp_group()
    src = (pp.rank_in_group - 1) % pp.world_size
    tensors = pp.recv_tensor_dict(src=src, all_gather_group=pp_send_allgather_group())
    return IntermediateTensors(tensors)


def _async_send_object(
    obj: object, dst_global: int, cpu_group: torch.distributed.ProcessGroup
) -> list[P2PWork]:
    """Async version of aiter GroupCoordinator.send_object."""
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
    """Non-blocking send of hidden_states/residual to the next PP stage."""
    pp = get_pp_group()
    dst_local = (pp.rank_in_group + 1) % pp.world_size
    dst_global = pp.ranks[dst_local]

    tensors = {k: it.tensors[k] for k in _PP_PROXY_KEYS if k in it.tensors}
    metadata_list, tensor_list = _split_tensor_dict(tensors)

    works = _async_send_object(metadata_list, dst_global, pp.cpu_group)

    ag_group = pp_send_allgather_group()
    ag_size = 1 if ag_group is None else ag_group.world_size
    ag_rank = 0 if ag_group is None else ag_group.rank_in_group

    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue
        if ag_group is not None and tensor.numel() % ag_size == 0:
            tensor = tensor.reshape(ag_size, -1)[ag_rank]
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
