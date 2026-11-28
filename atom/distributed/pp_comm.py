# SPDX-License-Identifier: MIT
# Pipeline-parallel communication and distributed-init helpers.

import logging
import pickle
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)
from aiter.ops.communication import set_custom_all_reduce

from atom.models.utils import IntermediateTensors
from atom.utils import envs

logger = logging.getLogger("atom")

# Keys carried between pipeline stages. `sparse_kv_indices` is optional: present
# only when a PP boundary splits a DSA IndexShare group (see model_runner).
# `block_residual` is Kimi-K3's written AttnRes prefix [T, closed_blocks, H].
# `PP_AUX_KEY` relays a drafter's target aux hidden states (`PPAuxRelay`,
# set/popped by model_runner). Keys missing here are silently dropped by the send.
PP_AUX_KEY = "pp_aux"

_PP_PROXY_KEYS = (
    "hidden_states",
    "residual",
    "block_residual",
    "sparse_kv_indices",
    PP_AUX_KEY,
)


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

    work: torch.distributed.Work | None
    payload: torch.Tensor | None


def init_pp_aware_dist_env(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    global_rank: int,
    world_size: int,
    distributed_init_method: str,
    backend: str = "nccl",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    disable_custom_all_reduce: bool = False,
    device_communicators: bool = True,
) -> None:
    """PP-aware distributed init (aiter's init_dist_env pins pp=1)."""
    set_custom_all_reduce(not disable_custom_all_reduce)
    init_distributed_environment(
        world_size=world_size,
        rank=global_rank,
        distributed_init_method=distributed_init_method,
        backend=backend,
        local_rank=local_rank,
        data_parallel_size=1,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        decode_context_model_parallel_size=decode_context_parallel_size,
        data_parallel_size=data_parallel_size,
        prefill_context_model_parallel_size=prefill_context_model_parallel_size,
        device_communicators=device_communicators,
    )

    # No per-rank signal/input-buffer registration here (this used to copy
    # aiter init_dist_env's signal/buffer block). All of it was vestigial --
    # ca_comm.signal / ca_comm.buffer are never read, and the registration
    # entry is keyed by the signal tensor's own address, which no allreduce
    # ever uses as input -- and it made raw IPC input pools unusable
    # (ROCm/aiter#4921): under PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    # the torch.zeros signal is VMM-backed so hipIpcGetMemHandle rejects it,
    # and the raw_cached input pool has no backing tensor so the
    # `_pool["input"].tensor` mirror raised. CustomAllreduce.__init__ builds
    # its own pools and copy-in path; nothing further is needed.

    logger.debug(
        "init_pp_aware_dist_env: global_rank=%d tp=%d pp=%d pcp=%d dp=%d world=%d",
        global_rank,
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        data_parallel_size,
        world_size,
    )


# A frame's metadata is one fixed-size gloo message on a tag of its own. Gloo
# delivers a message only once its receive is posted, so the receiving stage
# posts the next frame's metadata receive as soon as it has this one's: the
# message lands while the frame is computed, and asking for the frame costs no
# gloo round trip before its RCCL receives are posted.
_META_TAG = 0x50504D44
_META_BYTES = 4096


def _on_device_group(device: str) -> bool:
    """Whether a tensor on this device type goes over the device group rather
    than the CPU group."""
    return device != "cpu"


def _meta_message(entries: list) -> torch.Tensor:
    raw = pickle.dumps(entries)
    if len(raw) > _META_BYTES - 8:
        raise ValueError(
            f"PP frame metadata takes {len(raw)} bytes, over the "
            f"{_META_BYTES - 8} its message holds: {entries}"
        )
    # On the CPU whatever the default device: the model runner sets one.
    message = torch.zeros(_META_BYTES, dtype=torch.uint8, device="cpu")
    message[:8] = torch.tensor([len(raw)], dtype=torch.int64, device="cpu").view(
        torch.uint8
    )
    message[8 : 8 + len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return message


def _read_meta(message: torch.Tensor) -> list:
    size = int(message[:8].view(torch.int64)[0])
    return pickle.loads(message[8 : 8 + size].numpy().tobytes())


@dataclass
class _MetaRecv:
    """A posted receive of a frame's metadata message."""

    work: torch.distributed.Work
    message: torch.Tensor
    source: tuple  # (cpu_group, src global rank) it was posted for


_next_meta: _MetaRecv | None = None


def _post_meta_recv(cpu_group, src: int) -> _MetaRecv:
    message = torch.empty(_META_BYTES, dtype=torch.uint8, device="cpu")
    work = torch.distributed.irecv(message, src=src, group=cpu_group, tag=_META_TAG)
    return _MetaRecv(work, message, (cpu_group, src))


def send_intermediate_tensors(it: IntermediateTensors) -> None:
    """Blocking send of hidden_states/residual to the next PP stage."""
    commit_pp_send_work(async_send_intermediate_tensors(it))


def recv_intermediate_tensors() -> IntermediateTensors:
    """Receive hidden_states/residual from the previous PP stage."""
    global _next_meta
    pp = get_pp_group()
    src = pp.ranks[(pp.rank_in_group - 1) % pp.world_size]
    pending = _next_meta
    if pending is None or pending.source != (pp.cpu_group, src):
        pending = _post_meta_recv(pp.cpu_group, src)
    _next_meta = None
    pending.work.wait()
    entries = _read_meta(pending.message)
    _next_meta = _post_meta_recv(pp.cpu_group, src)

    ag_group = pp_send_allgather_group()
    tensors = {}
    for key, slot, value in entries:
        if slot is not None:
            device, dtype, shape = slot
            value = torch.empty(shape, dtype=dtype, device=device)
            if value.numel():
                gather = ag_group
                if gather is not None and value.numel() % gather.world_size:
                    gather = None
                part = value
                if gather is not None:
                    part = value.reshape(gather.world_size, -1)[gather.rank_in_group]
                group = pp.device_group if _on_device_group(device) else pp.cpu_group
                torch.distributed.recv(part, src=src, group=group)
                if gather is not None:
                    value = gather.all_gather(part, dim=0).reshape(shape)
        tensors[key] = value
    return IntermediateTensors(tensors)


def async_send_intermediate_tensors(
    it: IntermediateTensors, owned: tuple[str, ...] = ()
) -> list[P2PWork]:
    """Non-blocking send of hidden_states/residual to the next PP stage.

    ``owned`` keys name tensors the caller hands over and never touches
    again: they are sent as they are. The rest are cloned, since the model
    may overwrite them (a graph's output buffer, say) before the send lands.
    """
    pp = get_pp_group()
    dst_local = (pp.rank_in_group + 1) % pp.world_size
    dst_global = pp.ranks[dst_local]

    tensors = {k: it.tensors[k] for k in _PP_PROXY_KEYS if k in it.tensors}
    # (key, slot, value): a tensor's slot is its (device type, dtype, shape);
    # anything else has none and travels as the value itself.
    entries = [
        (
            (key, (value.device.type, value.dtype, tuple(value.shape)), None)
            if isinstance(value, torch.Tensor)
            else (key, None, value)
        )
        for key, value in tensors.items()
    ]
    message = _meta_message(entries)
    works = [
        P2PWork(
            torch.distributed.isend(
                message, dst=dst_global, group=pp.cpu_group, tag=_META_TAG
            ),
            message,
        )
    ]

    ag_group = pp_send_allgather_group()
    ag_size = 1 if ag_group is None else ag_group.world_size
    ag_rank = 0 if ag_group is None else ag_group.rank_in_group

    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if ag_group is not None and tensor.numel() % ag_size == 0:
            tensor = tensor.reshape(ag_size, -1)[ag_rank]
        buf = tensor if key in owned else tensor.clone()
        on_device = _on_device_group(tensor.device.type)
        group = pp.device_group if on_device else pp.cpu_group
        w = torch.distributed.isend(buf, dst=dst_global, group=group)
        works.append(P2PWork(w, buf))

    return works


def commit_pp_send_work(works: list[P2PWork]) -> None:
    """Block until all in-flight isend operations complete, then clear."""
    for p2p in works:
        if p2p.work is not None:
            p2p.work.wait()
    works.clear()
