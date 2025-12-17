# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import ipaddress
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                _unregister_process_group)
from torch.distributed import ProcessGroup
from torch.distributed.rendezvous import rendezvous
import torch
from datetime import timedelta

from atom.utils import is_torch_equal_or_newer


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def init_gloo_process_group(backend: Backend, prefix_store: PrefixStore,
                            group_rank: int, group_size: int,
                            timeout: timedelta) -> ProcessGroup:
    """
    Stateless init ProcessGroup with gloo backend compatible with 
    different torch versions.
    """
    if is_torch_equal_or_newer("2.6"):
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
    else:
        options = ProcessGroup.Options(backend=backend)
        pg = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
            options,
        )
    from torch.distributed.distributed_c10d import ProcessGroupGloo
    backend_class = ProcessGroupGloo(prefix_store,
                                     group_rank,
                                     group_size,
                                     timeout=timeout)
    backend_type = ProcessGroup.BackendType.GLOO
    device = torch.device("cpu")
    if is_torch_equal_or_newer("2.6"):
        # _set_default_backend is supported in torch >= 2.6
        pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)
    return pg


def stateless_init_torch_distributed_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.
    """
    init_method = get_tcp_uri(host, port)
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)
    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)
    if backend == "gloo":
        return init_gloo_process_group(backend=backend,
                                       prefix_store=prefix_store,
                                       group_rank=group_rank,
                                       group_size=group_size,
                                       timeout=timeout)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def stateless_destroy_torch_distributed_process_group(
        pg: ProcessGroup) -> None:
    """
    Destroy ProcessGroup returned by
        stateless_init_torch_distributed_process_group().
    """
    if is_torch_equal_or_newer("2.7"):
        pg.shutdown()
    else:
        # Lazy import for non-CUDA backends.
        from torch.distributed.distributed_c10d import _shutdown_backend
        _shutdown_backend(pg)

    _unregister_process_group(pg.group_name)