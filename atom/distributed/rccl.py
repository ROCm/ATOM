# SPDX-License-Identifier: MIT
"""Model-parallel initialization with RCCL reductions on every TP group."""

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)


def init_rccl_dist_env(
    tensor_model_parallel_size,
    rankID,
    backend="nccl",
    distributed_init_method="env://",
    local_rank=-1,
    data_parallel_size=1,
    data_parallel_rank=0,
    decode_context_parallel_size=1,
    prefill_context_model_parallel_size=1,
):
    # AITER's convenience init_dist_env unconditionally re-enables its custom
    # path. Select the supported policy before either group is constructed.
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=tensor_model_parallel_size * prefill_context_model_parallel_size,
        rank=rankID,
        backend=backend,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        1,
        decode_context_model_parallel_size=decode_context_parallel_size,
        data_parallel_size=data_parallel_size,
        prefill_context_model_parallel_size=prefill_context_model_parallel_size,
    )
