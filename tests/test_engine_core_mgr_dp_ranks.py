# SPDX-License-Identifier: MIT
# Tests for CoreManager distributed-DP rank assignment helpers.

from types import SimpleNamespace

from atom.model_engine.engine_core_mgr import (
    _distributed_dp_socket_plan,
    _iter_dp_rank_assignments,
)


def _config(
    *,
    data_parallel_size=1,
    data_parallel_size_local=1,
    data_parallel_rank=0,
    tensor_parallel_size=1,
    enable_dp_attention=False,
    distributed_dp=False,
):
    return SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        enable_dp_attention=enable_dp_attention,
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size_local,
            data_parallel_rank=data_parallel_rank,
            distributed_dp=distributed_dp,
        ),
    )


def test_single_node_assigns_all_global_ranks_locally():
    cfg = _config(data_parallel_size=4)

    assert list(_iter_dp_rank_assignments(cfg)) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_distributed_dp_assigns_local_slice_from_global_rank_offset():
    cfg = _config(
        data_parallel_size=16,
        data_parallel_size_local=8,
        data_parallel_rank=8,
        distributed_dp=True,
    )

    assert list(_iter_dp_rank_assignments(cfg)) == [
        (8, 0),
        (9, 1),
        (10, 2),
        (11, 3),
        (12, 4),
        (13, 5),
        (14, 6),
        (15, 7),
    ]


def test_dp_attention_flattens_tensor_parallel_into_global_and_local_dp():
    cfg = _config(
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=1,
        tensor_parallel_size=4,
        enable_dp_attention=True,
        distributed_dp=True,
    )

    assert list(_iter_dp_rank_assignments(cfg)) == [
        (4, 0),
        (5, 1),
        (6, 2),
        (7, 3),
    ]


def test_distributed_socket_plan_derives_ports_from_master_port():
    plan = _distributed_dp_socket_plan(
        data_parallel_size=4,
        master_ip="10.0.0.1",
        master_port=29500,
        bind_host="0.0.0.0",
    )

    assert plan[0].bind_input == "tcp://0.0.0.0:29600"
    assert plan[0].connect_input == "tcp://10.0.0.1:29600"
    assert plan[0].bind_output == "tcp://0.0.0.0:29601"
    assert plan[0].connect_output == "tcp://10.0.0.1:29601"
    assert plan[3].connect_input == "tcp://10.0.0.1:29606"
    assert plan[3].connect_output == "tcp://10.0.0.1:29607"


def test_worker_node_is_not_coordinator_when_rank_offset_nonzero():
    cfg = _config(
        data_parallel_size=16,
        data_parallel_size_local=8,
        data_parallel_rank=8,
        distributed_dp=True,
    )
    cfg.distributed_dp_serving = True

    # This mirrors CoreManager's role predicate without constructing sockets.
    is_coordinator = (
        not (cfg.parallel_config.distributed_dp and cfg.distributed_dp_serving)
        or cfg.parallel_config.data_parallel_rank == 0
    )

    assert is_coordinator is False
