# SPDX-License-Identifier: MIT
"""Run the real setup functions while recording only device/process boundaries."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _source_function(relative, name, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - Execute only these trusted local setup functions.
        compile(ast.fix_missing_locations(module), str(path), "exec"), namespace
    )
    return namespace[name]


def _initialize(pp, dcp, pp_rank=0):
    calls = []
    logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    pp_namespace = {
        "logger": logger,
        "set_custom_all_reduce": lambda enabled: None,
        "init_distributed_environment": lambda **kwargs: None,
        "ensure_model_parallel_initialized": lambda *args, **kwargs: calls.append(
            ("pp", kwargs["decode_context_model_parallel_size"])
        ),
    }
    pp_init = _source_function(
        "atom/distributed/pp_comm.py", "init_pp_aware_dist_env", pp_namespace
    )

    def source_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "atom.distributed.pp_comm":
            return SimpleNamespace(init_pp_aware_dist_env=pp_init)
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {
        "__builtins__": dict(vars(builtins), __import__=source_import),
        "logger": logger,
        "os": SimpleNamespace(environ={}),
        "torch": SimpleNamespace(
            device=lambda name: name,
            cuda=SimpleNamespace(
                device_count=lambda: 8, set_device=lambda device: None
            ),
        ),
        "set_current_atom_config": lambda config: None,
        "sp_is_enabled": lambda: False,
        "reject_simulated_tp": lambda *args: None,
        "local_device_index": lambda config, rank: pp_rank * 4 + rank,
        "get_distributed_init_method": lambda *args: "recorded-rendezvous",
        "raise_timeout_if_dp_spans_nodes": lambda config: None,
        "raise_timeout_if_tp_spans_nodes": lambda config: None,
        "apply_simulated_tp": lambda config: None,
        "init_dist_env": lambda *args, **kwargs: calls.append(
            ("non_pp", kwargs["decode_context_parallel_size"])
        ),
    }
    init = _source_function(
        "atom/model_engine/model_setup.py", "init_model_env", namespace
    )
    config = SimpleNamespace(
        tensor_parallel_size=4,
        tp_world_size=4,
        pcp_group_size=1,
        pipeline_parallel_size=pp,
        decode_context_parallel_size=dcp,
        master_addr="127.0.0.1",
        port=0,
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=0,
            data_parallel_rank=0,
            data_parallel_size=1,
            pipeline_parallel_rank=pp_rank,
            decode_context_parallel_size=1,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=0,
            _managed_distributed_store=False,
        ),
    )
    init(config, rank=0)
    return calls


@pytest.mark.parametrize(
    "pp,dcp,pp_rank,route",
    [(2, 4, 0, "pp"), (2, 4, 1, "pp"), (2, 1, 0, "pp"), (1, 4, 0, "non_pp")],
    ids=["pp-dcp-first", "pp-dcp-last", "pp-default-dcp", "non-pp-dcp"],
)
def test_model_initialization_preserves_dcp_width(pp, dcp, pp_rank, route):
    assert _initialize(pp, dcp, pp_rank) == [(route, dcp)]
