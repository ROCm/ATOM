"""Actual config method + OS sockets: stale open-port probes versus bind(0)."""
import ast
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace as NS


def config_factory(path, picker):
    names = {"from_vllm_config", "fold_local_rank", "get_pp_port_stride", "get_port_offset", "_positive_finite_timeout"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            body.append(node)
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    scope = {
        "math": math, "os": os, "get_open_port": picker,
        "_warn_deprecated_env_vars": lambda: None,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "get_pipeline_model_parallel_rank": lambda: 0,
        "resolve_host_ip": lambda extra: extra["host_ip"],
        "get_moriio_mode": lambda config: "write", "MoRIIOMode": NS(READ="read"),
        "MoRIIOConstants": NS(DEFAULT_TRANSFER_TIMEOUT=60, DEFAULT_DEFER_TIMEOUT=60, DEFAULT_RECV_ABORT_TIMEOUT=60),
    }
    exec(compile(tree, str(path), "exec"), scope)
    config = NS(
        parallel_config=NS(data_parallel_rank=0, data_parallel_size_local=1, data_parallel_size=1, pipeline_parallel_size=4),
        kv_transfer_config=NS(kv_connector_extra_config={"host_ip":"127.0.0.1", "http_port":6100, "handshake_port":26000, "notify_port":27000}),
    )
    return lambda: scope["from_vllm_config"](NS, config)


def open_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    before, after = map(Path, sys.argv[1:])
    # A listener starts between the config's free-port probe and native bind.
    requested = open_port()
    old = config_factory(before, lambda: requested)()
    with socket.socket() as contender, socket.socket() as native:
        contender.bind(("127.0.0.1", requested))
        contender.listen()
        try:
            native.bind((old.local_ip, old.local_kv_port))
        except OSError:
            pass
        else:
            raise AssertionError("The old stale port must collide with the contender")

    sockets, ports = [], []
    try:
        candidate = config_factory(after, lambda: requested)
        for _ in range(8):
            actual = candidate()
            s = socket.socket()
            sockets.append(s)
            s.bind((actual.local_ip, actual.local_kv_port))
            s.listen()
            ports.append(s.getsockname()[1])
        assert len(set(ports)) == 8 and all(0 < p < 65536 for p in ports)
        assert actual.handshake_port == 26000 and actual.notify_port == 27000
    finally:
        for s in sockets:
            s.close()
    print("PASS: actual config's stale port can collide; bind(0) keeps eight live endpoints distinct")


if __name__ == "__main__":
    main()
