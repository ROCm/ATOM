"""Exercise actual worker notification methods for asymmetric P/D DP pools."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(sys.argv[1])
names = {
    "update_state_after_alloc",
    "send_notify_block",
    "get_port_offset",
    "fold_local_rank",
    "pod_index",
    "_resolve_notify_endpoint",
}
body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
]
for name in ("moriio_connector.py", "moriio_common.py", "moriio_engine.py"):
    for node in ast.walk(ast.parse((root / name).read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
sent = []
scope = {
    "MoRIIOMode": SimpleNamespace(READ="read", WRITE="write"),
    "get_moriio_peer_tp_size": lambda params, tp: params.get("remote_tp_size", tp),
    "get_peer_zmq_from_request_id": lambda *args, **kwargs: None,
    "make_zmq_path": lambda proto, host, port: f"{proto}://{host}:{port}",
    "make_zmq_socket": lambda **kwargs: SimpleNamespace(
        send=lambda data: sent.append(data)
    ),
    "zmq": SimpleNamespace(Context=SimpleNamespace(instance=lambda: None), DEALER=5),
    "msgpack": SimpleNamespace(dumps=lambda data: data),
    "logger": SimpleNamespace(debug=lambda *args: None, warning=lambda *args: None),
}
exec(compile(module, "actual-moriio-notify-methods", "exec"), scope)
for rank in range(4):
    for override in (True, False):
        sent.clear()
        worker = SimpleNamespace(
            is_producer=False,
            _has_mamba=False,
            _is_hma_required=False,
            _is_kv_master=True,
            _global_dp_rank=rank,
            tp_size=1,
            mode="write",
            paths={},
            map_request_id=lambda *args: None,
        )
        from types import MethodType

        worker.send_notify_block = MethodType(scope["send_notify_block"], worker)
        params = {
            "do_remote_prefill": True,
            "transfer_id": "tx-test",
            "remote_host": "p",
            "remote_handshake_port": 6300,
            "remote_notify_port": 6400,
            "remote_engine_id": "p:6300",
            "remote_block_ids": None,
            "remote_dp_size": 1,
            "remote_dp_size_local": 1,
            "remote_dp_rank": 0,
            "remote_dp_rank_override": override,
            "remote_tp_size": 1,
        }
        scope["update_state_after_alloc"](
            worker,
            SimpleNamespace(request_id="tx-test", kv_transfer_params=params),
            SimpleNamespace(get_block_ids=lambda: [[11, 12]]),
            16,
        )
        assert len(sent) == (int(rank == 0) if override else 1)
        if sent:
            assert sent[0]["decode_rank"] == rank
writer = SimpleNamespace(worker=SimpleNamespace(tp_rank=0))
task = SimpleNamespace(
    remote_ip="d",
    multi_pod_hosts=["d"],
    remote_dp_size_local=4,
    remote_notify_port=6500,
)
for rank in range(4):
    assert scope["_resolve_notify_endpoint"](writer, task, rank) == ("d", 6500 + rank)
print("PASS: D ranks0..3 notify P0 and receive completion on their actual rank ports")
