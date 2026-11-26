"""CPU red/green checks for the real backported port and write-done methods."""

import ast
import collections
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


def methods(path, names):
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            node.decorator_list = []
            found.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *found,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {
        "logger": SimpleNamespace(debug=lambda *args: None),
        "ROLE": SimpleNamespace(PRODUCER="P"),
        "get_role": lambda: "D",
        "_MAX_TERMINAL_TRANSFER_IDS": 4096,
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope


def duplicate_finishes_early(root):
    scope = methods(
        root / "moriio_engine.py",
        {
            "_handle_write_done_message",
            "_is_transfer_terminal_locked",
            "_mark_transfer_terminal_locked",
        },
    )
    cls = type(
        "ActualMethods",
        (),
        {
            name: value
            for name, value in scope.items()
            if name.startswith("_") and callable(value)
        },
    )
    obj = cls()
    obj.lock = threading.Lock()
    obj._write_done_counts = {}
    obj._write_done_senders = {}
    obj._terminal_transfer_ids = collections.OrderedDict()
    obj.done_write_cache_req_ids = []
    message = {"transfer_id": "test", "num_writers": 4, "sender": [0, 0]}
    for _ in range(4):
        obj._handle_write_done_message(message)
    return obj


def main():
    before, after = map(Path, sys.argv[1:])
    assert duplicate_finishes_early(before).done_write_cache_req_ids == ["test"]
    obj = duplicate_finishes_early(after)
    assert obj.done_write_cache_req_ids == []
    for rank in (1, 2, 3):
        obj._handle_write_done_message(
            {"transfer_id": "test", "num_writers": 4, "sender": [rank, 0]}
        )
    assert obj.done_write_cache_req_ids == ["test"]
    obj._handle_write_done_message(
        {"transfer_id": "test", "num_writers": 4, "sender": [3, 0]}
    )
    assert obj.done_write_cache_req_ids == ["test"]
    scope = methods(
        after / "moriio_common.py",
        {"get_port_offset", "get_pp_port_stride", "later_stage_notify_ports"},
    )
    offset = scope["get_port_offset"]
    for pp, tp, dp in ((4, 1, 1), (1, 1, 4), (2, 2, 2)):
        stride = scope["get_pp_port_stride"](tp, dp)
        ports = [
            offset(d, t, tp, pp_rank=p, pp_stride=stride)
            for p in range(pp)
            for d in range(dp)
            for t in range(tp)
        ]
        assert len(set(ports)) == pp * tp * dp
        assert scope["later_stage_notify_ports"](
            SimpleNamespace(
                pp_rank=0, notify_port=61000, pp_size=pp, pp_port_stride=stride
            )
        ) == [61000 + p * stride for p in range(1, pp)]
    print(
        "PASS: old duplicate completion reproduced; patched four-stage fan-in and unique PP/DP ports verified"
    )


if __name__ == "__main__":
    main()
