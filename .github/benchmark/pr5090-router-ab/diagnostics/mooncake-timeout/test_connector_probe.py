"""Execute the real pinned _send_blocks AST with the bounded diagnostic added."""

import ast
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import atomesh_timeout_probe as probe
from prepare_worker import instrument


def main():
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    changed = instrument(source)
    tree = ast.parse(changed)
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_send_blocks"
    )
    module = ast.Module(body=[method], type_ignores=[])
    import time

    ns = {
        "time": time,
        "logger": types.SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
    }
    exec(compile(module, "pinned-_send_blocks", "exec"), ns)  # noqa: S102 - exact source hash checked by instrument
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ATOMESH_MOONCAKE_PROBE_DIR"] = tmp
        stats = {"success": 0, "failure": 0}
        calls = []
        worker = types.SimpleNamespace(
            is_kv_producer=True,
            pp_rank=3,
            hostname="localhost",
            rpc_port=123,
            device_id=3,
            num_sender_workers=10,
            num_sender_tasks=20,
            seen_base_addresses=[1000, 1100],
            _kv_data_lens=[100, 100],
            _kv_unregistered=False,
        )

        def send(*args):
            calls.append(args)
            return 0 if args[0] == "success" else -1

        worker.engine = types.SimpleNamespace(batch_transfer_sync_write=send)
        worker.xfer_stats = types.SimpleNamespace(
            record_transfer=lambda **kwargs: stats.update(success=stats["success"] + 1),
            record_failed_transfer=lambda: stats.update(failure=stats["failure"] + 1),
        )
        args = ([1050, 1190], [2000, 3000], [100, 20])
        assert ns["_send_blocks"](worker, "success", *args) == 0
        assert ns["_send_blocks"](worker, "failed", *args) == -1
        probe.record_registration(worker, "register")
        records = [
            json.loads(line)
            for line in next(Path(tmp).glob("connector-*.jsonl"))
            .read_text()
            .splitlines()
        ]
        assert len(records) == 2 and records[0]["source_outside_ranges"] == [1]
        assert records[0]["descriptor_count"] == 2 and records[0]["total_bytes"] == 120
        assert (
            records[0]["sender_workers"] == 10
            and records[0]["async_sender_tasks"] == 20
        )
        assert stats == {"success": 1, "failure": 1} and len(calls) == 2
        assert calls[0][1:] == calls[1][1:] == args
        assert probe.covered(1050, 150, [(1000, 100), (1100, 100)])
        assert not probe.covered(1050, 151, [(1000, 100), (1100, 100)])
        assert not probe.covered(999, 1, [(1000, 200)])
        # Broken output storage must not turn a failed write into a different result.
        os.environ["ATOMESH_MOONCAKE_PROBE_DIR"] = str(Path(tmp) / "missing")
        assert ns["_send_blocks"](worker, "failed", *args) == -1
    try:
        instrument(source + "\n")
    except ValueError:
        pass
    else:
        raise AssertionError("Different connector accepted")
    print(
        json.dumps(
            {
                "real_send_blocks_ast": True,
                "source_guard": True,
                "return_and_args_preserved": True,
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
