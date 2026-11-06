"""Extend the frozen linear preload with the native router's row domain."""

import functools
import json
import os
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_model_preload import preload_model as preload_linear
from step12_router_preload import preload_router


def preload_model(runner):
    from aiter.dist.parallel_state import get_tp_group

    preload_linear(runner)
    path = (
        Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
        / f"kernel_preload_rank{get_tp_group().rank_in_group}.json"
    )
    report = json.loads(path.read_text())
    report["status"] = "preloading_native_router"
    write_json(path, report)
    try:
        report["native_router"] = preload_router()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        write_json(path, report)
        raise
    report["status"] = "PASS"
    write_json(path, report)


def install():
    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        preload_model(self)
        return original(self)

    ModelRunner._maybe_warmup = warmup
