"""Common formal preload: native linear/router, BF16 families and late Triton."""

import functools
import json
import os
import time
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_bf16_preload import preload_model as preload_bf16
from step12_cold_cache_r2_source.step12_cold_preload import preload_triton
from step12_model_preload_r6 import preload_model as preload_linear_router


def preload_model(runner):
    from aiter.dist.parallel_state import get_tp_group

    start = time.perf_counter()
    preload_linear_router(runner)
    path = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"]) / (
        f"kernel_preload_rank{get_tp_group().rank_in_group}.json"
    )
    report = json.loads(path.read_text())
    report["status"] = "preloading_bf16_and_late_triton"
    write_json(path, report)
    try:
        report["native_bf16"] = preload_bf16(runner)
        report["late_triton"] = preload_triton()
        report.update(status="PASS", preload_seconds=time.perf_counter() - start)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(path, report)


def install():
    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        preload_model(self)
        return original(self)

    ModelRunner._maybe_warmup = warmup
