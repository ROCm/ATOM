"""Retain R3 preloads and cover native SWA cached-pointer/scalar classes."""

import functools
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_performance_r3_source.step12_model_preload_r3 import preload_model as prior
from step12_swa_preload_r2 import preload as preload_swa


def preload_model(runner):
    from aiter.dist.parallel_state import get_tp_group

    start = time.perf_counter()
    prior(runner)
    path = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"]) / (
        f"kernel_preload_rank{get_tp_group().rank_in_group}.json"
    )
    report = json.loads(path.read_text())
    report["status"] = "preloading_native_swa_domain"
    write_json(path, report)
    try:
        geometry = runner.attn_metadata_builder.geometry
        window = geometry.window(0, 1)
        assert not geometry.packed and geometry.head_dim == 512
        report["native_swa_geometry"] = asdict(window)
        write_json(path, report)
        report["native_swa"] = preload_swa(window)
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
