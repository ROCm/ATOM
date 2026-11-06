"""Add the GPU-proven native index domain after the unchanged native warmup."""

import functools
import json
import os
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from step12_indices_preload import preload


def install():
    from aiter.dist.parallel_state import get_tp_group

    from atom.model_engine.model_runner import ModelRunner

    original = ModelRunner._maybe_warmup

    @functools.wraps(original)
    def warmup(self):
        result = original(self)
        geometry = self.attn_metadata_builder.geometry
        rows = preload(geometry)
        path = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"]) / (
            f"kernel_preload_rank{get_tp_group().rank_in_group}.json"
        )
        report = json.loads(path.read_text())
        report["native_indices"] = rows
        report["native_indices_speculative_tokens"] = geometry.speculative_tokens
        write_json(path, report)
        return result

    ModelRunner._maybe_warmup = warmup
