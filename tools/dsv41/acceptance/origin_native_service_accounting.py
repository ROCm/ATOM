"""Observe origin/main's own FULL request graphs without installing optimizations."""

import collections
import hashlib
import json
import os
from functools import wraps
from pathlib import Path


def install():
    from aiter.dist.parallel_state import get_tp_group

    import atom
    from atom.model_engine.model_runner import ModelRunner
    from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel
    from atom.utils import backends

    source = Path(os.environ["DSV41_ORIGIN_ATOM_SOURCE"])
    output = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
    if not Path(atom.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Use the frozen origin/main source")
    for key, value in (
        ("DSV41_OWNER_SERVING", "0"),
        ("DSV41_OWNER_PACKET_PUBLICATION", "0"),
        ("ATOM_DSV41_ROW_SHARDED_MHC", "0"),
        ("ATOM_DSV41_PREFILL_TAIL_MODE", "off"),
    ):
        if os.environ.get(key) != value:
            raise RuntimeError("An optimization leaked into origin/main: " + key)
    backends.VLLM_CACHE_ROOT = os.environ["DSV41_OWNER_SERVING_COMPILE_CACHE"]
    counts = collections.Counter()
    live = False

    def save():
        raw = (
            json.dumps({"mode": "origin_main", "counts": dict(counts)}) + "\n"
        ).encode()
        if len(raw) > 128_000:
            raise RuntimeError("Native route receipt exceeds its bound")
        path = output / f"route_rank{get_tp_group().rank_in_group}.json"
        staging = path.with_suffix(".tmp")
        staging.write_bytes(raw)
        staging.replace(path)

    original_capture = ModelRunner.capture_cudagraph
    original_forward = ModelRunner.forward
    original_replayed = DeepseekV41RuntimeModel._replayed

    @wraps(original_capture)
    def capture(self):
        if self.world_size != 4 or hasattr(self, "drafter") or self._tr_hybrid_active():
            raise RuntimeError("Require native origin/main TP4/no-spec/FULL")
        result = original_capture(self)
        graph = self.req_graph
        original_replay = graph.replay

        @wraps(original_replay)
        def replay(*args, **kwargs):
            value = original_replay(*args, **kwargs)
            if live:
                counts["normal_FULL_replays"] += 1
            return value

        graph.replay = replay
        evidence = {
            "status": "origin_main_native_FULL_captured",
            "source": str(source),
            "runtime_sha256": hashlib.sha256(
                (source / "atom/models/deepseek_v41/runtime.py").read_bytes()
            ).hexdigest(),
            "graph_keys": graph.keys(),
            "private_owner_installed": False,
            "private_prefill_enabled": False,
            "scope": "Original ReqGraph replay, native prefill and arithmetic; host counters only",
        }
        (output / f"origin_capture_rank{get_tp_group().rank_in_group}.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        return result

    @wraps(original_replayed)
    def replayed(self, state, running_tokens, plan):
        if live:
            counts["native_prefill_replay_forwards"] += 1
            counts["native_prefill_replay_rows"] += running_tokens
        return original_replayed(self, state, running_tokens, plan)

    @wraps(original_forward)
    def forward(self, batch):
        nonlocal live
        previous = live
        live = not batch.is_dummy_run
        try:
            result = original_forward(self, batch)
            if live:
                counts["native_target_steps"] += 1
                counts["native_prefill_steps"] += int(batch.total_seqs_num_prefill > 0)
                if counts["native_target_steps"] % 64 == 0:
                    save()
            return result
        finally:
            live = previous

    ModelRunner.capture_cudagraph = capture
    ModelRunner.forward = forward
    DeepseekV41RuntimeModel._replayed = replayed
