"""Bounded host counters inside the real guarded prefill FFN consumer."""

import copy
import functools
import os


class RouteProof:
    def __init__(self, audit, enabled):
        self.audit = audit
        self.enabled = enabled
        self.layers = None
        self.contracts = []
        self.counts = {"warmup": {}, "sealed": {}}
        self.combine_counts = {"warmup": {}, "sealed": {}}
        self.active_layer = None

    def combine(self, rows, fused):
        with self.audit.lock:
            phase = "sealed" if self.audit.sealed else "warmup"
            key = f"{self.active_layer}:{'fused' if fused else 'native'}"
            row = self.combine_counts[phase].setdefault(
                key, {"calls": 0, "rows": 0, "min_rows": rows, "max_rows": rows}
            )
            row["calls"] += 1
            row["rows"] += rows
            row["min_rows"] = min(row["min_rows"], rows)
            row["max_rows"] = max(row["max_rows"], rows)

    def record(self, layer, rows):
        if not self.enabled or self.layers is None or layer not in self.layers:
            raise RuntimeError("Prefill FFN consumer outside validated layer contract")
        if not 4096 <= rows <= 16384:
            raise RuntimeError("Unexpected optimized prefill row domain")
        with self.audit.lock:
            phase = "sealed" if self.audit.sealed else "warmup"
            row = self.counts[phase].setdefault(
                str(layer), {"calls": 0, "rows": 0, "min_rows": rows, "max_rows": rows}
            )
            row["calls"] += 1
            row["rows"] += rows
            row["min_rows"] = min(row["min_rows"], rows)
            row["max_rows"] = max(row["max_rows"], rows)

    def snapshot(self):
        with self.audit.lock:
            return copy.deepcopy(
                {
                    "enabled": self.enabled,
                    "expected_layers": self.layers,
                    "contracts": self.contracts,
                    "counts": self.counts,
                    "combine_counts": self.combine_counts,
                    "scope": "Completed actual guarded prequantized FFN calls; host counters only, no kernel timing or decode occupancy claim.",
                }
            )


def validate_interval(before, after, enabled):
    assert before["enabled"] is after["enabled"] is enabled
    assert before["expected_layers"] == after["expected_layers"]
    assert before["contracts"] == after["contracts"]
    assert before["counts"]["warmup"] == after["counts"]["warmup"]
    layers = after["expected_layers"]
    assert layers is not None
    if not enabled:
        assert not layers and not after["contracts"]
        assert not any(before["counts"].values()) and not any(after["counts"].values())
        return {}
    assert layers and [r["layer"] for r in after["contracts"]] == layers
    assert set(after["counts"]["sealed"]) == {str(i) for i in layers}
    assert set(before["counts"]["warmup"]) == {str(i) for i in layers}
    deltas = {}
    for i in layers:
        key = str(i)
        old = before["counts"]["sealed"].get(key, {"calls": 0, "rows": 0})
        new = after["counts"]["sealed"][key]
        assert 4096 <= new["min_rows"] <= new["max_rows"] <= 16384
        delta = {k: new[k] - old[k] for k in ("calls", "rows")}
        assert delta["calls"] > 0 and delta["rows"] >= 4096 * delta["calls"]
        deltas[key] = delta
    assert len({(r["calls"], r["rows"]) for r in deltas.values()}) == 1
    return deltas


def bind(Runner, consumer, enabled, configure):
    audit = Runner._rebase_graph_audit
    if hasattr(Runner, "_step12_prefill_ffn_route"):
        raise RuntimeError("Prefill route already installed")
    proof = RouteProof(audit, enabled)
    Runner._step12_prefill_ffn_route = proof
    original_snapshot = audit.snapshot

    def snapshot(*args, **kwargs):
        with audit.lock:
            report = original_snapshot(*args, **kwargs)
            if report is not None:
                report["prefill_quantized_ffn"] = proof.snapshot()
            return report

    audit.snapshot = snapshot
    original_warmup = Runner._maybe_warmup

    @functools.wraps(original_warmup)
    def warmup(self):
        if proof.layers is not None:
            raise RuntimeError("Prefill layer contract must bind once per worker")
        proof.layers, proof.contracts = configure(self, enabled)
        return original_warmup(self)

    Runner._maybe_warmup = warmup
    if enabled:
        original_consumer = consumer.prequantized_owned_ffn
        # Both callbacks run inside the existing guarded FFN, outside Dynamo.
        # Count the actual exit call; graph replay does not re-enter Python.
        combined = hasattr(consumer, "v41_ffn_combine_scatter")
        if combined:
            from atom.model_ops.deepseek_v41.ffn_combine import supports_fused_combine

            original_exit = consumer.v41_ffn_combine_scatter

            @functools.wraps(original_exit)
            def exit_call(routed, shared):
                result = original_exit(routed, shared)
                proof.combine(
                    int(routed.shape[0]), supports_fused_combine(routed, shared, 4)
                )
                return result

            consumer.v41_ffn_combine_scatter = exit_call
        else:
            original_exit = consumer.v41_mhc_reduce_scatter

            @functools.wraps(original_exit)
            def exit_call(value):
                result = original_exit(value)
                proof.combine(int(value.shape[0]), False)
                return result

            consumer.v41_mhc_reduce_scatter = exit_call

        @functools.wraps(original_consumer)
        def observed(moe, local, rows):
            previous = proof.active_layer
            proof.active_layer = moe.layer_id
            try:
                result = original_consumer(moe, local, rows)
            finally:
                proof.active_layer = previous
            proof.record(moe.layer_id, int(rows))
            return result

        consumer.prequantized_owned_ffn = observed
    return proof


def configure(runner, enabled):
    from atom.models.deepseek_v41.bounded_replay import late_layer_start

    model = runner.model
    while not hasattr(model, "layers") and hasattr(model, "model"):
        model = model.model
    actual = [
        i
        for i, layer in enumerate(model.layers)
        if getattr(layer, "prequantized_prefill_ffn", False)
    ]
    expected = (
        list(range(2, late_layer_start(runner.config.hf_config) - 1)) if enabled else []
    )
    assert actual == expected
    records = []
    for i in expected:
        layer = model.layers[i]
        moe = layer.ffn
        assert layer.row_sharded_mhc and not moe.is_hash_layer
        assert not moe._use_comm_fused_dispatch
        assert moe.experts.supports_prequantized_input()
        assert moe.shared_experts.gate_up_proj.native_a8_group_rows == 32
        assert not moe.shared_experts.w2.reduce_results
        assert moe.layer_id == i
        records.append(
            {
                "layer": i,
                "quant_method": type(moe.experts.quant_method).__name__,
                "TP": moe.experts.tp_size,
                "EP": moe.experts.ep_size,
                "DP": moe.experts.dp_size,
                "shared_group_rows": 32,
            }
        )
    return expected, records


def install():
    from atom.model_engine.model_runner import ModelRunner

    enabled = os.environ.get("ATOM_DSV41_PREFILL_QUANTIZED_FFN") == "1"
    assert enabled == (os.environ["DSV41_OWNER_SERVING"] == "1")
    consumer = None
    if enabled:
        from atom.models.deepseek_v41 import prefill_ffn as consumer
    bind(ModelRunner, consumer, enabled, configure)
