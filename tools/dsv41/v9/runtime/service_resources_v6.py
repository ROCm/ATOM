# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Snapshot actual graph/IPC bindings outside formal timing; same observer in A/B."""

import functools
import os
import threading

LOCAL = threading.local()


def install():
    import torch  # noqa: F401
    import aiter.ops.flydsl.moe_kernels as mk
    from formal_counters import Counters
    import guarded_service_v2 as dispatcher

    enabled = os.environ.get("DSV41_COMBINED_GUARDED") == "1"
    if enabled:
        assert dispatcher.install()["installed"]
        from prefill_fp8_wait0_v1 import corrected
        import atom.models.deepseek_v41.prefill_ffn as pf

        original_stage, original_prefill = (
            mk.flydsl_moe_stage1,
            pf.prequantized_owned_ffn,
        )

        def stage(**kw):
            if (
                getattr(LOCAL, "prefill_layer", None) in range(2, 19)
                and tuple(kw["a"].shape) == (4096, 5120)
                and kw["out_dtype"] == "fp8"
                and kw.get("v2_output_layout") is True
            ):
                dispatcher.STATE["counts"]["scoped_prefill_fp8_wait0"] += 1
                return corrected(**kw)
            return original_stage(**kw)

        def prefill(moe, local, rows):
            previous = getattr(LOCAL, "prefill_layer", None)
            LOCAL.prefill_layer = getattr(moe.experts, "layer_id", None)
            try:
                return original_prefill(moe, local, rows)
            finally:
                LOCAL.prefill_layer = previous

        mk.flydsl_moe_stage1, pf.prequantized_owned_ffn = stage, prefill
    old_snapshot = Counters.snapshot

    def binding(tensor):
        return {
            "ptr": tensor.data_ptr(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "bytes": tensor.numel() * tensor.element_size(),
        }

    @functools.wraps(old_snapshot)
    def snapshot(self, request):
        data = old_snapshot(self, request)
        serving = self.serving
        comm = serving.group.device_communicator.ca_comm
        state = {
            "scope": "CPU binding snapshots under existing forward lock, no diagnostic GPU kernels",
            "owner_graphs": [
                {
                    "key": key,
                    "capacity": key[0],
                    "hidden": binding(value[1]),
                    "auxiliary": binding(value[2]),
                }
                for key, value in serving.graphs.items()
            ],
            "graph_pool": serving.graph_pool,
            "graph_coverage": serving.graph_coverage.report(),
            "comm_ptr": comm._ptr,
            "input_pool": {
                "ptr": comm._pool["input"].data_ptr,
                "max_size": comm._pool["input"].max_size,
            },
            "comm_capture_flags": {
                "active": comm._IS_CAPTURING,
                "register_enabled": comm.enable_register_for_capturing,
                "requested": comm._register_for_capturing_requested,
            },
            "prefill_adapters": [],
            "owner_order_codes": None,
        }
        if enabled:
            data["combined_service"] = dispatcher.snapshot()
            if dispatcher.STATE["ready"]:
                state["owner_order_codes"] = binding(dispatcher.STATE["rs"].codes)
            import explicit_routes

            state["prefill_adapters"] = [
                {
                    "rows": rows,
                    "padded": adapter.padded,
                    "codes": binding(adapter.codes),
                    "graph_scratch": [binding(t) for t in adapter.graph_scratch],
                }
                for rows, adapter in sorted(explicit_routes.candidates.items())
            ]
        data["graph_resources"] = state
        return data

    Counters.snapshot = snapshot
    return {"installed": True, "candidate": enabled, "shadow_numerics": False}
