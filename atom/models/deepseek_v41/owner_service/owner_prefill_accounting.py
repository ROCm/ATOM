"""Attach existing retained-prefill counters to the actual wrapped runtime."""

import json
from functools import wraps


def bind_runtime_accounting(serving):
    """The Mono wrapper delegates reads but stores assignments on itself."""
    outer = runtime = serving.runner.model
    visited = set()
    while id(runtime) not in visited:
        visited.add(id(runtime))
        inner = getattr(runtime, "model", runtime)
        if inner is runtime:
            break
        runtime = inner
    else:
        raise ValueError("Cyclic model wrapper cannot bind retained-prefill counters")
    if not getattr(runtime, "owner_service_prefill", False):
        raise ValueError("Bind counters only to the admitted owner-prefill runtime")
    if len(runtime.layers) != 40:
        raise ValueError("Bind the measured forty-layer runtime")
    runtime._owner_serving = serving
    return {
        "outer_type": type(outer).__name__,
        "runtime_type": type(runtime).__name__,
        "different_objects": runtime is not outer,
        "runtime_counter_owner_bound": runtime._owner_serving is serving,
        "scope": "Observability binding only; existing forward, kernel, state and scheduler execution unchanged",
    }


def install():
    from .owner_serving import OwnerServing

    original = OwnerServing.__init__

    @wraps(original)
    def initialize(self, runner):
        original(self, runner)
        receipt = bind_runtime_accounting(self)
        receipt["rank"] = self.rank
        raw = (json.dumps(receipt, indent=2) + "\n").encode()
        if len(raw) > 128_000:
            raise RuntimeError("Prefill accounting binding exceeds its reservation")
        self.evidence.with_name(f"prefill_accounting_rank{self.rank}.json").write_bytes(
            raw
        )

    OwnerServing.__init__ = initialize
