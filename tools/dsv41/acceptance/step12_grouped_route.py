"""Host-only proof that each captured owner body used 40 grouped H16 calls."""

import functools

CAPACITIES = (16, 32, 64, 96, 128, 192)


def validate_capture(capacity, graph_delta, native_delta, recorded_calls):
    if capacity not in CAPACITIES or graph_delta not in (0, 1):
        raise RuntimeError("Unexpected owner capture domain")
    if native_delta != 40 * graph_delta or recorded_calls < 40:
        raise RuntimeError("Every owner graph must contain all 40 grouped H16 calls")


def install():
    import torch

    from atom.models.deepseek_v41.owner_service import owner_grouped_attention
    from atom.models.deepseek_v41.owner_service.owner_serving import OwnerServing

    bind(OwnerServing, owner_grouped_attention, torch.cuda.is_current_stream_capturing)


def bind(OwnerServing, owner_grouped_attention, is_capturing):
    """Bind once; separated from imports so the real wrapper is host-testable."""
    if getattr(OwnerServing.run, "_step12_grouped_route", False):
        raise RuntimeError("Grouped route proof already installed")
    native = owner_grouped_attention.owner_grouped_attention_fwd
    original = OwnerServing.run
    active = [None]

    @functools.wraps(native)
    def grouped(query, *args, **kwargs):
        serving = active[0]
        if serving is None:
            raise RuntimeError("Grouped call outside the actual owner target seam")
        capacity = query.shape[0]
        if query.shape[1:] != (64, 512) or not query.is_contiguous():
            raise RuntimeError("Grouped model query layout changed")
        result = native(query, *args, **kwargs)
        mode = "capture" if is_capturing() else "eager"
        serving.counters[f"grouped_H16_{mode}_calls"] += 1
        serving.counters[f"grouped_H16_{mode}_N{capacity}"] += 1
        return result

    @functools.wraps(original)
    def run(self, *args, **kwargs):
        if active[0] is not None:
            raise RuntimeError("Unexpected nested owner target")
        graphs = self.counters["owner_graph_captures"]
        native_calls = self.counters["grouped_H16_capture_calls"]
        active[0] = self
        try:
            result = original(self, *args, **kwargs)
        finally:
            active[0] = None
        capacity = self.active[0].step.width
        validate_capture(
            capacity,
            self.counters["owner_graph_captures"] - graphs,
            self.counters["grouped_H16_capture_calls"] - native_calls,
            self.counters[f"grouped_H16_capture_N{capacity}"],
        )
        self.counters["grouped_graph_replay_steps"] += 1
        if self.counters["owner_graph_captures"] != graphs:
            self.publish()
        return result

    run._step12_grouped_route = True
    owner_grouped_attention.owner_grouped_attention_fwd = grouped
    OwnerServing.run = run
