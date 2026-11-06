"""One startup check of actual fixed placement and both retained router paths."""

import copy
import functools
import os


def flags(model, enabled):
    expected = list(range(2, 20)) if enabled else []
    actual = {
        flag: [
            i for i, layer in enumerate(model.layers) if getattr(layer.ffn, flag, False)
        ]
        for flag in ("prefill_local_router", "prefill_route_packet")
    }
    if any(layers != expected for layers in actual.values()):
        raise RuntimeError(
            "Actual local router/packet layers disagree with the selected package"
        )
    placement = getattr(model, "owner_ep4_placement", None)
    if placement != ("v41-prefill-census-ep4-v1" if enabled else None):
        raise RuntimeError(
            "Actual expert placement disagrees with the selected package"
        )
    return dict(actual, placement=placement)


def bind(Runner, enabled):
    if hasattr(Runner, "_step12_placement_service_audit"):
        raise RuntimeError("Fixed-placement audit already installed")
    state = {"enabled": enabled, "checks": 0, "actual": None}
    Runner._step12_placement_service_audit = state
    audit = Runner._rebase_graph_audit
    previous_snapshot, previous_warmup = audit.snapshot, Runner._maybe_warmup

    def snapshot(*args, **kwargs):
        with audit.lock:
            result = previous_snapshot(*args, **kwargs)
            if result is not None:
                result["fixed_placement_package"] = copy.deepcopy(state)
            return result

    @functools.wraps(previous_warmup)
    def warmup(self):
        if state["checks"] != 0:
            raise RuntimeError("Fixed-placement package must bind once before warmup")
        model = self.model
        while not hasattr(model, "layers") and hasattr(model, "model"):
            model = model.model
        state["actual"] = flags(model, enabled)
        state["checks"] = 1
        return previous_warmup(self)

    audit.snapshot, Runner._maybe_warmup = snapshot, warmup
    return state


def install():
    from atom.model_engine.model_runner import ModelRunner

    enabled = os.environ["DSV41_STEP12_OWNER_EP4"] == "1"
    assert enabled == (os.environ["ATOM_DSV41_EP_PROFILED_PLACEMENT"] == "1")
    bind(ModelRunner, enabled)
