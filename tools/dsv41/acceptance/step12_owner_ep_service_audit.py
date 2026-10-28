"""Startup-only loaded EP contract, attached to the existing graph audit."""

import copy
import functools
import os


def bind(Runner, enabled, validate):
    if hasattr(Runner, "_step12_owner_ep_audit"):
        raise RuntimeError("Owner EP audit already installed")
    audit = Runner._rebase_graph_audit
    state = {"enabled": enabled, "loaded_contract": None, "checks": 0}
    Runner._step12_owner_ep_audit = state
    original_snapshot = audit.snapshot
    original_warmup = Runner._maybe_warmup

    def snapshot(*args, **kwargs):
        with audit.lock:
            result = original_snapshot(*args, **kwargs)
            if result is not None:
                result["owner_ep4"] = copy.deepcopy(state)
            return result

    @functools.wraps(original_warmup)
    def warmup(self):
        if state["checks"] != 0:
            raise RuntimeError("Loaded EP contract must bind once per worker")
        if bool(self.config.enable_expert_parallel) is not enabled:
            raise RuntimeError("EP CLI/config disagrees with the selected arm")
        if enabled:
            model = self.model
            while not hasattr(model, "layers") and hasattr(model, "model"):
                model = model.model
            contract = validate(model, self.config, loaded=True)
            expected = dict(model.owner_ep4_contract, loaded=True)
            if contract != expected:
                raise RuntimeError("Loaded EP contract differs from construction")
            state["loaded_contract"] = contract
        state["checks"] = 1
        return original_warmup(self)

    audit.snapshot = snapshot
    Runner._maybe_warmup = warmup
    return state


def install():
    from atom.model_engine.model_runner import ModelRunner

    enabled = os.environ["DSV41_STEP12_OWNER_EP4"] == "1"
    assert enabled == (os.environ["DSV41_OWNER_SERVING"] == "1")
    validate = None
    if enabled:
        from atom.models.deepseek_v41.owner_ep import validate_owner_ep4

        validate = validate_owner_ep4
    bind(ModelRunner, enabled, validate)
