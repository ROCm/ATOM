"""One-time model-load evidence. No forward, scheduling, or sampling overrides."""
import json
import math
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


def classname(value):
    return None if value is None else type(value).__module__ + "." + type(value).__name__


def inspect_model(model):
    result = []
    if model is None:
        return result
    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        if method is None:
            continue
        tensors = {}
        for key in ("weight", "weight_scale", "weight_scale_inv", "input_scale", "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            tensor = getattr(layer, key, None)
            if tensor is not None and hasattr(tensor, "shape"):
                tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        result.append({"name": name, "layer": classname(layer), "quant_method": classname(method),
                       "fp8_linear": classname(getattr(method, "fp8_linear", None)),
                       "moe_kernel": classname(getattr(method, "moe_kernel", None)), "tensors": tensors})
    return result


def inspect_speculative(spec, expected_drafts, expected_al):
    rates = spec.synthetic_acceptance_rates
    resolved_al = 1 + sum(rates) if rates is not None else None
    if (spec.rejection_sample_method != "synthetic" or rates is None
            or spec.num_speculative_tokens != expected_drafts
            or len(rates) != expected_drafts
            or not math.isclose(resolved_al, expected_al, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"C48 synthetic acceptance mismatch: drafts={spec.num_speculative_tokens}, rates={rates}, expected_AL={expected_al}")
    return {"num_speculative_tokens": spec.num_speculative_tokens,
            "synthetic_acceptance_length": spec.synthetic_acceptance_length,
            "synthetic_acceptance_rates": rates,
            "resolved_acceptance_length": resolved_al,
            "rejection_sample_method": spec.rejection_sample_method}


class ManifestWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        runner = self.model_runner
        target = inspect_model(runner.get_model())
        draft = inspect_model(runner.get_draft_model())
        counts = {key: sum("Fp8PtpcOnlineLinearMethod" in item["quant_method"] for item in rows)
                  for key, rows in (("target", target), ("draft", draft))}
        spec = self.vllm_config.speculative_config
        data = {"role": os.environ["C48_DIAG_ROLE"], "rank": self.rank,
                "target": target, "draft": draft, "ptpc_counts": counts,
                "speculative": inspect_speculative(spec,
                    int(os.environ["C48_DIAG_DRAFT_TOKENS"]),
                    float(os.environ["C48_DIAG_AL"]))}
        out = Path(os.environ["C48_DIAG_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{data["role"]}-rank-{self.rank}.json').write_text(json.dumps(data, indent=2) + "\n")
        print(f"[C48] Loaded model quantization manifest rank={self.rank} PTPC={counts}", flush=True)
        assert bool(counts["target"]) == (os.environ["C48_DIAG_PTPC"] == "1"), counts
