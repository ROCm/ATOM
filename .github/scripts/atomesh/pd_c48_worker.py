"""Model evidence, bounded startup stacks and optional Prefill trace labels."""
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
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
        kernel = getattr(method, "kernel", None)
        result.append({"name": name, "layer": classname(layer), "quant_method": classname(method),
                       "fp8_linear": classname(getattr(method, "fp8_linear", None)),
                       "linear_kernel": classname(kernel), "use_asm_gemm": getattr(kernel, "use_asm_gemm", None),
                       "moe_kernel": classname(getattr(method, "moe_kernel", None)), "tensors": tensors})
    return result


def inspect_speculative(spec, expected_drafts, expected_al):
    rates = spec.synthetic_acceptance_rates
    resolved_al = 1 + sum(rates) if rates is not None else None
    if expected_al == 0:
        if (spec.rejection_sample_method != "standard" or rates is not None
                or spec.num_speculative_tokens != expected_drafts):
            raise ValueError(f"C48 standard rejection mismatch: method={spec.rejection_sample_method}, "
                             f"drafts={spec.num_speculative_tokens}, rates={rates}")
    elif (spec.rejection_sample_method != "synthetic" or rates is None
            or spec.num_speculative_tokens != expected_drafts
            or len(rates) != expected_drafts
            or not math.isclose(resolved_al, expected_al, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"C48 synthetic acceptance mismatch: drafts={spec.num_speculative_tokens}, rates={rates}, expected_AL={expected_al}")
    return {"num_speculative_tokens": spec.num_speculative_tokens,
            "synthetic_acceptance_length": spec.synthetic_acceptance_length,
            "synthetic_acceptance_rates": rates,
            "resolved_acceptance_length": resolved_al,
            "rejection_sample_method": spec.rejection_sample_method}


def gemm_m_buckets(max_m):
    """ATOM model_runner.gemm_m_buckets: one M per aiter getPaddedM bucket up to max_m."""
    buckets = [m for m in (1, 2, 4, 8) if m <= max_m]
    for step, upper in ((16, 256), (32, 1024), (64, 4096), (128, max_m)):
        start = (buckets[-1] // step + 1) * step if buckets else step
        buckets.extend(range(start, min(upper, max_m) + 1, step))
    if max_m > 0 and buckets[-1] != max_m:
        buckets.append(max_m)
    return buckets


def warm_gemm_m_buckets(worker):
    """Run dummy forwards at every tuned-GEMM M bucket so first-use FlyDSL builds happen before serving.

    The optional second sweep (C48_P_GEMM_WARMUP_SWEEPS=2, the default) times the
    same sizes with every kernel already built; the difference between the sweeps
    is the first-use cost a serving prefill would pay.
    """
    import time
    import torch
    num_sweeps = int(os.environ.get("C48_P_GEMM_WARMUP_SWEEPS", "2"))
    if num_sweeps not in (1, 2):
        raise ValueError(f"C48_P_GEMM_WARMUP_SWEEPS must be 1 or 2, got {num_sweeps}")
    max_m = worker.scheduler_config.max_num_batched_tokens
    # Other tuned tables key exact M at multiples of 8 up to 512 (ATOM's linear pass).
    sizes = sorted(set(gemm_m_buckets(max_m)) | set(range(8, min(512, max_m) + 1, 8)) | {max_m - 1})
    sweeps = []
    for sweep in range(num_sweeps):
        times = []
        for m in sizes:
            if worker.rank == 0:
                print(f"[C48] P GEMM warmup sweep={sweep} M={m}", flush=True)
            start = time.perf_counter()
            worker.model_runner._dummy_run(m, skip_eplb=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start, m))
        sweeps.append(times)
    summary = (f"[C48] P GEMM M-bucket warmup rank={worker.rank} sizes={len(sizes)} "
               f"first={sum(t for t, _ in sweeps[0]):.1f}s ")
    if num_sweeps == 2:
        slowest = sorted(((a - b, m) for (a, m), (b, _) in zip(*sweeps)), reverse=True)[:8]
        summary += f"second={sum(t for t, _ in sweeps[1]):.1f}s first-minus-second "
    else:
        slowest = sorted(sweeps[0], reverse=True)[:8]
        summary += "slowest-first "
    print(summary + " ".join(f"M={m}:{d:.2f}s" for d, m in slowest), flush=True)


class ManifestWorker(Worker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        if os.environ.get("C48_P_GEMM_WARMUP") == "1":
            warm_gemm_m_buckets(self)
        return result

    def initialize_from_config(self, kv_cache_config):
        from pd_c48_timeline import startup_stage
        with startup_stage(self, "initialize_from_config"):
            return super().initialize_from_config(kv_cache_config)

    def annotate_profile(self, scheduler_output):
        original = super().annotate_profile(scheduler_output)
        if (os.environ.get("C48_PREFILL_TIMELINE") != "1"
                or os.environ["C48_DIAG_ROLE"] != "prefill"
                or not self.profiler or not self.profiler.should_annotate):
            return original
        import torch
        from pd_c48_timeline import batch_key

        @contextmanager
        def annotated():
            with original, torch.profiler.record_function(
                "C48-PREFILL-BATCH:" + batch_key(scheduler_output)
            ):
                yield
        return annotated()

    def load_model(self, *, load_dummy_weights=False):
        from pd_c48_memory import LoadingMemoryProbe
        with LoadingMemoryProbe(self):
            super().load_model(load_dummy_weights=load_dummy_weights)
        runner = self.model_runner
        target = inspect_model(runner.get_model())
        draft = inspect_model(runner.get_draft_model())
        counts = {key: sum("Fp8PtpcOnlineLinearMethod" in item["quant_method"] for item in rows)
                  for key, rows in (("target", target), ("draft", draft))}
        spec = self.vllm_config.speculative_config
        data = {"role": os.environ["C48_DIAG_ROLE"], "rank": self.rank,
                "target": target, "draft": draft, "ptpc_counts": counts,
                "profiler_config": asdict(self.vllm_config.profiler_config),
                "speculative": inspect_speculative(spec,
                    int(os.environ["C48_DIAG_DRAFT_TOKENS"]),
                    float(os.environ["C48_DIAG_AL"]))}
        out = Path(os.environ["C48_DIAG_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f'{data["role"]}-rank-{self.rank}.json').write_text(json.dumps(data, indent=2) + "\n")
        print(f"[C48] Loaded model quantization manifest rank={self.rank} PTPC={counts}", flush=True)
        assert bool(counts["target"]) == (os.environ["C48_DIAG_PTPC"] == "1"), counts
        expect = json.loads(os.environ.get("C48_EXPECT_QUANT", "{}"))
        got = {k: sum(item["quant_method"].endswith("." + k) for item in target) for k in expect}
        assert got == expect, f"C48 quantization coverage mismatch: got={got} expected={expect}"
