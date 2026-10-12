"""Select the control before model loading, compilation and graph capture."""

import os

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

MODE = os.environ["K3_CONTEXT_MODE"]
assert MODE in ("fallback", "fused")
if MODE == "fallback":
    AiterMLAImpl._context_parallel_compute_prefill_context = (
        MLACommonImpl._context_parallel_compute_prefill_context
    )


class StaticControl:
    def experiment_info(self):
        model = self.model_runner.model
        layers = []
        for name, module in model.named_modules():
            impl = getattr(module, "impl", None)
            if not isinstance(impl, AiterMLAImpl):
                continue
            weight = getattr(impl.kv_b_proj, "weight", None)
            layers.append(
                {
                    "name": name,
                    "method": impl._context_parallel_compute_prefill_context.__qualname__,
                    "weight_dtype": str(weight.dtype) if weight is not None else None,
                    "weight_shape": list(weight.shape) if weight is not None else None,
                    "heads": impl.num_heads,
                }
            )
        return {
            "rank": self.rank,
            "mode": MODE,
            "context_implementation": (
                AiterMLAImpl._context_parallel_compute_prefill_context.__qualname__
            ),
            "layers": layers,
        }
