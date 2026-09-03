# SPDX-License-Identifier: MIT
"""Present a ``VllmConfig`` as the config ATOM's offload path expects.

ATOM's offload code was written against ``atom.config.Config``. It reads a
handful of fields off it -- block size, KV dtype, the HF config, PP geometry,
the transfer role -- all of which vLLM already owns in plugin mode. Rather than
build a second ATOM Config (two sources of truth that drift), this projects the
vLLM one, forwarding by reference wherever possible so there is nothing to keep
in sync.

Kept apart from ``connector.py`` so it stays importable, and testable, without
vLLM installed.
"""

from typing import Any

# vLLM spells its KV cache dtype in its own vocabulary; ATOM indexes
# ``aiter.dtypes.d_dtypes`` with its own. Only the values a KV cache can
# actually hold are mapped -- an unknown one must raise rather than silently
# pick a width, because the codec sizes every byte segment from it.
_VLLM_TO_ATOM_KV_DTYPE = {
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp8_inc": "fp8",
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "float": "fp32",
}


def _atom_kv_dtype(cache_config: Any, model_config: Any) -> str:
    """Translate vLLM's ``cache_dtype`` into an ``aiter.dtypes`` key."""
    raw = str(getattr(cache_config, "cache_dtype", "auto") or "auto")
    if raw == "auto":
        # "auto" means "same as the model", which vLLM resolves on model_config.
        raw = str(getattr(model_config, "dtype", "bfloat16")).replace("torch.", "")
    key = _VLLM_TO_ATOM_KV_DTYPE.get(raw)
    if key is None:
        raise ValueError(
            f"ATOM offload connector: unsupported KV cache dtype {raw!r}. "
            "The byte codec sizes each segment from it, so it cannot be guessed."
        )
    return key


class OffloadConfigShim:
    """The ATOM-offload-shaped view of a vLLM config.

    Attribute names match what ``atom.kv_transfer.offload`` reads; everything
    else on ATOM's Config is untouched by that path.
    """

    def __init__(self, vllm_config: Any) -> None:
        self._vllm_config = vllm_config
        cache_config = getattr(vllm_config, "cache_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)

        block_size = getattr(cache_config, "block_size", None)
        if not block_size:
            raise ValueError(
                "ATOM offload connector: vLLM reported no cache_config.block_size; "
                "the codec addresses KV by block and cannot proceed without it"
            )
        self.kv_cache_block_size = int(block_size)
        self.kv_cache_dtype = _atom_kv_dtype(cache_config, model_config)

        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        role = getattr(kv_transfer_config, "kv_role", None) or "kv_both"
        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None)
        self.kv_transfer_config = {
            "kv_role": role,
            **(dict(extra) if isinstance(extra, dict) else {}),
        }

        self.hf_config = getattr(model_config, "hf_config", None)
        if self.hf_config is None:
            raise ValueError(
                "ATOM offload connector: vLLM reported no model_config.hf_config; "
                "the LMCache namespace and layer count are derived from it"
            )

        self.parallel_config = parallel_config
        self.pipeline_parallel_size = int(
            getattr(parallel_config, "pipeline_parallel_size", 1) or 1
        )
        self.decode_context_parallel_size = int(
            getattr(parallel_config, "decode_context_parallel_size", 1) or 1
        )
        # Feeds the LMCache page namespace, so two different models never share
        # a key space. vLLM's served name is the closest analogue of ATOM's
        # model_tag.
        self.model = str(getattr(model_config, "model", "") or "atom-model")
        self.model_tag = self.model

    def __repr__(self) -> str:
        return (
            f"OffloadConfigShim(block_size={self.kv_cache_block_size}, "
            f"kv_dtype={self.kv_cache_dtype!r}, role="
            f"{self.kv_transfer_config.get('kv_role')!r}, pp="
            f"{self.pipeline_parallel_size}, dcp={self.decode_context_parallel_size})"
        )


def build_offload_config(vllm_config: Any) -> OffloadConfigShim:
    return OffloadConfigShim(vllm_config)
