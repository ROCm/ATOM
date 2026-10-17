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


class _HFConfigView:
    """Read a model's HF config without caring whether it is nested.

    Multimodal configs (MiniMax-M3, Qwen3.5-VL, ...) keep the transformer's own
    fields on a nested text config: ``MiniMaxM3Config`` has no
    ``num_hidden_layers`` at all, it lives on ``.text_config``. vLLM exposes the
    resolved inner config as ``model_config.hf_text_config`` while
    ``hf_config`` stays the outer one, and ATOM's offload code reads both
    spellings (``config.py`` notes this skew itself: its namespace guard reads
    ``hf_config.model_type`` while ``is_qwen_next`` reads
    ``hf_text_config.model_type``).

    Resolving inner-first and falling back to the outer config satisfies both
    readers without asking either side to change which attribute it wants.
    """

    __slots__ = ("_inner", "_outer")

    def __init__(self, inner: Any, outer: Any) -> None:
        self._inner = inner
        self._outer = outer

    def __getattr__(self, name: str) -> Any:
        if self._inner is not None:
            try:
                return getattr(self._inner, name)
            except AttributeError:
                pass
        return getattr(self._outer, name)

    def __repr__(self) -> str:
        return (
            f"_HFConfigView(inner={type(self._inner).__name__}, "
            f"outer={type(self._outer).__name__})"
        )


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
        extra = dict(extra) if isinstance(extra, dict) else {}
        # Published in BOTH spellings on purpose. ATOM's offload readers
        # disagree about where the option bag lives, and the disagreement is
        # silent in one direction: ``offload/config.py`` reads
        # ``kvc.get("kv_connector_extra_config", kvc)`` and so tolerates the
        # flattened form, but ``offload/mp/deployment.py`` reads
        # ``kvc.get("kv_connector_extra_config", {})`` with no such fallback.
        # Publishing only the flattened form would hand the MP path an empty
        # bag, and every option there has a default -- server host and port
        # included -- so nothing would raise: the worker would quietly dial
        # tcp://localhost:5555 instead of the configured server. ``kv_role``
        # stays out of the nested copy, which is forwarded to LMCache's own
        # storage-option parser once the ``lmcache.mp.*`` keys are stripped;
        # it is a transfer role, not a storage option.
        self.kv_transfer_config = {
            "kv_role": role,
            **extra,
            "kv_connector_extra_config": extra,
        }

        outer_hf = getattr(model_config, "hf_config", None)
        inner_hf = getattr(model_config, "hf_text_config", None)
        self.hf_config = (
            _HFConfigView(inner_hf, outer_hf) if outer_hf is not None else None
        )
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
        # The multiprocess offload path reads its deployment geometry off the
        # config object directly rather than off ``parallel_config``, and every
        # read has a ``getattr(..., 1)`` default. A missing field therefore does
        # not raise: it silently describes a TP8 deployment as TP1, which makes
        # the LMCache side size one rank's key space for the whole world and
        # lets eight ranks write the same keys with different bytes. Project
        # them explicitly so the default never applies.
        self.tensor_parallel_size = int(
            getattr(parallel_config, "tensor_parallel_size", 1) or 1
        )
        self.data_parallel_size = int(
            getattr(parallel_config, "data_parallel_size", 1) or 1
        )
        self.prefill_context_parallel_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1) or 1
        )
        # Part of the LMCache model namespace: two revisions of one model name
        # have different weights and must not share a key space.
        self.revision = getattr(model_config, "revision", None)
        # Read only to be rejected -- the MP path refuses to run under
        # speculative decoding. Forwarded so that refusal happens here rather
        # than after a draft model has already been offloaded against.
        self.speculative_config = getattr(vllm_config, "speculative_config", None)
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
            f"{self.pipeline_parallel_size}, tp={self.tensor_parallel_size}, "
            f"dcp={self.decode_context_parallel_size})"
        )


def build_offload_config(vllm_config: Any) -> OffloadConfigShim:
    return OffloadConfigShim(vllm_config)
