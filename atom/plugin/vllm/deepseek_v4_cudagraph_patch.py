"""Keep ATOM's DeepSeek-V4 on the CUDA-graph path it was built for.

For ``DeepseekV4ForCausalLM`` / ``DeepSeekV4MTPModel`` vLLM's
``VllmConfig.__post_init__`` sets ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` when the
variable is unset, and the breakable cudagraph then switches the compilation
mode to NONE. That default is meant for vLLM's own V4. The ATOM V4 plugin
model compiles with ATOM's torch.compile and splits at the V4 attention op
(``v4_attention_with_output``), which runs eagerly between the PIECEWISE
graphs and reads the metadata of the current batch. Under the breakable
cudagraph the whole forward is captured instead: the V4 attention (which has
no eager break) is recorded together with the capture batch's metadata, and
every replay of a prefill / mixed batch reuses it. The output is wrong.

So, before vLLM decides, the plugin opts ATOM's V4 out of the breakable
cudagraph. Only when the architecture resolves to the ATOM class in vLLM's
model registry; every other model (Kimi-K3, MiniMax-M3 rely on the breakable
cudagraph) is left alone, and an explicit user setting is kept.
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger("atom")

BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"

# The V4 architectures vLLM auto-enables the breakable cudagraph for and the
# plugin overrides with ATOM's model.
_V4_BREAKABLE_ARCHES = ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel")


def _plugin_disabled() -> bool:
    from atom.utils import envs

    return bool(envs.ATOM_DISABLE_VLLM_PLUGIN)


def _resolves_to_atom(arch: str) -> bool:
    """True iff vLLM's model registry maps ``arch`` to an ATOM class."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:  # noqa: BLE001
        return False
    entry = ModelRegistry.models.get(arch)
    if entry is None:
        return False
    module_name = getattr(entry, "module_name", None)
    if module_name is None:
        module_name = getattr(getattr(entry, "model_cls", None), "__module__", "")
    return module_name == "atom" or module_name.startswith("atom.")


def atom_v4_arch(architectures) -> str | None:
    """The ATOM-served V4 architecture among ``architectures``, or None."""
    for arch in architectures or ():
        if arch in _V4_BREAKABLE_ARCHES and _resolves_to_atom(arch):
            return arch
    return None


def _opt_out_of_breakable_cudagraph(vllm_config) -> None:
    if _plugin_disabled():
        return
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return
    arch = atom_v4_arch(getattr(model_config, "architectures", None))
    if arch is None:
        return
    if BREAKABLE_ENV not in os.environ:
        os.environ[BREAKABLE_ENV] = "0"
        logger.info(
            "[atom-v4-cg] %s is served by ATOM: setting %s=0 (pid=%d). ATOM's V4 "
            "runs prefill / mixed batches on ATOM's PIECEWISE graphs with the V4 "
            "attention eager; vLLM's breakable cudagraph would capture that "
            "attention with the capture batch's metadata.",
            arch,
            BREAKABLE_ENV,
            os.getpid(),
        )
    elif os.environ[BREAKABLE_ENV] == "1":
        logger.warning(
            "[atom-v4-cg] %s=1 is set explicitly for %s served by ATOM. The "
            "breakable cudagraph captures the V4 attention with the capture "
            "batch's metadata: prefill and mixed batches up to the largest "
            "capture size will produce wrong output. Unset it (or set 0).",
            BREAKABLE_ENV,
            arch,
        )


def apply_vllm_v4_breakable_cudagraph_opt_out() -> bool:
    """Wrap ``VllmConfig.__post_init__`` so ATOM's V4 opts out before vLLM's
    auto-enable runs. ``VllmConfig`` is a pydantic dataclass; pydantic calls
    ``__post_init__`` through the class, so the class-level wrap is honoured.
    Idempotent."""
    try:
        from vllm.config import VllmConfig
    except Exception as e:  # noqa: BLE001
        logger.debug("[atom-v4-cg] VllmConfig unavailable (%s), skip", e)
        return False
    original = VllmConfig.__post_init__
    if getattr(original, "_atom_v4_breakable_opt_out", False):
        return False

    @functools.wraps(original)
    def __post_init__(self, *args, **kwargs):
        _opt_out_of_breakable_cudagraph(self)
        return original(self, *args, **kwargs)

    __post_init__._atom_v4_breakable_opt_out = True  # type: ignore[attr-defined]
    VllmConfig.__post_init__ = __post_init__
    return True
