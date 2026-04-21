import functools
import logging

from atom.plugin.vllm.sampler_ops import greedy_sample_tokens
from atom.utils import envs

logger = logging.getLogger("atom")

_PATCH_ATTR = "_atom_vllm_sampler_patch_installed"


def _mark_patched(wrapper, original) -> None:
    setattr(wrapper, _PATCH_ATTR, True)
    setattr(wrapper, "_atom_vllm_sampler_original", original)


def _patch_v1_sampler() -> bool:
    try:
        from vllm.v1.sample.sampler import Sampler as V1Sampler
    except ImportError:
        logger.debug("vLLM V1 sampler module is unavailable; skip ATOM sampler patch.")
        return False

    original = V1Sampler.greedy_sample
    if getattr(original, _PATCH_ATTR, False):
        return False

    @functools.wraps(original)
    def _atom_greedy_sample(logits):
        # vLLM V1's original greedy path returns int64 token ids.
        return greedy_sample_tokens(logits).long()

    _mark_patched(_atom_greedy_sample, original)
    V1Sampler.greedy_sample = staticmethod(_atom_greedy_sample)
    return True


def apply_vllm_sampler_patch() -> None:
    """Install ATOM's greedy sampler patch into vLLM V1 only."""

    if envs.ATOM_DISABLE_VLLM_PLUGIN:
        logger.info("Skip ATOM vLLM sampler patch because ATOM platform is disabled.")
        return

    # Only vLLM V1 is patched. V2 keeps the original vLLM sampler path.
    try:
        if _patch_v1_sampler():
            logger.info("Enabled ATOM vLLM V1 greedy sampler patch.")
    except Exception:
        # Patching remains best-effort because vLLM internal symbols can drift.
        logger.exception("Failed to apply ATOM vLLM V1 sampler patch.")
