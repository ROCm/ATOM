"""Let vLLM's Kimi-K3 KDA kernels import under Triton 3.8.

The pinned vLLM keys the autotune of ``chunk_kda_fwd_kernel_intra_sub_chunk``
(``vllm/models/kimi_k3/{amd,nvidia}/ops/third_party/kda/chunk_intra.py``) on
``NC``, which is not one of the kernel's arguments. Triton 3.8 checks autotune
keys against the arguments when the decorator runs, so importing
``vllm.models.kimi_k3`` raises

    ValueError: key contains names that are not kernel arguments: NC.

and every Kimi-K3 deployment fails model inspection. vllm-project/vllm#60446
drops the key upstream, after the pinned commit.

The decorator runs at import, so the kernel cannot be patched afterwards; the
fix has to be in place before vLLM's module is imported. This wraps
``triton.autotune``: for a kernel defined under ``vllm.models.kimi_k3`` it
drops the key names that are not arguments -- what #60446 does by hand -- and
it passes every other kernel through unchanged. A vLLM that carries #60446
names no such key and the wrapper drops nothing; the INFO line naming a dropped
key is the sign this patch is still needed.
"""

import functools
import logging

logger = logging.getLogger("atom")

_KIMI_K3_MODULE_PREFIX = "vllm.models.kimi_k3."


def _keys_that_are_arguments(fn, key):
    """``key`` without the names ``fn`` does not take, for Kimi-K3 kernels only."""
    module = getattr(fn, "__module__", None) or ""
    arg_names = getattr(fn, "arg_names", None)
    if not module.startswith(_KIMI_K3_MODULE_PREFIX) or arg_names is None:
        return key
    dropped = [name for name in key if name not in arg_names]
    if not dropped:
        return key
    logger.info(
        "ATOM plugin: dropped autotune key(s) %s of %s.%s; they are not kernel "
        "arguments, which Triton 3.8 rejects (vllm-project/vllm#60446).",
        dropped,
        module,
        getattr(fn, "__name__", "<kernel>"),
    )
    return [name for name in key if name in arg_names]


def apply_vllm_kimi_k3_kda_autotune_patch() -> bool:
    """Wrap ``triton.autotune``. Idempotent; returns whether this call installed it.

    Must run before anything imports ``vllm.models.kimi_k3``. It is installed
    from ``register_model``, which vLLM's model-inspection subprocess also runs
    (``load_general_plugins``) before it imports the model class.
    """
    try:
        import triton
    except ImportError:
        return False
    original = triton.autotune
    if getattr(original, "_atom_kimi_k3_autotune_keys_patched", False):
        return False

    @functools.wraps(original)
    def autotune(configs, key, *args, **kwargs):
        def decorator(fn):
            return original(
                configs, _keys_that_are_arguments(fn, key), *args, **kwargs
            )(fn)

        return decorator

    autotune._atom_kimi_k3_autotune_keys_patched = True
    triton.autotune = autotune
    return True
