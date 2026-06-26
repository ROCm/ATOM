import functools
import inspect
import logging
import threading

logger = logging.getLogger("atom")

_TLS = threading.local()


def apply_mamba_hybrid_seq_lens_patch() -> None:
    """Patch vLLM mamba-hybrid metadata wiring for seq_lens_cpu_upper_bound."""
    import vllm.envs as vllm_envs

    if vllm_envs.VLLM_USE_V2_MODEL_RUNNER is not True:
        return

    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid
    except ImportError:
        return

    original_build_attn_metadata = mamba_hybrid.build_attn_metadata
    original_prepare_attn = mamba_hybrid.MambaHybridModelState.prepare_attn

    if getattr(original_prepare_attn, "_atom_seq_lens_upper_bound_patched", False):
        return

    try:
        sig = inspect.signature(original_build_attn_metadata)
    except (TypeError, ValueError):
        return
    if "seq_lens_cpu_upper_bound" not in sig.parameters:
        return

    @functools.wraps(original_build_attn_metadata)
    def wrapped_build_attn_metadata(*args, **kwargs):
        if kwargs.get("seq_lens_cpu_upper_bound") is None:
            seq_lens_cpu_upper_bound = getattr(_TLS, "seq_lens_cpu_upper_bound", None)
            if seq_lens_cpu_upper_bound is not None:
                kwargs["seq_lens_cpu_upper_bound"] = seq_lens_cpu_upper_bound
        return original_build_attn_metadata(*args, **kwargs)

    @functools.wraps(original_prepare_attn)
    def wrapped_prepare_attn(self, input_batch, *args, **kwargs):
        previous = getattr(_TLS, "seq_lens_cpu_upper_bound", None)
        _TLS.seq_lens_cpu_upper_bound = getattr(
            input_batch, "seq_lens_cpu_upper_bound", None
        )
        try:
            return original_prepare_attn(self, input_batch, *args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(_TLS, "seq_lens_cpu_upper_bound")
                except AttributeError:
                    pass
            else:
                _TLS.seq_lens_cpu_upper_bound = previous

    setattr(wrapped_prepare_attn, "_atom_seq_lens_upper_bound_patched", True)
    mamba_hybrid.build_attn_metadata = wrapped_build_attn_metadata
    mamba_hybrid.MambaHybridModelState.prepare_attn = wrapped_prepare_attn
    logger.info(
        "ATOM plugin: patched vLLM mamba-hybrid attention metadata to "
        "forward seq_lens_cpu_upper_bound."
    )
