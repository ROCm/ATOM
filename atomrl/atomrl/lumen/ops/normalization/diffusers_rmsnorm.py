"""diffusers ``RMSNorm`` with its forward fused into one kernel by ``torch.compile``.

diffusers' RMSNorm forward is an eager chain -- upcast, square, mean, rsqrt,
multiply, downcast, scale -- of 8 kernels. Compiling that function alone (the
model cannot be compiled whole under VeOmni's DiT trainer) collapses it to one.
Measured on MI350X at the Qwen-Image DiT's image branch, [1, 4096, 24, 128]
BF16, 3 repeats:

    impl      fwd us   fwd+bwd us
    stock       91         367
    compiled    22         298

The whole win is in the forward; neither the compiled graph nor Lumen's AITER
kernel improves the backward. It is also more accurate, 55.6 dB against an FP32
reference versus 52.6 dB, because the intermediate stays in FP32.

:func:`patch_diffusers_rmsnorm` is deliberately narrow:

* Only ``diffusers.models.normalization.RMSNorm`` with a weight and no bias.
  Never LayerNorm -- :meth:`atomrl.lumen.config.LumenConfig._patch_norms` replaces by
  class name and would also catch a DiT's ``elementwise_affine=False``
  LayerNorms, inventing trainable tensors the optimizer and FSDP2 never see.
* Only 4-D inputs with ``shape[1] >= min_seq``. A 13-token text branch is
  launch-bound and the fused version is not reliably faster there.
* Anything else, a DTensor weight included, takes the original forward.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_compiled = None
_orig_forward = None
_stats = {"fused": 0, "fallback": 0}


def _rmsnorm_weighted(hidden_states, weight, eps):
    """Structurally identical to diffusers 0.37.0 ``RMSNorm.forward``, weight path."""
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.to(weight.dtype)
    return hidden_states * weight


def fused_rmsnorm(hidden_states, weight, eps):
    """``_rmsnorm_weighted`` through ``torch.compile``, compiled on first use."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_rmsnorm_weighted, dynamic=False)
    return _compiled(hidden_states, weight, eps)


def patch_diffusers_rmsnorm(min_seq=1024):
    """Patch ``diffusers.models.normalization.RMSNorm.forward`` at the class level. Idempotent.

    Returns False, patching nothing, if diffusers is not importable.
    """
    global _orig_forward
    if _orig_forward is not None:
        return True
    try:
        from diffusers.models.normalization import RMSNorm
    except (ImportError, OSError) as exc:
        logger.info(
            "rmsnorm_compiled: diffusers unavailable (%s), not patching",
            type(exc).__name__,
        )
        return False

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()

    _orig_forward = RMSNorm.forward
    orig = _orig_forward
    threshold = int(min_seq)

    def forward(self, hidden_states):
        w = getattr(self, "weight", None)
        if (
            w is not None
            and getattr(self, "bias", None) is None
            and hidden_states.dim() == 4
            and hidden_states.shape[1] >= threshold
            and not isinstance(w, DTensor)
            and not isinstance(hidden_states, DTensor)
        ):
            _stats["fused"] += 1
            return fused_rmsnorm(hidden_states, w, self.eps)
        _stats["fallback"] += 1
        return orig(self, hidden_states)

    RMSNorm.forward = forward
    logger.info(
        "rmsnorm_compiled: diffusers RMSNorm.forward fused when 4-D and seq >= %d",
        threshold,
    )
    return True


def unpatch_diffusers_rmsnorm():
    global _orig_forward
    if _orig_forward is None:
        return
    from diffusers.models.normalization import RMSNorm

    RMSNorm.forward = _orig_forward
    _orig_forward = None


def stats():
    """Calls that took the fused forward and calls that fell back to diffusers'."""
    return dict(_stats)


__all__ = [
    "fused_rmsnorm",
    "patch_diffusers_rmsnorm",
    "stats",
    "unpatch_diffusers_rmsnorm",
]
