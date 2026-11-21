"""The optimizations of this example, as patches applied by ``train_dit_lumen.py``.

Nothing here modifies VeOmni's source or the Lumen library; each module rebinds
at runtime what it names, and only when its ``LUMEN_PATCH`` entry is set.

==================  ==============================================================
``LUMEN_PATCH``     module
==================  ==============================================================
``vae_conv*``       :mod:`.aiter_conv` -- aiter's FlyDSL convolution, used by
                    ``lumen_vae_conv.py``
``sdpa_efficient``  :mod:`.sdpa` -- SDPA flash backend off (aiter ``fmha_bwd``)
``attn_triton_fwd`` :mod:`.sdpa` -- aiter Triton forward on DiT attention
``rmsnorm_fuse``    :mod:`.rmsnorm` -- diffusers RMSNorm forward via ``torch.compile``
``local_adamw``     :mod:`.local_adamw` -- fused AdamW on FSDP2 local shards
``qwen_offline_fix`` :mod:`.qwen_offline` -- VeOmni Qwen-Image offline_training fix
==================  ==============================================================
"""


def _sdpa_efficient():
    from .sdpa import prefer_efficient_backend

    prefer_efficient_backend()


def _attn_triton_fwd():
    from .sdpa import install_triton_forward

    install_triton_forward()


def _rmsnorm_fuse():
    from .rmsnorm import patch_diffusers_rmsnorm

    patch_diffusers_rmsnorm()


def _local_adamw():
    from .local_adamw import install_local_shard_adamw

    install_local_shard_adamw()


def _qwen_offline_fix():
    from .qwen_offline import install_qwen_offline_fix

    install_qwen_offline_fix()


# In application order: attn_triton_fwd keeps the backward sdpa_efficient selects.
INSTALLERS = {
    "sdpa_efficient": _sdpa_efficient,
    "attn_triton_fwd": _attn_triton_fwd,
    "rmsnorm_fuse": _rmsnorm_fuse,
    "local_adamw": _local_adamw,
    "qwen_offline_fix": _qwen_offline_fix,
}
