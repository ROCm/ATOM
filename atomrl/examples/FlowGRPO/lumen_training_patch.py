# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Install the perf-veomni Lumen patches in the RL actor only.

verl imports this module in every Ray process through ``VERL_USE_EXTERNAL_MODULES=lumen_training_patch``. Importing it only
registers an import hook; the patches are installed when a process builds VeOmni's diffusion training engine
(``VeOmniDiffusionEngine._build_model_optimizer``), i.e. in the actor workers. The vLLM-Omni rollout and the
judge run in their own processes and are not touched.

``LUMEN_PATCH`` (``:`` or ``,`` separated; run_flowgrpo.sh defaults to ``rmsnorm_fuse:local_adamw``):

    rmsnorm_fuse, local_adamw  -- Lumen's examples/perf-veomni/veomni_patches, unchanged; run_flowgrpo.sh puts
                                  that directory on the workers' PYTHONPATH
    mask_all_true_to_none      -- a boolean SDPA mask that is all True is passed as None. Exact (an all-True mask
                                  masks nothing). diffusers' Qwen-Image always passes the text-padding mask; a
                                  training micro-batch holds the samples of one prompt, so its mask is all True and
                                  attention can use the flash kernel instead of the masked one. Off by default:
                                  with it, training reward lagged the baseline over steps 30-150.

Rank 0 prints ``[lumen-training]`` lines with per-patch call counts as evidence that the patches ran.
"""

import importlib.abc
import importlib.util
import os
import re
import sys

_TARGET = "verl_omni.workers.engine.veomni.diffusion_impl"
PATCHES = [p for p in re.split(r"[,:]", os.environ.get("LUMEN_PATCH", "")) if p]
_installed = False
_mask_stats = {"dropped_all_true": 0, "kept_partial": 0}
_calls = 0


def _log(msg):
    if os.environ.get("RANK", "0") == "0":
        print(f"[lumen-training] {msg}", flush=True)


def _install_mask_adapter():
    import weakref

    import torch
    import torch.nn.functional as F

    inner = F.scaled_dot_product_attention
    # The DiT builds one mask per forward and passes views of it to all 60 blocks; check it once.
    cache = {"ref": None}

    def sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            base = attn_mask._base if attn_mask._base is not None else attn_mask
            ref = cache["ref"]
            if (
                ref is not None
                and ref() is base
                and cache["key"] == (tuple(attn_mask.shape), base._version)
            ):
                all_true = cache["all_true"]
            else:
                all_true = bool(attn_mask.all())
                cache.update(
                    ref=weakref.ref(base),
                    key=(tuple(attn_mask.shape), base._version),
                    all_true=all_true,
                )
            _mask_stats["dropped_all_true" if all_true else "kept_partial"] += 1
            if all_true:
                attn_mask = None
        return inner(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    F.scaled_dot_product_attention = sdpa


def install(target):
    global _installed
    if _installed:
        return
    _installed = True
    from atomrl.lumen.config import LumenConfig

    unknown = set(PATCHES) - {"rmsnorm_fuse", "local_adamw", "mask_all_true_to_none"}
    if unknown:
        raise ValueError(f"unsupported LUMEN_PATCH entries for RL: {sorted(unknown)}")
    LumenConfig(
        scaling="none",
        diffusers_rmsnorm_fuse="rmsnorm_fuse" in PATCHES,
        local_shard_adamw="local_adamw" in PATCHES,
    ).enable(target)
    if "mask_all_true_to_none" in PATCHES:
        _install_mask_adapter()
    _log(f"installed in pid {os.getpid()}: {PATCHES}")


def stats():
    out = {"mask": dict(_mask_stats)}
    if "rmsnorm_fuse" in PATCHES:
        from atomrl.lumen.ops.normalization import rmsnorm

        out["rmsnorm"] = rmsnorm.diffusers_rmsnorm_stats()
    return out


def _wrap_engine(module):
    cls = module.VeOmniDiffusionEngine
    build, fwd_bwd = cls._build_model_optimizer, cls.forward_backward_batch

    def _build_model_optimizer(self, *args, **kwargs):
        result = build(self, *args, **kwargs)
        install(self)
        return result

    def forward_backward_batch(self, *args, **kwargs):
        global _calls
        out = fwd_bwd(self, *args, **kwargs)
        _calls += 1
        if _calls <= 4 or _calls % 30 == 0:
            _log(f"after forward_backward_batch #{_calls}: {stats()}")
        return out

    cls._build_model_optimizer = _build_model_optimizer
    cls.forward_backward_batch = forward_backward_batch


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def _exec(module):
            exec_module(module)
            _wrap_engine(module)

        spec.loader.exec_module = _exec
        return spec


if PATCHES:
    if _TARGET in sys.modules:
        _wrap_engine(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
