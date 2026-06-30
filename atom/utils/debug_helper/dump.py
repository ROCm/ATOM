# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Generic, env-gated debug dump for model bisecting (forward / weights / sampler).

All entry points are no-ops when their controlling env var is unset, so this
module is safe to wire into hot paths in production.

Env vars (all defined in atom/utils/envs.py):
  ATOM_FWD_DUMP_DIR / ATOM_FWD_DUMP_LAYERS / ATOM_FWD_DUMP_BLOCK_CLASS /
  ATOM_FWD_DUMP_LAYER_ATTR / ATOM_FWD_DUMP_ONE_SHOT
  ATOM_WEIGHT_DUMP_DIR / ATOM_WEIGHT_DUMP_LAYERS / ATOM_WEIGHT_DUMP_EXIT
  ATOM_DEBUG_TOPK / ATOM_DEBUG_TOPK_PATH

Output file naming
------------------
Forward:  {ATOM_FWD_DUMP_DIR}/layer{LL}_rank{R}.pt    (key: "hidden", "shape")
Weights:  {ATOM_WEIGHT_DUMP_DIR}/weight_rank{R}_layer{L}.pt
          (keys: "_tp_rank", "_tp_size", "_layer", + param/buffer dotted names)

Typical wiring (one line per integration point)
-----------------------------------------------
After model load (model_runner.py):
    from atom.utils.debug_helper import (
        install_block_forward_hooks, maybe_dump_weights_and_exit,
    )
    install_block_forward_hooks(self.model)   # no-op without env
    maybe_dump_weights_and_exit(self.model)   # no-op without env (or sys.exit)

Inside Sampler.forward (optional):
    from atom.utils.debug_helper import maybe_log_topk
    maybe_log_topk(logits)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

from atom.utils import envs

# === helpers =========================================================


def _parse_layer_set(env_value: str) -> Optional[set[int]]:
    """Return None for empty (= dump all), else parsed integer set."""
    if not env_value:
        return None
    return {int(x) for x in env_value.split(",") if x}


def _get_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_initialized() else 1


# === Forward dump ====================================================


def install_block_forward_hooks(model: torch.nn.Module) -> int:
    """Install per-Block forward hooks that dump hidden_out per layer.

    No-op when ATOM_FWD_DUMP_DIR is unset. Returns number of hooks installed.

    Block detection: a submodule whose class name matches one of the
    comma-separated names in ATOM_FWD_DUMP_BLOCK_CLASS (default "Block")
    AND has the layer-index attribute named ATOM_FWD_DUMP_LAYER_ATTR
    (default "layer_id"). For sub-stage bisecting, list multiple class names
    e.g. "Block,DeepseekV4Attention,FusedMoE" — each is matched against the
    `layer_id` of its parent block and tagged in the output filename by
    class name. The layer index is filtered by ATOM_FWD_DUMP_LAYERS
    (default: all layers).

    Output filename: layer{LL}_{ClassName}_rank{R}.pt
    """
    dump_dir = envs.ATOM_FWD_DUMP_DIR
    if not dump_dir:
        return 0

    os.makedirs(dump_dir, exist_ok=True)
    wanted = _parse_layer_set(envs.ATOM_FWD_DUMP_LAYERS)
    block_classes = {
        c.strip() for c in envs.ATOM_FWD_DUMP_BLOCK_CLASS.split(",") if c.strip()
    }
    layer_attr = envs.ATOM_FWD_DUMP_LAYER_ATTR
    one_shot = envs.ATOM_FWD_DUMP_ONE_SHOT
    rank = _get_rank()

    # Per-(layer, class) call counter — used when one_shot=False to distinguish
    # warmup vs prefill vs per-seq dispatched calls.
    _call_counters: dict[tuple[int, str], int] = {}

    def _make_hook(layer_id: int, cls_name: str):
        base = os.path.join(dump_dir, f"layer{layer_id:02d}_{cls_name}_rank{rank}")
        one_shot_fname = base + ".pt"

        def _hook(_mod, _args, output):
            if one_shot:
                if os.path.exists(one_shot_fname):
                    return
                fname = one_shot_fname
            else:
                key = (layer_id, cls_name)
                n = _call_counters.get(key, 0)
                _call_counters[key] = n + 1
                fname = f"{base}_call{n:03d}.pt"
            t = output[0] if isinstance(output, tuple) else output
            if not isinstance(t, torch.Tensor):
                return
            torch.save(
                {"hidden": t.detach().cpu(), "shape": tuple(t.shape)},
                fname,
            )

        return _hook

    # Build map: id(module) -> layer_id, by walking the model and matching
    # parent blocks (which carry layer_attr). Sub-modules of a block share its
    # layer_id; we discover this by traversing named_modules with prefix matching.
    block_layer_ids: dict[str, int] = {}  # module dotted name -> layer_id
    for name, mod in model.named_modules():
        lid = getattr(mod, layer_attr, None)
        if lid is not None:
            block_layer_ids[name] = int(lid)

    def _find_layer_id(mod_name: str) -> Optional[int]:
        """Walk up the dotted name to find the nearest enclosing block layer_id."""
        if mod_name in block_layer_ids:
            return block_layer_ids[mod_name]
        parts = mod_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in block_layer_ids:
                return block_layer_ids[parent]
        return None

    n = 0
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls not in block_classes:
            continue
        lid = _find_layer_id(name)
        if lid is None:
            continue
        if wanted is not None and lid not in wanted:
            continue
        mod.register_forward_hook(_make_hook(lid, cls))
        n += 1
    return n


# === Weight dump =====================================================


def maybe_dump_weights_and_exit(model: torch.nn.Module) -> None:
    """Dump per-layer params + buffers to ATOM_WEIGHT_DUMP_DIR, then sys.exit(0).

    No-op when ATOM_WEIGHT_DUMP_DIR is unset. Skips expert weights (FP4 packed,
    too large; weight loading is verified separately for those).

    Each rank writes its own file: weight_rank{R}_layer{L}.pt with keys:
      _tp_rank, _tp_size, _layer, plus all param/buffer names containing
      'layers.{L}.' and not '.experts.'.
    """
    dump_dir = envs.ATOM_WEIGHT_DUMP_DIR
    if not dump_dir:
        return

    os.makedirs(dump_dir, exist_ok=True)
    wanted = [int(x) for x in envs.ATOM_WEIGHT_DUMP_LAYERS.split(",") if x]
    rank = _get_rank()
    world = _get_world_size()

    for layer in wanted:
        prefix = f"layers.{layer}."
        pkt: dict = {"_tp_rank": rank, "_tp_size": world, "_layer": layer}
        for n, p in model.named_parameters():
            if prefix in n and ".experts." not in n:
                pkt[n] = p.detach().cpu()
        for n, b in model.named_buffers():
            if prefix in n and ".experts." not in n:
                pkt[f"buffer:{n}"] = b.detach().cpu()
        out = os.path.join(dump_dir, f"weight_rank{rank}_layer{layer}.pt")
        torch.save(pkt, out)

    if envs.ATOM_WEIGHT_DUMP_EXIT:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)


# === Sampler top-K dump ==============================================


def maybe_log_topk(logits: torch.Tensor, prefix: str = "") -> None:
    """Log top-K (id, prob) pairs per row. No-op when ATOM_DEBUG_TOPK == 0.

    Writes one line per row to ATOM_DEBUG_TOPK_PATH (or stderr if unset).
    Only rank 0 writes (TP-replicated logits).
    """
    k = envs.ATOM_DEBUG_TOPK
    if k <= 0 or logits.ndim != 2:
        return
    if _get_rank() != 0:
        return

    probs = logits.float().softmax(dim=-1)
    top = probs.topk(k, dim=-1)
    out_path = envs.ATOM_DEBUG_TOPK_PATH
    fp = open(out_path, "a", encoding="utf-8") if out_path else sys.stderr
    try:
        for row in range(logits.size(0)):
            triples = " ".join(
                f"{int(top.indices[row, j].item())}:"
                f"{float(top.values[row, j].item()):.3f}"
                for j in range(k)
            )
            print(f"{prefix}row{row} top{k}: {triples}", file=fp, flush=True)
    finally:
        if out_path:
            fp.close()
