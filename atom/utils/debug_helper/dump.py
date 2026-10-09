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

import torch

from atom.utils import envs

# === helpers =========================================================


def _parse_layer_set(env_value: str) -> set[int] | None:
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


def _output_tensor(output) -> torch.Tensor | None:
    """Best-effort "the hidden state" out of whatever a block returned.

    Plain tensor and tuple-of-tensors cover most blocks. DeepSeek-V4 style mHC
    blocks instead return a state object carrying the residual stream in a
    field, so fall back to the first tensor attribute rather than silently
    dumping nothing.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("x_prev", "hidden_states", "hidden", "last_hidden_state"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    if isinstance(output, (tuple, list)):
        # State objects first: a hook's args are (hc_state, positions) and the
        # bare `positions` tensor would otherwise win and silently compare as
        # identical everywhere.
        for item in output:
            if not isinstance(item, torch.Tensor):
                found = _output_tensor(item)
                if found is not None:
                    return found
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _tensor_fields(obj) -> dict:
    """Every tensor attribute of a state object, keyed by attribute name.

    Returns {} for plain tensors/tuples — there is nothing extra to say about
    those beyond what `_output_tensor` already picked.
    """
    if isinstance(obj, torch.Tensor):
        return {}
    if isinstance(obj, (tuple, list)):
        for item in obj:
            if not isinstance(item, torch.Tensor):
                fields = _tensor_fields(item)
                if fields:
                    return fields
        return {}
    out = {}
    for name in getattr(obj, "__dataclass_fields__", ()) or dir(obj):
        if name.startswith("_"):
            continue
        # A non-dataclass state object is walked via `dir()`, which surfaces
        # properties that raise on access (`Tensor.imag` on a real dtype, a
        # lazily-built view whose backing buffer is unbound). A dump helper
        # must never be the thing that kills the forward pass.
        # Blind catch + silent skip on purpose: the raising property is not the
        # subject of the dump, and logging one line per attribute per layer per
        # forward would bury the dump it is meant to support.
        try:
            val = getattr(obj, name, None)
        except Exception:  # noqa: BLE001, S112
            continue
        if isinstance(val, torch.Tensor):
            out[name] = val.detach().cpu()
    return out


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

        def _hook(_mod, args, output):
            # A D2H copy is illegal mid-capture and aborts the whole graph.
            # CUDAGraph replay reruns the captured kernels without invoking
            # Python hooks anyway, so there is nothing to dump on this path.
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return
            if one_shot:
                if os.path.exists(one_shot_fname):
                    return
                fname = one_shot_fname
            else:
                key = (layer_id, cls_name)
                n = _call_counters.get(key, 0)
                _call_counters[key] = n + 1
                fname = f"{base}_call{n:03d}.pt"
            t = _output_tensor(output)
            if t is None:
                return
            # The input is saved alongside the output so a divergence can be
            # attributed to this block rather than inherited from the previous
            # one without dumping every layer.
            inp = _output_tensor(args)
            torch.save(
                {
                    "hidden": t.detach().cpu(),
                    "shape": tuple(t.shape),
                    "input": None if inp is None else inp.detach().cpu(),
                    # A block whose input is a multi-field state object (mHC
                    # carries residual / post_mix / comb_mix beside the hidden)
                    # can diverge through any field, so keep them all.
                    "input_fields": _tensor_fields(args),
                    "output_fields": _tensor_fields(output),
                },
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

    def _find_layer_id(mod_name: str) -> int | None:
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
            if dist.get_world_size() > 1:
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
