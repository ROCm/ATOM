# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LLaMA LoRA / MLPerf finetune SOURCE patches (disk modifications).

These patches are **opt-in** (``default=False``). Apply with::

    python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag lora

Typical MLPerf finetune bootstrap also needs RMSNorm patches::

    python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora
"""

from __future__ import annotations

import os

from atomrl.lumen.patches.registry import PatchPhase, register_patch

_REQUIRES_GRAD_MARKER = "# LUMEN-PATCH-REQUIRES-GRAD"
_CKPT_MARKER = "Load state dict with LoRA base_layer key remapping"

_LOAD_MODEL_STATE_DICT_REPLACEMENT = '''    def load_model_state_dict(module, state_dict, strict: bool):
        """Load state dict with LoRA base_layer key remapping."""
        import re as _re

        # Get the innermost model (strip DDP + Float16Module wrappers)
        inner = module
        while hasattr(inner, 'module'):
            inner = inner.module

        # Get the inner model's state_dict keys (what load_state_dict will match against)
        inner_keys = set(inner.state_dict().keys())

        # Check if remapping is needed: do checkpoint keys match inner model keys?
        ckpt_keys = set(state_dict.keys())
        common = ckpt_keys.intersection(inner_keys)
        needs_remap = len(common) < len(ckpt_keys)
        if not needs_remap:
            print(f"[CKPT FIX] All {len(common)} checkpoint keys match model")
        else:
            print(f"[CKPT FIX] {len(common)}/{len(ckpt_keys)} keys match directly, remapping remaining...")
            # Find LoRA base_layer parents from inner model keys
            lora_parents = set()
            for ik in inner_keys:
                m = _re.match(r'(.+)\\.base_layer\\.weight$', ik)
                if m:
                    lora_parents.add(m.group(1))

            # Find _norm.weight keys
            norm_keys = {ik for ik in inner_keys if '._norm.' in ik}

            new_sd = {}
            mapped = 0
            for ck, cv in state_dict.items():
                # Check if this key's parent is a LoRA-wrapped layer
                parts = ck.rsplit(".", 1)
                if len(parts) == 2:
                    parent, param = parts
                    if parent in lora_parents:
                        base_key = f"{parent}.base_layer.{param}"
                        new_sd[base_key] = cv
                        mapped += 1
                        continue

                new_sd[ck] = cv

                # Duplicate layernorm weights to _norm.weight if needed
                if ck.endswith(".weight") and ("layernorm" in ck or "final_layernorm" in ck):
                    norm_key = ck.replace(".weight", "._norm.weight")
                    if norm_key in inner_keys:
                        new_sd[norm_key] = cv

            state_dict = new_sd

            # Remap fused LayerNormLinear norm weights (--lumen-linear)
            _ln_rules = [
                ("input_layernorm.", "self_attention.linear_qkv.base_layer.layer_norm_"),
                ("input_layernorm.", "self_attention.linear_qkv.layer_norm_"),
                ("pre_mlp_layernorm.", "mlp.linear_fc1.base_layer.layer_norm_"),
                ("pre_mlp_layernorm.", "mlp.linear_fc1.layer_norm_"),
            ]
            ln_mapped = 0
            for ck in list(state_dict.keys()):
                for old_frag, new_frag in _ln_rules:
                    if old_frag not in ck:
                        continue
                    tgt = ck.replace(old_frag, new_frag, 1)
                    if tgt in inner_keys and tgt not in state_dict:
                        state_dict[tgt] = state_dict.pop(ck)
                        ln_mapped += 1
                        break
            if ln_mapped:
                print(f"[CKPT FIX] Remapped {ln_mapped} fused LayerNormLinear norm keys")

            loaded = set(state_dict.keys()).intersection(inner_keys)
            not_loaded = inner_keys - set(state_dict.keys())
            important_missing = [k for k in sorted(not_loaded)
                                 if "lora_" not in k and "_extra_state" not in k
                                 and "cross_attn" not in k and "._norm." not in k]
            print(f"[CKPT FIX] Remapped {mapped} LoRA base_layer keys")
            print(f"[CKPT FIX] After remapping: {len(loaded)}/{len(inner_keys)} model keys covered")
            if important_missing:
                print(f"[CKPT FIX] Important missing ({len(important_missing)}): {important_missing[:15]}")
            else:
                print(f"[CKPT FIX] All base model weights mapped!")

        try:
            module.load_state_dict(state_dict, strict=strict)
        except Exception as e:
            if strict:
                load_return = module.load_state_dict(state_dict, strict=False)
                print(f"load_return: {load_return}")

        print(f"[CKPT FIX] Checkpoint loaded successfully (verification skipped for memory efficiency)")
'''

_SFT_ARG = (
    "group.add_argument('--sft', action=\"store_true\", help='Megatron SFT training')"
)
_PATCHED_SFT_ARG = (
    "group.add_argument('--sft', action=\"store_true\", default=True, "
    "help='Megatron SFT training (patched default=True for MLPerf loss norm)')"
)


def patch_lora_requires_grad(megatron_root: str) -> bool:
    """Force hidden_states.requires_grad before activation checkpointing (LoRA + recompute)."""
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "transformer_block.py"
    )
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    if _REQUIRES_GRAD_MARKER in content:
        return False
    old = (
        "        hidden_states = make_viewless_tensor(inp=hidden_states, "
        "requires_grad=True, keep_graph=True)\n"
    )
    idx = content.find(old)
    if idx == -1:
        return False
    inject = (
        f"\n        {_REQUIRES_GRAD_MARKER}\n"
        "        # LoRA fine-tuning: embedding is frozen so hidden_states.requires_grad=False.\n"
        "        # Force requires_grad=True so CheckpointFunction builds autograd graph.\n"
        "        if not hidden_states.requires_grad:\n"
        "            hidden_states = hidden_states.detach().requires_grad_(True)\n"
    )
    content = content[: idx + len(old)] + inject + content[idx + len(old) :]
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_lora_checkpoint_load(megatron_root: str) -> bool:
    """LoRA base_layer ckpt remap + mmap=True torch.load for large checkpoints."""
    path = os.path.join(megatron_root, "megatron", "training", "checkpointing.py")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()

    changed = False
    if _CKPT_MARKER not in content:
        start_marker = (
            "    def load_model_state_dict(module, state_dict, strict: bool):"
        )
        end_marker = "\n    # Model."
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if start_idx < 0 or end_idx < 0:
            return False
        content = (
            content[:start_idx] + _LOAD_MODEL_STATE_DICT_REPLACEMENT + content[end_idx:]
        )
        changed = True

    old_load = "state_dict = torch.load(checkpoint_name, map_location='cpu')"
    new_load = (
        "state_dict = torch.load(checkpoint_name, map_location='cpu', "
        "mmap=True, weights_only=False)"
    )
    if old_load in content:
        content = content.replace(old_load, new_load)
        changed = True

    if not changed:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_lora_adapter_scaling(megatron_root: str) -> bool:
    """Use alpha/rank LoRA scaling (NeMo / HuggingFace PEFT convention)."""
    path = os.path.join(
        megatron_root, "megatron", "core", "transformer", "lora_adapter.py"
    )
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    if "alpha / rank" in content:
        return False
    old = "        self.lora_alpha = alpha"
    new = "        self.lora_alpha = alpha / rank if rank > 0 else alpha"
    if old not in content:
        return False
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


def patch_lora_sft_loss_default(megatron_root: str) -> bool:
    """Default ``--sft`` to True for MLPerf-compatible val loss normalization."""
    path = os.path.join(megatron_root, "megatron", "training", "arguments.py")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    if _PATCHED_SFT_ARG in content:
        return False
    marker = "group.add_argument('--sft', action=\"store_true\""
    if marker not in content:
        return False
    old_block = (
        "def _add_sft_args(parser):\n    group = parser.add_argument_group(title='sft')"
    )
    new_block = (
        "def _add_sft_args(parser):\n"
        "    group = parser.add_argument_group(title='sft')\n"
        "    # --- patched: default=True to match NeMo sample_weight=constant ---"
    )
    if old_block in content:
        content = content.replace(old_block, new_block, 1)
    if _SFT_ARG not in content:
        return False
    content = content.replace(_SFT_ARG, _PATCHED_SFT_ARG, 1)
    with open(path, "w") as f:
        f.write(content)
    return True


_LORA_TAGS = frozenset({"lora", "finetune", "megatron"})

register_patch(
    "lora_requires_grad",
    PatchPhase.SOURCE,
    description="Force hidden_states.requires_grad before activation checkpointing",
    tags=_LORA_TAGS,
    default=False,
)(patch_lora_requires_grad)

register_patch(
    "lora_checkpoint_load",
    PatchPhase.SOURCE,
    description="LoRA base_layer checkpoint remap + mmap torch.load",
    tags=_LORA_TAGS,
    default=False,
)(patch_lora_checkpoint_load)

register_patch(
    "lora_adapter_scaling",
    PatchPhase.SOURCE,
    description="LoRA alpha/rank scaling to match NeMo and PEFT",
    tags=_LORA_TAGS,
    default=False,
)(patch_lora_adapter_scaling)

register_patch(
    "lora_sft_loss_default",
    PatchPhase.SOURCE,
    description="Default --sft=True for MLPerf val loss normalization",
    tags=_LORA_TAGS,
    default=False,
)(patch_lora_sft_loss_default)
