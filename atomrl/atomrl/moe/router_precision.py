# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP32 MoE router for train/rollout expert-selection agreement.

A routed-expert model picks top-k of N experts by comparing router logits. That
is a *discrete* decision, so any numerical disagreement between the training
forward (HF) and the rollout engine (vLLM fused MoE) can flip which experts a
token is sent to, which changes its log-prob by ~0.1 rather than ~1e-3.

With a BF16 router the logits are quantized coarsely enough that the gap between
the kept k-th and the dropped (k+1)-th expert is *exactly zero* for a large
fraction of tokens, and the two implementations then break the tie differently.
Measured on Qwen3-30B-A3B-Base (128 experts, top-8, 48 layers):

    router top-k margin percentiles : p10=0.0000  p25=0.0312  p50=0.0625
    margin < 1e-3                   : 11.65% of (token, layer) decisions
    top-k set changes if the same    :  6.40% of (token, layer) decisions
      router is recomputed in fp32
    => only ~4% of tokens route identically through all 48 layers
    => mean |rollout_logp - train_logp| = 0.027 with identical weights

Computing the gate in fp32 on *both* sides removes the quantization: margins
become continuous, so a flip needs a genuine near-tie rather than a bf16
rounding collision.

The gate parameters stay BF16 — only the matmul is promoted. That keeps the
state_dict dtype unchanged, so FSDP2 sharding and the ZMQ CUDA-IPC weight sync
are untouched.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

from atomrl.moe.rollout_routing import apply_replay, layer_index_of

logger = logging.getLogger(__name__)

_MARKER = "_lumen_fp32_router"

# vLLM quantization ``ignore`` entry matching the routers ``_is_router`` patches.
ROUTER_PATTERN = r"re:.*\.(mlp|ffn)\.gate$"


def fp32_router_enabled() -> bool:
    """Env override; on by default (MoEConfig.moe_router_dtype defaults to fp32)."""
    return os.environ.get("ATOMRL_FP32_MOE_ROUTER", "1") == "1"


def vllm_quantization_config(quantization: str | None) -> dict | None:
    """vLLM ``quantization_config`` for a rollout engine that gets the fp32 router.

    vLLM's online quantizations (``fp8_per_block``, ...) quantize every linear
    layer, the routers included, and the fp32 router needs the gate's real weights.
    Keeping the gates BF16 also keeps the rollout's routing on the trainer's
    weights; they are a few MB per model.
    """
    if not quantization or not fp32_router_enabled():
        return None
    from vllm.config.quantization import ONLINE_QUANT_SHORTHAND_NAMES

    if quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        return None
    return {"ignore": [ROUTER_PATTERN]}


def _is_router(name: str, mod: torch.nn.Module) -> bool:
    # Qwen names the router ".mlp.gate"; DSV4 also uses ".ffn.gate".
    if not name.endswith((".mlp.gate", ".ffn.gate")):
        return False
    w = getattr(mod, "weight", None)
    return isinstance(w, torch.Tensor) and w.dim() == 2


def _is_hf_topk_router(mod: torch.nn.Module) -> bool:
    """HF ``Qwen3MoeTopKRouter``: does the top-k itself, returns a 3-tuple."""
    return all(hasattr(mod, a) for a in ("top_k", "num_experts", "norm_topk_prob"))


def _patch_hf_topk_router(
    mod: torch.nn.Module, layer_idx: int | None, fp32: bool
) -> None:
    """Reimplement Qwen3MoeTopKRouter.forward with fp32 logits and an R3 hook.

    Upstream computes ``router_logits`` in BF16 and only upcasts at the softmax,
    which is too late — the ranking has already been decided by the rounded
    logits. Everything the block consumes downstream (logits and scores) is cast
    back to the original dtype, so only the *selection* changes.

    When Rollout Routing Replay has installed routing for the current
    micro-batch, the selection is replaced by the rollout's and the weights are
    regathered from the live probabilities, which keeps the router differentiable.
    """

    def forward(hidden_states, _m=mod, _layer=layer_idx, _fp32=fp32):
        out_dtype = _m.weight.dtype
        hidden_states = hidden_states.reshape(-1, _m.hidden_dim)
        if _fp32:
            router_logits = F.linear(hidden_states.float(), _m.weight.float())
        else:
            router_logits = F.linear(hidden_states, _m.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, _m.top_k, dim=-1)

        replayed = apply_replay(router_probs, router_indices, _layer)
        if replayed is not None:
            router_top_value, router_indices = replayed

        if _m.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        return (
            router_logits.to(out_dtype),
            router_top_value.to(out_dtype),
            router_indices,
        )

    mod.forward = forward


def _patch_linear_router(mod: torch.nn.Module, returns_tuple: bool) -> None:
    """Plain linear gate: promote the matmul and hand back fp32 logits.

    The caller (vLLM's FusedMoE runner, or an older HF block) does the top-k, so
    the logits must STAY fp32 — rounding them back to BF16 here would restore the
    ties this patch exists to remove.
    """

    def forward(x, _m=mod, _tuple=returns_tuple):
        bias = getattr(_m, "bias", None)
        out = F.linear(
            x.float(),
            _m.weight.float(),
            bias.float() if bias is not None else None,
        )
        return (out, None) if _tuple else out

    mod.forward = forward


def enable_fp32_moe_router(model: torch.nn.Module) -> int:
    """Patch every MoE router in ``model``. Returns #patched.

    Idempotent, and a no-op on dense models. Handles the HF ``Qwen3MoeTopKRouter``
    (top-k inside the router, 3-tuple return), vLLM's ``ReplicatedLinear`` gate
    (``(output, bias)`` return, top-k done by FusedMoE), and a plain
    ``nn.Linear`` gate.

    The HF patch is installed even when fp32 is switched off, because it is also
    where Rollout Routing Replay intercepts the top-k. fp32 then only controls
    the precision of the matmul inside it.
    """
    fp32 = fp32_router_enabled()

    patched = 0
    for name, mod in model.named_modules():
        if not _is_router(name, mod) or getattr(mod, _MARKER, False):
            continue

        if _is_hf_topk_router(mod):
            _patch_hf_topk_router(mod, layer_index_of(name), fp32)
        elif fp32:
            if mod.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                # Quantized weights keep their scales outside .weight, so an fp32
                # matmul over them routes every token to the wrong experts.
                raise RuntimeError(
                    f"fp32 MoE router: {name} is quantized ({mod.weight.dtype}); keep "
                    f"routers unquantized (vLLM quantization_config ignore "
                    f"[{ROUTER_PATTERN!r}]) or set ATOMRL_FP32_MOE_ROUTER=0"
                )
            _patch_linear_router(
                mod, returns_tuple=not isinstance(mod, torch.nn.Linear)
            )
        else:
            continue

        setattr(mod, _MARKER, True)
        patched += 1

    if patched:
        # print, not logger: this module runs inside Ray actors and vLLM workers
        # whose logging config swallows library-level records, and a long run
        # needs positive confirmation that the patch actually took effect.
        print(
            f"[atomrl] MoE router patched on {patched} gates (fp32={fp32})", flush=True
        )
    return patched
