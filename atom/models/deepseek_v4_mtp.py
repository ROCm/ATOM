# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 MTP wrapper for ATOM's EagleProposer.

Mirrors the V2/V3 + Qwen MTP convention:

1. The target (`DeepseekV4ForCausalLM`) constructs and loads ONLY the main
   stack — no MTP modules, no MTP weights.

2. This wrapper owns the MTP block(s) and loads `mtp.{i}.*` ckpt entries
   into them via the standard `load_model` path (with `spec_decode=True`).
   `weights_mapper`, `packed_modules_mapping`, `get_expert_mapping`, and
   `remap_mtp_weight_name` are declared so the loader recognizes V4 MTP
   ckpt naming + filters out target-only entries silently.

3. `EagleProposer.load_model` then calls `share_with_target(...)` which
   rebinds `self.model.{embed,head}` to point at the target's already-loaded
   instances, and propagates them onto each MTPBlock (which requires
   `embed`/`head` set externally). No second weight load, no double KV.

4. V4's MTP block consumes the **un-reduced mHC residual stack** `[N, hc, dim]`
   from the target — enabled by deferring `hc_head + RMSNorm + LM head` from
   `DeepseekV4Model.forward` to `DeepseekV4ForCausalLM.compute_logits`. The
   wrapper's `forward` contract matches ATOM's EagleProposer:
     - input  `hidden_states` is the target's `[N, hc, dim]` residual
     - output is `[N, dim]` post-(MTP block + its own hc_head + norm)
"""

from typing import Optional

import torch
from torch import nn

from atom.config import Config
from atom.model_loader.loader import WeightsMapper
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.utils.forward_context import get_forward_context

from .deepseek_v4 import (
    Block,
    DeepseekV4Args,
    HCState,
    ParallelHead,
    make_v4_quant_config,
)


class MTPBlock(Block):
    """MTP block: V4 dense block + e_proj/h_proj/enorm/hnorm + own hc_head params + LM head.

    Port of inference/model.py:739-767. Subclass of Block reusing all HC + Attention + FFN
    machinery; adds a token-embed projection (`e_proj`), a hidden-state projection
    (`h_proj`), per-input RMSNorms, and its own `hc_head_fn/base/scale` parameters
    for the final LM head reduction.

    `embed` and `head` are assigned externally by the wrapper (shared with
    the target's embedding and LM head via `share_with_target`).
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__(layer_id, args, prefix=prefix)
        # e_proj / h_proj are FP8 on disk per index; ATOM Linear with V4QuantConfig
        # picks per_1x128 automatically. nn.Linear at construction works for the
        # toy/dummy path; for real-checkpoint loading, switch to ReplicatedLinear.
        qc = args.quant_config
        if qc is None:
            self.e_proj = nn.Linear(args.dim, args.dim, bias=False)
            self.h_proj = nn.Linear(args.dim, args.dim, bias=False)
        else:
            self.e_proj = ReplicatedLinear(
                args.dim,
                args.dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.e_proj",
            )
            self.h_proj = ReplicatedLinear(
                args.dim,
                args.dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.h_proj",
            )
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        # Per-MTP hc_head params (distinct from Block's hc_attn/hc_ffn params).
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = atom_parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = atom_parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = atom_parameter(torch.empty(1, dtype=torch.float32))
        # Externally-assigned by the wrapper (shared with the target).
        self.embed: Optional[nn.Module] = None
        self.head: Optional[ParallelHead] = None

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]  residual stream from main model
        positions: torch.Tensor,  # [num_tokens] int  absolute positions
        input_ids: torch.Tensor,  # [num_tokens] int
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  pre-hc_head residual
        """Run one MTP step. Returns the un-reduced mHC residual stack
        `[num_tokens, hc, dim]` — same shape contract as
        `DeepseekV4Model.forward`. The hc_head reduction + RMSNorm + LM head
        are all deferred to `DeepseekV4MTP.compute_logits` so the wrapper's
        forward output can be fed back in as `x` for the NEXT MTP draft
        step (mtp_k > 1) without re-expanding from a `[N, dim]` post-reduction
        state.
        """
        assert (
            self.embed is not None
        ), "MTPBlock requires .embed to be assigned by the wrapper"
        e = self.enorm(self.embed(input_ids))  # [num_tokens, dim]
        x = self.hnorm(x)  # [num_tokens, hc, dim]
        # Mix token-embed + hidden into a fresh residual. h_proj is FP8
        # (V4QuantConfig → ReplicatedLinear over `gemm_a8w8_blockscale_preshuffle`),
        # which only supports 2D `[M, K]` input. Flatten the hc axis into the
        # batch dim and reshape back after the projection. e_proj's input is
        # already 2D [num_tokens, dim]; unsqueeze adds the hc axis so it
        # broadcasts over h_proj_out's hc dim.
        n_tok, hc, d = x.shape
        h_proj_out = self.h_proj(x.reshape(n_tok * hc, d)).reshape(n_tok, hc, d)
        x = self.e_proj(e).unsqueeze(-2) + h_proj_out  # [num_tokens, hc, dim]
        hc_state = HCState(residual=x, post_mix=None, comb_mix=None, x_prev=None)
        hc_state = super().forward(hc_state, positions)
        return self.hc_post(
            hc_state.x_prev,
            hc_state.residual,
            hc_state.post_mix,
            hc_state.comb_mix,
        )


class DeepseekV4MTPModel(nn.Module):
    """V4 MTP inner model: owns the MTP blocks. Each block's `embed` / `head`
    are set externally by `DeepseekV4MTP.share_with_target` to point at the
    already-loaded target instances; this module itself holds no embed/head.
    """

    def __init__(self, atom_config: Config, args: DeepseekV4Args) -> None:
        super().__init__()
        # ModelRunner reads `drafter.model.model.mtp_start_layer_idx` to bind
        # the draft attention to KV slots `[n_layers, n_layers + n_mtp)`.
        self.mtp_start_layer_idx = args.n_layers
        # Real MTP blocks loaded via the standard load_model path with
        # spec_decode=True. Each block's `.embed` / `.head` are set by
        # share_with_target after load.
        self.mtp = nn.ModuleList(
            [
                MTPBlock(args.n_layers + i, args, prefix=f"mtp.{i}")
                for i in range(args.n_mtp_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int
        positions: torch.Tensor,  # [num_tokens] int
        hidden_states: torch.Tensor,  # [num_tokens, hc, dim]  pre-hc_head from target
        spec_step_idx: int = 0,
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  pre-hc_head residual
        """Returns the un-reduced mHC residual `[N, hc, dim]` from the
        selected MTP block — same shape contract as `DeepseekV4Model.forward`.
        EagleProposer feeds this back in as `hidden_states` for the NEXT
        draft step (mtp_k > 1)."""
        idx = spec_step_idx % len(self.mtp)
        return self.mtp[idx](hidden_states, positions, input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hc, dim]  pre-hc_head residual
        spec_step_idx: int = 0,
    ) -> torch.Tensor:  # [bs, vocab]
        """Mirror `DeepseekV4ForCausalLM.compute_logits`: hc_head + RMSNorm +
        LM head all here so MTPBlock.forward can return the un-reduced
        residual stack. Each MTP block has its own `hc_head_fn/base/scale`
        and own `norm` (the LM head is target-shared via `share_with_target`)."""
        idx = spec_step_idx % len(self.mtp)
        blk = self.mtp[idx]
        x = blk.head.hc_head(
            hidden_states, blk.hc_head_fn, blk.hc_head_scale, blk.hc_head_base
        )  # [num_tokens, dim]
        x = blk.norm(x)
        return blk.head.get_logits(x)


class DeepseekV4MTP(nn.Module):
    """Top-level V4 MTP wrapper. Owns its own MTP weights (loaded via the
    standard `load_model` path with `spec_decode=True`) and shares only
    `embed` / `head` with the target through `share_with_target`.
    """

    # Disk `mtp.{i}.*` -> wrapper param `model.mtp.{i}.*`. Prefix-anchored to
    # avoid the `mtp.` substring colliding with anything inside MTPBlock subtree.
    weights_mapper = WeightsMapper(orig_to_new_prefix={"mtp.": "model.mtp."})
    # Same on-disk -> internal name conventions as the target (V4 ckpt quirks).
    weights_mapping = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }
    # Same packed-module fusions as the target — MTPBlock subclasses Block, so
    # the same attention / shared-experts fused param layouts apply.
    packed_modules_mapping = {
        "attn.wq_a": ("attn.wqkv_a", 0),
        "attn.wkv": ("attn.wqkv_a", 1),
        "compressor.wkv": ("compressor.wkv_gate", 0),
        "compressor.wgate": ("compressor.wkv_gate", 1),
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
    }

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        self.args.quant_config = make_v4_quant_config(
            self.hf_config,
            online_quant_config=getattr(config, "online_quant_config", None),
        )
        self.atom_config.quant_config = self.args.quant_config
        self.model = DeepseekV4MTPModel(atom_config=config, args=self.args)

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Filter loader input to MTP-only weights.

        Called per ckpt entry AFTER `weights_mapper` rewrites `mtp.` ->
        `model.mtp.`. Target-only entries (`embed.weight`, `layers.X.*`,
        `head.weight`, `hc_head_*`, `norm.weight`) pass through unchanged
        and have no matching wrapper param, which would otherwise generate
        loud `dropped_ckpt_keys` warnings. Returning None drops them silently.
        """
        return name if "mtp." in name else None

    @property
    def disable_fused_shared_loading(self) -> bool:
        """True when MTP shared experts are standalone, not fused into FusedMoE."""
        for m in self.model.modules():
            if m.__class__.__name__ == "MoE":
                return not getattr(m, "_fuse_shared_into_routed", True)
        return False

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """FusedMoE expert param mapping for MTPBlock's MoE layer. Same
        ckpt convention as the target (`ffn.experts.{e}.w{1,2,3}`).
        """
        num_fused_shared = 0
        for m in self.model.modules():
            if m.__class__.__name__ == "FusedMoE":
                num_fused_shared = getattr(m, "num_fused_shared_experts", 0)
                break
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.args.n_routed_experts + num_fused_shared,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Hash MoE routing inside MTPBlock's MoE looks at this — same as
        # `DeepseekV4ForCausalLM.forward`.
        get_forward_context().context.input_ids = input_ids
        return self.model(input_ids, positions, hidden_states, spec_step_idx)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def share_with_target(self, target_base: nn.Module, loaded: set[str]) -> None:
        """Bind embed/head on each MTPBlock to the already-loaded target's
        instances. MTPBlock requires `.embed` / `.head` set externally before
        its first forward; `compute_logits` then reaches them via `mtp[0].head`.
        """
        for blk in self.model.mtp:
            blk.embed = target_base.model.embed
            blk.head = target_base.model.head
