"""vLLM-specific DeepSeek-V4 model.

This module reuses the native ATOM DeepSeek-V4 implementation
(:mod:`atom.models.deepseek_v4`) unchanged and only layers on the behaviour
that vLLM's graph-mode execution requires. The single vLLM-specific concern is
reconciling the padded CUDA-graph bucket width against the sparse-attention
metadata in the graph-break attention op (see ``DeepseekV4AttentionVllm``).

It follows the same construction-swap pattern as ``qwen3_next``: the
``DeepseekV4ForCausalLM`` subclass temporarily rebinds the module-global
``DeepseekV4Attention`` so the whole model tree is built with the vLLM attention
variant, then restores it.
"""

import torch

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4 import (
    DeepseekV4Attention as DeepseekV4AttentionBase,
    DeepseekV4ForCausalLM as DeepseekV4ForCausalLMBase,
    DeepseekV4Model as DeepseekV4ModelBase,
)
from atom.utils.forward_context import AttnState, get_forward_context


class DeepseekV4AttentionVllm(DeepseekV4AttentionBase):
    """DeepSeek-V4 attention with vLLM piecewise-CUDA-graph reconciliation.

    Under ``cudagraph_mode=FULL_AND_PIECEWISE`` vLLM captures/replays the dense
    regions of ATOM's torch.compiled graph at the padded bucket width, while the
    ``deepseek_v4_attention`` op (a graph break, marked as a splitting op) runs
    eagerly. So for a prefill/mixed batch whose bucket was captured, ``x`` /
    ``positions`` arrive padded to ``T_pad``, but the sparse-attention metadata
    is built for the *real* token count (the bridge's prefill path sets
    ``batch_id_per_token`` to length == real tokens).

    Slice the inputs to the real tokens before the (unchanged) native attention
    so per-token Q rows match the ``kv_indptr`` arrays — otherwise the
    paged-prefill kernel aborts with ``kv_indptr_prefix length must be N+1`` —
    then pad the output back to ``T_pad`` so the next captured dense region (and
    this op's ``empty_like(x)`` fake-meta) see the full bucket width.

    Decode is fully captured (incl. this op) with metadata already padded to the
    bucket, so it runs at the padded width and must NOT be sliced. The padded
    rows are never sampled (``logits_indices`` reference real positions only).
    """

    def forward_impl(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        fc = get_forward_context()
        # Dummy/bypass forwards short-circuit inside the native impl; defer to it
        # at full width (no metadata to reconcile against).
        if not fc.context.is_dummy_run:
            attn_md = fc.attn_metadata
            if attn_md is not None and attn_md.state is not AttnState.DECODE:
                num_in = x.size(0)
                bid = attn_md.batch_id_per_token
                num_real = bid.shape[0] if bid is not None else num_in
                if num_real < num_in:
                    out = super().forward_impl(x[:num_real], positions[:num_real])
                    return torch.nn.functional.pad(out, (0, 0, 0, num_in - num_real))
        return super().forward_impl(x, positions)


class DeepseekV4ModelVllm(DeepseekV4ModelBase):
    """DeepSeek-V4 model with a persistent MTP-draft hidden-state buffer.

    vLLM's DeepSeek-V4 MTP draft reads the target's pre-hc_head residual from
    Python *outside* the CUDAGraph (via the plugin's ``get_mtp_*`` hook). Under
    ``cudagraph_mode=FULL*`` the target model's Python ``forward`` body does not
    re-run on replay, so a plain Python stash of the return value would freeze
    the draft on the capture-time residual and draft acceptance collapses
    (~4%). Mirror vLLM's native DeepSeek-V4 ``_mtp_hidden_buffer``: allocate a
    stable-address buffer once (outside the graph pool) and refresh it every
    forward with an *in-graph* ``copy_`` that is captured and thus re-runs on
    every replay.

    This lives in the vLLM plugin only; native ATOM serving does not need it
    because its ModelRunner already routes the model output through a persistent
    ``forward_vars["outputs"]`` buffer that the drafter reads after replay.
    """

    def __init__(self, *, atom_config, args):
        super().__init__(atom_config=atom_config, args=args)
        hc_dim = self.hc_mult * self.args.dim
        sched_cfg = getattr(atom_config, "scheduler_config", None)
        max_num_batched_tokens = getattr(
            sched_cfg, "max_num_batched_tokens", None
        ) or getattr(atom_config, "max_num_batched_tokens", None)
        self._mtp_hidden_buffer = torch.empty(
            max_num_batched_tokens,
            hc_dim,
            dtype=self.embed.weight.dtype,
            device=self.embed.weight.device,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = super().forward(input_ids, positions)
        # In-graph copy_: captured into the CUDAGraph so it refreshes the buffer
        # on every replay, keeping the MTP draft's input hidden states current.
        num_tokens = h.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(h.flatten(1))
        return h


class DeepseekV4ForCausalLM(DeepseekV4ForCausalLMBase):
    """Native DeepSeek-V4 model built with the vLLM attention + model variants.

    Temporarily rebinds the module-global ``DeepseekV4Attention`` /
    ``DeepseekV4Model`` to their vLLM subclasses while the base ``__init__``
    constructs the tree (each layer does ``self.attn = DeepseekV4Attention(...)``
    and the wrapper does ``self.model = DeepseekV4Model(...)`` via those
    globals), then restores them. Class attributes used by the plugin wrapper
    (``weights_mapper`` / ``weights_mapping`` / ``packed_modules_mapping`` /
    ``extra_output_dims``) are inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        original_model_cls = deepseek_v4_base.DeepseekV4Model
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        deepseek_v4_base.DeepseekV4Model = DeepseekV4ModelVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
            deepseek_v4_base.DeepseekV4Model = original_model_cls
