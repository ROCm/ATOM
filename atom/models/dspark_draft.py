# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared surface for the DSpark block drafters.

Every DSpark flavor -- the inline DeepSeek-V4 draft that ships inside the target
checkpoint (``deepseek_v4_dspark.py``), the standalone Kimi-K3 draft
(``kimi_k3_dspark.py``), whatever lands next -- owes ``DSparkProposer`` the same
two things: absorb the target's context into the draft's own KV, then draft a
whole block in one backbone pass.

Absorbing the context has the same shape in every flavor, and that shape is the
only thing implemented here:

    project the concatenated aux hidden states ONCE, then hand that single
    tensor to every layer, each of which recomputes KV from it using its OWN
    weights (the q half of the fused projection is computed and dropped).

Below that line the flavors diverge at every step, so the per-layer write stays
in the subclass. V4 writes a 512-wide MQA row into a private rolling window
addressed by absolute position, with the RMSNorm covering the RoPE lanes and
fp8 QAT on the rest; K3 writes a 576-wide MLA latent into its rows of the
engine's paged pool addressed by slot mapping, and norms only the compressed
half. The steps line up one for one and not one of them is the same computation
-- which is why the seam is here, above the per-layer write, and not inside it.

The heads after the backbone are the same in every flavor, so they live here
too: ``DSparkMarkovHead``, ``DSparkConfidenceHead`` and ``linear_out``.
"""

import torch
from torch import nn

from atom.utils import envs


def linear_out(output):
    """ATOM quantized linears may return (tensor, scale); take the tensor."""
    return output[0] if isinstance(output, tuple) else output


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order Markov transition bias over the draft logits.

    The full ``V x V`` transition matrix is factorized as ``B = W1 @ W2^T`` with
    ``W1, W2 in R^{V x r}`` (checkpoint: ``markov_w{1,2}.weight [V, r]``). Given
    the previously sampled token ``x_{k-1}``, the bias added to position ``k``'s
    base logits is ``W1[x_{k-1}] @ W2^T in R^V``. ``r = dspark_markov_rank``
    keeps both storage and per-step compute small, so the sequential sampling
    loop stays light relative to the parallel backbone.

    Because the bias is added INSIDE a per-position softmax (a local correction,
    not a global renormalization), per-token probabilities stay exact, which is
    what lets speculative verification remain lossless.

    Both tables stay replicated across TP ranks: the bias is added to the
    full-vocab logits the shared LM head produces, so a sharded ``W2`` would have
    to be gathered anyway.
    """

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Embedding(vocab_size, rank)
        # Read once here rather than per sampled position: envs re-reads the
        # environment on every attribute access.
        self.fused_sample = envs.ATOM_DSPARK_FUSED_MARKOV_SAMPLE

    def forward(self, token_ids: torch.Tensor):
        """Args:
            token_ids: [*]  ids of the previously sampled token x_{k-1}.
        Returns:
            logits_bias:  [*, V]  bias to add to the next position's logits.
            markov_embed: [*, r]  W1[x_{k-1}], reused by the confidence head.
        """
        markov_embed = self.markov_w1(token_ids)
        # fp32 matmul: the bias enters the softmax that gates acceptance.
        logits_bias = torch.matmul(
            markov_embed.float(), self.markov_w2.weight.float().t()
        )
        return logits_bias, markov_embed

    def sample_next(
        self, token_ids: torch.Tensor, base_logits: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """One greedy block position: the argmax of the biased logits into `out`.

        The bias itself is never returned, which is what lets the fused path
        skip materializing it: ``dspark_markov_argmax`` keeps ``W2`` bf16 and
        reduces straight to ids, with the same fp32 accumulation the softmax
        guarantee above asks for (see that module for the numerics argument).
        ``markov_embed`` is the return because the confidence head consumes it;
        the ids are not, so that `out` stays the only place to read them whether
        or not the caller is compiled.

        Args:
            token_ids:   [B]     ids of the previously sampled token x_{k-1}.
            base_logits: [B, V]  this position's base logits.
            out:         [B]     where the ids land; a strided view is fine,
                                 which is how the caller's own block becomes
                                 the destination instead of a copy's source.
        Returns:
            markov_embed: [B, r]  W1[x_{k-1}].
        """
        if self.fused_sample:
            # Imported here, not at module scope: the op imports Triton and
            # AITER, and this module stays importable without them.
            from atom.model_ops.dspark_markov_sample import dspark_markov_argmax

            return dspark_markov_argmax(
                base_logits,
                token_ids,
                self.markov_w1.weight,
                self.markov_w2.weight,
                out,
            )
        bias, markov_embed = self(token_ids)
        # bf16 + fp32 promotes the slice to fp32 before the add, so an explicit
        # .float() would only materialize it twice for the same sum.
        out.copy_((base_logits + bias).argmax(dim=-1))
        return markov_embed


class DSparkConfidenceHead(nn.Module):
    """Per-position survival-probability estimator.

        c_k = sigma( w^T [ h_k ; W1[x_{k-1}] ] (+ b) )

    Input is the concatenation of ``h_k`` (hidden) and the Markov embedding
    ``W1[x_{k-1}]`` (rank), so the projection weight is ``[1, hidden + rank]``.
    Which ``h_k`` a flavor feeds is part of its checkpoint's training: V4 feeds
    the PRE-norm mHC head reduction, MiniMax-M3.1 the post-final-norm hidden the
    LM head reads. Only MiniMax-M3.1's projection has a bias.

    The sigmoid lives here rather than in the caller: VerifyScheduler consumes
    ``c_k`` as an absolute probability, so the (0, 1) range is part of this
    module's contract.
    """

    def __init__(self, hidden_size: int, rank: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(hidden_size + rank, 1, bias=bias)

    def forward(
        self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            hidden_states: [*, hidden]  h_k.
            markov_embeds: [*, rank]    W1[x_{k-1}].
        Returns:
            confidence: [*]  sigmoid survival probability in (0, 1).
        """
        # FP32 out (the downstream scheduler needs calibrated absolute
        # probabilities). The checkpoint head is BF16 [1, hidden + rank]: the
        # GEMV takes it as is and accumulates and writes FP32, which is what
        # casting both operands to FP32 first would compute, minus the casts.
        # aiter is imported here, not at module load: this module imports
        # without it.
        x = torch.cat([hidden_states, markov_embeds], dim=-1)
        weight, bias = self.proj.weight, self.proj.bias
        if not x.is_cuda:
            logit = torch.nn.functional.linear(
                x.float(), weight.float(), None if bias is None else bias.float()
            )
            return torch.sigmoid(logit).squeeze(-1)

        import aiter
        from aiter.tuned_gemm import tgemm

        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        logit = tgemm.mm(x.reshape(-1, x.shape[-1]), weight, otype=torch.float32)
        if bias is not None:
            logit = logit + bias.float()
        return aiter.sigmoid(logit).view(x.shape[:-1])


class DSparkDraftModel(nn.Module):
    """What a DSpark draft model must provide to ``DSparkProposer``."""

    supports_block_graph = True
    # Whether the block attends through MLA decode, whose persistent metadata
    # the proposer plans. A draft that attends its block another way (MiniMax-
    # M3.1's per-row MHA decode) builds what it reads in `prepare_paged_block`.
    block_attention_is_mla = True

    def prepare_block(self, metadata_builder, num_draft, scheduled_bs, running_bs):
        """Mask padded writes for models whose draft pass mutates private KV."""
        return

    @property
    def first_attention_layer_num(self) -> int:
        """``kv_cache_data`` key of this draft's first attention layer.

        Paged MLA drafts bind that layer on ``layers[0].self_attn.mla_attn``;
        this is the lookup ``DSparkProposer`` used before the hook existed, so
        Kimi-K3 and every other MLA draft keep resolving without an override.
        A draft whose attention module lives somewhere else (MiniMax-M3.1's
        ``KVCacheBinding``) overrides this. Inline drafts that keep a private
        KV window never reach the call site.
        """
        return self.layers[0].self_attn.mla_attn.layer_num

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page ids
        block_positions: torch.Tensor,  # [running_bs, T]
        slots: torch.Tensor,  # [running_bs, T] the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
    ) -> None:
        """The per-step metadata only a non-MLA block attention reads, built
        outside the captured block from the proposer's paged block metadata."""
        return

    def write_context_kv(
        self,
        aux_concat: torch.Tensor,  # [N, target_hidden * num_target_layers]
        positions: torch.Tensor,  # [N] absolute positions
    ) -> None:
        """Absorb one target forward's hidden states into this draft's KV.

        Where those rows land -- a private rolling window, a paged pool -- and
        how they are addressed is the layer's business, not this method's. A
        layer reads its own addressing (``cu_seqlens_q``, ``slot_mapping``, ...)
        off the live forward context rather than taking it as an argument,
        because the two flavors need different pieces of the same metadata.

        That makes the call site load-bearing: this must run while
        ``attn_metadata`` still describes the TARGET forward that produced
        ``aux_concat``. A drafter that has already retargeted the metadata to
        its own block pass (see ``DSparkProposer._propose_with_draft``) is
        past the point where these rows can be placed.
        """
        from atom.utils.forward_context import get_forward_context

        # warmup_model() runs at the end of ModelRunner.__init__, BEFORE
        # allocate_kv_cache(), so on a dummy run there is nothing to write into:
        # every layer's cache is still the empty init tensor and the store would
        # abort inside the cache kernels. Skipping is safe -- warmup discards
        # the draft's output. The block forward needs no such guard; it is also
        # the memory-profiling pass, so it must still run.
        if get_forward_context().context.is_dummy_run:
            return
        ctx_hidden = self.project_context(aux_concat)
        for layer in self.context_layers:
            layer.write_context_kv(ctx_hidden, positions)

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """Fuse the concatenated target aux hidden states into one [N, hidden]
        context tensor, shared by every layer."""
        raise NotImplementedError

    @property
    def context_layers(self):
        """The layers that hold context KV, in order."""
        raise NotImplementedError

    def forward_spec(
        self,
        input_ids: torch.Tensor,  # [B] verified anchor token per request
        positions: torch.Tensor,  # flavor-defined, see `block_backbone`
        num_draft: "int | None" = None,
    ):
        """One DSpark block: parallel backbone pass + sequential Markov
        sampling. Returns ``(draft_token_ids [B, T], confidence | None)``.

        Implement as the composition of the two halves below, so the whole
        block and the captured block cannot drift apart."""
        raise NotImplementedError

    def block_backbone(
        self,
        input_ids: torch.Tensor,  # [B] verified anchor token per request
        positions: torch.Tensor,  # flavor-defined, see below
        num_draft: int,
    ):
        """The parallel half: embed the block, run the layers, norm.

        Returns whatever this flavor's ``head_and_sample`` takes -- a hidden
        tensor, a tuple of them, its own business. Nothing in between looks
        inside: the proposer captures the backbone as a pass and the head as
        its epilogue, and hands one's return straight to the other.

        ``positions`` is the one argument the flavors disagree on: K3 takes the
        block's absolute positions ``[B*T]``, because its caller built the paged
        slot mapping from those exact numbers; V4 takes the anchor position
        ``[B]`` and expands inside its compiled region.
        ``DSparkProposer._block_positions`` is the only place that knows which.
        """
        raise NotImplementedError

    def head_and_sample(self, out, anchor_ids: torch.Tensor, num_draft: int):
        """The sequential half: LM head, then Markov sampling over the block.

        ``out`` is what ``block_backbone`` returned; ``anchor_ids`` is the one
        argument whose shape says how many requests there are.
        """
        raise NotImplementedError
