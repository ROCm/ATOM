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
The standalone GQA drafts (Kimi-K3 DSpark GQA, MiniMax-M3.1) also share one layer
forward, ``DSparkGQADecoderLayer``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from atom.config import QuantizationConfig
from atom.spec_decode.draft_kv import DraftKvPoolSpec
from atom.utils import envs

if TYPE_CHECKING:
    from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore


@dataclass(frozen=True)
class DraftSampling:
    """Everything the block sampler needs to draw -- and publish -- a real q.

    Built once, when the runner has allocated the target's own sampling
    inputs (`DSparkLossless.bind_target`), before any warmup or capture: the block head is cudagraph-captured
    (``DSparkProposer._declare_draft_graphs``, ``capture_epilogue=True``), so
    every address here is baked into the recording. That is also why the
    per-step seed and row bases are device tensors rather than python ints:
    their values are refreshed outside it.

    The per-row tensors are sized at ``max_num_seqs``; the block runs at
    ``running_bs <= max_num_seqs`` and slices.
    """

    #: [slots * T, K] -- q, the distribution each draft token was drawn from
    #: (its top K entries; see `SparseQStore`), one block of T rows per live
    #: request. Read a step later by the accept kernel, so it outlives the pass
    #: and cannot live in the graph's private pool.
    q_store: "SparseQStore"
    #: [max_bs] int32, the first ``q_store`` row each batch row writes; block
    #: position k goes to row ``base + k``. Negative writes nothing.
    q_row_base: torch.Tensor
    #: [1] int32, the Philox key. Must hold the same value on every TP rank.
    seed: torch.Tensor
    #: Preallocated scores and partials for the sampled op
    #: (`dspark_markov_sample_scratch`), so the draft step allocates nothing and
    #: the profiling run sees it.
    scratch: torch.Tensor
    #: The target's own per-row sampling inputs (`ModelRunner.forward_vars`),
    #: so draft and target cannot disagree on them. [max_bs] fp32 temperatures,
    #: clamped to ``SAMPLER_EPS``, where a row at ``SAMPLER_EPS`` is greedy
    #: (`Sampler`'s predicate) and drafts its argmax.
    temperatures: torch.Tensor
    #: [max_bs] int32, the request's top-k the draft samples within; -1 is off.
    top_k: torch.Tensor
    #: [max_bs] fp32, the request's top-p, over the unfiltered q and
    #: intersected with top-k (aiter's rule); 1.0 is off.
    top_p: torch.Tensor


def linear_out(output):
    """ATOM quantized linears may return (tensor, scale); take the tensor."""
    return output[0] if isinstance(output, tuple) else output


def draft_rope_kwargs(config, *, is_neox_style: bool, rope_type=None) -> dict:
    """`get_rope` arguments from a standalone draft's OWN config.

    A draft is trained with its own rope, not the target's, and the context
    rows it writes are rotated with it, so it must be read from the draft
    config: the target's loads clean, so nothing would catch the draft
    inheriting it. The neox style and, for YaRN, the ``rope_type`` are the
    caller's, because they differ between drafts. ``rope_scaling`` is ``None``
    for a plain rope.
    """
    rope_params = dict(getattr(config, "rope_parameters", None) or {})
    rope_theta = rope_params.get("rope_theta") or getattr(config, "rope_theta", 10000.0)
    use_yarn = rope_params.get("rope_type") in ("yarn", "deepseek_yarn") or (
        rope_params.get("factor", 1.0) not in (1.0, None)
    )
    rope_scaling = None
    if use_yarn:
        rope_scaling = dict(rope_params)
        if rope_type is not None:
            rope_scaling["rope_type"] = rope_type
        rope_scaling.setdefault(
            "original_max_position_embeddings", config.max_position_embeddings
        )
    return {
        "max_position": config.max_position_embeddings,
        "base": rope_theta,
        "rope_scaling": rope_scaling,
        "is_neox_style": is_neox_style,
    }


def draft_quant_config(config, *, draft: str, accepted=()) -> QuantizationConfig:
    """A standalone draft's own quantization, independent of the target's.

    The atom config copied onto the draft carries the TARGET's offline and
    online quantization, which does not describe the draft checkpoint, so the
    draft's own block goes through the registered parser here; a draft config
    without one is BF16. ``accepted`` holds the ``(quant_type, quant_dtype)``
    pairs the ``draft`` runs besides BF16; anything else is refused.
    """
    draft_qc = QuantizationConfig(config)
    spec = draft_qc.global_spec
    if draft_qc.quant_method and (spec.quant_type, spec.quant_dtype) not in accepted:
        raise ValueError(
            f"the {draft} draft does not support quant_method="
            f"{draft_qc.quant_method!r} ({spec.quant_type}, {spec.quant_dtype})"
        )
    return draft_qc


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
            from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax

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

    def sample_block(
        self,
        base_logits: torch.Tensor,  # [B, T, V]
        anchor_ids: torch.Tensor,  # [B]
        sampling: "DraftSampling | None",
    ) -> torch.Tensor:
        """The block, left to right, each position from ``x_{k-1}`` into its
        own column of one ``[B, T + 1]`` block whose first column is the
        anchor; returns the drafts ``[B, T]``:

            logits_k <- U_k + B(x_{k-1}, .)
            x_k      <- argmax(logits_k)                      (``sampling`` None)
            x_k      <- drawn from the top of softmax(logits_k / T),
                        that q stored at q_store[base + k]     (lossless)

        Kimi-K3's loop: one for greedy and lossless, so the column convention
        and the noise counters cannot drift apart. V4, V4.1 and MiniMax-M3.1
        keep their own greedy loops, because their confidence head reads each
        position's ``W1[x_{k-1}]``, which the sampled op does not produce.
        Lossless, each row writes its q
        straight into the row block its request owns (``sampling.q_row_base``),
        which the accept kernel reads a step later. The branch on
        ``sampling`` is python, so a recording holds one side; an armed
        drafter records both (`DSparkProposer.block_greedy`).
        """
        bs, num_draft, vocab = base_logits.shape
        if sampling is not None:
            # Imported here for the reason `sample_next` gives.
            from atom.model_ops.spec_sampling.dspark_markov_sample import (
                dspark_markov_sample,
            )

            # The noise counter stride is the buffer's max batch, not this
            # step's, so the uniform a (position, row, id) draws does not move
            # with the batch, and no two positions of one block share one.
            position_stride = sampling.q_row_base.shape[0] * vocab
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        for k in range(num_draft):
            if sampling is None:
                # Greedy: temperature/sampling is applied by the target's
                # verify. The column is the op's destination, so no id is
                # written twice.
                self.sample_next(out_ids[:, k], base_logits[:, k], out_ids[:, k + 1])
                continue
            dspark_markov_sample(
                base_logits[:, k],
                out_ids[:, k],
                self.markov_w1.weight,
                self.markov_w2.weight,
                out_ids[:, k + 1],
                *sampling.q_store,
                sampling.q_row_base[:bs],
                k,
                sampling.temperatures[:bs],
                sampling.top_k[:bs],
                sampling.top_p[:bs],
                sampling.seed,
                k * position_stride,
                sampling.scratch,
            )
        return out_ids[:, 1:]


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


class KVCacheBinding(nn.Module):
    """What a draft KV pool binds one layer's cache onto, and nothing else.

    ``DraftKvBuilder`` recognizes a non-MLA attention module by these
    attributes and sets ``k_cache`` / ``v_cache`` (in the layout that draft's
    pool binds), the scales and ``max_model_len`` on it; the layer then reads
    and writes the cache itself.

    ``check_cache``, if given, is called with the views before they are bound,
    and raises for a pool its reader was not built for.
    ``page_tokens`` declares the page the reader views; None uses the pool's
    own block, while a reader that re-pages it must name that page explicitly.
    """

    base_attention = None
    use_mla = False

    def __init__(
        self, layer_num: int, check_cache=None, *, page_tokens: int | None = None
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.check_cache = check_cache
        self.page_tokens = page_tokens
        self.max_model_len = 0
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None

    @property
    def bound(self) -> bool:
        """Whether a pool is bound. Unbound is a real state, not an error:
        `warmup_model()` runs before `allocate_kv_cache()`, so the binding still
        holds its empty init tensor."""
        return self.k_cache.numel() > 0


class AllocationCap:
    """Each request's allocation, in tokens, for a paged block draft.

    The block writes up to T positions past the anchor, which can run past the
    pages the scheduler allocated; a table column that was never allocated
    reads as block 0, another request's entry. ``apply`` drops those positions'
    writes (slot -1) and returns the allocation so the caller can cap what the
    block reads. Buffers are sized at ``max_bs`` and only sliced, so a captured
    block replays against each step's numbers.
    """

    def __init__(self, max_bs: int, device) -> None:
        self._alloc = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self._alloc_host = torch.zeros(
            max_bs, dtype=torch.int32, device="cpu", pin_memory=True
        )
        # Recorded after each copy out of `_alloc_host`; a fresh event has
        # nothing to wait for.
        self._alloc_copied = torch.cuda.Event()

    def apply(
        self,
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
        block_positions: torch.Tensor,  # [running_bs, T]
        slots: torch.Tensor,  # [running_bs, T] the block's KV write slots
        unit: int,  # tokens one allocated page id spans
    ) -> torch.Tensor:
        """Mask ``slots`` past each row's allocation; return it in tokens."""
        running_bs = block_positions.shape[0]
        # The host can run a step ahead of the device: the previous copy must
        # have read the pinned buffer before it is rewritten.
        self._alloc_copied.synchronize()
        pages = self._alloc_host.numpy()[:running_bs]
        n = min(len(allocated_pages), scheduled_bs)
        pages[:n] = allocated_pages[:n]
        # Pad rows and the warmup batch address page 0 by convention and read
        # one page.
        pages[n:] = 1
        pages.clip(min=1, out=pages)
        alloc = self._alloc[:running_bs]
        alloc.copy_(self._alloc_host[:running_bs], non_blocking=True)
        self._alloc_copied.record()
        alloc.mul_(unit)
        slots.masked_fill_(block_positions >= alloc.view(running_bs, 1), -1)
        return alloc


class DSparkGQADecoderLayer(nn.Module):
    """The forward of a standalone GQA draft layer: pre-norm attention, then a
    pre-norm MLP. A subclass builds ``self_attn``, ``mlp``, ``input_layernorm``
    and ``post_attention_layernorm``."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Residual add fused into the norm, as in Qwen3DecoderLayer.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected + normed target hidden
        positions: torch.Tensor,  # [N] absolute positions
        slot_mapping: torch.Tensor,  # [N] destination slots
    ) -> None:
        """The context rows bypass ``input_layernorm``: the reference feeds
        ``hidden_norm(fc(aux))`` straight into the K/V projection; only the
        block stream goes through the layer's input norm."""
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)


class DSparkDraftModel(nn.Module):
    """What a DSpark draft model must provide to ``DSparkProposer``."""

    supports_block_graph = True
    # Whether the block attends through MLA decode, whose persistent metadata
    # the proposer plans. A draft that attends its block another way (MiniMax-
    # M3.1's per-row MHA decode) builds what it reads in `prepare_paged_block`.
    block_attention_is_mla = True
    # A pool of the draft's own, outside the target's DCP group. None is the
    # generic pool, inside the group.
    kv_pool_spec: ClassVar[DraftKvPoolSpec | None] = None

    #: Does this flavor's block sampler hand ``draft_sampling`` to
    #: ``DSparkMarkovHead.sample_block``, so it draws from, and publishes, a
    #: real q? A class flag, not a probe: every flavor's head has the method.
    #: ``DSparkProposer`` binds sampling buffers only where it is set.
    supports_lossless_draft = False

    #: The buffers the block sampler draws into, set once lossless speculative
    #: sampling arms (`DSparkLossless.bind_target`), before any warmup or
    #: capture; None keeps the greedy block sampler, as does
    #: ``head_and_sample(..., greedy=True)`` whatever is set. The branch is
    #: python, so a captured graph holds whichever side it was recorded with.
    draft_sampling: "DraftSampling | None" = None

    def init_block_fields(self, config) -> None:
        """The fields every standalone draft reads off its config alike: the
        block's mask id, the target layers it taps, and the Markov head.

        Call it where the draft registers its Markov head: registration order
        is ``named_parameters`` order, which dummy weight init walks.
        """
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.markov_head = DSparkMarkovHead(
            int(config.vocab_size), int(config.dspark_markov_rank)
        )

    def prepare_block(self, metadata_builder, num_draft, scheduled_bs, running_bs):
        """Mask padded writes for models whose draft pass mutates private KV."""
        return

    @property
    def first_attention_layer_num(self) -> int:
        """``kv_cache_data`` key of this draft's first attention layer.

        A paged draft binds that layer on ``layers[0].self_attn.kv`` (a
        ``KVCacheBinding``); a draft whose attention module lives somewhere
        else (Kimi-K3's MLA draft) overrides this. Inline drafts that keep a
        private KV window never reach the call site.
        """
        return self.layers[0].self_attn.kv.layer_num

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] page ids
        block_positions: torch.Tensor,  # [running_bs, T]
        slots: torch.Tensor,  # [running_bs, T] the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
        entry_tokens: int,  # tokens one block table id spans, the slots' unit
    ) -> None:
        """The per-step metadata only a non-MLA block attention reads, built
        outside the captured block from the proposer's paged block metadata."""
        return

    def write_context_kv(
        self,
        aux_concat: torch.Tensor,  # [N, target_hidden * num_target_layers]
        positions: torch.Tensor,  # [N] absolute positions
        slot_mapping: torch.Tensor | None = None,  # [N] paged-pool slots
    ) -> None:
        """Absorb one target forward's hidden states into this draft's KV.

        Where those rows land -- a private rolling window, a paged pool -- is
        the layer's business, not this method's. A paged layer writes at
        ``slot_mapping`` (see :meth:`DSparkProposer.compute_draft_kv` for which
        slots); a window layer ignores it and reads ``cu_seqlens_q`` off the
        live forward context.

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
            layer.write_context_kv(ctx_hidden, positions, slot_mapping)

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """Fuse the concatenated target aux hidden states into one [N, hidden]
        context tensor, shared by every layer: ``hidden_norm(fc(aux))``.

        A target layer id is that layer's output, the residual stream before
        any final norm, which is what the standalone drafts are trained on.
        """
        return self.hidden_norm(linear_out(self.fc(aux_concat)))

    @property
    def context_layers(self):
        """The layers that hold context KV, in order."""
        return self.layers

    def forward_spec(
        self,
        input_ids: torch.Tensor,  # [B] verified anchor token per request
        positions: torch.Tensor,  # flavor-defined, see `block_backbone`
        num_draft: "int | None" = None,
    ):
        """One DSpark block: parallel backbone pass + sequential Markov
        sampling. Returns ``(draft_token_ids [B, T], confidence | None)``.

        The composition of the two halves below, so the whole block and the
        captured block cannot drift apart: the proposer captures the backbone
        and runs the head as its epilogue."""
        return self.head_and_sample(
            self.block_backbone(input_ids, positions, num_draft),
            input_ids,
            num_draft,
        )

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

        By default, the standalone drafts' block: ``[anchor, MASK, ...]`` of
        width T embedded, every layer over ``rope_positions(positions)`` (the
        ``[B*T]`` form), then ``final_norm``. Returns ``[B*T, hidden]``, flat,
        which is what the LM head takes. A flavor whose block differs (V4,
        V4.1) overrides this.

        T positions in, T draft tokens out, the convention vLLM and SGLang use.
        (dflash.py's standalone ``spec_generate`` demo slices
        ``[:, -block_size+1:]`` instead; no production implementation does.)
        """
        bs = input_ids.shape[0]
        draft_ids = input_ids.new_full((bs, num_draft), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed_tokens(draft_ids.view(-1))
        positions = self.rope_positions(positions)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.final_norm(hidden, residual)
        return hidden

    def rope_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """The positions the default ``block_backbone``'s layers rope at: the
        block's own, unless a draft's rope cache cannot reach them."""
        return positions

    def head_and_sample(
        self, out, anchor_ids: torch.Tensor, num_draft: int, *, greedy: bool = False
    ):
        """The sequential half: LM head, then Markov sampling over the block.

        ``out`` is what ``block_backbone`` returned; ``anchor_ids`` is the one
        argument whose shape says how many requests there are. ``greedy``
        takes the greedy sampler whatever ``draft_sampling`` holds; only a
        flavor with ``supports_lossless_draft`` is asked for it. By default a
        draft with no confidence head: ``(draft_token_ids [B, T], None)``.
        """
        base_logits = self.lm_head(out).view(anchor_ids.shape[0], num_draft, -1)
        return self._sample_block(base_logits, anchor_ids, greedy), None

    def _sample_block(
        self,
        base_logits: torch.Tensor,  # [B, T, V]
        anchor_ids: torch.Tensor,  # [B]
        greedy: bool = False,
    ) -> torch.Tensor:
        """Markov-biased left-to-right sampling over the already-computed block:
        the head's ``sample_block`` with this model's ``draft_sampling``, or
        with none when ``greedy``.

            logits_k <- U_k + B(x_{k-1}, .)
            x_k      <- argmax(logits_k)       (``greedy``, or ``draft_sampling`` None)
            x_k      <- drawn from the top of softmax(logits_k / T), its q
                        published for the accept rule   (lossless)

        Only this loop is sequential, and it is T tiny rank-256 matmuls rather
        than T backbone passes -- which is the entire point of DSpark.
        """
        return self.markov_head.sample_block(
            base_logits, anchor_ids, None if greedy else self.draft_sampling
        )

    def share_with_target(self, target_base: nn.Module, loaded: set) -> None:
        """Bind the embedding and LM head to the target's loaded instances.

        ``target_base`` is the target ``*ForCausalLM`` (ModelRunner unwraps a
        multimodal ``language_model`` wrapper first), whose embedding lives on
        ``.model``. The vocabularies must agree or the shared LM head would
        silently score the wrong rows.
        """
        # The head the draft scores with; embed_tokens may be a PPMissingLayer
        # under PP.
        target_vocab = int(target_base.lm_head.num_embeddings)
        if target_vocab != self.vocab_size:
            raise ValueError(
                f"{type(self).__name__} draft vocab {self.vocab_size} != target "
                f"vocab {target_vocab}. The draft shares the target's embedding "
                "and LM head, so the two must agree."
            )
        # A stage without the target's embedding binds its PPMissingLayer,
        # which the drafter's check_shared_embedding refuses.
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head
