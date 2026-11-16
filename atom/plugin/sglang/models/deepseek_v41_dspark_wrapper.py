# SPDX-License-Identifier: MIT
"""ATOM-backed ``DeepseekV4ForCausalLMDSpark`` for SGLang's DSpark worker.

SGLang renames a DSpark-bundling V4.1 draft runner to
``DeepseekV4ForCausalLMDSpark``; this wrapper keeps that public name and
delegates the draft to ATOM's native ``DeepseekV41DSpark`` (three MoE draft
stages, ``DraftAttention`` over the target's window rings 40-42 plus the
block). SGLang keeps the worker lifecycle, the Markov sampling, the rejection
and the TP token sync; ATOM owns the weights, the block math and the KV.

The draft is a separate enablement from the target: its own config (ATOM's
draft parse, ``model_type=deepseek_v41_dspark``), its own loader pass
(``spec_decode=True``: ``mtp.*`` only), and its own forward context built from
the draft runner's current ``ForwardBatch`` and pool.

V4 (not V4.1) DSpark is not a plugin path: this wrapper refuses it.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dspark import BaseDSparkMarkovHead
from torch import nn

from atom.plugin.sglang.deepseek_v41_bridge import (
    V41_DSPARK_ALGORITHM,
    V41_DSPARK_DRAFT_ARCH,
    V41_DSPARK_DRAFT_MODEL_TYPE,
    V41_DSPARK_DRAFT_TOKENS,
    V41_DSPARK_STAGES,
    _sglang_bag,
    is_deepseek_v41_config,
    is_v41_dspark_atom_config,
    speculative_algorithm_name,
    v41_dspark_capability_error,
)
from atom.plugin.sglang.runtime import plugin_runtime_scope

logger = logging.getLogger("atom.plugin.sglang.models")

# Kill switch: "0" keeps SGLang's Markov chain for a greedy draft
# (SGLANG_DSPARK_USE_GREEDY_DRAFT=1) instead of ATOM's sharded fused argmax.
V41_DSPARK_FUSED_GREEDY_ENV = "ATOM_SGLANG_V41_DSPARK_FUSED_GREEDY_DRAFT"


class ATOMV41MarkovHeadAdapter(BaseDSparkMarkovHead):
    """SGLang's Markov-head interface over ATOM's ``DSparkMarkovHead``.

    The bias is ATOM's own forward: ``W1[x_{k-1}] @ W2^T`` in fp32
    (``markov_w{1,2}`` are ``[V, r]``), with W2's fp32 cast held once
    (`refresh_w2_cache`) instead of redone per block position. The vocab is whole on every rank (ATOM
    keeps both tables replicated), so ``vocab_shard`` stays None and SGLang
    samples from the gathered base logits. No fused greedy weight
    (SGLang's one-row kernel): a greedy draft block runs ATOM's sharded fused
    argmax instead (`DeepseekV4ForCausalLMDSpark.atom_v41_greedy_draft_block`).

    ATOM's head is held outside the module tree so its weights are not
    registered twice.
    """

    def __init__(self, atom_markov_head: nn.Module) -> None:
        super().__init__()
        self.__dict__["_atom"] = atom_markov_head
        self.vocab_size = int(atom_markov_head.vocab_size)
        self.markov_rank = int(atom_markov_head.rank)

    @property
    def atom_markov_head(self) -> nn.Module:
        return self.__dict__["_atom"]

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.atom_markov_head.markov_w1(token_ids.long())

    def refresh_w2_cache(self) -> torch.Tensor:
        """``markov_w2`` as fp32, cast once instead of on every block position.

        ``DSparkMarkovHead.forward`` casts the whole ``[V, r]`` table per
        position (5 x 66 MB read + 132 MB write a round, ~0.35 ms on MI308X).
        The copy is the same fp32 values in the same layout, so
        ``emb.float() @ w2.t()`` is the identical GEMM; +V*r*4 B (132 MB) a
        rank. Built at load (`load_weights`), inside the memory the KV profile
        sees. A reload refreshes it in place: a captured draft graph reads its
        address."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark Markov W2 cache built under a graph capture")
        weight = self.atom_markov_head.markov_w2.weight
        cache = self.__dict__.get("_w2_f32")
        if (
            cache is not None
            and cache.shape == weight.shape
            and cache.device == weight.device
        ):
            cache.copy_(weight.detach())
        else:
            cache = weight.detach().float().contiguous()
            self.__dict__["_w2_f32"] = cache
        self.__dict__["_w2_src"] = weight.data_ptr()
        return cache

    def _w2_fp32(self) -> torch.Tensor | None:
        """The fp32 cache of the live ``markov_w2``, or None to cast per step
        (no cache yet, or the weight was replaced under a capture)."""
        cache = self.__dict__.get("_w2_f32")
        weight = self.atom_markov_head.markov_w2.weight
        if cache is not None and self.__dict__.get("_w2_src") == weight.data_ptr():
            return cache
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return None
        return self.refresh_w2_cache()

    def step_bias(self, prev_tokens, *, hidden_states, state):
        del hidden_states
        w2 = self._w2_fp32()
        if w2 is None:
            bias, _ = self.atom_markov_head(prev_tokens.long())
            return bias, state
        embed = self.atom_markov_head.markov_w1(prev_tokens.long())
        # fp32 matmul, as `DSparkMarkovHead.forward`: the bias enters the
        # softmax that gates acceptance.
        return torch.matmul(embed.float(), w2.t()), state

    def step_logits_dtype(self, base_dtype: torch.dtype) -> torch.dtype:
        del base_dtype
        return torch.float32

    def greedy_w2_weight(self):
        return None


def _speculative_algorithm():
    spec = _sglang_bag("spec")
    if spec is not None and hasattr(spec, "speculative_algorithm"):
        return spec.speculative_algorithm
    from atom.plugin.config import get_sglang_server_args

    return getattr(get_sglang_server_args(), "speculative_algorithm", None)


class DeepseekV4ForCausalLMDSpark(nn.Module):
    """SGLang's DSpark draft contract, backed by ATOM's ``DeepseekV41DSpark``."""

    # ATOM owns the draft's quantization (block-32 FP8 dense, FP4 experts).
    sglang_skip_quant_config = True
    # The block's row 0 is the bonus token: SGLang samples five rows from it.
    sample_from_anchor = True
    uses_own_vocab_modules = False
    supports_pre_gather_target_hidden_projection = False
    # Static verify only: the confidence head's weights load (coverage) but
    # no schedule reads it.
    confidence_head = None

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del quant_config, prefix
        super().__init__()
        if not is_deepseek_v41_config(config):
            raise NotImplementedError(
                "ATOM's SGLang DSpark draft serves DeepSeek-V4.1 only; "
                "DeepSeek-V4 DSpark is not a plugin path"
            )
        algorithm = speculative_algorithm_name(_speculative_algorithm())
        reason = v41_dspark_capability_error(config)
        if algorithm != V41_DSPARK_ALGORITHM or reason is not None:
            raise NotImplementedError(
                f"DeepSeek-V4.1 DSpark draft refused: algorithm={algorithm!r}, "
                f"checkpoint: {reason or 'ok'}"
            )
        from atom.config import get_current_atom_config

        target_atom_config = get_current_atom_config()
        if not is_v41_dspark_atom_config(target_atom_config):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft needs the V4.1 target's ATOM config "
                "with method=dspark as the current config"
            )
        speculative = target_atom_config.speculative_config
        if speculative.num_speculative_tokens != V41_DSPARK_DRAFT_TOKENS:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark proposes {V41_DSPARK_DRAFT_TOKENS} tokens; "
                f"the ATOM config says {speculative.num_speculative_tokens}"
            )
        draft_hf = speculative.draft_model_hf_config
        arches = tuple(getattr(draft_hf, "architectures", None) or ())
        if getattr(
            draft_hf, "model_type", None
        ) != V41_DSPARK_DRAFT_MODEL_TYPE or arches[:1] != (V41_DSPARK_DRAFT_ARCH,):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft config was not normalized by ATOM "
                f"(model_type={getattr(draft_hf, 'model_type', None)}, "
                f"architectures={arches})"
            )
        from atom.plugin.sglang.runtime.model_arch import resolve_model_arch_spec

        resolved, _ = resolve_model_arch_spec(draft_hf)
        if resolved != V41_DSPARK_DRAFT_ARCH:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark draft config resolves to {resolved}"
            )

        self.config = config
        self.quant_config = None
        # Plain attributes: `Config` objects are not modules.
        self.target_atom_config = target_atom_config
        draft_atom_config = copy.copy(target_atom_config)
        draft_atom_config.hf_config = draft_hf
        self.atom_config = draft_atom_config

        with plugin_runtime_scope(framework="sglang", atom_config=target_atom_config):
            from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

            # As native's DSparkProposer builds it (`_build_draft_model`);
            # the backbone's side stream is not reachable here, so the draft
            # MoE runs without one.
            self.model = DeepseekV41DSpark(draft_atom_config)

        if int(self.model.block_size) != V41_DSPARK_DRAFT_TOKENS:
            raise RuntimeError(
                f"ATOM DSpark block {self.model.block_size} != "
                f"{V41_DSPARK_DRAFT_TOKENS}"
            )
        stages = len(self.model.mtp)
        if stages != V41_DSPARK_STAGES:
            raise RuntimeError(f"ATOM DSpark has {stages} stages")
        self.gamma = self.block_size = int(self.model.block_size)
        self.num_stages = stages
        self.start_layer, self.end_layer = 0, stages
        self.num_context_features = len(draft_hf.dspark_target_layer_ids)
        self.noise_token_id = int(draft_hf.dspark_noise_token_id)
        self.hidden_size = int(draft_hf.hidden_size)
        self.vocab_size = int(draft_hf.vocab_size)
        self.markov_head = ATOMV41MarkovHeadAdapter(self.model.mtp[-1].markov_head)
        self.logits_mup_width_multiplier = None
        # Why the fused sharded greedy block (`atom_v41_greedy_draft_block`)
        # cannot run, or None once it can; resolved at its first call (the
        # graph runner's eager warmup, before any capture). Host checks only.
        self._fused_greedy_reason = "not resolved"
        self._check_mask_token(config)
        self._log_route()

    def _check_mask_token(self, config) -> None:
        from sglang.srt.speculative.dspark_components.dspark_config import (
            parse_dspark_draft_config,
        )

        mask = parse_dspark_draft_config(draft_hf_config=config).mask_token_id
        if mask is not None and int(mask) != self.noise_token_id:
            raise RuntimeError(
                f"SGLang's DSpark mask token {mask} != the checkpoint's "
                f"dspark_noise_token_id {self.noise_token_id}"
            )

    def _log_route(self) -> None:
        stage = self.model.mtp[0]
        moe = getattr(stage, "ffn", None)
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark draft: role=draft "
            "wrapper=DeepseekV4ForCausalLMDSpark atom=%s stages=%d layer_ids=%s "
            "moe=%s/top%s attn=%s block=%d noise=%d markov_rank=%d",
            type(self.model).__name__,
            self.num_stages,
            tuple(layer.attn.spec.layer_id for layer in self.model.mtp),
            getattr(self.atom_config.hf_config, "dspark_n_routed_experts", "?"),
            getattr(self.atom_config.hf_config, "dspark_num_experts_per_tok", "?"),
            type(stage.attn).__name__,
            self.block_size,
            self.noise_token_id,
            self.markov_head.markov_rank,
        )
        del moe

    # -- weights ---------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """ATOM's native drafter load: the target checkpoint, ``mtp.*`` only
        (``remap_mtp_weight_name``), the draft's own mapper and quant rules.
        SGLang's iterator is not read."""
        del weights
        from atom.model_loader.loader import load_model

        target = self.target_atom_config
        speculative = target.speculative_config
        with plugin_runtime_scope(framework="sglang", atom_config=target):
            loaded = load_model(
                self.model,
                target.model,
                speculative.draft_model_hf_config,
                target.load_dummy,
                True,
            )
        self._report_coverage(loaded)
        # The Markov tables are plain embeddings (no quant post-processing),
        # final here; a reload refreshes the fp32 W2 in place.
        self.markov_head.refresh_w2_cache()
        return loaded

    def _report_coverage(self, loaded) -> None:
        """The draft's weight coverage as one line a rank, the gate G1 reads.

        Every real parameter must be loaded; the loader's own warnings name
        the first misses, this line counts them and checks the routed experts
        of every stage materialized."""
        names = {name for name, _ in self.model.named_parameters()}
        loaded = set(loaded or ())
        derived = ("_shuffled", "_packed", "_meta_for_quant")
        unloaded = sorted(
            name
            for name in names - loaded
            if not any(name.endswith(suffix) for suffix in derived)
        )
        experts = {
            f"mtp.{i}.ffn.experts.{w}": any(
                n.startswith(f"mtp.{i}.ffn.experts.{w}") for n in loaded
            )
            for i in range(self.num_stages)
            for w in ("w13_weight", "w2_weight")
        }
        complete = not unloaded and all(experts.values())
        summary = (
            f"ATOM DeepSeek-V4.1 DSpark draft weight coverage: "
            f"{len(names & loaded)}/{len(names)} parameters loaded, "
            f"{len(unloaded)} unloaded {unloaded[:32]}, routed experts "
            f"{'ok' if all(experts.values()) else experts}"
        )
        if complete:
            logger.info(summary)
            return
        # A partially loaded draft still serves correct text (the target
        # verifies every token) and only shows up as collapsed acceptance.
        if os.environ.get("ATOM_SGLANG_V41_DSPARK_ALLOW_PARTIAL_DRAFT") == "1":
            logger.warning(
                "%s (allowed by ATOM_SGLANG_V41_DSPARK_ALLOW_PARTIAL_DRAFT)", summary
            )
            return
        raise RuntimeError(summary)

    # -- shared modules --------------------------------------------------

    def attach_shared_modules(self, *, embed_tokens, lm_head) -> None:
        """SGLang's handshake: the target's embedding and head. Both are
        ATOM's (the target wrapper's ``lm_head`` is a view of ATOM's head),
        bound to the draft the way native ``share_with_target`` binds them;
        the target is also told which draft's window it writes."""
        backbone = getattr(lm_head, "atom_backbone", None)
        if backbone is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft needs the ATOM V4.1 target's lm_head "
                f"view, got {type(lm_head).__name__}"
            )
        if embed_tokens is not backbone.embed:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft was handed an embedding other than "
                "the ATOM target's"
            )
        if int(backbone.head.num_embeddings) != self.vocab_size:
            raise RuntimeError("DeepSeek-V4.1 DSpark draft and target vocab differ")
        self.model.share_with_target(backbone)
        lm_head.atom_wrapper.attach_dspark_draft(self.model)
        from atom.plugin.sglang.patches.dspark_fused_greedy_draft_patch import (
            install_dspark_fused_greedy_draft_patch,
        )

        install_dspark_fused_greedy_draft_patch()
        # SGLang's folded draft sampler sizes its step logits off the draft's
        # ``lm_head.org_vocab_size``. The target's head view, not a module:
        # nothing registers twice.
        self.__dict__["lm_head"] = lm_head
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark draft shares the target's embed and head "
            "(tp=%d, vocab=%d)",
            int(backbone.head.tp_size),
            self.vocab_size,
        )

    # -- forward ---------------------------------------------------------

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Defined so SGLang leaves the block's embedding to the draft
        # (`SGLANG_DSPARK_EMBED_IN_GRAPH`): ATOM embeds the anchors and keeps
        # the noise row. The forward never takes precomputed embeddings.
        raise NotImplementedError(
            "DeepSeek-V4.1 DSpark embeds its block inside block_backbone"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        del kwargs
        if input_embeds is not None or get_embedding or pp_proxy_tensors is not None:
            raise NotImplementedError(
                "DeepSeek-V4.1 DSpark draft takes token ids only (no input "
                "embeddings, embedding mode or PP proxies)"
            )
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        from atom.plugin.sglang.deepseek_v41_bridge import count_v41_dspark
        from atom.plugin.sglang.deepseek_v41_dspark import (
            _in_graph_capture,
            draft_block_inputs,
            v41_dspark_draft_forward_context,
        )

        if not _in_graph_capture():
            # A replayed block runs no Python; this is an eager block.
            count_v41_dspark("draft_eager")
        pool = get_token_to_kv_pool()
        if not getattr(pool, "is_atom_v41_draft_proxy", False):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft forward resolved a pool other than "
                f"the draft role of the ATOM proxy: {type(pool).__name__}"
            )
        block = draft_block_inputs(
            forward_batch,
            input_ids,
            positions,
            noise_token_id=self.noise_token_id,
            width=self.block_size,
        )
        with (
            plugin_runtime_scope(
                framework="sglang", atom_config=self.target_atom_config
            ),
            v41_dspark_draft_forward_context(
                self.target_atom_config, cache=pool.atom_cache, block=block
            ),
        ):
            normed, _ = self.model.block_backbone(
                block.anchor_ids, block.anchor_positions, block.width
            )
        rows = block.bs * block.width
        if normed.dim() != 2 or int(normed.shape[0]) != rows:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark draft returned {tuple(normed.shape)} for "
                f"{rows} block rows"
            )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=normed)

    def prepare_v41_draft_graph(self) -> None:
        """Build the block's lazy caches -- the stages' layer ids and the
        noise row (a TP all-reduce of the shared embedding) -- before any
        draft graph capture, so neither is first built under a capture."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft graph state built under a capture"
            )
        embed = getattr(self.model, "embed", None)
        if embed is None or getattr(self.model, "head", None) is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft graphs before the target's embed and "
                "head were attached"
            )
        with plugin_runtime_scope(
            framework="sglang", atom_config=self.target_atom_config
        ):
            self.model._context_layer_ids(embed.weight.device)
            self.model._noise_embedding()

    # -- greedy draft: ATOM's sharded fused Markov argmax -------------------

    def _resolve_fused_greedy(self) -> str | None:
        """Whether native's sharded greedy block can run here (the gate of
        ``DeepseekV41DSpark._sharded_markov_sample``, plus a vocab the TP
        shards cover exactly), logged once a rank."""
        resolved = getattr(self, "_fused_greedy_reason", "not resolved")
        if resolved != "not resolved":
            return resolved
        from atom.model_ops.dspark_markov_argmax import markov_weights_unsupported

        head = getattr(self.model, "head", None)
        markov = self.model.mtp[-1].markov_head
        if os.environ.get(V41_DSPARK_FUSED_GREEDY_ENV, "1") == "0":
            reason = f"{V41_DSPARK_FUSED_GREEDY_ENV}=0"
        elif head is None:
            reason = "the target head is not attached"
        elif int(head.tp_size) <= 1:
            reason = "the head is not TP-sharded"
        elif getattr(head, "bias", None) is not None:
            reason = "the head has a bias"
        elif not getattr(markov, "fused_sample", False):
            reason = "ATOM_DSPARK_FUSED_MARKOV_SAMPLE=0"
        elif int(head.num_embeddings_per_partition) * int(head.tp_size) != int(
            markov.vocab_size
        ):
            reason = "the head's TP shards do not cover the Markov vocab exactly"
        else:
            reason = markov_weights_unsupported(
                markov.markov_w1.weight, markov.markov_w2.weight
            )
        self._fused_greedy_reason = reason
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark greedy draft block: %s",
            (
                "ATOM dspark_markov_argmax_tp on the vocab shard (no step logits, "
                "no [B*T, V] all-gather)"
                if reason is None
                else f"SGLang's Markov chain ({reason})"
            ),
        )
        return reason

    def atom_v41_greedy_draft_block(
        self, hidden: torch.Tensor, anchor_tokens: torch.Tensor, gamma: int
    ) -> torch.Tensor | None:
        """The greedy draft block ``[bs, gamma]`` from the draft's normed rows,
        as native ``_head_and_sample_sharded`` drafts it: this rank's LM-head
        shard ``[bs * gamma, V / tp]``, then per position ATOM's fused
        ``W1[x_{k-1}] @ W2^T`` + argmax over the shard, the ranks exchanging
        only each row's ``(max, id)``. The same argmax SGLang's greedy chain
        takes over the gathered logits plus the fp32 bias (fp32 accumulation,
        lowest id on ties), up to the GEMV's summation order; identical on
        every rank by construction. None when the gate is off
        (`_resolve_fused_greedy`): SGLang's chain runs instead."""
        if self._resolve_fused_greedy() is not None:
            return None
        from aiter.dist.parallel_state import get_tp_group
        from aiter.tuned_gemm import tgemm

        from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax_tp
        from atom.utils import envs

        rows = int(hidden.shape[0])
        if rows % gamma or int(anchor_tokens.numel()) * gamma != rows:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark greedy block of {rows} rows for "
                f"{int(anchor_tokens.numel())} anchors x {gamma}"
            )
        bs = rows // gamma
        head = self.model.head
        markov = self.model.mtp[-1].markov_head
        with plugin_runtime_scope(
            framework="sglang", atom_config=self.target_atom_config
        ):
            base_logits = tgemm.mm(hidden, head.weight).view(bs, gamma, -1)
            out = torch.empty((bs, gamma), dtype=torch.int64, device=hidden.device)
            tp_group = get_tp_group()
            use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
            for k in range(gamma):
                dspark_markov_argmax_tp(
                    base_logits[:, k],
                    anchor_tokens if k == 0 else out[:, k - 1],
                    markov.markov_w1.weight,
                    markov.markov_w2.weight,
                    out[:, k],
                    int(head.vocab_start_idx),
                    tp_group,
                    use_custom=use_custom,
                )
        return out

    def compute_base_logits(self, x: torch.Tensor):
        """The shared target head over the draft's normed rows, full vocab
        (ATOM's head gathers it across TP), as native ``head_and_sample``
        computes it. The second value is SGLang's confidence feature."""
        if self.logits_mup_width_multiplier is not None:
            raise RuntimeError("DeepSeek-V4.1 has no muP logits multiplier")
        with plugin_runtime_scope(
            framework="sglang", atom_config=self.target_atom_config
        ):
            logits = self.model.head.get_logits(x)
        if int(logits.shape[-1]) != self.vocab_size:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark base logits are {int(logits.shape[-1])} "
                f"wide, the vocab {self.vocab_size}"
            )
        return logits, x

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor | None = None,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        """SGLang's context-KV injection, after a target prefill or verify.

        The rows are already in the draft's window: the target forward that
        produced them wrote them (`V41DSparkTargetSurface.finish_forward`),
        under its own step, before the verdict -- native's order. Rejected
        verify rows sit above the next anchor, masked by the block's indices,
        in ring slots no committed row of the window shares. After a radix
        hit at ``P`` SGLang injects ``[P, L)`` only; the rows below ``P`` the
        window needs were written by the same forward's replay rows
        (``deepseek_v41_radix``). So this only checks the contract and writes
        nothing."""
        del cache_loc, cache_loc_2d, commit_lens
        if not getattr(pool, "is_atom_v41_draft_proxy", False):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark injection into a pool other than the "
                f"draft role of the ATOM proxy: {type(pool).__name__}"
            )
        if target_hidden_is_projected:
            raise RuntimeError("DeepSeek-V4.1 DSpark takes unprojected target hidden")
        width = self.num_context_features * self.hidden_size
        if target_hidden.dim() != 2 or int(target_hidden.shape[-1]) != width:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark target hidden {tuple(target_hidden.shape)} "
                f"is not [rows, {width}]"
            )
        if int(target_hidden.shape[0]) != int(positions.shape[0]):
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark target hidden has {int(target_hidden.shape[0])} "
                f"rows, positions {int(positions.shape[0])}"
            )

    # -- not on this path --------------------------------------------------

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSeek-V4.1 DSpark projects target hidden inside the target forward"
        )

    def set_dspark_layers_to_capture(self, layer_ids) -> None:
        raise NotImplementedError("The DSpark draft captures no aux hidden states")

    def prune_to_ctx_kv_injection(self) -> None:
        raise NotImplementedError(
            "DeepSeek-V4.1 DSpark has no PD-prefill draft on the plugin"
        )

    def compute_confidence(self, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V4.1 DSpark on the plugin runs static verify (no confidence)"
        )


EntryClass = [DeepseekV4ForCausalLMDSpark]
