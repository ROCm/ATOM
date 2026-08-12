"""ATOM-backed Kimi-K3 DSpark draft wrapper for sglang-main's DSpark worker.

Kimi-K3 DSpark is a *block* drafter: sglang-main's ``DSparkWorkerV2`` embeds the
draft block (anchor + ``mask_token_id`` fillers) with the TARGET embedding and
runs this draft model once over the whole block, then reads back the raw hidden
states and drives ``compute_base_logits`` + ``markov_head.sample_block`` itself.

This wrapper keeps ATOM's ``KimiK3DSpark`` backbone (so the draft MLA attention
runs through ATOM's native kernels, matching the target's parity requirement) and
adapts it to the sglang DSpark draft contract:

    forward(input_ids, positions, forward_batch, input_embeds) -> hidden
    compute_base_logits(hidden) -> (base_logits, confidence_tap)
    markov_head.sample_block(base_logits, first_prev_tokens=, hidden_states=, sampler=)
    attach_shared_modules(embed_tokens=, lm_head=)
    write_target_hidden_kv(...)  -> project target aux hidden into the draft cache

The block attends NON-causally (every block position sees the whole block; order
is carried by RoPE). The KV/metadata translation and the sibling bf16 MLA pool
binding live in ``kimi_k3_dspark_bridge``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from atom.plugin.sglang.models.base_model_wrapper import _AtomCausalLMBaseForSglang
from atom.plugin.sglang.runtime import SGLangPluginRuntime, plugin_runtime_scope

logger = logging.getLogger("atom.plugin.sglang.models")


def _linear_out(output: Any) -> torch.Tensor:
    """ATOM quantized linears may return ``(tensor, scale)``; take the tensor."""
    return output[0] if isinstance(output, tuple) else output


class _K3DSparkMarkovAdapter(nn.Module):
    """Expose ATOM ``DSparkMarkovHead`` through sglang's ``sample_block`` API.

    sglang's ``VanillaMarkov.sample_block`` samples the block left-to-right,
    adding the first-order Markov bias ``B(x_{k-1})`` to each position's base
    logits before applying the caller's per-step ``sampler``. ATOM's
    ``DSparkMarkovHead(token_ids) -> (bias, embed)`` computes the same bias, so
    this adapter reuses ATOM's weights while honoring the sglang sampler hook
    (which carries greedy/temperature/top-k handling).
    """

    def __init__(self, atom_markov_head: nn.Module) -> None:
        super().__init__()
        self.atom_markov_head = atom_markov_head

    @property
    def markov_w1(self):  # exposed so weight loading finds the ATOM tensors
        return self.atom_markov_head.markov_w1

    @property
    def markov_w2(self):
        return self.atom_markov_head.markov_w2

    def sample_block(
        self,
        base_logits: torch.Tensor,  # [bs, gamma, V]
        *,
        first_prev_tokens: torch.Tensor,  # [bs]
        hidden_states: torch.Tensor | None,
        sampler,
    ):
        del hidden_states  # vanilla markov bias does not use draft hidden
        bs, gamma, _ = base_logits.shape
        out = base_logits.new_empty((bs, gamma), dtype=torch.long)
        prev = first_prev_tokens
        for k in range(gamma):
            bias, _ = self.atom_markov_head(prev)
            step_logits = base_logits[:, k].float() + bias
            token = sampler(step_logits, k)
            out[:, k] = token
            prev = token
        return out, None


class K3DSparkModel(_AtomCausalLMBaseForSglang):
    """sglang EntryClass for the Kimi-K3 DSpark standalone block drafter.

    Reuses the ATOM base wrapper for construction (``prepare_model``) and weight
    loading, but replaces the causal-LM forward with the DSpark block contract.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # sglang's DSpark worker drives sampling itself; expose the markov head
        # under its expected ``sample_block`` API backed by ATOM weights.
        self.markov_head = _K3DSparkMarkovAdapter(self.model.markov_head)
        # confidence_head is training-only for this checkpoint and deliberately
        # unloaded; run a fixed verify length (no confidence schedule).
        self.confidence_head = None

    # ---- sglang DSpark draft contract -------------------------------------

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        """Bind the target's embedding + lm_head onto the ATOM draft model.

        The proposer embeds the block with the target embedding externally and
        passes ``input_embeds``; the draft only needs the shared lm_head for
        ``compute_base_logits``. Keep the embedding too for parity with ATOM's
        ``share_with_target`` (harmless, used only if we ever self-embed).
        """
        self.model.embed_tokens = embed_tokens
        self.model.lm_head = lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def compute_base_logits(self, hidden: torch.Tensor):
        """Project post-final-norm hidden through the shared lm_head.

        The DSpark worker calls this OUTSIDE the draft forward (no ATOM forward
        context), but ATOM's ``ParallelLMHead.forward`` reads
        ``get_forward_context().context.is_prefill``. Establish a minimal decode
        context (``is_draft=True`` so the head skips its prefill-only path).
        """
        lm_head = self.model.lm_head
        if lm_head is None:
            raise RuntimeError(
                "K3DSparkModel.compute_base_logits called before "
                "attach_shared_modules bound the target lm_head."
            )
        from atom.utils.forward_context import Context, set_forward_context

        positions = torch.zeros(
            int(hidden.shape[0]), dtype=torch.int64, device=hidden.device
        )
        set_forward_context(
            attn_metadata=None,
            atom_config=self.atom_config,
            context=Context(positions=positions, is_prefill=False, is_draft=True),
        )
        base_logits = _linear_out(lm_head(hidden))
        return base_logits, None

    def write_target_hidden_kv(self, **kwargs: Any) -> None:
        """Project verified target hidden states into the draft's KV cache.

        Delegates to the bridge, which adapts sglang's injector call shape to
        ATOM's ``project_context`` + per-layer ``_pcp_write_full_kv`` on the
        sibling bf16 MLA pool.
        """
        from atom.plugin.sglang.kimi_k3_dspark_bridge import (
            write_kimi_k3_dspark_target_hidden_kv,
        )

        write_kimi_k3_dspark_target_hidden_kv(self.model, **kwargs)

    # ---- block forward -----------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Any = None,
        **model_kwargs: Any,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors, model_kwargs
        if input_embeds is None:
            raise ValueError(
                "K3DSparkModel.forward requires input_embeds (the target "
                "embedding of the draft block)."
            )
        with (
            plugin_runtime_scope(framework="sglang", atom_config=self.atom_config),
            SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
                set_forward_context=True,
            ) as runtime,
        ):
            # Bind the sibling bf16 MLA pool views and build the non-causal
            # block attention metadata into the ATOM forward context.
            self.model_arch_spec.bind_cache_views(self.model, runtime)

            hidden = runtime.input_embeds
            for layer in self.model.layers:
                hidden = layer(runtime.positions, hidden)
            hidden = self.model.final_norm(hidden)
            hidden = runtime.trim_output(hidden)

        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def load_weights(self, weights: Any = None):
        """Load the DRAFT checkpoint, not the target.

        The generic base loader resolves the checkpoint from
        ``plugin_config.model_config.model_path`` -- which SGLang fills from the
        global (target) ServerArgs. Force the draft path (set on
        ``atom_config.model`` by ``_prepare_k3_dspark_draft_model_config``) so
        the ATOM loader reads the standalone DSpark checkpoint and its bf16
        weight names, exactly as ATOM core / ATOM-vLLM do.
        """
        del weights
        from atom.model_loader.loader import load_model_in_plugin_mode

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model_in_plugin_mode(
                model=self.model,
                config=self.atom_config,
                prefix=self.model_arch_spec.load_weights_prefix,
                model_name_or_path_override=self.atom_config.model,
            )


EntryClass = [K3DSparkModel]
