"""vLLM-specific Kimi-K3 DSpark draft.

Two layers of adaptation, kept out of the native model:

* :class:`KimiK3DSparkDraft` adds the plain backbone pass vLLM's speculator
  calls. The native module only exposes ``forward_spec``, which also builds the
  block and samples it -- work vLLM does itself, in its own buffers.
* :class:`KimiK3DSparkVllm` is what vLLM instantiates, and translates the
  speculator's drafting contract onto that module.
"""

import torch

from atom.models.kimi_k3_dspark import KimiK3DSpark as KimiK3DSparkBase
from atom.plugin.vllm.model_wrapper import ATOMForCausalLM


class KimiK3DSparkDraft(KimiK3DSparkBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One parallel pass over the draft block.

        vLLM has already laid the block out as ``[anchor, MASK, ...]`` in its
        own input buffer, so this neither seeds the block nor samples it -- the
        speculator's Markov loop runs over the hidden states returned here.
        """
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.final_norm(hidden_states)

    def write_combined_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor | list[torch.Tensor],
    ) -> None:
        """Scatter the target-derived context rows into every draft layer.

        Unlike the native ``write_context_kv`` this does not project first: vLLM
        combines the aux hidden states in a separate step (so it can stage them
        in a persistent buffer) and hands the result straight here. Slots may
        differ per layer when the draft spans several KV cache groups.

        This enters at the attention module rather than at the decoder layer:
        the layer-level entry point reads its slots off ATOM's native forward
        context, which does not exist under vLLM. Skipping it changes nothing
        else -- the context rows bypass ``input_layernorm`` either way.
        """
        per_layer_slots = isinstance(slot_mappings, (list, tuple))
        for i, layer in enumerate(self.layers):
            layer.self_attn.write_context_kv(
                ctx_hidden,
                positions,
                slot_mappings[i] if per_layer_slots else slot_mappings,
            )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.mla_attn.layer_name for layer in self.layers]


class KimiK3DSparkVllm(ATOMForCausalLM):
    """The DSpark drafting contract, backed by the ATOM draft module."""

    # The draft ships neither an embedding table nor an LM head; vLLM's loader
    # consults these before binding the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Plain attributes, not properties: vLLM's DSpark loader rebinds
        # ``lm_head`` to the target's head after construction.
        self.lm_head = None
        # The draft scores the full target vocabulary, so its sampled ids are
        # already target ids and need no remap table.
        self.draft_id_to_target_id = None

    def combine_hidden_states(self, aux_concat: torch.Tensor) -> torch.Tensor:
        return self.model.project_context(aux_concat)

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> None:
        # Dummy and memory-profiling steps pass no slots; the block tables are
        # placeholders there and writing would clobber real cache entries.
        if slot_mappings is None:
            return
        self.model.write_combined_context_kv(hidden_states, positions, slot_mappings)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.markov_w1(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        # fp32: this bias lands inside the softmax that decides acceptance.
        weight = self.model.markov_head.markov_w2.weight
        return torch.matmul(markov_embed.float(), weight.float().t())

    def map_draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return draft_token_ids

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()
