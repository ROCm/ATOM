# SPDX-License-Identifier: MIT
"""V4.1 DSpark math and checkpoint layout; context storage is caller-owned."""

from copy import copy

import torch
import torch.nn.functional as F
from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.dspark import draft_attention, draft_step, rotate_rows
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.deepseek_v41.normalization import RMSNorm
from atom.model_ops.deepseek_v41.projections import grouped_output_projection
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from torch import nn

from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.dspark_draft import DSparkDraftModel

from .attention import Attention
from .config import build_attention_topology
from .layers import native_quant_config, reduce_output
from .model import Block, DeepseekV41ForCausalLM, LogitsHead


class DraftAttention(Attention):
    def project_context(self, hidden, positions, rope, *, packed=False):
        keys = rotate_rows(rope, self.kv_norm(self.wkv(hidden)), positions)
        # Unlike V4's mixed NoPE/RoPE layout, V4.1 QAT covers all head lanes.
        return quantize_fp8(keys, dequantize=not packed)

    def forward(self, hidden, context_kv, step, rope):
        qr = self.q_norm(self.wq_a(hidden))
        query = self.wq_b(qr).unflatten(-1, (self.heads, self.head_dim))
        query = rotate_rows(rope, query, step.positions)
        keys = self.project_context(hidden, step.positions, rope)
        output = draft_attention(
            query,
            context_kv[self.spec.layer_id],
            keys,
            self.attn_sink,
            step,
            self.head_dim**-0.5,
        )
        output = rotate_rows(rope, output, step.positions, inverse=True)
        output = output.unflatten(-2, (self.groups, -1)).flatten(-2)
        weight = self.wo_a.weight.view(self.groups, self.o_rank, -1)
        return reduce_output(
            self.wo_b(grouped_output_projection(output, weight).flatten(-2))
        )


class MarkovHead(nn.Module):
    def __init__(self, vocab_size, rank):
        super().__init__()
        self.embed = VocabParallelEmbedding(vocab_size, rank)
        self.head = LogitsHead(rank, vocab_size)

    def forward(self, token_ids):
        hidden = self.embed(token_ids.flatten()).view(*token_ids.shape, -1)
        return self.head(hidden), hidden


class ConfidenceHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, 1, bias=False, dtype=torch.bfloat16)
        self.register_buffer("fp32_weight", None, persistent=False)

    def process_weights_after_loading(self):
        self.fp32_weight = self.proj.weight.float()

    def forward(self, hidden, markov_embeddings):
        if self.fp32_weight is None:
            raise RuntimeError("Confidence weights must be processed after loading")
        values = torch.cat((hidden, markov_embeddings), dim=-1).float()
        return F.linear(values, self.fp32_weight).squeeze(-1)


class DraftBlock(Block):
    attention_cls = DraftAttention

    def __init__(self, config, spec, stage, prefix: str = ""):
        super().__init__(config, spec, prefix=prefix)
        if stage == 0:
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                quant_config=native_quant_config(),
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if stage == config.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = MarkovHead(config.vocab_size, config.dspark_markov_rank)
            self.confidence_head = ConfidenceHead(
                config.hidden_size + config.dspark_markov_rank
            )


class DeepseekV41DSpark(DSparkDraftModel):
    # Loads through the shared path like the backbone, off the same checkpoint
    # and with the same layer types, so the rename rules are the backbone's --
    # referenced, not restated, so the two cannot drift apart.
    weights_mapper = DeepseekV41ForCausalLM.weights_mapper
    weights_mapping = DeepseekV41ForCausalLM.weights_mapping
    packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping
    disable_fused_shared_loading = DeepseekV41ForCausalLM.disable_fused_shared_loading
    # The block gathers from request-owned windows through live Python
    # metadata, and eager expert dispatch synchronizes route counts. Only
    # target tensor stages are capturable until draft inputs are staged too.
    supports_block_graph = False

    def __init__(self, config, *, max_length=None):
        super().__init__()
        args = getattr(config, "hf_config", config)
        if args.num_nextn_predict_layers < 1 or args.dspark_block_size < 1:
            raise ValueError(
                "V4.1 DSpark needs draft stages and a positive block width"
            )
        self.config = args
        self.block_size = args.dspark_block_size
        self.window_size, self.vocab_size = args.sliding_window, args.vocab_size
        draft = copy(args)
        draft.n_routed_experts = args.dspark_n_routed_experts
        draft.num_experts_per_tok = args.dspark_num_experts_per_tok
        topology = build_attention_topology(args)[args.num_hidden_layers :]
        self.mtp = nn.ModuleList(
            # The stage index, not the topology's layer id: this prefix names
            # the module's own parameters, and `nn.ModuleList` numbers them
            # from zero -- which is also how the checkpoint numbers them.
            DraftBlock(draft, spec, i, prefix=f"mtp.{i}")
            for i, spec in enumerate(topology)
        )
        capacity = max_length or getattr(
            config, "max_model_len", args.max_position_embeddings
        )
        self.rope = RotaryEmbedding(
            args.qk_rope_head_dim, capacity + self.block_size, base=args.rope_theta
        )
        self.embed = self.head = None

    @property
    def model(self):
        """The one indirection the inherited loader attributes reach through."""
        return self

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Keep the draft's own stages; drop the rest of the checkpoint.

        The drafter is loaded from the same file as the target, in a second
        pass over every tensor in it. Its parameters are named exactly as the
        checkpoint names them, so the remap is identity -- what this is really
        for is the `None`, which is how the loader is told a tensor belongs to
        somebody else rather than that it failed to route.
        """
        return name if name.startswith("mtp.") else None

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Same mapping as the backbone, over the draft's smaller expert set."""
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.dspark_n_routed_experts,
        )

    def share_with_target(self, target_base, loaded=None):
        self.embed, self.head = target_base.embed, target_base.head

    def target_aux_capture_spec(self, layer_ids, hidden_size):
        from atom.spec_decode.drafter import AuxCaptureSpec

        # Published taps (37, 38, 39) have no Engram. A block pre-hook at an
        # Engram layer would read before its injection, violating the checkpoint.
        if set(layer_ids).intersection(self.config.engram_layer_ids):
            raise ValueError("V4.1 DSpark input taps cannot precede Engram injection")
        return AuxCaptureSpec(
            layer_ids, hidden_size, self._target_layer_input, capture="input"
        )

    @staticmethod
    def _target_layer_input(inputs, block):
        residual = inputs[0].residual
        return residual.mean(dim=-2).reshape(-1, residual.shape[-1])

    def project_context(self, aux_concat):
        first = self.mtp[0]
        return first.main_norm(first.main_proj(aux_concat))

    @property
    def context_layers(self):
        return tuple(layer.attn for layer in self.mtp)

    def project_context_kv(self, aux_concat, positions):
        """Project once, then derive each stage's own quantized target keys."""
        hidden = self.project_context(aux_concat)
        return {
            layer.attn.spec.layer_id: layer.attn.project_context(
                hidden, positions, self.rope
            )
            for layer in self.mtp
        }

    def write_context_kv(self, aux_concat, positions):
        from atom.utils.forward_context import get_forward_context

        forward = get_forward_context()
        if forward.context.is_dummy_run:
            return
        metadata = forward.attn_metadata
        cache, step = metadata.cache, metadata.step
        hidden = self.project_context(aux_concat[: step.length].unsqueeze(0))
        for attention in self.context_layers:
            keys = attention.project_context(
                hidden, positions[: step.length][None], self.rope, packed=cache.packed
            )
            cache.write_window(
                attention.spec.layer_id,
                keys,
                step,
            )

    def block_backbone(self, input_ids, positions, num_draft):
        from atom.utils.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        cache, step = metadata.cache, metadata.step
        slots = F.pad(step.slots, (0, input_ids.numel() - step.slots.numel()))
        context = {
            layer.spec.layer_id: cache.read_window(layer.spec.layer_id, slots)
            for layer in self.context_layers
        }
        ring = cache.geometry.ring_slots
        physical = torch.arange(ring, device=positions.device)
        context_positions = positions[:, None] - (positions[:, None] - physical) % ring
        return self.draft_hidden(
            input_ids, positions, context, context_positions, num_draft=num_draft
        )

    def forward_spec(self, input_ids, positions, num_draft=None):
        width = self.block_size if num_draft is None else num_draft
        return self.head_and_sample(
            self.block_backbone(input_ids, positions, width), input_ids, width
        )

    def draft_hidden(
        self, anchor_ids, anchors, context_kv, context_positions, *, num_draft=None
    ):
        """Pure block math; the caller supplies a committed per-request window."""
        if self.embed is None or self.head is None:
            raise RuntimeError("DSpark must share the loaded target embedding and head")
        width = self.block_size if num_draft is None else num_draft
        if not 1 <= width <= self.block_size:
            raise ValueError("Draft width must fit the published DSpark block")
        tokens = anchor_ids.new_full(
            (anchor_ids.numel(), width), self.config.dspark_noise_token_id
        )
        tokens[:, 0] = anchor_ids
        hidden = self.embed(tokens.flatten()).view(
            *tokens.shape, self.config.hidden_size
        )
        state = SinglePassHCState.from_embeddings(hidden, self.config.hc_mult)
        step = draft_step(context_positions, anchors, width, self.window_size)
        for layer in self.mtp:
            state = layer(state, context_kv, step, self.rope)
        hidden = state.collapse()
        return self.mtp[-1].norm(hidden), hidden

    def head_and_sample(self, out, anchor_ids, num_draft):
        normed, hidden = out
        logits = self.head(normed)
        if logits.shape[1] != num_draft:
            raise ValueError("Draft backbone and sampling widths disagree")
        previous = anchor_ids
        tokens, embeddings = [], []
        last = self.mtp[-1]
        for position in range(num_draft):
            bias, embedding = last.markov_head(previous)
            previous = (logits[:, position] + bias).argmax(-1)
            tokens.append(previous)
            embeddings.append(embedding)
        confidence = last.confidence_head(
            hidden, torch.stack(embeddings, dim=1)
        ).sigmoid()
        return torch.stack(tokens, dim=1), confidence
