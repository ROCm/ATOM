# SPDX-License-Identifier: MIT
"""ModelRunner interface over the accepted V4.1 text backbone."""

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context

from .model import Block
from .multimodal import DeepseekV41MultimodalModel


def _fake_layer_output(hidden, layer_name):
    return torch.empty_like(hidden)


# The boundary ops carry a dependency on hidden so the compiler retains and
# orders the stream fork/join with the model's tensor work.
@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_begin_forward(hidden: torch.Tensor) -> None:
    """Read live request state on every execution, including graph capture."""
    metadata = get_forward_context().attn_metadata
    metadata.step.begin_forward()
    if not metadata.step.requests:
        return
    if hidden.shape[-2] != metadata.step.width:
        raise ValueError("Token rows disagree with the width this step declared")
    if metadata.image_mask is not None:
        raise NotImplementedError("V4.1 compiled runtime supports text-only routing")
    stage = getattr(metadata.engram_embeddings, "stage", None)
    if stage is not None:
        stage()


@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_end_forward(hidden: torch.Tensor) -> None:
    metadata = get_forward_context().attn_metadata
    if not metadata.step.requests:
        hidden.zero_()
        return
    rows = metadata.engram_embeddings
    if getattr(rows, "stage", None) is not None:
        rows.join()


@torch_compile_guard(mutates_args=[], gen_fake=_fake_layer_output)
def v41_attention(hidden: torch.Tensor, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return torch.zeros_like(hidden)
    layer, rope = context.no_compile_layers[layer_name]
    return layer.attn(hidden, metadata.cache, metadata.step, rope)


@torch_compile_guard(mutates_args=[], gen_fake=_fake_layer_output)
def v41_engram(residual: torch.Tensor, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return residual.clone()
    layer, _ = context.no_compile_layers[layer_name]
    embeddings = metadata.engram_embeddings.get(layer.engram.layer_id)
    if embeddings is None:
        raise ValueError("Engram rows must be prepared before model execution")
    return layer.engram(residual, embeddings, None)


class RuntimeBlock(Block):
    """Serving uses the same guarded ops in eager and compiled execution."""

    def attention_forward(self, hidden, cache, step, rope):
        return v41_attention(hidden, self.layer_name)

    def engram_forward(self, residual, embeddings, image_mask):
        return v41_engram(residual, self.layer_name)


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # Weights arrive through the shared loader, the way V4's do, so the
    # renames, the packed projections and the expert mapping are declared once
    # as tables on `DeepseekV41ForCausalLM` and inherited here rather than
    # restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which the Engram runtime
    # imports directly and does not route through here.

    block_cls = RuntimeBlock

    def __init__(self, atom_config):
        config = atom_config
        super().__init__(
            config.hf_config,
            max_length=config.max_model_len,
            online_quant_config=config.online_quant_config,
        )
        for spec, layer in zip(self.topology, self.layers):
            rope = self.global_rope if spec.ratio else self.window_rope
            atom_config.compilation_config.static_forward_context[layer.layer_name] = (
                layer,
                rope,
            )

    def begin_forward(self, hidden, engram_embeddings):
        v41_begin_forward(hidden)

    def end_forward(self, hidden, engram_embeddings):
        v41_end_forward(hidden)

    def forward(self, input_ids, positions, inputs_embeds=None):
        """Tensor-only serving entry; guarded ops read the live forward context."""
        return self.forward_hidden(
            input_ids.unsqueeze(0),
            None,
            None,
            inputs_embeds=(
                None if inputs_embeds is None else inputs_embeds.unsqueeze(0)
            ),
        ).squeeze(0)

    def compute_logits(self, hidden):
        return self.head.get_logits(self.norm(hidden))
