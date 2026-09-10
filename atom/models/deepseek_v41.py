"""DeepSeek-V4.1 with only Engram implemented.

The real architecture -- CSA2, Single-Pass mHC, DSpark, CED, the vision tower,
the MoE -- is NOT implemented. Attention and the feed-forward network are
deliberate stubs that preserve the residual and compute nothing, so this model
produces no meaningful output and must not be used for accuracy.

What it does do is load and run through ATOM's ordinary serving path, which is
what the Engram host path needs in order to be exercised at all: the model
registry resolves it, the checkpoint's engram tensors are loaded, ModelRunner
finds `build_engram_runtime`, and the engram modules at layers 1 and 14 read
embeddings that were gathered on the host and staged to the device.

Replace the stubs as the real layers land. The engram wiring is three lines and
is marked below.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from atom.model_ops.engram_module import EngramModules

logger = logging.getLogger(__name__)


class DeepseekV41StubLayer(nn.Module):
    """Placeholder for a real decoder layer: identity on the residual stream.

    A real layer would run CSA2 attention and the MoE here. Neither is
    implemented; the residual passes through untouched so the layers around
    engram behave predictably while engram itself is exercised for real.
    """

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states


class DeepseekV41Model(nn.Module):
    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        hf = config.hf_config
        self.hidden_size = int(hf.hidden_size)
        self.hc_mult = int(getattr(hf, "hc_mult", 4))
        self.num_layers = int(hf.num_hidden_layers)
        dtype = getattr(config, "dtype", torch.bfloat16)

        self.embed_tokens = nn.Embedding(
            int(hf.vocab_size), self.hidden_size, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [DeepseekV41StubLayer(i) for i in range(self.num_layers)]
        )
        self.norm = nn.RMSNorm(
            self.hidden_size, eps=float(getattr(hf, "rms_norm_eps", 1e-6)), dtype=dtype
        )

        # --- engram wiring, 1 of 3 ---------------------------------------
        self.engram = EngramModules.from_checkpoint(
            config.model, hf_config=hf.to_dict(), dtype=dtype
        )
        # Set by build_engram_runtime; ModelRunner stages into it before each
        # forward. Not a module attribute, so it stays out of the state dict.
        self.engram_runtime = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[num_tokens] -> [num_tokens, hc_mult, hidden].

        The multi-branch residual is what the real mHC stack carries and what
        EngramOp gates against, so it is the shape this returns even though the
        branches are identical here.
        """
        hidden = self.get_input_embeddings(input_ids)
        hidden = hidden.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()

        for layer_id, layer in enumerate(self.layers):
            hidden = layer(hidden, positions)
            # --- engram wiring, 2 of 3 -----------------------------------
            if self.engram is not None and layer_id in self.engram:
                if self.engram_runtime is None:
                    raise RuntimeError(
                        f"layer {layer_id} carries engram but no runtime was "
                        f"supplied; ModelRunner stages the embeddings before "
                        f"calling forward"
                    )
                hidden = hidden + self.engram[layer_id](
                    hidden, self.engram_runtime.embeddings(layer_id)
                )
        return hidden


class DeepseekV41ForCausalLM(nn.Module):
    """ATOM model contract. Engram is real; everything else is a stub."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        hf = config.hf_config
        dtype = getattr(config, "dtype", torch.bfloat16)
        self.model = DeepseekV41Model(config, prefix=prefix)
        # ModelRunner unpacks these straight into torch.empty, so they are the
        # actual sizes between N and dim -- ints, set per instance, matching the
        # [num_tokens, hc, dim] residual forward returns.
        self.extra_output_dims: tuple[int, ...] = (self.model.hc_mult,)
        self.lm_head = nn.Linear(
            int(hf.hidden_size), int(hf.vocab_size), bias=False, dtype=dtype
        )
        logger.warning(
            "DeepseekV41ForCausalLM: attention and MoE are stubs; output is not "
            "meaningful. Engram on layers %s is real.",
            list(self.model.engram.layer_ids) if self.model.engram else [],
        )

    # --- engram wiring, 3 of 3 -------------------------------------------
    def build_engram_runtime(self, device: torch.device, max_num_tokens: int):
        """ModelRunner probes for this name and skips the engram path without it.

        The runtime is kept on the model as well as returned: `run_model` calls
        forward with a fixed (input_ids, positions) signature, so there is no
        argument to pass it through, and the layers have to reach it themselves.
        """
        if self.model.engram is None:
            return None
        runtime = self.model.engram.build_engram_runtime(device, max_num_tokens)
        self.model.engram_runtime = runtime
        return runtime

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids, positions) -> torch.Tensor:
        """The signature ModelRunner calls: no room for engram state, so the
        runtime is read off the model where build_engram_runtime left it."""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[num_tokens, hc_mult, hidden] -> [num_tokens, vocab].

        The real model reduces the branches through an mHC head; averaging is a
        stand-in that keeps the shapes honest.
        """
        return self.lm_head(self.model.norm(hidden_states.mean(dim=-2)))

    def load_weights(self, weights) -> set[str]:
        """Consume the checkpoint without storing it.

        Engram tensors were loaded from the checkpoint directly in
        EngramModules.from_checkpoint; the stub layers have nothing to fill, so
        every other tensor is counted and dropped. Returning an empty set tells
        the loader nothing here was claimed.
        """
        seen = 0
        for _name, _tensor in weights:
            seen += 1
        logger.info("DeepseekV41ForCausalLM: ignored %d checkpoint tensors", seen)
        return set()
