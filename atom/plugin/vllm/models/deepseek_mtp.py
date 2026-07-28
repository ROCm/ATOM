"""vLLM-specific DeepSeek MTP model extensions."""

import torch

from atom.models.deepseek_mtp import DeepSeekMTP as DeepSeekMTPBase


class DeepSeekMTP(DeepSeekMTPBase):
    """Adapt the native DeepSeek MTP model to vLLM's recycle-state contract."""

    def get_recycle_hidden(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.model.num_mtp_layers
        mtp_layer = self.model.layers[
            str(self.model.mtp_start_layer_idx + current_step_idx)
        ]
        return mtp_layer.shared_head(hidden_states)
