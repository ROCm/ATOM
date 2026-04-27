"""RTP-LLM scoped patch for ATOM qwen3_next model path."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("atom.plugin.rtpllm.models.qwen3_next")

_PATCHED = False


def apply_qwen3_next_rtpllm_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import atom.models.qwen3_next as qwen3_next

    def _split_router_logits(self, router_logits: torch.Tensor):
        n_shared = int(getattr(self, "n_shared_experts", 0) or 0)
        if n_shared <= 0:
            n_routed = int(getattr(self, "n_routed_experts", 0) or 0)
            total_experts = int(router_logits.shape[-1])
            # Backward-compatible inference when main path has no `n_shared_experts`.
            if n_routed > 0 and total_experts > n_routed:
                n_shared = total_experts - n_routed
        if self.shared_expert is None or n_shared <= 0:
            return router_logits, None
        return torch.split(
            router_logits,
            [self.n_routed_experts, n_shared],
            dim=-1,
        )

    def _apply_shared_expert_gate(
        shared_output: torch.Tensor, shared_expert_gate_logits: torch.Tensor | None
    ) -> torch.Tensor:
        if shared_expert_gate_logits is None:
            return shared_output
        return torch.sigmoid(shared_expert_gate_logits) * shared_output

    def _patched_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        router_logits, shared_expert_gate_logits = self._split_router_logits(
            router_logits
        )
        routed_output = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if (
            not qwen3_next.is_rocm_aiter_fusion_shared_expert_enabled()
            and self.shared_expert is not None
        ):
            shared_output = self.shared_expert(hidden_states)
            shared_output = self._apply_shared_expert_gate(
                shared_output, shared_expert_gate_logits
            )
            final_hidden_states = shared_output + routed_output
        else:
            final_hidden_states = routed_output

        if self.tp_size > 1:
            final_hidden_states = qwen3_next.tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        return final_hidden_states.view(orig_shape)

    cls = qwen3_next.Qwen3NextSparseMoeBlock
    # Main path references `self.shared_expert_gate` but does not always initialize it.
    # Set a class-level default so plugin mode won't crash on attribute lookup.
    cls.shared_expert_gate = None
    cls._split_router_logits = _split_router_logits
    cls._apply_shared_expert_gate = staticmethod(_apply_shared_expert_gate)
    cls.forward = _patched_sparse_moe_forward

    _PATCHED = True
    logger.info("Applied RTP patch for atom.models.qwen3_next.Qwen3NextSparseMoeBlock")

