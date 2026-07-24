"""GLM-5.2 ATOM-native multi-step draft backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
    ATOMGLM52DSABackendForSgl,
)


@dataclass(frozen=True)
class GLM52NativeDraftBackendContract:
    topk: int
    speculative_num_steps: int
    page_size: int

    def validate(self) -> None:
        if self.topk != 1:
            raise ValueError(
                "GLM52NativeMultiStepDraftBackend currently requires topk=1, "
                f"got {self.topk}"
            )
        if self.speculative_num_steps < 2:
            raise ValueError(
                "GLM52NativeMultiStepDraftBackend requires at least two "
                f"speculative steps, got {self.speculative_num_steps}"
            )
        if self.page_size <= 0:
            raise ValueError(f"GLM-5.2 draft page size must be positive, got {self.page_size}")


class GLM52NativeDraftStepBackend(ATOMGLM52DSABackendForSgl):
    """One native ATOM attention backend instance for one draft substep."""

    def __init__(self, model_runner, sub_step: int):
        super().__init__(model_runner)
        self.sub_step = int(sub_step)

    @staticmethod
    def get_name() -> str:
        return "atom_glm52_native_draft_step"

    def set_eager_metadata(self, forward_batch, metadata) -> None:
        self.forward_metadata = forward_batch
        self.atom_glm52_graph_metadata = metadata


class GLM52NativeMultiStepDraftBackend:
    """Own eager ATOM metadata for every GLM-5.2 draft-decode substep.

    P1 intentionally supports eager initialization only. CUDA graph state,
    capture, and replay belong to P4 and fail loudly until that phase is
    implemented against this same owner.
    """

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        self.model_runner = model_runner
        draft_model = getattr(model_runner, "model", None)
        if not bool(
            getattr(draft_model, "_atom_glm52_uses_native_draft_frontend", False)
        ):
            raise RuntimeError(
                "GLM52NativeMultiStepDraftBackend requires the P2 native draft "
                "frontend; refusing a native-backend/generic-frontend hybrid"
            )
        self.topk = int(topk)
        self.speculative_num_steps = int(speculative_num_steps)
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.atom_config = getattr(draft_model, "atom_config", None)
        if self.atom_config is None:
            raise RuntimeError(
                "GLM52NativeMultiStepDraftBackend requires draft model atom_config"
            )
        self.contract = GLM52NativeDraftBackendContract(
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
            page_size=int(model_runner.server_args.page_size),
        )
        self.contract.validate()
        self._bind_native_model_cache_views(draft_model)
        self.attn_backends = [
            GLM52NativeDraftStepBackend(model_runner, sub_step)
            for sub_step in range(self.speculative_num_steps - 1)
        ]

    @staticmethod
    def get_name() -> str:
        return "atom_glm52_native_multi_step_draft"

    def _bind_native_model_cache_views(self, draft_model) -> None:
        from atom.plugin.sglang.glm52_mtp.cache_bind import (
            bind_glm52_dsa_cache_views,
        )
        from atom.plugin.sglang.glm52_mtp.common import (
            ensure_shared_sparse_buffer,
            get_index_topk,
            validate_page_size,
        )

        validate_page_size(self.token_to_kv_pool, self.atom_config)
        shared_sparse = ensure_shared_sparse_buffer(
            self.token_to_kv_pool,
            num_tokens=int(self.model_runner.req_to_token_pool.size),
            topk=get_index_topk(self.atom_config),
            device=self.device,
        )
        if not bind_glm52_dsa_cache_views(draft_model, self.token_to_kv_pool):
            raise RuntimeError(
                "GLM52NativeMultiStepDraftBackend failed to bind native KV/indexer "
                "cache views"
            )

        from atom.models.deepseek_v2 import DeepseekV2MLAAttention

        for module in draft_model.modules():
            if not isinstance(module, DeepseekV2MLAAttention):
                continue
            indexer = getattr(module, "indexer", None)
            if indexer is not None and indexer.sparse_kv_indices_buffer.data_ptr() != shared_sparse.data_ptr():
                raise RuntimeError("GLM-5.2 draft indexer is not bound to backend sparse buffer")
            mla_attn = getattr(module, "mla_attn", None)
            bound = getattr(mla_attn, "sparse_kv_indices_buffer", None)
            if torch.is_tensor(bound) and bound.data_ptr() != shared_sparse.data_ptr():
                raise RuntimeError("GLM-5.2 draft attention is not bound to backend sparse buffer")

    def _out_cache_rows(self, forward_batch) -> torch.Tensor:
        out_cache_loc = forward_batch.out_cache_loc
        if not torch.is_tensor(out_cache_loc):
            raise RuntimeError("GLM-5.2 native draft requires out_cache_loc")
        batch_size = int(forward_batch.batch_size)
        expected = batch_size * self.topk * self.speculative_num_steps
        if int(out_cache_loc.numel()) != expected:
            raise RuntimeError(
                "GLM-5.2 native draft out_cache_loc layout mismatch: "
                f"got={int(out_cache_loc.numel())}, expected={expected}, "
                f"batch_size={batch_size}, topk={self.topk}, "
                f"steps={self.speculative_num_steps}"
            )
        return (
            out_cache_loc.reshape(batch_size, self.topk, self.speculative_num_steps)
            .permute(2, 0, 1)
            .reshape(self.speculative_num_steps, -1)
            .contiguous()
        )

    def init_forward_metadata(self, forward_batch) -> None:
        """Build fresh native ATOM metadata for each eager draft substep."""
        from atom.plugin.sglang.glm52_mtp.draft_decode import (
            clear_draft_decode_sub_step,
            set_draft_decode_sub_step,
        )
        from atom.plugin.sglang.glm52_mtp.dispatcher import (
            build_atom_glm52_attention_metadata_from_sglang,
        )

        if forward_batch.spec_info is None:
            raise RuntimeError("GLM-5.2 native draft metadata requires spec_info")
        if int(forward_batch.batch_size) <= 0:
            raise RuntimeError("GLM-5.2 native draft metadata requires a live batch")

        original_out_cache_loc = forward_batch.out_cache_loc
        original_positions = forward_batch.positions
        out_cache_rows = self._out_cache_rows(forward_batch)
        forward_batch._atom_glm52_speculative_num_steps = self.speculative_num_steps

        try:
            for sub_step, backend in enumerate(self.attn_backends):
                set_draft_decode_sub_step(forward_batch, sub_step)
                forward_batch.out_cache_loc = out_cache_rows[sub_step]
                forward_batch.positions = original_positions + sub_step
                metadata = build_atom_glm52_attention_metadata_from_sglang(
                    forward_batch,
                    forward_batch.positions,
                    token_to_kv_pool=self.token_to_kv_pool,
                    req_to_token_pool=self.req_to_token_pool,
                    atom_config=self.atom_config,
                )
                backend.set_eager_metadata(forward_batch, metadata)
        finally:
            clear_draft_decode_sub_step(forward_batch)
            forward_batch.out_cache_loc = original_out_cache_loc
            forward_batch.positions = original_positions

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        del max_bs, max_num_tokens
        raise NotImplementedError(
            "GLM52NativeMultiStepDraftBackend CUDA graph state belongs to P4"
        )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch) -> None:
        del forward_batch
        raise NotImplementedError(
            "GLM52NativeMultiStepDraftBackend graph capture belongs to P4"
        )

    def init_forward_metadata_replay_cuda_graph(self, forward_batch, bs: int) -> None:
        del forward_batch, bs
        raise NotImplementedError(
            "GLM52NativeMultiStepDraftBackend graph replay belongs to P4"
        )