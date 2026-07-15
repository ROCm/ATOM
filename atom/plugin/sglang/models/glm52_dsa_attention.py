"""GLM-5.2 native MLA attention frontend for SGLang plugin mode."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from atom.model_ops.attention_mla import MLAAttention


@contextmanager
def glm52_native_mla_attention_construction():
    """Temporarily make GLM sparse MLA layers construct native ATOM attention."""

    import atom.models.deepseek_v2 as deepseek_v2

    previous = deepseek_v2.Attention

    def _build_glm52_native_mla_attention(*args: Any, **kwargs: Any):
        mla_modules = kwargs.get("mla_modules", None)
        if (
            kwargs.get("use_mla", False)
            and mla_modules is not None
            and getattr(mla_modules, "is_sparse", False)
        ):
            return SGLangATOMGLM52MLAAttention(*args, **kwargs)
        return previous(*args, **kwargs)

    deepseek_v2.Attention = _build_glm52_native_mla_attention
    try:
        yield
    finally:
        deepseek_v2.Attention = previous


class SGLangATOMGLM52MLAAttention(MLAAttention):
    """Use ATOM native ``MLAAttention`` under SGLang plugin runtime.

    ``DeepseekV2MLAAttention.forward`` calls ``self.mla_attn`` with the ATOM
    model-side argument order, while ``MLAAttention.forward`` has the lower-level
    backend argument order.  This frontend keeps the model-side call contract and
    delegates directly to ``forward_impl``.
    """

    def __init__(self, *args: Any, prefix: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer_name = prefix if prefix is not None else f"GLM52_MLA_{self.layer_num}"
        from atom.config import get_current_atom_config

        get_current_atom_config().compilation_config.static_forward_context[
            self.layer_name
        ] = self

    def _get_forward_batch(self):
        from atom.plugin.sglang.runtime import get_current_forward_batch

        forward_batch = get_current_forward_batch()
        if forward_batch is None:
            raise RuntimeError(
                "forward_batch is required for SGLang GLM-5.2 native MLA attention"
            )
        return forward_batch

    def _infer_total_tokens(self, forward_batch, tensor: torch.Tensor) -> int:
        if hasattr(forward_batch, "input_ids") and forward_batch.input_ids is not None:
            return int(forward_batch.input_ids.shape[0])
        if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
            return int(forward_batch.positions.shape[0])
        if hasattr(forward_batch, "seq_lens_sum"):
            return int(forward_batch.seq_lens_sum)
        return int(tensor.shape[0])

    def _maybe_all_gather(
        self,
        tensor: torch.Tensor | None,
        *,
        total_tokens: int,
        input_scattered: bool,
    ):
        if tensor is None or not input_scattered:
            return tensor
        from sglang.srt.distributed import get_tp_group

        output = tensor.new_empty((total_tokens, *tensor.shape[1:]))
        get_tp_group().all_gather_into_tensor(output, tensor)
        return output

    def forward(
        self,
        q_input: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        forward_batch = self._get_forward_batch()

        from sglang.srt.layers.communicator import get_attn_tp_context

        attn_tp_context = get_attn_tp_context()
        with attn_tp_context.maybe_input_scattered(forward_batch):
            total_tokens = self._infer_total_tokens(forward_batch, q_input)
            q_input = self._maybe_all_gather(
                q_input,
                total_tokens=total_tokens,
                input_scattered=attn_tp_context.input_scattered,
            )
            kv_c_normed = self._maybe_all_gather(
                kv_c_normed,
                total_tokens=total_tokens,
                input_scattered=attn_tp_context.input_scattered,
            )
            k_pe = self._maybe_all_gather(
                k_pe,
                total_tokens=total_tokens,
                input_scattered=attn_tp_context.input_scattered,
            )
            positions = self._maybe_all_gather(
                positions,
                total_tokens=total_tokens,
                input_scattered=attn_tp_context.input_scattered,
            )
            q_scale = self._maybe_all_gather(
                q_scale,
                total_tokens=total_tokens,
                input_scattered=attn_tp_context.input_scattered,
            )

            return self.forward_impl(q_input, kv_c_normed, k_pe, positions, q_scale)
