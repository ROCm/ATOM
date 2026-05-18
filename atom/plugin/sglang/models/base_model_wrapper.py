"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.

To add a new model, append its architecture class name to _MODEL_NAMES.
"""

import copy

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger("atom.plugin.sglang.models")


_current_forward_batch: ContextVar[Optional[ForwardBatch]] = ContextVar(
    "atom_sglang_current_forward_batch", default=None
)


def get_current_forward_batch():
    return _current_forward_batch.get()


def _is_dummy_forward(forward_batch: ForwardBatch) -> bool:
    # SGLang's IDLE batch is the plugin-side equivalent of ATOM dummy run.
    forward_mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_idle")
        and forward_mode.is_idle()
    )


def _pad_dummy_like(
    tensor: Optional[torch.Tensor],
    *,
    length: int,
    fill_value: int | float = 0,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    shape = (length, *tensor.shape[1:])
    return torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)


def _materialize_atom_dummy_forward(
    input_ids: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    input_embeds: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    ForwardBatch,
]:
    """Convert an empty SGLang IDLE batch into ATOM-style dummy forward inputs."""
    dummy_positions = positions.new_zeros((1,))
    dummy_input_ids = input_ids.new_zeros((1,))
    dummy_input_embeds = _pad_dummy_like(input_embeds, length=1, fill_value=0)

    model_forward_batch = copy.copy(forward_batch)
    model_forward_batch.positions = dummy_positions
    model_forward_batch.batch_size = 1
    model_forward_batch.seq_lens_sum = 1
    model_forward_batch.seq_lens = forward_batch.seq_lens.new_ones((1,))
    model_forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.new_ones((1,))

    return dummy_input_ids, dummy_positions, dummy_input_embeds, model_forward_batch


def _trim_hidden_states_for_output(hidden_states, num_tokens: int):
    if torch.is_tensor(hidden_states):
        return hidden_states[:num_tokens]
    if isinstance(hidden_states, tuple):
        return tuple(
            tensor[:num_tokens] if torch.is_tensor(tensor) else tensor
            for tensor in hidden_states
        )
    return hidden_states


def _resolve_num_tokens_across_dp(
    atom_config: Any,
    forward_batch: ForwardBatch,
    num_tokens: int,
    is_dummy_run: bool,
) -> torch.Tensor:
    """Resolve per-DP token counts for ATOM's CPU-side DPMetadata.

    Real SGLang dp-attention batches carry ``global_num_tokens_cpu`` from the
    scheduler.  That list is the source of truth for mixed prefill/decode/idle
    batches, where token counts may look like [8, 1, 8, 8].

    Some SGLang synthetic/static batches, especially CUDA graph capture batches,
    only keep the global token buffer on GPU.  ATOM's DPMetadata is CPU-side and
    needs a CPU tensor before model forward, so avoid reading the GPU buffer back
    to CPU.  We only fallback when the batch advertises the same-shape DP buffer
    layout (global_dp_buffer_len == local_num_tokens * dp_size), where the CPU
    equivalent is exactly [local_num_tokens] * dp_size.

    IDLE batches are reported by SGLang as 0 tokens on the current rank, but
    this wrapper materializes them as one local dummy token before entering
    ATOM.  Patch the current DP rank after resolving the distribution so
    ``DPMetadata`` sees a local count that matches the actual ATOM input.
    """
    global_num_tokens_cpu = getattr(forward_batch, "global_num_tokens_cpu", None)
    if global_num_tokens_cpu is not None:
        num_tokens_across_dp = torch.tensor(
            global_num_tokens_cpu, dtype=torch.int32, device="cpu"
        )
    else:
        dp_size = atom_config.parallel_config.data_parallel_size
        global_num_tokens_gpu = getattr(forward_batch, "global_num_tokens_gpu", None)
        global_dp_buffer_len = getattr(forward_batch, "global_dp_buffer_len", None)
        is_static_same_shape_batch = (
            global_num_tokens_gpu is not None
            and global_dp_buffer_len == num_tokens * dp_size
        )
        if not is_static_same_shape_batch:
            raise RuntimeError(
                "[SGL+ATOM] SGLang dp-attention requires "
                "forward_batch.global_num_tokens_cpu unless the batch uses static "
                "same-shape DP metadata."
            )

        # Static batches, such as CUDA graph capture batches, may only keep
        # global token counts on GPU. Avoid GPU-to-CPU reads here and mirror
        # their same-shape layout directly for ATOM's CPU DPMetadata.
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )

    if is_dummy_run:
        # SGLang reports idle ranks as 0 tokens, but ATOM materializes them
        # as one local dummy token so collectives and DPMetadata stay aligned.
        dp_rank = atom_config.parallel_config.data_parallel_rank
        num_tokens_across_dp[dp_rank] = num_tokens
    return num_tokens_across_dp


def _set_sglang_forward_context(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
) -> None:
    """Bridge SGLang batch metadata into ATOM's global forward context."""
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    forward_mode = forward_batch.forward_mode
    # TODO: This max_seqlen_q is not the source of truth for prefill attention;
    # SGLang plugin attention consumes forward_batch.attn_backend.forward_metadata
    # directly.  In this wrapper it is only needed by ATOM MoE padding: under
    # dp-attention + TP (non-EP all_gather/reduce_scatter), decode/idle batches
    # must use 1 so pad_for_all_gather keeps fixed-shape collectives aligned.
    # Leaving it as 0 there can make active and dummy ranks send different
    # shapes to DP all_gather and hang.
    max_seqlen_q = 1 if forward_mode.is_decode_or_idle() else 0
    attn_metadata = AttentionMetaData(max_seqlen_q=max_seqlen_q)
    batch_size = int(forward_batch.batch_size)
    is_dummy_run = _is_dummy_forward(forward_batch)
    is_prefill = forward_mode.is_prefill()
    num_tokens = int(positions.shape[0])

    enable_dp_attention = bool(atom_config.enable_dp_attention)
    if enable_dp_attention:
        # SGLang owns the cross-DP token distribution under dp-attention; ATOM
        # uses it to derive graph_bs and fixed-size MoE gather/scatter buffers.
        num_tokens_across_dp = _resolve_num_tokens_across_dp(
            atom_config, forward_batch, num_tokens, is_dummy_run
        )
        graph_bs = int(torch.max(num_tokens_across_dp).item())
    else:
        # Without dp-attention, ATOM runs with local-rank shapes only.  There is
        # no cross-DP token distribution to pass into DPMetadata, so graph_bs
        # follows the local prefill token count or decode batch size.
        num_tokens_across_dp = None
        graph_bs = num_tokens if is_prefill else batch_size
    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=is_dummy_run,
        batch_size=batch_size,
        graph_bs=graph_bs,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def _reset_sglang_forward_context() -> None:
    from atom.utils.forward_context import reset_forward_context

    reset_forward_context()


@dataclass(frozen=True)
class SGLangForwardBatchMetadata:
    """Small context object for one SGLang model forward."""

    forward_batch: Optional[ForwardBatch]
    pp_proxy_tensors: Optional[PPProxyTensors] = None
    save_kv_cache: bool = True
    _current: ClassVar[ContextVar[Optional["SGLangForwardBatchMetadata"]]] = ContextVar(
        "atom_sglang_current_forward_batch_metadata",
        default=None,
    )

    @classmethod
    def current(cls) -> Optional["SGLangForwardBatchMetadata"]:
        return cls._current.get()

    @classmethod
    def build(
        cls,
        forward_batch: Optional[
            Union[ForwardBatch, "SGLangForwardBatchMetadata"]
        ] = None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        save_kv_cache: Optional[bool] = None,
    ) -> Optional["SGLangForwardBatchMetadata"]:
        if isinstance(forward_batch, cls):
            return forward_batch
        if forward_batch is None and pp_proxy_tensors is None and save_kv_cache is None:
            return cls.current()
        return cls(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=True if save_kv_cache is None else save_kv_cache,
        )

    @classmethod
    @contextmanager
    def bind(cls, metadata: Optional["SGLangForwardBatchMetadata"]):
        meta_token = cls._current.set(metadata)
        batch_token = _current_forward_batch.set(
            None if metadata is None else metadata.forward_batch
        )
        try:
            yield metadata
        finally:
            _current_forward_batch.reset(batch_token)
            cls._current.reset(meta_token)

    @staticmethod
    def to_intermediate_tensors(
        intermediate_tensors,
        metadata: Optional["SGLangForwardBatchMetadata"],
    ):
        if intermediate_tensors is not None or metadata is None:
            return intermediate_tensors
        pp_proxy_tensors = metadata.pp_proxy_tensors
        if pp_proxy_tensors is None:
            return intermediate_tensors
        tensors = getattr(pp_proxy_tensors, "tensors", None)
        if tensors is None:
            return intermediate_tensors
        from atom.models.utils import IntermediateTensors

        return IntermediateTensors(dict(tensors))


@dataclass(frozen=True)
class ModelArchSpec:
    wrapper_binds_gdn_context: bool = False
    apply_deepseek_patch: bool = False


_MODEL_ARCH_SPECS = {
    "DeepseekV3ForCausalLM": ModelArchSpec(apply_deepseek_patch=True),
    "Qwen3MoeForCausalLM": ModelArchSpec(),
    "Qwen3NextForCausalLM": ModelArchSpec(wrapper_binds_gdn_context=True),
}


class _AtomCausalLMBaseForSglang(nn.Module):
    """Base ATOM model wrapper conforming to sglang's model interface.

    Delegates model creation and weight loading to ATOM's plugin system,
    while providing the forward signature and LogitsProcessorOutput return
    type that sglang expects.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size
        self.model_arch = getattr(config, "architectures", [""])[0]
        self.model_arch_spec = _MODEL_ARCH_SPECS.get(self.model_arch, ModelArchSpec())

        import atom

        # TODO: prepare_model() currently handles model construction, config
        # generation, attention backend registration, and distributed init.
        # Refactor so this wrapper only dispatches the attention backend
        # (register_ops_to_sglang + set_attn_cls), and let sglang handle
        # model construction directly
        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            raise ValueError(
                f"ATOM failed to create model for architecture {self.model_arch}"
            )

        # Under SGLang dp-attention, ATOM runtime interprets non-MoE modules
        # like lm_head with tp=1 semantics, so plugin logits must not perform
        # an extra TP all-gather after local lm_head matmul.
        plugin_skip_all_gather = bool(self.model.atom_config.enable_dp_attention)
        self.logits_processor = LogitsProcessor(
            config, skip_all_gather=plugin_skip_all_gather
        )

        # Apply ds model-specific sglang patches (attn dispatch, weight hooks, etc.)
        # TODO: will remove this after sglang supports atom attention backend
        if self.model_arch_spec.apply_deepseek_patch:
            from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
                setup_deepseek_for_sglang,
            )

            setup_deepseek_for_sglang(self.model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        metadata = SGLangForwardBatchMetadata.build(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=model_kwargs.get("save_kv_cache"),
        )

        if _is_dummy_forward(forward_batch):
            (
                model_input_ids,
                model_positions,
                model_input_embeds,
                model_forward_batch,
            ) = _materialize_atom_dummy_forward(
                input_ids,
                positions,
                input_embeds,
                forward_batch,
            )
        else:
            (
                model_input_ids,
                model_positions,
                model_input_embeds,
                model_forward_batch,
            ) = (
                input_ids,
                positions,
                input_embeds,
                forward_batch,
            )
        model_inputs = dict(
            input_ids=model_input_ids,
            positions=model_positions,
            intermediate_tensors=SGLangForwardBatchMetadata.to_intermediate_tensors(
                pp_proxy_tensors, metadata
            ),
            inputs_embeds=model_input_embeds,
        )
        with SGLangForwardBatchMetadata.bind(metadata):
            if self.model_arch_spec.wrapper_binds_gdn_context:
                from atom.plugin.sglang.attention_backend.attention_gdn import (
                    SGLangGDNForwardContext,
                )

                with SGLangGDNForwardContext.bind(metadata):
                    hidden_states = self.model(**model_inputs)
            else:
                try:
                    _set_sglang_forward_context(
                        self.model.atom_config, model_forward_batch, model_positions
                    )
                    hidden_states = self.model(**model_inputs)
                finally:
                    _reset_sglang_forward_context()

        if self.pp_group.is_last_rank:
            if _is_dummy_forward(forward_batch):
                # TODO: Revisit if SGLang ever sends non-empty dummy batches.
                # Today this path only runs when an empty IDLE batch is expanded
                # to one ATOM dummy token, so the output boundary must trim back to
                # the original SGLang-visible length: 0 tokens.
                hidden_states = _trim_hidden_states_for_output(hidden_states, 0)
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.model.lm_head,
                forward_batch,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self.model, config=self.model.atom_config, prefix="model."
        )


EntryClass = []
for _name in _MODEL_ARCH_SPECS:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)
