from typing import Generic, Optional, TypeVar
import logging

from dataclasses import dataclass

import torch

from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.dist.parallel_state import get_tp_group
from atom.plugin.prepare import is_vllm, is_sglang
from atom.utils import CpuGpuBuffer, envs
from atom.config import get_current_atom_config

from atom.utils.forward_context import Context, AttentionMetaData
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.attention_mla import MLAAttention

logger = logging.getLogger("atom")

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION


@dataclass
class AiterFlashAttentionPhaseMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


AiterFlashAttentionDecodeMetadata = AiterFlashAttentionPhaseMetadata
AiterFlashAttentionPrefillMetadata = AiterFlashAttentionPhaseMetadata


@dataclass
class AiterChunkSlidingWindowMetadata:
    swa_seqlens: torch.Tensor
    swa_cu_seqlens: torch.Tensor
    swa_seq_starts: torch.Tensor
    swa_token_to_batch: torch.Tensor
    swa_max_seqlens: int
    swa_total_tokens: int
    swa_workspace: torch.Tensor


@dataclass
class AiterChunkContextMetadata:
    workspace: torch.Tensor
    cu_seq_lens_chunk: torch.Tensor
    chunk_starts: torch.Tensor
    token_to_batch: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    seq_lens: torch.Tensor
    num_chunks: int
    total_token_per_batch: list[int]
    swa_metadata: Optional[AiterChunkSlidingWindowMetadata] = None


@dataclass
class AiterFlashAttentionChunkPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    chunk_context_metadata: AiterChunkContextMetadata


@dataclass
class AiterFlashAttentionMetadataForPluginMode:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor

    # prefill and decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_extends: int
    num_extend_tokens: int

    decode_metadata: Optional[AiterFlashAttentionDecodeMetadata] = None
    prefill_metadata: Optional[AiterFlashAttentionPrefillMetadata] = None
    extend_metadata: Optional[AiterFlashAttentionChunkPrefillMetadata] = None

    use_cascade: bool = False
    common_prefix_len: int = 0
    total_tokens: int = 0

    context: Optional[Context] = None


class vllmAiterAttentionBackendMethods:
    # here attention in ATOM doesn't accept the output buffer because
    # ATOM works as a model impl backend, it needs the maximum freedom
    # to decide the output buffer and shape, thus here use this flag to
    # let vllm don't allocate the output buffer for ATOM. ATOM will
    # handle the output buffer by itself
    accept_output_buffer: bool = False
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False


def create_attn_metadata_builder_init_method(base_class):
    """
    Create the init method for metadata builder
    """

    def init_method_under_plugin_mode(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        base_class.__init__(self, kv_cache_spec, layer_names, config, device)
        logger.info("init AiterAttentionMetadataBuilder for plugin mode")
        from vllm.config import VllmConfig, get_layers_from_vllm_config

        try:
            from vllm.attention.layer import Attention
        except ImportError:
            from vllm.model_executor.layers.attention import Attention

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config

        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.aot_sliding_window: Optional[tuple[int, int]] = None
        self.total_tokens: int = 0

        self.scheduler_config = config.scheduler_config
        self.block_ratio = 1

        sliding_window_sizes: set[tuple[int, int] | None] = set()
        layers = get_layers_from_vllm_config(config, Attention)
        for layer in layers.values():
            assert isinstance(layer.impl, PagedAttentionImpl)
            sliding_window = layer.impl.sliding_window
            if sliding_window is None or sliding_window == -1:
                sliding_window_sizes.add(None)
            elif isinstance(sliding_window, tuple):
                sliding_window_sizes.add(sliding_window)
            else:
                sliding_window_sizes.add((sliding_window - 1, 0))

        while len(sliding_window_sizes) > 0:
            sliding_window_config = sliding_window_sizes.pop()
            if sliding_window_config is not None and sliding_window_config[0] != -1:
                assert (
                    self.aot_sliding_window is None
                ), "Aiter Backend only support one valid sliding window"
                self.aot_sliding_window = sliding_window_config

        # for extend path to store the fetched key and value
        # here buffer used for extend path is not calculated by vLLM and SGLang
        # when profile_run, it is possible to exhaust the GPU memory when
        # gpu_mem_utilization is much higher
        self.extend_workspace = torch.empty(
            [2, _CP_TOKENS_PER_ITER_ROCM, self.num_heads_kv, self.head_dim],
            dtype=self.model_config.dtype,
            device=device,
        )
        workspace_bytes = (
            2
            * _CP_TOKENS_PER_ITER_ROCM
            * self.num_heads_kv
            * self.head_dim
            * torch.tensor([], dtype=self.model_config.dtype).element_size()
        )
        workspace_mib = workspace_bytes / (1024 * 1024)
        logger.warning(
            "ATOM allocates extend_workspace outside vLLM memory accounting: "
            "shape=%s dtype=%s size=%.2f MiB. "
            "This untracked GPU memory can increase OOM risk when "
            "gpu_mem_utilization is high.",
            tuple(self.extend_workspace.shape),
            self.model_config.dtype,
            workspace_mib,
        )

        # used for ROPE
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": device}
        self.positions = CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs)

    return init_method_under_plugin_mode


def setup_attn_metadata_builder_base_class_and_attributes(class_dict: dict):
    """
    Setup the base class and attributes for attention metadata builder
    """
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionMetadataBuilder,
    )

    base_class = AttentionMetadataBuilder
    generic_base = AttentionMetadataBuilder
    needs_generic = True

    # align with vllm rocm aiter fa
    class_dict["_cudagraph_support"] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    class_dict["reorder_batch_threshold"] = 1

    return base_class, generic_base, needs_generic, class_dict


class vllmAttentionMetadataBuilderMethods:
    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata=None,
        fast_build: bool = False,
    ):
        if common_prefix_len > 0:
            raise ValueError("ATOM does not support cascade attention yet")

        from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

        # here assume the decode num token is 1 per request
        split_ret = split_decodes_prefills_and_extends(
            common_attn_metadata=common_attn_metadata, decode_threshold=1
        )

        (
            num_decodes,
            num_extends,
            num_prefills,
            num_decode_tokens,
            num_extend_tokens,
            num_prefill_tokens,
        ) = split_ret

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens.cpu()
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=query_lens_cpu[:num_decodes].max().item(),
                min_query_len=query_lens_cpu[:num_decodes].min().item(),
                max_seq_len=seq_lens[:num_decodes].max().item(),
                query_start_loc=common_attn_metadata.query_start_loc[: num_decodes + 1],
            )

        extend_metadata = None
        if num_extends > 0:
            num_extends_slice = slice(num_decodes, num_decodes + num_extends)
            query_lens_for_extend = query_lens_cpu[num_extends_slice]
            seq_lens_for_extend = seq_lens[num_extends_slice]
            computed_kv_lens = seq_lens_for_extend - query_lens_for_extend
            swa_metadata = None
            if self.aot_sliding_window is not None:
                swa_seqlen_for_extend = torch.minimum(
                    seq_lens_for_extend,
                    query_lens_for_extend + self.aot_sliding_window[0] + 1,
                )
                cu_seq_lens = torch.zeros(
                    num_extends + 1,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                torch.cumsum(
                    swa_seqlen_for_extend,
                    dim=0,
                    dtype=cu_seq_lens.dtype,
                    out=cu_seq_lens[1:],
                )
                token_to_seq = torch.arange(
                    0,
                    num_extends,
                    dtype=torch.int32,
                    device=seq_lens_for_extend.device,
                )
                token_to_seq = torch.repeat_interleave(
                    token_to_seq, swa_seqlen_for_extend
                )
                fetched_shape = cu_seq_lens[-1].item()
                swa_workspace = torch.empty(
                    (2, fetched_shape, self.num_heads_kv, self.head_dim),
                    dtype=self.vllm_config.model_config.dtype,
                    device=self.device,
                )

                seq_starts = seq_lens_for_extend - swa_seqlen_for_extend
                max_seqlen_k = swa_seqlen_for_extend.max().item()
                total_tokens = cu_seq_lens[-1].item()

                swa_metadata = AiterChunkSlidingWindowMetadata(
                    swa_seqlens=swa_seqlen_for_extend.to(
                        self.device, non_blocking=True
                    ),
                    swa_cu_seqlens=cu_seq_lens.to(self.device, non_blocking=True),
                    swa_seq_starts=seq_starts.to(self.device, non_blocking=True),
                    swa_token_to_batch=token_to_seq.to(self.device, non_blocking=True),
                    swa_max_seqlens=max_seqlen_k,
                    swa_total_tokens=total_tokens,
                    swa_workspace=swa_workspace,
                )

            # allocate the equal amount of workspace for
            # each chunk prefill request
            max_context_chunk = _CP_TOKENS_PER_ITER_ROCM // num_extends
            from vllm.utils.math_utils import cdiv

            num_chunks = cdiv(computed_kv_lens.max().item(), max_context_chunk)

            chunk_starts = (
                torch.arange(num_chunks, dtype=torch.int32)
                .unsqueeze(1)
                .expand(-1, num_extends)
                * max_context_chunk
            )
            chunk_ends = torch.min(
                computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
            )
            chunk_seq_lens = (chunk_ends - chunk_starts).clamp(
                min=0
            )  # [num_chunks, num_extends]
            cu_seq_lens_cpu = torch.zeros(
                [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
            )
            torch.cumsum(
                chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
            )
            max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()

            # Build token->batch mapping robustly, even with zero-length batches.
            token_to_batch_tensor = torch.zeros(
                (num_chunks, max_cum_tokens), dtype=torch.int32, pin_memory=True
            )
            batch_ids = torch.arange(num_extends, dtype=torch.int32)
            for chunk_idx in range(num_chunks):
                total_tokens = cu_seq_lens_cpu[chunk_idx, -1].item()
                if total_tokens == 0:
                    continue
                token_to_batch = torch.repeat_interleave(
                    batch_ids, chunk_seq_lens[chunk_idx].to(torch.int64)
                )
                token_to_batch_tensor[chunk_idx, :total_tokens] = token_to_batch

            chunk_context_metadata = AiterChunkContextMetadata(
                workspace=self.extend_workspace,
                cu_seq_lens_chunk=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                chunk_starts=chunk_starts.to(self.device, non_blocking=True),
                seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                seq_lens=chunk_seq_lens,
                token_to_batch=token_to_batch_tensor.to(self.device, non_blocking=True),
                num_chunks=num_chunks,
                total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
                swa_metadata=swa_metadata,
            )

            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes : num_decodes + num_extends + 1
            ]
            seq_lens_device = common_attn_metadata.seq_lens[num_extends_slice]
            cu_seq_lens = torch.zeros(
                num_extends + 1, dtype=torch.int32, device=seq_lens_device.device
            )
            torch.cumsum(
                seq_lens_device, dim=0, dtype=cu_seq_lens.dtype, out=cu_seq_lens[1:]
            )
            extend_metadata = AiterFlashAttentionChunkPrefillMetadata(
                max_query_len=query_lens_for_extend.max().item(),
                min_query_len=query_lens_for_extend.min().item(),
                max_seq_len=seq_lens[num_extends_slice].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
                chunk_context_metadata=chunk_context_metadata,
            )

        prefill_metadata = None
        if num_prefills > 0:
            query_lens_for_prefill = query_lens_cpu[num_decodes + num_extends :]
            query_start_loc_device = common_attn_metadata.query_start_loc[
                num_decodes + num_extends :
            ]
            prefill_metadata = AiterFlashAttentionPrefillMetadata(
                max_query_len=query_lens_for_prefill.max().item(),
                min_query_len=query_lens_for_prefill.min().item(),
                max_seq_len=seq_lens[num_decodes + num_extends :].max().item(),
                query_start_loc=query_start_loc_device - query_start_loc_device[0],
            )

        num_actual_kv_tokens = torch.sum(seq_lens).item()

        use_cascade = False

        context_batch_size = 0
        has_prefill = bool(num_prefills > 0 or num_extends > 0)
        if has_prefill:
            context_batch_size = num_prefills + num_extends
        else:
            context_batch_size = num_decodes
        context_graph_bs = context_batch_size

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        context = Context(
            positions=None,
            is_prefill=has_prefill,
            batch_size=context_batch_size,
            graph_bs=context_graph_bs,
        )

        attn_metadata_for_plugin_mode = AiterFlashAttentionMetadataForPluginMode(
            num_actual_tokens=num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_extends=num_extends,
            num_extend_tokens=num_extend_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
            extend_metadata=extend_metadata,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            total_tokens=self.total_tokens,
            context=context,
        )

        attn_metadata = AttentionMetaData(
            max_seqlen_q=common_attn_metadata.max_query_len,
            block_tables=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            plugin_metadata=attn_metadata_for_plugin_mode,
        )

        return attn_metadata

    # this method will be called by vllm, so it follows the vllm's interface convention
    def build_for_cudagraph_capture(
        self,
        common_attn_metadata=None,
    ):
        self.total_tokens = (
            self.model_config.max_model_len
            * self.vllm_config.scheduler_config.max_num_partial_prefills
        )
        attn_metadata = self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )
        self.total_tokens = 0
        return attn_metadata


def AiterAttentionMetadataBuilderDecoratorForPluginMode(default_base_class):
    def decorator(cls):
        is_vllm_mode = is_vllm()
        is_sglang_mode = is_sglang()

        base_class = default_base_class
        class_dict = {}

        # record original decorated cls methods
        for key, value in cls.__dict__.items():
            if not key.startswith("__") or key in (
                "__annotations__",
                "__init__",
                "__module__",
                "__qualname__",
                "__doc__",
            ):
                class_dict[key] = value

        # handle the generic base class
        needs_generic = False
        generic_base = None

        if is_vllm_mode:
            # get the base class and generic base class
            base_class, generic_base, needs_generic, class_dict = (
                setup_attn_metadata_builder_base_class_and_attributes(class_dict)
            )

            # replace the __init__ method in the decorated class
            class_dict["__init__"] = create_attn_metadata_builder_init_method(
                base_class
            )

            # add the methods to the decorated class
            for method_name in dir(vllmAttentionMetadataBuilderMethods):
                if not method_name.startswith("_"):
                    method = getattr(vllmAttentionMetadataBuilderMethods, method_name)
                    if callable(method):
                        class_dict[method_name] = method
        elif is_sglang_mode:
            raise NotImplementedError(
                "AttentionMetadataBuilder for sglang is not implemented yet"
            )

        # create the new class
        new_class = type(cls.__name__, (base_class,), class_dict)

        # replace the inherit base class for plugin mode, meanwhile support generic base class
        is_generic_builder_base = (
            isinstance(generic_base, type)
            and issubclass(generic_base, Generic)
            and len(getattr(generic_base, "__parameters__", ())) > 0
        )
        if needs_generic and is_generic_builder_base:
            new_class.__orig_bases__ = (generic_base[AttentionMetaData],)
        else:
            new_class.__orig_bases__ = (generic_base,)

        return new_class

    return decorator


# for MLA attention metadata for plugin mode
@dataclass
class AiterMLACommonDecodeMetadataForPluginMode:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None


@dataclass
class AiterMLADecodeMetadataForPluginMode(AiterMLACommonDecodeMetadataForPluginMode):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None


@dataclass
class AiterMLACommonPrefillMetadataForPluginMode:
    """Prefill Specific Metadata"""

    @dataclass
    class AiterMLAChunkedContextMetadataForPluginMode:
        # New for MLA (compared to FlashAttention)
        # For handling chunked prefill
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        workspace: torch.Tensor
        token_to_seq: torch.Tensor
        chunk_total_token: list[int]

        # for mla DCP
        padded_local_chunk_seq_lens: list[list[int]] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        padded_local_cu_seq_lens: torch.Tensor | None = None
        cu_seq_lens_lst: list[list[int]] | None = None
        chunk_size: int | None = None

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: AiterMLAChunkedContextMetadataForPluginMode | None = None
    query_seq_lens: torch.Tensor | None = None
    workspace_buffer: torch.Tensor | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None


D = TypeVar("D", bound=AiterMLACommonDecodeMetadataForPluginMode)


@dataclass
class AiterMLACommonMetadataForPluginMode(Generic[D]):
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None

    decode: D | None = None
    prefill: AiterMLACommonPrefillMetadataForPluginMode | None = None

    def __post_init__(self):
        pass
        # if self.head_dim is not None and not MLACommonBackend.supports_head_size(
        #     self.head_dim
        # ):
        #     raise ValueError(f"Head dimension {self.head_dim} is not supported by MLA.")


class vllmMLAAttentionMetadataBuilderMethods:
    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    # TODO: support mtp and sparse
    def _set_mla_persistent_worker_buffers(
        self, bs: int, cu_seqlens_q: torch.Tensor, max_q_len: int = 1
    ):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": max_q_len,
            "uni_seqlen_qo": max_q_len,
            "fast_mode": 1,
            "max_split_per_batch": 16,
        }
        var = self.mla_persistent_metadata
        work_meta_data = var["work_meta_data"]
        work_info_set = var["work_info_set"]
        work_indptr = var["work_indptr"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]
        get_mla_metadata_v1(
            cu_seqlens_q,
            self.paged_kv_indptr[: bs + 1],  # TODO: support sparse
            self.paged_kv_last_page_len[:bs],
            self.num_attention_heads,
            1,  # nhead_kv,
            True,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=self.block_size,
            **split_params,
        )
        return {
            "work_meta_data": work_meta_data,
            "work_info_set": work_info_set,
            "work_indptr": work_indptr,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ):
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        mask = torch.arange(
            block_table_tensor.size(1),
            dtype=block_table_tensor.dtype,
            device=device,
        ).unsqueeze(0) < seq_lens_device.unsqueeze(1)
        paged_kv_indices = block_table_tensor[mask]

        # kernel block size is always 1, so each page has exactly 1 token.
        # last_page_len is always 1 - just slice the pre-initialized buffer.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        num_actual_pages = paged_kv_indices.size(0)

        self.paged_kv_indices[:num_actual_pages].copy_(
            paged_kv_indices, non_blocking=True
        )
        self.paged_kv_indices[num_actual_pages:].fill_(-1)
        paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

        self.paged_kv_indptr[: 1 + num_reqs].copy_(paged_kv_indptr, non_blocking=True)
        self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
        paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

        # paged_kv_last_page_len already uses the pre-initialized buffer slice
        # (set above), so no copy needed - buffer is always 1s.

        self.qo_indptr[: 1 + num_reqs].copy_(query_start_loc_device, non_blocking=True)
        self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
        qo_indptr = self.qo_indptr[: 1 + num_reqs]

        ctx_mla_ps = self._set_mla_persistent_worker_buffers(
            num_reqs, query_start_loc_device, 1
        )
        self.mla_persistent_metadata.update(ctx_mla_ps)

        attn_metadata = AiterMLADecodeMetadataForPluginMode(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
        )

        return attn_metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata=None,
    ):
        return self.build(0, common_attn_metadata)

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata=None,
        fast_build: bool = False,
    ):

        from vllm.v1.attention.backends.utils import split_decodes_and_prefills
        from vllm.model_executor.layers.attention.mla_attention import (
            QueryLenSupport,
        )

        from vllm.utils.math_utils import cdiv, round_down
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            num_computed_tokens_cpu = (
                common_attn_metadata.compute_num_computed_tokens().cpu()
            )

            reqs_start = num_decodes  # prefill_start

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )

            chunked_context_metadata = None
            if max_context_len_cpu > 0:
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                max_context_chunk = (
                    self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
                )

                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_and_maybe_dequant_cache` kernel
                    # cannot handle `context_chunk_starts` that are not aligned
                    # to page_size
                    max_context_chunk = round_down(max_context_chunk, self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like
                #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = (
                    torch.arange(num_chunks, dtype=torch.int32)
                    .unsqueeze(1)
                    .expand(-1, num_prefills)
                    * max_context_chunk
                )
                chunk_ends = torch.min(
                    context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
                )
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(
                    num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                )
                torch.cumsum(
                    chunk_seq_lens,
                    dim=1,
                    out=cu_seq_lens_cpu[:, 1:],
                    dtype=torch.int32,
                )
                chunk_total_token = cu_seq_lens_cpu[:, -1]

                max_token_num_over_chunk = chunk_total_token.max().item()
                token_to_seq_tensor_cpu = torch.zeros(
                    [num_chunks, max_token_num_over_chunk], dtype=torch.int32
                )
                range_idx = torch.arange(num_prefills, dtype=torch.int32)
                for i in range(num_chunks):
                    chunk_token_to_seq_tensor = torch.repeat_interleave(
                        range_idx, chunk_seq_lens[i]
                    )
                    chunk_len = chunk_token_to_seq_tensor.shape[0]
                    token_to_seq_tensor_cpu[i, :chunk_len] = chunk_token_to_seq_tensor

                if self.dcp_world_size > 1:
                    local_context_lens_allranks = get_dcp_local_seq_lens(
                        context_lens_cpu,
                        self.dcp_world_size,
                        None,
                        self.dcp_local_block_size,
                    )
                    # Note(qcs): The max local context lengths
                    # padded to `dcp_local_block_size`.
                    padded_local_context_lens_cpu: torch.Tensor = (
                        cdiv(
                            context_lens_cpu,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    # Note(hc): The above max_context_chunk already enforces
                    # block_size alignment, DCP just need the block_size can
                    # be divisible by dcp_world_size, because DCP use
                    # cp_gather_cache which not require `cp_chunk_starts`
                    # aligned to page_size.
                    assert max_context_chunk % self.dcp_world_size == 0
                    padded_local_max_context_chunk_across_ranks = (
                        cdiv(
                            max_context_chunk,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    local_chunk_starts = (
                        torch.arange(num_chunks, dtype=torch.int32)
                        .unsqueeze(1)
                        .expand(-1, num_prefills)
                        * padded_local_max_context_chunk_across_ranks
                    )
                    local_chunk_ends = torch.min(
                        padded_local_context_lens_cpu.unsqueeze(0),
                        local_chunk_starts
                        + padded_local_max_context_chunk_across_ranks,
                    )
                    padded_local_chunk_seq_lens = (
                        local_chunk_ends - local_chunk_starts
                    ).clamp(min=0)

                    padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
                        num_chunks,
                        num_prefills + 1,
                        dtype=torch.int32,
                        pin_memory=True,
                    )
                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                chunked_context_metadata_cls = (
                    AiterMLACommonPrefillMetadataForPluginMode.AiterMLAChunkedContextMetadataForPluginMode
                )
                if self.dcp_world_size > 1:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=local_chunk_starts.to(device, non_blocking=True),
                        seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token.tolist(),
                        workspace=self.chunked_prefill_workspace,
                        padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
                        local_context_lens_allranks=local_context_lens_allranks.tolist(),
                        padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                            device, non_blocking=True
                        ),
                        cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
                        chunk_size=padded_local_max_context_chunk_across_ranks,
                    )
                else:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=chunk_starts.to(device, non_blocking=True),
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token,
                        workspace=self.chunked_prefill_workspace,
                    )

                if self._use_cudnn_prefill:
                    chunked_context_metadata.seq_lens = chunk_seq_lens

                assert (
                    max(chunked_context_metadata.max_seq_lens)
                    <= self.chunked_prefill_workspace_size
                )

            prefill_metadata = AiterMLACommonPrefillMetadataForPluginMode(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
            )

        decode_metadata = None
        if num_decodes > 0:
            dcp_tot_seq_lens_device = None
            if self.dcp_world_size > 1:
                dcp_tot_seq_lens_device = seq_lens[:num_decodes]
                seq_lens = dcp_local_seq_lens

                # After DCP distribution, the maximum number of tokens for any rank is
                # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
                # and I is cp_kv_cache_interleave_size.
                # This eliminates GPU->CPU sync while minimizing workspace
                # over-allocation.
                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_seq_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
            )

        attn_metadata_for_plugin_mode = AiterMLACommonMetadataForPluginMode(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # TODO: support mtp
        ctx_mla_ps = self.mla_persistent_metadata

        attn_metadata = AttentionMetaData(
            max_seqlen_q=common_attn_metadata.max_query_len,
            block_tables=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            plugin_metadata=attn_metadata_for_plugin_mode,
            **ctx_mla_ps,
        )
        return attn_metadata


def create_mla_attn_metadata_builder_init_method(base_class):
    """
    Create the init method for metadata builder
    """

    def init_method_under_plugin_mode(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        base_class.__init__(self, kv_cache_spec, layer_names, config, device)
        logger.info("init AiterAttentionMetadataBuilder for plugin mode")
        from vllm.config import VllmConfig

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config

        self.compilation_config = self.vllm_config.compilation_config
        self.decode_attn_out_dtype = self.vllm_config.model_config.dtype
        # kernel block size is always 1.
        max_num_pages_per_req = self.vllm_config.model_config.max_model_len
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        self.num_attention_heads = (
            config.model_config.hf_config.num_attention_heads
            // get_tp_group().world_size
        )
        self.block_size = kv_cache_spec.block_size
        self.max_bs = max_num_reqs

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (kernel block size is always 1),
        # so we create it once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        self.paged_kv_indptr = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        self.qo_indptr = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            1,
            self.num_attention_heads,
            torch.bfloat16,
            dtypes.d_dtypes[config.cache_config.cache_dtype],
            is_sparse=False,  # TODO: support sparse
            fast_mode=True,
        )

        self.mla_persistent_metadata = {
            "work_meta_data": torch.empty(
                work_meta_data_size, dtype=work_meta_data_type, device=self.device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_type, device=self.device
            ),
            "work_info_set": torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=self.device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            ),
            "reduce_final_map": torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            ),
        }

    return init_method_under_plugin_mode


def setup_mla_attn_metadata_builder_base_class_and_attributes(class_dict: dict):
    """
    Setup the base class and attributes for attention metadata builder
    """
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
        QueryLenSupport,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    base_class = MLACommonMetadataBuilder
    generic_base = MLACommonMetadataBuilder
    needs_generic = True

    # align with vllm rocm aiter fa
    class_dict["_cudagraph_support"] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    class_dict["reorder_batch_threshold"] = 1
    class_dict["query_len_support"] = QueryLenSupport.UNIFORM

    return base_class, generic_base, needs_generic, class_dict


def AiterMLAAttentionMetadataBuilderDecoratorForPluginMode(default_base_class):
    def decorator(cls):
        is_vllm_mode = is_vllm()
        is_sglang_mode = is_sglang()

        base_class = default_base_class
        class_dict = {}

        # record original decorated cls methods
        for key, value in cls.__dict__.items():
            if not key.startswith("__") or key in (
                "__annotations__",
                "__init__",
                "__module__",
                "__qualname__",
                "__doc__",
            ):
                class_dict[key] = value

        # handle the generic base class
        needs_generic = False
        generic_base = None

        if is_vllm_mode:
            # get the base class and generic base class
            base_class, generic_base, needs_generic, class_dict = (
                setup_mla_attn_metadata_builder_base_class_and_attributes(class_dict)
            )

            # replace the __init__ method in the decorated class
            class_dict["__init__"] = create_mla_attn_metadata_builder_init_method(
                base_class
            )

            # add the methods to the decorated class
            for method_name in dir(vllmMLAAttentionMetadataBuilderMethods):
                if not method_name.startswith("__"):
                    method = getattr(
                        vllmMLAAttentionMetadataBuilderMethods, method_name
                    )
                    if callable(method):
                        class_dict[method_name] = method
        elif is_sglang_mode:
            raise NotImplementedError(
                "AttentionMetadataBuilder for sglang is not implemented yet"
            )

        # create the new class
        new_class = type(cls.__name__, (base_class,), class_dict)

        # replace the inherit base class for plugin mode, meanwhile support generic base class
        if needs_generic and generic_base is not None:
            new_class.__orig_bases__ = (generic_base[new_class],)

        return new_class

    return decorator


class vllmAiterMLABackendMethods:
    accept_output_buffer: bool = True
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        # (num_blocks, num_layers, block_size, head_size)
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)


def AiterBackendDecoratorForPluginMode(cls):
    """
    Decorator for AiterBackend to add specific methods and attributes for plugin mode
    """
    is_vllm_mode = is_vllm()
    if is_vllm_mode:
        if not issubclass(cls.get_impl_cls(), MLAAttention):
            methods_cls = vllmAiterAttentionBackendMethods
        else:
            methods_cls = vllmAiterMLABackendMethods
        for name in dir(methods_cls):
            if name.startswith("_"):
                continue
            setattr(cls, name, getattr(methods_cls, name))
    return cls


# here not register it as a custom op and mark split because vllm
# will register attention impl forward as a custom op, so here
# avoid duplicated registration, and the split op is registered
# into the atom support_torch_compile decorator
def unified_attention_with_output_base_for_plugin_mode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    if use_mla:
        # raise NotImplementedError("MLA is not supported for plugin mode for now")
        kv_c_normed = k
        k_pe = v
        self = atom_config.compilation_config.static_forward_context[layer_name]
        q = self.q_proj(q, q_scale)
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        if disable_vllm_plugin_attention:
            k_pe = k_pe.unsqueeze(1)
            if self.rotary_emb is not None:
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )
        # positions written at model entry (model_wrapper.forward)
        output = self.attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.num_heads * self.v_head_dim),
        )
        return self.o_proj(output)
    else:
        self = atom_config.compilation_config.static_forward_context[layer_name]
        # here is the standard vllm attention impl interface
        # when using fusion, we need to pass the qkv and positions through the q,k,v
        # [watch out] accept_output_buffer must be False for plugin mode
        # because we don't want vllm to manipulate the q k v and output buffer
        # ATOM needs to handle all of the buffer here
        if envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            output = self.attn(q, positions, qkv)
        else:
            # calculate the q and k with rotary embedding
            if self.rotary_emb is not None:
                q, k = self.rotary_emb(positions, q, k)
            output = self.attn(q, k, v)
        return output
