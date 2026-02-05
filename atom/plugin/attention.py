from typing import Optional
import logging

from dataclasses import dataclass

import torch

from atom.plugin.prepare import is_vllm, is_sglang
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import Context, AttentionMetaData
from atom.model_ops.attention_mha import PagedAttentionImpl

logger = logging.getLogger("atom")

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024


@dataclass
class AiterFlashAttentionDecodeMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


@dataclass
class AiterFlashAttentionPrefillMetadata:
    max_query_len: int
    min_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


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
class MetadataForPluginMode:
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


class vllmAiterBackendMethods:
    # here attention in ATOM doesn't accept the output buffer because
    # ATOM works as a model impl backend, it needs the maximum freedom
    # to decide the output buffer and shape, thus here use this flag to
    # let vllm don't allocate the output buffer for ATOM. ATOM will
    # handle the output buffer by itself
    accept_output_buffer: bool = False
    supported_dtypes: list = [torch.float16, torch.bfloat16]

    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    @staticmethod
    def get_supported_kernel_block_sizes():
        from vllm.v1.attention.backend import MultipleOf

        return [MultipleOf(16)]

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


def AiterBackendDecoratorForPluginMode(cls):
    """
    Decorator for AiterBackend to add specific methods and attributes for plugin mode
    """
    is_vllm_mode = is_vllm()

    if is_vllm_mode:
        # for vllm, add the required methods
        cls.full_cls_name = vllmAiterBackendMethods.full_cls_name
        cls.accept_output_buffer = vllmAiterBackendMethods.accept_output_buffer
        cls.supported_dtypes = vllmAiterBackendMethods.supported_dtypes
        cls.get_supported_kernel_block_sizes = (
            vllmAiterBackendMethods.get_supported_kernel_block_sizes
        )
        cls.get_kv_cache_shape = vllmAiterBackendMethods.get_kv_cache_shape
        cls.is_mla = vllmAiterBackendMethods.is_mla
        cls.get_required_kv_cache_layout = (
            vllmAiterBackendMethods.get_required_kv_cache_layout
        )
        cls.get_supported_head_sizes = vllmAiterBackendMethods.get_supported_head_sizes
        cls.supports_alibi_sqrt = vllmAiterBackendMethods.supports_alibi_sqrt
    return cls


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
        from vllm.attention.layer import Attention

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
            sliding_window_sizes.add((layer.impl.sliding_window - 1, 0))

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

        # used to store the positions of each tokens of each request
        # for computing ROPE
        positions = []

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AiterFlashAttentionDecodeMetadata(
                max_query_len=query_lens_cpu[:num_decodes].max().item(),
                min_query_len=query_lens_cpu[:num_decodes].min().item(),
                max_seq_len=seq_lens[:num_decodes].max().item(),
                query_start_loc=common_attn_metadata.query_start_loc[: num_decodes + 1],
            )
            for seq_len in seq_lens[:num_decodes]:
                positions.append(seq_len - 1)

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

            for idx in range(num_extends):
                extend_start_seq_len = (
                    seq_lens_for_extend[idx] - query_lens_for_extend[idx]
                )
                extend_end_seq_len = seq_lens_for_extend[idx]
                for pos in range(extend_start_seq_len, extend_end_seq_len):
                    positions.append(pos)

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
            for prefill_seq_len in seq_lens[num_decodes + num_extends :]:
                for pos in range(prefill_seq_len):
                    positions.append(pos)

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
        self.positions.np[:num_actual_tokens] = positions
        context = Context(
            positions=self.positions.copy_to_gpu(num_actual_tokens),
            is_prefill=has_prefill,
            batch_size=context_batch_size,
            graph_bs=context_graph_bs,
        )

        attn_metadata_for_plugin_mode = MetadataForPluginMode(
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
        if needs_generic and generic_base is not None:
            new_class.__orig_bases__ = (generic_base[new_class],)

        return new_class

    return decorator


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
    from atom.config import get_current_atom_config
    from atom.utils import envs

    atom_config = get_current_atom_config()
    if use_mla:
        raise NotImplementedError("MLA is not supported for plugin mode for now")
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
            output = self.attn(q, k, v)
        return output
