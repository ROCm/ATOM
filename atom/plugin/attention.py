from typing import Type
import logging

import torch

from atom.plugin.prepare import is_vllm, is_sglang
from atom.utils import CpuGpuBuffer
from atom.model_ops.attentions.backends import AttentionMetadataBuilder
from atom.utils.forward_context import AttentionMetaData, Context

logger = logging.getLogger("atom")

class vllmAiterBackendMethods:
    # here attention in ATOM doesn't accept the output buffer because
    # ATOM works as a model impl backend, it needs the maximum freedom
    # to decide the output buffer and shape, thus here use this flag to
    # let vllm don't allocate the output buffer for ATOM
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


def AiterBackendDecoratorForPluginMode(cls):
    '''
    Decorator for AiterBackend to add specific methods and attributes for plugin mode
    '''
    is_vllm_mode = is_vllm()

    if is_vllm_mode:
        # for vllm, add the required methods
        cls.full_cls_name = vllmAiterBackendMethods.full_cls_name
        cls.accept_output_buffer = vllmAiterBackendMethods.accept_output_buffer
        cls.supported_dtypes = vllmAiterBackendMethods.supported_dtypes
        cls.get_supported_kernel_block_sizes = vllmAiterBackendMethods.get_supported_kernel_block_sizes
        cls.get_kv_cache_shape = vllmAiterBackendMethods.get_kv_cache_shape
        cls.is_mla = vllmAiterBackendMethods.is_mla
        cls.get_required_kv_cache_layout = vllmAiterBackendMethods.get_required_kv_cache_layout
        cls.get_supported_head_sizes = vllmAiterBackendMethods.get_supported_head_sizes
    return cls


def create_attn_metadata_builder_init_method(base_class: Type[AttentionMetadataBuilder]):
    '''
    Create the init method for metadata builder
    '''
    def init_method_under_plugin_mode(self,
                                      kv_cache_spec=None,
                                      layer_names=None,
                                      config=None,
                                      device=None,
                                      model_runner=None):
        base_class.__init__(self,
                            kv_cache_spec,
                            layer_names,
                            config,
                            device)

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.scheduler_config = config.scheduler_config
        self.total_tokens = 0
        self.block_size = config.cache_config.block_size
        self.block_ratio = 1
        self.has_sliding_window = hasattr(config.model_config.hf_config, "sliding_window")
        
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
        # max_bs = config.scheduler_config.max_num_seqs

        i64_kwargs = {"dtype": torch.int64, "device": device}
        # i32_kwargs = {"dtype": torch.int32, "device": device}

        # self.slot_mapping = CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs)
        # self.context_lens = CpuGpuBuffer(max_num_batched_tokens, **i32_kwargs)
        self.positions = CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs)
        # max_num_blocks_per_seq = (self.scheduler_config.max_model_len + self.block_size - 1) // self.block_size
        # self.block_tables = CpuGpuBuffer(
        #     max_bs,
        #     max_num_blocks_per_seq // self.block_ratio,
        #     **i32_kwargs
        # )
        # self.context_lens = CpuGpuBuffer(max_bs, **i32_kwargs)
        # self.cu_seqlens_q = CpuGpuBuffer(max_bs + 1, **i32_kwargs)
        # self.cu_seqlens_k = CpuGpuBuffer(max_bs + 1, **i32_kwargs)

        # Create block_tables_converted if block_ratio > 1
        # if self.block_ratio > 1:
        #     self.block_tables_converted = CpuGpuBuffer(
        #         max_bs,
        #         max_num_blocks_per_seq,
        #         **i32_kwargs
        #     )

        # self.cu_seqlens_q.cpu.copy_(torch.arange(0, max_bs + 1, step=1, dtype=torch.int32))
        # self.cu_seqlens_q.copy_to_gpu()

    return init_method_under_plugin_mode


def setup_attn_metadata_builder_base_class_and_attributes(class_dict: dict):
    '''
    Setup the base class and attributes for attention metadata builder
    '''
    from vllm.v1.attention.backends.utils import (
        AttentionCGSupport,
        AttentionMetadataBuilder,
    )

    base_class = AttentionMetadataBuilder
    generic_base = AttentionMetadataBuilder
    needs_generic = True

    class_dict['_cudagraph_support'] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    class_dict['reorder_batch_threshold'] = 1

    return base_class, generic_base, needs_generic, class_dict


class vllmAttentionMetadataBuilderMethods:
    def __init__(self):
        raise TypeError(
            f"{self.__class__.__name__} is a utility class and should not be instantiated. "
            "Its methods are meant to be added to other classes via decorators."
        )

    def build_decode(
        self,
        common_prefix_len: int,
        common_attn_metadata = None,
        fast_build: bool = False
    ):
        scheduled_bs = common_attn_metadata.num_reqs
        # TODO: for we find the bs and scheduled_bs are equal
        bs = scheduled_bs
        # print(f"[zejun] ATOM build_decode - scheduled_bs: {scheduled_bs}", flush=True)
        # print(f"[zejun] ATOM build_decode - bs: {bs}", flush=True)
        dropout_p = 0.0
        max_q_len = 1
        min_seqlen_q = 0

        context_lens = common_attn_metadata.seq_lens_cpu.tolist()
        max_seqlen_k = max(context_lens)
        positions = [i - 1 for i in context_lens]

        # block_size = self.block_size
        # last_block_num_tokens = []
        # block_tables = []
        # for i in range(scheduled_bs):
        #     seqlen = common_attn_metadata.seq_lens_cpu[i]
        #     last_block_num_tokens.append(seqlen % block_size)

        #     block_table = common_attn_metadata.block_table_tensor[i]
        #     nonzero_indices = torch.nonzero(block_table, as_tuple=False).squeeze()
        #     last_nonzero_idx = nonzero_indices.max().item()
        #     block_table = block_table[:last_nonzero_idx + 1].cpu().tolist()
            # block_tables.append(block_table)

        # slot_mapping = [
        #     block_table[-1] * block_size + last_block_num - 1
        #     for block_table, last_block_num in zip(
        #         block_tables, last_block_num_tokens
        #     )
        # ]
        # slot_mapping.extend([-1] * (bs - scheduled_bs))

        # Prepare block_tables: write to CPU buffer first
        # self.block_tables.np.fill(0)
        # for i, block_table in enumerate(block_tables):
        #     self.block_tables.np[i, : len(block_table)] = block_table

        # for decode, sum scheduled tokens is equal to scheduled_bs
        sum_scheduled_tokens = scheduled_bs

        # Write data to CPU buffers (equivalent to var["..."].np[:...] = ...)
        # self.slot_mapping.np[:bs] = slot_mapping
        self.positions.np[:sum_scheduled_tokens] = positions
        # self.context_lens.np[:scheduled_bs] = context_lens
        # self.context_lens.np[scheduled_bs:bs] = 0

        # Prepare vars_used list (what needs to be copied to GPU)
        vars_used = [
            # ("slot_mapping", bs),
            # ("context_lens", bs),
            # ("block_tables", bs),
            # ("cu_seqlens_q", bs + 1),
        ]

        # Copy to GPU (equivalent to ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used})
        ctx = {}
        for var_name, num in vars_used:
            buffer = getattr(self, var_name)
            result = buffer.copy_to_gpu(num)
            # print('[zejun] ATOM build_decode - var_name: ', var_name, flush=True)
            # print('[zejun] ATOM build_decode - num: ', num, flush=True)
            # print('[zejun] ATOM build_decode - buffer: ', buffer, flush=True)
            # print('[zejun] ATOM build_decode - buffer.cpu: ', buffer.cpu, flush=True)
            # print('[zejun] ATOM build_decode - buffer.gpu: ', buffer.gpu, flush=True)
            # print('[zejun] ATOM build_decode - result: ', result, flush=True)
            # if var_name == "cu_seqlens_q":
            #     print(f"[zejun] ATOM build_decode - cu_seqlens_q debug:", flush=True)
            #     print(f"  buffer.cpu[:num]: {buffer.cpu[:num]}", flush=True)
            #     print(f"  result (gpu tensor): {result}", flush=True)
            #     print(f"  result is None: {result is None}", flush=True)
            ctx[var_name] = result

        # Get positions from GPU and create Context
        positions_gpu = self.positions.copy_to_gpu(sum_scheduled_tokens)

        # Build AttentionMetaData with ctx dict (unpacked as kwargs) and Context
        attn_metadata = AttentionMetaData(
            # TODO: handle the situation that prefill mixed with decode
            cu_seqlens_q=common_attn_metadata.query_start_loc,
            # TODO: add back to handle the mixed scenario
            # cu_seqlens_k=common_attn_metadata.query_start_loc,
            max_q_len=max_q_len,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=common_attn_metadata.slot_mapping,
            context_lens=common_attn_metadata.seq_lens, # here use cuda tensor
            dropout_p=dropout_p,
            block_tables=common_attn_metadata.block_table_tensor,
            context=Context(
                positions=positions_gpu,
                is_prefill=False,
                batch_size=bs,
                graph_bs=bs
            ),
            **ctx,
        )
        # print(f"[zejun] ATOM build_decode - AttentionMetaData members:", flush=True)
        # print(f"  cu_seqlens_q: {attn_metadata.cu_seqlens_q}", flush=True)
        # print(f"  cu_seqlens_k: {attn_metadata.cu_seqlens_k}", flush=True)
        # print(f"  max_seqlen_q: {attn_metadata.max_seqlen_q}", flush=True)
        # print(f"  max_seqlen_k: {attn_metadata.max_seqlen_k}", flush=True)
        # print(f"  min_seqlen_q: {attn_metadata.min_seqlen_q}", flush=True)
        # print(f"  slot_mapping: {attn_metadata.slot_mapping}", flush=True)
        # print(f"  context_lens: {attn_metadata.context_lens}", flush=True)
        # if attn_metadata.block_tables is not None:
        #     print(f"  block_tables shape : {attn_metadata.block_tables.shape}", flush=True)
        # print(f"  dropout_p: {attn_metadata.dropout_p}", flush=True)
        # print(f"  max_q_len: {attn_metadata.max_q_len}", flush=True)
        # print(f"  kv_indptr: {attn_metadata.kv_indptr}", flush=True)
        # print(f"  kv_indices: {attn_metadata.kv_indices}", flush=True)
        # print(f"  kv_last_page_lens: {attn_metadata.kv_last_page_lens}", flush=True)
        # print(f"  cu_seqlen_ks: {attn_metadata.cu_seqlen_ks}", flush=True)
        # print(f"  cu_seqlen_ke: {attn_metadata.cu_seqlen_ke}", flush=True)
        # print(f"  sparse_kv_indptr: {attn_metadata.sparse_kv_indptr}", flush=True)
        # print(f"  work_meta_data: {attn_metadata.work_meta_data}", flush=True)
        # print(f"  work_indptr: {attn_metadata.work_indptr}", flush=True)
        # print(f"  work_info_set: {attn_metadata.work_info_set}", flush=True)
        # print(f"  reduce_indptr: {attn_metadata.reduce_indptr}", flush=True)
        # print(f"  reduce_final_map: {attn_metadata.reduce_final_map}", flush=True)
        # print(f"  reduce_partial_map: {attn_metadata.reduce_partial_map}", flush=True)
        # print(f"  context: {attn_metadata.context}", flush=True)
        # print(f"  block_tables_converted: {attn_metadata.block_tables_converted}", flush=True)
        # print(f"  kv_indices_converted: {attn_metadata.kv_indices_converted}", flush=True)

        return attn_metadata

    def build_prefill(
        self,
        common_prefix_len: int,
        common_attn_metadata = None,
        fast_build: bool = False
    ):
        # print('[zejun] ATOM ATOMAttentionMetadataBuilder build_prefill', flush=True)
        bs = common_attn_metadata.num_reqs
        sum_scheduled_tokens = common_attn_metadata.num_actual_tokens
        block_tables = common_attn_metadata.block_table_tensor
        # var = self.model_runner.forward_vars
        positions = []
        # cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        block_size = self.block_size
        # for vllm, the slot mapping is passed in common_attn_metadata
        # it is determined by vllm
        # slot_mapping = []
        # seqs = list(batch.seqs.values())
        # seqs = seqs[:bs]
        # TODO: remove later
        for i in range(bs):
            seqlen = common_attn_metadata.seq_lens_cpu[i].item()
            cached_seqlen = common_attn_metadata.num_computed_tokens_cpu[i].item()
            positions.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            # cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            num_blocks = (
                seqlen + block_size - 1
            ) // block_size
            # print('[zejun] ATOM(plugin) num_blocks = ', num_blocks, flush=True)
            num_cached_blocks = (
                cached_seqlen + block_size - 1
            ) // block_size
            # print('[zejun] ATOM(plugin) num_cached_blocks = ', num_cached_blocks, flush=True)
            # last_block_tokens = batch.last_block_num_tokens[i]
            last_block_tokens = seqlen % block_size
            # print('[zejun] ATOM(plugin) last_block_tokens = ', last_block_tokens, flush=True)

            # block_table = batch.block_tables[i]
            block_table = common_attn_metadata.block_table_tensor[i]
            nonzero_indices = torch.nonzero(block_table, as_tuple=False).squeeze()
            last_nonzero_idx = nonzero_indices.max().item()
            block_table = block_table[:last_nonzero_idx + 1].cpu().tolist()
            # print('[zejun] ATOM(plugin) block_table = ', block_table, flush=True)

            for j in range(num_cached_blocks, num_blocks):
                start = block_table[j] * block_size
                # print('[zejun] ATOM(plugin)[', j, '] start = ', start, flush=True)
                if j != num_blocks - 1:
                    end = start + block_size
                else:
                    end = start + last_block_tokens
                # print('[zejun] ATOM(plugin)[', j, '] end = ', end, flush=True)
                # slot_mapping.extend(list(range(start, end)))
                # print('[zejun] ATOM(plugin)[', j, '] slot_mapping = ', slot_mapping, flush=True)

        # Write data to CPU buffers
        self.positions.np[:sum_scheduled_tokens] = positions
        # self.slot_mapping.np[:len(slot_mapping)] = slot_mapping

        # Build cu_seqlens_q and cu_seqlens_k tensors
        # self.cu_seqlens_q.np[:bs + 1] = cu_seqlens_q
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)

        # Write context_lens to CPU buffer
        # context_lens = common_attn_metadata.seq_lens_cpu.tolist()
        # self.context_lens.np[:bs] = context_lens[:bs]

        min_seqlen_q = 0
        dropout_p = 0.0

        # Prepare vars_used list (what needs to be copied to GPU)
        vars_used = [
            # ("cu_seqlens_q", bs + 1),
            # ("slot_mapping", len(slot_mapping)),
            # ("context_lens", bs),
        ]

        # Copy to GPU (equivalent to ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used})
        ctx = {}
        for var_name, num in vars_used:
            buffer = getattr(self, var_name)
            ctx[var_name] = buffer.copy_to_gpu(num)

        # Get positions from GPU and create Context
        positions_gpu = self.positions.copy_to_gpu(sum_scheduled_tokens)

        # Build AttentionMetaData with ctx dict (unpacked as kwargs) and Context
        attn_metadata = AttentionMetaData(
            # TODO: handle the situation that prefill mixed with decode
            cu_seqlens_q=common_attn_metadata.query_start_loc,
            cu_seqlens_k=common_attn_metadata.query_start_loc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=common_attn_metadata.slot_mapping,
            context_lens=common_attn_metadata.seq_lens_cpu,
            dropout_p=dropout_p,
            block_tables=block_tables,
            context=Context(
                positions=positions_gpu,
                is_prefill=True,
                batch_size=bs,
                graph_bs=bs
            ),
            **ctx,
        )
        # print(f"[zejun] ATOM build_prefill - AttentionMetaData members:", flush=True)
        # print(f"  cu_seqlens_q: {attn_metadata.cu_seqlens_q}", flush=True)
        # print(f"  cu_seqlens_k: {attn_metadata.cu_seqlens_k}", flush=True)
        # print(f"  max_seqlen_q: {attn_metadata.max_seqlen_q}", flush=True)
        # print(f"  max_seqlen_k: {attn_metadata.max_seqlen_k}", flush=True)
        # print(f"  min_seqlen_q: {attn_metadata.min_seqlen_q}", flush=True)
        # print(f"  slot_mapping: {attn_metadata.slot_mapping}", flush=True)
        # print(f"  context_lens: {attn_metadata.context_lens}", flush=True)
        # if attn_metadata.block_tables is not None:
        #     print(f"  block_tables shape : {attn_metadata.block_tables.shape}", flush=True)
        # print(f"  dropout_p: {attn_metadata.dropout_p}", flush=True)
        # print(f"  max_q_len: {attn_metadata.max_q_len}", flush=True)
        # print(f"  kv_indptr: {attn_metadata.kv_indptr}", flush=True)
        # print(f"  kv_indices: {attn_metadata.kv_indices}", flush=True)
        # print(f"  kv_last_page_lens: {attn_metadata.kv_last_page_lens}", flush=True)
        # print(f"  cu_seqlen_ks: {attn_metadata.cu_seqlen_ks}", flush=True)
        # print(f"  cu_seqlen_ke: {attn_metadata.cu_seqlen_ke}", flush=True)
        # print(f"  sparse_kv_indptr: {attn_metadata.sparse_kv_indptr}", flush=True)
        # print(f"  work_meta_data: {attn_metadata.work_meta_data}", flush=True)
        # print(f"  work_indptr: {attn_metadata.work_indptr}", flush=True)
        # print(f"  work_info_set: {attn_metadata.work_info_set}", flush=True)
        # print(f"  reduce_indptr: {attn_metadata.reduce_indptr}", flush=True)
        # print(f"  reduce_final_map: {attn_metadata.reduce_final_map}", flush=True)
        # print(f"  reduce_partial_map: {attn_metadata.reduce_partial_map}", flush=True)
        # print(f"  context: {attn_metadata.context}", flush=True)
        # print(f"  block_tables_converted: {attn_metadata.block_tables_converted}", flush=True)
        # print(f"  kv_indices_converted: {attn_metadata.kv_indices_converted}", flush=True)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int = 0,
        common_attn_metadata = None,
        fast_build: bool = False,
    ):
        logger.info(f'Build attention metadata for plugin mode')
        if common_prefix_len > 0:
            raise ValueError("ATOM does not support cascade attention yet")

        from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends
        num_decodes, num_extends, num_prefills, _, _, _ = split_decodes_prefills_and_extends(
            common_attn_metadata=common_attn_metadata,
            decode_threshold=1
        )

        # TODO: support mixed prefill and decode
        if num_decodes > 0 and num_prefills > 0:
            raise ValueError("ATOM does not support mixed prefill and decode yet for now")

        if num_extends > 0:
            raise ValueError("ATOM does not support chunked prefill yet, please add \
                --no-enable-chunked-prefill when launch the server")

        # print('--------------------------------', flush=True)
        # print('[zejun] ATOM, common_attn_metadata query_start_loc = ', common_attn_metadata.query_start_loc, flush=True)
        # print('[zejun] ATOM, common_attn_metadata query_start_loc_cpu = ', common_attn_metadata.query_start_loc_cpu, flush=True)
        # print('[zejun] ATOM, common_attn_metadata seq_lens = ', common_attn_metadata.seq_lens, flush=True)
        # print('[zejun] ATOM, common_attn_metadata seq_lens_cpu = ', common_attn_metadata.seq_lens_cpu, flush=True)
        # print('[zejun] ATOM, common_attn_metadata num_computed_tokens_cpu = ', common_attn_metadata.num_computed_tokens_cpu, flush=True)
        # print('[zejun] ATOM, common_attn_metadata num_reqs = ', common_attn_metadata.num_reqs, flush=True)
        # print('[zejun] ATOM, common_attn_metadata num_actual_tokens = ', common_attn_metadata.num_actual_tokens, flush=True)
        # print('[zejun] ATOM, common_attn_metadata max_query_len = ', common_attn_metadata.max_query_len, flush=True)
        # print('[zejun] ATOM, common_attn_metadata max_seq_len = ', common_attn_metadata.max_seq_len, flush=True)
        # print('[zejun] ATOM, common_attn_metadata block_table_tensor = ', common_attn_metadata.block_table_tensor.shape, flush=True)
        # print('[zejun] ATOM, common_attn_metadata block_table_tensor[0, : 16] = ', common_attn_metadata.block_table_tensor[0, : 16], flush=True)
        # print('[zejun] ATOM, common_attn_metadata slot_mapping = ', common_attn_metadata.slot_mapping, flush=True)
        # print('--------------------------------', flush=True)

        _build_prefill = common_attn_metadata.max_query_len > 1
        if _build_prefill:
            return self.build_prefill(common_prefix_len, common_attn_metadata, fast_build)
        else:
            return self.build_decode(common_prefix_len, common_attn_metadata, fast_build)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata = None,
    ) -> AttentionMetaData:
        logger.info('Build attention metadata for cudagraph capture for plugin mode')
        # TODO: refine here
        self.total_tokens = (
            self.model_config.max_model_len
            * self.scheduler_config.max_num_partial_prefills
        )
        attn_metadata = self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.total_tokens = 0
        return attn_metadata


def AiterAttentionMetadataBuilderDecoratorForPluginMode(default_base_class):
    def decorator(cls):
        is_vllm_mode = is_vllm()
        is_sglang_mode = is_sglang()

        base_class = default_base_class
        class_dict = {}

        for key, value in cls.__dict__.items():
            if not key.startswith('__') or key in ('__annotations__',):
                class_dict[key] = value

        # handle the generic base class
        needs_generic = False
        generic_base = None

        if is_vllm_mode:
            # get the base class and generic base class
            base_class, generic_base, needs_generic, class_dict = \
                setup_attn_metadata_builder_base_class_and_attributes(class_dict)

            # replace the __init__ method to the decorated class
            class_dict['__init__'] = create_attn_metadata_builder_init_method(base_class)

            # add the methods to the decorated class
            for method_name in dir(vllmAttentionMetadataBuilderMethods):
                if not method_name.startswith('_'):
                    method = getattr(vllmAttentionMetadataBuilderMethods, method_name)
                    if callable(method):
                        class_dict[method_name] = method
        elif is_sglang_mode:
            raise NotImplementedError("AttentionMetadataBuilder for sglang is not implemented yet")

        # create the new class
        new_class = type(cls.__name__, (base_class,), class_dict)

        # replace the inherit base class for plugin mode, meanwhile support generic base class
        if needs_generic and generic_base is not None:
            new_class.__orig_bases__ = (generic_base[new_class],)

        return new_class

    return decorator
