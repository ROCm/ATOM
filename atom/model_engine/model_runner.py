import logging
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter.dist.utils import get_distributed_init_method
from aiter import destroy_dist_env, dtypes, init_dist_env
from aiter.dist.parallel_state import graph_capture, get_tp_group
from atom.config import Config, set_current_atom_config, KVCacheTensor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence
from atom.model_loader.loader import load_model
from atom.model_ops.sampler import Sampler
from atom.models.deepseek_v2 import DeepseekV2ForCausalLM
from atom.models.llama import LlamaForCausalLM
from atom.models.mixtral import MixtralForCausalLM
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.utils import CpuGpuBuffer, init_exit_handler, get_hf_text_config
from atom.utils.selector import get_attn_backend
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.utils import CpuGpuBuffer, envs, init_exit_handler, get_hf_text_config
from aiter.dist.parallel_state import get_dp_group, get_tp_group

logger = logging.getLogger("atom")
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
    DPMetadata,
)

suppot_model_arch_dict = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV32ForCausalLM": DeepseekV2ForCausalLM,
}
# seed = 34567
# np.random.seed(seed)
# torch.cuda.manual_seed_all(seed)


class tokenIDProcessor:

    def __init__(self, max_num_batched_tokens: int, device: torch.device):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        # self.is_deferred_out = False
        self.is_deferred_out = True
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.input_ids_loc = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_event = torch.cuda.Event()
        self.async_copy_stream = torch.cuda.Stream()
        self.clean()

    def send_to_cpu_async(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            self.async_copy_event.record(self.async_copy_stream)
        self.token_ids_cpu.append(cpu_tensor)
        self.token_ids_gpu.append(gpu_tensor)

    def recv_async_output(self) -> list[int]:
        self.async_copy_event.synchronize()
        for _ in self.token_ids_cpu:
            token_ids = self.token_ids_cpu.pop(0).tolist()
            return token_ids
        return []

    def clean(self):
        self.token_ids_gpu: list[torch.Tensor] = []
        self.token_ids_cpu: list[torch.Tensor] = []

        self.prev_batch: Optional[ScheduledBatch] = None

    def prepare_sampled_ids(
        self, batch: ScheduledBatch, sampled_token_ids: torch.Tensor
    ) -> dict[int, int]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            seq_ids = batch.seqs.keys()
            ret = {seq_id: token_id for seq_id, token_id in zip(seq_ids, token_ids)}
            ret[-1] = 0
            return ret
        token_ids = self.recv_async_output()
        self.send_to_cpu_async(sampled_token_ids)

        if self.prev_batch is not None:
            seq_ids = self.prev_batch.seqs.keys()
            token_ids = {
                seq_id: token_id for seq_id, token_id in zip(seq_ids, token_ids)
            }
        else:
            # first time, no previous tokens
            token_ids = {}

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids
        token_ids[-1] = 1
        return token_ids

    def get_prev_alive_locations(self, batch: ScheduledBatch) -> tuple[list[int], bool]:
        token_ids = self.prev_token_ids
        deferred_prev_indices = [
            i
            for i, seq_id in enumerate(self.prev_batch.seqs.keys())
            if seq_id in batch.seqs
        ]
        return (
            deferred_prev_indices,
            len(deferred_prev_indices) == token_ids.size(0),
        )

    def prepare_input_ids(
        self,
        batch: ScheduledBatch,
    ) -> torch.Tensor:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""
        seqs = list(batch.seqs.values())
        token_nums = batch.num_scheduled_tokens
        total_tokens = batch.total_tokens_num
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        total_reqs_decode = batch.total_seqs_num_decode
        """for prefill: all input ids are new"""
        start_loc = 0
        for seq, new_token_num in zip(
            seqs[:total_reqs_prefill], token_nums[:total_reqs_prefill]
        ):
            self.input_ids.np[start_loc : start_loc + new_token_num] = seq[
                seq.num_cached_tokens :
            ]
            start_loc += new_token_num
        self.input_ids.copy_to_gpu(total_tokens_prefill)

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            return self.input_ids.gpu[:total_tokens_prefill]

        if not self.is_deferred_out:
            token_ids = [
                seq.token_ids[-1]
                for seq in seqs[
                    total_reqs_prefill : total_reqs_prefill + total_reqs_decode
                ]
            ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            self.input_ids.copy_to_gpu(total_tokens_decode)
            return self.input_ids.gpu[:total_tokens_decode]

        """for decode: input ids are from prev_sampled_token_ids"""
        locations, is_all_alive = self.get_prev_alive_locations(batch)
        num_deferred_tokens = len(locations)
        if is_all_alive:
            num_norm_tokens = total_tokens_decode - num_deferred_tokens
            if num_norm_tokens > 0:
                token_ids = [
                    seq.token_ids[-1]
                    for seq in seqs[
                        total_reqs_prefill : total_reqs_prefill + num_norm_tokens
                    ]
                ]
                self.input_ids.np[:num_norm_tokens] = token_ids
                self.input_ids.copy_to_gpu(num_norm_tokens)
            # no new requests added and old requests finished
            self.input_ids.gpu[
                num_norm_tokens : num_norm_tokens + num_deferred_tokens
            ] = self.prev_token_ids

        elif num_deferred_tokens == total_tokens_decode:
            # no new requests added but some old requests finished
            self.input_ids_loc.np[:num_deferred_tokens] = locations
            self.input_ids_loc.copy_to_gpu(num_deferred_tokens)
            torch.gather(
                self.prev_token_ids,
                0,
                self.input_ids_loc.gpu[:num_deferred_tokens],
                out=self.input_ids.gpu[:num_deferred_tokens],
            )
        else:
            # TODO: new requests' input_ids need to be filled in
            assert False, "TODO new requests' input_ids need to be filled in"
        return self.input_ids.gpu[:total_tokens]


class ModelRunner:

    def __init__(self, rank: int, config: Config):
        self.config = config
        set_current_atom_config(config)
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.label = f"Model Runner{rank}/{self.world_size}"
        self.hf_text_config = get_hf_text_config(hf_config)
        self.use_mla = self.is_deepseek_mla()
        self.is_deepseek_v32 = (
            hasattr(hf_config, "index_topk") if self.use_mla else False
        )
        # Calculate local device rank considering both TP and DP
        # When data parallelism is enabled on the same node, different DP ranks
        # need to use different sets of GPUs
        dp_rank_local = config.parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = 0
        local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        device = torch.device(f"cuda:{local_device_rank}")
        print(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, local_device_rank={local_device_rank}, device={device}"
        )
        self.device = device
        
        # Initialize profiler for this rank
        self.profiler = None
        self.profiler_dir = None
        if config.torch_profiler_dir is not None:
            # Create rank-specific profiler directory
            self.profiler_dir = os.path.join(config.torch_profiler_dir, f"rank_{rank}")
            os.makedirs(self.profiler_dir, exist_ok=True)

        self.graph_bs = [0]  # for eager fallback

        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
        )
        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )
        init_exit_handler(self)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.attn_backend = get_attn_backend(
            self.block_size,
            use_mla=self.use_mla,
        )
        self.tokenID_processor = tokenIDProcessor(
            self.config.max_num_batched_tokens, self.device
        )
        self.sampler = Sampler()
        self.arange_np = np.arange(
            max(
                self.config.max_num_seqs + 1,
                self.config.max_model_len,
                self.config.max_num_batched_tokens,
            ),
            dtype=np.int64,
        )
        self.async_output_copy_stream = torch.cuda.Stream()
        self.model = suppot_model_arch_dict[hf_config.architectures[0]](config)
        self.use_kv_indptr = False
        torch.set_default_device(None)
        load_model(self.model, config.model, config.hf_config, config.load_dummy)
        if isinstance(self.model, DeepseekV2ForCausalLM):
            self.use_kv_indptr = True
        torch.set_default_device("cuda")
        self.allocate_forward_vars()
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(self)
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type in ("deepseek_v2", "deepseek_v3")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # Bfloat16 torch tensors cannot be directly cast to a numpy array, so
        # if a bfloat16 buffer is needed without a corresponding numpy array,
        # don't bother instantiating the numpy array.
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        destroy_dist_env()
        return True

    def start_profiler(self):
        """Start profiling for this rank"""
        if self.profiler_dir is not None and self.profiler is None:
            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(
                    self.profiler_dir, use_gzip=True
                ),
            )
            self.profiler.__enter__()

    def stop_profiler(self):
        """Stop profiling for this rank"""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        return True

    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization. """
        num_tokens_original = 1

        seq = Sequence([0] * num_tokens_original, block_size=self.block_size)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = [0]
        bs = 1

        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32), 
            total_tokens_num=num_tokens_original,  # original value
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
        )

        attn_metadata, positions = self.attn_metadata_builder.build(dummy_batch, bs)
        context_bs = dummy_batch.total_seqs_num_decode

        num_input_tokens, num_tokens_across_dp = self._preprocess(dummy_batch)

        context = Context(
            positions=positions,
            is_prefill=False,
            batch_size=context_bs,
            graph_bs=bs,
        )

        actual_num_tokens = dummy_batch.total_tokens_num
        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=actual_num_tokens,  # original value, not with padding
            num_tokens_across_dp=num_tokens_across_dp,
        )

        self.tokenID_processor.input_ids.np[:num_input_tokens] = [0] * num_input_tokens
        self.tokenID_processor.input_ids.copy_to_gpu(num_input_tokens)
        input_ids = self.tokenID_processor.input_ids.gpu[:num_input_tokens]

        logits = self.run_model(input_ids)

        reset_forward_context()
        logger.debug(f"{self.label}: dummy batch executed with {num_input_tokens} tokens")
        return True

    def warmup_model(self):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        dp_size = get_dp_group().world_size
        warmup_max_tokens = max_num_batched_tokens // dp_size
        
        num_seqs = min(
            warmup_max_tokens // max_model_len, self.config.max_num_seqs
        )
        
        if num_seqs == 0:
            num_seqs = 1
            seq_len = min(warmup_max_tokens, max_model_len)
            if seq_len == 0:
                seq_len = 1
            logger.warning(
                f"{self.label}: DP size={dp_size} too large, warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
                f"Using {num_seqs} seq with length {seq_len} for warmup."
            )
        else:
            seq_len = max_model_len

        seqs = [
            Sequence([0] * seq_len, block_size=self.block_size)
            for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([seq_len] * num_seqs, dtype=np.int32)
        total_tokens_num = num_scheduled_tokens.sum()

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num,
            total_tokens_num_prefill=total_tokens_num,
            total_seqs_num=num_seqs,
            total_seqs_num_prefill=num_seqs,
        )
        self.forward(dummy_batch)
        self.tokenID_processor.clean()
        torch.cuda.empty_cache()
        logger.info(
            f"{self.label}: warmup_model {time.time() - start_time:.2f} seconds with {num_seqs} reqs {total_tokens_num} tokens"
        )

    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.hf_config.torch_dtype
        self.max_bs = min(self.config.max_num_seqs, 512)
        self.max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}

        # TODO: remove it in forward_context
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "positions": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "temperatures": CpuGpuBuffer(self.max_bs, **f32_kwargs),
            "outputs": torch.empty(self.max_bs, hidden_size, dtype=hidden_type),
        }

    def get_num_blocks(self):
        torch.set_default_device("cuda")
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        torch.set_default_device("cpu")
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        if self.use_mla:
            block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * 576
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )
            if self.is_deepseek_v32:
                index_dim = hf_config.index_head_dim + 4
                aligned_index_dim = ((index_dim + 15) // 16) * 16
                block_bytes += (
                    hf_config.num_hidden_layers
                    * self.block_size
                    * aligned_index_dim
                    * dtypes.fp8.itemsize
                )
        else:
            block_bytes = (
                2
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * hf_config.head_dim
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )
        num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert num_kvcache_blocks > 0, f"need at least {block_bytes} KV cache"
        return num_kvcache_blocks

    def allocate_kv_cache(self, num_kvcache_blocks):
        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        if self.use_mla:
            self.kv_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                576,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
            if self.is_deepseek_v32:
                # Align last dimension to 16 bytes for fp8 (1 byte per element)
                # to avoid unaligned memory access in torch inductor
                index_dim = hf_config.index_head_dim + 4
                aligned_index_dim = ((index_dim + 15) // 16) * 16
                self.index_cache = torch.zeros(
                    hf_config.num_hidden_layers,
                    config.num_kvcache_blocks,
                    self.block_size,
                    aligned_index_dim,
                    dtype=dtypes.fp8,
                    device="cuda",
                )
        else:
            self.kv_cache = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                hf_config.head_dim,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )

            self.kv_scale = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                dtype=dtypes.fp32,
                device="cuda",
            )
        # Build KVCacheConfig
        # lirong TODO: This is a simple solution to build KVCacheConfig,
        # models with only one type of attention, but not support multi-type of attention models.
        # We need to support it by kv_cache_group in the future.
        kv_cache_tensors = []
        layer_id = 0
        x = 16 // self.kv_cache.element_size()
        for module in self.model.modules():
            # Since use attention base and there are child in attention, add base condition
            if hasattr(module, "base_attention"):
                if hasattr(module, "use_mla") and not module.use_mla:
                    # Non-MLA attention
                    k_cache = self.kv_cache[0, layer_id].view(
                        config.num_kvcache_blocks,
                        num_kv_heads,
                        hf_config.head_dim // x,
                        self.block_size,
                        x,
                    )
                    v_cache = self.kv_cache[1, layer_id].view(
                        config.num_kvcache_blocks,
                        num_kv_heads,
                        hf_config.head_dim,
                        self.block_size,
                    )
                    module.max_model_len = self.config.max_model_len
                    if config.kv_cache_dtype == "fp8":
                        module.k_scale = self.kv_scale[0, layer_id]
                        module.v_scale = self.kv_scale[1, layer_id]

                    k_scale = module.k_scale
                    v_scale = module.v_scale

                    # Store in KVCacheTensor
                    kv_cache_tensor = KVCacheTensor(
                        layer_num=layer_id,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                    kv_cache_tensors.append(kv_cache_tensor)

                    module.k_cache = k_cache
                    module.v_cache = v_cache

                    layer_id += 1
                elif hasattr(module, "use_mla") and module.use_mla:
                    # MLA attention
                    kv_cache = self.kv_cache[layer_id].view(
                        config.num_kvcache_blocks * self.block_size,
                        1,
                        576,
                    )
                    module.max_model_len = self.config.max_model_len
                    if self.is_deepseek_v32 and module.indexer is not None:
                        # Use aligned dimension to avoid memory copy in torch inductor
                        module.indexer.k_cache.kv_cache[0] = self.index_cache[
                            layer_id
                        ].view(
                            config.num_kvcache_blocks * self.block_size,
                            1,
                            aligned_index_dim,
                        )
                    # Store in KVCacheTensor
                    kv_cache_tensor = KVCacheTensor(
                        layer_num=layer_id,
                        k_cache=kv_cache,
                        v_cache=None,
                        k_scale=None,
                        v_scale=None,
                    )
                    kv_cache_tensors.append(kv_cache_tensor)

                    module.kv_cache = kv_cache
                    module.max_model_len = self.config.max_model_len
                    layer_id += 1

        # Store KVCacheConfig
        kv_cache_data = {
            f"layer_{i}": kv_cache_tensor
            for i, kv_cache_tensor in enumerate(kv_cache_tensors)
        }
        # vllm use register_kv_caches to register kv_cache_data. We just set it to global here
        set_kv_cache_data(kv_cache_data)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return True

    def get_dp_padding(self, num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1 or self.enforce_eager:
            # Early exit.
            return 0, None
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()

        return max_tokens_across_dp_cpu - num_tokens, num_tokens_across_dp

    def _preprocess(self, batch: ScheduledBatch):
        num_input_tokens = batch.total_tokens_num
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad
        return num_input_tokens, num_tokens_across_dp

    def prepare_intputs(self, batch: ScheduledBatch):
        is_prefill = batch.total_tokens_num_prefill > 0
        bs = batch.total_seqs_num
        num_scheduled_tokens = batch.num_scheduled_tokens
        cu_seqlens_q, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        self.forward_vars["cu_seqlens_q"].np[1 : bs + 1] = cu_seqlens_q
        if not is_prefill:
            scheduled_bs = batch.total_seqs_num_decode
            seqs = list(batch.seqs.values())
            seqs = seqs[batch.total_seqs_num_prefill :]
            assert len(seqs) == scheduled_bs
            bs = (
                scheduled_bs
                if self.enforce_eager
                else next((x for x in self.graph_bs if x >= scheduled_bs), scheduled_bs)
                # Use cudagraph and padding to batch_size, if bs > graph_bs, use eager mode
            )
            assert (
                bs >= scheduled_bs
            ), f"current decode {scheduled_bs=} > max graph_bs{bs}"
            self.forward_vars["cu_seqlens_q"].np[scheduled_bs + 1 : bs + 1] = (
                self.forward_vars["cu_seqlens_q"].np[scheduled_bs]
            )
        attn_metadata, positions = self.attn_metadata_builder.build(batch, bs)
        context_bs = (
            batch.total_seqs_num_prefill if is_prefill else batch.total_seqs_num_decode
        )

        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            batch_size=context_bs,
            graph_bs=bs,
        )
        num_input_tokens, num_tokens_across_dp = self._preprocess(batch)        
        actual_num_tokens = batch.total_tokens_num
        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=actual_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        return num_input_tokens

    def prepare_sample(self, batch: ScheduledBatch) -> torch.Tensor:
        temperatures = [seq.temperature for seq in batch.seqs.values()]
        bs = batch.total_seqs_num
        buffer = self.forward_vars["temperatures"]
        buffer.np[:bs] = temperatures
        return buffer.copy_to_gpu(bs)

    def prepare_model(self, batch: ScheduledBatch):
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        input_ids = self.tokenID_processor.prepare_input_ids(batch)
        # if self.rank == 0:
        #     print(f"input_ids: {input_ids}")

        self.prepare_intputs(batch)
        temperatures = self.prepare_sample(batch)
        return (
            input_ids,
            temperatures,
        )

    def run_model(self, input_ids: torch.Tensor):
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.batch_size
        is_prefill = context.is_prefill
        positions = context.positions
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            hidden_states = self.model(input_ids, positions)
        else:
            graph_bs = context.graph_bs
            self.graphs[graph_bs].replay()
            hidden_states = self.forward_vars["outputs"][:bs]
        return self.model.compute_logits(hidden_states)

    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> dict[int, int]:
        sampled_tokens = self.sampler(logits, temperatures)
        if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)
        token_ids = self.tokenID_processor.prepare_sampled_ids(
            batch,
            sampled_tokens,
        )
        return token_ids

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> dict[int, int]:
        input_ids, temperatures = self.prepare_model(batch)
        logits = self.run_model(input_ids)
        reset_forward_context()
        return self.postprocess(batch, logits, temperatures)

    @torch.inference_mode()
    def capture_cudagraph(self):
        start_time = time.time()
        # self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if self.config.compilation_config.cudagraph_capture_sizes:
            self.graph_bs = self.config.compilation_config.cudagraph_capture_sizes
        else:
            cuda_graph_sizes = self.config.compilation_config.cuda_graph_sizes
            if len(cuda_graph_sizes) == 1:
                self.graph_bs = [1, 2, 4, 8] + [
                    i for i in range(16, cuda_graph_sizes[0] + 1, 16)
                ]
            elif len(cuda_graph_sizes) > 1:
                self.graph_bs = cuda_graph_sizes
        self.graph_bs.sort(reverse=True)

        self.forward_vars["cu_seqlens_q"].np[: self.graph_bs[0] + 1] = np.arange(
            0, self.graph_bs[0] + 1
        )
        self.forward_vars["cu_seqlens_q"].copy_to_gpu(self.graph_bs[0] + 1)
        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]

        self.graphs: dict[int, torch.cuda.CUDAGraph] = dict()
        self.graph_pool = None

        with graph_capture() as gc:
            capture_range = (
                tqdm.tqdm(self.graph_bs) if self.rank == 0 else self.graph_bs
            )
            for bs in capture_range:
                if self.rank == 0:
                    capture_range.set_description(f"Capturing {bs=}")
                graph = torch.cuda.CUDAGraph()

                attn_metadata, context = (
                    self.attn_metadata_builder.build_for_cudagraph_capture(bs)
                )
                num_tokens = bs
                num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                num_tokens += num_pad
                set_forward_context(
                    attn_metadata=attn_metadata,
                    atom_config=self.config,
                    context=context,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                )

                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup

                with torch.cuda.graph(graph, self.graph_pool, stream=gc.stream):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
        self.graph_bs.sort(reverse=False)
        return time.time() - start_time, self.graph_bs
