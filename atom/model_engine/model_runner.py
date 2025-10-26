import logging
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, dtypes, init_dist_env
from aiter.dist.parallel_state import graph_capture
from atom.config import Config, set_current_atom_config
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence
from atom.model_loader.loader import load_model
from atom.model_ops.sampler import Sampler
from atom.models.deepseek_v2 import DeepseekV2ForCausalLM
from atom.models.llama import LlamaForCausalLM
from atom.models.mixtral import MixtralForCausalLM
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.utils import CpuGpuBuffer, init_exit_handler
from atom.utils.context import get_context, reset_context, set_context

logger = logging.getLogger("atom")
from atom.utils.forward_context import AttentionMetadata, set_forward_context

suppot_model_arch_dict = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV2ForCausalLM,
}
# seed=1
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

        # Initialize profiler for this rank
        self.profiler = None
        self.profiler_dir = None
        if config.torch_profiler_dir is not None:
            # Create rank-specific profiler directory
            self.profiler_dir = os.path.join(config.torch_profiler_dir, f"rank_{rank}")
            os.makedirs(self.profiler_dir, exist_ok=True)

        self.graph_bs = [0]  # for eager fallback

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        self.device = device
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        init_dist_env(self.world_size, rankID=rank)
        init_exit_handler(self)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
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
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")

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

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [
            Sequence([0] * max_model_len, block_size=self.block_size)
            for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([max_model_len] * num_seqs, dtype=np.int32)
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

    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.hf_config.torch_dtype
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "positions": CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs),
            "slot_mapping": CpuGpuBuffer(max_num_batched_tokens, **i64_kwargs),
            "context_lens": CpuGpuBuffer(max_bs, **i32_kwargs),
            "block_tables": CpuGpuBuffer(max_bs, max_num_blocks, **i32_kwargs),
            "temperatures": CpuGpuBuffer(max_bs, **f32_kwargs),
            "cu_seqlens_q": CpuGpuBuffer(max_bs + 1, **i32_kwargs),
            "kv_indptr": CpuGpuBuffer(max_bs + 1, **i32_kwargs),
            "kv_indices": CpuGpuBuffer(max_bs * max_num_blocks, **i32_kwargs),
            "kv_last_page_lens": CpuGpuBuffer(max_bs, **i32_kwargs),
            "outputs": torch.empty(max_bs, hidden_size, dtype=hidden_type),
        }
        self.forward_vars["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, max_bs + 1, step=1, dtype=torch.int32)
        )
        self.forward_vars["cu_seqlens_q"].copy_to_gpu()

        self.forward_vars["kv_last_page_lens"].cpu.fill_(1)
        self.forward_vars["kv_last_page_lens"].copy_to_gpu()

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
        isMLA = hf_config.architectures[0].startswith("DeepseekV")
        if isMLA:
            block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * 576
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
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
        isMLA = hf_config.architectures[0].startswith("DeepseekV")
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        if isMLA:
            self.kv_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                576,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
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

        layer_id = 0
        x = 16 // self.kv_cache.element_size()
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id].view(
                    config.num_kvcache_blocks,
                    num_kv_heads,
                    hf_config.head_dim // x,
                    self.block_size,
                    x,
                )
                module.v_cache = self.kv_cache[1, layer_id].view(
                    config.num_kvcache_blocks,
                    num_kv_heads,
                    hf_config.head_dim,
                    self.block_size,
                )
                module.max_model_len = self.config.max_model_len
                if config.kv_cache_dtype == "fp8":
                    module.k_scale = self.kv_scale[0, layer_id]
                    module.v_scale = self.kv_scale[1, layer_id]
                attention_metadata = AttentionMetadata(
                    k_cache=module.k_cache,
                    v_cache=module.v_cache,
                    k_scale=module.k_scale,
                    v_scale=module.v_scale,
                )
                set_forward_context(module.layer_num, attention_metadata)

                layer_id += 1
            elif hasattr(module, "kv_cache") and isMLA:
                module.kv_cache = self.kv_cache[layer_id].view(
                    config.num_kvcache_blocks * self.block_size,
                    1,
                    576,
                )
                module.max_model_len = self.config.max_model_len
                attention_metadata = AttentionMetadata(
                    k_cache=module.kv_cache,
                    v_cache=None,
                    k_scale=None,
                    v_scale=None,
                )
                set_forward_context(module.layer_num, attention_metadata)
                layer_id += 1
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return True

    def prepare_block_tables(self, seqs: list[Sequence]):
        block_tables = self.forward_vars["block_tables"].np
        for i, seq in enumerate(seqs):
            block_tables[i] = 0
            block_tables[i, : seq.num_blocks] = seq.block_table

    def prepare_prefill(self, batch: ScheduledBatch):
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.forward_vars
        positions = []
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        seqs = list(batch.seqs.values())
        seqs = seqs[:bs]
        for seq in seqs:
            seqlen = seq.num_tokens
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > batch.total_tokens_num:  # prefix cache
            self.prepare_block_tables(seqs)
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)

        min_seqlen_q = 0
        dropout_p = 0.0
        vars_used = [
            ("block_tables", bs),
            ("cu_seqlens_q", bs + 1),
            ("slot_mapping", len(slot_mapping)),
        ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        set_context(
            True,
            batch_size=bs,
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            # slot_mapping=slot_mapping.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            **ctx,
        )
        return var["positions"].copy_to_gpu(sum_scheduled_tokens)

    def prepare_decode(self, batch: ScheduledBatch):
        scheduled_bs = batch.total_seqs_num_decode
        seqs = list(batch.seqs.values())
        seqs = seqs[batch.total_seqs_num_prefill :]
        assert len(seqs) == scheduled_bs
        bs = (
            scheduled_bs
            if self.enforce_eager
            else next(x for x in self.graph_bs if x >= scheduled_bs)
        )
        assert bs >= scheduled_bs, f"current decode {scheduled_bs=} > max graph_bs{bs}"
        dropout_p = 0.0
        max_q_len = 1
        self.total_blocks = 0

        context_lens = [seq.num_tokens for seq in seqs]
        positions = context_lens
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in seqs
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        if self.use_kv_indptr:
            assert self.block_size == 1, "AITER MLA requires only block size 1."
            self.kv_indices: list[int] = []
            self.kv_indptr: list[int] = [0]
            self.kv_last_page_lens: list[int] = []

            for seq in seqs:
                current_seq_len = seq.num_tokens
                self._update_paged_kv_tensors(seq.block_table, current_seq_len)

        self.prepare_block_tables(seqs)

        sum_scheduled_tokens = batch.total_tokens_num_decode
        var = self.forward_vars
        var["slot_mapping"].np[:bs] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
        ]

        if self.use_kv_indptr and len(self.kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the scheduler
            self.kv_indices.extend([0] * (self.total_blocks - len(self.kv_indices)))
            var["kv_indices"].np[: self.total_blocks] = np.array(
                self.kv_indices, dtype=np.int64
            )
            var["kv_indptr"].np[: scheduled_bs + 1] = np.array(
                self.kv_indptr, dtype=np.int64
            )
            var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = var["kv_indptr"].np[
                scheduled_bs
            ]
            var["kv_last_page_lens"].np[:scheduled_bs] = np.array(
                self.kv_last_page_lens, dtype=np.int64
            )
            var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
            vars_used = [
                ("slot_mapping", bs),  # TODO: MTP support
                ("context_lens", bs),
                ("block_tables", bs),
                ("cu_seqlens_q", bs + 1),
                ("kv_indices", sum(context_lens)),
                ("kv_indptr", bs + 1),
                ("kv_last_page_lens", bs),
            ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        set_context(
            False,
            batch_size=scheduled_bs,
            graph_bs=bs,
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        # for el in ctx:
        #     print(f"{el}: {ctx[el]}")
        # print(f"positions: {positions}")
        return positions

    def _update_paged_kv_tensors(self, block_table: list[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = (
            seq_len // self.block_size + 1
            if seq_len % self.block_size != 0
            else seq_len // self.block_size
        )
        self.kv_indices.extend(block_table[:block_table_bound])
        self.kv_indptr.append(self.kv_indptr[-1] + block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.kv_last_page_lens.append(last_page_len)

    def prepare_sample(self, batch: ScheduledBatch) -> torch.Tensor:
        temperatures = [seq.temperature for seq in batch.seqs.values()]
        bs = batch.total_seqs_num
        buffer = self.forward_vars["temperatures"]
        buffer.np[:bs] = temperatures
        return buffer.copy_to_gpu(bs)

    def prepare_model(self, batch: ScheduledBatch):
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0
        bs = batch.total_seqs_num
        seqs = batch.seqs.values()

        num_scheduled_tokens = batch.num_scheduled_tokens
        cu_seqlens_q, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        self.forward_vars["cu_seqlens_q"].np[1 : bs + 1] = cu_seqlens_q

        input_ids = self.tokenID_processor.prepare_input_ids(batch)
        # print(f"{input_ids=}")

        is_prefill = batch.total_tokens_num_prefill > 0
        prepare_func = self.prepare_prefill if is_prefill else self.prepare_decode
        positions = prepare_func(batch)
        temperatures = self.prepare_sample(batch)
        return (
            input_ids,
            positions,
            temperatures,
        )

    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        ctx = get_context()
        bs = ctx.batch_size
        is_prefill = ctx.is_prefill
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            hidden_states = self.model(input_ids, positions)
        else:
            graph_bs = ctx.graph_bs
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
        # print(f"{logits=}")
        # print(f"{sampled_tokens=}")
        token_ids = self.tokenID_processor.prepare_sampled_ids(
            batch,
            sampled_tokens,
        )
        return token_ids

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> dict[int, int]:
        input_ids, positions, temperatures = self.prepare_model(batch)
        logits = self.run_model(input_ids, positions)
        reset_context()
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
        slot_mapping = self.forward_vars["slot_mapping"].gpu
        context_lens = self.forward_vars["context_lens"].gpu
        block_tables = self.forward_vars["block_tables"].gpu
        cu_seqlens_q = self.forward_vars["cu_seqlens_q"].gpu
        kv_indptr = self.forward_vars["kv_indptr"].gpu
        kv_indices = self.forward_vars["kv_indices"].gpu
        kv_last_page_lens = self.forward_vars["kv_last_page_lens"].gpu
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

                set_context(
                    False,
                    batch_size=bs,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    max_q_len=1,
                    cu_seqlens_q=cu_seqlens_q[: bs + 1],
                    kv_indptr=kv_indptr[: bs + 1],
                    kv_indices=kv_indices[:],
                    kv_last_page_lens=kv_last_page_lens[:bs],
                )

                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup

                with torch.cuda.graph(graph, self.graph_pool, stream=gc.stream):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()
        self.graph_bs.sort(reverse=False)
        return time.time() - start_time, self.graph_bs
