import logging
import os
import signal
import time
import weakref
from typing import Optional, Union, Any
import queue
from collections import deque

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, dtypes, init_dist_env
from aiter.dist.parallel_state import get_tensor_model_parallel_rank, graph_capture

from atom.config import Config, set_current_atom_config
from atom.model_engine.scheduler import ScheduledBatchs, InputBatch
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

class OutputProcessor:

    def __init__(self):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_event = torch.cuda.Event()
        self.async_copy_stream = torch.cuda.Stream()
        self.deferred_request_id: list[list[int]] = []
        self.deferred_token_id: list[list[int]] = []
        self.token_ids_gpu: list[torch.Tensor] = []
        self.token_ids_cpu: list[torch.Tensor] = []

        self.token_req_ids: list[list[int]] = []

        self.pending_outputs: deque = deque()  # saving (cpu_tensor, seqs)


    def send_to_cpu_async(self, batch: ScheduledBatchs, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            self.async_copy_event.record(self.async_copy_stream)
        # self.token_ids_gpu.append(gpu_tensor)
        self.token_ids_cpu.append(cpu_tensor)
        self.pending_outputs.append((cpu_tensor, batch.seqs))

    def revc_async_output_tuple(self) -> tuple[list[int], list[Sequence]]:
        if not self.pending_outputs:
            return [], []
            
        self.async_copy_event.synchronize()
        if self.pending_outputs:
            cpu_tensor, seqs = self.pending_outputs.popleft()
            token_ids = cpu_tensor.tolist()
            return token_ids, seqs
        return [], []

    def recv_async_output(self) -> list[int]:
        self.async_copy_event.synchronize()
        for _ in self.token_ids_cpu:
            token_ids = self.token_ids_cpu.pop(0).tolist()
            return token_ids
        return []

    def clean_token(self):
        self.token_ids_cpu: list[torch.Tensor] = []
        self.pending_outputs.clear()

    def update_and_ret(
        self, batch: ScheduledBatchs, sampled_token_ids: torch.Tensor
    ) -> list[int]:
        token_ids = self.recv_async_output()
        # prev_token_ids, prev_seqs = self.revc_async_output_tuple()
        self.send_to_cpu_async(batch, sampled_token_ids)
        # return token_ids
        # return prev_token_ids, prev_seqs

        # deferred_reqIDs = []
        returned_reqIDs = []
        for req in batch.seqs:
            if req.id in self.deferred_request_id:
                returned_reqIDs.append(req.id)
            else:
                self.deferred_request_id.append(req.id)

        if returned_reqIDs:
            # for ids in self.token_ids_cpu[:-1]:
            #     ids_list = ids.tolist()
                for req_id in returned_reqIDs:
                    self.deferred_request_id.remove(req_id)
                    # req_index = batch.unfinished_prev_req.index(req_id)
                    # batch.unfinished_prev_req.remove(req_id)
                    # req = batch.seqs[req_index]
                    # req.append_token(ids_list[req_index])

        # self.async_copy_event.synchronize()

        return token_ids

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
        self.out_processor = OutputProcessor()
        self.sampler = Sampler()

        self.input_batch = InputBatch(
            max_num_reqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len)

        self.input_ids = self._make_buffer(self.config.max_num_batched_tokens,
                                           dtype=torch.int32)
        self.positions = self._make_buffer(self.config.max_num_batched_tokens,
                                           dtype=torch.int64)
        self.use_async_scheduling = True
        self.async_output_copy_stream = torch.cuda.Stream() if \
            self.use_async_scheduling else None
    
        self.out_processor = OutputProcessor()


        self.model = suppot_model_arch_dict[hf_config.architectures[0]](config)
        torch.set_default_device(None)
        load_model(self.model, config.model, config.hf_config, config.load_dummy)
        torch.set_default_device("cuda")
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")
        self.allocate_decode_vars()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")


    def _make_buffer(self,
                     *size: Union[int, torch.SymInt],
                     dtype: torch.dtype,
                     numpy: bool = True) -> CpuGpuBuffer:
        # Bfloat16 torch tensors cannot be directly cast to a numpy array, so
        # if a bfloat16 buffer is needed without a corresponding numpy array,
        # don't bother instantiating the numpy array.
        return CpuGpuBuffer(*size,
                            dtype=dtype,
                            device=self.device,
                            pin_memory=True,
                            with_numpy=numpy)


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

        num_scheduled_tokens: dict[str, int] = {}

        for seq in seqs:
            num_scheduled_tokens[seq.id] = seq.num_tokens

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        # print("total_num_scheduled_tokens dummy_batch", total_num_scheduled_tokens)
        dummy_batch = ScheduledBatchs(seqs, True, False, seqs, num_scheduled_tokens, total_num_scheduled_tokens)
        self.forward(dummy_batch)
        self.out_processor.clean_token()
        torch.cuda.empty_cache()

    def allocate_decode_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.hf_config.torch_dtype
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.decode_vars = {
            "input_ids": CpuGpuBuffer(max_bs, dtype=torch.int64, device=self.device),
            "positions": CpuGpuBuffer(max_bs, dtype=torch.int64, device=self.device),
            "slot_mapping": CpuGpuBuffer(max_bs, dtype=torch.int64, device=self.device),
            "context_lens": CpuGpuBuffer(max_bs, dtype=torch.int32, device=self.device),
            "block_tables": CpuGpuBuffer(
                max_bs, max_num_blocks, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": CpuGpuBuffer(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "kv_indptr": CpuGpuBuffer(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "kv_indices": CpuGpuBuffer(
                max_bs * max_num_blocks, dtype=torch.int32, device=self.device
            ),
            "kv_last_page_lens": CpuGpuBuffer(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "outputs": torch.empty(max_bs, hidden_size, dtype=hidden_type),
        }
        self.decode_vars["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, max_bs + 1, step=1, dtype=torch.int32)
        )
        self.decode_vars["cu_seqlens_q"].copy_to_gpu()

        self.decode_vars["kv_last_page_lens"].cpu.fill_(1)
        self.decode_vars["kv_last_page_lens"].copy_to_gpu()

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
                layer_id += 1
        return True

    def prepare_block_tables(self, seqs: list[Sequence]):
        lens = [len(seq.block_table) for seq in seqs]
        block_tables = np.empty((len(seqs), max(lens)), dtype=np.int32)
        for i, seq in enumerate(seqs):
            block_tables[i, : lens[i]] = seq.block_table
        return block_tables

    def _prepare_input_ids(self, total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray) -> None:
        """Prepare the input IDs for the current batch.
        
        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # flattened_indices.append(flattened_index)
                flattened_indices.append(prev_index)
                indices_match &= (prev_index == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(flattened_indices)
        if num_commmon_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens,
                                                        0],
                non_blocking=True)
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=True).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=True).to(self.device, non_blocking=True)
        # print('self.input_batch.prev_sampled_token_ids', self.input_batch.prev_sampled_token_ids)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=input_ids_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
            prev_common_req_indices_tensor])
        #         prev_common_req_indices_tensor, 0])

        # self.positions.gpu[:num_commmon_tokens].copy_()


    def prepare_prefill(self, scheduled_batchs: ScheduledBatchs):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in scheduled_batchs.seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
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
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(scheduled_batchs.seqs)
            block_tables = torch.from_numpy(block_tables).cuda(non_blocking=True)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, pin_memory=True)
        min_seqlen_q = 0
        dropout_p = 0.0
        set_context(
            True,
            len(scheduled_batchs.seqs),
            cu_seqlens_q=cu_seqlens_q.cuda(non_blocking=True),
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=slot_mapping.cuda(non_blocking=True),
            block_tables=block_tables,
            dropout_p=dropout_p,
        )
        return input_ids.cuda(non_blocking=True), positions.cuda(non_blocking=True)

    def prepare_decode(self, scheduled_batchs: ScheduledBatchs):
        bs = len(scheduled_batchs.seqs)
        graph_bs = (
            bs if self.enforce_eager else next(x for x in self.graph_bs if x >= bs)
        )
        assert graph_bs >= bs, f"current decode {bs=} > max graph_bs{graph_bs}"
        input_ids = []
        # positions = []
        slot_mapping = []
        context_lens = []
        dropout_p = 0.0
        max_q_len = 1
        self.total_blocks = 0

        context_lens = [seq.num_tokens for seq in scheduled_batchs.seqs]
        # positions = np.array(context_lens, dtype=np.int64)
        context_lens = np.array(context_lens, dtype=np.int32)
        input_ids = [seq.last_token for seq in scheduled_batchs.seqs]
        input_ids = np.array(input_ids, dtype=np.int64)
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            for seq in scheduled_batchs.seqs
        ]
        slot_mapping.extend([-1] * (graph_bs - bs))
        slot_mapping = np.array(slot_mapping, dtype=np.int64)

        if isinstance(self.model, DeepseekV2ForCausalLM):
            assert self.block_size == 1, "AITER MLA requires only block size 1."
            self.kv_indices: list[int] = []
            self.kv_indptr: list[int] = [0]
            self.kv_last_page_lens: list[int] = []
            qlens = [1 for seq in scheduled_batchs.seqs]  # TODO support mtp
            num_scheduled_tokens = np.array(qlens, dtype=np.int32)
            max_q_len = max(qlens)
            cu_seqlens_q = np.cumsum(num_scheduled_tokens, dtype=np.int32)

            for seq in scheduled_batchs.seqs:
                current_seq_len = len(seq)
                self._update_paged_kv_tensors(seq.block_table, current_seq_len)

        block_tables = self.prepare_block_tables(scheduled_batchs.seqs)
        # cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        # cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
        num_scheduled_tokens = scheduled_batchs.total_num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens
        prev_input_ids_ = self.input_ids.gpu[:num_input_tokens]
        prev_positions_ = self.input_batch.prev_position_ids

        var = self.decode_vars
        var["slot_mapping"].np[:graph_bs] = slot_mapping
        var["input_ids"].np[:bs] = prev_input_ids_[:bs].cpu()
        var["positions"].np[:bs] = prev_positions_[:bs]
        # var["input_ids"].np[:bs] = input_ids
        # var["positions"].np[:bs] = positions

        var["context_lens"].np[:bs] = context_lens
        var["block_tables"].np[:bs, : block_tables.shape[1]] = block_tables

        if isinstance(self.model, DeepseekV2ForCausalLM) and len(self.kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the scheduler
            self.kv_indices.extend([0] * (self.total_blocks - len(self.kv_indices)))
            var["kv_indices"].np[: self.total_blocks] = np.array(
                self.kv_indices, dtype=np.int64
            )
            var["kv_indptr"].np[: bs + 1] = np.array(self.kv_indptr, dtype=np.int64)
            var["kv_indptr"].np[bs + 1 : graph_bs + 1] = var["kv_indptr"].np[bs]
            var["kv_last_page_lens"].np[:bs] = np.array(
                self.kv_last_page_lens, dtype=np.int64
            )
            var["cu_seqlens_q"].np[1 : bs + 1] = np.array(cu_seqlens_q, dtype=np.int64)

        for el in [
            "slot_mapping",
            "input_ids",
            "positions",
            "context_lens",
            "block_tables",
            "cu_seqlens_q",
            "kv_indices",
            "kv_indptr",
            "kv_last_page_lens",
        ]:
            var[el].copy_to_gpu()
        set_context(
            False,
            batch_size=bs,
            graph_bs=graph_bs,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs, : block_tables.shape[1]],
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            kv_last_page_lens=var["kv_last_page_lens"].gpu[:bs],
        )
        return prev_input_ids_, prev_positions_
        return var["input_ids"].gpu[:bs], var["positions"].gpu[:bs]

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

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures


    def prepare_model(self, scheduled_batchs: ScheduledBatchs):
        self._update_states(scheduled_batchs)

        total_num_scheduled_tokens = scheduled_batchs.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        _req_ids = self.input_batch.req_ids
        req_ids = [req_id for req_id in _req_ids if req_id is not None]

        tokens = [scheduled_batchs.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # cu_num_tokens, arange = self._get_cumsum_and_arange(
        #     num_scheduled_tokens)
        cu_num_tokens = np.concatenate([np.arange(n) for n in num_scheduled_tokens])

        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
        seqs = scheduled_batchs.seqs
        is_prefill = scheduled_batchs.is_prefill
        prepare_func = self.prepare_prefill if is_prefill else self.prepare_decode
        input_ids, positions = prepare_func(scheduled_batchs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        return input_ids, positions, temperatures

    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        torch.cuda.empty_cache()
        ctx = get_context()
        bs = ctx.batch_size
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            graph_bs = ctx.graph_bs
            self.graphs[graph_bs].replay()
            return self.model.compute_logits(self.decode_vars["outputs"][:bs])

    def postprocess(
        self,
        batch: ScheduledBatchs,
        logits: torch.Tensor,
        temperatures: Optional[torch.Tensor],
    ) -> list[int]:
        if self.rank == 0:
            sampled_tokens = self.sampler(logits, temperatures)

            # num_sampled_tokens = sampled_tokens.shape[0]
            sampled_token_ids = sampled_tokens
            if  self.use_async_scheduling:
                # assert sampled_token_ids.shape[-1] == 1

                # Cache the sampled tokens on the GPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                # if self.input_batch.prev_sampled_token_ids is None:
                #     output = None
                # else:
                #     output = self.input_batch.prev_sampled_token_ids

                self.input_batch.prev_sampled_token_ids = \
                    sampled_token_ids
                # print("self.input_batch.prev_sampled_token_ids in postprocess", self.input_batch.prev_sampled_token_ids)
                self.input_batch.prev_req_id_to_index = {
                    req_id: i
                    for i, req_id in enumerate(self.input_batch.req_ids)
                }
                # plus 1 to fit next iter seq len
                context_lens = [seq.num_tokens + 1 for seq in batch.seqs]
                # positions = np.array(context_lens, dtype=np.int64)

                self.input_batch.prev_position_ids = np.array(context_lens, dtype=np.int64)


            # output  = AsyncGPUModelRunnerOutput(
            #     sampled_token_ids=sampled_token_ids,
            #     # invalid_req_indices=invalid_req_indices,
            #     async_output_copy_stream=self.async_output_copy_stream,
            # )

            # .get_output()
            # self.handle_output(output)

            # token_ids = sampled_tokens.tolist()
            output = self.out_processor.update_and_ret(
                batch,
                sampled_tokens,
            )
        else:
            output = None
        # self.worker_response_mq.enqueue(result)
        return output

    def _update_states(self, batch: ScheduledBatchs):
        for req_id in self.input_batch.finished_req_ids:
            self.input_batch.remove_request(req_id)

        for request in batch.seqs:
            self.input_batch.add_request(request)
    
    def finish_req(self, batch: ScheduledBatchs):
        request_id = self.input_batch.req_ids
        for id in request_id:
            self.input_batch.finished_req_ids.add(id)
        # print("finish_req:", self.input_batch.finished_req_ids)


    @torch.inference_mode()
    def forward(self, batch: ScheduledBatchs, is_wramup = False) -> list[int]:
        # self._update_states(batch)
        input_ids, positions, temperatures = self.prepare_model(batch)
        logits = self.run_model(input_ids, positions, batch.is_prefill)
        reset_context()
        self.finish_req(batch)
        # self.postprocess(batch, logits, temperatures)
        # return self.async_get_output()
        return self.postprocess(batch, logits, temperatures)

    @torch.inference_mode()
    def capture_cudagraph(self):
        start_time = time.time()
        input_ids = self.decode_vars["input_ids"].gpu
        positions = self.decode_vars["positions"].gpu
        slot_mapping = self.decode_vars["slot_mapping"].gpu
        context_lens = self.decode_vars["context_lens"].gpu
        block_tables = self.decode_vars["block_tables"].gpu
        cu_seqlens_q = self.decode_vars["cu_seqlens_q"].gpu
        kv_indptr = self.decode_vars["kv_indptr"].gpu
        kv_indices = self.decode_vars["kv_indices"].gpu
        kv_last_page_lens = self.decode_vars["kv_last_page_lens"].gpu
        outputs = self.decode_vars["outputs"]

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

        self.graphs: dict[int, torch.cuda.CUDAGraph] = dict()
        self.graph_pool = None

        with graph_capture() as gc:
            capture_range = (
                tqdm.tqdm(self.graph_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.graph_bs
            )
            for bs in capture_range:
                if get_tensor_model_parallel_rank() == 0:
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
