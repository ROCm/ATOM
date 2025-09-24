import logging
import os
import pickle
import signal
import weakref
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, dtypes, init_dist_env
from aiter.dist.parallel_state import get_tensor_model_parallel_rank, graph_capture

from atom.config import Config
from atom.model_engine.scheduler import ScheduledBatchs
from atom.model_engine.sequence import Sequence
from atom.model_loader.loader import load_model
from atom.model_ops.sampler import Sampler
from atom.models.llama import LlamaForCausalLM
from atom.models.mixtral import MixtralForCausalLM
from atom.models.qwen3 import Qwen3ForCausalLM
from atom.utils.context import get_context, reset_context, set_context

logger = logging.getLogger("atom")

suppot_model_arch_dict = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize profiler for this rank
        self.profiler = None
        self.profiler_dir = None
        if config.torch_profiler_dir is not None:
            # Create rank-specific profiler directory
            self.profiler_dir = os.path.join(config.torch_profiler_dir, f"rank_{rank}")
            os.makedirs(self.profiler_dir, exist_ok=True)

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        init_dist_env(self.world_size, rankID=rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.graph_bs = [0]  # for eager fallback

        self.model = suppot_model_arch_dict[hf_config.architectures[0]](config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.allocate_decode_vars()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        if self.config.compilation_config.level == 3:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="atom", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="atom")
                self.loop()

        def signal_handler(signum, frame):
            raise SystemExit(
                f"Rank{self.rank}/{self.world_size}: received signal {signum}, exiting..."
            )

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info(f"Initializing ModelRunner for rank {rank}")
        self._finalizer = weakref.finalize(self, self.exit)

    def exit(self, msg=None):
        logger.info(f"Exiting ModelRunner for rank {self.rank}/{self.world_size} {msg}")
        if dist.is_initialized():
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None
        torch.cuda.synchronize()
        destroy_dist_env()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

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
        dummy_batch = ScheduledBatchs(seqs, True, False)
        self.run(dummy_batch)
        torch.cuda.empty_cache()

    def allocate_decode_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.hf_config.torch_dtype
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.decode_vars = {
            "input_ids": torch.zeros(max_bs, dtype=torch.int64),
            "positions": torch.zeros(max_bs, dtype=torch.int64),
            "slot_mapping": torch.zeros(max_bs, dtype=torch.int64),
            "context_lens": torch.zeros(max_bs, dtype=torch.int32),
            "block_tables": torch.zeros(max_bs, max_num_blocks, dtype=torch.int32),
            "outputs": torch.empty(max_bs, hidden_size, dtype=hidden_type),
        }

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0, f"need at least {block_bytes} KV cache"
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

                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

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
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        min_seqlen_q = 0
        dropout_p = 0.0
        set_context(
            True,
            len(scheduled_batchs.seqs),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            dropout_p=dropout_p,
        )
        return input_ids, positions

    def prepare_decode(self, scheduled_batchs: ScheduledBatchs):
        bs = len(scheduled_batchs.seqs)
        graph_bs = (
            bs if self.enforce_eager else next(x for x in self.graph_bs if x >= bs)
        )
        assert graph_bs >= bs, f"current decode {bs=} > max graph_bs{graph_bs}"
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 1
        max_seqlen_k = 0
        min_seqlen_q = 0
        dropout_p = 0.0

        for seq in scheduled_batchs.seqs:
            current_seq_len = len(seq)
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)

            cu_seqlens_k.append(cu_seqlens_k[-1] + current_seq_len)
            max_seqlen_k = max(current_seq_len, max_seqlen_k)

            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, pin_memory=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True)
        block_tables = self.prepare_block_tables(scheduled_batchs.seqs)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)

        decode_vars = self.decode_vars
        decode_vars["slot_mapping"][bs : graph_bs + 1] = -1
        decode_vars["slot_mapping"][:bs].copy_(slot_mapping, non_blocking=True)
        decode_vars["input_ids"][:bs].copy_(input_ids, non_blocking=True)
        decode_vars["positions"][:bs].copy_(positions, non_blocking=True)
        decode_vars["context_lens"][:bs].copy_(context_lens, non_blocking=True)
        decode_vars["block_tables"][:bs, : block_tables.size(1)].copy_(
            block_tables, non_blocking=True
        )
        set_context(
            False,
            batch_size=bs,
            graph_bs=graph_bs,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=decode_vars["slot_mapping"][:bs],
            context_lens=decode_vars["context_lens"][:bs],
            block_tables=decode_vars["block_tables"][:bs, : block_tables.size(1)],
            dropout_p=dropout_p,
        )

        return decode_vars["input_ids"][:bs], decode_vars["positions"][:bs]

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def prepare_model(self, scheduled_batchs: ScheduledBatchs):
        seqs = scheduled_batchs.seqs
        is_prefill = scheduled_batchs.is_prefill
        prepare_func = self.prepare_prefill if is_prefill else self.prepare_decode
        input_ids, positions = prepare_func(scheduled_batchs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        return input_ids, positions, temperatures

    @torch.inference_mode()
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

    def run(self, scheduled_batchs: ScheduledBatchs) -> list[int]:
        input_ids, positions, temperatures = self.prepare_model(scheduled_batchs)
        logits = self.run_model(input_ids, positions, scheduled_batchs.is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        input_ids = self.decode_vars["input_ids"]
        positions = self.decode_vars["positions"]
        slot_mapping = self.decode_vars["slot_mapping"]
        context_lens = self.decode_vars["context_lens"]
        block_tables = self.decode_vars["block_tables"]
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
