"""Attention backend for Qwen3.8-Flash-Next (`qwen3_8_flash_next`).

Qwen3.8-Flash-Next is a hybrid whose cache needs are unusual even by ATOM's standards.
Per request it holds:

  * paged K/V for the 12 full-attention (QSA) layers -- the ordinary pool;
  * a paged RAW index-key cache per QSA layer, so the indexer can mean-pool
    groups of `compress_ratio` keys across chunk boundaries;
  * a paged COMPRESSED index-key cache per QSA layer, one row per complete
    group, i.e. `block_size / compress_ratio` rows per block;
  * GDN conv + temporal state for the 36 linear-attention layers (inherited);
  * a short-convolution state for the single PLE layer, whose dilated kernel
    reaches back `(ple_conv_kernel_size - 1) * ngram_size` tokens.

The two index-key caches ride the SAME block table as the main K/V pool, so
they need no separate block allocator -- only their own byte budget and their
own tensors. The PLE state shares the GDN state class for the same reason: one
per-request slot index, two backends' bytes.

Unlike the other MHA backends here, the main K/V is kept in the plain
`[blocks, block_size, heads, head_dim]` layout rather than AITER's pre-shuffled
one. QSA never calls AITER paged attention -- every token, prefill or decode,
goes through the sparse GQA kernel -- so the shuffle would buy nothing and the
sparse kernels want the natural layout. For the same reason the builder pins
its physical block size to the scheduler's, making `block_ratio == 1` so a
block table entry is directly a page index into every one of these caches.
"""

from dataclasses import dataclass

import numpy as np
import torch
from aiter import dtypes

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils import CpuGpuBuffer

from .gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder
from .sub_pool_spec import SubPoolSpec, page_pool, state_pool

# The PLE short-conv window is per-request state with exactly the GDN
# recurrent state's lifetime and multiplicity, so it shares its index space.
PLE_STATE_SLOT_CLASS = STATE_SLOT_CLASS


@dataclass
class Qwen3_8FlashNextQSAMetadata:
    """Per-forward addressing for one QSA layer's three paged caches."""

    block_tables: torch.Tensor  # [reqs, pages] int32, scheduler block ids
    slot_mapping: torch.Tensor  # [tokens] int64, flat row in the token caches
    compressed_slot_mapping: torch.Tensor  # [tokens] int64, -1 off group ends
    token_to_req: torch.Tensor  # [tokens] int32
    logical_positions: torch.Tensor  # [tokens] int64, -1 for padded rows
    seq_lens: torch.Tensor  # [reqs] int32
    # Host-side longest sequence in the batch. Bounds the scored/selected width
    # so a short request does not pay for the whole engine context.
    max_seq_len: int


@dataclass
class Qwen3_8FlashNextPLEMetadata:
    """Per-forward inputs for the single PLE layer."""

    query_start_loc: torch.Tensor  # [reqs + 1] int32
    ngram_context: torch.Tensor  # [reqs, ngram_size - 1] int64
    state_indices_in: torch.Tensor  # [reqs] int32, slot the state is read from
    state_indices_out: torch.Tensor  # [reqs] int32, slot it is written to
    has_initial_state: torch.Tensor  # [reqs] bool
    conv_state: torch.Tensor  # [slots, channels, state_len] short-conv pool
    num_reqs: int
    is_prefill: bool
    # Host-side packing width for the prefill convolution; taking it from the
    # device would sync every step.
    max_query_len: int


class Qwen3_8FlashNextBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_QWEN4_EXP"

    @staticmethod
    def get_builder_cls() -> type["Qwen3_8FlashNextMetadataBuilder"]:
        return Qwen3_8FlashNextMetadataBuilder


class Qwen3_8FlashNextMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN hybrid plus the QSA side caches and the PLE conv state."""

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        # Pin the physical page to the scheduler block. Every QSA cache is
        # addressed straight off `block_tables`, and AITER's shuffled-layout
        # paged attention -- the only reason a smaller physical page exists --
        # is never called on this model.
        self.block_size = model_runner.block_size
        self.block_ratio = 1
        self.max_num_blocks_per_seq = (
            model_runner.config.max_model_len + self.block_size - 1
        ) // self.block_size

        hf = model_runner.config.hf_config
        if self.block_size % self._compress_ratio:
            raise ValueError(
                f"--block-size must be divisible by indexer_compress_ratio "
                f"({self._compress_ratio}); got {self.block_size}"
            )
        # Image/video requests give mRoPE three genuinely different position
        # rows, and then a compressed key's group position cannot be recomputed
        # arithmetically. Cache the per-token axes only when the engine can
        # actually be handed an image -- it costs ~1.4 GB of the paged budget.
        self.cache_rope_positions = (
            getattr(model_runner.config, "multimodal_config", None) is not None
        )
        self.ngram_context_len = max(int(hf.ngram_size) - 1, 0)
        eos = hf.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        max_tokens = self.max_num_batched_tokens
        i32 = {"dtype": torch.int32, "device": self.device}
        i64 = {"dtype": torch.int64, "device": self.device}
        self._token_to_req = CpuGpuBuffer(max_tokens, **i32)
        self._logical_positions = CpuGpuBuffer(max_tokens, **i64)
        # Written in place, never reallocated: a captured decode graph bakes in
        # the address it reads the compressed slots from.
        self._compressed_slots = torch.empty(max_tokens, **i64)
        self._ngram_context = CpuGpuBuffer(
            self.max_bs, max(self.ngram_context_len, 1), **i64
        )
        self._has_initial_state = CpuGpuBuffer(
            self.max_bs, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Geometry                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _qsa_layers(self) -> int:
        return self.model_runner.num_full_attn

    @property
    def _index_head_dim(self) -> int:
        return int(self.model_runner.config.hf_config.indexer_head_dim)

    @property
    def _compress_ratio(self) -> int:
        return int(self.model_runner.config.hf_config.indexer_compress_ratio)

    def _ple_state_shape(self) -> tuple[int, int]:
        hf = self.model_runner.config.hf_config
        state_len = (int(hf.ple_conv_kernel_size) - 1) * int(hf.ngram_size)
        channels = int(hf.hidden_size) * int(hf.hc_count)
        # The PLE convolution is NOT tensor-parallel: every rank runs the full
        # hc_count * hidden width, matching the reference's tp_world_size=1.
        return state_len + self.num_spec, channels

    # ------------------------------------------------------------------ #
    # Pool sizing and allocation                                          #
    # ------------------------------------------------------------------ #

    def _page_bytes(self) -> int:
        """Per-block bytes: main K/V plus both index caches, all QSA layers."""
        runner = self.model_runner
        block = self.block_size
        item = dtypes.bf16.itemsize
        head_dim = int(runner.config.hf_config.head_dim)
        num_kv_heads = runner._get_num_kv_heads()

        main = 2 * self._qsa_layers * block * num_kv_heads * head_dim * item
        raw = self._qsa_layers * block * self._index_head_dim * item
        compressed = (
            self._qsa_layers
            * (block // self._compress_ratio)
            * self._index_head_dim
            * item
        )
        positions = (
            self._qsa_layers * block * 3 * torch.int64.itemsize
            if self.cache_rope_positions
            else 0
        )
        return main + raw + compressed + positions

    def _ple_state_spec(self) -> SubPoolSpec | None:
        """Per-request PLE short-conv window, or None when the model has no PLE."""
        hf = self.model_runner.config.hf_config
        if not getattr(hf, "ple_layer_ids", None):
            return None
        state_len, channels = self._ple_state_shape()
        entry = state_len * channels * self.model_runner.config.torch_dtype.itemsize
        return state_pool(
            PLE_STATE_SLOT_CLASS, entry, entries_per_req=1 + self.num_spec
        )

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        # The GDN recurrent state carries over unchanged; the paged pool does
        # not, because QSA's layout and side caches replace the MHA one.
        specs = [page_pool(self._page_bytes()), self.state_spec()]
        ple_spec = self._ple_state_spec()
        if ple_spec is not None:
            specs.append(ple_spec)
        return specs

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        if num_draft_layers:
            raise NotImplementedError("Qwen3.8-Flash-Next has no speculative draft path yet")
        runner = self.model_runner
        head_dim = int(runner.config.hf_config.head_dim)
        blocks = runner.num_physical_kvcache_blocks
        block = self.block_size
        return {
            "kv_cache": torch.zeros(
                2,
                self._qsa_layers,
                blocks,
                block,
                num_kv_heads,
                head_dim,
                dtype=dtypes.bf16,
                device="cuda",
            ),
            "qsa_raw_key_cache": torch.zeros(
                self._qsa_layers,
                blocks,
                block,
                1,
                self._index_head_dim,
                dtype=dtypes.bf16,
                device="cuda",
            ),
            "qsa_compressed_key_cache": torch.zeros(
                self._qsa_layers,
                blocks,
                block // self._compress_ratio,
                1,
                self._index_head_dim,
                dtype=dtypes.bf16,
                device="cuda",
            ),
            "qsa_rope_position_cache": (
                torch.zeros(
                    self._qsa_layers,
                    blocks,
                    block,
                    1,
                    3,
                    dtype=torch.int64,
                    device="cuda",
                )
                if self.cache_rope_positions
                else None
            ),
        }

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict[str, object]:
        caches = super().allocate_per_req_cache(entries)
        hf = self.model_runner.config.hf_config
        if not getattr(hf, "ple_layer_ids", None):
            return caches
        state_len, channels = self._ple_state_shape()
        caches["ple_conv_state"] = torch.zeros(
            entries.get(PLE_STATE_SLOT_CLASS, 0),
            channels,
            state_len,
            dtype=self.model_runner.config.torch_dtype,
            device="cuda",
        )
        return caches

    def relocate_state_slots(self, pairs) -> None:
        super().relocate_state_slots(pairs)
        state = getattr(self.model_runner, "ple_conv_state", None)
        if state is None:
            return
        span = 1 + self.num_spec
        destinations, sources = [], []
        for src_group, dst_group in pairs:
            src, dst = src_group * span, dst_group * span
            destinations.append(state[dst : dst + span])
            sources.append(state[src : src + span])
        if destinations:
            torch._foreach_copy_(destinations, sources)

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind the three caches a QSA layer owns; defer everything else."""
        if not getattr(module, "is_qsa_attention", False):
            return super().build_kv_cache_tensor(layer_id, module)

        from atom.config import KVCacheTensor

        runner = self.model_runner
        qsa_idx = layer_id // runner.full_attention_interval
        position_cache = getattr(runner, "qsa_rope_position_cache", None)
        module.bind_caches(
            runner.kv_cache[0, qsa_idx],
            runner.kv_cache[1, qsa_idx],
            runner.qsa_raw_key_cache[qsa_idx],
            runner.qsa_compressed_key_cache[qsa_idx],
            None if position_cache is None else position_cache[qsa_idx],
        )
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=runner.kv_cache[0, qsa_idx],
            v_cache=runner.kv_cache[1, qsa_idx],
            k_scale=None,
            v_scale=None,
        )

    # ------------------------------------------------------------------ #
    # Per-forward metadata                                                #
    # ------------------------------------------------------------------ #

    def _build_qsa_metadata(
        self,
        attn_metadata,
        num_reqs: int,
        num_tokens: int,
        tokens_per_req: np.ndarray,
        mapped_tokens: int,
        max_seq_len: int | None = None,
    ) -> Qwen3_8FlashNextQSAMetadata:
        """Token->request, logical positions, and the compressed slot mapping.

        `mapped_tokens` is how many of `num_tokens` are real; the rest are
        CUDA-graph padding and are marked `-1` so no cache row is touched.
        """
        token_to_req = self._token_to_req.np
        logical = self._logical_positions.np
        token_to_req[:num_tokens] = 0
        logical[:num_tokens] = -1
        if mapped_tokens:
            token_to_req[:mapped_tokens] = np.repeat(
                np.arange(num_reqs, dtype=np.int32), tokens_per_req
            )[:mapped_tokens]
            logical[:mapped_tokens] = self.model_runner.forward_vars["positions"].np[
                :mapped_tokens
            ]
        token_to_req_gpu = self._token_to_req.copy_to_gpu(num_tokens)
        logical_gpu = self._logical_positions.copy_to_gpu(num_tokens)

        block_tables = attn_metadata.block_tables
        slot_mapping = attn_metadata.slot_mapping[:num_tokens]
        storage_block = self.block_size // self._compress_ratio

        # One compressed row per COMPLETE group. `logical // ratio // storage`
        # is the same logical block the token itself lives in, so the main
        # block table addresses this cache unchanged.
        compressed_pos = torch.div(
            logical_gpu.clamp_min(0), self._compress_ratio, rounding_mode="floor"
        )
        logical_block = torch.div(
            compressed_pos, storage_block, rounding_mode="floor"
        ).clamp_(0, max(block_tables.shape[1] - 1, 0))
        requests = token_to_req_gpu.long().clamp_(0, max(block_tables.shape[0] - 1, 0))
        physical = block_tables[requests, logical_block].long()
        compressed_slots = physical * storage_block + compressed_pos.remainder(
            storage_block
        )
        closes_group = (logical_gpu >= 0) & (
            (logical_gpu + 1).remainder(self._compress_ratio) == 0
        )
        compressed_slots = torch.where(
            closes_group & (slot_mapping >= 0) & (physical >= 0),
            compressed_slots,
            torch.full_like(compressed_slots, -1),
        )
        persistent_slots = self._compressed_slots[:num_tokens]
        persistent_slots.copy_(compressed_slots)

        return Qwen3_8FlashNextQSAMetadata(
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            compressed_slot_mapping=persistent_slots,
            token_to_req=token_to_req_gpu,
            logical_positions=logical_gpu,
            seq_lens=attn_metadata.context_lens[:num_reqs],
            max_seq_len=(
                int(attn_metadata.max_seqlen_k)
                if max_seq_len is None
                else int(max_seq_len)
            ),
        )

    def _build_ple_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata,
        num_reqs: int,
        chunk_starts: np.ndarray,
        is_prefill: bool,
        max_query_len: int,
    ) -> Qwen3_8FlashNextPLEMetadata | None:
        """n-gram context tokens plus the short-conv state slots."""
        if not self.ngram_context_len:
            return None
        gdn = attn_metadata.gdn_metadata
        if gdn is None or gdn.non_spec_state_indices_tensor is None:
            return None
        conv_state = getattr(self.model_runner, "ple_conv_state", None)
        if conv_state is None:
            return None

        context = self._ngram_context.np
        context[:num_reqs] = self.eos_token_id
        token_ids = getattr(batch, "seq_token_ids", None)
        if token_ids is not None:
            for req in range(min(num_reqs, len(token_ids))):
                start = int(chunk_starts[req])
                ids = token_ids[req]
                for offset in range(self.ngram_context_len):
                    position = start - self.ngram_context_len + offset
                    if position >= 0:
                        context[req, offset] = ids[position]

        # A cold first chunk must not fold in whatever the recycled state slot
        # still held; anything with cached tokens continues its own window.
        has_initial = self._has_initial_state.np
        if is_prefill:
            has_initial[:num_reqs] = np.asarray(
                batch.num_cached_tokens[:num_reqs], dtype=np.int64
            ) > 0
        else:
            has_initial[:num_reqs] = True

        return Qwen3_8FlashNextPLEMetadata(
            query_start_loc=attn_metadata.cu_seqlens_q[: num_reqs + 1],
            ngram_context=self._ngram_context.copy_to_gpu(num_reqs),
            state_indices_in=(
                gdn.non_spec_state_indices_in_tensor
                if gdn.non_spec_state_indices_in_tensor is not None
                else gdn.non_spec_state_indices_tensor
            )[:num_reqs],
            state_indices_out=gdn.non_spec_state_indices_tensor[:num_reqs],
            has_initial_state=self._has_initial_state.copy_to_gpu(num_reqs),
            conv_state=conv_state,
            num_reqs=num_reqs,
            is_prefill=is_prefill,
            max_query_len=max_query_len,
        )

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)
        num_reqs = batch.total_seqs_num_prefill
        num_tokens = batch.total_tokens_num_prefill
        # QSA reads the compressed cache through the block table on every
        # forward, so unlike dense prefill it cannot wait for `has_cached`.
        if attn_metadata.block_tables is None and batch.block_tables:
            self.prepare_block_tables(batch)
            attn_metadata.block_tables = self.model_runner.forward_vars[
                "block_tables"
            ].copy_to_gpu(num_reqs)
        if attn_metadata.block_tables is None:
            attn_metadata.qsa_metadata = None
            attn_metadata.ple_metadata = None
            return attn_metadata, positions

        query_lens = np.asarray(
            batch.num_scheduled_tokens[:num_reqs], dtype=np.int64
        )
        attn_metadata.qsa_metadata = self._build_qsa_metadata(
            attn_metadata, num_reqs, num_tokens, query_lens, num_tokens
        )
        attn_metadata.ple_metadata = self._build_ple_metadata(
            batch,
            attn_metadata,
            num_reqs,
            np.asarray(batch.num_cached_tokens[:num_reqs], dtype=np.int64),
            is_prefill=True,
            max_query_len=int(attn_metadata.max_seqlen_q),
        )
        return attn_metadata, positions

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        attn_metadata, positions = super().prepare_decode(batch, bs)
        scheduled_bs = batch.total_seqs_num_decode
        query_len = attn_metadata.max_seqlen_q
        num_tokens = bs * query_len
        per_req = np.full(bs, query_len, dtype=np.int64)
        attn_metadata.qsa_metadata = self._build_qsa_metadata(
            attn_metadata,
            bs,
            num_tokens,
            per_req,
            batch.total_tokens_num_decode,
            # A decode graph is captured once and replayed at every sequence
            # length, so the QSA selection width has to be a constant. The
            # engine's context bound is the only one that holds for every
            # replay; prefill still narrows it to the batch it actually sees.
            max_seq_len=self.model_runner.config.max_model_len,
        )
        context_lens = np.asarray(batch.context_lens[:scheduled_bs], dtype=np.int64)
        chunk_starts = np.zeros(bs, dtype=np.int64)
        chunk_starts[:scheduled_bs] = context_lens - query_len
        if query_len != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next PLE decode assumes one token per request; "
                "speculative decode is not wired up yet"
            )
        attn_metadata.ple_metadata = self._build_ple_metadata(
            batch,
            attn_metadata,
            bs,
            chunk_starts,
            is_prefill=False,
            max_query_len=query_len,
        )
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int):
        """Decode-graph metadata pointing at the same buffers replay writes.

        Every tensor here is a slice of a persistent buffer that
        `prepare_decode` later fills in place, so the addresses baked into the
        captured graph stay valid. The two host-side widths that reach a
        Triton `constexpr` -- the QSA selection width and the PLE packing
        width -- are pinned to values that hold for every replay.
        """
        attn_metadata, context = super().build_for_cudagraph_capture(bs)
        runner = self.model_runner
        var = runner.forward_vars
        num_tokens = bs * int(attn_metadata.max_seqlen_q)

        self._token_to_req.np[:num_tokens] = np.repeat(
            np.arange(bs, dtype=np.int32), int(attn_metadata.max_seqlen_q)
        )
        self._logical_positions.np[:num_tokens] = var["positions"].np[:num_tokens]
        attn_metadata.qsa_metadata = self._build_qsa_metadata(
            attn_metadata,
            bs,
            num_tokens,
            np.full(bs, int(attn_metadata.max_seqlen_q), dtype=np.int64),
            num_tokens,
            max_seq_len=runner.config.max_model_len,
        )

        gdn = attn_metadata.gdn_metadata
        conv_state = getattr(runner, "ple_conv_state", None)
        if not self.ngram_context_len or conv_state is None or gdn is None:
            attn_metadata.ple_metadata = None
            return attn_metadata, context
        self._ngram_context.np[:bs] = self.eos_token_id
        self._has_initial_state.np[:bs] = True
        attn_metadata.ple_metadata = Qwen3_8FlashNextPLEMetadata(
            query_start_loc=attn_metadata.cu_seqlens_q[: bs + 1],
            ngram_context=self._ngram_context.copy_to_gpu(bs),
            state_indices_in=gdn.non_spec_state_indices_in_tensor[:bs],
            state_indices_out=gdn.non_spec_state_indices_tensor[:bs],
            has_initial_state=self._has_initial_state.copy_to_gpu(bs),
            conv_state=conv_state,
            num_reqs=bs,
            is_prefill=False,
            max_query_len=int(attn_metadata.max_seqlen_q),
        )
        return attn_metadata, context
