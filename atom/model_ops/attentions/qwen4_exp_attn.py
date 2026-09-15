"""Attention backend for Qwen3.8-Flash-Next (`qwen4_exp`).

The model keeps the following caches per request:

  * paged K/V for the 12 full-attention (QSA) layers -- the ordinary pool;
  * a paged RAW index-key cache per QSA layer, so the indexer can mean-pool
    groups of `compress_ratio` keys across chunk boundaries;
  * a paged COMPRESSED index-key cache per QSA layer, one row per complete
    group, i.e. `block_size / compress_ratio` rows per block;
  * GDN conv + temporal state for the 36 linear-attention layers (inherited);
  * short-convolution and n-gram token windows for the single PLE layer.
    The dilated convolution reaches back
    `(ple_conv_kernel_size - 1) * ngram_size` tokens.

The two index-key caches ride the SAME block table as the main K/V pool, so
they need no separate block allocator -- only their own byte budget and their
own tensors. The PLE state shares the GDN state class for the same reason: one
per-request slot index, two backends' bytes.

The sparse GQA kernel reads main K/V in the plain
`[blocks, block_size, heads, head_dim]` layout for both prefill and decode.
The physical block size matches the scheduler's (`block_ratio == 1`), so each
block table entry directly addresses a page in both main and index-key caches.
"""

from dataclasses import dataclass

import numpy as np
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots
from atom.utils import CpuGpuBuffer

from .gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder
from .pool_layout.entry_arena import EntryField, LayerMajorArena, entry_bytes_for
from .pool_layout.sub_pool_spec import SubPoolSpec, page_pool, state_pool
from .token_layout.batch_ids import build_batch_ids

# The PLE convolution and n-gram windows have exactly the GDN
# recurrent state's lifetime and multiplicity, so it shares its index space.
PLE_STATE_SLOT_CLASS = STATE_SLOT_CLASS

# The QSA layers' row space. Named rather than derived from
# `layer_id // full_attention_interval`, because the row a layer holds is the
# order it was bound in: on a non-first PP stage the global layer numbers do
# not start at zero, and that arithmetic would index past the pool.
QSA_ROWS = "qsa"


@dataclass
class Qwen4ExpQSAMetadata:
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
class Qwen4ExpPLEMetadata:
    """Per-forward inputs for the single PLE layer."""

    query_start_loc: torch.Tensor  # [reqs + 1] int32
    ngram_state: torch.Tensor  # [slots, ngram_size - 1] int64
    state_indices_in: torch.Tensor  # [reqs] int32, slot the state is read from
    state_indices_out: torch.Tensor  # [reqs] int32, slot it is written to
    has_initial_state: torch.Tensor  # [reqs] bool
    conv_state: torch.Tensor  # [slots, channels, state_len] short-conv pool


class Qwen4ExpBackend(GDNAttentionBackend):
    @staticmethod
    def validate_config(config) -> None:
        """Check the metadata and state layouts implemented by this backend."""
        if config.speculative_config is not None:
            # PLE updates one committed convolution state per request; it has
            # no speculative-token rollback metadata.
            raise ValueError("Qwen PLE state does not support speculative decoding")
        if (
            config.enable_dp_attention
            or config.prefill_context_parallel_size > 1
            or config.decode_context_parallel_size > 1
        ):
            raise ValueError(
                "Qwen QSA requires unsplit request/page metadata; "
                "DP attention, PCP and DCP are not implemented"
            )
        if config.enable_tbo or config.enable_tbo_decode:
            # QSA/PLE metadata are not sliced by the shared ubatch dispatcher.
            raise ValueError("Qwen QSA/PLE metadata do not support TBO slicing")
        if config.kv_transfer_config:
            raise ValueError(
                "Qwen KV transfer/offload requires QSA side caches and PLE state"
            )
        if len(set(config.hf_config.ple_layer_ids or [])) > 1:
            raise ValueError("Qwen backend currently allocates state for one PLE layer")
        if config.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
            raise ValueError("Qwen QSA requires BF16 KV cache")
        ratio = int(config.hf_config.indexer_compress_ratio)
        if ratio <= 0 or config.kv_cache_block_size % ratio:
            raise ValueError(
                "QSA block size must be divisible by positive indexer_compress_ratio"
            )

    @staticmethod
    def attn_block_size(hf_config, scheduler_block_size: int) -> int:
        # QSA reads the scheduler's page IDs directly. This must be selected
        # before CommonAttentionBuilder sizes the block-table buffers.
        return scheduler_block_size

    @staticmethod
    def get_name() -> str:
        return "ROCM_QWEN4_EXP"

    @staticmethod
    def get_builder_cls() -> type["Qwen4ExpMetadataBuilder"]:
        return Qwen4ExpMetadataBuilder


class Qwen4ExpMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN hybrid plus the QSA side caches and the PLE state."""

    BACKEND = Qwen4ExpBackend

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        hf = model_runner.config.hf_config
        # Image/video requests give mRoPE three genuinely different position
        # rows, and then a compressed key's group position cannot be recomputed
        # arithmetically. Cache the per-token axes only when the engine can
        # actually be handed an image -- it costs ~1.4 GB of the paged budget.
        self.cache_rope_positions = (
            getattr(model_runner.config, "multimodal_config", None) is not None
        )
        self.ngram_context_len = int(hf.ngram_size) - 1 if hf.ple_layer_ids else 0
        eos = hf.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        max_tokens = self.max_num_batched_tokens
        i32 = {"dtype": torch.int32, "device": self.device}
        i64 = {"dtype": torch.int64, "device": self.device}
        self.model_runner.forward_vars["qsa_token_to_req"] = CpuGpuBuffer(
            max_tokens, **i32
        )
        self.model_runner.forward_vars["qsa_logical_positions"] = CpuGpuBuffer(
            max_tokens, **i64
        )
        # Written in place, never reallocated: a captured decode graph bakes in
        # the address it reads the compressed slots from.
        self.model_runner.forward_vars["qsa_compressed_slots"] = torch.empty(
            max_tokens, **i64
        )
        self.model_runner.forward_vars["ple_has_initial_state"] = CpuGpuBuffer(
            self.max_bs, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Geometry                                                            #
    # ------------------------------------------------------------------ #

    def _module_kinds(self, module) -> tuple:
        """A QSA layer takes a row of the QSA pool; everything else is the
        hybrid's (the GDN state slots, handled by the parent)."""
        if getattr(module, "is_qsa_attention", False):
            return (QSA_ROWS,)
        return super()._module_kinds(module)

    @property
    def _qsa_layers(self) -> int:
        """QSA layers this rank actually built, counted off the bind walk."""
        return self.row_counts().get(QSA_ROWS, 0)

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

    def _qsa_fields(self) -> list[EntryField]:
        """Single declaration used for sizing, allocation and transfer views."""
        hf = self.model_runner.config.hf_config
        layers, block = self._qsa_layers, self.block_size
        shape = (block, self.model_runner._get_num_kv_heads(), int(hf.head_dim))
        fields = [
            EntryField("k", layers, shape, torch.bfloat16),
            EntryField("v", layers, shape, torch.bfloat16),
            EntryField(
                "index_raw", layers, (block, 1, self._index_head_dim), torch.bfloat16
            ),
            EntryField(
                "index_compressed",
                layers,
                (block // self._compress_ratio, 1, self._index_head_dim),
                torch.bfloat16,
            ),
        ]
        if self.cache_rope_positions:
            fields.append(
                EntryField("rope_positions", layers, (block, 1, 3), torch.int64)
            )
        return fields

    def _page_bytes(self) -> int:
        return entry_bytes_for(self._qsa_fields())

    def paged_pool_bytes(self, blocks: int) -> int:
        return self._page_bytes() * blocks

    def _state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        # Honor the checkpoint's recurrent storage contract. The convolution
        # still stores the model dtype, as in the shared GDN backend.
        requested = getattr(
            self.model_runner.config.hf_config, "mamba_ssm_dtype", "float32"
        )
        types = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        if requested not in types:
            raise ValueError(f"unsupported mamba_ssm_dtype: {requested}")
        return self.model_runner.config.torch_dtype, types[requested]

    def _ple_state_spec(self) -> SubPoolSpec | None:
        """Both PLE windows share one state slot; absent when there is no PLE."""
        hf = self.model_runner.config.hf_config
        if not getattr(hf, "ple_layer_ids", None):
            return None
        state_len, channels = self._ple_state_shape()
        entry = state_len * channels * self.model_runner.config.torch_dtype.itemsize
        entry += (int(hf.ngram_size) - 1) * torch.int64.itemsize
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

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Back every paged field with the runner's shared allocation."""
        self.num_blocks = blocks
        self.qsa_arena = LayerMajorArena(
            self._qsa_fields(), blocks, self.device, buf=buf
        )
        return {}

    def release_kv_pools(self) -> None:
        for module in self.pool_rows.get(QSA_ROWS, {}):
            module.bind_caches(None, None, None, None, None)
        self.qsa_arena = None
        super().release_kv_pools()

    def get_kv_transfer_tensors(self):
        # Expose all regions for diagnostics/lifecycle consumers. External
        # transfer is rejected by config until the connector owns this layout.
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        arena = getattr(self, "qsa_arena", None)
        if arena is None:
            return None

        def region(tensor, role):
            return KVTransferRegion(
                base_addr=tensor.data_ptr(),
                total_bytes=tensor.numel() * tensor.element_size(),
                unit_bytes=tensor.stride(0) * tensor.element_size(),
                semantic_role=role,
            )

        blocks = [
            region(tensor, f"qsa.{field.name}.{layer}")
            for field in arena.fields
            for layer, tensor in enumerate(arena.view(field.name))
        ]
        runner = self.model_runner
        slots = [
            region(tensor, f"gdn.{name}.{layer}")
            for name in ("mamba_k_cache", "mamba_v_cache")
            for layer, tensor in enumerate(getattr(runner, name, ()))
        ]
        for name, role in (
            ("ple_conv_state", "ple.conv"),
            ("ple_ngram_state", "ple.ngram"),
        ):
            ple = getattr(runner, name, None)
            if ple is not None:
                slots.append(region(ple, role))
        return KVTransferTensors(block_regions=blocks, slot_regions=slots)

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
        caches["ple_ngram_state"] = torch.full(
            (entries.get(PLE_STATE_SLOT_CLASS, 0), self.ngram_context_len),
            self.eos_token_id,
            dtype=torch.int64,
            device="cuda",
        )
        return caches

    def relocate_state_slots(self, pairs) -> None:
        super().relocate_state_slots(pairs)
        destinations, sources = [], []
        # The shared pool relocates individual slots, never request groups.
        for name in ("ple_conv_state", "ple_ngram_state"):
            state = getattr(self.model_runner, name, None)
            if state is not None:
                for src, dst in pairs:
                    destinations.append(state[dst : dst + 1])
                    sources.append(state[src : src + 1])
        if destinations:
            torch._foreach_copy_(destinations, sources)

    def build_kv_cache_tensor(self, module):
        """Bind the three caches a QSA layer owns; defer everything else."""
        if not getattr(module, "is_qsa_attention", False):
            return super().build_kv_cache_tensor(module)

        from atom.config import KVCacheTensor

        row = self.pool_rows[QSA_ROWS][module]
        arena = self.qsa_arena
        key, value = arena.view("k")[row], arena.view("v")[row]
        module.bind_caches(
            key,
            value,
            arena.view("index_raw")[row],
            arena.view("index_compressed")[row],
            arena.view("rope_positions")[row] if self.cache_rope_positions else None,
        )
        return KVCacheTensor(
            layer_num=module.layer_num,
            k_cache=key,
            v_cache=value,
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
    ) -> Qwen4ExpQSAMetadata:
        """Token->request, logical positions, and the compressed slot mapping.

        `mapped_tokens` is how many of `num_tokens` are real; the rest are
        CUDA-graph padding and are marked `-1` so no cache row is touched.
        """
        token_to_req = self.model_runner.forward_vars["qsa_token_to_req"].np
        logical = self.model_runner.forward_vars["qsa_logical_positions"].np
        token_to_req[:num_tokens] = -1
        logical[:num_tokens] = -1
        if mapped_tokens:
            # Decode's running shape includes graph padding. Only scheduled
            # tokens own a request; the common token layout owns the mapping.
            counts = tokens_per_req.copy()
            remaining = mapped_tokens
            for req in range(len(counts)):
                counts[req] = min(int(counts[req]), remaining)
                remaining -= int(counts[req])
            build_batch_ids(counts, pad_to=num_tokens, out=token_to_req)
            logical[:mapped_tokens] = self.model_runner.forward_vars["positions"].np[
                :mapped_tokens
            ]
        token_to_req_gpu = self.model_runner.forward_vars[
            "qsa_token_to_req"
        ].copy_to_gpu(num_tokens)
        logical_gpu = self.model_runner.forward_vars[
            "qsa_logical_positions"
        ].copy_to_gpu(num_tokens)

        block_tables = attn_metadata.block_tables
        slot_mapping = attn_metadata.slot_mapping[:num_tokens]
        persistent_slots = self.model_runner.forward_vars["qsa_compressed_slots"][
            :num_tokens
        ]
        qsa_compressed_slots(
            slot_mapping, logical_gpu, self._compress_ratio, persistent_slots
        )

        return Qwen4ExpQSAMetadata(
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

    @staticmethod
    def _ple_state_slots(gdn) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The `(in, out)` GDN state slots PLE shares, or None if it has none.

        Metadata without a separate read-side tensor denotes an in-place
        update; otherwise PLE must preserve the supplied fork source.
        """
        if gdn is None or gdn.non_spec_state_indices_tensor is None:
            return None
        source = gdn.non_spec_state_indices_in_tensor
        if source is None:
            source = gdn.non_spec_state_indices_tensor
        return source, gdn.non_spec_state_indices_tensor

    def _build_ple_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata,
        num_reqs: int,
        is_prefill: bool,
    ) -> Qwen4ExpPLEMetadata | None:
        """Address both PLE windows using the existing GDN state-slot metadata."""
        if not self.ngram_context_len:
            return None
        slots = self._ple_state_slots(attn_metadata.gdn_metadata)
        if slots is None:
            return None
        state_indices_in, state_indices_out = slots
        conv_state = getattr(self.model_runner, "ple_conv_state", None)
        if conv_state is None:
            return None

        # A cold first chunk must not fold in whatever the recycled state slot
        # still held; anything with cached tokens continues its own window.
        has_initial = self.model_runner.forward_vars["ple_has_initial_state"].np
        if is_prefill:
            has_initial[:num_reqs] = (
                np.asarray(batch.num_cached_tokens[:num_reqs], dtype=np.int64) > 0
            )
        else:
            has_initial[:num_reqs] = True

        return Qwen4ExpPLEMetadata(
            query_start_loc=attn_metadata.cu_seqlens_q[: num_reqs + 1],
            ngram_state=self.model_runner.ple_ngram_state,
            state_indices_in=state_indices_in[:num_reqs],
            state_indices_out=state_indices_out[:num_reqs],
            has_initial_state=self.model_runner.forward_vars[
                "ple_has_initial_state"
            ].copy_to_gpu(num_reqs),
            conv_state=conv_state,
        )

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = super().prepare_prefill(batch, running_bs)
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

        query_lens = np.asarray(batch.num_scheduled_tokens[:num_reqs], dtype=np.int64)
        attn_metadata.qsa_metadata = self._build_qsa_metadata(
            attn_metadata, num_reqs, num_tokens, query_lens, num_tokens
        )
        attn_metadata.ple_metadata = self._build_ple_metadata(
            batch,
            attn_metadata,
            num_reqs,
            is_prefill=True,
        )
        return attn_metadata, positions

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        attn_metadata, positions = super().prepare_decode(
            batch, running_bs, running_tokens, max_seqlen_q
        )
        bs = running_bs
        query_len = attn_metadata.max_seqlen_q
        num_tokens = running_tokens
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
        if query_len != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next PLE decode assumes one token per request; "
                "speculative decode is not wired up yet"
            )
        attn_metadata.ple_metadata = self._build_ple_metadata(
            batch,
            attn_metadata,
            bs,
            is_prefill=False,
        )
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int):
        """Decode-graph metadata pointing at the same buffers replay writes.

        Every tensor here is a slice of a persistent buffer that
        `prepare_decode` later fills in place, so the addresses baked into the
        captured graph stay valid. The QSA selection width is pinned to the
        engine's context bound, which holds for every replay.
        """
        # Capture has no scheduled state slots. Give its synthetic requests
        # distinct slots so the in-place windows cannot race on slot zero.
        # Normal metadata preparation overwrites these same buffers on replay.
        for indices in (
            self.non_spec_state_indices_tensor,
            self.non_spec_state_indices_in_tensor,
        ):
            indices.np[:bs] = np.arange(bs, dtype=np.int32)
            indices.copy_to_gpu(bs)
        attn_metadata, context = super().build_for_cudagraph_capture(bs)
        runner = self.model_runner
        num_tokens = bs * int(attn_metadata.max_seqlen_q)

        attn_metadata.qsa_metadata = self._build_qsa_metadata(
            attn_metadata,
            bs,
            num_tokens,
            np.full(bs, int(attn_metadata.max_seqlen_q), dtype=np.int64),
            num_tokens,
            max_seq_len=runner.config.max_model_len,
        )

        slots = self._ple_state_slots(attn_metadata.gdn_metadata)
        conv_state = getattr(runner, "ple_conv_state", None)
        if not self.ngram_context_len or conv_state is None or slots is None:
            attn_metadata.ple_metadata = None
            return attn_metadata, context
        state_indices_in, state_indices_out = slots
        self.model_runner.forward_vars["ple_has_initial_state"].np[:bs] = True
        attn_metadata.ple_metadata = Qwen4ExpPLEMetadata(
            query_start_loc=attn_metadata.cu_seqlens_q[: bs + 1],
            ngram_state=runner.ple_ngram_state,
            state_indices_in=state_indices_in[:bs],
            state_indices_out=state_indices_out[:bs],
            has_initial_state=self.model_runner.forward_vars[
                "ple_has_initial_state"
            ].copy_to_gpu(bs),
            conv_state=conv_state,
        )
        return attn_metadata, context
