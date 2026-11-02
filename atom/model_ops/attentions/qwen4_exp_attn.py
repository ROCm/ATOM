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

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import NamedTuple

import numpy as np
import torch

from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_cache_formats,
    qsa_cache_row_layout,
    qsa_compressed_slots,
    qsa_draft_decode_metadata,
)
from atom.utils import CpuGpuBuffer, envs
from atom.utils.block_tables import block_table_state

from .gdn_attn import (
    LINEAR_STATE_ROWS,
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
    paged_checkpoints_deployable,
)
from .pool_layout.entry_arena import (
    EntryField,
    LayerMajorArena,
    entry_bytes_for,
    published_bytes_for,
)
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
# MTP draft QSA layers that keep BF16 caches while the target's are quantized
# (`ATOM_QSA_DRAFT_KV_BF16=1`): their own rows, their own `*_draft` fields.
QSA_DRAFT_ROWS = "qsa_draft"


# The QSA field rules. The arena, the layers binding its rows, the PAGE
# namespace and `_check_kv_transfer` all read these.


def qsa_caches_quantized(config) -> bool:
    return qsa_cache_formats(config) != ("bf16", "bf16")


def qsa_draft_rows(config) -> bool:
    return bool(envs.ATOM_QSA_DRAFT_KV_BF16) and qsa_caches_quantized(config)


def qsa_row_space(config, layer_num: int, num_target_layers: int) -> str:
    """The MTP draft layer sits past the `num_target_layers` target stack."""
    if layer_num >= num_target_layers and qsa_draft_rows(config):
        return QSA_DRAFT_ROWS
    return QSA_ROWS


def qsa_caches_rope_positions(config) -> bool:
    # Image/video requests give mRoPE three genuinely different position rows,
    # and then a compressed key's group position cannot be recomputed
    # arithmetically. Cache the per-token axes only when the engine can
    # actually be handed an image -- it costs ~1.4 GB of the paged budget.
    return getattr(config, "multimodal_config", None) is not None


def qsa_page_segments(config) -> dict:
    """What besides the namespace's own fields changes the PAGE segment set."""
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "method", None) == "mtp" and qsa_draft_rows(config)
    return {
        "image_input": qsa_caches_rope_positions(config),
        "draft_kv": "bf16" if draft else "target",
    }


def _kv_transfer_leaves(config) -> list[str]:
    return [name for name, _ in KVConnectorFactory.leaf_connectors(config) or ()]


def _check_kv_transfer(config) -> None:
    """Qwen3.8's GDN and PLE state ride the PAGE checkpoint image, so only an
    offload tier registered with `reads_page_images` may run."""
    leaves = _kv_transfer_leaves(config)
    for name in leaves:
        if not KVConnectorFactory.is_offload_backend(name):
            # A P/D connector moves the QSA pages alone, and decode would
            # continue from a zero recurrent state: wrong output, no error.
            raise ValueError(
                "Qwen3.8 disaggregated serving is not supported: "
                f"kv_connector={name!r} would transfer the QSA pages without "
                "the GDN and PLE state. Use an offload connector "
                "(lmcache_offload or lmcache_mp) or run without "
                "kv_transfer_config."
            )
        if not KVConnectorFactory.reads_page_images(name):
            raise ValueError(
                "Qwen3.8 offload supports "
                f"{', '.join(KVConnectorFactory.page_image_readers())}; "
                f"kv_connector={name!r} does not read the QSA/GDN/PLE PAGE "
                "checkpoint image."
            )
    if "lmcache_offload" in leaves and qsa_caches_quantized(config):
        # The in-process codec copies what `build_kv_cache_tensor` hands it:
        # no scale planes, so a CPU hit restores rows under stale scales.
        raise ValueError(
            "Qwen3.8 lmcache_offload does not carry the scale planes of a "
            "quantized QSA cache yet (kv_cache_dtype="
            f"{getattr(config, 'kv_cache_dtype', None)!r}, index_cache_dtype="
            f"{getattr(config, 'index_cache_dtype', None)!r}). Use "
            "--kv-cache-dtype bf16 --index-cache-dtype bf16, or lmcache_mp, "
            "which stores every QSA field; ROCm/ATOM#2655 lifts this."
        )


class _PlePlane(NamedTuple):
    """One PLE state tensor: `shape` is a single slot's, without the slot dim."""

    name: str
    tag: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    fill: int

    @property
    def slot_bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


def _shape_and_dtype(field) -> str:
    """`(shape),dtype` of one slot or entry, as the layout id spells it."""
    return f"{tuple(field.shape)},{field.dtype}".replace(" ", "")


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

    def for_tokens(self, num_tokens: int) -> "Qwen4ExpQSAMetadata":
        """Draft step zero uses real rows, without the target graph's padding."""
        if self.slot_mapping.numel() == num_tokens:
            return self
        return replace(
            self,
            slot_mapping=self.slot_mapping[:num_tokens],
            compressed_slot_mapping=self.compressed_slot_mapping[:num_tokens],
            token_to_req=self.token_to_req[:num_tokens],
            logical_positions=self.logical_positions[:num_tokens],
        )


@dataclass
class Qwen4ExpPLEMetadata:
    """Per-forward inputs for the single PLE layer."""

    query_start_loc: torch.Tensor  # [reqs + 1] int32
    ngram_state: torch.Tensor  # [slots, ngram_size - 1 + num_spec] int64
    state_indices_in: torch.Tensor  # [reqs] int32, slot the state is read from
    state_indices_out: torch.Tensor  # [reqs] int32, slot it is written to
    has_initial_state: torch.Tensor  # [reqs] bool
    conv_state: torch.Tensor  # [slots, channels, state_len] short-conv pool
    num_accepted_tokens: torch.Tensor | None = None


class Qwen4ExpBackend(GDNAttentionBackend):
    @staticmethod
    def validate_config(config) -> None:
        """Check the metadata and state layouts implemented by this backend."""
        if (
            config.speculative_config is not None
            and config.speculative_config.method != "mtp"
        ):
            raise ValueError("Qwen3.8-Flash-Next supports native MTP speculation only")
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
        _check_kv_transfer(config)
        if config.kv_transfer_config and not paged_checkpoints_deployable(config):
            # Restoring QSA pages from fork slots would leave GDN and PLE stale.
            raise ValueError(
                "Qwen3.8 offload needs PAGE checkpoints; pipeline parallelism "
                "and RapidServe stay on fork slots"
            )
        if len(set(config.hf_config.ple_layer_ids or [])) > 1:
            raise ValueError("Qwen backend currently allocates state for one PLE layer")
        qsa_cache_formats(config)
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

    def _gdn_flydsl_policy(self, module, state):
        from atom.model_ops.fla_ops.gdn_flydsl import select_policy

        attention = module.impl
        return select_policy(
            allowed=attention.allow_aiter_flydsl,
            replayssm=self.replayssm,
            lossy_decode=envs.ATOM_ENABLE_GDN_DECODE_LOSSY_FAST,
            state=state,
            activation_dtype=attention.dt_bias.dtype,
        )

    @cached_property
    def _ssm_storage(self) -> str:
        """`vk` when FlyDSL decode keeps the SSM state transposed, else `kv`.

        Asked of a one-slot probe: the layout id is spelled before any state
        is allocated.
        """
        _, shape_v = self._state_shape_for_runner()
        _, dtype_v = self._state_dtypes()
        probe = torch.empty((1, *shape_v), dtype=dtype_v, device=self.device)
        transposed = {
            self._gdn_flydsl_policy(module, probe).decode
            for module in self.pool_rows.get(LINEAR_STATE_ROWS, {})
            if hasattr(module, "base_linear_attention")
        }
        if len(transposed) > 1:
            raise RuntimeError("GDN layers disagree on the SSM storage order")
        return "vk" if True in transposed else "kv"

    def _build_gdn_cache_tensor(self, module):
        cache = super().build_kv_cache_tensor(module)
        if hasattr(module, "base_linear_attention"):
            attention = module.impl
            policy = self._gdn_flydsl_policy(module, cache.v_cache)
            if self._uses_paged_checkpoints() and policy.decode != (
                self._ssm_storage == "vk"
            ):
                raise RuntimeError(
                    "GDN SSM state is bound "
                    f"{'VK' if policy.decode else 'KV'} but the checkpoint "
                    f"layout id says ssm_storage={self._ssm_storage}"
                )
            attention.gdn_flydsl_policy = policy
            self._flydsl_prefill_enabled = (
                getattr(self, "_flydsl_prefill_enabled", False) or policy.prefill
            )
            if policy.decode:
                # Physical VK storage, exposed as a logical KV view. This is
                # zero-copy and preserves all gather/scatter/fork interfaces.
                # Triton fallback and checkpoint stores honor the inner strides.
                cache.v_cache = cache.v_cache.transpose(-1, -2)
        return cache

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        hf = model_runner.config.hf_config
        self.cache_rope_positions = qsa_caches_rope_positions(model_runner.config)
        self.ngram_context_len = int(hf.ngram_size) - 1 if hf.ple_layer_ids else 0
        eos = hf.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        max_tokens = self.max_num_batched_tokens
        i32 = {"dtype": torch.int32, "device": self.device}
        i64 = {"dtype": torch.int64, "device": self.device}
        self.model_runner.forward_vars["qsa_token_to_req"] = CpuGpuBuffer(
            max_tokens, publication_group="qsa", **i32
        )
        self.model_runner.forward_vars["qsa_logical_positions"] = CpuGpuBuffer(
            max_tokens, publication_group="qsa", **i64
        )
        # Written in place, never reallocated: a captured decode graph bakes in
        # the address it reads the compressed slots from.
        self.model_runner.forward_vars["qsa_compressed_slots"] = torch.empty(
            max_tokens, **i64
        )
        self.model_runner.forward_vars["ple_has_initial_state"] = CpuGpuBuffer(
            self.max_bs, dtype=torch.bool, device=self.device, publication_group="ple"
        )

    # ------------------------------------------------------------------ #
    # Geometry                                                            #
    # ------------------------------------------------------------------ #

    def _module_kinds(self, module) -> tuple:
        """A QSA layer takes a row of the QSA pool (or of the BF16 draft rows,
        when the draft keeps BF16 caches under quantized target caches);
        everything else is the hybrid's (the GDN state slots, handled by the
        parent)."""
        if getattr(module, "is_qsa_attention", False):
            return (getattr(module, "qsa_row_space", QSA_ROWS),)
        return super()._module_kinds(module)

    @property
    def _qsa_layers(self) -> int:
        """QSA layers this rank actually built, counted off the bind walk."""
        return self.row_counts().get(QSA_ROWS, 0)

    def _qsa_row_spaces(self):
        """`(row space, field-name suffix, (kv format, index format), rows)`."""
        counts = self.row_counts()
        spaces = [
            (QSA_ROWS, "", qsa_cache_formats(self.model_runner.config)),
            (QSA_DRAFT_ROWS, "_draft", ("bf16", "bf16")),
        ]
        return [
            (space, suffix, formats, counts[space])
            for space, suffix, formats in spaces
            if counts.get(space, 0)
        ]

    @property
    def _index_head_dim(self) -> int:
        return int(self.model_runner.config.hf_config.indexer_head_dim)

    @property
    def _compress_ratio(self) -> int:
        return int(self.model_runner.config.hf_config.indexer_compress_ratio)

    @cached_property
    def _ple_planes(self) -> list[_PlePlane]:
        """One PLE slot's tensors, in checkpoint order; empty without PLE."""
        hf = self.model_runner.config.hf_config
        if not getattr(hf, "ple_layer_ids", None):
            return []
        state_len = (int(hf.ple_conv_kernel_size) - 1) * int(hf.ngram_size)
        channels = int(hf.hidden_size) * int(hf.hc_count)
        # The PLE convolution is NOT tensor-parallel: every rank runs the full
        # hc_count * hidden width, matching the reference's tp_world_size=1.
        return [
            _PlePlane(
                "ple_conv_state",
                "conv",
                (channels, state_len + self.num_spec),
                self.model_runner.config.torch_dtype,
                0,
            ),
            _PlePlane(
                "ple_ngram_state",
                "ngram",
                (self.ngram_context_len + self.num_spec,),
                torch.int64,
                self.eos_token_id,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Pool sizing and allocation                                          #
    # ------------------------------------------------------------------ #

    def _qsa_fields(self) -> list[EntryField]:
        """Single declaration used for sizing, allocation and transfer views.

        A quantized K/V or compressed-key cache adds its scale plane
        (`*_scale`), paged like the rows it scales. BF16 draft rows under
        quantized target rows get their own `*_draft` fields.
        """
        hf = self.model_runner.config.hf_config
        block = self.block_size
        heads = self.model_runner._get_num_kv_heads()
        rows = block // self._compress_ratio
        fields = []
        for _, sfx, (kv_format, index_format), layers in self._qsa_row_spaces():
            kv_dim, kv_dtype, kv_scale, kv_scale_dtype = qsa_cache_row_layout(
                kv_format, int(hf.head_dim)
            )
            fields += [
                EntryField(f"k{sfx}", layers, (block, heads, kv_dim), kv_dtype),
                EntryField(f"v{sfx}", layers, (block, heads, kv_dim), kv_dtype),
            ]
            if kv_scale is not None:
                for name in ("k_scale", "v_scale"):
                    fields.append(
                        EntryField(
                            f"{name}{sfx}",
                            layers,
                            (block, heads, *kv_scale),
                            kv_scale_dtype,
                        )
                    )
            index_dim, index_dtype, index_scale, index_scale_dtype = (
                qsa_cache_row_layout(index_format, self._index_head_dim)
            )
            fields += [
                EntryField(
                    f"index_raw{sfx}",
                    layers,
                    (block, 1, self._index_head_dim),
                    torch.bfloat16,
                ),
                EntryField(
                    f"index_compressed{sfx}", layers, (rows, 1, index_dim), index_dtype
                ),
            ]
            if index_scale is not None:
                fields.append(
                    EntryField(
                        f"index_compressed_scale{sfx}",
                        layers,
                        (rows, 1, *index_scale),
                        index_scale_dtype,
                    )
                )
            if self.cache_rope_positions:
                fields.append(
                    EntryField(
                        f"rope_positions{sfx}", layers, (block, 1, 3), torch.int64
                    )
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
        planes = self._ple_planes
        if not planes:
            return None
        return state_pool(
            PLE_STATE_SLOT_CLASS,
            sum(plane.slot_bytes for plane in planes),
            entries_per_req=self.slots_per_req(),
        )

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        # The GDN recurrent state carries over unchanged; the paged pool does
        # not, because QSA's layout and side caches replace the MHA one.
        specs = [page_pool(self._page_bytes()), self.state_spec()]
        ple_spec = self._ple_state_spec()
        if ple_spec is not None:
            specs.append(ple_spec)
        return specs

    def _wants_paged_checkpoints(self, config) -> bool:
        # No model-type test: only `Qwen4ExpBackend` builds this class.
        return any(
            KVConnectorFactory.reads_page_images(name)
            for name in _kv_transfer_leaves(config)
        )

    def state_transfer(self):
        transfer = super().state_transfer()
        if not transfer.copies:
            return transfer
        # What the parent's id cannot see: the PAGE fields, the SSM's physical
        # order and the PLE planes.
        planes = self._ple_planes
        layout_id = (
            transfer.paged_layout_id
            + f":ssm_storage={self._ssm_storage}"
            + ":page="
            + ";".join(f"{f.name}={_shape_and_dtype(f)}" for f in self._qsa_fields())
            + "".join(f":ple_{p.tag}={_shape_and_dtype(p)}" for p in planes)
        )
        if not planes:
            return replace(transfer, paged_layout_id=layout_id)
        layout_id += ":ple_order=" + ",".join(plane.tag for plane in planes)
        # Not readable midstep: nothing snapshots the PLE windows inside a
        # forward, so prefill stays cut onto rungs.
        return StateTransfer.copy(layout_id)

    def _checkpoint_plane_shapes(self) -> list[tuple[int, int]]:
        return super()._checkpoint_plane_shapes() + [
            (plane.slot_bytes, 1) for plane in self._ple_planes
        ]

    def _ple_tensors(self) -> list[torch.Tensor]:
        """The runner's PLE tensors in `_ple_planes` order; empty before allocation."""
        states = [getattr(self.model_runner, p.name, None) for p in self._ple_planes]
        return [] if all(state is None for state in states) else states

    def _checkpoint_plane_tensors(self) -> list[torch.Tensor]:
        # GDN planes are `(layers, slots, ...)`. PLE is allocated `(slots, ...)`.
        return super()._checkpoint_plane_tensors() + [
            state.unsqueeze(0) for state in self._ple_tensors()
        ]

    def state_entry_views(self, slot: int) -> list[torch.Tensor]:
        # The parent range-checks `slot`; PLE shares its slot class.
        return super().state_entry_views(slot) + [
            state[slot : slot + 1] for state in self._ple_tensors()
        ]

    # The parent's PAGE contract reads the image through these two hooks.

    def _published_page_unit_bytes(self) -> int:
        # Asked at sizing, before the arena exists.
        return published_bytes_for(self._qsa_fields())

    def _page_unit_pools(self) -> list:
        arena = getattr(self, "qsa_arena", None)
        if arena is None:
            raise RuntimeError("Qwen3.8 PAGE checkpoint has no QSA arena")
        return [arena]

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        """Back every paged field with the runner's shared allocation."""
        self.num_blocks = blocks
        self.qsa_arena = LayerMajorArena(
            self._qsa_fields(), blocks, self.device, buf=buf
        )
        return {}

    def release_kv_pools(self) -> None:
        for space in (QSA_ROWS, QSA_DRAFT_ROWS):
            for module in self.pool_rows.get(space, {}):
                module.bind_caches(None, None, None, None, None)
        self.qsa_arena = None
        super().release_kv_pools()

    def get_kv_transfer_tensors(self):
        """QSA PAGE regions; GDN and PLE ride the checkpoint image."""
        arena = getattr(self, "qsa_arena", None)
        if arena is None:
            return None
        transfer = KVTransferTensors(
            pages=[
                page_region(tensor, semantic_role=role)
                for role, tensor in arena.region_tensors()
            ],
        )
        return self._with_paged_state_contract(transfer)

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict[str, object]:
        caches = super().allocate_per_req_cache(entries)
        slots = entries.get(PLE_STATE_SLOT_CLASS, 0)
        for plane in self._ple_planes:
            caches[plane.name] = torch.full(
                (slots, *plane.shape), plane.fill, dtype=plane.dtype, device="cuda"
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
            return self._build_gdn_cache_tensor(module)

        from atom.config import KVCacheTensor

        space = getattr(module, "qsa_row_space", QSA_ROWS)
        sfx = "_draft" if space == QSA_DRAFT_ROWS else ""
        row = self.pool_rows[space][module]
        arena = self.qsa_arena
        names = {field.name for field in arena.fields}

        def optional(name):
            name += sfx
            return arena.view(name)[row] if name in names else None

        key, value = optional("k"), optional("v")
        module.bind_caches(
            key,
            value,
            optional("index_raw"),
            optional("index_compressed"),
            optional("rope_positions"),
            k_scale_cache=optional("k_scale"),
            v_scale_cache=optional("v_scale"),
            compressed_key_scale=optional("index_compressed_scale"),
        )
        return KVCacheTensor(
            layer_num=module.layer_num,
            k_cache=key,
            v_cache=value,
            k_scale=None,
            v_scale=None,
            extra_planes=tuple(
                plane
                for plane in map(
                    optional, ("index_raw", "index_compressed", "rope_positions")
                )
                if plane is not None
            ),
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
        self._check_metadata_writable("qsa")
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
        if gdn is None:
            return None
        if gdn.spec_state_indices_tensor is not None:
            # Like GDN's convolution, PLE retains the candidate tail in slot 0
            # and selects the accepted window on the next verification pass.
            slots = gdn.spec_state_indices_tensor[:, 0]
            return slots, slots
        if gdn.non_spec_state_indices_tensor is None:
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
        self._check_metadata_writable("ple")
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
            num_accepted_tokens=attn_metadata.gdn_metadata.num_accepted_tokens,
        )

    def prepare_prefill(
        self, batch: ScheduledBatch, running_bs: int, running_tokens: int
    ):
        attn_metadata, positions = super().prepare_prefill(
            batch, running_bs, running_tokens
        )
        num_reqs = batch.total_seqs_num_prefill
        num_tokens = batch.total_tokens_num_prefill
        # QSA reads the compressed cache through the block table on every
        # forward, so unlike dense prefill it cannot wait for `has_cached`.
        if attn_metadata.block_tables is None and batch.block_tables:
            # The parent already packed this source; only its upload was optional.
            attn_metadata.block_tables = block_table_state(
                self.model_runner.forward_vars["block_tables"]
            ).publish(num_reqs)
        if attn_metadata.block_tables is None:
            attn_metadata.qsa_metadata = None
            attn_metadata.ple_metadata = None
            return attn_metadata, positions

        query_lens = np.asarray(batch.num_scheduled_tokens[:num_reqs], dtype=np.int64)
        if (
            attn_metadata.gdn_metadata is not None
            and getattr(self, "_flydsl_prefill_enabled", False)
            and not self.replayssm
        ):
            from atom.model_ops.fla_ops.gdn_flydsl import build_prefill_metadata

            attn_metadata.gdn_metadata.flydsl_prefill_metadata = build_prefill_metadata(
                query_lens, attn_metadata.gdn_metadata.non_spec_query_start_loc
            )
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
        attn_metadata.ple_metadata = self._build_ple_metadata(
            batch,
            attn_metadata,
            bs,
            is_prefill=False,
        )
        return attn_metadata, positions

    def _refreshed_flydsl_plan(self, var, running_bs: int):
        """The draft's own plan, never the target's.

        Each draft pass advances context_lens and the planner bakes per-task
        tile ranges from it, so an inherited plan attends over the pre-bump
        context. The op cannot catch it: its guard compares batch and kv-head
        count, and neither changes.
        """
        return self.refresh_flydsl_plan(var["context_lens"].gpu[:running_bs])

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor | None = None,
    ) -> dict:
        """Move the draft's paged QSA view to the accepted sequence tail."""
        running_bs = positions.shape[-1]
        var = self.model_runner.forward_vars
        slots = var["slot_mapping"].gpu[:running_bs]
        if getattr(self, "qsa_arena", None) is None:
            # Allocation profiling runs all draft steps before the pools exist.
            return {
                "slot_mapping": slots,
                "qsa_metadata": None,
                "flydsl_work_plan": self._refreshed_flydsl_plan(var, running_bs),
            }
        logical = var["qsa_logical_positions"].gpu[:running_bs]
        req_ids = var["qsa_token_to_req"].gpu[:running_bs]
        compressed = var["qsa_compressed_slots"][:running_bs]
        lengths = var["context_lens"].gpu[:running_bs]
        tables = var["block_tables"].gpu[:running_bs]
        qsa_draft_decode_metadata(
            lengths,
            tables,
            num_reject_tokens,
            slots,
            logical,
            req_ids,
            compressed,
            bs,
            self.block_size,
            self._compress_ratio,
        )
        return {
            "slot_mapping": slots,
            "flydsl_work_plan": self._refreshed_flydsl_plan(var, running_bs),
            "qsa_metadata": Qwen4ExpQSAMetadata(
                tables,
                slots,
                compressed,
                req_ids,
                logical,
                lengths,
                self.model_runner.config.max_model_len,
            ),
        }

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
        self._prepare_state_indices_for_capture(bs)
        attn_metadata, context = super().build_for_cudagraph_capture(bs)
        runner = self.model_runner
        num_tokens = bs * int(attn_metadata.max_seqlen_q)
        attn_metadata.slot_mapping = runner.forward_vars["slot_mapping"].gpu[
            :num_tokens
        ]
        context.positions = runner.forward_vars["positions"].gpu[:num_tokens]

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
        self._check_metadata_writable("ple")
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
            num_accepted_tokens=attn_metadata.gdn_metadata.num_accepted_tokens,
        )
        return attn_metadata, context
