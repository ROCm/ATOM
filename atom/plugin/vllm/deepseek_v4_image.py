"""DeepSeek-V4 adapter for prefix caching through checkpoint images.

V4 keeps every request's sliding-window ring and compressor tails in a per-request
slot carved from the proxy pool, outside any block vLLM hashes. The image
mechanism itself is model-agnostic (``paged_state_image``,
``paged_state_image_scheduler``); this module supplies V4's side of it
(``DeepseekV4ImageAdapter``):

* The paged cache stays one opaque proxy layer, priced at its real per-block
  bytes. The slot area is withheld from the block pool as a tail, not amortized
  into every page (``V4ImageSizing.usable_blocks``).
* An image is the part of a slot a native checkpoint carries (CSA main/indexer
  compressor kv/score and every window row; HCA owes a 128-aligned boundary
  nothing): native ``PagedStateCheckpointSpec`` with ``k = units_per_checkpoint``
  PAGE units at block size 128.
* The copy between a slot and ``k`` non-contiguous PAGE units is native ATOM's
  ``DeepseekV4AttentionMetadataBuilder.execute_paged_state_copies``, run over the
  plugin's planes by ``_PluginV4StateCopier``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from atom.plugin.vllm.paged_state_image import (
    PagedStateImageAdapter,
    PagedStateImageBackend,
    PagedStateImageLayer,
    image_state_copy_funcs,
    images_on,
    install,
    register_image_adapter,
    register_image_layers,
)
from atom.plugin.vllm.paged_state_image_scheduler import (
    ImagePlacement,
    SchedulerImages,
)

logger = logging.getLogger("atom")

ATOM_DEEPSEEK_V4_IMAGE_LAYER_SUFFIX = "atom_deepseek_v4_image"
_V4_IMAGE_BLOCK_SIZE = 128
_V4_PROXY_ALIGNMENT = 256
_V4_IMAGE_LAYOUT_PREFIX = "dsv4-paged-state-v3-plugin"


def deepseek_v4_image_layer_names(k: int) -> list[str]:
    # Layer index 1+i: index 0 is the proxy, and the draft proxy sits at
    # num_hidden_layers, which is far above any k this layout produces.
    return [
        f"model.layers.{1 + i}.{ATOM_DEEPSEEK_V4_IMAGE_LAYER_SUFFIX}" for i in range(k)
    ]


def is_deepseek_v4_config(vllm_config) -> bool:
    from atom.plugin.vllm.platform import _is_deepseek_v4

    mc = getattr(vllm_config, "model_config", None)
    return mc is not None and _is_deepseek_v4(mc)


def deepseek_v4_images_on(vllm_config) -> bool:
    """Whether V4 prefix hits are restored from checkpoint images."""
    return images_on(register_v4_image_adapter(), vllm_config)


# ---------------------------------------------------------------------------
# The copier: native checkpoint copy over the plugin's planes
# ---------------------------------------------------------------------------


def _native_builder_cls():
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    return DeepseekV4AttentionMetadataBuilder


def _native_staging_fn():
    from atom.model_ops.attentions.backends import CommonAttentionBuilder

    return CommonAttentionBuilder._checkpoint_staging


class _PluginV4StateCopier:
    """Native V4 slot <-> PAGE-unit checkpoint copy, bound to the plugin's pool.

    Every method that decides bytes or addresses is native ATOM's own function
    (attached in ``_attach_native``), so a store/restore here cuts the image at the
    same ranges, in the same order, through the same descriptor kernel as a native
    checkpoint. This class only supplies what those methods read off ``self``: the
    planes, the indexer pool, the geometry, the arena field split and a
    ``model_runner`` carrying the checkpoint spec.
    """

    def __init__(
        self,
        *,
        geometry,
        arena_planes,
        row_widths,
        ratios,
        num_slots: int,
        max_num_seqs: int,
        kv_planes=None,
        indexer_pool=None,
        num_blocks: int = 0,
        device=None,
        spec=None,
    ) -> None:
        self.pool_geometry = geometry
        self._arena_planes = arena_planes
        self._row_widths = list(row_widths)
        self.compress_ratios = [int(r) for r in ratios]
        self.csa_layers = [i for i, r in enumerate(self.compress_ratios) if r == 4]
        self.num_blocks = int(num_blocks)
        self._num_slots = int(num_slots)
        self._planes = kv_planes
        self._indexer_pool = indexer_pool
        self._indexer_fp4 = False
        self._device = device
        self.model_runner = SimpleNamespace(
            config=SimpleNamespace(
                kv_cache_block_size=_V4_IMAGE_BLOCK_SIZE,
                decode_context_parallel_size=1,
                max_num_seqs=int(max_num_seqs),
            ),
            state_runtime=SimpleNamespace(checkpoint_spec=spec),
        )
        self._slot_view_cache = None
        self._checkpoint_range_cache = None
        self._checkpoint_plan_cache = None
        self._checkpoint_slot_base_cache = None
        self._checkpoint_staging_cache = None
        self._page_unit_region_cache = None
        self._page_unit_region_owners = ()

    @property
    def num_state_slots(self) -> int:
        return self._num_slots

    def _plane_row_widths(self) -> list[int]:
        return list(self._row_widths)

    def _kv_planes(self) -> list[torch.Tensor]:
        return list(self._planes)

    def _indexer_page_pools(self):
        return [(self._indexer_pool, "dsv4.csa_indexer")]

    def _checkpoint_descriptor_device(self) -> torch.device:
        return self._device

    @property
    def checkpoint_spec(self):
        return self.model_runner.state_runtime.checkpoint_spec


_NATIVE_COPY_METHODS = (
    "execute_paged_state_copies",
    "_validate_paged_state_op",
    "_checkpoint_slot_ranges",
    "_checkpoint_segment_sizes",
    "checkpoint_image_bytes",
    "_assert_ratios_divide_the_alignment",
    "_checkpoint_copy_plan",
    "_checkpoint_slot_bases",
    "_page_unit_regions",
    "_page_unit_bases",
    "_page_unit_stream_sizes",
    "_slot_views",
    "warmup_per_req_cache",
)


def _attach_native() -> None:
    if getattr(_PluginV4StateCopier, "_native_attached", False):
        return
    native = _native_builder_cls()
    for name in _NATIVE_COPY_METHODS:
        setattr(_PluginV4StateCopier, name, native.__dict__[name])
    _PluginV4StateCopier._checkpoint_staging = _native_staging_fn()
    _PluginV4StateCopier._native_attached = True


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V4ImageSizing:
    ratios: tuple[int, ...]
    n_csa: int
    win_with_spec: int
    num_slots: int
    max_num_seqs: int
    arena_rows: int
    row_widths: tuple[int, ...]
    index_row_bytes: int
    # One 128-token block in ATOM's carve: both plane envelopes + every CSA
    # layer's indexer block. A PAGE unit is exactly this.
    page_unit_bytes: int
    # What vLLM prices a block at: the above, 256-aligned.
    page_bytes: int
    slot_bytes: int
    image_bytes: int
    layout_id: str

    @property
    def spec(self):
        """Native ATOM's checkpoint spec for this layout."""
        from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec

        return PagedStateCheckpointSpec(
            page_unit_bytes=self.page_unit_bytes,
            slot_bytes=self.slot_bytes,
            layout_id=self.layout_id,
            image_bytes=self.image_bytes,
        )

    @property
    def k(self) -> int:
        return self.spec.units_per_checkpoint

    def geometry(self, num_blocks: int, num_slots: int | None = None):
        from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
            UnifiedPoolGeometry,
        )

        return UnifiedPoolGeometry(
            list(self.ratios),
            num_blocks=int(num_blocks),
            num_slots=int(self.num_slots if num_slots is None else num_slots),
            ring_slots=self.win_with_spec,
            block_size=_V4_IMAGE_BLOCK_SIZE,
            arena_rows=self.arena_rows,
        )

    def carve_bytes(self, num_blocks: int) -> int:
        """Bytes the plugin carve takes for `num_blocks` blocks + every slot."""
        from atom.model_ops.attentions.pool_layout.entry_arena import plan_regions

        geo = self.geometry(num_blocks)
        regions = [geo.plane_bytes(w) for w in self.row_widths]
        regions.append(
            self.n_csa * num_blocks * (_V4_IMAGE_BLOCK_SIZE // 4) * self.index_row_bytes
        )
        _, total = plan_regions(regions)
        # A packed allocation may start this cache off the 256 B boundary.
        return total + _V4_PROXY_ALIGNMENT - 1

    def usable_blocks(self, tensor_blocks: int) -> int:
        """Blocks the pool may hand out once the slot tail is withheld.

        A function of the tensor's block count alone, so EngineCore (which sets
        the scheduler's count) and every worker (which carves) agree.
        """
        total = int(tensor_blocks) * self.page_bytes
        usable = int(tensor_blocks) - math.ceil(
            (self.num_slots * self.slot_bytes + _V4_PROXY_ALIGNMENT) / self.page_bytes
        )
        while usable > 0 and self.carve_bytes(usable) > total:
            usable -= 1
        return max(usable, 0)


def v4_image_sizing(vllm_config) -> V4ImageSizing:
    from atom.plugin.vllm.deepseek_v4_bridge import (
        _index_row_bytes,
        _layer_counts,
        _v4_kv_fp8,
        _v4_state_layout,
        _v4_win_with_spec,
    )

    _attach_native()
    hf = vllm_config.model_config.hf_config
    # Every ratio the checkpoint lists, MTP layer included: the target carves
    # its slot over all of them, as native does (a draft ring lives in the
    # target's slot), and the image is a function of that slot.
    ratios, _dense, n_csa, _hca = _layer_counts(hf)
    kv_fp8 = _v4_kv_fp8(vllm_config)
    arena_planes, arena_rows, row_widths = _v4_state_layout(vllm_config, kv_fp8)
    win = _v4_win_with_spec(vllm_config, int(getattr(hf, "sliding_window", 128)))
    max_num_seqs = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    index_row = _index_row_bytes(int(getattr(hf, "index_head_dim", 128)))
    head_dim = int(getattr(hf, "head_dim", 512))
    rope = int(getattr(hf, "qk_rope_head_dim", 64))
    index_head_dim = int(getattr(hf, "index_head_dim", 128))

    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        UnifiedPoolGeometry,
    )

    geo = UnifiedPoolGeometry(
        ratios,
        num_blocks=1,
        num_slots=max_num_seqs,
        ring_slots=win,
        block_size=_V4_IMAGE_BLOCK_SIZE,
        arena_rows=arena_rows,
    )
    page_unit = (
        sum(geo.block_bytes(w) for w in row_widths)
        + n_csa * (_V4_IMAGE_BLOCK_SIZE // 4) * index_row
    )
    page = -(-page_unit // _V4_PROXY_ALIGNMENT) * _V4_PROXY_ALIGNMENT
    slot = sum(geo.slot_bytes(w) for w in row_widths)
    sizer = _PluginV4StateCopier(
        geometry=geo,
        arena_planes=arena_planes,
        row_widths=row_widths,
        ratios=ratios,
        num_slots=max_num_seqs,
        max_num_seqs=max_num_seqs,
    )
    image = int(sizer.checkpoint_image_bytes())
    nocopy = ",".join(
        f.name for plane in arena_planes for f in plane if not f.in_checkpoint
    )
    layout_id = (
        f"{_V4_IMAGE_LAYOUT_PREFIX}:block={_V4_IMAGE_BLOCK_SIZE}:ring={win}"
        f":dims={head_dim},{rope},{index_head_dim}"
        f":main={'fp8-2buff' if kv_fp8 else 'bf16'}"
        f":ratios={','.join(str(r) for r in ratios)}:nocopy={nocopy}:entry=packed"
    )
    return V4ImageSizing(
        ratios=tuple(int(r) for r in ratios),
        n_csa=n_csa,
        win_with_spec=win,
        num_slots=max_num_seqs,
        max_num_seqs=max_num_seqs,
        arena_rows=arena_rows,
        row_widths=tuple(int(w) for w in row_widths),
        index_row_bytes=index_row,
        page_unit_bytes=int(page_unit),
        page_bytes=int(page),
        slot_bytes=int(slot),
        image_bytes=image,
        layout_id=layout_id,
    )


def make_v4_image_copier(sizing: V4ImageSizing, views, *, num_blocks: int, device):
    """The copier over a bound pool, cross-checked against the sizing."""
    _attach_native()
    geo = views["geometry"]
    planes = [views["kv_plane"]]
    if views.get("kv_plane_rope") is not None:
        planes.append(views["kv_plane_rope"])
    spec = sizing.spec
    copier = _PluginV4StateCopier(
        geometry=geo,
        arena_planes=views["arena_planes"],
        row_widths=sizing.row_widths,
        ratios=sizing.ratios,
        num_slots=sizing.num_slots,
        max_num_seqs=sizing.max_num_seqs,
        kv_planes=planes,
        indexer_pool=views["csa_indexer_pool"],
        num_blocks=num_blocks,
        device=device,
        spec=spec,
    )
    _, strides = copier._page_unit_regions()
    actual_page = int(np.asarray(strides).sum())
    actual_slot = sum(geo.slot_bytes(w) for w in sizing.row_widths)
    actual_image = int(copier.checkpoint_image_bytes())
    if (
        actual_page != sizing.page_unit_bytes
        or actual_slot != sizing.slot_bytes
        or actual_image != sizing.image_bytes
    ):
        raise RuntimeError(
            "DeepSeek-V4 image geometry differs from sizing: "
            f"page={actual_page}/{sizing.page_unit_bytes}, "
            f"slot={actual_slot}/{sizing.slot_bytes}, "
            f"image={actual_image}/{sizing.image_bytes}"
        )
    return copier


# ---------------------------------------------------------------------------
# Image layers, placement defaults
# ---------------------------------------------------------------------------


class AtomDeepseekV4ImageBackend(PagedStateImageBackend):
    block_size = _V4_IMAGE_BLOCK_SIZE

    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V4_IMAGE"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [_V4_IMAGE_BLOCK_SIZE]


class AtomDeepseekV4ImageLayer(PagedStateImageLayer):
    """One PAGE unit of a V4 checkpoint image, as vLLM sees it."""

    def __init__(self, prefix: str, part: int, page_bytes: int, adapter=None):
        super().__init__(
            prefix, part, page_bytes, adapter or register_v4_image_adapter()
        )


def register_deepseek_v4_image_layers(vllm_config) -> list[str]:
    """Register the k image layers next to the proxy (worker side)."""
    return register_image_layers(
        register_v4_image_adapter(), vllm_config, AtomDeepseekV4ImageLayer
    )


def deepseek_v4_image_state_copy_funcs():
    return image_state_copy_funcs()


def _native_checkpoint_defaults() -> tuple[int, bool]:
    from atom.config import Config

    fields = Config.__dataclass_fields__
    return (
        int(fields["state_checkpoint_interval_tokens"].default),
        bool(fields["state_checkpoint_demand"].default),
    )


@dataclass(frozen=True)
class V4ImagePlacement(ImagePlacement):
    """Native placement at V4's 128-token block."""

    block: int = _V4_IMAGE_BLOCK_SIZE

    @classmethod
    def from_env(cls, defaults=None, block=_V4_IMAGE_BLOCK_SIZE) -> V4ImagePlacement:
        return super().from_env(defaults or _native_checkpoint_defaults(), block)


V4SchedulerImages = SchedulerImages


# ---------------------------------------------------------------------------
# The adapter and the install entry
# ---------------------------------------------------------------------------


class DeepseekV4ImageAdapter(PagedStateImageAdapter):
    name = "DeepSeek-V4"
    layer_suffix = ATOM_DEEPSEEK_V4_IMAGE_LAYER_SUFFIX
    block_size = _V4_IMAGE_BLOCK_SIZE
    backend_cls = AtomDeepseekV4ImageBackend

    @property
    def proxy_layer_name(self) -> str:
        from atom.plugin.vllm.deepseek_v4_bridge import (
            ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME,
        )

        return ATOM_DEEPSEEK_V4_PROXY_LAYER_NAME

    def matches(self, vllm_config) -> bool:
        return is_deepseek_v4_config(vllm_config)

    def sizing(self, vllm_config) -> V4ImageSizing:
        return v4_image_sizing(vllm_config)

    def image_layer_names(self, k: int) -> list[str]:
        return deepseek_v4_image_layer_names(k)

    def checkpoint_defaults(self) -> tuple[int, bool]:
        return _native_checkpoint_defaults()


_V4_ADAPTER = DeepseekV4ImageAdapter()


def register_v4_image_adapter() -> DeepseekV4ImageAdapter:
    return register_image_adapter(_V4_ADAPTER)


def apply_vllm_v4_prefix_install() -> None:
    """Register the V4 adapter and install every image hook. Idempotent; called
    from register_model.

    ``ATOMPlatform.check_and_update_config`` is not a reliable site: vLLM may
    never activate the platform (``register_platform`` runs inside ``import vllm``
    and its failure is swallowed). ``register_model`` runs in EngineCore and
    in every worker.
    """
    register_v4_image_adapter()
    a, b, c, d = install()
    if a or b or c or d:
        logger.info(
            "ATOM DeepSeek-V4: installed image prefix hooks "
            "(get_kv_cache_configs=%s, kv zeroing=%s, runner events=%s, "
            "placement scheduler=%s)",
            a,
            b,
            c,
            d,
        )
