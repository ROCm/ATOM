# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's side of the PAGE-backed state checkpoint path.

The mechanism is `paged_state_checkpoint.py`, which landed on main with
ROCm/ATOM#2502; this file is the family adapter it dispatches to. The vendored
copy that used to sit beside it is gone -- upstream's is the one in use, under
upstream's names. Every V4.1-specific and V2-model-runner difference belongs
here, not there.

V4.1 is the simple case of that mechanism, in two ways V4 is not:

  * **One PAGE unit is exactly one proxy block.** V4's unit is a 128-token
    carve whose bytes are spread over several planes, so its adapter has to
    rebuild the geometry to price a block. V4.1's `paged_bytes` *is* the
    block at 256 tokens, so the sizing is read straight off
    `V41PoolGeometry` and `StateCopies` already asserts the three numbers
    agree (`checkpoints.py:21-26`).
  * **The copier is native's, not a borrowed one.** #2502's V4 copier steals
    fifteen methods off the V4 metadata builder (`_attach_native`) and its
    own PR calls that a hack to be replaced "together with the V4.1 adapter".
    Taking it at its word: V4.1 wraps `StateCopies` directly.

Both rest on the same property as #2502: every group overlays one
`KVCacheTensor` from byte 0 at the same block stride, so a block id means
the same unit to vLLM and to ATOM, and native's `proxy_base + unit * bytes`
addressing is already right. `check_state_checkpoint_groups` asserts it at startup.

What V4.1 decides differently from V4, and why:

  * **DSpark is accepted.** V4's adapter refuses every drafter but MTP,
    because a draft's state sits outside the target's slot. V4.1's does not:
    the window ring is sized `window_size + speculative_tokens` and the
    compressor ring is widened by the same slack, so a rejected round's rows
    land past the window a resumer reads and neither needs a rollback
    (`attentions/deepseek_v41/speculative.py:4-6`). The bytes a speculative
    step leaves are already the right bytes for the accepted prefix; only
    the position they are filed under has to wait for `commit_tentative`.
  * **No fp8 requirement.** V4's checkpoints are validated only over its fp8
    2buff plane layout and its bf16 path does not run. V4.1's target pool is
    BF16 and that is the configuration it ships in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from atom.plugin.vllm.paged_state_checkpoint import (
    PagedStateCheckpointAdapter,
    PagedStateCheckpointBackend,
    PagedStateCheckpointLayer,
    install,
    native_checkpoint_defaults,
    register_state_checkpoint_adapter,
    register_state_checkpoint_layers,
    state_checkpoints_on,
)
from atom.plugin.vllm.paged_state_checkpoint_scheduler import (
    SchedulerStateCheckpoints,
    StateCheckpointPlacement,
)

#: Suffix of the checkpoint layer names. Positional: unit *i* of the checkpoint is
#: layer *i*, so the order is the key space and renumbering makes every
#: stored checkpoint unreadable.
ATOM_DEEPSEEK_V41_STATE_CHECKPOINT_LAYER_SUFFIX = "atom_deepseek_v41_state_checkpoint"


def _block_size() -> int:
    from atom.plugin.vllm.deepseek_v41_bridge import ATOM_DEEPSEEK_V41_BLOCK_SIZE

    return int(ATOM_DEEPSEEK_V41_BLOCK_SIZE)


def _proxy_alignment() -> int:
    from atom.plugin.vllm.deepseek_v41_bridge import ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT

    return int(ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT)


def deepseek_v41_state_checkpoint_layer_names(k: int) -> list[str]:
    return [
        f"model.layers.{1 + i}.{ATOM_DEEPSEEK_V41_STATE_CHECKPOINT_LAYER_SUFFIX}"
        for i in range(k)
    ]


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V41StateCheckpointSizing:
    """What vLLM must be told, derived from the one geometry that owns it.

    Every field is read off `V41PoolGeometry` rather than recomputed, because
    `StateCopies.__init__` refuses a spec whose three byte counts differ from
    the allocated cache's -- so a second derivation here could only ever
    disagree with the thing it has to match.
    """

    num_slots: int
    max_num_seqs: int
    #: One 256-token V4.1 PAGE: the main page plus each owner's index rows.
    page_unit_bytes: int
    #: What vLLM prices a block at -- the above, proxy-aligned.
    page_bytes: int
    slot_bytes: int
    image_bytes: int
    layout_id: str

    @property
    def spec(self):
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

    def carve_bytes(self, num_blocks: int) -> int:
        """Bytes the V4.1 carve takes for `num_blocks` blocks plus every slot.

        `paged_extents(pages)[1]` is where the PAGE region ends and the STATE
        region begins -- the one boundary between the two currencies -- so the
        carve is that plus one slot each. The alignment slack is the same one
        `bind_deepseek_v41_proxy_cache` checks the storage offset against.

        Asks the geometry this sizing was built from rather than fetching it
        again from the config: the second lookup would be a second derivation
        of a number that has to match the first exactly.
        """
        boundary = int(self._geometry.paged_extents(int(num_blocks))[1])
        return boundary + self.num_slots * self.slot_bytes + _proxy_alignment() - 1

    def usable_blocks(self, tensor_blocks: int) -> int:
        """Blocks the pool may hand out once the slot tail is withheld.

        A pure function of the tensor's block count, so EngineCore (which
        sets the scheduler's count) and every worker (which carves) reach the
        same answer without talking to each other.
        """
        total = int(tensor_blocks) * self.page_bytes
        usable = int(tensor_blocks) - math.ceil(
            (self.num_slots * self.slot_bytes + _proxy_alignment()) / self.page_bytes
        )
        while usable > 0 and self.carve_bytes(usable) > total:
            usable -= 1
        return max(usable, 0)


#: `carve_bytes` needs the geometry the sizing was read off, and the dataclass
#: is frozen, so it is attached rather than being a field: a field would make
#: it part of equality and of the log line, and it is neither.
def _with_geometry(
    sizing: V41StateCheckpointSizing, geometry
) -> V41StateCheckpointSizing:
    object.__setattr__(sizing, "_geometry", geometry)
    return sizing


def v41_state_checkpoint_sizing(vllm_config) -> V41StateCheckpointSizing:
    from atom.plugin.vllm.deepseek_v41_bridge import (
        v41_num_state_slots,
        v41_proxy_geometry,
        v41_proxy_page_size_bytes,
    )

    geometry = v41_proxy_geometry(vllm_config)
    return _with_geometry(
        V41StateCheckpointSizing(
            num_slots=int(v41_num_state_slots(vllm_config)),
            max_num_seqs=max(1, int(vllm_config.scheduler_config.max_num_seqs)),
            page_unit_bytes=int(geometry.paged_bytes),
            page_bytes=int(v41_proxy_page_size_bytes(vllm_config)),
            slot_bytes=int(geometry.state_bytes),
            image_bytes=int(geometry.checkpoint_image_bytes),
            layout_id=str(geometry.layout_id),
        ),
        geometry,
    )


# ---------------------------------------------------------------------------
# The copier: native's, wrapped to the mechanism's two-method interface
# ---------------------------------------------------------------------------


class _PluginV41StateCopier:
    """`StateCopies` behind the names `paged_state_checkpoint` calls.

    No method borrowing: the checkpoint blocks alias the proxy arena at the same
    block stride, which is what makes native's own addressing correct here.
    The wrapper exists only because the mechanism asks for `checkpoint_spec`
    as an attribute and `execute_paged_state_copies` as a method, while
    native spells them `spec` and `execute`.
    """

    def __init__(self, cache, spec, max_batch: int):
        from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies

        # Raises when the three byte counts disagree with the allocated
        # cache, which is the cross-check this adapter relies on instead of
        # re-deriving the geometry.
        self._copies = StateCopies(cache, spec, max_batch)
        # The store buffers too (`execute_state_checkpoint_stores`): a first
        # use allocates pinned memory, which synchronizes.
        self._copies.staging.reserve([1, 2])
        self.checkpoint_spec = spec

    def execute_paged_state_copies(self, stores, restores, descriptor_slot: int = 0):
        self._copies.execute(stores, restores, descriptor_slot=descriptor_slot)

    def reserve_checkpoint_descriptors(self, descriptor_slots) -> None:
        self._copies.staging.reserve(descriptor_slots)

    def state_entry_views(self, slot):
        return [self._copies.entry(slot)]


def build_v41_state_checkpoint_copier(vllm_config, max_batch: int = 64):
    """The copier for this worker's bound pool, or None when checkpoints are off."""
    from atom.plugin.vllm.deepseek_v41_bridge import ensure_v41_proxy_bound

    cache, _slots = ensure_v41_proxy_bound(vllm_config)
    sizing = v41_state_checkpoint_sizing(vllm_config)
    return _PluginV41StateCopier(cache, sizing.spec, max_batch)


# ---------------------------------------------------------------------------
# Checkpoint layers, placement
# ---------------------------------------------------------------------------


class AtomDeepseekV41StateCheckpointBackend(PagedStateCheckpointBackend):
    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V41_STATE_CHECKPOINT"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [_block_size()]


class AtomDeepseekV41StateCheckpointLayer(PagedStateCheckpointLayer):
    """One PAGE unit of a V4.1 checkpoint, as vLLM sees it."""

    def __init__(self, prefix: str, part: int, page_bytes: int, adapter=None):
        super().__init__(
            prefix, part, page_bytes, adapter or register_v41_state_checkpoint_adapter()
        )


def v41_state_checkpoints_requested() -> bool:
    """Whether this run registers V4.1's state-checkpoint layers.

    The one owner of `ATOM_V41_STATE_CHECKPOINTS`. Both the model wrapper,
    which registers the layers, and the platform gate, which admits prefix
    caching on the promise that they exist, read it here -- two readings of
    one switch are two answers waiting to disagree, and the disagreement is a
    platform that admits prefix caching for a pool with no checkpoint groups.
    """
    from atom.utils import envs

    return envs.ATOM_V41_STATE_CHECKPOINTS


def v41_state_checkpoints_active(vllm_config) -> bool:
    """Whether checkpoints restore V4.1's STATE this run.

    The one predicate the platform gate and the model's runtime check share.
    Both halves: the switch registers the checkpoint layers, and
    `state_checkpoints_on` (any V4.1 run with prefix caching) is what uses them.
    """
    return v41_state_checkpoints_requested() and bool(
        state_checkpoints_on(register_v41_state_checkpoint_adapter(), vllm_config)
    )


def register_deepseek_v41_state_checkpoint_layers(vllm_config) -> list[str]:
    return register_state_checkpoint_layers(
        register_v41_state_checkpoint_adapter(),
        vllm_config,
        AtomDeepseekV41StateCheckpointLayer,
    )


@dataclass(frozen=True)
class V41CheckpointPlacement(StateCheckpointPlacement):
    """Native placement at V4.1's 256-token block."""

    block: int = 256

    @classmethod
    def from_env(
        cls, defaults=None, block: int | None = None
    ) -> V41CheckpointPlacement:
        return super().from_env(
            defaults or native_checkpoint_defaults(), block or _block_size()
        )


V41SchedulerStateCheckpoints = SchedulerStateCheckpoints


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def v41_drafter_in_target_slot(vllm_config) -> bool:
    """Whether a prefix hit may resume the configured drafter from a checkpoint.

    True for DSpark, and this is the one place V4.1 departs from #2502's V4
    adapter, so the argument is written out rather than assumed:

    A V4.1 checkpoint carries the window ring and the Engram cursor; the two
    compressor rings are `in_checkpoint=False` because a checkpoint sits on a
    PAGE boundary, a multiple of every ratio, so a resumer reads none of them
    (`v41_pool_geometry.py:318-319`). DSpark adds no state outside that --
    its drafts write the same rings, and the rings are sized
    `window_size + speculative_tokens` / `max_ratio + speculative_tokens`
    exactly so a rejected round's rows fall outside the window the next round
    reads (`attentions/deepseek_v41/speculative.py:4-6`). That is also why
    V4.1 needs no byte rollback: `TentativeState.commit` rewrites `cursor`
    alone.

    The consequence for the store, handled where the store is: a speculative
    step's own ends are the DRAFTED ends, so the checkpoint must be filed under
    the committed position instead, which exists only after
    `commit_tentative`.

    True with no drafter at all, for the same reason it is true with one.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return True
    method = getattr(spec, "method", None)
    return method in (None, "dspark")


class DeepseekV41StateCheckpointAdapter(PagedStateCheckpointAdapter):
    name = "DeepSeek-V4.1"
    layer_suffix = ATOM_DEEPSEEK_V41_STATE_CHECKPOINT_LAYER_SUFFIX
    block_size = 256
    backend_cls = AtomDeepseekV41StateCheckpointBackend
    state_checkpoint_drafters = "no drafter, or speculative method 'dspark'"
    # Under speculation a hit leaves two tokens (`prefill_tail`).
    speculative_prefill_tail = 2
    # Without checkpoints `deepseek_v41_state_reserve_patch` withholds the STATE
    # tail and auto-fits the length, as it did before this adapter existed.
    sizes_pool_without_checkpoints = False

    @property
    def proxy_layer_name(self) -> str:
        from atom.plugin.vllm.deepseek_v41_bridge import (
            ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
        )

        return ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME

    def matches(self, vllm_config) -> bool:
        """V4.1 with the switch on. Without it no checkpoint layer exists, and
        every hook keyed on this adapter (pool sizing, placement) would size
        and place checkpoints for groups nobody registered."""
        from atom.plugin.vllm.deepseek_v41_bridge import is_deepseek_v41_vllm_config

        return v41_state_checkpoints_requested() and bool(
            is_deepseek_v41_vllm_config(vllm_config)
        )

    def sizing(self, vllm_config) -> V41StateCheckpointSizing:
        return v41_state_checkpoint_sizing(vllm_config)

    def state_checkpoint_layer_names(self, k: int) -> list[str]:
        return deepseek_v41_state_checkpoint_layer_names(k)

    def drafter_state_in_checkpoint(self, vllm_config) -> bool:
        return v41_drafter_in_target_slot(vllm_config)

    def check_state_checkpoints_supported(self, vllm_config) -> None:
        """Refuse at startup what this design cannot serve. Never degrade.

        Each clause is a configuration under which vLLM would offer a prefix
        hit at a position no V4.1 forward left state at. The failure it
        replaces is `needs state at X, found Y` mid-run -- an engine death
        hundreds of requests after the mistake, with nothing pointing back at
        the flag that caused it.

        An absent attribute is a vLLM version difference rather than a
        misconfiguration, so each default is the value this design requires.
        Every problem is reported at once: fixing one and restarting to find
        the next costs a model load.
        """
        cache = vllm_config.cache_config
        block = _block_size()
        bad: list[str] = []

        mode = getattr(cache, "mamba_cache_mode", "align")
        if mode != "align":
            bad.append(
                f"mamba_cache_mode={mode!r}: the checkpoint groups are MambaSpecs and "
                "only 'align' writes their state at chunk ends (pass "
                "--mamba-cache-mode align)"
            )

        unit = getattr(cache, "prefix_match_unit", None)
        if unit is not None and int(unit) != block:
            bad.append(
                f"prefix_match_unit={unit} against a checkpoint block of {block}: a "
                "finer unit lets MambaManager serve a hit at a sub-block "
                "position, which is not a V4.1 PAGE edge and has no checkpoint"
            )

        retention = getattr(cache, "prefix_cache_retention_interval", 0) or 0
        if int(retention) != 0:
            bad.append(
                f"prefix_cache_retention_interval={retention}: the design relies "
                "on only the replay boundary being retained; a wider interval "
                "retains boundaries no chunk ended on"
            )

        if getattr(cache, "enable_mamba_shared_prefix_checkpoint", False):
            bad.append(
                "enable_mamba_shared_prefix_checkpoint=True: the shared-prefix "
                "junction is a second retained boundary this build does not "
                "land on"
            )

        if bad:
            raise ValueError(
                "DeepSeek-V4.1 checkpoints cannot be served under this "
                "configuration:\n  - " + "\n  - ".join(bad)
            )


_V41_ADAPTER = DeepseekV41StateCheckpointAdapter()


def register_v41_state_checkpoint_adapter() -> DeepseekV41StateCheckpointAdapter:
    return register_state_checkpoint_adapter(_V41_ADAPTER)


def current_state_checkpoint_worker(adapter=None):
    """This process's `StateCheckpointWorker` for V4.1, or None before the pool binds.

    `paged_state_checkpoint` keeps the workers in a private dict keyed by adapter
    name and only exposes a setter (`register_state_checkpoint_worker`); the
    V4.1 reader lives with the V4.1 adapter.
    """
    from atom.plugin.vllm.paged_state_checkpoint import _workers

    name = (adapter or register_v41_state_checkpoint_adapter()).name
    return _workers.get(name)


def apply_vllm_v41_prefix_install() -> None:
    """Register the V4.1 adapter and install every checkpoint hook. Idempotent.

    Called from `register_model`, which runs in EngineCore and in every
    worker. `ATOMPlatform.check_and_update_config` is not a reliable site:
    vLLM may never activate the platform.
    """
    register_v41_state_checkpoint_adapter()
    install()
    install_v41_store_sites()
    install_v41_state_checkpoint_group_ids()
    install_v41_slot_owner_release()
    install_v41_group_check_scope()
    from atom.plugin.vllm.deepseek_v41_bridge import is_deepseek_v41_vllm_config
    from atom.plugin.vllm.req_id_passthrough_patch import install_v2_host_block_rows

    # Only V4.1's runner reads the rows; every other model skips keeping them.
    install_v2_host_block_rows(
        lambda runner: is_deepseek_v41_vllm_config(getattr(runner, "vllm_config", None))
    )


def install_v41_group_check_scope() -> bool:
    """Scope the vendored group check to the pool the checkpoints actually live in.

    `check_state_checkpoint_groups` walks EVERY group in the config and requires the
    family's block size and page of each. That holds for the proxy and the
    checkpoint layers -- they are one overlaid pool -- but a DSpark draft group is
    a second, separate pool by design (`_build_v41_proxy_draft_groups`), with
    its own block size and page, and no checkpoint is ever stored in it. Left
    unscoped the check refuses the one combination the adapter advertises as
    supported (`state_checkpoint_drafters`), so checkpoints and speculative decoding could
    never run together.

    Done by wrapping the module attribute; the check itself is unchanged and
    still runs over the proxy and all k checkpoint groups. Upstream fix: have `check_state_checkpoint_groups`
    skip groups that hold neither the proxy nor a checkpoint layer.
    """
    import functools
    from types import SimpleNamespace

    from atom.plugin.vllm import paged_state_checkpoint as PSI

    original = PSI.check_state_checkpoint_groups
    if getattr(original, "_atom_v41_scoped", False):
        return False

    @functools.wraps(original)
    def check_state_checkpoint_groups(cfg, adapter, sizing):
        groups = list(cfg.kv_cache_groups)
        proxy_name = adapter.proxy_layer_name
        ours = [
            g
            for g in groups
            if any(
                name.endswith(proxy_name) or adapter.is_state_checkpoint_layer(name)
                for name in g.layer_names
            )
        ]
        if len(ours) == len(groups):
            return original(cfg, adapter, sizing)
        scoped = SimpleNamespace(
            kv_cache_groups=ours,
            kv_cache_tensors=cfg.kv_cache_tensors,
            num_blocks=cfg.num_blocks,
        )
        return original(scoped, adapter, sizing)

    check_state_checkpoint_groups._atom_v41_scoped = True
    PSI.check_state_checkpoint_groups = check_state_checkpoint_groups
    return True


def install_v41_slot_owner_release() -> None:
    """Release the bridge's slot-owner records with the slots.

    `_apply_runner_patches` hands every runner's finished, preempted and
    resumed ids to the registered slot allocators; registered among them, the
    records go when the request does, so a request id that comes back does
    not inherit its old slot.
    """
    from atom.plugin.vllm import paged_state_checkpoint as PSI
    from atom.plugin.vllm.deepseek_v41_bridge import V41_SLOT_OWNER_RECORDS

    PSI.register_slot_allocator(V41_SLOT_OWNER_RECORDS)


def install_v41_state_checkpoint_group_ids() -> int:
    """Record each checkpoint layer's KV cache group id on whichever runner runs.

    Same gap as `install_v41_store_sites`, on a different site and with a
    worse failure: the vendored installer wraps
    `vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_kv_cache` to
    write `_atom_state_checkpoint_group_id`, but DeepSeek-V4.1 runs on the V2 runner, so
    the write never happened and the hook still reported success -- the checkpoint
    worker then refused to start, with the pool already sized for checkpoints.

    Written BEFORE the original runs, not after as the vendored hook does:
    V4.1 binds its proxy pool (and stands up the checkpoint worker) inside
    `initialize_kv_cache`, so a write afterwards is already too late. The id
    comes from this worker's own `kv_cache_config` argument, which is what the
    vendored hook reads too, so both writes agree.
    """
    import functools

    from atom.plugin.vllm.paged_state_checkpoint import PagedStateCheckpointLayer
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    patched = 0
    for cls in gpu_model_runner_classes():
        original = getattr(cls, "initialize_kv_cache", None)
        if original is None or getattr(original, "_atom_v41_checkpoint_gids", False):
            continue

        @functools.wraps(original)
        def initialize_kv_cache(self, kv_cache_config, *args, _orig=original, **kwargs):
            sfc = self.compilation_config.static_forward_context
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                for name in group.layer_names:
                    layer = sfc.get(name)
                    if isinstance(layer, PagedStateCheckpointLayer):
                        layer._atom_state_checkpoint_group_id = gid
            return _orig(self, kv_cache_config, *args, **kwargs)

        initialize_kv_cache._atom_v41_checkpoint_gids = True
        cls.initialize_kv_cache = initialize_kv_cache
        patched += 1
    return patched


def install_v41_store_sites() -> int:
    """Put the store sites on whichever model runner this vLLM ships.

    The vendored mechanism's own installer hardcodes
    `vllm.v1.worker.gpu_model_runner.GPUModelRunner` -- the V1 runner -- and
    DeepSeek-V4.1 runs on the V2 one. `install_store_sites` takes the class,
    so this needs no edit to the vendored file; `gpu_model_runner_classes()`
    is the same enumeration `v41_speculative.py` already patches over.

    The V2 runner has no `_model_forward` at all: its forward is inlined in
    `execute_model`. A `getattr(cls, "_model_forward", None)` guard therefore
    skipped the only runner V4.1 uses, silently -- the scheduler planned an
    checkpoint at the anchor, nothing ever wrote it, and the next request's hit on
    that boundary died with `needs state at N, found 1`. So wrap
    `execute_model` there instead: it returns after the step's last forward,
    on the same stream and outside any capture, which is what the store site
    requires.
    """
    import functools

    from atom.plugin.vllm.paged_state_checkpoint import (
        _workers,
        flush_worker,
        install_store_sites,
    )
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    patched = 0
    for cls in gpu_model_runner_classes():
        original = getattr(cls, "_model_forward", None)
        if original is not None:
            if getattr(original, "_atom_state_checkpoint_patched", False):
                continue
            install_store_sites(cls)
            patched += 1
            continue
        original = getattr(cls, "execute_model", None)
        if original is None or getattr(original, "_atom_v41_store_site", False):
            continue

        @functools.wraps(original)
        def execute_model(self, *args, _orig=original, **kwargs):
            output = _orig(self, *args, **kwargs)
            # Flush here, drafter or not -- do NOT defer to the next build the
            # way the vendored V1 site does. The V2 runner has no
            # `_model_forward`: `execute_model` returns after the target
            # forward, and `sample_tokens` then samples, commits a speculative
            # step and runs the drafter. For a step that did not speculate the
            # slot is final here -- the drafter writes past the window the
            # checkpoint holds (`v41_drafter_in_target_slot`) -- and deferring
            # instead carries the store across the NEXT step's forward. (V4.1
            # refuses pipeline parallelism, so every rank reaches the commit.)
            #
            # Measured with DSpark on, before this: every checkpoint held the state
            # of the following boundary -- planned 2560 stored at cursor 3072,
            # planned 3072 at 3584, planned 3584 at 3630 -- and a later hit on
            # one of those boundaries got back a state one checkpoint ahead.
            # ... unless a speculative step is still outstanding. DSpark's
            # accepted prefix is committed by `postprocess_sampled`, which
            # runs after this returns, and a store issued before that commit
            # reaches `StateCopies._validate` -> `cache.require_committed()`
            # and raises "Commit the accepted prefix before reusing or
            # checkpointing state". The speculative wrapper flushes instead,
            # immediately after it commits.
            from atom.plugin.vllm.deepseek_v41_bridge import (
                v41_speculative_step_is_pending,
            )

            if not v41_speculative_step_is_pending():
                for worker in list(_workers.values()):
                    flush_worker(worker)
            return output

        execute_model._atom_v41_store_site = True
        cls.execute_model = execute_model
        patched += 1
    return patched
