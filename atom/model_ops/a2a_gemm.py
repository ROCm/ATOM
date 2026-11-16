# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 SP4 attention output + ``o_proj`` through ``aiter.a2a_gemm_fused``.

The frozen neutral contract (aiter owns it, version 1, fingerprint
``388d9816496c260e``) is::

    a2a_gemm_fused(x, x_scale, w, w_scale, out, total_tokens=...,
                   bucket_m=..., registered=..., group=..., stream=...)

on gfx950 with SP=4 / TP=1, one node. ``x`` is this rank's **pre-head-exchange**
FP8 send buffer ``[4*M_real, 2048]``: the BF16 attention output for this rank's
head shard over the padded global token rows, quantized with one scale per
token over all 8192 channels (exactly ATOM's existing o-proj quantizer
contract). ``x_scale`` is those scales ``[M_real]``; ``w``/``w_scale`` the
replicated 16x16-preshuffled fp8 ``o_proj`` weight and its per-channel scale;
``out`` the caller-owned BF16 ``[bucket_m, 6144]`` result; FP32 accumulation,
no bias. The kernel folds the SP head exchange into the GEMM: rows
``[d*M_real:(d+1)*M_real]`` of ``x`` are destination rank ``d``'s token block.

This module is the *consumer*. It binds the op behind a default-OFF config
switch and, for a covered call class, replaces the attention tail -- the head
exchange (`sp_attention.gather_heads`) followed by `o_proj` -- with one routed
call, at the pre-exchange seam ATOM already owns
(`sp_attention.ulysses_attention(..., output_transform=...)`).

The tail is one fixed-schema opaque custom op
(`torch.ops.aiter.a2a_gemm_attend_project`, registered by
``mark_spliting_op`` with a fake impl), so Dynamo traces the attention forward
without a graph break and CUDA graphs can capture/replay the tail:

* **torch.compile**: the model-side branch is static (site installed + shape/
  dtype metadata); the opaque op decides fused-vs-native at runtime, so the
  fused path serves under piecewise compilation too.
* **CUDA graph capture**: every per-layer/per-M decision and the provider
  pre-bind happen in the eager capture-prep phase (rank-uniformly), never
  inside capture. For each captured M class the site allocates address-stable
  send/scale/output cells once, runs the routed call eagerly once (the
  provider binds that class on those pointers) and only then lets capture
  record the same call: capture sees a decision-cache hit plus a bound launch,
  no allocation/bind/sync/collective new decision, and replay repeats it.
  Every un-pre-bound class is decided native on *all* ranks before capture, so
  no rank-open risk appears inside the graph.

Everything is fail-open: no provider interface, no routed op, a provider
decline, an uncovered M class, a missing registered producer view for an
in-band class, a failed pre-bind, or a shape the op does not serve keeps
ATOM's existing composition unchanged, with one warning. Once a routed call
starts an error propagates (aiter never retries native).

Enablement is the aiter provider rail (`atom/aiter_provider.py`): the
``ModelRunner`` calls ``aiter.plugin.configure(enable=["a2a-gemm-provider"])``
once per worker before the model is built, installs these sites after weight
load and before warm-up/CUDA-graph capture, and warms the provider up over the
bound ``o_proj`` shapes. The class table itself is agreed once at install: the
provider's cheap ``covers(M)`` predicate (or its verified coverage table)
drives a single ``all_gather_object`` vote, and the capture M classes are
pre-bound under one more rank-uniform vote in the eager capture-prep phase.
"""

import logging

import torch

from atom.utils import mark_spliting_op

logger = logging.getLogger("atom")

# -- the frozen neutral-op identity (aiter owns the contract) -----------------
_OP_NAME = "a2a_gemm_fused"
_OP_VERSION = 1
_FINGERPRINT = "388d9816496c260e"

# -- the frozen gfx950 SP4 / TP1 deployment ------------------------------------
_SP_WORLD = 4
_TP_SHARDS = 1
_SHARD = 2048  # pre-exchange x width: 8192 channels / four head shards
_K_IN = 8192  # o_proj input width (all attention heads, SP4)
_N_OUT = 6144  # o_proj output width (hidden size)
_M_MAX = 16384  # provider capacity bound
_REGISTERED_M_MIN = 2048  # producer-registered ingress band (real M)
_REGISTERED_M_MAX = 8192
_ROWS_PER_M = 4  # pre-exchange x rows per real local token

_FP8 = torch.float8_e4m3fn

# One shared caller-owned output per (device, real M) cell. Layers run one
# after another on the same stream and consume their o_proj result before the
# next layer calls it, so the buffer may be reused. A capture cell also owns
# its address-stable send/scale buffers, pre-allocated before capture.
_OUTPUTS: dict[tuple, torch.Tensor] = {}
_OUTPUT_LIMIT_BYTES = 1 << 30

# Non-capture eager scratch, cached per (device, M)/padded shape for the same
# reason as the capture cells: the provider keys a binding on the send/scale
# pointers, so a fresh per-call tensor would miss the binding and re-run its
# collective SP4 bind on every layer/prefill chunk. One buffer per class is
# overwritten in place every call (content changes, pointers do not).
_EAGER_SEND: dict[tuple, torch.Tensor] = {}
_EAGER_SCALE: dict[tuple, torch.Tensor] = {}

# Capture epoch: begins at every capture phase (ModelRunner.begin_capture).
# Pre-bound cells/bindings from an earlier epoch are dropped so a recapture
# (sleep/wake, provider session reset) never replays stale pointers.
_CAPTURE_EPOCH = 0

# Per-(device, rows) zero BF16 pre-exchange x used only by the eager pre-bind
# call: the quantizer needs an x, its pointer never reaches the provider (the
# routed call's x is the FP8 send buffer), and it keeps pre-bind allocation-free
# on the post-capture path.
_WARM_X: dict[tuple, torch.Tensor] = {}

# "capture-served M=.." / replay counters; the replay log is what Phase F and
# gpu-runner2 read to prove a captured graph actually replays the fused tail.
_REPLAY_LOGGED: set[int] = set()
_REPLAY_COUNT = 0

# Pointers ATOM has taken as its own registered producer views: `registered=True`
# is only passed for a tensor in this set (the provider honors it only then, and
# an in-band class without its producer view declines rather than stage-flips
# the verified kernel).
_PRODUCER_PTRS: set[int] = set()

# layer name -> installed site (the model-side gate/execution hand-off).
_SITES: dict[str, "A2AGemmSite"] = {}

# Process-local provider lifecycle state (one provider session per worker).
_PLUGIN = None
_ENSURE_DONE = False


# ---------------------------------------------------------------------------
# lazy aiter / distributed access (module scope imports neither)
# ---------------------------------------------------------------------------
def _aiter():
    import aiter

    return aiter


def _aiter_plugin():
    global _PLUGIN
    if _PLUGIN is None:
        import importlib

        try:
            _PLUGIN = importlib.import_module("aiter.plugin")
        except Exception:  # noqa: BLE001 - interface absent/not a package in tests
            try:
                _PLUGIN = getattr(_aiter(), "plugin", None)
            except Exception:  # noqa: BLE001 - no aiter at all
                _PLUGIN = None
    return _PLUGIN


def _pcp_group():
    """The SP group (AITER's PCP rank dimension) as the fused op executes it."""
    from aiter.dist.parallel_state import get_pcp_group

    return get_pcp_group()


def _group_or_none():
    try:
        return _pcp_group().device_group
    except Exception:  # noqa: BLE001 - CPU/test process, or no SP interface
        return None


def _sp_attention():
    """The SP attention module: the pre-exchange seam and its quantizer."""
    from atom.model_ops import sp_attention

    return sp_attention


def _transport():
    """The pre-exchange FP8 transport (amax gather + ATOM's quantizer)."""
    from atom.model_ops.minimax_m3 import attention_fp8

    return attention_fp8


def _gfx_runtime() -> str | None:
    """The device's gfx name, or None in a CPU process / without aiter."""
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime()
    except Exception:  # noqa: BLE001 - no aiter/device
        return None


def _device_index() -> int | None:
    try:
        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - CPU/test process
        return None


def _in_hipgraph() -> bool:
    """Whether the current forward runs under a CUDA-graph capture.

    ``in_hipgraph`` is a ``ForwardContext`` field (the ``Context`` object holds
    only the step's own state), so read it off the context object itself.
    """
    try:
        from atom.utils.forward_context import get_forward_context

        return bool(getattr(get_forward_context(), "in_hipgraph", False))
    except Exception:  # noqa: BLE001 - no context means ordinary eager
        return False


def _is_dummy_run() -> bool:
    """Whether the current forward is a dummy/warm-up run.

    ``is_dummy_run`` is a ``Context`` field (the forward context repeats it
    nowhere), unlike ``in_hipgraph``.
    """
    try:
        from atom.utils.forward_context import get_forward_context

        return bool(getattr(get_forward_context().context, "is_dummy_run", False))
    except Exception:  # noqa: BLE001 - no context means ordinary eager
        return False


def _is_compiling() -> bool:
    """Whether torch.compile is tracing this forward (fusion is eager-only)."""
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - older torch without the helper
        return False


def _eager_fused_enabled() -> bool:
    """Whether the operator opted into fusing the non-captured/eager tail.

    Default off (``ATOM_A2A_GEMM_EAGER_FUSED=0``): captured decode graphs are
    the advertised mode, while non-captured prefill stays native because the
    provider's per-new-M-class preparation is expensive in-model. Read lazily so
    an operator can set it before the first forward.
    """
    try:
        from atom.utils import envs

        return bool(envs.ATOM_A2A_GEMM_EAGER_FUSED)
    except Exception:  # noqa: BLE001 - envs always present; stay fail-open
        return False


def _on_device(tensor) -> bool:
    """Whether a tensor lives on a device (the fused op is CUDA-only).

    Separate from the call-shape checks so CPU tests can drive the whole gate
    and the routed call with mocked device tensors.
    """
    return bool(getattr(tensor, "is_cuda", False))


# ---------------------------------------------------------------------------
# static deployment / site gates
# ---------------------------------------------------------------------------
def deployment_reason(config) -> str | None:
    """Why this deployment cannot use the frozen op, or None.

    The fused kernel is frozen for gfx950, SP=4 and one TP shard on one node;
    anything else keeps ATOM's existing composition.
    """
    if int(getattr(config, "sequence_parallel_size", 1) or 1) != _SP_WORLD:
        return "sp-world"
    if int(getattr(config, "tensor_parallel_size", 1) or 1) != _TP_SHARDS:
        return "tp-world"
    if int(getattr(config, "pipeline_parallel_size", 1) or 1) != 1:
        return "pipeline-parallel"
    if int(getattr(config, "data_parallel_size", 1) or 1) != 1:
        return "data-parallel"
    gfx = _gfx_runtime()
    if gfx != "gfx950":
        return "gfx" if gfx is not None else "gfx-unknown"
    return None


def static_weight_reason(linear) -> str | None:
    """Why this layer's ``o_proj`` is not the frozen [6144, 8192] FP8 weight, or None."""
    w = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    if w is None or scale is None:
        return "weight-missing"
    if getattr(w, "dtype", None) != _FP8 or w.dim() != 2:
        return "weight-dtype"
    if tuple(w.shape) != (_N_OUT, _K_IN):
        return "weight-shape"
    if getattr(w, "is_shuffled", False) is not True:
        return "weight-layout"
    if not w.is_contiguous():
        return "weight-contiguous"
    if getattr(scale, "dtype", None) != torch.float32 or scale.numel() != _N_OUT:
        return "weight-scale"
    if not scale.is_contiguous():
        return "weight-scale-contiguous"
    if int(getattr(linear, "input_size", _K_IN)) != _K_IN:
        return "input-size"
    if int(getattr(linear, "output_size", _N_OUT)) != _N_OUT:
        return "output-size"
    if int(getattr(linear, "tp_size", 1) or 1) != 1:
        return "tp-shard"
    return None


def _consumes_per_token_fp8(linear) -> bool:
    """Whether the linear consumes the attention's per-token FP8 tuple."""
    quant_type = getattr(linear, "quant_type", None)
    quant_value = getattr(quant_type, "value", quant_type)
    # QuantType.per_Token is 2 in every ATOM/aiter generation (see linear.py).
    try:
        from atom.model_ops.linear import QuantType

        per_token = QuantType.per_Token.value
    except Exception:  # noqa: BLE001 - CPU test stub without the enum
        per_token = 2
    return quant_value == per_token and getattr(linear, "params_dtype", None) == _FP8


def _contract_weight_scale(scale):
    """The frozen v1 contract's contiguous fp32 ``[6144]`` weight scale, or None.

    The per-token FP8 ``o_proj`` keeps ATOM's per-channel scale as ``(N, 1)``;
    the frozen op requires a contiguous ``(N,)`` and its native reference
    raises ``w_scale must have shape (6144,), got (6144, 1)`` otherwise. A
    contiguous ``(N, 1)`` flattens to a zero-copy view, so the served path adds
    no copy; a scale that cannot be normalized is a native fallback before the
    call, never an ATOM failure.
    """
    if scale is None or getattr(scale, "dtype", None) != torch.float32:
        return None
    try:
        flat = scale.reshape(-1)
    except Exception:  # noqa: BLE001 - exotic/non-tensor scale
        return None
    if int(flat.numel()) != _N_OUT:
        return None
    flat = flat.contiguous()
    if not flat.is_contiguous() or int(flat.numel()) != _N_OUT:
        return None
    return flat


# ---------------------------------------------------------------------------
# provider coverage: the cheap, non-executing covers predicate / class table
# ---------------------------------------------------------------------------
def _normalize_coverage(provider) -> tuple[dict[int, int], set[int]] | None:
    """The provider's frozen *verified* direct call classes, or None.

    ``{M_real: bucket_M}`` plus the registered-ingress subset, from the
    provider's stdlib-only ``coverage()`` -- no kernel, no bind, nothing
    executed. Only direct bucket classes are kept (``M_real == bucket_M``):
    ATOM never stages a padded real M, so Pad seams stay native. A provider
    that no longer ships an explicit table may expose ``covers(M)`` instead
    (see `_provider_covers`); a missing/broken table is then not fatal.
    """
    coverage = getattr(provider, "coverage", None)
    if not callable(coverage):
        return None
    try:
        raw = coverage()
    except Exception:  # older/other provider: no static table
        logger.debug("a2a_gemm: provider coverage() unavailable", exc_info=True)
        return None
    direct: dict[int, int] = {}
    registered: set[int] = set()
    try:
        for item in raw or ():
            if isinstance(item, (tuple, list)):
                m_real, bucket_m, ingress = int(item[0]), int(item[1]), item[2]
            else:
                m_real = int(item.m_real)
                bucket_m = int(item.bucket_m)
                ingress = item.ingress
            if bucket_m != m_real:
                continue  # Pad seam: ATOM does not pad a real M
            if not 1 <= m_real <= _M_MAX:
                continue
            direct[m_real] = bucket_m
            if str(ingress) == "registered":
                registered.add(m_real)
    except Exception:  # malformed interface: no usable table
        logger.debug("a2a_gemm: unusable provider coverage()", exc_info=True)
        return None
    return direct, registered


def _provider_covers(provider):
    """The provider's cheap ``covers(M_real) -> bool`` predicate, or None.

    Newer providers serve arbitrary M 1..16384 through the same frozen rule
    (the 82+7 class table becomes the independently verified subset), so they
    expose a per-M covers predicate instead of a policy-exhaustive list. It is
    metadata-only, like ``select``'s first half: no kernel, no bind, no
    allocation.
    """
    covers = getattr(provider, "covers", None)
    return covers if callable(covers) else None


def _provider_policy(provider):
    """A rank-comparable identity of the provider's serving policy, or None.

    The rank-uniform install vote must agree the *policy*, which need not be an
    explicit class list any more. Providers may name it (an attribute or a
    ``policy_version()`` callable); the covers/table mode is part of the identity
    regardless. Only a value that is itself rank-comparable may be voted on: a
    bound method's ``str`` carries its address, so callables are resolved first.
    """
    for attr in ("policy_version", "coverage_version", "policy", "version"):
        value = getattr(provider, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:  # an unusable identity is not voted
                logger.debug("a2a_gemm: provider %s() failed", attr, exc_info=True)
                continue
            if value is None:
                continue
        try:
            return (attr, str(value))
        except Exception:  # noqa: BLE001 - exotic identity object
            return (attr, repr(value))
    return None


def registered_band(m: int) -> bool:
    """Whether this real M is in the frozen producer-registered ingress band."""
    return _REGISTERED_M_MIN <= int(m) <= _REGISTERED_M_MAX


def agree_class_table(payload: dict, group) -> dict | None:
    """Agree one deployment/class-table payload across the SP4 ranks, once.

    Returns the agreed payload, or None when the ranks disagree (or the
    collective fails): then every rank keeps ATOM's native path for the whole
    run. Called at install time only -- never in a forward, never in capture.
    """
    if not payload.get("deployment"):
        return None
    if group is None:
        # Single-rank/CPU process: the local verdict is the decision.
        return payload
    try:
        import torch.distributed as dist

        world = int(dist.get_world_size(group))
        gathered: list = [None] * world
        dist.all_gather_object(gathered, payload, group=group)
    except Exception:
        logger.warning(
            "a2a_gemm: rank-uniform coverage vote failed; native path", exc_info=True
        )
        return None
    if len(gathered) != _SP_WORLD or any(item is None for item in gathered):
        logger.warning("a2a_gemm: rank-uniform coverage vote incomplete; native path")
        return None
    first = gathered[0]
    if any(item != first for item in gathered[1:]):
        logger.warning(
            "a2a_gemm: SP4 ranks disagree on the routed class table; native path"
        )
        return None
    return first


# ---------------------------------------------------------------------------
# producer ingress
# ---------------------------------------------------------------------------
def _mark_producer(tensor) -> None:
    _PRODUCER_PTRS.add(int(tensor.data_ptr()))


def owns_producer_pointer(tensor) -> bool:
    """Whether ATOM's producer wrote a provider-registered view at this address."""
    return int(tensor.data_ptr()) in _PRODUCER_PTRS


def _output_buffer(device, m: int) -> torch.Tensor:
    """The shared caller-owned ``[m, 6144]`` BF16 output of one M cell."""
    key = (str(device), int(m))
    out = _OUTPUTS.get(key)
    if out is None:
        out = torch.empty((int(m), _N_OUT), dtype=torch.bfloat16, device=device)
        _OUTPUTS[key] = out
        live = sum(t.numel() * t.element_size() for t in _OUTPUTS.values())
        while live > _OUTPUT_LIMIT_BYTES and len(_OUTPUTS) > 1:
            _, dropped = _OUTPUTS.popitem()
            live -= dropped.numel() * dropped.element_size()
    return out


def _eager_send_buffer(device, rows: int) -> torch.Tensor:
    """The address-stable FP8 send buffer of one eager (device, padded rows)."""
    key = (str(device), int(rows))
    q = _EAGER_SEND.get(key)
    if q is None:
        q = torch.empty((int(rows), _SHARD), dtype=_FP8, device=device)
        _EAGER_SEND[key] = q
        live = sum(t.numel() * t.element_size() for t in _EAGER_SEND.values())
        while live > _OUTPUT_LIMIT_BYTES and len(_EAGER_SEND) > 1:
            _, dropped = _EAGER_SEND.popitem()
            live -= dropped.numel() * dropped.element_size()
    return q


def _eager_scale_buffer(device, m: int) -> torch.Tensor:
    """The address-stable FP32 ``[m, 1]`` per-token scale of one eager M class."""
    key = (str(device), int(m))
    scale = _EAGER_SCALE.get(key)
    if scale is None:
        scale = torch.empty((int(m), 1), dtype=torch.float32, device=device)
        _EAGER_SCALE[key] = scale
    return scale


def _is_cuda_capturing() -> bool:
    """Whether torch is recording a CUDA graph right now (host-side check)."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - CPU/test process
        return False


def _warm_x(device, rows: int) -> torch.Tensor:
    """Address-stable zero BF16 pre-exchange x for the eager capture pre-bind.

    Only its shape feeds the quantizer; the routed call's ``x`` is the FP8 send
    buffer, so this tensor never reaches the provider and one per (device, rows)
    is enough.
    """
    key = (str(device), int(rows))
    x = _WARM_X.get(key)
    if x is None:
        x = torch.zeros((int(rows), _SHARD), dtype=torch.bfloat16, device=device)
        _WARM_X[key] = x
    return x


def agree_capture_classes(local: dict, group) -> dict | None:
    """Agree one capture M class' per-layer pre-bind eligibility across ranks.

    One ``all_gather_object`` in the eager capture-prep phase. Returns
    ``{layer_name: bool}`` (True only where every rank can pre-bind); a failed
    or incomplete vote returns None so every rank falls back to native for the
    class, never a rank-divergent capture graph.
    """
    if group is None:
        # Single-rank/CPU process: the local verdict is the decision.
        return dict(local)
    try:
        import torch.distributed as dist

        world = int(dist.get_world_size(group))
        gathered: list = [None] * world
        dist.all_gather_object(gathered, dict(local), group=group)
    except Exception:
        logger.warning(
            "a2a_gemm: capture pre-bind vote failed; class stays native",
            exc_info=True,
        )
        return None
    if len(gathered) != _SP_WORLD or any(item is None for item in gathered):
        logger.warning("a2a_gemm: capture pre-bind vote incomplete; class stays native")
        return None
    return {name: all(bool(item.get(name)) for item in gathered) for name in local}


# ---------------------------------------------------------------------------
# the site
# ---------------------------------------------------------------------------
class _CaptureCell:
    """Address-stable buffers of one captured M class.

    The provider keys its prepared call and its per-class decision on these
    pointers, so they are allocated once in the eager capture-prep phase and
    reused verbatim by every capture recording and replay.
    """

    __slots__ = ("m", "out", "q", "registered", "scale")

    def __init__(self, m, q, scale, out, registered):
        self.m = int(m)
        self.q = q
        self.scale = scale
        self.out = out
        self.registered = bool(registered)


class A2AGemmSite:
    """One MiniMax-M3 attention layer's fused attention-tail + o_proj.

    One fixed-schema opaque custom op owns the whole tail, so the same site
    serves under eager, torch.compile and CUDA-graph capture/replay. Capture
    never decides or binds: `capture_prebind` does that eagerly, before the
    graph is recorded, on the address-stable `_CaptureCell` of each class.
    """

    def __init__(self, name, attn, linear, config, provider, agreed, covers=None):
        self.name = name
        self.attn = attn
        self.linear = linear
        self.config = config
        self.provider = provider
        self.covers = covers
        self.direct = dict(agreed.get("direct") or {})
        self.registered = frozenset(agreed.get("registered") or ())
        self.ready = True
        self._group = None
        self._logged: set[str] = set()
        self._producer_views: dict[int, torch.Tensor] = {}
        # Capture state, reset at every capture phase (epoch).
        self._capture_epoch = -1
        self._capture_cells: dict[int, _CaptureCell] = {}
        self._capture_classes: frozenset[int] = frozenset()
        self._capture_native: set[int] = set()
        self.capture_prebound = 0
        self.capture_served = 0

    # -- logging ---------------------------------------------------------
    def _log_once(self, key, msg, *args):
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg, *args)

    def _world(self) -> int:
        try:
            return int(_pcp_group().world_size)
        except Exception:  # noqa: BLE001 - no SP interface: keep the frozen world
            return _SP_WORLD

    def covered(self, m: int) -> bool:
        """Whether the provider serves this real M (cheap, nothing executed).

        Prefers the provider's per-M ``covers(M)`` predicate (the generalized
        policy: arbitrary M through the frozen rule, with the 82+7 table as the
        independently verified subset); a provider that only ships the old
        static table keeps working through the table hit. A predicate that
        raises is a decline for that call, never an ATOM failure -- but the
        static table is still consulted as the fallback.
        """
        covers = self.covers
        if covers is not None:
            try:
                return bool(covers(int(m)))
            except Exception:  # predicate failure = the static table fallback
                logger.debug(
                    "a2a_gemm %s: covers(%d) failed; falling back to the static table",
                    self.name,
                    m,
                    exc_info=True,
                )
        return int(m) in self.direct

    def can_serve(self, m: int) -> bool:
        """Whether this M class can take the routed op (provider view included)."""
        if not self.covered(int(m)):
            return False
        return not (registered_band(int(m)) and self._producer_view(int(m)) is None)

    @property
    def capture_classes(self) -> frozenset[int]:
        return self._capture_classes

    # -- capture lifecycle -------------------------------------------------
    def begin_capture(self, epoch: int) -> None:
        """Start a capture phase: drop pre-bind state from an earlier epoch."""
        self._capture_epoch = int(epoch)
        self._capture_cells.clear()
        self._capture_classes = frozenset()
        self._capture_native.clear()
        self._producer_views.clear()
        self.capture_prebound = 0
        self.capture_served = 0

    def mark_capture_native(self, m: int) -> None:
        """Record the rank-uniform decision that this capture class is native."""
        self._capture_native.add(int(m))
        self._capture_cells.pop(int(m), None)
        self._capture_classes = frozenset(self._capture_cells)

    def capture_prebind(self, m: int) -> bool:
        """Eagerly bind one capture M class on address-stable buffers.

        Runs before any ``torch.cuda.graph`` recording: allocate the cell, run
        the routed call once on the zero pre-exchange x so the provider binds
        this class on the cell's pointers, and verify the provider (not aiter's
        native reference) served it. False leaves the class native everywhere.
        """
        m = int(m)
        if _is_cuda_capturing():
            logger.warning(
                "a2a_gemm %s: capture pre-bind of M=%d requested inside graph "
                "capture; class stays native",
                self.name,
                m,
            )
            return False
        if not self.can_serve(m):
            return False
        padded = _ROWS_PER_M * m
        device = self.linear.weight.device
        if registered_band(m):
            q = self._producer_view(m)
            if q is None:
                return False
            registered = True
        else:
            q = torch.empty((padded, _SHARD), dtype=_FP8, device=device)
            registered = False
        cell = _CaptureCell(
            m,
            q,
            torch.empty((m, 1), dtype=torch.float32, device=device),
            _output_buffer(device, m),
            registered,
        )
        try:
            out = self._serve(
                x=_warm_x(device, padded), padded_tokens=padded, cell=cell
            )
        except Exception:  # pre-bind failure = that class stays native, fail-open
            logger.warning(
                "a2a_gemm %s: capture pre-bind of M=%d failed; class stays native",
                self.name,
                m,
                exc_info=True,
            )
            return False
        if out is None or not self._provider_served():
            logger.warning(
                "a2a_gemm %s: capture pre-bind of M=%d was not served by the "
                "provider; class stays native",
                self.name,
                m,
            )
            return False
        self._capture_cells[m] = cell
        self._capture_classes = frozenset(self._capture_cells)
        self.capture_prebound += 1
        logger.info("a2a_gemm capture prebind layer=%s M=%d", self.name, m)
        # gpu-runner2 also matches the #3084 wording.
        logger.info(
            "provider capture %s mode=o_proj m=%d: fused (prepare)", self.name, m
        )
        return True

    def _provider_served(self) -> bool:
        """Whether the last routed call was served by the provider (not native).

        The aiter plugin interface records every decision; when it cannot be
        queried (older interface) the pre-bind is trusted, since ``warmup`` and
        ``covers`` already proved the class and AITER_PLUGINS=0 disables the
        provider before this seam is installed.
        """
        plugin = _aiter_plugin()
        get_decision = getattr(plugin, "last_decision", None) if plugin else None
        if not callable(get_decision):
            return True
        try:
            decision = get_decision()
        except Exception:  # noqa: BLE001 - observability only
            return True
        if decision is None:
            # Nothing recorded: routing was never enabled in this process.
            return False
        return str(getattr(decision, "route", "")) == "provider"

    # -- execution ---------------------------------------------------------
    def fuse(self, attn, o_proj, q, k, v, positions, qkv):
        """Run the attention tail; the transform picks fused vs native."""
        sp = _sp_attention()
        result = sp.ulysses_attention(
            self.attn,
            q,
            k,
            v,
            positions,
            None,
            qkv,
            output_transform=self._output_transform,
        )
        return result[0] if isinstance(result, tuple) else result

    def attend_and_project(self, q, k, v, positions, qkv):
        """The opaque custom op's body (eager, compiled and captured alike)."""
        return self.fuse(self.attn, self.linear, q, k, v, positions, qkv)

    def _native_tail(self, x, padded_tokens):
        """ATOM's existing composition: head exchange + FP8 o_proj."""
        sp = _sp_attention()
        world = self._world()
        scale = x.new_empty((int(padded_tokens) // world, 1), dtype=torch.float32)
        fp8_output = sp.gather_heads(
            x, int(padded_tokens), self.attn.sp_head_exchange, scale
        )
        return (self.linear(fp8_output, scale), None)

    def _output_transform(self, x, padded_tokens):
        """Pre-exchange tail: amax gather + quantize + one routed call.

        ``x`` is the attention impl's BF16 output for this rank's head shard
        over the global (padded) token rows. Under CUDA-graph capture only a
        class pre-bound for this capture epoch may take the routed op (its
        address-stable cell); every other class -- decided rank-uniformly in the
        eager capture-prep phase -- stays on ATOM's composition.
        """
        world = self._world()
        padded_tokens = int(padded_tokens)
        m = padded_tokens // world
        if (
            not _on_device(x)
            or x.dim() != 2
            or x.dtype != torch.bfloat16
            or x.stride(1) != 1
            or x.shape[1] != _SHARD
            or padded_tokens % world
            or padded_tokens != _ROWS_PER_M * m
        ):
            return self._native_tail(x, padded_tokens)
        if _in_hipgraph():
            cell = self._capture_cells.get(m)
            if cell is None:
                self._log_once(
                    "capture-native-" + str(m),
                    "a2a_gemm %s: capture-native M=%d (not pre-bound)",
                    self.name,
                    m,
                )
                return self._native_tail(x, padded_tokens)
            out = self._serve(x=x, padded_tokens=padded_tokens, cell=cell)
            if out is None:  # pragma: no cover - pre-bound classes always serve
                return self._native_tail(x, padded_tokens)
            self.capture_served += 1
            self._log_once(
                "capture-served-" + str(m),
                "a2a_gemm %s: capture-served M=%d",
                self.name,
                m,
            )
            return (out, None)
        if _is_dummy_run() or not self.can_serve(m):
            return self._native_tail(x, padded_tokens)
        if not _eager_fused_enabled():
            # Deployment default: the advertised fused mode is the captured
            # decode graph. A non-captured prefill would pay the provider's
            # collective SP4 bind on the first call of every new M class per
            # layer (seconds each), which turns P99 TTFT into minutes -- so
            # non-captured calls stay on ATOM's composition unless the operator
            # explicitly opts in with ATOM_A2A_GEMM_EAGER_FUSED=1.
            self._log_once(
                "eager-native",
                "a2a_gemm %s: non-captured tail stays native "
                "(ATOM_A2A_GEMM_EAGER_FUSED=1 opts in)",
                self.name,
            )
            return self._native_tail(x, padded_tokens)
        out = self._serve(x=x, padded_tokens=padded_tokens, cell=None)
        if out is None:
            return self._native_tail(x, padded_tokens)
        return (out, None)

    def _serve(self, *, x, padded_tokens, cell):
        """One fused call; the cell's stable buffers when capturing/prebinding.

        Returns the caller-owned BF16 output, or None when the frozen weight or
        send-buffer contract does not hold (the caller falls back to native
        before the routed op started).
        """
        m = int(padded_tokens) // self._world()
        # Frozen weight contract, checked from metadata only: the preshuffled
        # fp8 [6144,8192] weight and the contiguous fp32 [6144] scale. The
        # per-token o_proj stores the scale as (6144,1); normalize to the
        # contract's (6144,) (zero-copy for the contiguous production layout)
        # before anything is quantized or called.
        w = self.linear.weight
        if (
            w is None
            or getattr(w, "dtype", None) != _FP8
            or w.dim() != 2
            or tuple(w.shape) != (_N_OUT, _K_IN)
            or not w.is_contiguous()
        ):
            self._log_once(
                "bad-weight-" + str(m),
                "a2a_gemm %s: o_proj weight is not the frozen [6144,8192] FP8 "
                "layout; native",
                self.name,
            )
            return None
        w_scale = _contract_weight_scale(self.linear.weight_scale)
        if w_scale is None:
            self._log_once(
                "bad-weight-scale-" + str(m),
                "a2a_gemm %s: o_proj weight_scale cannot be normalized to the "
                "contract's contiguous [6144]; native",
                self.name,
            )
            return None
        if cell is not None:
            q = cell.q
            scale = cell.scale
            out = cell.out
            registered = cell.registered
        else:
            # Non-capture path: the same address-stable scratch rule as a
            # capture cell. A fresh send/scale tensor per call would change the
            # provider's binding key and re-run its collective SP4 bind on every
            # layer/prefill chunk; cached buffers are overwritten in place.
            view = self._producer_view(m) if registered_band(m) else None
            if registered_band(m) and (view is None or not owns_producer_pointer(view)):
                return None
            q = (
                view
                if view is not None
                else _eager_send_buffer(x.device, padded_tokens)
            )
            scale = _eager_scale_buffer(x.device, m)
            out = _output_buffer(x.device, m)
            registered = view is not None
        transport = _transport()
        q, scale = transport.gather_amax_and_quantize(
            x, int(padded_tokens), out=q, scale=scale
        )
        # Frozen pre-exchange semantics: x is [4*M, 2048]; total_tokens == its
        # row count; x_scale is [M]. Anything else keeps ATOM's composition.
        if (
            q is None
            or q.dtype != _FP8
            or q.dim() != 2
            or tuple(q.shape) != (int(padded_tokens), _SHARD)
            or not q.is_contiguous()
            or scale is None
            or tuple(scale.shape) != (m, 1)
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            self._log_once(
                "bad-send-buffer-" + str(m),
                "a2a_gemm %s: quantizer did not produce the frozen [4M,2048] "
                "send buffer + [M,1] scales; native",
                self.name,
            )
            return None
        self._log_once(
            "first-served",
            "a2a_gemm %s: routed attention tail first served (M=%d registered=%s)",
            self.name,
            m,
            registered,
        )
        aiter = _aiter()
        # One call. No try/except: a routed run error must propagate (aiter
        # does not retry native), and there is no Python fallback after this.
        aiter.a2a_gemm_fused(
            q,
            scale.reshape(-1).contiguous(),
            w,
            w_scale,
            out,
            total_tokens=int(q.shape[0]),
            bucket_m=m,
            registered=registered,
            group=self._group_value(),
            stream=None,
        )
        return out

    def _producer_view(self, m: int):
        cached = self._producer_views.get(m)
        if cached is not None:
            return cached
        provider = self.provider
        get_view = getattr(provider, "registered_input", None)
        if not callable(get_view):
            return None
        try:
            view = get_view(m)
        except Exception:  # noqa: BLE001 - the producer view is an optional extra
            logger.debug("a2a_gemm %s: registered_input(%d) failed", self.name, m)
            return None
        if (
            view is None
            or getattr(view, "dtype", None) != _FP8
            or tuple(getattr(view, "shape", ())) != (_ROWS_PER_M * m, _SHARD)
            or not view.is_contiguous()
        ):
            return None
        _mark_producer(view)
        self._producer_views[m] = view
        return view

    def _group_value(self):
        if self._group is None:
            self._group = _group_or_none() or False
        return None if self._group is False else self._group


# ---------------------------------------------------------------------------
# the model-side hand-off
# ---------------------------------------------------------------------------
def attends_via_routed_op(attn, o_proj, q) -> bool:
    """Static eligibility of the routed tail (safe under Dynamo and capture).

    Everything here is metadata/constant: the installed site, the per-token FP8
    o_proj shape and the pre-exchange attention output width. The dynamic
    fused-vs-native decision (covered M class, registered view, dummy/capture
    phase, provider decline) lives inside the opaque custom op, so tracing is
    stable and the compiled graph serves the fused path.
    """
    site = _SITES.get(getattr(attn, "layer_name", None))
    if site is None or not site.ready:
        return False
    if getattr(attn, "_m3_tp_replicated_o_proj", False):
        return False
    if q is None or getattr(q, "dtype", None) != torch.bfloat16:
        return False
    if getattr(q, "dim", None) is None or q.dim() != 2 or int(q.shape[-1]) != _K_IN:
        return False
    return int(getattr(o_proj, "input_size", 0) or 0) == _K_IN


def fuse_attention_o_proj(attn, o_proj, q, k, v, positions, qkv):
    """Eager entry point for one covered layer's attention tail + o_proj.

    Kept for the eager path and tests; the traced/captured model calls the
    opaque custom op (`a2a_gemm_attend_project`) with the same body.
    """
    site = _SITES.get(getattr(attn, "layer_name", None))
    if site is None:
        raise RuntimeError(
            f"a2a_gemm: no installed site for {getattr(attn, 'layer_name', None)!r}"
        )
    return site.fuse(attn, o_proj, q, k, v, positions, qkv)


def _a2a_gemm_attend_project_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((q.shape[0], _N_OUT), dtype=torch.bfloat16, device=q.device)


@mark_spliting_op(
    is_custom=True, gen_fake=_a2a_gemm_attend_project_fake, mutates_args=[]
)
def a2a_gemm_attend_project(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
) -> torch.Tensor:
    """One layer's attention tail + o_proj, opaque to Dynamo and CUDA graphs."""
    site = _SITES.get(layer_name)
    if site is None:
        raise RuntimeError(f"a2a_gemm: no installed site for {layer_name!r}")
    return site.attend_and_project(q, k, v, positions, qkv)


# ---------------------------------------------------------------------------
# install / process lifecycle
# ---------------------------------------------------------------------------
class A2AGemmSites:
    """The installed sites plus capture-phase and provider lifecycle state."""

    def __init__(self, sites, config, group=None):
        self.sites = sites
        self.config = config
        self.group = group
        self.capture_epoch = 0
        self.capture_bound_classes: set[int] = set()
        # The capture ladder maps many batch sizes onto the same local M
        # (sp_local_len), so each class is pre-bound exactly once per epoch.
        self.capture_prepared_classes: set[int] = set()

    def begin_capture(self) -> int:
        """Open a capture phase: new epoch, drop earlier pre-binds/bindings."""
        global _CAPTURE_EPOCH, _REPLAY_LOGGED, _REPLAY_COUNT
        _CAPTURE_EPOCH += 1
        _REPLAY_LOGGED = set()
        _REPLAY_COUNT = 0
        self.capture_epoch = _CAPTURE_EPOCH
        self.capture_bound_classes = set()
        self.capture_prepared_classes = set()
        for site in self.sites:
            site.begin_capture(_CAPTURE_EPOCH)
        logger.info(
            "a2a_gemm: capture epoch=%d begins for %d layers (eager pre-bind follows)",
            _CAPTURE_EPOCH,
            len(self.sites),
        )
        return _CAPTURE_EPOCH

    def prepare_capture(self, local_m: int) -> int:
        """Eagerly pre-bind one captured M class, rank-uniformly, before capture.

        The rank vote runs first: a layer serves the class only if *every* rank
        can pre-bind it, so the captured graph is uniform (an un-pre-bound class
        is native on all ranks). Pre-bind itself runs the routed call eagerly on
        address-stable buffers, which is what lets capture see a bound class.

        Deduped per M class per epoch: the capture ladder's batch sizes collapse
        onto the same local M (``sp_local_len``), so the second request for a
        class is a no-op (0), keeping the vote and the eager routed call once.
        """
        local_m = int(local_m)
        if _is_cuda_capturing():
            logger.warning(
                "a2a_gemm: capture pre-bind for M=%d requested inside graph "
                "capture; leaving it native",
                local_m,
            )
            return 0
        if not self.sites:
            return 0
        if local_m in self.capture_prepared_classes:
            logger.debug(
                "a2a_gemm: capture M=%d already prepared in epoch %d; skipping",
                local_m,
                self.capture_epoch,
            )
            return 0
        self.capture_prepared_classes.add(local_m)
        local = {site.name: bool(site.can_serve(local_m)) for site in self.sites}
        if not any(local.values()):
            for site in self.sites:
                site.mark_capture_native(local_m)
            logger.info(
                "a2a_gemm capture M=%d: native on every layer (uncovered or "
                "no producer view)",
                local_m,
            )
            return 0
        agreed = agree_capture_classes(local, self.group)
        if agreed is None:
            for site in self.sites:
                site.mark_capture_native(local_m)
            logger.warning(
                "a2a_gemm capture M=%d: rank vote failed; native on every rank",
                local_m,
            )
            return 0
        bound = 0
        for site in self.sites:
            if agreed.get(site.name) and site.capture_prebind(local_m):
                bound += 1
            else:
                site.mark_capture_native(local_m)
                if not agreed.get(site.name):
                    logger.info(
                        "a2a_gemm capture prebind layer=%s M=%d: native "
                        "(rank-uniform decline)",
                        site.name,
                        local_m,
                    )
        if bound:
            self.capture_bound_classes.add(local_m)
            logger.info(
                "a2a_gemm capture prebind M=%d: %d/%d layers fused (rank-uniform)",
                local_m,
                bound,
                len(self.sites),
            )
        else:
            logger.warning(
                "a2a_gemm capture prebind M=%d: no layer could pre-bind; native",
                local_m,
            )
        return bound

    def note_graph_replay(self, bs: int) -> None:
        """Replay-side observability: the graph replayed, counters advance."""
        global _REPLAY_COUNT
        _REPLAY_COUNT += 1
        bs = int(bs)
        if bs not in _REPLAY_LOGGED:
            _REPLAY_LOGGED.add(bs)
            logger.info(
                "a2a_gemm graph replay bs=%d (replay#%d, pre-bound M classes=%s)",
                bs,
                _REPLAY_COUNT,
                sorted(self.capture_bound_classes),
            )

    def close(self):
        """Best-effort provider teardown (the aiter interface has no hook)."""
        from atom.aiter_provider import shutdown_aiter_providers

        shutdown_aiter_providers(self.config, group=_group_or_none())


def _ensure_enabled(config) -> bool:
    """Make this process's aiter plugin configuration match the deployment.

    ``ModelRunner`` configures before the model is built, but installation can
    run in a worker that never saw that call (a separately constructed model).
    The interface has one process-local config, so re-assert it once.
    """
    global _ENSURE_DONE
    from atom.aiter_provider import atom_version, normalize_provider_names

    try:
        enable = normalize_provider_names(
            getattr(config, "aiter_provider_enable", ()) or ()
        )
    except (TypeError, ValueError):
        return False
    if not enable:
        return False
    plugin = _aiter_plugin()
    if plugin is None:
        return False
    if _ENSURE_DONE:
        return True
    _ENSURE_DONE = True
    try:
        status = plugin.status() or {}
        if not tuple(status.get("enabled", ()) or ()):
            plugin.configure(source=f"atom {atom_version()}", enable=enable)
            status = plugin.status() or {}
        if _OP_NAME not in (status.get("routed_ops") or {}):
            logger.warning(
                "a2a_gemm: this aiter does not route %r; native path", _OP_NAME
            )
            return False
    except Exception:
        logger.warning(
            "a2a_gemm: aiter provider status/configure failed", exc_info=True
        )
        return False
    return True


def _minimax_m3_attention_modules(model):
    from atom.models.minimax_m3.model import (
        MiniMaxM3Attention,
        MiniMaxM3SparseAttention,
    )

    for module in model.modules():
        if isinstance(module, (MiniMaxM3Attention, MiniMaxM3SparseAttention)):
            yield module


def initialize_a2a_gemm(model, config, *, group=None, device=None):
    """Install the routed attention-tail sites; call after weight load.

    Returns the installed site set, or None when the switch is off / the
    deployment is not the frozen one / the provider cannot serve. Every
    failure means "native path" with one warning, never an ATOM failure.
    """
    if not getattr(config, "enable_a2a_gemm", False):
        return None
    if not tuple(getattr(config, "aiter_provider_enable", ()) or ()):
        logger.warning(
            "a2a_gemm: --enable-a2a-gemm needs --aiter-provider-enable with a "
            "provider that serves the op; native path"
        )
        return None
    reason = deployment_reason(config)
    if reason is not None:
        logger.info("a2a_gemm: native path (%s)", reason)
        return None
    if not _ensure_enabled(config):
        return None

    group = group if group is not None else _group_or_none()
    device = device if device is not None else _device_index()

    from atom.aiter_provider import (
        load_aiter_provider,
        warmup_aiter_providers,
    )

    # Framework-owned warm-up, after weight load and before warm-up/capture.
    # The provider opens its SP4 session and does its own one rank-uniform
    # vote; False means "declines every call", i.e. native forever.
    warmed = warmup_aiter_providers(config, None, group=group, device=device)
    provider = load_aiter_provider(config, op=_OP_NAME) if warmed else None
    if provider is None:
        logger.warning("a2a_gemm: aiter provider did not warm up; native path")
        return None
    coverage = _normalize_coverage(provider)
    covers = _provider_covers(provider)
    if coverage is None and covers is None:
        logger.warning(
            "a2a_gemm: provider exposes neither coverage() nor covers(); native path"
        )
        return None
    if covers is None and not coverage[0]:
        logger.warning("a2a_gemm: provider coverage table is empty; native path")
        return None

    # Static per-layer gate first: only identical deployments report the same
    # site names, so the rank-uniform vote below also proves every rank
    # installs the same layers.
    eligible: list[tuple[str, object, object]] = []
    for module in _minimax_m3_attention_modules(model):
        attn = getattr(module, "attn", None)
        linear = getattr(module, "o_proj", None)
        name = getattr(attn, "layer_name", None)
        if attn is None or linear is None or not name:
            continue
        why = static_weight_reason(linear)
        if why is None and not _consumes_per_token_fp8(linear):
            why = "o_proj-input"
        if why is None and getattr(attn, "_m3_tp_replicated_o_proj", False):
            why = "tp-replicated-o-proj"
        if why is not None:
            logger.info("a2a_gemm %s: native (%s)", name, why)
            continue
        eligible.append((name, attn, linear))
    if not eligible:
        logger.warning("a2a_gemm: no MiniMax-M3 layer can take the routed op")
        return None

    direct, registered = coverage if coverage is not None else ({}, set())
    # The rank-uniform vote agrees the serving *policy*, not necessarily an
    # explicit class list: newer providers serve arbitrary M through the frozen
    # rule (covers(M)) and keep the 82+7 table as the verified subset, while an
    # older provider still votes its table. Both the policy identity and the
    # table (when present) must match on every rank.
    payload = {
        "deployment": True,
        "sites": tuple(name for name, _, _ in eligible),
        "policy": _provider_policy(provider),
        "covers": covers is not None,
        "direct": tuple(sorted(direct)),
        "registered": tuple(sorted(registered)),
    }
    agreed = agree_class_table(payload, group)
    if agreed is None:
        return None
    # The agreed payload is structurally the local one (rank disagreement is a
    # whole-run decline), so keep the provider's real table.
    agreed = {"direct": direct, "registered": registered}

    sites: list[A2AGemmSite] = []
    for name, attn, linear in eligible:
        site = A2AGemmSite(name, attn, linear, config, provider, agreed, covers=covers)
        _SITES[name] = site
        sites.append(site)
    logger.info(
        "a2a_gemm: installed the routed %s attention tail on %d layers "
        "(%d direct classes, %d registered, covers=%s)",
        _OP_NAME,
        len(sites),
        len(agreed["direct"]),
        len(agreed["registered"]),
        covers is not None,
    )
    return A2AGemmSites(sites, config, group=group)
