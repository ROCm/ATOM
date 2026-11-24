# SPDX-License-Identifier: MIT
"""`model_options.qpath_fuse`: opt-in launch fusion of V4.1's decode query path.

Between the projections and the attention a decode layer launches
`wq_a+wkv GEMM -> q/kv norm -> wq_b GEMM -> RoPE / window`. The last one,
`rope_quant_window`, does two unrelated things: it rotates and MXFP8-quantizes
the query `wq_b` just produced, and it rotates, quantizes and stores the KV
row the norm produced. This switch moves each half onto the launch that
produced its input; with both on the RoPE / window launch is gone. A list of
item names; empty (the default) leaves every path as it was and loads nothing.

  kv  the q/kv norm launch also runs each normed KV row through the KV half
      of `rope_quant_window` (`dual_rmsnorm_mxfp8_kv_window`).
  q   `wq_b` runs the tuned GEMM kernel of this row count with an epilogue
      that also writes the rotated MXFP8 query attention reads: the
      `v41.attn.query_rope_quant` seam, served by a planned plugin owner or
      by AITER's `gemm_a8w8_mxscale_bpreshuffle_rope_quant` where the
      AITER in use has it.

Both items serve a decode step of the MXFP8 cache only (FP8 KV and window,
not packed), on gfx950, at V4.1-Flash widths, where the q/kv norm is the
native MXFP8 one and no fused attention (`model_options.fused_attn`) serves
the step; `q` further needs a fused epilogue for the tuned kernel at this
row count (the seam's owner, or AITER's). Anything else -- prefill, bounded replay, other shapes or
arches, an item that is off -- keeps today's launches for that half.

What the two launches give is fixed up to one thing, the rotation's rounding.
`rope_window` writes `even * cos - odd * sin` and `even * sin + odd * cos` and
lets the compiler fuse one product of each sum; which one differs between
compilations of that kernel -- by the tables' dtype (this model's are BF16)
and by its program layout (how many heads a query program takes; the KV row
sits beside either). The two outcomes are

  P  even' = fma(cos, even, -(sin * odd))   odd' = fma(sin, even, cos * odd)
  G  even' = fma(cos, even, -(sin * odd))   odd' = fma(cos, odd, sin * even)

and differ in the last bit of a rotated odd value about once in a million,
enough to move an e4m3 code. The fused launches therefore take the form as an
argument and compute it with explicit fma. `REFERENCE_FORMS` records what each
compilation of one reference kernel does, read off its compiled code;
`reference_program_heads` is that reference's layout rule, and
`PROVEN_FINGERPRINTS` the kernel source and Triton build the record was taken
from. `plan_forms` opens only for a table dtype and program in the record and
names the recorded form.

On the recorded build the fused output is bit for bit what the two launches
give. The recorded kernel is `rope_window_reference`, which a recipe selects
with `model_options.rope_window_reference`; `rope_window`'s own kernel is
another source (`reference_fingerprint`). Without that option the fused
launches still compute the recorded forms, which are fixed by the record and
not by the compiler, so against `rope_window`'s own launch a rotated value
may differ in its last bit, at the rate above. One line in the log says which
case a process is in. Recording another kernel takes its compiled gfx950 code.
"""

import functools
import hashlib
import logging

import torch

logger = logging.getLogger("atom")

KNOWN_ITEMS = frozenset({"kv", "q"})

# The decode query path this switch is validated for: heads x head_dim, the
# rotated tail, the q latent's width.
HEADS, HEAD_DIM, ROPE_DIM, Q_RANK = 64, 512, 64, 1280
# The fewest rows `q` fuses at: the fused epilogue was compared with the two
# launches, byte for byte on a device, from 6 rows up (the tuned kernel of 5..8
# rows); the two kernels that serve 1..4 rows have only been checked statically.
Q_MIN_ROWS = 5

# The rotation form (module docstring) of every compilation of the reference
# `_rope_window_kernel` a decode step can launch, by `(tables, program)`:
# tables "bf16" | "f32"; program `full_fh<heads a query program>` (query and
# KV row, the decode launch), `query_fh<..>` (the query alone) and `kv_only`
# (the KV row alone) -- the last two are what is left to launch when one item
# fuses and the other does not. Read off the reference's compiled gfx950
# code; it is a property of that build, hence `PROVEN_FINGERPRINTS`.
REFERENCE_FORMS = {
    ("bf16", "full_fh1"): {"q": "G", "kv": "G"},
    ("bf16", "full_fh8"): {"q": "G", "kv": "P"},
    ("bf16", "query_fh1"): {"q": "G"},
    ("bf16", "query_fh8"): {"q": "G"},
    ("bf16", "kv_only"): {"kv": "G"},
    ("f32", "full_fh1"): {"q": "P", "kv": "P"},
    ("f32", "full_fh8"): {"q": "G", "kv": "P"},
    ("f32", "query_fh1"): {"q": "P"},
    ("f32", "query_fh8"): {"q": "G"},
    ("f32", "kv_only"): {"kv": "P"},
}
# `reference_fingerprint()` of the builds `REFERENCE_FORMS` was read from.
PROVEN_FINGERPRINTS = frozenset({"10aa874fab0393fd"})

# Seams whose native body the fused path stands in for. An owner of any of
# them keeps the calls it would have got.
_SEAMS = ("v41.attn.qk_norm",)
# Item q's launch: wq_b's GEMM that also writes the rotated MXFP8 query.
QUERY_SEAM = "v41.attn.query_rope_quant"


def items() -> tuple[str, ...]:
    """The recipe's `model_options.qpath_fuse`."""
    from atom import plugin_host

    return plugin_host.model_options().qpath_fuse


# Decisions made (calls, not graph replays): what a boot check reads.
COUNTS = {"kv": 0, "q": 0, "rope_window_launches": 0, "unfused": 0}
_logged: set[str] = set()


def _log_once(key: str, message: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(message, *args)


def enabled(item: str) -> bool:
    return item in items()


def _gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx950"
    except Exception:  # noqa: BLE001 - no device to ask
        return False


@functools.lru_cache(maxsize=1)
def _aiter_rope_quant():
    """AITER's `(op, fuses)` pair, or `(None, None)` on a build without it or
    with one that cannot be told the rotation form (`fuse_cos`) or read BF16
    tables. Asked once a process."""
    try:
        import inspect

        from aiter.ops.gemm_rope_quant import (
            gemm_a8w8_mxscale_bpreshuffle_rope_quant,
            gemm_a8w8_mxscale_bpreshuffle_rope_quant_fuses,
        )

        op = gemm_a8w8_mxscale_bpreshuffle_rope_quant
        if "fuse_cos" not in inspect.signature(op).parameters:
            return None, None
    except (ImportError, TypeError, ValueError):
        return None, None
    return op, gemm_a8w8_mxscale_bpreshuffle_rope_quant_fuses


def query_fuses(wq_b, rows, positions, cos, sin, fuse_cos) -> bool | None:
    """Whether item q's launch serves this step: a planned owner of
    `QUERY_SEAM` answers first (its select, asked through `explain` with what
    exists before the norm ran -- the weight, its scale, the positions and
    the tables), then AITER's own predicate. None: neither exists."""
    from atom import plugin_host

    weight, scale = wq_b.weight, wq_b.weight_scale
    if plugin_host.owns(QUERY_SEAM):
        captured, bucket = plugin_host._graph_state()
        decision = plugin_host.explain(
            QUERY_SEAM,
            None,
            None,
            weight,
            scale,
            positions,
            cos,
            sin,
            None,
            None,
            captured=captured,
            bucket=bucket,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
            fuse_cos=fuse_cos,
        )
        if decision.route == "plugin":
            return True
    op, fuses = _aiter_rope_quant()
    if op is None:
        return None
    return bool(
        fuses(
            rows,
            weight.shape[0],
            weight.shape[1],
            tuple(scale.shape),
            HEAD_DIM,
            ROPE_DIM,
        )
    )


def rope_quant_gemm(
    x_fp8,
    x_scale,
    w,
    w_scale,
    positions,
    cos,
    sin,
    q_fp8,
    q_scale,
    *,
    unfused,
    **form,
):
    """Item q's launch through `QUERY_SEAM`: bf16 `[M, N]`, the rotated MXFP8
    query written into `q_fp8` / `q_scale`. ``form``: `head_dim`, `rope_dim`
    and, for form G, `fuse_cos=True` (the default form is left unnamed). An
    owner that declines a call it accepted at plan time leaves it to AITER's
    op, or, on an AITER without one, to ``unfused`` (the GEMM, then the
    query's own rotate-and-quantize launch)."""
    from atom import plugin_host

    op, _ = _aiter_rope_quant()

    def native(x_fp8, x_scale, w, w_scale, positions, cos, sin, q_fp8, q_scale, **form):
        if op is None:
            return unfused(x_fp8, x_scale, positions, q_fp8, q_scale)
        return op(
            x_fp8, w, x_scale, w_scale, positions, cos, sin, q_fp8, q_scale, **form
        )

    return plugin_host.call(
        "v41.attn.query_rope_quant",
        native,
        x_fp8,
        x_scale,
        w,
        w_scale,
        positions,
        cos,
        sin,
        q_fp8,
        q_scale,
        **form,
    )


@functools.lru_cache(maxsize=1)
def reference_fingerprint() -> str:
    """What `REFERENCE_FORMS` is a property of: the source of the
    `_rope_window_kernel` this process launches (`rope_window_reference`'s
    under `model_options.rope_window_reference`, else `rope_window`'s) and
    of everything it calls (Triton's own hash of it), and the Triton build
    that compiles it. "" when it cannot be taken."""
    try:
        from triton.runtime.cache import triton_key

        from atom import plugin_host
        from atom.model_ops.deepseek_v41 import rope_window, rope_window_reference

        if plugin_host.model_options().rope_window_reference:
            rope_window = rope_window_reference
        text = f"{rope_window._rope_window_kernel.cache_key}-{triton_key()}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


def _log_reference() -> None:
    """Once a process: whether the RoPE / window kernel of this tree is the
    one the rotation forms were read from (module docstring)."""
    fingerprint = reference_fingerprint()
    if fingerprint in PROVEN_FINGERPRINTS:
        _log_once(
            "reference:proven",
            "[QPATH-FUSE] rotation forms: the recorded reference kernel "
            "(fingerprint %s); the fused launches give its bytes",
            fingerprint,
        )
    else:
        _log_once(
            "reference:other",
            "[QPATH-FUSE] rotation forms: the RoPE / window kernel or the Triton "
            "build is not the recorded reference (fingerprint %s; "
            "model_options.rope_window_reference selects the recorded kernel); "
            "the fused launches compute the recorded forms, and a rotated value "
            "may differ in its last bit from this process's RoPE / window launch",
            fingerprint or "unavailable",
        )


def rope_tables(rope):
    """`(kind, cos, sin)` of the tables the RoPE / window launch reads off
    ``rope``, as the fused launches take them, or None: ``kind`` "bf16" or
    "f32", both tables contiguous ``[positions, pairs]`` of that dtype (2-D
    views of the same memory when singleton dims sit in between). The fused
    launches read them in place and widen a BF16 entry as the reference
    does; no fp32 copy is involved."""
    cos, sin = rope.cos_cache, rope.sin_cache
    kind = {torch.bfloat16: "bf16", torch.float32: "f32"}.get(cos.dtype)
    if (
        kind is None
        or cos.dim() < 2
        or cos.shape != sin.shape
        or cos.dtype != sin.dtype
        or any(n != 1 for n in cos.shape[1:-1])
        or not (cos.is_contiguous() and sin.is_contiguous())
    ):
        return None
    pairs = cos.shape[-1]
    return kind, cos.view(-1, pairs), sin.view(-1, pairs)


def reference_program_heads(rows: int, heads: int, device):
    """How many heads one query program of the reference launch takes at
    ``rows`` (`rope_window_reference.query_fp8_heads`: 1 or 8), or None when
    it cannot be told: which record of `REFERENCE_FORMS` a step's fused
    launches compute. Asked with that launch's own inputs, the row count and
    the device's CU count, so every rank of a group answers alike for the
    same rows."""
    from atom.model_ops.deepseek_v41 import rope_window_reference

    try:
        cu = None
        if device.type != "cuda":
            # Nothing launches without a device (a wiring check): AITER's count.
            from aiter.jit.utils.chip_info import get_cu_num

            cu = int(get_cu_num())
            if cu <= 0:
                return None
        return int(rope_window_reference.query_fp8_heads(rows, heads, device, cu))
    except Exception:  # noqa: BLE001 - no CU count to size the programs by
        return None


def _aiter_norm_may_be_served() -> bool:
    """Whether an AITER plugin provider could take the q/kv norm call the `kv`
    item stands in for. Unknown counts as yes."""
    try:
        from aiter import plugin as aiter_plugin
        from aiter.ops.triton.quant import fused_mxfp8_quant

        config = aiter_plugin._config()
        if config.kill_switch or not config.enabled:
            return False
        routed = fused_mxfp8_quant._fused_dual_rmsnorm_mxfp8_quant_routed
        operation = getattr(routed, "__aiter_plugin_op__", None)
        if operation is None:
            return False
        return bool(aiter_plugin._eligible(operation, config).providers)
    except Exception:  # noqa: BLE001
        return True


def layer_supported(attn) -> tuple[bool, str]:
    """Whether an attention layer has the shape and the weights the fused
    path is written for (checked once, after the weights load)."""
    if (attn.heads, attn.head_dim) != (HEADS, HEAD_DIM):
        return False, f"heads x head_dim {attn.heads}x{attn.head_dim}"
    if type(attn.qk_norm).__name__ != "DualRMSNormMXFP8":
        return False, "the q/kv norm is not the MXFP8 one"
    if not _gfx950():
        return False, "not gfx950"
    return True, "ok"


def wq_b_supported(wq_b) -> tuple[bool, str]:
    """Whether `wq_b.forward` is exactly the native preshuffled group-32
    MXFP8 GEMM, with nothing after it: the call the `q` item replaces."""
    weight = getattr(wq_b, "weight", None)
    scale = getattr(wq_b, "weight_scale", None)
    if weight is None or scale is None or weight.numel() == 0 or scale.numel() == 0:
        return False, "weight or scale empty"
    if getattr(wq_b, "native_a8_group_rows", None) != 32:
        return False, "not a native 32x32 MXFP8 weight"
    if not getattr(weight, "is_shuffled", False):
        return False, "weight not preshuffled"
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return False, f"weight {weight.dtype} {tuple(weight.shape)}"
    if tuple(weight.shape) != (HEADS * HEAD_DIM, Q_RANK):
        return False, f"weight shape {tuple(weight.shape)}"
    if getattr(wq_b, "bias", None) is not None:
        return False, "has a bias"
    if getattr(wq_b, "is_output_padded", False):
        return False, "padded output"
    if getattr(wq_b, "gemm_seam", None) is not None:
        return False, "GEMM behind a plugin seam"
    if (
        getattr(wq_b, "tp_dim", None) == 1
        and getattr(wq_b, "tp_size", 1) > 1
        and getattr(wq_b, "reduce_results", False)
    ):
        return False, "all-reduced output"
    return True, "ok"


def plan(attn, hidden, cache, step, rope, queries) -> tuple[bool, bool]:
    """`(kv, q)`: which halves of `rope_quant_window` this call moves onto the
    launches ahead of it (`plan_forms` without the forms). `(False, False)`
    is today's path, untouched."""
    kv, q = plan_forms(attn, hidden, cache, step, rope, queries)
    return kv is not None, q is not None


def plan_forms(attn, hidden, cache, step, rope, queries):
    """`(kv, q)`: for each half of `rope_quant_window` this call moves onto the
    launch ahead of it, the rotation form ("P" or "G", module docstring) that
    launch has to compute -- the form of the reference compilation this step
    would have launched -- else None. `(None, None)` is today's path."""
    wanted = items()
    if not wanted:
        return None, None
    if torch.compiler.is_compiling():
        # Traced Python is today's path, as a plugin seam's is: the fused norm
        # launches its kernel from Python, which a tracer cannot keep.
        return None, None
    state = attn._qpath_fuse_state
    if state is None:
        layer_ok, layer_why = layer_supported(attn)
        wq_b_ok, wq_b_why = wq_b_supported(attn.wq_b)
        state = attn._qpath_fuse_state = (layer_ok, layer_ok and wq_b_ok)
        _log_once(
            f"layer:{layer_ok}:{layer_why}:{wq_b_ok}:{wq_b_why}",
            "[QPATH-FUSE] items %s: layer %s (%s), wq_b %s (%s)",
            sorted(wanted),
            "supported" if layer_ok else "unsupported",
            layer_why,
            "supported" if wq_b_ok else "unsupported",
            wq_b_why,
        )
    if not state[0]:
        return None, None
    from atom import plugin_host

    why = tables = full = heads = None
    rows = step.running_tokens
    if not step.decode or queries is not None:
        why = "not a decode step over all rows"
    elif not (cache.kv_fp8 and cache.window_fp8) or cache.packed:
        why = "not the MXFP8 KV + window cache"
    elif not rows:
        why = "no rows"
    elif any(plugin_host.owns(seam) for seam in _SEAMS):
        why = "a query-path seam has a plugin owner"
    else:
        positions = step.positions
        tables = rope_tables(rope)
        if (
            rope.rope_dim != ROPE_DIM
            or tables is None
            or tables[1].shape[1] != ROPE_DIM // 2
        ):
            why = "rope tables are not bf16 or fp32 [positions, 32]"
        elif (
            positions.dtype != torch.int64
            or positions.numel() != rows
            or not positions.is_contiguous()
        ):
            why = "positions are not one int64 per row"
        elif hidden.shape[:-1].numel() != rows:
            why = "rows disagree with the step"
        else:
            heads = reference_program_heads(rows, attn.heads, hidden.device)
            full = REFERENCE_FORMS.get((tables[0], f"full_fh{heads}"))
            if full is None:
                why = "no recorded rotation form for this step's launch"
    if why is not None:
        COUNTS["unfused"] += 1
        _log_once(f"plan:{why}", "[QPATH-FUSE] not fused: %s", why)
        return None, None
    _log_reference()

    kind = tables[0]
    kv = enabled("kv") and not _aiter_norm_may_be_served()
    if enabled("kv") and not kv:
        _log_once(
            "kv:provider", "[QPATH-FUSE] kv: an AITER provider may serve the norm"
        )
    q, q_why = False, "not fused"
    if enabled("q") and state[1]:
        if rows < Q_MIN_ROWS:
            q_why = f"not fused (fewer than {Q_MIN_ROWS} rows)"
        else:
            fuses = query_fuses(
                attn.wq_b, rows, positions, tables[1], tables[2], full["q"] == "G"
            )
            if fuses is None:
                q_why = (
                    "not fused (no plugin owner of the query seam, and AITER "
                    "has no fused epilogue with a rotation form)"
                )
            q = bool(fuses)
    # A half that stays behind is launched by the program that runs it alone.
    # That program must round as the full launch does, or the pair is not
    # what the two launches give.
    if q and not kv:
        alone = REFERENCE_FORMS.get((kind, "kv_only"), {}).get("kv")
        if alone != full["kv"]:
            q, q_why = False, (
                "not fused (without item kv the KV row would be launched alone, "
                "and that program rounds its rotation differently here)"
            )
    if kv and not q:
        alone = REFERENCE_FORMS.get((kind, f"query_fh{heads}"), {}).get("q")
        if alone != full["q"]:
            kv = False
            _log_once(
                f"kv:alone:{kind}:{heads}",
                "[QPATH-FUSE] kv: not fused at %d rows (the query would be "
                "launched alone, and that program rounds its rotation "
                "differently here)",
                rows,
            )
    if enabled("q") and state[1]:
        _log_once(
            f"q:{rows}:{q}:{q_why}",
            "[QPATH-FUSE] q: %s at %d rows",
            (
                f"wq_b writes the rotated MXFP8 query (form {full['q']}, {kind} tables)"
                if q
                else q_why
            ),
            rows,
        )
    if kv:
        _log_once(
            f"kv:on:{kind}:{full['kv']}",
            "[QPATH-FUSE] kv: the q/kv norm launch writes the KV window "
            "(form %s, %s tables, first at %d rows)",
            full["kv"],
            kind,
            rows,
        )
    COUNTS["kv"] += kv
    COUNTS["q"] += q
    COUNTS["rope_window_launches"] += not (kv and q)
    return (full["kv"] if kv else None), (full["q"] if q else None)
