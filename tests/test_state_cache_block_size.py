# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`state_cache_block_size` (M): checkpoint granularity for recurrent state.

Linear-attention hybrids (GDN / KDA — Qwen3-Next, Qwen3.5, Kimi-K3) carry a
conv_state + ssm_state per request. Unlike paged KV, that state is a whole
layer snapshot rather than a per-token row, so checkpointing it at every
`kv_cache_block_size` boundary would cost orders of magnitude more HBM than
the paged KV it parallels. M is the coarser granularity at which state IS
checkpointed; prefix caching for such models then works at
max(M, kv_cache_block_size).

Two things have to hold for M to be usable:

  * **Alignment.** A checkpoint boundary must be simultaneously a paged-KV
    block boundary (so one prefix-cache hit covers both pools at the same
    token count) and a conv/scan chunk boundary (so the kernels can emit a
    state there). That is lcm(64, kv_cache_block_size).
  * **Pool ratio.** The checkpoint pool and the paged pool must cover the
    same token span, i.e. split the budget as
    `checkpoint_bytes : (M / block_size) * block_bytes`. Otherwise one pool
    runs dry while the other still has capacity, which is exactly the waste
    M exists to avoid.

These tests pin the derivation math and the source-level contract (no GPU
required); the byte-cost hooks themselves are exercised at the builder level.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "atom" / "config.py"
MODEL_RUNNER = REPO / "atom" / "model_engine" / "model_runner.py"
ENGINE_CORE = REPO / "atom" / "model_engine" / "engine_core.py"
ARG_UTILS = REPO / "atom" / "model_engine" / "arg_utils.py"
BACKENDS = REPO / "atom" / "model_ops" / "attentions" / "backends.py"
GDN_ATTN = REPO / "atom" / "model_ops" / "attentions" / "gdn_attn.py"


def _func(tree: ast.Module, fn_name: str, cls_name: str | None = None):
    scope: ast.AST = tree
    if cls_name is not None:
        scope = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == cls_name
        )
    return next(
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )


def _body_src(fn: ast.FunctionDef) -> str:
    """`fn`'s source with its docstring removed.

    Matching on `ast.unparse(fn)` wholesale is a trap: a docstring that
    *describes* the contract ("must pass num_spec=0") satisfies a substring
    assertion even after the code stops doing it, so the test silently stops
    testing. Strip the docstring so only real code is matched.
    """
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return "\n".join(ast.unparse(n) for n in body)


def _load_state_cache_helpers():
    """The state-cache helpers, exec'd straight out of atom/config.py.

    conftest stubs `atom.config`, and importing the real one drags in
    HuggingFace + torch. All three helpers are self-contained (stdlib `math`
    only), so exec'ing their source keeps this test dependency-free while
    still exercising the SHIPPED code rather than a reimplementation — a
    reimplementation would just re-derive the bug.
    """
    tree = ast.parse(CONFIG.read_text())
    align = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "STATE_CACHE_BLOCK_SIZE_ALIGN"
    )
    wanted = (
        "state_cache_align",
        "validate_state_cache_config",
        "derive_state_cache_block_size",
    )
    body = [_func(tree, name) for name in wanted]
    ns: dict = {"STATE_CACHE_BLOCK_SIZE_ALIGN": align, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<config>", "exec"), ns)
    return ns, align


_HELPERS, ALIGN = _load_state_cache_helpers()
derive = _HELPERS["derive_state_cache_block_size"]
align_for = _HELPERS["state_cache_align"]
validate = _HELPERS["validate_state_cache_config"]


# ── derivation math ────────────────────────────────────────────────────────


def test_align_is_64():
    """The conv kernel tiles at BLOCK_M=8 and the GDN scan at BT=64; 64 is the
    smallest value satisfying both, and it is what the CLI help advertises."""
    assert ALIGN == 64


@pytest.mark.parametrize("fraction", [0.05, 0.15, 0.3, 0.5])
def test_derived_m_lands_near_target_fraction(fraction):
    """M is chosen so the state pool is ~`fraction` of the combined budget.

    Checked against the definition rather than the closed form: at the derived
    M, amortized state bytes / (state + paged) bytes per token should sit at
    the target. Alignment rounds M UP, which biases the realized fraction
    DOWN (a coarser M means fewer checkpoints), so the realized value must not
    exceed the target.
    """
    state_bytes = 96 * (1 << 20)  # ~96MB of recurrent state per checkpoint
    paged_per_token = 8192  # bytes of full-attn KV per token
    m = derive(state_bytes, paged_per_token, fraction)

    assert m % ALIGN == 0 and m > 0
    realized = (state_bytes / m) / ((state_bytes / m) + paged_per_token)
    assert realized <= fraction + 1e-9, (
        f"align rounds M up, so the realized fraction must not exceed the "
        f"target: got {realized:.4f} > {fraction}"
    )
    # And the next-coarser M must be the one that overshoots, i.e. M is the
    # tightest aligned value at or under the target — not an arbitrary
    # over-round that starves the state pool.
    if m > ALIGN:
        finer = m - ALIGN
        realized_finer = (state_bytes / finer) / (
            (state_bytes / finer) + paged_per_token
        )
        assert realized_finer > fraction


def test_larger_state_needs_coarser_m():
    """More expensive state => checkpoint less often, to hold the same share."""
    paged, frac = 8192, 0.15
    small = derive(16 * (1 << 20), paged, frac)
    large = derive(128 * (1 << 20), paged, frac)
    assert large > small


def test_cheaper_paged_kv_needs_coarser_m():
    """When paged KV per token is cheap, the same state share needs a wider
    span to amortize against."""
    state, frac = 96 * (1 << 20), 0.15
    cheap = derive(state, 1024, frac)
    pricey = derive(state, 16384, frac)
    assert cheap > pricey


def test_derived_m_is_never_below_align():
    """Even a tiny state must round up to one aligned block — a sub-align M
    is unusable by the conv/scan kernels."""
    assert derive(1, 1 << 30, 0.15) == ALIGN


@pytest.mark.parametrize("state,paged", [(0, 8192), (1 << 20, 0), (-1, 8192)])
def test_no_state_or_no_paged_kv_disables_checkpointing(state, paged):
    """0 means "no checkpoint pool"; the callers gate on `> 0`."""
    assert derive(state, paged, 0.15) == 0


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_out_of_range_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="state_cache_target_fraction"):
        derive(1 << 20, 8192, fraction)


# ── alignment contract ─────────────────────────────────────────────────────


@pytest.mark.parametrize("kv_block", [16, 64, 128, 256, 1024])
def test_align_is_lcm_of_kernel_and_kv_block(kv_block):
    """Alignment must be lcm(ALIGN, kv_cache_block_size).

    A checkpoint that is a multiple of only one of the two is unusable: land
    off a paged-KV boundary and a hit covers different token counts in the two
    pools; land off a conv/scan chunk and no kernel can emit the state.
    Dropping either factor (e.g. returning the kernel alignment alone) breaks
    the 1024-page backends, so both are checked directly.
    """
    got = align_for(kv_block)
    assert got == math.lcm(ALIGN, kv_block)
    assert got % ALIGN == 0, "must land on a conv/scan chunk boundary"
    assert got % kv_block == 0, "must land on a paged-KV block boundary"


@pytest.mark.parametrize(
    "m,kv_block",
    [
        (100, 16),  # not a multiple of the kernel alignment
        (32, 16),  # multiple of kv_block but below/off the kernel alignment
        (64, 128),  # multiple of the kernel alignment but off the KV block
        (192, 128),  # multiple of neither lcm factor jointly
    ],
)
def test_misaligned_m_rejected(m, kv_block):
    with pytest.raises(ValueError, match="state_cache_block_size"):
        validate(m, kv_block, 32768, 0.15)


@pytest.mark.parametrize("kv_block", [16, 128])
def test_aligned_m_accepted(kv_block):
    validate(align_for(kv_block) * 3, kv_block, 32768, 0.15)


def test_m_larger_than_context_rejected():
    """A boundary no sequence can reach is dead memory."""
    with pytest.raises(ValueError, match="max_model_len"):
        validate(8192, 16, 4096, 0.15)


@pytest.mark.parametrize("sentinel", [-1, 0])
def test_sentinels_accepted_without_alignment_checks(sentinel):
    """-1 (derive) and 0 (disabled) are not token counts, so alignment does
    not apply to them."""
    validate(sentinel, 16, 4096, 0.15)


@pytest.mark.parametrize("m", [-2, -100])
def test_negative_non_sentinel_rejected(m):
    with pytest.raises(ValueError, match="state_cache_block_size"):
        validate(m, 16, 32768, 0.15)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_validate_rejects_out_of_range_fraction(fraction):
    with pytest.raises(ValueError, match="state_cache_target_fraction"):
        validate(-1, 16, 32768, fraction)


def test_validation_runs_after_the_v4_block_size_override():
    """DeepSeek-V4 rewrites `kv_cache_block_size` to 128 in `__post_init__`.

    The alignment M must satisfy is derived from that value, so validating
    before the override would accept an M that is misaligned by the time the
    pool is built — e.g. M=64 is fine against the default block size 16 but
    illegal once the block size becomes 128.
    """
    validate(64, 16, 32768, 0.15)  # legal pre-override
    with pytest.raises(ValueError):
        validate(64, 128, 32768, 0.15)  # illegal post-override

    tree = ast.parse(CONFIG.read_text())
    fn = _func(tree, "__post_init__", cls_name="Config")
    lines = _body_src(fn).splitlines()
    v4 = next(i for i, ln in enumerate(lines) if "DeepseekV4" in ln)
    call = next(i for i, ln in enumerate(lines) if "validate_state_cache_config" in ln)
    assert call > v4


def test_post_init_actually_calls_the_validator():
    """The AST ordering check above is only meaningful if the call is there."""
    tree = ast.parse(CONFIG.read_text())
    fn = _func(tree, "__post_init__", cls_name="Config")
    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "validate_state_cache_config" in calls


# ── pool sizing contract ───────────────────────────────────────────────────


@pytest.mark.parametrize("m", [64, 512, 4096])
@pytest.mark.parametrize("checkpoint_mb", [4, 96, 512])
def test_pool_split_covers_equal_token_spans(m, checkpoint_mb):
    """Budgeting in whole spans makes both pools cover the same token count.

    One span = M tokens = 1 checkpoint + M/block_size paged blocks. Reserving
    `available // span_bytes` checkpoints therefore pairs every checkpoint
    with the paged blocks it needs, with the remainder flowing to the paged
    pool — so the paged span is never SHORT, only ever slightly ahead.
    """
    budget = 40 * (1 << 30)
    checkpoint_bytes = checkpoint_mb * (1 << 20)
    block_bytes = 128 * (1 << 10)
    block_size = 64

    paged_per_span = (m // block_size) * block_bytes
    span_bytes = checkpoint_bytes + paged_per_span
    n_checkpoints = budget // span_bytes
    n_paged_blocks = (budget - n_checkpoints * checkpoint_bytes) // block_bytes

    tokens_state = n_checkpoints * m
    tokens_paged = n_paged_blocks * block_size
    assert tokens_paged >= tokens_state, (
        f"paged pool must cover at least the state pool's span: "
        f"state={tokens_state} paged={tokens_paged}"
    )
    # The only excess is the sub-span remainder of the floor division, spent
    # on paged blocks. Bound it: strictly less than one span's worth of budget
    # converted to paged tokens. A larger gap would mean the split itself is
    # skewed, not just rounded.
    max_excess = (span_bytes // block_bytes + 1) * block_size
    assert tokens_paged - tokens_state < max_excess


def test_derived_m_makes_the_two_spans_nearly_equal():
    """At a DERIVED M the pools are balanced, so the leftover is negligible.

    The one-span bound above is exact but loose when M is far too small for
    the state cost (one 96MB checkpoint paired with a single 128KB paged
    block — the regime the derivation exists to avoid). Feeding the
    derivation's own M back into the sizing math closes the gap to a couple
    of percent, which is the property that actually matters: no meaningful
    HBM is stranded in either pool.
    """
    budget = 40 * (1 << 30)
    checkpoint_bytes = 96 * (1 << 20)
    block_bytes, block_size = 128 * (1 << 10), 64

    m = derive(checkpoint_bytes, block_bytes / block_size, 0.15)
    paged_per_span = (m // block_size) * block_bytes
    span_bytes = checkpoint_bytes + paged_per_span
    n_checkpoints = budget // span_bytes
    n_paged_blocks = (budget - n_checkpoints * checkpoint_bytes) // block_bytes

    tokens_state = n_checkpoints * m
    tokens_paged = n_paged_blocks * block_size
    assert tokens_state > 0
    assert tokens_paged == pytest.approx(tokens_state, rel=0.05)

    # ...and the realized state share should sit near the requested 0.15.
    realized = (n_checkpoints * checkpoint_bytes) / budget
    assert realized == pytest.approx(0.15, abs=0.02)


def test_runner_budgets_in_whole_spans():
    tree = ast.parse(MODEL_RUNNER.read_text())
    fn = _func(tree, "get_num_blocks", cls_name="ModelRunner")
    src = _body_src(fn)
    assert "state_checkpoint_bytes()" in src
    assert "num_state_cache_blocks" in src
    # The span term: M / block_size blocks of block_bytes each, plus one
    # checkpoint; capacity is a floor division by that, not a float share.
    assert "state_cache_block_size // self.block_size" in src
    assert "span_bytes = checkpoint_bytes + paged_bytes_per_span" in src
    assert "available_for_pool // span_bytes" in src


def test_runner_deducts_the_pool_before_sizing_kv_blocks():
    """`available_for_pool` must shrink BEFORE num_kvcache_blocks is computed,
    or the two pools would each be sized against the full budget and OOM."""
    tree = ast.parse(MODEL_RUNNER.read_text())
    fn = _func(tree, "get_num_blocks", cls_name="ModelRunner")
    lines = _body_src(fn).splitlines()
    deduct = next(
        i for i, ln in enumerate(lines) if "available_for_pool -= state_reserved" in ln
    )
    sized = [i for i, ln in enumerate(lines) if "num_kvcache_blocks =" in ln]
    assert sized, "expected num_kvcache_blocks to be assigned in get_num_blocks"
    assert all(i > deduct for i in sized)


def test_zero_capacity_pool_also_zeros_the_granularity():
    """If the budget holds no checkpoints, `state_cache_block_size` must go to
    0 too — otherwise downstream `> 0` gates would read as "checkpointing on"
    against an empty pool."""
    tree = ast.parse(MODEL_RUNNER.read_text())
    fn = _func(tree, "get_num_blocks", cls_name="ModelRunner")
    src = _body_src(fn)
    assert "num_state_cache_blocks == 0" in src
    assert "config.state_cache_block_size = 0" in src


def test_pool_needs_prefix_caching_and_chunked_prefill():
    """Both of the pool's enabling conditions must be decided in ONE place.

    `StateCachePool.__init__` disables itself without prefix caching (no
    consumer for a checkpoint) or without chunked prefill (its write path works
    by cutting a chunk short at a boundary, which a deployment that turned
    chunking off has opted out of). If the runner sized a pool anyway,
    `EngineCore`'s gate would see `state_cache_block_size > 0` and leave prefix
    caching ON for a recurrent model whose BlockManager pool is inert — hits
    with nothing bounding them, which is the exact corruption the gate exists
    to prevent.
    """
    tree = ast.parse(MODEL_RUNNER.read_text())
    fn = _func(tree, "_resolve_state_cache_block_size", cls_name="ModelRunner")
    src = _body_src(fn)
    assert "config.enable_prefix_caching" in src
    assert "config.enable_chunked_prefill" in src
    assert "config.state_cache_block_size = 0" in src

    pool = ast.parse(
        (REPO / "atom" / "model_engine" / "state_cache_pool.py").read_text()
    )
    init = _func(pool, "__init__", cls_name="StateCachePool")
    enabled = next(
        n
        for n in ast.walk(init)
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and isinstance(getattr(n, "target", None) or n.targets[0], ast.Attribute)
        and (getattr(n, "target", None) or n.targets[0]).attr == "enabled"
    )
    cond = ast.unparse(enabled.value)
    assert "enable_prefix_caching" in cond and "enable_chunked_prefill" in cond


def test_block_manager_forwards_both_flags_to_the_pool():
    """The pool cannot read Config itself, so BlockManager must pass them."""
    bm = ast.parse((REPO / "atom" / "model_engine" / "block_manager.py").read_text())
    fn = _func(bm, "__init__", cls_name="BlockManager")
    call = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "StateCachePool"
    )
    passed = {kw.arg for kw in call.keywords}
    assert {"enable_prefix_caching", "enable_chunked_prefill"} <= passed


def test_stateless_attention_gets_no_pool():
    """`_resolve_state_cache_block_size` must force 0 when the builder has no
    checkpointable state, even if the user passed an explicit M."""
    tree = ast.parse(MODEL_RUNNER.read_text())
    fn = _func(tree, "_resolve_state_cache_block_size", cls_name="ModelRunner")
    src = _body_src(fn)
    assert "checkpoint_bytes <= 0" in src
    assert "config.state_cache_block_size = 0" in src


def test_resolution_happens_in_the_runner_and_is_echoed_to_the_engine():
    """M is resolved in the RUNNER subprocess (it needs per-block byte costs
    only the loaded builder knows), so the engine process must read the
    resolved value back out of block_info — same mechanism as num_swa_blocks.
    Without it the engine would still hold the -1 sentinel.
    """
    runner_src = MODEL_RUNNER.read_text()
    assert '"state_cache_block_size": int(config.state_cache_block_size)' in runner_src

    tree = ast.parse(ENGINE_CORE.read_text())
    fn = _func(tree, "__init__", cls_name="EngineCore")
    # ast.unparse normalizes string quoting, so match on single quotes.
    src = _body_src(fn)
    assert "block_info.get('state_cache_block_size'" in src
    assert "block_info.get('num_state_cache_blocks'" in src

    # ...and before the prefix-caching gate, which reports the resolved value.
    lines = src.splitlines()
    read = next(i for i, ln in enumerate(lines) if "state_cache_block_size" in ln)
    gate = next(i for i, ln in enumerate(lines) if "_has_recurrent_state" in ln)
    assert read < gate


# ── builder hook contract ──────────────────────────────────────────────────


def test_backends_declares_the_checkpoint_hook_defaulting_to_zero():
    """Stateless attentions must opt OUT by default, so the runner stays
    model-agnostic (same shape as the paged-SWA hooks)."""
    tree = ast.parse(BACKENDS.read_text())
    fn = _func(tree, "state_checkpoint_bytes")
    assert len(fn.body) == 2  # docstring + return
    ret = fn.body[-1]
    assert isinstance(ret, ast.Return) and ret.value.value == 0


def test_gdn_checkpoint_excludes_speculative_slots():
    """A checkpoint stores the COMMITTED state only.

    `_state_shape` folds `num_spec` into the conv-state length, so reusing the
    working-slot shape would over-size every checkpoint by `num_spec` conv
    rows per layer. The override must pass num_spec=0.
    """
    tree = ast.parse(GDN_ATTN.read_text())
    fn = _func(tree, "state_checkpoint_bytes", cls_name="GDNAttentionMetadataBuilder")
    src = _body_src(fn)
    assert "num_spec=0" in src


def test_gdn_working_slot_bytes_still_include_speculative_slots():
    """The working pool keeps the rollback slots — the checkpoint change must
    not have narrowed it."""
    tree = ast.parse(GDN_ATTN.read_text())
    per_req = _func(
        tree, "compute_per_req_cache_bytes", cls_name="GDNAttentionMetadataBuilder"
    )
    assert "num_spec" not in _body_src(per_req)  # defaults to the runner's width
    slots = _func(tree, "slots_per_req", cls_name="GDNAttentionMetadataBuilder")
    assert "num_spec" in _body_src(slots)


# ── CLI surface ────────────────────────────────────────────────────────────


def test_cli_exposes_both_names_and_the_fraction():
    """`--mamba-cache-block-size` is the name the feature was requested under;
    `--state-cache-block-size` is the name that matches what it actually
    governs (any recurrent state, not just mamba). Keep both."""
    src = ARG_UTILS.read_text()
    assert '"--state-cache-block-size"' in src
    assert '"--mamba-cache-block-size"' in src
    assert '"--state-cache-target-fraction"' in src
    assert 'dest="state_cache_block_size"' in src


def test_engine_args_defaults_match_config_defaults():
    """A drifted default would silently override the Config default via
    `_get_engine_kwargs`, which passes every EngineArgs field through."""
    cfg = ast.parse(CONFIG.read_text())
    cls = next(
        n for n in ast.walk(cfg) if isinstance(n, ast.ClassDef) and n.name == "Config"
    )
    cfg_defaults = {
        n.target.id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("state_cache_")
        and n.value is not None
    }

    args = ast.parse(ARG_UTILS.read_text())
    eng = next(
        n
        for n in ast.walk(args)
        if isinstance(n, ast.ClassDef) and n.name == "EngineArgs"
    )
    arg_defaults = {
        n.target.id: ast.literal_eval(n.value)
        for n in eng.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("state_cache_")
        and n.value is not None
    }

    assert arg_defaults, "EngineArgs must carry the state-cache fields"
    for name, value in arg_defaults.items():
        assert cfg_defaults.get(name) == value, (
            f"EngineArgs.{name}={value} drifts from Config.{name}="
            f"{cfg_defaults.get(name)}"
        )
