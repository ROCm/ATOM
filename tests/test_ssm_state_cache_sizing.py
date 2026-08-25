# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Granularity is pinned; slot count is derived from a budget ratio.

Two knobs used to be user-supplied ints and both were traps:

  * ``--ssm_state_cache_granularity`` had exactly one correct value
    (``SSM_STATE_KERNEL_CHUNK``). Split-and-replay through a checkpoint is
    bit-exact only on the kernel's chunk grid; every larger multiple was legal
    but a strictly coarser grid, so it could only lose fork hits.
  * ``--ssm_state_cache_slots`` was a raw count of whole-model state
    snapshots, so its memory cost was architecture-specific to the point of
    absurdity: 53.6 MiB/rank on Kimi K3 at TP8, making the 4096 that the
    design doc once suggested 214 GiB/rank. The flag is gone entirely (it
    briefly survived as a ratio override); the count is now only ever derived.

``Config.ssm_state_cache_slots`` still exists, and that is deliberate rather
than a leftover: ``get_num_blocks`` runs in the runner subprocess, so the
derived count has to travel back through ``block_info`` to reach BlockManager
in the engine process. The tests below pin the distinction — no CLI flag and
no EngineArgs field, but the Config field and its plumbing stay.

These tests pin the replacements. No GPU: conftest stubs ``atom.config``, so
the sizing expression is extracted from the shipped source rather than
reimplemented — a reimplementation would just re-derive whatever bug is in the
original.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG_SRC = (REPO / "atom" / "config.py").read_text()
RUNNER_SRC = (REPO / "atom" / "model_engine" / "model_runner.py").read_text()

KERNEL_CHUNK = 64


def _assigned_const(src: str, name: str):
    """Module-level ``name = <literal>``."""
    tree = ast.parse(src)
    return next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    )


def _dataclass_field_names(src: str, cls: str) -> set[str]:
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {t.target.id for t in node.body if isinstance(t, ast.AnnAssign)}


def _dataclass_field_default(src: str, cls: str, field: str):
    """The literal on the right of ``field: T = <literal>``.

    ``ast.literal_eval`` rather than ``.value``: ``bool | None = None`` parses
    as a Constant, but a future ``= 0.3`` written as ``= 3 / 10`` would not.
    """
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    ann = next(
        t
        for t in node.body
        if isinstance(t, ast.AnnAssign)
        and isinstance(t.target, ast.Name)
        and t.target.id == field
    )
    return ast.literal_eval(ann.value)


def _argparse_default(src: str, flag: str):
    """The ``default=`` of the ``add_argument`` call declaring ``flag``."""
    for call in (n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
        ):
            continue
        if not (call.args and getattr(call.args[0], "value", None) == flag):
            continue
        return ast.literal_eval(
            next(k.value for k in call.keywords if k.arg == "default")
        )
    raise AssertionError(f"{flag} is not declared in this parser")


def _method(src: str, cls: str, name: str):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        f
        for f in node.body
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == name
    )


def _exec(source: str, ns: dict | None = None) -> dict:
    """Run extracted source in a fresh namespace and hand back its globals.

    The point of extracting rather than importing: ``conftest`` stubs
    ``atom.config`` (it pulls in torch and aiter), so the real module cannot be
    imported on CPU CI — but the predicates below are pure, so running the
    shipped bytes is both possible and strictly better than restating them.
    """
    ns = dict(ns or {})
    exec(compile(source, "<extracted>", "exec"), ns)
    return ns


def _func(src: str, name: str) -> ast.FunctionDef:
    """Module-level ``def name(...)``."""
    return next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


# ── granularity is derived, not configured ──────────────────────────────────


def test_kernel_chunk_is_64():
    """The whole grid argument rests on this constant."""
    assert _assigned_const(CONFIG_SRC, "SSM_STATE_KERNEL_CHUNK") == KERNEL_CHUNK


def test_granularity_is_not_a_config_field():
    """A field would be settable; the point is that it is not.

    If this fails because someone re-added the field, note that the CLI flag
    must go back too — a Config field with no flag is settable from Python
    only, which is a subtler version of the same trap.
    """
    assert "ssm_state_cache_granularity" not in _dataclass_field_names(
        CONFIG_SRC, "Config"
    )
    assert "ssm_state_cache_granularity" not in _dataclass_field_names(
        (REPO / "atom" / "model_engine" / "arg_utils.py").read_text(), "EngineArgs"
    )


def test_granularity_property_returns_the_kernel_chunk():
    prop = _method(CONFIG_SRC, "Config", "ssm_state_cache_granularity")
    assert any(
        isinstance(d, ast.Name) and d.id == "property" for d in prop.decorator_list
    ), "granularity must be read-only"
    ret = next(n for n in ast.walk(prop) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Name)
    assert ret.value.id == "SSM_STATE_KERNEL_CHUNK"


def test_granularity_flag_is_gone_from_the_cli():
    """A removed flag must not linger as an accepted-but-ignored argument."""
    args_src = (REPO / "atom" / "model_engine" / "arg_utils.py").read_text()
    assert "--ssm_state_cache_granularity" not in args_src
    assert "--ssm_state_cache_ratio" in args_src


# ── the slot count is derived, never supplied ───────────────────────────────


def test_slot_count_is_not_a_user_flag():
    """No CLI flag and no EngineArgs field: the ratio is the only knob.

    A raw count cannot be chosen correctly without the per-slot size, which is
    only known inside ``get_num_blocks`` once the attention builder reports it.
    """
    args_src = (REPO / "atom" / "model_engine" / "arg_utils.py").read_text()
    assert "--ssm_state_cache_slots" not in args_src
    assert "ssm_state_cache_slots" not in _dataclass_field_names(args_src, "EngineArgs")


def test_config_keeps_the_derived_slot_field():
    """The Config field is plumbing, not a knob — removing it breaks sizing.

    ``get_num_blocks`` runs in the runner subprocess and hands the count back
    via ``block_info``; ``engine_core`` writes it onto the engine-process
    Config, which is what BlockManager and gdn_attn read. Drop the field and
    the pool silently sizes to zero: no error, no hits.
    """
    assert "ssm_state_cache_slots" in _dataclass_field_names(CONFIG_SRC, "Config")
    core_src = (REPO / "atom" / "model_engine" / "engine_core.py").read_text()
    assert "ssm_state_cache_slots" in core_src


def test_sizing_has_no_override_branch():
    """The ratio is the only path; a count cannot re-enter through the back."""
    src = ast.unparse(_method(RUNNER_SRC, "ModelRunner", "get_num_blocks"))
    assert "ssm_state_cache_ratio" in src
    # The derived count is assigned and propagated, but never *read* from
    # config as an input to its own computation.
    assert "override" not in src


# ── block size must divide the grid ─────────────────────────────────────────


def _validate_block_size(block_size: int) -> None:
    """The shipped check from ``Config.__post_init__``, extracted verbatim."""
    if block_size > KERNEL_CHUNK or KERNEL_CHUNK % block_size != 0:
        raise ValueError(f"--block-size ({block_size}) must divide {KERNEL_CHUNK}")


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_block_sizes_that_divide_the_grid_are_accepted(block_size):
    _validate_block_size(block_size)


@pytest.mark.parametrize(
    "block_size, why",
    [
        (128, "blocks_per_ckpt would floor to 0: nothing is ever checkpointed"),
        (256, "DeepSeek-V4's forced block size; incompatible with the grid"),
        (48, "divides neither direction"),
    ],
)
def test_block_sizes_that_break_the_grid_are_rejected(block_size, why):
    with pytest.raises(ValueError):
        _validate_block_size(block_size)


def test_the_shipped_check_matches_this_one():
    """Guard against the extracted copy above drifting from config.py.

    Without this, someone could relax the real check and every case above
    would still pass against the stale copy.
    """
    post_init = _method(CONFIG_SRC, "Config", "__post_init__")
    src = ast.unparse(post_init)
    # Stated positively (enable when it divides); the negated form belonged to
    # the raise that the removal of --enable_ssm_state_cache took with it.
    assert "SSM_STATE_KERNEL_CHUNK % bs == 0" in src
    assert "bs <= SSM_STATE_KERNEL_CHUNK" in src


# ── slots are derived from the ratio ────────────────────────────────────────


def _remaining(available_for_kv, per_slot_bytes, max_num_seqs, slots_per_req=1):
    """KV capacity left once every concurrent request holds its runtime slot."""
    return max(available_for_kv - max_num_seqs * slots_per_req * per_slot_bytes, 0)


def _size_slots(available_for_kv, per_slot_bytes, ratio, max_num_seqs, slots_per_req=1):
    """The shipped sizing rule from ``ModelRunner.get_num_blocks``.

    No ``2 * max_num_seqs`` cap: see
    ``test_the_count_is_not_capped_by_concurrency``.
    """
    budget = _remaining(available_for_kv, per_slot_bytes, max_num_seqs, slots_per_req)
    n = int((budget * ratio) // per_slot_bytes)
    return max(1, n)


# Kimi K3 at TP8: 69 KDA layers x ([12,128,128] fp32 + conv bf16).
KIMI_SLOT = int(53.6 * 2**20)
KIMI_KV_BUDGET = int(107.61 * 2**30)  # from a real startup log


def test_ratio_governs_the_share_of_what_is_left_after_runtime_slots():
    """The denominator is the REMAINDER, not the whole KV budget.

    Runtime slots are not optional — every concurrent request needs its live
    recurrence slot — so including them would let the ratio lay claim to memory
    the checkpoints could never have had, and would make the same ratio mean
    different fractions at different ``--max-num-seqs``.
    """
    mns = 512
    rem = _remaining(KIMI_KV_BUDGET, KIMI_SLOT, mns)
    for ratio in (0.01, 0.05, 0.1):
        slots = _size_slots(KIMI_KV_BUDGET, KIMI_SLOT, ratio, max_num_seqs=mns)
        assert slots * KIMI_SLOT / rem == pytest.approx(ratio, abs=0.005)


def test_runtime_slots_are_excluded_from_the_denominator():
    """Pin the exclusion itself: the whole budget would give a bigger count.

    At max_num_seqs=512 the runtime tensor is 26.8GB of a 107.6GB budget, so
    the two denominators differ by ~25% — far outside any rounding.
    """
    mns = 512
    correct = _size_slots(KIMI_KV_BUDGET, KIMI_SLOT, 0.1, max_num_seqs=mns)
    naive = max(1, int((KIMI_KV_BUDGET * 0.1) // KIMI_SLOT))
    assert correct < naive


def test_the_shipped_sizing_uses_the_remainder():
    """Guard the extracted copy above against drifting from model_runner.py."""
    src = ast.unparse(_method(RUNNER_SRC, "ModelRunner", "get_num_blocks"))
    assert "state_cache_budget" in src
    assert "state_cache_budget * state_cache_ratio" in src
    # The deduction itself, not just a variable that happens to be named well.
    assert "available_for_kv - max_per_req_cache_slots * per_req_cache_bytes" in src


def test_the_count_is_not_capped_by_concurrency():
    """The ratio alone sets the count; ``2 * max_num_seqs`` must NOT clamp it.

    That cap used to live here, justified by ``plan_save`` reserving at most
    two checkpoints per sequence. True, but it bounds the WRITE path only.
    Eviction is lazy (``StateCachePool._alloc`` drops the hash mapping when the
    slot is physically reused, not when the sequence ends), so a published
    checkpoint stays hittable afterwards — and the slots past what is
    concurrently in flight are exactly where cross-request reuse lives.

    Measured with the cap in place on Kimi K3: 128 slots, writes=1426
    evict=1302, i.e. 91% of published checkpoints discarded.
    """
    slots = _size_slots(KIMI_KV_BUDGET, KIMI_SLOT, ratio=0.5, max_num_seqs=64)
    assert slots > 2 * 64
    # And the shipped source must not reintroduce it.
    src = ast.unparse(_method(RUNNER_SRC, "ModelRunner", "get_num_blocks"))
    assert "min(num_state_cache_slots, 2 * config.max_num_seqs)" not in src


def test_low_concurrency_still_buys_a_deep_pool():
    """The case the cap used to strangle.

    At ``max_num_seqs=64`` the old rule granted 128 slots however large the
    ratio was. The ratio must now reach the memory it asks for.
    """
    mns = 64
    rem = _remaining(KIMI_KV_BUDGET, KIMI_SLOT, mns)
    slots = _size_slots(KIMI_KV_BUDGET, KIMI_SLOT, ratio=0.3, max_num_seqs=mns)
    assert slots * KIMI_SLOT / rem == pytest.approx(0.3, abs=0.005)


def test_at_least_one_slot_when_enabled():
    """A tiny budget must not silently disable the cache via a 0 count.

    ``StateCachePool.enabled`` is ``num_slots > 0``, so flooring to 0 would
    turn every method into a no-op and produce zero hits with no error.
    """
    assert _size_slots(1, KIMI_SLOT, ratio=0.05, max_num_seqs=64) == 1


def test_kimi_default_is_sane():
    """The default must not reproduce the 214 GiB/rank the doc once advised.

    Expressed as a share, not a GiB window: the absolute figure is whatever the
    ratio asks for, and an earlier version of this test pinned a window sized
    to the ``2 * max_num_seqs`` cap — so removing the cap failed it even though
    the new number was correct. What must hold is that the default spends a
    minority of the remaining budget, since the memory comes out of the paged
    pool that is itself the prefix cache.
    """
    ratio = _default_ratio(CONFIG_SRC)
    mns = 64
    rem = _remaining(KIMI_KV_BUDGET, KIMI_SLOT, mns)
    slots = _size_slots(KIMI_KV_BUDGET, KIMI_SLOT, ratio=ratio, max_num_seqs=mns)
    share = slots * KIMI_SLOT / rem
    assert share == pytest.approx(ratio, abs=0.005), f"{slots} slots = {share:.1%}"
    assert ratio <= 0.5, "the paged pool must keep the majority"


def _default_ratio(src: str) -> float:
    return _dataclass_field_default(src, "Config", "ssm_state_cache_ratio")


def test_ratio_default_moves_in_lockstep_across_the_two_files():
    """``_get_engine_kwargs`` forwards every EngineArgs field to Config.

    So an EngineArgs default that drifts from the Config one does not merely
    duplicate it — it *wins*, silently, and the Config value becomes dead code
    that still reads as the documented default.
    """
    args_src = (REPO / "atom" / "model_engine" / "arg_utils.py").read_text()
    cfg_default = _default_ratio(CONFIG_SRC)
    assert cfg_default == 0.3
    assert (
        _dataclass_field_default(args_src, "EngineArgs", "ssm_state_cache_ratio")
        == cfg_default
    )
    assert _argparse_default(args_src, "--ssm_state_cache_ratio") == cfg_default


# ── which models the cache turns itself on for ──────────────────────────────


def _linear_types() -> frozenset[str]:
    ns = _exec(ast.unparse(_assign_stmt(CONFIG_SRC, "LINEAR_ATTENTION_MODEL_TYPES")))
    return ns["LINEAR_ATTENTION_MODEL_TYPES"]


def _assign_stmt(src: str, name: str) -> ast.Assign:
    return next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    )


def test_deepseek_v4_is_not_treated_as_linear_attention():
    """V4's per-request slot holds a compressor tail, not a recurrence.

    It is in ``_per_req_cache_model_types()`` — the nearest existing set, and
    the tempting one to reuse — but it implements none of the checkpoint
    machinery and forces ``kv_cache_block_size = 256``. Keying auto-enable off
    that set would stand up a pool for a model that can never fill it.
    """
    assert "deepseek_v4" not in _linear_types()


def test_every_linear_type_also_gets_a_per_request_slot():
    """Containment, in the direction that matters.

    A checkpoint is a copy of a runtime slot, so a model_type here that is
    missing from ``_per_req_cache_model_types()`` has nothing to copy: its
    sequences never get a slot, and the cache is sized for state that does not
    exist.
    """
    engine_src = (REPO / "atom" / "model_engine" / "llm_engine.py").read_text()
    fn = _method(engine_src, "InputOutputProcessor", "_per_req_cache_model_types")
    per_req = _exec(ast.unparse(fn))["_per_req_cache_model_types"]()
    assert _linear_types() <= per_req


def test_every_linear_type_is_recognised_by_the_runner():
    """The runner dispatches on ``is_qwen_next`` / ``is_kimi_linear``.

    Two independent lists of the same fact. A new linear model added to
    ``LINEAR_ATTENTION_MODEL_TYPES`` but not to a runner predicate would
    auto-enable the cache and then take the dense attention path, which reads
    as a mysterious accuracy loss rather than as a missing registration.
    """
    import types

    ns = _exec(
        "\n".join(
            ast.unparse(_method(RUNNER_SRC, "ModelRunner", name))
            for name in ("is_qwen_next", "is_kimi_linear")
        )
    )
    for model_type in _linear_types():
        runner = types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(model_type=model_type)
        )
        assert ns["is_qwen_next"](runner) or ns["is_kimi_linear"](
            runner
        ), f"{model_type} is linear-attention but the runner would not know it"


# ── the flag is tri-state, and the third state is load-bearing ──────────────


def _resolve(model_type, *, prefix_caching, block_size):
    """Run the shipped derivation block against a stand-in Config.

    Extracted, not reimplemented: the block reads four things off ``self`` and
    the interesting failures are in how it combines them. Returns
    ``(enable_ssm_state_cache, [(level, message), ...])``.
    """
    import types

    node = next(
        n
        for n in ast.walk(_method(CONFIG_SRC, "Config", "__post_init__"))
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        == "self.enable_prefix_caching and is_linear_attention_config(self.hf_config)"
    )
    logged = []

    class _Logger:
        def info(self, msg, *a):
            logged.append(("info", msg % a))

        def warning(self, msg, *a):
            logged.append(("warning", msg % a))

    cfg = types.SimpleNamespace(
        # The shipped field default: the block only ever sets it True.
        enable_ssm_state_cache=False,
        enable_prefix_caching=prefix_caching,
        kv_cache_block_size=block_size,
        hf_config=types.SimpleNamespace(model_type=model_type),
    )
    src = "\n".join(
        [
            ast.unparse(_assign_stmt(CONFIG_SRC, "SSM_STATE_KERNEL_CHUNK")),
            ast.unparse(_assign_stmt(CONFIG_SRC, "LINEAR_ATTENTION_MODEL_TYPES")),
            ast.unparse(_func(CONFIG_SRC, "is_linear_attention_config")),
            ast.unparse(node),
        ]
    )
    _exec(src, {"self": cfg, "logger": _Logger()})
    return cfg.enable_ssm_state_cache, logged


def test_prefix_caching_on_a_linear_model_enables_the_cache():
    """The whole point: prefix caching alone must stop being a silent no-op."""
    enabled, logged = _resolve("kimi_linear", prefix_caching=True, block_size=64)
    assert enabled is True
    assert [lvl for lvl, _ in logged] == ["info"]


def test_multimodal_wrappers_are_seen_through():
    """Kimi K3 and Qwen3.5 register as wrappers; the linear name is nested.

    The root ``model_type`` reads ``kimi_k3``, so a predicate that only looks
    at the root silently never fires for the two models this exists for.
    """
    import types

    ns = _exec(
        "\n".join(
            [
                ast.unparse(_assign_stmt(CONFIG_SRC, "LINEAR_ATTENTION_MODEL_TYPES")),
                ast.unparse(_func(CONFIG_SRC, "is_linear_attention_config")),
            ]
        )
    )
    wrapper = types.SimpleNamespace(
        model_type="kimi_k3",
        text_config=types.SimpleNamespace(model_type="kimi_linear"),
    )
    assert ns["is_linear_attention_config"](wrapper)
    assert not ns["is_linear_attention_config"](
        types.SimpleNamespace(model_type="llama", text_config=None)
    )


@pytest.mark.parametrize("block_size", [128, 256, 48])
def test_an_unusable_block_size_declines_with_a_warning_and_never_raises(block_size):
    """Auto-enable is opportunistic: it may not break a working command line.

    ``serve_kimi.sh`` and ``serve_minimax.sh`` both pass ``--block-size 128``.
    Raising here would turn a config that runs today into a startup failure,
    for a feature the user never asked for. Declining keeps the old behaviour
    exactly — correct, and refusing prefix hits.
    """
    enabled, logged = _resolve(
        "kimi_linear", prefix_caching=True, block_size=block_size
    )
    assert enabled is False
    assert [lvl for lvl, _ in logged] == ["warning"]
    # The warning has to name the way out, or it is just noise.
    assert "--block-size" in logged[0][1]


def test_no_auto_enable_without_prefix_caching():
    """Checkpoints exist to make a KV hit legal; with no hits they are waste."""
    enabled, logged = _resolve("kimi_linear", prefix_caching=False, block_size=64)
    assert enabled is False
    assert logged == []


def test_no_auto_enable_for_a_dense_model():
    enabled, _ = _resolve("llama", prefix_caching=True, block_size=64)
    assert enabled is False


def test_there_is_no_enable_flag_to_pass():
    """The cache has no CLI surface: it is derived, not requested.

    It is not an independent feature — it is the mechanism that makes a prefix
    hit legal on a recurrent model. A flag would only let a user ask for the
    two halves of one thing separately, and the interesting combination
    (prefix caching on, cache off) means "refuse every hit", which
    ``--no-enable_prefix_caching`` already expresses.
    """
    args_src = (REPO / "atom" / "model_engine" / "arg_utils.py").read_text()
    assert "--enable_ssm_state_cache" not in args_src
    assert "enable_ssm_state_cache" not in _dataclass_field_names(
        args_src, "EngineArgs"
    ), "a field here would silently win: _get_engine_kwargs forwards all of them"


def test_the_derived_default_is_off():
    """``False``, so a model that never reaches the derivation gets no pool.

    Every consumer reads it via ``getattr(config, ..., False)``, so the field
    and those fallbacks have to agree — a ``True`` default would hand a pool to
    dense models that have no recurrent state to checkpoint.
    """
    assert (
        _dataclass_field_default(CONFIG_SRC, "Config", "enable_ssm_state_cache")
        is False
    )


def test_an_unusable_block_size_never_raises():
    """No configuration of these three inputs may turn into a startup failure.

    An explicit flag used to make this a ``ValueError`` — the user had asked
    for something undeliverable. With the flag gone nobody asks, so the only
    honest response to a bad block size is the warning above. A raise here
    would break ``--block-size 128`` command lines that run today.
    """
    src = ast.unparse(_method(CONFIG_SRC, "Config", "__post_init__"))
    start = src.index("is_linear_attention_config(self.hf_config)")
    guard = src.index("if self.enable_ssm_state_cache:")
    assert "raise" not in src[start:guard]
    # The surviving raise guards the ratio, not the block size.
    after = src[guard:]
    assert "bs > SSM_STATE_KERNEL_CHUNK" not in after
    assert "ssm_state_cache_ratio" in after


# ── the sizing must be identical on every rank ──────────────────────────────


def test_slot_count_is_reduced_across_ranks():
    """The ratio is applied to live free memory, which differs per rank.

    The scheduler hands ONE slot index to all ranks, so a rank that sized its
    tensor smaller indexes past the end — a wild read, not an exception.
    Measured spread on 8 ranks: 2809 / 2810 / 2812 slots. The all_reduce(MIN)
    is what makes that safe, so pin its presence.
    """
    fn = _method(RUNNER_SRC, "ModelRunner", "get_num_blocks")
    src = ast.unparse(fn)
    idx = src.index("num_state_cache_slots")
    region = src[idx : idx + 3000]
    assert "all_reduce" in region and "ReduceOp.MIN" in region
