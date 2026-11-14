# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_DSV41_MONO_HI`` (``mono/hi_gate.py``): "auto" (the default) serves exactly the supported deployment and
leaves every other on ATOM's mono path, with one info line; "1" refuses loudly; "0" is off. The decision reads the
configuration, the environment and the device properties only: no hi module is imported.

The configuration is ATOM's own ``Config`` of the served command line (serve_v41.sh's arguments, the InferenceX
recipe at TP4), built the way ``LLMEngine`` builds it, so the gate sees every value ATOM settles on (its overrides
included: V4.1's 128-token KV page whatever ``--block-size`` asks); each refusal case changes one of those fields.

What only the loaded model shows is refused at install (``hi_dispatch.install_hi``): every rank's check agreed over
the TP group before anything is built, so one rank refusing turns the hi path off on every rank.
"""

import argparse
import copy
import logging
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiter_stub import atom_without_aiter

from atom.models.deepseek_v41.mono import hi_gate
from atom.utils import envs

FIXTURES = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures"
# serve_v41.sh's ATOM arguments (the InferenceX dsv41flash recipe at TP4) but the model path and the server's own
CAPTURE = "[" + ",".join(map(str, [*range(1, 33), 48, 64, 128])) + "]"
SERVED_ARGS = (
    "--trust-remote-code --tensor-parallel-size 4 --kv_cache_dtype bf16 --index-cache-dtype fp4 "
    "--gpu-memory-utilization 0.9 --max-num-seqs 128 --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 "
    "--enable_prefix_caching --block-size 16 --state-checkpoint-interval-tokens 8192 --level 3 --cudagraph-mode FULL "
    f"--cudagraph-capture-sizes {CAPTURE} --method dspark --num-speculative-tokens 5"
)


def _atom_config(args: str = SERVED_ARGS):
    """ATOM's ``Config`` of a server command line over the V4.1-Flash fixture: ``EngineArgs`` parsed as the server
    parses it, then ``Config`` built from its engine kwargs as ``LLMEngine`` builds it (every override applied).
    """
    with atom_without_aiter() as load:
        config = load("atom.config")
        arg_utils = load("atom.model_engine.arg_utils")
        parser = argparse.ArgumentParser()
        arg_utils.EngineArgs.add_cli_args(parser)
        ns, _ = parser.parse_known_args(["--model", str(FIXTURES), *args.split()])
        engine = arg_utils.EngineArgs.from_cli_args(ns)
        names = {f.name for f in fields(config.Config)}
        kwargs = engine._get_engine_kwargs()
        return config.Config(
            engine.model, **{k: v for k, v in kwargs.items() if k in names}
        )


@pytest.fixture(scope="module")
def served():
    return _atom_config()


@pytest.fixture(autouse=True)
def fresh_log_once(monkeypatch):
    """Each test sees its own lines: ``log_once`` logs a line once a process."""
    monkeypatch.setattr(hi_gate, "_LOGGED", set())


def _config(served, **over):
    """``served`` (ATOM's ``Config`` of the served command line) with ``over`` changed (``hf``: hf_config fields)."""
    cfg = copy.copy(served)
    hf = over.pop("hf", None)
    if hf:
        cfg.hf_config = copy.copy(served.hf_config)
        for k, v in hf.items():
            setattr(cfg.hf_config, k, v)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _drafter():
    return SimpleNamespace(model=SimpleNamespace(block_backbone=None))


def _builder(uva=True, overlap=True, engram=True):
    if not engram:
        return SimpleNamespace(engram=None)
    host = SimpleNamespace(uva=uva, overlap=object() if overlap else None)
    return SimpleNamespace(engram=SimpleNamespace(host=host))


@pytest.fixture
def mi355x(monkeypatch):
    """A whole MI355X, FlyDSL installed, the defaults' environment."""
    monkeypatch.setattr(hi_gate, "_device", lambda: ("gfx950", 256))
    monkeypatch.setattr(
        hi_gate.importlib.util,
        "find_spec",
        lambda name: object() if name == "flydsl" else None,
    )
    monkeypatch.delenv("ATOM_DSV41_MONO_HI", raising=False)


def test_the_default_is_auto(mi355x):
    assert envs.ATOM_DSV41_MONO_HI == "auto" and hi_gate.mode() == "auto"


def test_auto_serves_the_served_command_line(mi355x, served, caplog):
    before = set(sys.modules)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert hi_gate.decide(served, _drafter(), _builder(), 0) is True
    assert not caplog.records
    # deciding imports nothing of the hi path
    assert not [m for m in set(sys.modules) - before if ".mono.hi" in m]


def test_the_gate_reads_the_values_atom_settled_on(served):
    """Every configuration field the gate reads, as ATOM's ``Config`` holds it for the served command line: the value
    the hook and ATOM's cache use (``--block-size 16`` becomes V4.1's 128-token page).
    """
    assert served.kv_cache_block_size == hi_gate.BLOCK_SIZE == 128
    assert (
        served.tensor_parallel_size,
        served.kv_cache_dtype,
        served.index_cache_dtype,
    ) == (4, "bf16", "fp4")
    assert {
        k: getattr(served.hf_config, k) for k in hi_gate.GEOMETRY
    } == hi_gate.GEOMETRY
    spec = served.speculative_config
    assert (
        spec.method,
        spec.num_speculative_tokens,
        served.hf_config.dspark_block_size,
    ) == ("dspark", 5, 5)
    assert served.max_model_len <= hi_gate.MAX_CTX and not served.kv_transfer_config


@pytest.mark.parametrize("block", [1, 16, 64, 256])
def test_any_block_size_flag_takes_atoms_v41_page(mi355x, block):
    """ATOM lays V4.1's KV out in 128-token pages whatever ``--block-size`` asks: served either way."""
    cfg = _atom_config(SERVED_ARGS.replace("--block-size 16", f"--block-size {block}"))
    assert cfg.kv_cache_block_size == 128
    assert hi_gate.refusal(cfg, _drafter(), _builder(), 0) == ""


def test_tp8_is_served(mi355x, served):
    """TP8 next to TP4 (``hi_runner.install`` builds one request's verify width there; wider steps stay on ATOM's
    mono path)."""
    cfg = _config(served, tensor_parallel_size=8)
    assert hi_gate.refusal(cfg, _drafter(), _builder(), 0) == ""


UNSUPPORTED = {
    "TP1": ({"tensor_parallel_size": 1}, {}, "TP1"),
    "TP2": ({"tensor_parallel_size": 2}, {}, "TP2"),
    "TP16": ({"tensor_parallel_size": 16}, {}, "TP16"),
    "DP attention": ({}, {"dp_ranks": 2}, "DP attention"),
    "fp8 KV": ({"kv_cache_dtype": "fp8"}, {}, "kv_cache_dtype fp8"),
    "fp8 index cache": ({"index_cache_dtype": "fp8"}, {}, "index-cache-dtype fp8"),
    "gfx1250's FP4 index layout": (
        {"index_cache_layout": "natural"},
        {},
        "FP4 index cache layout natural",
    ),
    "another KV page": (
        {"kv_cache_block_size": 256},
        {},
        "KV cache pages of 256 tokens",
    ),
    "no MTP": ({"speculative_config": None}, {"drafter": None}, "no DSpark draft"),
    "another drafter": (
        {
            "speculative_config": SimpleNamespace(
                method="eagle3", num_speculative_tokens=3
            )
        },
        {},
        "no DSpark draft",
    ),
    "DSpark without the block draft": (
        {},
        {"drafter": SimpleNamespace(model=object())},
        "block draft",
    ),
    "3 drafts": (
        {
            "speculative_config": SimpleNamespace(
                method="dspark", num_speculative_tokens=3
            )
        },
        {},
        "3 speculative tokens",
    ),
    "other geometry": ({"hf": {"hidden_size": 7168}}, {}, "geometry"),
    "context past the bucket": ({"max_model_len": 2 * 1048576}, {}, "context bucket"),
    "P/D": ({"kv_transfer_config": {"kv_role": "kv_consumer"}}, {}, "KV transfer"),
    "no metadata builder": ({}, {"builder": None}, "metadata builder"),
    "no Engram": ({}, {"builder": _builder(engram=False)}, "Engram runtime"),
    "Engram without UVA": ({}, {"builder": _builder(uva=False)}, "UVA"),
    "Engram without overlap staging": (
        {},
        {"builder": _builder(overlap=False)},
        "overlap",
    ),
}


@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_auto_leaves_any_other_deployment_on_atoms_path(mi355x, served, caplog, case):
    over, args, why = UNSUPPORTED[case]
    call = {"drafter": _drafter(), "builder": _builder(), "dp_ranks": 0, **args}
    with caplog.at_level(logging.INFO, logger="atom"):
        assert not hi_gate.decide(
            _config(served, **over), call["drafter"], call["builder"], call["dp_ranks"]
        )
    lines = [r.getMessage() for r in caplog.records]
    assert (
        len(lines) == 1
        and lines[0].startswith("V4.1 mono hi off: ")
        and why in lines[0]
    ), lines
    assert caplog.records[0].levelno == logging.INFO


@pytest.mark.parametrize(
    "device, why",
    [
        (("gfx942", 304), "gfx942"),
        (("gfx950", 128), "128 CUs"),
        (("gfx1250", 256), "gfx1250"),
    ],
)
def test_auto_leaves_other_gpus_on_atoms_path(mi355x, served, monkeypatch, device, why):
    monkeypatch.setattr(hi_gate, "_device", lambda: device)
    assert why in hi_gate.refusal(served, _drafter(), _builder(), 0)
    assert not hi_gate.decide(served, _drafter(), _builder(), 0)


def test_auto_without_flydsl(mi355x, served, monkeypatch):
    monkeypatch.setattr(hi_gate.importlib.util, "find_spec", lambda name: None)
    assert "FlyDSL" in hi_gate.refusal(served, _drafter(), _builder(), 0)


@pytest.mark.parametrize("layout", hi_gate.INDEX_LAYOUTS)
def test_both_gfx950_fp4_index_layouts_are_served(mi355x, served, layout):
    """ATOM's FP4 index plane in either gfx950 layout (``ATOM_V4_UNIFIED_MQA``: opus -> mfma, flydsl ->
    preshuffle): the hi programs address both."""
    cfg = _config(served, index_cache_layout=layout)
    assert hi_gate.refusal(cfg, _drafter(), _builder(), 0) == ""


def test_off(mi355x, served, monkeypatch, caplog):
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", "0")
    with caplog.at_level(logging.INFO, logger="atom"):
        assert not hi_gate.decide(served, _drafter(), _builder(), 0)
    assert [r.getMessage() for r in caplog.records] == [
        "V4.1 mono hi off: ATOM_DSV41_MONO_HI=0"
    ]


def test_forced(mi355x, served, monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", "1")
    assert hi_gate.decide(served, _drafter(), _builder(), 0)
    with pytest.raises(RuntimeError, match="TP2"):
        hi_gate.decide(
            _config(served, tensor_parallel_size=2), _drafter(), _builder(), 0
        )


def test_an_unknown_mode_is_an_error(mi355x, served, monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", "yes")
    with pytest.raises(ValueError, match="auto, 1, 0"):
        hi_gate.decide(served, _drafter(), _builder(), 0)


@pytest.mark.parametrize(
    "over, why",
    [
        ({"enable_tbo": True}, "two-batch overlap"),
        ({"enable_tbo_decode": True}, "two-batch overlap"),
        ({"hf": {"vocab_size": 102400}}, "geometry"),
        ({"hf": {"rms_norm_eps": 1e-6}}, "geometry"),
        ({"hf": {"sliding_window": 256}}, "geometry"),
        ({"hf": {"dspark_target_layer_ids": [36, 37, 38, 39]}}, "geometry"),
        ({"hf": {"engram_max_ngram_size": 3}}, "geometry"),
    ],
)
def test_the_loaded_models_assumptions_are_the_gates(mi355x, served, over, why):
    """What the hi programs assume of the model and the engine, answered before anything is built: the verify head's
    vocabulary, the norms' eps, the window, the draft's taps, the Engram's n-grams; no two-batch overlap.
    """
    assert why in hi_gate.refusal(_config(served, **over), _drafter(), _builder(), 0)


def test_triton_moe_keeps_atoms_path(mi355x, served, monkeypatch):
    """``ATOM_USE_TRITON_MOE=1`` (recipes/DeepSeek-V4.md) lays the routed experts out for Triton, not the gfx950 FlyDSL
    layout the hi path reads: "auto" stays on ATOM's path, "1" refuses."""
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    assert "ATOM_USE_TRITON_MOE" in hi_gate.refusal(served, _drafter(), _builder(), 0)
    assert not hi_gate.decide(served, _drafter(), _builder(), 0)
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", "1")
    with pytest.raises(RuntimeError, match="ATOM_USE_TRITON_MOE"):
        hi_gate.decide(served, _drafter(), _builder(), 0)


@pytest.fixture
def dispatch_module(monkeypatch):
    """``mono.dispatch`` with ATOM's own mono runners (FlyDSL kernels) stood in for: loaded fresh, dropped after."""
    import importlib
    import types

    from aiter_stub import _drop

    name = "atom.models.deepseek_v41.mono.dispatch"
    saved = sys.modules.get(name)
    for mod, cls in (
        ("runner", "V41MonoDecodeRunner"),
        ("draft_runner", "DraftMonoRunner"),
    ):
        stub = types.ModuleType(f"atom.models.deepseek_v41.mono.{mod}")
        setattr(stub, cls, object)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    _drop(name)
    try:
        yield importlib.import_module(name)
    finally:
        _drop(name)
        if saved is not None:
            sys.modules[name] = saved


@pytest.mark.parametrize("mode", ["auto", "1"])
def test_a_model_the_hi_path_does_not_serve_falls_back(
    mi355x, served, monkeypatch, caplog, dispatch_module, mode
):
    """A mismatch only the loaded model shows (``MonoUnsupported`` from ``install_hi``, e.g. the routed experts outside
    the gfx950 FlyDSL layout): under "auto" ``install_mono_decode`` logs one line and keeps ATOM's mono decode; under
    "1" it raises."""
    import types

    from atom.mono.runtime.consensus import MonoUnsupported

    dispatch = dispatch_module

    def install_hi(*a, **k):
        raise MonoUnsupported("routed experts not in the gfx950 FlyDSL layout")

    stub = types.ModuleType("atom.models.deepseek_v41.mono.hi_dispatch")
    stub.install_hi = install_hi
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", mode)
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(dispatch, "dp_attention_ranks", lambda cfg: 0)
    monkeypatch.setattr(dispatch.hi_gate, "decide", lambda *a: True)
    monkeypatch.setattr(dispatch, "MonoDecodeModel", lambda *a: "atom mono")
    if mode == "1":
        with pytest.raises(MonoUnsupported, match="FlyDSL layout"):
            dispatch.install_mono_decode("model", served, None, metadata_builder=None)
        return
    with caplog.at_level(logging.INFO, logger="atom"):
        got = dispatch.install_mono_decode("model", served, None, metadata_builder=None)
    assert got == "atom mono"
    assert [r.getMessage() for r in caplog.records] == [
        "V4.1 mono hi off: routed experts not in the gfx950 FlyDSL layout"
    ]


@pytest.mark.parametrize("rank, logged", [(0, 1), (1, 0), (3, 0)])
def test_a_deployments_line_is_logged_once(monkeypatch, caplog, rank, logged):
    """``log_once``: the lines every rank would log alike (the gate's and the fallback's "hi off", the install's) come
    from rank 0 only, each line once a process (another argument is another line)."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    with caplog.at_level(logging.INFO, logger="atom"):
        for why in ("why", "why", "other"):
            hi_gate.log_once(logging.INFO, "V4.1 mono hi off: %s", why)
    assert [r.getMessage() for r in caplog.records] == (
        ["V4.1 mono hi off: why", "V4.1 mono hi off: other"] if logged else []
    )


# ---- the install's refusals, agreed over the TP group (``hi_dispatch.install_hi``)


@pytest.fixture
def hi_dispatch_module(monkeypatch):
    """``mono.hi_dispatch`` (Triton stood in for where AITER is missing) with ATOM's own mono runners stood in for:
    loaded fresh, dropped after; this process is TP rank 1 of 4 (``_tp``)."""
    import types

    from aiter_stub import _drop

    names = [f"atom.models.deepseek_v41.mono.{m}" for m in ("hi_dispatch", "dispatch")]
    saved = {name: sys.modules.get(name) for name in names}
    for mod, cls in (
        ("runner", "V41MonoDecodeRunner"),
        ("draft_runner", "DraftMonoRunner"),
    ):
        stub = types.ModuleType(f"atom.models.deepseek_v41.mono.{mod}")
        setattr(stub, cls, object)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    for name in names:
        _drop(name)
    try:
        with atom_without_aiter() as load:
            module = load(names[0])
            monkeypatch.setattr(module, "_tp", lambda: ("tp cpu group", 4, 1))
            yield module
    finally:
        for name in names:
            _drop(name)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def _install_world(hi_dispatch, monkeypatch, *, local=None, others=(None, None, None)):
    """``install_hi``'s collaborators recorded in ``log``: this rank's check (raising ``local`` when given), the
    agreement's gather (the other ranks' reports ``others``: 0, 2, 3) and every build step.
    """
    import torch.distributed as dist

    log = []

    def check(model, atom_config, engram, draft, tp, rank):
        log.append(("check", tp, rank))
        if local is not None:
            raise local

    def gather(reports, report, group=None):
        log.append(("agree", report, group))
        reports[:] = [others[0], report, *others[1:]]

    def built(name):
        def step(*a, **k):
            log.append(name)
            return SimpleNamespace(
                eng={}, bind_draft=lambda d: log.append("bind_draft")
            )

        return step

    monkeypatch.setattr(hi_dispatch.hi_runner, "check", check)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(hi_dispatch.hi_runner, "install", built("hi layers"))
    monkeypatch.setattr(hi_dispatch.hi_round, "install", built("round"))
    monkeypatch.setattr(hi_dispatch, "HiDraftModel", built("draft model"))
    monkeypatch.setattr(hi_dispatch, "HiDecodeModel", built("decode model"))
    draft = SimpleNamespace(block_backbone=None)
    drafter = SimpleNamespace(model=draft)
    builder = SimpleNamespace(
        engram=SimpleNamespace(mapping=None, host=None),
        model_runner=None,
        use_decode_round=lambda rnd: log.append("use round"),
    )
    cfg = SimpleNamespace(max_model_len=4096, index_cache_dtype="fp4")
    return log, (object(), cfg, drafter, builder, 0), draft


@pytest.mark.parametrize(
    "local, others, why",
    [
        # this rank's own expert shard (a rank-local data check)
        (
            "padding",
            (None, None, None),
            (
                "refused at install: rank 1: MonoUnsupported: V4.1 mono hi weights: a routed expert's padding scale "
                "code is 255 (E8M0 NaN)"
            ),
        ),
        # another rank's; this one passed
        (
            None,
            (None, "MonoUnsupported: Engram host shard", None),
            "refused at install: rank 2: MonoUnsupported: Engram host shard",
        ),
        # any failure of the check is a refusal every rank takes (alone, it would leave the others waiting)
        (
            "oom",
            (None, None, None),
            "refused at install: rank 1: RuntimeError: out of memory",
        ),
    ],
)
def test_one_rank_refusing_at_install_refuses_on_every_rank_before_anything_is_built(
    hi_dispatch_module, monkeypatch, local, others, why
):
    """``install_hi``: this rank's check, then every rank's verdict agreed (one gather, this rank's report in it),
    then the build. One rank refusing raises ``MonoUnsupported`` on every rank, naming it, before the first
    collective of the build and before anything is built or allocated (no peer group, no packed weight); the draft
    model is left as it was."""
    from atom.mono.runtime.consensus import MonoUnsupported

    error = {
        "padding": MonoUnsupported(
            "V4.1 mono hi weights: a routed expert's padding scale code is 255 (E8M0 NaN)"
        ),
        "oom": RuntimeError("out of memory"),
        None: None,
    }[local]
    log, args, draft = _install_world(
        hi_dispatch_module, monkeypatch, local=error, others=others
    )
    with pytest.raises(MonoUnsupported) as raised:
        hi_dispatch_module.install_hi(*args)
    assert str(raised.value) == why
    report = None if error is None else f"{type(error).__name__}: {error}"
    assert log == [("check", 4, 1), ("agree", report, "tp cpu group")]
    assert args[2].model is draft


def test_every_rank_passing_builds_after_the_agreement(hi_dispatch_module, monkeypatch):
    """Every rank's check passed: the build follows the agreement; the decode round goes to V4.1's metadata builder
    (``use_decode_round``) last, before the model is wrapped."""
    log, args, _ = _install_world(hi_dispatch_module, monkeypatch)
    hi_dispatch_module.install_hi(*args)
    assert log == [("check", 4, 1), ("agree", None, "tp cpu group"), "hi layers", "draft model", "bind_draft",
                   "round", "use round", "decode model"]  # fmt: skip


def test_a_refusal_on_another_rank_keeps_atoms_mono_path_here(
    mi355x, served, monkeypatch, caplog, hi_dispatch_module
):
    """Through ``install_mono_decode`` under "auto": a refusal agreed at install (here another rank's) leaves this
    rank on ATOM's mono decode, as every other rank, with one line naming the rank and its reason.
    """
    dispatch = sys.modules["atom.models.deepseek_v41.mono.dispatch"]
    _, args, draft = _install_world(
        hi_dispatch_module, monkeypatch, others=("MonoUnsupported: shard", None, None)
    )
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(dispatch, "dp_attention_ranks", lambda cfg: 0)
    monkeypatch.setattr(dispatch.hi_gate, "decide", lambda *a: True)
    monkeypatch.setattr(dispatch, "MonoDecodeModel", lambda *a: "atom mono")
    monkeypatch.setattr(
        dispatch, "MonoDraftModel", lambda m, dp: ("atom mono draft", m)
    )
    model, cfg, drafter, builder, _ = args
    with caplog.at_level(logging.INFO, logger="atom"):
        got = dispatch.install_mono_decode(
            model, cfg, drafter, metadata_builder=builder
        )
    assert got == "atom mono"
    assert drafter.model == ("atom mono draft", draft)
    assert [r.getMessage() for r in caplog.records] == [
        "V4.1 mono hi off: refused at install: rank 0: MonoUnsupported: shard"
    ]
