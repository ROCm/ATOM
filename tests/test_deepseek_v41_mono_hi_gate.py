# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_DSV41_MONO_HI`` (``mono/hi_gate.py``): "auto" (the default) serves exactly the supported deployment and
leaves every other on ATOM's mono path, with one info line; "1" refuses loudly; "0" is off. The decision reads the
configuration, the environment and the device properties only: no hi module is imported.

The configuration is ATOM's own ``Config`` of the served command line (serve_v41.sh's arguments, the InferenceX
recipe at TP4), built the way ``LLMEngine`` builds it, so the gate sees every value ATOM settles on (its overrides
included: V4.1's 128-token KV page whatever ``--block-size`` asks); each refusal case changes one of those fields.
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
    monkeypatch.delenv("ATOM_H2D_BACKEND", raising=False)


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


UNSUPPORTED = {
    "TP1": (dict(tensor_parallel_size=1), {}, "TP1"),
    "TP2": (dict(tensor_parallel_size=2), {}, "TP2"),
    "TP8": (dict(tensor_parallel_size=8), {}, "TP8"),
    "DP attention": ({}, dict(dp_ranks=2), "DP attention"),
    "fp8 KV": (dict(kv_cache_dtype="fp8"), {}, "kv_cache_dtype fp8"),
    "fp8 index cache": (dict(index_cache_dtype="fp8"), {}, "index-cache-dtype fp8"),
    "another KV page": (
        dict(kv_cache_block_size=256),
        {},
        "KV cache pages of 256 tokens",
    ),
    "no MTP": (dict(speculative_config=None), dict(drafter=None), "no DSpark draft"),
    "another drafter": (
        dict(
            speculative_config=SimpleNamespace(
                method="eagle3", num_speculative_tokens=3
            )
        ),
        {},
        "no DSpark draft",
    ),
    "DSpark without the block draft": (
        {},
        dict(drafter=SimpleNamespace(model=object())),
        "block draft",
    ),
    "3 drafts": (
        dict(
            speculative_config=SimpleNamespace(
                method="dspark", num_speculative_tokens=3
            )
        ),
        {},
        "3 speculative tokens",
    ),
    "other geometry": (dict(hf=dict(hidden_size=7168)), {}, "geometry"),
    "context past the bucket": (dict(max_model_len=2 * 1048576), {}, "context bucket"),
    "P/D": (dict(kv_transfer_config={"kv_role": "kv_consumer"}), {}, "KV transfer"),
    "no metadata builder": ({}, dict(builder=None), "metadata builder"),
    "no Engram": ({}, dict(builder=_builder(engram=False)), "Engram runtime"),
    "Engram without UVA": ({}, dict(builder=_builder(uva=False)), "UVA"),
    "Engram without overlap staging": (
        {},
        dict(builder=_builder(overlap=False)),
        "overlap",
    ),
}


@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_auto_leaves_any_other_deployment_on_atoms_path(mi355x, served, caplog, case):
    over, args, why = UNSUPPORTED[case]
    call = {**dict(drafter=_drafter(), builder=_builder(), dp_ranks=0), **args}
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


def test_auto_without_the_packed_h2d(mi355x, served, monkeypatch):
    monkeypatch.setenv("ATOM_H2D_BACKEND", "copy")
    assert "ATOM_H2D_BACKEND" in hi_gate.refusal(served, _drafter(), _builder(), 0)


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
    with pytest.raises(RuntimeError, match="TP8"):
        hi_gate.decide(
            _config(served, tensor_parallel_size=8), _drafter(), _builder(), 0
        )


def test_an_unknown_mode_is_an_error(mi355x, served, monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MONO_HI", "yes")
    with pytest.raises(ValueError, match="auto, 1, 0"):
        hi_gate.decide(served, _drafter(), _builder(), 0)
