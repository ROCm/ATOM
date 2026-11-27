# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 BF16 keep-out resolution, and that no engine-config key is shadowed.

Review feedback on #39. The duplicate-key test earns its place: #39 originally
repeated ``"r3_enabled"`` in one dict literal, Python kept the later entry, and
the whole release suite passed with R3 resolution silently replaced.
"""

import ast
import pathlib

from atomrl.core.config import (
    QuantizationConfig,
    RolloutQuantConfig,
    TrainingQuantConfig,
)
from atomrl.quantization.fp8_config import FP8Config
from atomrl.workers.actor_worker import _layers_in_bf16

_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIRST = "num_first_layers_in_bf16"
LAST = "num_last_layers_in_bf16"


def _cfg(*, training=None, rollout=None):
    quant = {}
    if training is not None:
        quant["training"] = training
    if rollout is not None:
        quant["rollout"] = rollout
    return {"quantization": quant}


# --- precedence --------------------------------------------------------------


def test_training_wins_over_rollout():
    cfg = _cfg(training={FIRST: 3}, rollout={FIRST: 1})
    assert _layers_in_bf16(cfg, FIRST) == 3


def test_rollout_is_the_fallback():
    """configs/grpo_dense_fp8.yaml sets these under rollout only."""
    cfg = _cfg(rollout={FIRST: 2, LAST: 4})
    assert _layers_in_bf16(cfg, FIRST) == 2
    assert _layers_in_bf16(cfg, LAST) == 4


def test_an_explicit_training_zero_overrides_a_nonzero_rollout():
    """0 means 0, not "unset" -- asymmetry has to be requestable."""
    cfg = _cfg(training={FIRST: 0}, rollout={FIRST: 2})
    assert _layers_in_bf16(cfg, FIRST) == 0


def test_absent_everywhere_is_zero():
    for cfg in ({}, {"quantization": {}}, _cfg(training={}, rollout={})):
        assert _layers_in_bf16(cfg, FIRST) == 0


def test_a_non_dict_section_does_not_raise():
    """get_nested_config defaults on a dataclass section."""
    assert _layers_in_bf16({"quantization": {"training": object()}}, FIRST) == 0


# --- the same precedence in FP8Config ----------------------------------------


def test_fp8config_defaults_to_the_rollout_value():
    quant = QuantizationConfig(
        rollout=RolloutQuantConfig(
            num_first_layers_in_bf16=2, num_last_layers_in_bf16=4
        ),
        training=TrainingQuantConfig(fp8="hybrid"),
    )
    cfg = FP8Config.from_config(quant)
    assert (cfg.num_first_layers_in_bf16, cfg.num_last_layers_in_bf16) == (2, 4)


def test_fp8config_lets_training_deviate():
    quant = QuantizationConfig(
        rollout=RolloutQuantConfig(
            num_first_layers_in_bf16=2, num_last_layers_in_bf16=4
        ),
        training=TrainingQuantConfig(
            fp8="hybrid", num_first_layers_in_bf16=0, num_last_layers_in_bf16=1
        ),
    )
    cfg = FP8Config.from_config(quant)
    assert (cfg.num_first_layers_in_bf16, cfg.num_last_layers_in_bf16) == (0, 1)


def test_training_keepouts_default_to_follow_rollout():
    """None, not 0, or the field could never mean "unset"."""
    t = TrainingQuantConfig()
    assert t.num_first_layers_in_bf16 is None
    assert t.num_last_layers_in_bf16 is None


# --- no dict literal in the worker may repeat a key --------------------------


def test_no_duplicate_keys_in_actor_worker_dict_literals():
    src = (_ROOT / "atomrl/workers/actor_worker.py").read_text()
    dupes = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Dict):
            continue
        seen = set()
        for k in node.keys:
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                continue
            if k.value in seen:
                dupes.append(f"{k.value!r} at line {k.lineno}")
            seen.add(k.value)
    assert not dupes, "later entry silently wins: " + ", ".join(dupes)


# --- the R3 flag has exactly one name in the engine config -------------------


def test_r3_has_a_single_engine_config_key():
    """``moe_enable_routing_replay`` is the survivor: the Megatron field name,
    declared in both config dataclasses, read by all three engines."""
    offenders = []
    for path in sorted(_ROOT.glob("atomrl/**/*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if 'ec.get("r3_enabled"' in line or '["r3_enabled"]' in line:
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}")
    assert not offenders, "read moe_enable_routing_replay instead: " + ", ".join(
        offenders
    )


def test_actor_worker_emits_the_canonical_r3_key():
    rel = "atomrl/workers/actor_worker.py"
    src = (_ROOT / rel).read_text()
    assert '"moe_enable_routing_replay"' in src
    assert '"r3_enabled":' not in src
