# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 under vLLM's V1 runner: a still-prefilling row never takes the
FULL decode graph, and the builder refuses a FULL step it did not build as
DECODE.

V1's ``_is_uniform_decode`` looks at the shape only. A step whose every row
has ``uniform_decode_query_len`` tokens but where some row is still in its
prompt (the 1-token tail after a prefix hit, a 1-token second chunk, a fresh
1-token prompt; 2 tokens with MTP k=1) dispatched the FULL decode graph, while
the V4 bridge built it as PREFILL and left the persistent decode buffers the
graph reads untouched.
"""

import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from atom.plugin.vllm import deepseek_v4_cudagraph_patch as cg

FULL = NS(name="FULL")
PIECEWISE = NS(name="PIECEWISE")
ATOM_WRAPPER = NS(module_name="atom.plugin.vllm.model_wrapper", class_name="X")


class _Runner:
    """The parts of V1 GPUModelRunner the patch touches."""

    uniform_decode_query_len = 1

    def __init__(self, arch, computed, prompt, query_len=None):
        self.model_config = NS(architectures=[arch])
        self.input_batch = NS(
            num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
            num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
        )
        if query_len is not None:
            self.uniform_decode_query_len = query_len
        self.calls = []
        self.armed_during_build = "unset"

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens,
        uniform_decode_query_len,
        num_tokens,
        num_reqs,
        force_uniform_decode=None,
    ):
        if force_uniform_decode is not None:
            return force_uniform_decode
        return (
            max_num_scheduled_tokens == uniform_decode_query_len
            and num_tokens == max_num_scheduled_tokens * num_reqs
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens,
        num_reqs,
        num_scheduled_tokens_np,
        max_num_scheduled_tokens,
        use_cascade_attn,
        allow_microbatching=True,
        force_eager=False,
        force_uniform_decode=None,
        **kwargs,
    ):
        self.calls.append(force_uniform_decode)
        uniform = self._is_uniform_decode(
            max_num_scheduled_tokens,
            self.uniform_decode_query_len,
            num_tokens,
            num_reqs,
            force_uniform_decode,
        )
        return (
            (FULL if uniform else PIECEWISE),
            NS(num_tokens=num_tokens),
            False,
            None,
            None,
        )

    def _build_attention_metadata(self, *args, **kwargs):
        self.armed_during_build = cg.current_v4_dispatch()
        return "md"


@pytest.fixture
def patched(monkeypatch, tmp_path):
    class Runner(_Runner):
        pass

    registry = NS(
        models={
            "DeepseekV4ForCausalLM": ATOM_WRAPPER,
            "KimiK3ForConditionalGeneration": ATOM_WRAPPER,
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        NS(GPUModelRunner=Runner),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.registry", NS(ModelRegistry=registry)
    )
    monkeypatch.delenv("ATOM_DISABLE_VLLM_PLUGIN", raising=False)
    monkeypatch.setattr(cg, "_INSTR_DIR", str(tmp_path))
    monkeypatch.setattr(cg, "_INSTR", cg._new_counters())
    assert cg.apply_vllm_v4_prefill_not_uniform_patch()
    return Runner


def _step(runner, lens):
    lens = np.asarray(lens, dtype=np.int32)
    out = runner._determine_batch_execution_and_padding(
        num_tokens=int(lens.sum()),
        num_reqs=len(lens),
        num_scheduled_tokens_np=lens,
        max_num_scheduled_tokens=int(lens.max()),
        use_cascade_attn=False,
    )
    runner._build_attention_metadata()
    return out[0]


def test_one_token_tail_after_a_prefix_hit_is_not_uniform(patched):
    # prompt 1025, 1024 hit: one token left, still prefilling
    runner = patched("DeepseekV4ForCausalLM", computed=[1024], prompt=[1025])
    assert _step(runner, [1]) is PIECEWISE
    assert runner.calls == [False]
    assert runner.armed_during_build == (PIECEWISE, True)
    assert cg._INSTR["flipped"] == 1


def test_decodes_mixed_with_a_one_token_tail_are_not_uniform(patched):
    runner = patched(
        "DeepseekV4ForCausalLM", computed=[3000, 40, 8192], prompt=[2000, 30, 8193]
    )
    assert _step(runner, [1, 1, 1]) is PIECEWISE
    assert runner.calls == [False]


def test_pure_decode_is_untouched(patched):
    runner = patched("DeepseekV4ForCausalLM", computed=[3000, 40], prompt=[2000, 30])
    assert _step(runner, [1, 1]) is FULL
    assert runner.calls == [None]
    assert runner.armed_during_build == (FULL, False)
    assert cg._INSTR["flipped"] == 0
    assert cg._INSTR["modes"] == {"FULL": 1}


def test_first_decode_after_the_prompt_is_a_decode(patched):
    # num_computed == num_prompt: the prompt is done, the row is decoding
    runner = patched("DeepseekV4ForCausalLM", computed=[1025], prompt=[1025])
    assert _step(runner, [1]) is FULL
    assert runner.calls == [None]


def test_mtp_two_token_tail_is_not_uniform(patched):
    runner = patched(
        "DeepseekV4ForCausalLM", computed=[3000, 896], prompt=[2000, 898], query_len=2
    )
    assert _step(runner, [2, 2]) is PIECEWISE
    assert runner.calls == [False]


def test_mtp_verify_is_untouched(patched):
    runner = patched(
        "DeepseekV4ForCausalLM", computed=[3000, 900], prompt=[2000, 898], query_len=2
    )
    assert _step(runner, [2, 2]) is FULL
    assert runner.calls == [None]


def test_non_v4_runner_is_untouched(patched):
    runner = patched("KimiK3ForConditionalGeneration", computed=[1024], prompt=[1025])
    assert _step(runner, [1]) is FULL
    assert runner.calls == [None]
    assert runner.armed_during_build is None


def test_padded_rows_beyond_num_reqs_are_ignored(patched):
    # rows past num_reqs hold stale data from condense()
    runner = patched("DeepseekV4ForCausalLM", computed=[3000, 0], prompt=[2000, 9])
    lens = np.asarray([1], dtype=np.int32)
    mode = runner._determine_batch_execution_and_padding(
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=lens,
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )[0]
    assert mode is FULL
    assert runner.calls == [None]


def test_capture_and_dummy_runs_are_untouched(patched):
    runner = patched("DeepseekV4ForCausalLM", computed=[1024], prompt=[1025])
    lens = np.asarray([1], dtype=np.int32)
    mode = runner._determine_batch_execution_and_padding(
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=lens,
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
        force_uniform_decode=True,
    )[0]
    assert mode is FULL
    assert runner.calls == [True]
    runner._build_attention_metadata()
    assert runner.armed_during_build is None
    assert cg._INSTR["steps"] == 0


def test_dispatch_is_armed_only_inside_the_build(patched):
    runner = patched("DeepseekV4ForCausalLM", computed=[3000], prompt=[2000])
    _step(runner, [1])
    assert cg.current_v4_dispatch() is None
    # a second build in the same step (e.g. a drafter) is not armed
    runner._build_attention_metadata()
    assert runner.armed_during_build is None


def test_apply_is_idempotent(patched):
    d = patched._determine_batch_execution_and_padding
    b = patched._build_attention_metadata
    assert not cg.apply_vllm_v4_prefill_not_uniform_patch()
    assert patched._determine_batch_execution_and_padding is d
    assert patched._build_attention_metadata is b


@pytest.fixture
def attn_state():
    fc = pytest.importorskip("atom.utils.forward_context")
    return fc.AttnState


def _armed(mode, flipped=False):
    return cg._armed_dispatch(mode, flipped)


def test_guard_raises_on_full_with_a_prefill_build(attn_state, monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "_INSTR_DIR", str(tmp_path))
    monkeypatch.setattr(cg, "_INSTR", cg._new_counters())
    md = NS(
        num_reqs=1, max_query_len=1, num_actual_tokens=1, is_prefilling=np.array([True])
    )
    with _armed(FULL), pytest.raises(RuntimeError, match="FULL.*PREFILL"):
        cg.check_v4_dispatch_consistency(attn_state.PREFILL_PREFIX, md)
    assert cg._INSTR["guard_trips"] == 1


@pytest.mark.parametrize(
    "mode,state",
    [("FULL", "DECODE"), ("PIECEWISE", "PREFILL_PREFIX"), ("PIECEWISE", "DECODE")],
)
def test_guard_is_quiet_when_consistent(attn_state, mode, state):
    md = NS(num_reqs=1, max_query_len=1, num_actual_tokens=1, is_prefilling=None)
    with _armed(NS(name=mode)):
        cg.check_v4_dispatch_consistency(getattr(attn_state, state), md)


def test_guard_is_quiet_when_unarmed(attn_state):
    md = NS(num_reqs=1, max_query_len=1, num_actual_tokens=1, is_prefilling=None)
    cg.check_v4_dispatch_consistency(attn_state.PREFILL_PREFIX, md)


def test_counters_are_written(patched, tmp_path):
    runner = patched("DeepseekV4ForCausalLM", computed=[1024], prompt=[1025])
    _step(runner, [1])
    files = list(tmp_path.glob("cg_*.json"))
    assert len(files) == 1
    import json

    data = json.loads(files[0].read_text())
    assert data["flipped"] == 1 and data["steps"] == 1
