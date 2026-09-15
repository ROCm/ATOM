import argparse
from types import SimpleNamespace

import torch

from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.persistent_decoder import (
    PERSISTENT_MAX_QUANTUM_TOKENS,
    PERSISTENT_PLANE_BLOCK_BYTES,
    decide_persistent_decode,
)


def _seq(**overrides):
    values = dict(
        id=7,
        temperature=0.0,
        return_logprobs=False,
        stop_strings=None,
        stop_token_sequences=[],
        kv_transfer_params=None,
        num_tokens=1025,
        max_tokens=1024,
        output_tokens=[],
        ignore_eos=False,
        last_token=42,
        block_table=list(range(33)),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _decision(seq=None, **overrides):
    args = dict(
        mode="required",
        seqs=[seq or _seq()],
        num_scheduled_tokens=[1],
        has_queued_work=False,
        use_spec=False,
        max_model_len=131072,
        eos_token_id=200002,
        extra_eos_token_ids=(),
    )
    args.update(overrides)
    return decide_persistent_decode(**args)


def test_eligible_batch_one_uses_eight_token_quantum():
    decision = _decision()
    assert decision.selected
    assert decision.plan.max_tokens == PERSISTENT_MAX_QUANTUM_TOKENS
    assert decision.plan.committed_kv_length == 1024
    assert decision.plan.pending_token == 42


def test_multiple_eos_ids_reduce_quantum_to_one():
    decision = _decision(extra_eos_token_ids=(200007,))
    assert decision.selected
    assert decision.plan.max_tokens == 1


def test_persistent_provider_rejection_rules_are_preserved():
    assert _decision(seq=_seq(temperature=0.1)).rejection_reason == "non_greedy"
    assert _decision(seq=_seq(return_logprobs=True)).rejection_reason == "logprobs"
    assert _decision(seq=_seq(stop_strings=["done"])).rejection_reason == "stop_strings"
    assert _decision(has_queued_work=True).rejection_reason == "queued_work"
    assert _decision(seqs=[_seq(id=1), _seq(id=2)]).rejection_reason == (
        "unsupported_decode_batch"
    )


def test_atom_native_bf16_cache_keeps_separate_shuffled_kv_planes():
    raw = torch.zeros(2, 1, 16, 8, 64, dtype=torch.bfloat16)
    assert raw[0].numel() * raw.element_size() == PERSISTENT_PLANE_BLOCK_BYTES

    # These are ATOM/AITER's existing views over the K/V-plane-major storage.
    k_cache = raw[0].view(1, 8, 8, 16, 8)
    v_cache = raw[1].view(1, 8, 2, 64, 8)
    k_cache[0, 2, 0, 3, 5] = 11
    v_cache[0, 4, 0, 9, 7] = 13

    assert raw[0].flatten()[2 * 2048 // 2 + 3 * 8 + 5].item() == 11
    assert raw[1].flatten()[4 * 2048 // 2 + 9 * 8 + 7].item() == 13

def test_persistent_cli_exposes_only_mode_and_checkpoint():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    persistent_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--persistent-decoder")
    }
    assert persistent_options == {
        "--persistent-decoder",
        "--persistent-decoder-checkpoint",
    }


def test_persistent_defaults_use_atom_bf16_cache():
    args = EngineArgs()
    assert args.kv_cache_dtype == "bf16"
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    kv_cache_action = next(
        action
        for action in parser._actions
        if "--kv_cache_dtype" in action.option_strings
    )
    assert tuple(kv_cache_action.choices) == ("bf16", "fp8")
    kwargs = args._get_engine_kwargs()
    assert kwargs["kv_cache_dtype"] == "bf16"
    assert "persistent_decoder_backend" not in kwargs
    assert "persistent_decoder_kv_dtype" not in kwargs
    assert "persistent_decoder_max_quantum" not in kwargs
