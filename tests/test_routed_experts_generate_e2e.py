# SPDX-License-Identifier: MIT
"""GPU e2e: native LLM.generate() routed-expert capture.

Covers tensor-parallel, CUDA-graph decode, a prefix-cache hit, concurrent
requests in one batch, chunked prefill, and MTP verify on a GDN model. Dummy
weights keep the run small; tokenizer comes from a real Qwen3 checkpoint so token ids stay
in-vocab.

Usage (ATOM Docker image, 2+ GPUs)::

    ATOM_RUN_ROUTED_EXPERTS_E2E=1 pytest tests/test_routed_experts_generate_e2e.py -v --timeout=600
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HAS_GPU = torch.cuda.is_available() or (
    hasattr(torch, "hip") and torch.hip.is_available()
)
_OPT_IN = os.environ.get("ATOM_RUN_ROUTED_EXPERTS_E2E") == "1"
_NUM_LAYERS = 2
_NUM_EXPERTS = 8
_TOP_K = 2
_MAX_NEW = 8

pytestmark = pytest.mark.skipif(
    not (_HAS_GPU and _OPT_IN),
    reason="GPU generate e2e; set ATOM_RUN_ROUTED_EXPERTS_E2E=1 (needs 2 GPUs)",
)


def _tokenizer_path() -> str:
    candidates = [
        os.environ.get("ATOM_TEST_TOKENIZER", ""),
        "/models/Qwen3-8B-Base",
        "/data/rl_data/models/Qwen3-8B-Base",
        "/data/rl_data/models/Qwen3-30B-A3B-Base",
    ]
    for path in candidates:
        if path and (Path(path) / "tokenizer.json").is_file():
            return path
    pytest.skip("no Qwen3 tokenizer.json found (set ATOM_TEST_TOKENIZER)")


def _write_tiny_qwen3_moe(model_dir: Path, vocab_size: int) -> None:
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "head_dim": 32,
        "max_position_embeddings": 2048,
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 128,
        "num_attention_heads": 8,
        "num_experts": _NUM_EXPERTS,
        "num_experts_per_tok": _TOP_K,
        "num_hidden_layers": _NUM_LAYERS,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))


_NEXT_NUM_LAYERS = 4


def _write_tiny_qwen3_next(model_dir: Path, vocab_size: int) -> None:
    """Three GDN layers then one full-attention layer, every layer MoE, plus
    the one MTP layer the drafter reuses."""
    config = {
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "hidden_act": "silu",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "partial_rotary_factor": 0.25,
        "attention_bias": False,
        "num_hidden_layers": _NEXT_NUM_LAYERS,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "num_experts": _NUM_EXPERTS,
        "num_experts_per_tok": _TOP_K,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "num_nextn_predict_layers": 1,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))


def _generate_tracked(engine, prompts, sampling_params):
    """Like LLMEngine.generate, plus the largest prefix-cache hit seen."""
    engine.core_mgr.reset_dp_router()
    engine.add_request(prompts, sampling_params)
    outputs = {}
    max_cached = 0
    while not engine.is_finished() and (
        engine.core_mgr.is_alive() or engine.core_mgr.is_rest()
    ):
        seqs = engine.step()
        for seq in seqs:
            max_cached = max(
                max_cached,
                int(getattr(seq, "num_cached_tokens", 0) or 0),
                int(getattr(seq, "prefix_cache_hit_tokens", 0) or 0),
            )
        outputs.update(engine.io_processor.postprocess(seqs))
    ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
    return ordered, max_cached


def _assert_routes(routes, seq_len: int, num_layers: int = _NUM_LAYERS) -> None:
    assert routes is not None
    arr = np.asarray(routes)
    assert arr.dtype == np.int16
    assert arr.shape == (seq_len - 1, num_layers, _TOP_K)
    assert arr.min() >= 0
    assert arr.max() < _NUM_EXPERTS


_PROMPT = "Count from 1 to 40: " + " ".join(str(i) for i in range(1, 41))


def _tp_engine(tmp_path, write_model=_write_tiny_qwen3_moe, **overrides):
    """A TP=2, FULL-cudagraph engine on a tiny dummy model."""
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        pytest.skip(f"TP e2e needs 2 GPUs, found {n_gpu}")

    from transformers import AutoTokenizer

    from atom.model_engine.arg_utils import EngineArgs

    tok_path = _tokenizer_path()
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    # VocabParallelEmbedding requires vocab % tp == 0.
    vocab_size = ((vocab_size + 63) // 64) * 64
    model_dir = tmp_path / "tiny_model"
    model_dir.mkdir()
    write_model(model_dir, vocab_size)

    engine_args = EngineArgs(
        model=str(model_dir),
        tensor_parallel_size=2,
        load_dummy="xavier",
        enable_return_routed_experts=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        cudagraph_mode="FULL",
        cudagraph_capture_sizes="[1, 2]",
        level=0,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.08,
    )
    engine_args = dataclasses.replace(engine_args, **overrides)
    return engine_args.create_engine(tokenizer=tokenizer)


def _greedy():
    from atom.sampling_params import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=_MAX_NEW, ignore_eos=True)


def _assert_output_routes(out, num_layers: int = _NUM_LAYERS) -> None:
    assert out["num_tokens_output"] == _MAX_NEW
    _assert_routes(
        out["routed_experts"],
        out["num_tokens_input"] + out["num_tokens_output"],
        num_layers,
    )


def test_generate_tp_cudagraph_prefix_hit_routes(tmp_path):
    engine = _tp_engine(tmp_path)
    params = _greedy()
    try:
        first, cached_first = _generate_tracked(engine, [_PROMPT], params)
        assert cached_first == 0
        out0 = first[0]
        _assert_output_routes(out0)

        second, cached_second = _generate_tracked(engine, [_PROMPT], params)
        assert cached_second > 0, "second generate must prefix-hit the first prompt"
        out1 = second[0]
        assert out1["num_tokens_input"] == out0["num_tokens_input"]
        _assert_output_routes(out1)
        np.testing.assert_array_equal(out0["routed_experts"], out1["routed_experts"])
        assert out0["token_ids"] == out1["token_ids"]
    finally:
        engine.close()


def test_concurrent_requests_keep_their_own_routes(tmp_path):
    """[A, B, A] in one batch: each request gets rows for its own length, and
    the two copies of A route identically, so no request reads another's
    slots."""
    engine = _tp_engine(tmp_path)
    other = "Name the planets in order from the sun:"
    try:
        outs, _ = _generate_tracked(engine, [_PROMPT, other, _PROMPT], _greedy())
        assert len(outs) == 3
        for out in outs:
            _assert_output_routes(out)
        a0, b, a1 = outs
        assert b["num_tokens_input"] != a0["num_tokens_input"]
        np.testing.assert_array_equal(a0["routed_experts"], a1["routed_experts"])
        assert a0["token_ids"] == a1["token_ids"]
    finally:
        engine.close()


def test_chunked_prefill_then_prefix_hit_routes(tmp_path):
    """A prompt prefilled in several chunks, then re-run on a prefix hit whose
    cached rows come from the KV blocks: both runs export the same routes."""
    budget = 32
    engine = _tp_engine(tmp_path, max_num_batched_tokens=budget)
    params = _greedy()
    try:
        first, cached_first = _generate_tracked(engine, [_PROMPT], params)
        assert cached_first == 0
        out0 = first[0]
        assert out0["num_tokens_input"] > 2 * budget, "prompt must span chunks"
        _assert_output_routes(out0)

        second, cached_second = _generate_tracked(engine, [_PROMPT], params)
        assert cached_second > 0, "second generate must prefix-hit the first prompt"
        out1 = second[0]
        _assert_output_routes(out1)
        np.testing.assert_array_equal(out0["routed_experts"], out1["routed_experts"])
        assert out0["token_ids"] == out1["token_ids"]
    finally:
        engine.close()


def _route_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of (token, layer) cells whose top-k expert sets differ."""
    return float(np.mean(np.any(np.sort(a, -1) != np.sort(b, -1), axis=-1)))


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_mtp_verify_routes_align_with_a_prefill_of_the_same_tokens(
    tmp_path, num_speculative_tokens
):
    """GDN + MTP under FULL graphs. Dummy weights make the drafter mostly
    wrong, so most verify steps reject drafts. Re-prefilling prompt + output
    computes every token's route directly; the verify-built routes must line
    up with it row for row, where a misplaced verify start would shift them."""
    max_new = 32
    engine = _tp_engine(
        tmp_path,
        write_model=_write_tiny_qwen3_next,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
        enable_prefix_caching=False,
    )
    from atom.sampling_params import SamplingParams

    try:
        prompt_ids = list(engine.tokenizer.encode(_PROMPT))
        n_in = len(prompt_ids)
        params = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=True)
        (out,), _ = _generate_tracked(engine, [prompt_ids], params)
        assert out["num_tokens_output"] == max_new
        verify = np.asarray(out["routed_experts"])
        _assert_routes(verify, n_in + max_new, _NEXT_NUM_LAYERS)

        replay = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        (ref_out,), _ = _generate_tracked(
            engine, [prompt_ids + list(out["token_ids"])], replay
        )
        ref = np.asarray(ref_out["routed_experts"])[: len(verify)]

        decode = slice(n_in, len(verify))
        aligned = _route_disagreement(verify[decode], ref[decode])
        shifted = min(
            _route_disagreement(verify[n_in + 1 :], ref[n_in:-1]),
            _route_disagreement(verify[n_in:-1], ref[n_in + 1 :]),
        )
        assert _route_disagreement(verify[:n_in], ref[:n_in]) <= 0.05
        assert aligned <= 0.1
        assert shifted >= 0.4, "routes must be position-specific for this check"
    finally:
        engine.close()
