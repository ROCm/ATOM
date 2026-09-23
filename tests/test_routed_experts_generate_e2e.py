# SPDX-License-Identifier: MIT
"""GPU e2e: native LLM.generate() routed-expert capture.

Covers tensor-parallel, CUDA-graph decode, and a prefix-cache hit. Dummy
Qwen3-MoE weights keep the run small; tokenizer comes from a real Qwen3
checkpoint so token ids stay in-vocab.

Usage (ATOM Docker image, 2+ GPUs)::

    ATOM_RUN_ROUTED_EXPERTS_E2E=1 pytest tests/test_routed_experts_generate_e2e.py -v --timeout=600
"""

from __future__ import annotations

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


def _assert_routes(routes, seq_len: int) -> None:
    assert routes is not None
    arr = np.asarray(routes)
    assert arr.dtype == np.int16
    assert arr.shape == (seq_len - 1, _NUM_LAYERS, _TOP_K)
    assert arr.min() >= 0
    assert arr.max() < _NUM_EXPERTS


def test_generate_tp_cudagraph_prefix_hit_routes(tmp_path):
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        pytest.skip(f"TP e2e needs 2 GPUs, found {n_gpu}")

    from transformers import AutoTokenizer

    from atom.model_engine.arg_utils import EngineArgs
    from atom.sampling_params import SamplingParams

    tok_path = _tokenizer_path()
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    # VocabParallelEmbedding requires vocab % tp == 0.
    vocab_size = ((vocab_size + 63) // 64) * 64
    model_dir = tmp_path / "tiny_qwen3_moe"
    model_dir.mkdir()
    _write_tiny_qwen3_moe(model_dir, vocab_size)

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
    engine = engine_args.create_engine(tokenizer=tokenizer)
    prompt = "Count from 1 to 40: " + " ".join(str(i) for i in range(1, 41))
    params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW, ignore_eos=True)
    try:
        first, cached_first = _generate_tracked(engine, [prompt], params)
        assert cached_first == 0
        out0 = first[0]
        n_in = out0["num_tokens_input"]
        n_out = out0["num_tokens_output"]
        assert n_out == _MAX_NEW
        seq_len = n_in + n_out
        _assert_routes(out0["routed_experts"], seq_len)

        second, cached_second = _generate_tracked(engine, [prompt], params)
        assert cached_second > 0, "second generate must prefix-hit the first prompt"
        out1 = second[0]
        assert out1["num_tokens_input"] == n_in
        assert out1["num_tokens_output"] == n_out
        _assert_routes(out1["routed_experts"], seq_len)
        np.testing.assert_array_equal(out0["routed_experts"], out1["routed_experts"])
        assert out0["token_ids"] == out1["token_ids"]
    finally:
        engine.close()
