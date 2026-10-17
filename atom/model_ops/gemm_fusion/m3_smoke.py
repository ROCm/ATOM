# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weights-free MiniMax-M3 smoke entry for gemm_fusion compile/lifecycle evidence.

STATUS: written but NEVER RUN (no torch/aiter/GPU on the authoring host). The
config below lists every HF field ``atom/models/minimax_m3/model.py`` reads
(grep of ``config.<x>``); it is a starting point, not a verified-loadable model.
Missing tokenizer/quantization/MoE fields will surface as errors on first run.

What it gives Evidence (E2 matrix, E3/E4/E5/E1/E7 repro):
  * ``make_model_dir(path, layers=...)``  writes config.json (+ a tokenizer stub is NOT
    provided: pass a real tokenizer dir via ``--tokenizer`` or copy one in).
  * ``serve_args(...)`` returns the ATOM CLI argv for ``--load_dummy`` (random weights,
    no checkpoint), ``--level 0|1|2|3``, optional ``--enforce-eager`` and
    ``--gemm-fusion-profile``. ON and OFF differ ONLY by that flag.
  * ``python -m atom.model_ops.gemm_fusion.m3_smoke --dir D --level 3 [--profile P]``
    prints the argv and the env for the fault seams (it does not launch).
SP4/gfx950 gating: the fused site needs q width 8192 (64 heads x 128), SP world 4,
tp 1, bf16, per-token-FP8 o_proj, gfx950; a smaller model silently takes the OLD
path (declare returns None), which would make an ON run vacuous. So the config keeps
the real attention geometry and shrinks only layer count / experts / vocab.
Check ``gemm_fusion: installed fused QKV+attention on N layers`` (N>0) in the log;
N == 0 means the run did not exercise the fused path.
Layer pattern (``sparse_attention_freq``) covers the three layer types the site
declares: dense, sparse indexed, sparse skip-index (``index_topk_freq`` > 1).
"""

import argparse
import json
import os


def m3_config(layers=4, experts=8, vocab=32000, index_topk_freq=2):
    """HF-style config dict (no weights). Geometry matches the golden cases."""
    return {
        "model_type": "minimax_m3",
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000.0,
        "partial_rotary_factor": 0.5,
        "hidden_act": "swigluoai",
        "intermediate_size": 1536,
        "dense_intermediate_size": 6144,
        "num_local_experts": experts,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        # layer 0 dense MoE-free; layers alternate dense attn / sparse attn
        "moe_layer_freq": [0] + [1] * (layers - 1),
        "index_topk_freq": index_topk_freq,
        "index_skip_topk_offset": 0,
        "sparse_attention_config": {
            "sparse_block_size": 128,  # must equal SPARSE_BLOCK_SIZE (checked at init)
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_topk_blocks": 8,
            "sparse_init_block": 1,
            "sparse_local_block": 2,
            "sparse_score_type": "max",
            # 0 = dense attention layer, nonzero = sparse (first layer dense)
            "sparse_attention_freq": [0] + [1] * (layers - 1),
        },
        "torch_dtype": "bfloat16",
    }


def register_hf_config(model_type="minimax_m3"):
    """Make ``AutoConfig.from_pretrained`` accept this model_type WITHOUT a transformers
    upgrade or trust_remote_code. Call it in the process that builds ``Config`` (the one
    that calls ``atom.config.get_hf_config``) BEFORE ``create_engine``.

    Why it is needed: ATOM's native path registers no config class for minimax_m3 /
    minimax_m3_vl (only the vLLM plugin does, via the same kind of shim), so
    ``get_hf_config`` falls through to ``AutoConfig.from_pretrained`` and raises for a
    model_type the installed transformers does not know. The shim is a bare
    ``PretrainedConfig``: it keeps every field of the (flat, text-only) config.json as an
    attribute, and ``architectures`` still selects ``MiniMaxM3SparseForCausalLM``.

    UNVERIFIED against real transformers 5.16.1 (not installed on the authoring host).
    Idempotent: a model_type transformers already knows is left alone.
    """
    from transformers import AutoConfig, PretrainedConfig

    cls = type("MiniMaxM3SmokeConfig", (PretrainedConfig,), {"model_type": model_type})
    try:
        AutoConfig.register(model_type, cls)
    except ValueError as exc:  # already registered (by transformers or an earlier call)
        if "already" not in str(exc):
            raise
    return cls


VL_MODEL_TYPE = "minimax_m3_vl"  # outer type of the released checkpoint (repo tests + vLLM shim)
TEXT_MODEL_TYPE = "minimax_m3_text"  # nested text_config type (repo tests)
VL_ARCH = "MiniMaxM3SparseForConditionalGeneration"  # what the recipes' VL checkpoint selects


def m3_vl_config(layers=4, experts=8, vocab=32000, index_topk_freq=2):
    """The released checkpoint's LAYOUT: an outer ``minimax_m3_vl`` config whose text model
    is nested under ``text_config`` (``minimax_m3_text``). ATOM reads the text fields from
    ``text_config`` (``_get_text_config``) and copies the index-cache overrides across
    (``_normalize_minimax_m3_text_config``). ``auto_map`` points at a config class that
    ships NEXT TO config.json, which is how ATOM's own recipes load M3 (``--trust-remote-
    code``); ATOM's native path registers no class for it. Attention geometry is unchanged.
    """
    text = m3_config(layers=layers, experts=experts, vocab=vocab, index_topk_freq=index_topk_freq)
    for k in ("architectures", "torch_dtype"):
        text.pop(k, None)
    text["model_type"] = TEXT_MODEL_TYPE
    # The index-cache controls live on the OUTER config (the recipes set them with
    # --hf-overrides, which updates the outer config) and ATOM copies them DOWN into
    # text_config (_normalize_minimax_m3_text_config). Keep them out of text_config here so
    # the run exercises that same copy, which is what decides skip-index layers.
    outer_index = {k: text.pop(k) for k in ("index_topk_freq", "index_skip_topk_offset") if k in text}
    return {
        "model_type": VL_MODEL_TYPE,
        "architectures": [VL_ARCH],
        "auto_map": {"AutoConfig": "configuration_minimax_m3.MiniMaxM3Config"},
        **outer_index,
        "text_config": text,
        # vision tower is never built on the text-only native path (model.py skips
        # vision_tower./multi_modal_projector./patch_merge_mlp. weights); kept minimal
        "vision_config": {"model_type": "minimax_m3_vision"},
        "torch_dtype": "bfloat16",
    }


# Shipped next to config.json. A bare PretrainedConfig keeps every field as an attribute
# (nested dicts become sub-configs), mirroring atom/plugin/vllm/register.py MiniMaxM3Config.
CONFIGURATION_PY = '''\
from transformers import PretrainedConfig


class MiniMaxM3TextConfig(PretrainedConfig):
    model_type = "minimax_m3_text"


class MiniMaxM3Config(PretrainedConfig):
    model_type = "minimax_m3_vl"
    sub_configs = {"text_config": MiniMaxM3TextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = MiniMaxM3TextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.hidden_size = getattr(text_config, "hidden_size", None)
        super().__init__(**kwargs)
'''


def make_model_dir(path, *, layout="vl", **kw):
    """Write config.json (+ configuration_minimax_m3.py for the ``vl`` layout).

    layout="vl"   (default) the released checkpoint's layout; load with --trust-remote-code.
    layout="flat" the old single-level ``minimax_m3`` config; needs ``register_hf_config()``.
    """
    os.makedirs(path, exist_ok=True)
    cfg = m3_vl_config(**kw) if layout == "vl" else m3_config(**kw)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if layout == "vl":
        with open(os.path.join(path, "configuration_minimax_m3.py"), "w", encoding="utf-8") as f:
            f.write(CONFIGURATION_PY)
    return path


SPARSE_BLOCK_SIZE = 128  # atom.model_ops.minimax_m3.sparse_attn.SPARSE_BLOCK_SIZE


def serve_args(model_dir, *, level=3, profile="", enforce_eager=False, sp=4, extra=(),
               trust_remote_code=True, block_size=SPARSE_BLOCK_SIZE):
    """ATOM CLI argv. ``--block-size`` must be a multiple of the sparse block (128):
    the attention metadata builder asserts ``model_runner.block_size % self.block_size == 0``
    (attentions/backends.py:511) and ATOM's CLI default of 16 fails it for M3's sparse layers
    (found by the E2 OFF arm, job-e2_off01). The same value also fixes the KV page size the
    dense layers' fusion signature derives from ``config.kv_cache_block_size``."""
    if block_size % SPARSE_BLOCK_SIZE:
        raise ValueError(f"block_size must be a multiple of {SPARSE_BLOCK_SIZE}, got {block_size}")
    argv = [
        # xavier: finite, roughly real-scale random weights (arg_utils --load_dummy)
        "--model", model_dir, "--load_dummy", "xavier",
        "-tp", "1", "-sp", str(sp), "--kv-cache-dtype", "fp8",
        "--level", str(level), "--block-size", str(block_size),
    ]
    if trust_remote_code:  # the vl layout resolves its config class through auto_map
        argv.append("--trust-remote-code")
    if enforce_eager:
        argv.append("--enforce-eager")
    if profile:
        argv += ["--gemm-fusion-profile", profile]
    return argv + list(extra)


FAULT_ENV = "ATOM_GEMM_FUSION_FAULT"  # see faults.py for kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--profile", default="")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()
    make_model_dir(a.dir, layers=a.layers)
    print("argv:", " ".join(serve_args(a.dir, level=a.level, profile=a.profile,
                                       enforce_eager=a.enforce_eager)))
    print("fault env:", FAULT_ENV, "(see atom/model_ops/gemm_fusion/faults.py)")


if __name__ == "__main__":
    main()
