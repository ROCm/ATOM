"""Image-side smoke for RTP-LLM Qwen3.8-Flash-Next plugin wiring.

This does not start rtp-llm (the SGL ATOM image does not ship it). It checks
the ATOM-side contract: HF config parse, hybrid layer table, and prepare_model
text-only adaptations.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

CKPT = os.environ.get(
    "QWEN38_FLASH_CKPT",
    "/data/pretrained_model/Qwen/Qwen3.8-Flash-Next-PTPC-FP8",
)


def main() -> int:
    from atom.plugin.prepare import _set_framework_backbone
    from atom.plugin.register import _ATOM_SUPPORTED_MODELS

    _set_framework_backbone("rtpllm")

    arch = "Qwen4ExpForConditionalGeneration"
    if arch not in _ATOM_SUPPORTED_MODELS:
        raise SystemExit(f"{arch} missing from _ATOM_SUPPORTED_MODELS")
    print(f"supported: {arch} -> {_ATOM_SUPPORTED_MODELS[arch]}")

    raw = json.loads(open(os.path.join(CKPT, "config.json")).read())
    text = raw["text_config"]
    print(
        "checkpoint:",
        CKPT,
        "layers=",
        text["num_hidden_layers"],
        "experts=",
        text["num_experts"],
        "qsa_interval=",
        text["full_attention_interval"],
        "ple_layer_ids=",
        text["ple_layer_ids"],
        "hc_count=",
        text["hc_count"],
    )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    rtp_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            ckpt_path=CKPT,
            max_seq_len=4096,
            attn_config=SimpleNamespace(kv_cache_dtype="bf16"),
        ),
        parallelism_config=SimpleNamespace(tp_size=1, tp_rank=0, ep_size=1),
        max_generate_batch_size=4,
    )
    atom_cfg = generate_atom_config_for_plugin_mode(rtp_cfg)
    hf = atom_cfg.hf_config
    print(
        "atom_config:",
        "arch=",
        getattr(hf, "architectures", None),
        "ple=",
        getattr(hf, "ple_layer_ids", None),
        "hc=",
        getattr(hf, "hc_count", None),
        "kv=",
        atom_cfg.kv_cache_dtype,
        "mm=",
        atom_cfg.multimodal_config,
    )
    if list(getattr(hf, "architectures", []) or [])[:1] != [arch]:
        raise SystemExit(f"unexpected architectures: {hf.architectures}")
    if list(getattr(hf, "ple_layer_ids", []) or []) != [2]:
        raise SystemExit(f"unexpected ple_layer_ids: {hf.ple_layer_ids}")

    from atom.plugin.prepare import _prepare_model_atom_rtpllm

    class _Sentinel:
        packed_modules_mapping = {}
        quant_exclude_name_mapping = {}

        def __init__(self, atom_config=None, config=None):
            self.atom_config = atom_config or config

    called = {}

    def _set_attn():
        called["set_attn"] = True

    def _init_dist(config):
        called["init_dist"] = config is atom_cfg

    model = _prepare_model_atom_rtpllm(
        rtp_cfg,
        atom_cfg,
        arch,
        _Sentinel,
        _set_attn,
        _init_dist,
    )
    if atom_cfg.multimodal_config is not None:
        raise SystemExit("vision tower was not disabled for RTP text path")
    if atom_cfg.kv_cache_dtype not in {"auto", "bf16", "bfloat16"}:
        raise SystemExit(f"kv_cache_dtype={atom_cfg.kv_cache_dtype}")
    if not called.get("set_attn") or not called.get("init_dist"):
        raise SystemExit(f"prepare hooks not run: {called}")
    print("prepare_model rtpllm path: text-only, gdn-only, kv=bf16")
    print("smoke ok", type(model).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
