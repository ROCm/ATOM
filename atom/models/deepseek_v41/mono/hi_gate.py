# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether DeepSeek-V4.1's high-interactivity (hi) monokernels serve this deployment (``ATOM_DSV41_MONO_HI``).

- "auto" (the default): on for exactly the deployment they are built for, gfx950 (MI355X, every CU) at TP2, TP4 or
  TP8 (``SERVED_TPS``) without DP attention, a bf16 KV cache with the FP4 index cache (either gfx950 layout,
  ``INDEX_LAYOUTS``), ATOM's 128-token V4.1 pages, the DSpark draft with its 5 drafts, ATOM's Engram runtime on its
  UVA lookup with overlap staging, no KV transfer, a context up to ``MAX_CTX`` (the programs' context bucket),
  DeepSeek-V4.1-Flash's geometry and FlyDSL installed. Any other deployment keeps ATOM's own mono decode, unchanged,
  with one info line saying why.
- "1": the same, but a deployment they do not serve is an error (raised, naming why).
- "0": never.

Decided once, at ``install_mono_decode`` (after ATOM's own mono refusals, ``config.config_refusal``), before anything
of the hi path is imported or allocated: this module reads the configuration, the environment and the device's
properties only. Every configuration field it reads is the value ATOM's ``Config`` settled on (after its own
overrides: V4.1's KV page is 128 tokens whatever ``--block-size`` says), the one the hook and ATOM's cache use.
"""

from __future__ import annotations

import importlib.util
import logging

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

MODES = ("auto", "1", "0")
# the tensor-parallel sizes the hi programs are built for (MI355X, the validated configurations)
SERVED_TPS = (2, 4, 8)
REQ_ROWS = 6  # a verify's rows a request: DSpark's 5 drafts + the anchor
# the KV cache's page in tokens: ATOM's for V4.1 (``Config`` sets ``kv_cache_block_size`` to 128 for
# ``deepseek_v41_text`` whatever ``--block-size`` asks); the cache bridge reads the page geometry from ATOM's cache
# (``rows_per_page``, the page's rows), validated at this one
BLOCK_SIZE = 128
# the hi programs' context bucket in tokens (their compressed caches' and indexers' sizes): a --max-model-len past it
# is refused
MAX_CTX = 1 << 20
CUS = 256  # MI355X's CUs (one CTA a CU: a compute partition has fewer)
# ATOM's FP4 index plane layouts the hi programs read and write in place (``Config.index_cache_layout`` on gfx950:
# "mfma" with ``ATOM_V4_UNIFIED_MQA=opus``, the default, "preshuffle" with ``flydsl``; ``hi.device.gfx950.paged``)
INDEX_LAYOUTS = ("preshuffle", "mfma")
ARCH = "gfx950"
# DeepSeek-V4.1-Flash: the geometry the hi programs are composed for (``hi.v41.config.Geometry``, the verify head's
# vocabulary, the draft's taps and stages, the Engram's n-grams and heads)
GEOMETRY = {
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "n_routed_experts": 384,
    "hc_mult": 4,
    "vocab_size": 129280,
    "rms_norm_eps": 1e-20,
    "sliding_window": 128,
    "num_nextn_predict_layers": 3,
    "dspark_target_layer_ids": [37, 38, 39],
    "engram_layer_ids": [1, 14],
    "engram_max_ngram_size": 4,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
}


_LOGGED: set = set()  # the lines ``log_once`` logged: (level, the formatted line)


def log_once(level: int, msg: str, *args) -> None:
    """A line every rank would log alike (the deployment's, not a rank's), logged once a process: the first time it
    comes, on rank 0 (on every rank without ``torch.distributed``)."""
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    key = (level, msg % args if args else msg)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.log(level, msg, *args)


def mode() -> str:
    m = str(envs.ATOM_DSV41_MONO_HI).strip().lower()
    if m not in MODES:
        raise ValueError(
            f"ATOM_DSV41_MONO_HI={envs.ATOM_DSV41_MONO_HI!r}: one of {', '.join(MODES)}"
        )
    return m


def _device() -> tuple[str | None, int | None]:
    """(gfx arch, CUs) of the current device, (None, None) without one."""
    if not torch.cuda.is_available():
        return None, None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return p.gcnArchName.split(":")[0], int(p.multi_processor_count)


def refusal(atom_config, drafter, metadata_builder, dp_ranks: int) -> str:
    """Why the hi monokernels do not serve this deployment ("" when they do; module docstring)."""
    cfg = atom_config
    tp = int(getattr(cfg, "tensor_parallel_size", 1) or 1)
    if dp_ranks:
        return "DP attention"
    if tp not in SERVED_TPS:
        return f"TP{tp} (the hi path serves {' / '.join(f'TP{t}' for t in SERVED_TPS)})"
    arch, cus = _device()
    if arch is not None and arch != ARCH:
        return f"{arch} (the hi path serves {ARCH})"
    if cus is not None and cus != CUS:
        return f"{cus} CUs (the hi path serves a whole MI355X, {CUS})"
    kv = getattr(cfg, "kv_cache_dtype", None)
    if kv != "bf16":
        return f"--kv_cache_dtype {kv} (the hi path serves bf16)"
    index = getattr(cfg, "index_cache_dtype", None)
    if index != "fp4":
        return f"--index-cache-dtype {index} (the hi path serves fp4)"
    layout = getattr(cfg, "index_cache_layout", None)
    if layout not in INDEX_LAYOUTS:
        return f"FP4 index cache layout {layout} (the hi path addresses {', '.join(INDEX_LAYOUTS)})"
    block = getattr(cfg, "kv_cache_block_size", None)
    if block != BLOCK_SIZE:
        return f"KV cache pages of {block} tokens (the hi path serves ATOM's V4.1 page, {BLOCK_SIZE})"
    hf = getattr(cfg, "hf_config", None)
    geometry = {k: _plain(getattr(hf, k, None)) for k in GEOMETRY}
    if geometry != GEOMETRY:
        return f"model geometry {geometry} (the hi path serves DeepSeek-V4.1-Flash's)"
    spec = getattr(cfg, "speculative_config", None)
    if getattr(spec, "method", None) != "dspark" or drafter is None:
        return "no DSpark draft"
    if not hasattr(getattr(drafter, "model", None), "block_backbone"):
        return "no DSpark block draft"
    drafts = getattr(spec, "num_speculative_tokens", None)
    block_drafts = getattr(hf, "dspark_block_size", None)
    if (drafts, block_drafts) != (REQ_ROWS - 1, REQ_ROWS - 1):
        return (
            f"{drafts} speculative tokens, DSpark blocks of {block_drafts} "
            f"(the hi path serves {REQ_ROWS - 1})"
        )
    max_len = getattr(cfg, "max_model_len", None)
    if max_len is None or int(max_len) > MAX_CTX:
        return f"--max-model-len {max_len} > the hi programs' context bucket {MAX_CTX}"
    if getattr(cfg, "kv_transfer_config", None):
        return "KV transfer configured (P/D or an offload tier)"
    if getattr(cfg, "enable_tbo", False) or getattr(cfg, "enable_tbo_decode", False):
        return "two-batch overlap (the hi layers run a step whole)"
    if envs.ATOM_USE_TRITON_MOE:
        return "ATOM_USE_TRITON_MOE=1 (the hi path reads the routed experts in the gfx950 FlyDSL layout)"
    if metadata_builder is None:
        return "no metadata builder"
    engram = getattr(metadata_builder, "engram", None)
    if engram is None:
        return "no Engram runtime"
    host = getattr(engram, "host", None)
    if not getattr(host, "uva", False) or getattr(host, "overlap", None) is None:
        return "Engram lookup without UVA host shards and overlap staging"
    if importlib.util.find_spec("flydsl") is None:
        return "FlyDSL is not installed"
    return ""


def _plain(value):
    """A configuration value as ``GEOMETRY`` spells it (a tuple of ids as a list)."""
    return list(value) if isinstance(value, tuple) else value


def decide(atom_config, drafter, metadata_builder, dp_ranks: int) -> bool:
    """True when the hi monokernels serve this deployment; one info line when "auto" leaves it on ATOM's path. Raises
    under "1" when they cannot serve it."""
    m = mode()
    if m == "0":
        log_once(logging.INFO, "V4.1 mono hi off: ATOM_DSV41_MONO_HI=0")
        return False
    why = refusal(atom_config, drafter, metadata_builder, dp_ranks)
    if not why:
        return True
    if m == "1":
        raise RuntimeError(
            f"ATOM_DSV41_MONO_HI=1, but the hi path does not serve this deployment: {why}"
        )
    log_once(logging.INFO, "V4.1 mono hi off: %s; ATOM's mono path", why)
    return False
