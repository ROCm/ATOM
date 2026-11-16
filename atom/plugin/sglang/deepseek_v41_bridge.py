# SPDX-License-Identifier: MIT
"""Run DeepSeek-V4.1 (CSA2) on SGLang's plugin path against an ATOM-owned pool.

SGLang 0.5.22 knows V4.1 natively: it normalizes the checkpoint config, rewrites
``architectures`` to ``DeepseekV4ForCausalLM`` (``model_type`` stays
``deepseek_v41``), forces ``page_size=256`` and the ``dsv4`` attention backend,
and builds a ``DeepSeekV4TokenToKVPool``. Everything here keys on
``model_type``, never on the architecture name, so V4.1 cannot fall into the
V4 bridge.

Ownership (Design A):

- SGLang owns scheduling, ``req_to_token`` / page allocation and sampling.
- ATOM owns the model, its weights and the CSA2 pool layout. The pool is one
  uint8 arena, PAGE region then STATE region, because
  ``V41PoolGeometry.window`` addresses a slot's ring past the absolute end of
  the paged extent. ``ATOMDeepSeekV41ProxyKVPool`` replaces SGLang's DSV4 pool,
  allocates that arena and hands it to ``PagedAttentionCache`` through the
  ``_allocate_backing`` seam.
- SGLang sizes that pool by what the arena costs
  (``ATOMDeepSeekV41PoolConfigurator``), not by its own DSV4 layout.
- The plugin translates each ``ForwardBatch`` into ATOM's scheduled-batch
  protocol and runs ATOM's own ``DeepseekV41MetadataBuilder`` on it, outside
  the forward, exactly once per step.
- Decode CUDA graphs (one per batch-size bucket) are staged by the backend
  shim outside the captured region, on reserved capture STATE slots at
  capture and into the same fixed addresses before every replay; Engram runs
  inside the graph as natively (see "Decode CUDA graphs" below).

Identities the translation relies on:

- SGLang page id == V4.1 PAGE id (both 256 tokens), so block tables are
  ``req_to_token[req, ::256] // 256``. Page 0 is SGLang's padding page.
- STATE slot == ``req_pool_idx``: stable for a request's lifetime and unique
  among live requests (row 0 of ``req_to_token`` is padding, never handed
  out). A request starting at position 0 resets its slot
  (``PagedAttentionCache.prepare_state``). With radix cache on, a prefix hit
  at ``P`` rebuilds its slot from ``P - R`` (``deepseek_v41_radix``): in the
  step's own forward, its span widened to start there (fused), or in a side
  forward before it; either is translated here into ATOM's
  ``ScheduledBatch.replaying``.
- V4.1's sliding window lives in the STATE ring, so the proxy asks SGLang for
  a plain paged allocator (``needs_paged_swa_allocator = False``): no SWA
  token bookkeeping, no SWA tree cache.

DSpark (SGLang ``DSparkWorkerV2``, static verify, with or without the overlap
schedule):

- TARGET_VERIFY is a tentative ATOM step of ``spec_info.draft_token_num`` (6)
  rows a request at positions ``p..p+5``: the builder stages every prefix the
  sampler may accept (``cache.pending``), and the accepted one is committed
  once per round from SGLang's own ``commit_lens``
  (``commit_deepseek_v41_dspark_verify``), never from stored metadata.
- The draft's context KV (window rings 40-42 of the same STATE slots) is
  written by the target forward that produced the rows, under that forward's
  own step, exactly as native ATOM's ``compute_draft_kv`` does. The draft
  runner's proxy pool owns no bytes: it is a role view of the target arena.
- With radix cache on, a prefix hit's EXTEND replays ``[P - R, P)`` like any
  other (``deepseek_v41_radix``), and that forward writes the draft windows'
  rows below ``P`` too; verify rounds never carry a replay.
- There is no DRAFT_EXTEND and no draft DECODE: the draft block runs as a
  TARGET_VERIFY of width 5 on the draft runner.
- With ``--cuda-graph-backend-decode full`` both runners capture TARGET_VERIFY
  graphs (see "DSpark graph stages" below): the draft block (x5) reads only
  SGLang's fixed buffers inside its forward; the target verify (x6) is a
  tentative step staged outside the graph per replay, its accepted prefix
  committed eagerly after it. Prefill stays eager.

DSpark under the overlap schedule (SGLang 0.5.22 ``event_loop_overlap``):

- Overlap does not split a round. ``DSparkWorkerV2._forward_decode`` runs the
  whole round -- draft block, verify, accept, ``on_publish``, the commit hook,
  the hidden commit -- in one ``forward_batch_generation`` call on the
  scheduler thread under ``forward_stream``. What overlaps the GPU is the
  scheduler's own work: processing round t-1's result and forming round
  t+1's batch. Round t+1's verify is staged only after round t's publish
  (upstream ``FutureMap.resolve_seq_lens_cpu`` waits for it at the top of
  ``run_batch``; with `dspark_deferred_seq_lens_patch` the device gather is
  stream-ordered after it, the draft block is launched, and the host waits
  for the published lengths in ``run_non_compact``), so a verify is never
  staged over an uncommitted one (``require_committed`` and ``begin_step``
  refuse it if it ever were).
- The tentative step's publication epoch is opened by the verify's staging
  and resumed by its commit on the same thread and stream
  (``PublicationOwner.resume`` checks both); the next ``begin`` waits on the
  commit's completion event before the pinned sources are rewritten.
- Host lengths: the ScheduleBatch's ``seq_lens_cpu`` is seeded one round late
  by ``prepare_for_decode`` and overwritten with the published exact lengths
  (this backend's ``needs_cpu_seq_lens``) before the verify, so
  ``live_seq_lens_cpu`` is the committed prefix as without overlap. Deferred,
  the draft runs with ``seq_lens_cpu`` None as for a ``needs_cpu_seq_lens=False``
  backend (host planning sized by ``spec_info.nxt_kv_lens_cpu``).
- Slots and PAGE rows: the verify forward batch carries the ScheduleBatch's
  host ``req_pool_indices`` (`_install_host_mirror_patch`), and a verify
  replay stages its block tables off `HostPageRows` like a decode replay. A
  DSpark request's row only grows while it lives: spec-v2 over-allocation
  (``DFlashDraftInputV2.prepare_for_decode``) reserves whole pages ahead and
  never shrinks ``kv_allocated_len``; KV is given back only by
  ``release_kv_cache`` (finish, abort, retraction), which ends the request id's
  tenancy of the slot, and a retracted request re-enters through an EXTEND
  that records its new row.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("atom")

DEEPSEEK_V41_ARCH = "DeepseekV41ForCausalLM"
DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
# SGLang's DSV4 pool asserts page_size == 256 on GPU; V4.1's PAGE follows it.
ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE = 256

# DSpark on V4.1: the one speculative deployment this bridge serves.
V41_DSPARK_ALGORITHM = "DSPARK"
# Draft tokens a round proposes (SGLang's gamma, ATOM's num_speculative_tokens).
V41_DSPARK_DRAFT_TOKENS = 5
# Rows a request verifies: the bonus token and the drafts
# (SGLang's speculative_num_draft_tokens / spec_info.draft_token_num).
V41_DSPARK_VERIFY_TOKENS = V41_DSPARK_DRAFT_TOKENS + 1
# Draft stages (checkpoint `num_nextn_predict_layers`), layers 40..42.
V41_DSPARK_STAGES = 3
# ATOM's draft config after `SpeculativeConfig.hf_config_override`.
V41_DSPARK_DRAFT_ARCH = "DeepseekV41DSparkModel"
V41_DSPARK_DRAFT_MODEL_TYPE = "deepseek_v41_dspark"
# Bring-up contract checks that read the device (a sync each): positions and
# mask tokens of the draft block, commit lengths in range and agreed across TP.
# On by default while the path is being validated; "0" turns them off for
# performance runs (record that in the run's notes).
V41_DSPARK_CHECKS_ENV = "ATOM_SGLANG_V41_DSPARK_CHECKS"
# Perf runs only: allow SGLANG_SIMULATE_ACC_LEN > 1 (a fixed mean accept
# length, the output text is then not the model's).
V41_DSPARK_SIMULATED_ACC_ENV = "ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC"
# Kill switch: "0" turns decoder SWA bounded replay off under DSpark only
# (every prefill row runs every layer, the draft context takes every row).
# ATOM_SGLANG_V41_DECODER_REPLAY=0 turns it off everywhere.
V41_DSPARK_DECODER_REPLAY_ENV = "ATOM_SGLANG_V41_DSPARK_DECODER_REPLAY"
# Debug: a directory where replayed and plain DSpark prefills dump the aux
# rows the draft's first block reads (each request's last window rows), to
# compare replay on and off on the same request offline. Eager prefills only.
V41_DSPARK_AUX_DUMP_ENV = "ATOM_SGLANG_V41_DSPARK_AUX_DUMP"

# The metadata builder of this process, created right before SGLang profiles
# the memory its KV pool may take (`_install_pool_sizing_patch`).
_V41_BUILDER = None

# Debug only: compare ATOM's staged step positions with SGLang's on every
# eager forward. The comparison is a device sync per step, so it is off by
# default; positions are derived from the same host lengths SGLang uses.
_CHECK_POSITIONS = os.environ.get("ATOM_SGLANG_V41_CHECK_POSITIONS", "0") == "1"


def is_deepseek_v41_config(config: Any) -> bool:
    """Whether an HF / SGLang / ATOM config describes DeepSeek-V4.1."""
    if config is None:
        return False
    for source in (
        config,
        getattr(config, "hf_config", None),
        getattr(config, "text_config", None),
    ):
        if source is None:
            continue
        if str(getattr(source, "model_type", "") or "") in DEEPSEEK_V41_MODEL_TYPES:
            return True
        architectures = getattr(source, "architectures", None) or ()
        if any(str(arch) == DEEPSEEK_V41_ARCH for arch in architectures):
            return True
    return False


_V41_DSPARK_CHECKS_WARNED = False


def v41_dspark_checks_enabled() -> bool:
    # Off by default: validation-only device reads and a TP all_gather per
    # round, which also put a host sync in the graph replay path.
    enabled = os.environ.get(V41_DSPARK_CHECKS_ENV, "0") == "1"
    global _V41_DSPARK_CHECKS_WARNED
    if enabled and not _V41_DSPARK_CHECKS_WARNED:
        _V41_DSPARK_CHECKS_WARNED = True
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark bring-up checks are on (%s=1): they add "
            "host syncs and a TP all_gather per round; not for perf runs",
            V41_DSPARK_CHECKS_ENV,
        )
    return enabled


def speculative_algorithm_name(value: Any) -> str:
    """SGLang's speculative algorithm (enum, string or None) as an upper-case
    name, "" when speculation is off."""
    if value is None:
        return ""
    name = str(getattr(value, "name", value)).strip().upper()
    return "" if name in ("", "NONE", "NULL", "FALSE") else name


def _dspark_fields(config: Any):
    """The config object that carries V4.1's DSpark keys: the config itself
    (SGLang's normalized config and ATOM's text config are both flat), or a
    nested ``hf_config`` / ``text_config``."""
    hf = getattr(config, "hf_config", None)
    for source in (
        config,
        hf,
        getattr(config, "text_config", None),
        getattr(hf, "text_config", None),
    ):
        if source is not None and getattr(source, "dspark_block_size", None):
            return source
    return None


def v41_dspark_capability_error(config: Any) -> str | None:
    """Why ``config`` is not the V4.1 + bundled DSpark checkpoint this bridge
    serves, or None when it is.

    The capability is read off the checkpoint's own contract rather than the
    family name: a five-wide block, the published target taps, a noise token,
    a Markov head, three draft stages, and those stages window-only (no
    compressor), which is the layout the draft rings 40-42 assume."""
    if not is_deepseek_v41_config(config):
        return "not a DeepSeek-V4.1 checkpoint"
    source = _dspark_fields(config)
    if source is None:
        return "the checkpoint bundles no DSpark head (dspark_block_size)"
    block = int(getattr(source, "dspark_block_size", 0) or 0)
    if block != V41_DSPARK_DRAFT_TOKENS:
        return f"dspark_block_size {block} != {V41_DSPARK_DRAFT_TOKENS}"
    if not list(getattr(source, "dspark_target_layer_ids", None) or ()):
        return "no dspark_target_layer_ids"
    if getattr(source, "dspark_noise_token_id", None) is None:
        return "no dspark_noise_token_id"
    if int(getattr(source, "dspark_markov_rank", 0) or 0) <= 0:
        return "dspark_markov_rank is not positive"
    stages = int(getattr(source, "num_nextn_predict_layers", 0) or 0)
    if stages != V41_DSPARK_STAGES:
        return f"num_nextn_predict_layers {stages} != {V41_DSPARK_STAGES}"
    layers = int(getattr(source, "num_hidden_layers", 0) or 0)
    ratios = [int(r) for r in (getattr(source, "compress_ratios", None) or ())]
    draft_ratios = ratios[layers : layers + stages]
    if len(draft_ratios) != stages or any(draft_ratios):
        return (
            "the draft stages are not window-only "
            f"(compress_ratios[{layers}:] = {draft_ratios})"
        )
    return None


def is_deepseek_v41_dspark(config: Any, speculative_algorithm: Any) -> bool:
    """The positive predicate of the V4.1 DSpark path: a V4.1 checkpoint with
    the bundled DSpark head, served with SGLang's DSPARK algorithm."""
    return (
        speculative_algorithm_name(speculative_algorithm) == V41_DSPARK_ALGORITHM
        and v41_dspark_capability_error(config) is None
    )


def is_v41_dspark_atom_config(atom_config: Any) -> bool:
    """Whether an ATOM ``Config`` is the V4.1 target of a DSpark deployment
    (what `build_v41_dspark_speculative_config` produced)."""
    speculative = getattr(atom_config, "speculative_config", None)
    return (
        speculative is not None
        and getattr(speculative, "method", None) == "dspark"
        and is_deepseek_v41_config(getattr(atom_config, "hf_config", None))
    )


def build_v41_dspark_speculative_config(server_args: Any, hf_config: Any):
    """SGLang DSPARK on a V4.1 checkpoint -> ATOM's native ``SpeculativeConfig``.

    The same object native ATOM builds from ``--method dspark
    --num-speculative-tokens 5`` (`arg_utils`): ``draft_model_hf_config`` is
    left to ATOM's own loader, so the draft config is ATOM's parse of the
    checkpoint, never a mutated view of SGLang's (`hf_config_override` writes
    into whatever it is handed). Its effect on the target is the pool geometry
    (`build_v41_pool_geometry(speculative_tokens=5)`), and native
    `validate_speculative_config` checks it again when ``Config`` is built.
    """
    spec = _sglang_bag("spec")

    def setting(name, default=None):
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        return getattr(server_args, name, default)

    algorithm = speculative_algorithm_name(setting("speculative_algorithm"))
    if algorithm != V41_DSPARK_ALGORITHM:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin supports only "
            f"--speculative-algorithm {V41_DSPARK_ALGORITHM}; got {algorithm}"
        )
    reason = v41_dspark_capability_error(hf_config)
    if reason is not None:
        raise ValueError(f"DeepSeek-V4.1 DSpark is not servable: {reason}")
    verify_tokens = setting("speculative_num_draft_tokens")
    if verify_tokens is None or int(verify_tokens) != V41_DSPARK_VERIFY_TOKENS:
        raise ValueError(
            "DeepSeek-V4.1 DSpark verifies the bonus token and five drafts: "
            f"launch with --speculative-num-draft-tokens {V41_DSPARK_VERIFY_TOKENS}"
            f" (got {verify_tokens})"
        )
    model_path = getattr(server_args, "model_path", None)
    draft_path = setting("speculative_draft_model_path")
    if draft_path and draft_path != model_path:
        raise ValueError(
            "DeepSeek-V4.1 DSpark loads its draft from the target checkpoint; "
            f"got --speculative-draft-model-path {draft_path}"
        )

    from atom.config import SpeculativeConfig

    logger.info(
        "SGLang+ATOM speculative_config method=dspark tokens=%d model=%s",
        V41_DSPARK_DRAFT_TOKENS,
        model_path,
    )
    return SpeculativeConfig(
        method="dspark",
        model=model_path,
        num_speculative_tokens=V41_DSPARK_DRAFT_TOKENS,
    )


def sglang_v41_kv_cache_dtype(kv_cache_dtype: Any) -> str:
    """Map SGLang's ``--kv-cache-dtype`` onto the V4.1 main pool format.

    This reads the operator's input: SGLang's DSV4 hook resolves ``auto`` to
    ``fp8_e4m3`` only in its own config bags, which the ATOM pool ignores.
    """
    dtype = str(kv_cache_dtype or "auto").lower()
    if dtype in ("auto", "bf16", "bfloat16"):
        return "bf16"
    if dtype == "fp8_e4m3":
        return "fp8"
    raise ValueError(
        "DeepSeek-V4.1 on the SGLang plugin supports --kv-cache-dtype "
        f"auto/bfloat16/fp8_e4m3; got {kv_cache_dtype!r}"
    )


def v41_checkpoint_is_fp4ptpc(hf_config: Any) -> bool:
    """Whether a V4.1 checkpoint is FP4PTPC (``quant_method: w8a8_fp8``).

    Reads SGLang's config, whose normalization keeps the checkpoint's
    ``quantization_config``, or an ATOM-normalized one (``fp4ptpc``).
    """
    from atom.models.deepseek_v41.config import FP4PTPC_METHOD

    for source in (hf_config, getattr(hf_config, "text_config", None)):
        if source is None:
            continue
        marked = getattr(source, "fp4ptpc", None)
        if marked is not None:
            return bool(marked)
        quant = getattr(source, "quantization_config", None)
        if quant:
            method = (
                quant.get("quant_method")
                if isinstance(quant, dict)
                else getattr(quant, "quant_method", None)
            )
            return method == FP4PTPC_METHOD
    return False


def v41_default_online_quant_config(
    online_quant_config: Any, gfx: str, hf_config: Any
) -> Any:
    """The online quantization a V4.1 ``Config`` is built with.

    gfx942 has no FP4 MFMA: unless the operator passed one (SGLang carries
    ATOM's ``--online_quant_config`` in ``--model-loader-extra-config``), an
    FP4PTPC checkpoint's routed experts are converted to per-channel FP8 at
    load, the native MI308X recipe. It has to be in place before ``Config``
    is built and the model is constructed: V4.1's MoE refuses an FP4PTPC
    layer whose shared expert did not fuse into the converted routed
    experts, and the DSpark draft refuses unconverted routed experts. A
    native checkpoint gets none: ``Config``'s V4.1 runtime gate refuses
    online quantization of it.
    """
    if online_quant_config is not None or gfx != "gfx942":
        return online_quant_config
    if not v41_checkpoint_is_fp4ptpc(hf_config):
        return None
    return {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------


def _sglang_bag(name: str):
    try:
        from sglang.srt import runtime_context

        return getattr(runtime_context, f"get_{name}")()
    except Exception:  # noqa: BLE001 - bags are published per process
        return None


def _sglang_setting(bag, name, default=None):
    """A resolved SGLang setting off its config bag, else the raw ServerArgs."""
    if bag is not None and hasattr(bag, name):
        return getattr(bag, name)
    from atom.plugin.config import get_sglang_server_args

    return getattr(get_sglang_server_args(), name, default)


def sglang_radix_cache_enabled() -> bool:
    """Whether SGLang serves this launch with its radix (prefix) cache."""
    return not _sglang_setting(_sglang_bag("memory"), "disable_radix_cache", False)


def _prefill_graphs_enabled(exec_) -> bool:
    graph_config = getattr(getattr(exec_, "graph", None), "cuda_graph_config", None)
    prefill = getattr(graph_config, "prefill", None)
    if prefill is not None:
        backend = getattr(prefill, "backend", "disabled")
    else:
        from atom.plugin.config import get_sglang_server_args

        backend = getattr(get_sglang_server_args(), "cuda_graph_backend_prefill", None)
    return str(backend or "disabled").lower() in ("full", "breakable")


def _radix_refusals(memory, exec_, schedule, spec) -> tuple:
    """What radix cache cannot run beside (``deepseek_v41_radix``)."""
    from atom.plugin.sglang.deepseek_v41_radix import (
        REPLAY_ROWS_ENV,
        radix_speculation_refusal,
        resolve_replay_mode,
        resolve_replay_rows,
    )

    rows = resolve_replay_rows()
    resolve_replay_mode()
    spec_reason = radix_speculation_refusal(
        _sglang_setting(spec, "speculative_algorithm", None)
    )
    chunk = _sglang_setting(schedule, "chunked_prefill_size", None)
    return (
        # A MIXED batch skips the replay seam.
        (
            "mixed chunk with radix cache",
            bool(_sglang_setting(schedule, "enable_mixed_chunk", False)),
        ),
        (
            "session radix cache",
            bool(_sglang_setting(memory, "enable_session_radix_cache", False)),
        ),
        (
            "a custom radix cache backend",
            _sglang_setting(memory, "radix_cache_backend", None) is not None,
        ),
        # The only caller of `move_kv_cache`, which the proxy pool refuses.
        (
            "unified memory with radix cache",
            bool(_sglang_setting(memory, "enable_unified_memory", False)),
        ),
        # Trees that rewrite a live request's row outside
        # `UnifiedRadixCache.checkpoint_into_tree`, whose rewrites drop the
        # decode staging's host rows (`deepseek_v41_radix`).
        (
            (
                "a KV storage backend with radix cache (LMCache, FlexKV, "
                "external cache linker)"
            ),
            any(
                bool(_sglang_setting(memory, name, False))
                for name in (
                    "enable_lmcache",
                    "enable_flexkv",
                    "enable_unified_cache_external_linker",
                )
            ),
        ),
        ("prefill CUDA graphs with radix cache", _prefill_graphs_enabled(exec_)),
        (
            f"--chunked-prefill-size {chunk} below {REPLAY_ROWS_ENV}={rows}",
            chunk is not None and 0 < int(chunk) < rows,
        ),
        # DSpark runs under the replay (its draft windows are rebuilt with the
        # target's, `deepseek_v41_radix`); ATOM_V41_RADIX_DSPARK=0 turns that
        # off. Any other algorithm is refused below in any case.
        (
            (
                f"speculative decoding with radix cache: {spec_reason} "
                "(launch with --disable-radix-cache)"
            ),
            spec_reason is not None,
        ),
    )


def refuse_unsupported_sglang_features() -> None:
    """Fail before weights load on SGLang features this bridge cannot serve."""
    from atom.plugin.config import get_sglang_server_args

    args = get_sglang_server_args()
    memory, exec_, spec, disagg, schedule = (
        _sglang_bag(name) for name in ("memory", "exec", "spec", "disagg", "schedule")
    )
    setting = _sglang_setting

    features = getattr(exec_, "features", None)
    unsupported = (
        ("hierarchical cache", setting(memory, "enable_hierarchical_cache", False)),
        ("HiSparse", setting(memory, "enable_hisparse", False)),
        (
            "encoder SWA bounded replay",
            setting(features, "enable_encoder_swa_bounded_replay", False),
        ),
        (
            "PD disaggregation",
            str(setting(disagg, "disaggregation_mode", "null")) != "null",
        ),
        ("DP attention", bool(getattr(args, "enable_dp_attention", False))),
        ("pipeline parallelism", int(getattr(args, "pp_size", 1) or 1) > 1),
        ("two-batch overlap", bool(getattr(args, "enable_two_batch_overlap", False))),
        # Image rows need ATOM's vision path, which the wrapper does not feed;
        # SGLang otherwise also sends an image warmup request.
        (
            "multimodal requests (launch with --language-model-only)",
            not setting(disagg, "language_model_only", False),
        ),
    )
    if sglang_radix_cache_enabled():
        from atom.plugin.sglang import deepseek_v41_envs as envs

        # The launch policy (`deepseek_v41_radix.v41_radix_launch_override`)
        # turns SGLang's radix cache off before this runs wherever the replay
        # cannot serve; these refusals stand in where it did not run.
        unsupported += (
            (
                "radix cache with ATOM_V41_RADIX=0 (launch with --disable-radix-cache)",
                not envs.ATOM_V41_RADIX,
            ),
            # The replay side batch carries no sampling info.
            (
                "watermarking with radix cache",
                setting(features, "enable_watermark", False),
            ),
        )
        unsupported += _radix_refusals(memory, exec_, schedule, spec)
    refused = [name for name, enabled in unsupported if enabled]
    algorithm = speculative_algorithm_name(setting(spec, "speculative_algorithm"))
    if algorithm:
        refused.extend(_dspark_launch_refusals(args, setting, spec, exec_, algorithm))
    if refused:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin does not support " + ", ".join(refused)
        )


def _graph_backend(exec_, args, phase: str) -> str:
    """The CUDA graph backend SGLang resolved for ``phase``, lower-case."""
    try:
        config = exec_.graph.cuda_graph_config
    except AttributeError:
        config = getattr(args, "cuda_graph_config", None)
    backend = getattr(getattr(config, phase, None), "backend", None)
    if backend is None:
        return "disabled" if getattr(args, "disable_cuda_graph", False) else "unknown"
    return str(getattr(backend, "value", backend)).lower()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _dspark_decode_graph_refusals(args, exec_) -> list:
    """What a DSpark launch with full decode graphs (the target's verify and
    the draft's block, both TARGET_VERIFY runners) asks for that the staging
    cannot serve."""
    refused = []
    graph = getattr(exec_, "graph", None)
    padding_off = getattr(graph, "disable_cuda_graph_padding", None)
    if padding_off is None:
        padding_off = getattr(args, "disable_cuda_graph_padding", False)
    if padding_off:
        # One graph per bucket, padded rows staged as zero-length requests.
        refused.append("DSpark decode graphs with --disable-cuda-graph-padding")
    if _env_flag("SGLANG_ENABLE_METADATA_GLUE_GRAPH"):
        # The glue graph would freeze the per-round host staging.
        refused.append("DSpark decode graphs with SGLANG_ENABLE_METADATA_GLUE_GRAPH")
    try:
        config = exec_.graph.cuda_graph_config
    except AttributeError:
        config = getattr(args, "cuda_graph_config", None)
    sizes = getattr(getattr(config, "decode", None), "bs", None) or ()
    schedule = _sglang_bag("schedule")
    max_running = getattr(schedule, "max_running_requests", None) or getattr(
        args, "max_running_requests", None
    )
    if sizes and max_running and max(int(s) for s in sizes) < int(max_running):
        # Not refused: SGLang runs larger rounds eagerly, both stages and the
        # accept, at several times the graph round's cost.
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark decode graphs stop at bs %d below "
            "--max-running-requests %s: rounds above it run eagerly (slower); "
            "raise --cuda-graph-max-bs-decode to cover the running batch",
            max(int(s) for s in sizes),
            max_running,
        )
    if sizes and max_running and max(int(s) for s in sizes) > int(max_running):
        # The builder (max_num_seqs) and the reserved capture STATE slots
        # are both sized by --max-running-requests.
        refused.append(
            f"DSpark decode graphs up to bs {max(int(s) for s in sizes)} above "
            f"--max-running-requests {max_running}"
        )
    return refused


def _dspark_launch_refusals(args, setting, spec, exec_, algorithm: str) -> list:
    """What a speculative V4.1 launch asks for that the DSpark path does not
    serve. Only DSPARK, five drafts, static verify, synchronous; decode graphs
    full or off, prefill eager."""
    if algorithm != V41_DSPARK_ALGORITHM:
        return [f"speculative algorithm {algorithm} (only {V41_DSPARK_ALGORITHM})"]
    refused = []
    verify_tokens = setting(spec, "speculative_num_draft_tokens")
    if verify_tokens is None or int(verify_tokens) != V41_DSPARK_VERIFY_TOKENS:
        refused.append(
            f"DSpark with --speculative-num-draft-tokens {verify_tokens} "
            f"(launch with {V41_DSPARK_VERIFY_TOKENS})"
        )
    draft_path = setting(spec, "speculative_draft_model_path")
    if draft_path and draft_path != getattr(args, "model_path", None):
        refused.append("a DSpark draft checkpoint other than the target's")
    mode = os.environ.get("SGLANG_RAGGED_VERIFY_MODE", "static").strip().lower()
    if mode != "static":
        refused.append(
            f"SGLANG_RAGGED_VERIFY_MODE={mode} (DSpark verify must be static)"
        )
    prefill = _graph_backend(exec_, args, "prefill")
    if prefill != "disabled":
        # DSpark's prefill captures FULL hidden states and runs every row
        # (decoder replay off): eager.
        refused.append(
            f"DSpark with the prefill CUDA graph ({prefill}); launch with "
            "--cuda-graph-backend-prefill disabled"
        )
    decode = _graph_backend(exec_, args, "decode")
    if decode == "full":
        refused.extend(_dspark_decode_graph_refusals(args, exec_))
    elif decode != "disabled":
        refused.append(
            f"DSpark with the {decode} decode CUDA graph backend (launch with "
            "--cuda-graph-backend-decode full or disabled)"
        )
    # The overlap schedule is served (see "DSpark under the overlap schedule"
    # in the module docstring): a round's verify, accept and commit are one
    # `forward_batch_generation` call on the scheduler thread and the forward
    # stream, so the verify's tentative step is committed, in the epoch it
    # opened, before the next round is staged.
    embed_in_graph = os.environ.get("SGLANG_DSPARK_EMBED_IN_GRAPH", "1").strip()
    if embed_in_graph.lower() in ("0", "false", "no", "off"):
        # The ATOM draft embeds its own block; SGLang would pass input_embeds.
        refused.append("SGLANG_DSPARK_EMBED_IN_GRAPH=0 (the ATOM draft embeds)")
    simulated = os.environ.get("SGLANG_SIMULATE_ACC_LEN", "").strip()
    if (
        simulated
        and not (float(simulated) <= 0 or float(simulated) == 1.0)
        and not _env_flag(V41_DSPARK_SIMULATED_ACC_ENV)
    ):
        # A fixed acceptance length commits draft tokens the target did not
        # choose: the text is not the model's. Benchmarks at a fixed mean
        # accept length (native's --spec-decode-acceptance-length) opt in.
        refused.append(
            f"SGLANG_SIMULATE_ACC_LEN={simulated} (only 1.0, the verify-only "
            f"test, unless {V41_DSPARK_SIMULATED_ACC_ENV}=1 for a perf run)"
        )
    if setting(spec, "speculative_use_block_verification", False):
        refused.append("DSpark block verification")
    try:
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        unified_kv = bool(is_unified_kv_triton())
    except Exception:  # noqa: BLE001 - absent kernels mean no unified-kv path
        unified_kv = False
    if unified_kv:
        refused.append("DSpark over SGLang's unified-kv DSV4 ring")
    return refused


def align_atom_config_with_sglang(atom_config: Any) -> None:
    """Size the builder by SGLang's resolved schedule, not the operator input.

    ``Config`` is built from the raw ServerArgs; SGLang's model hooks resolve
    ``max_running_requests`` (256 for DSV4) and the prefill chunk only in
    its config bags, and the builder's fixed buffers must hold every step the
    scheduler can form.
    """
    schedule = _sglang_bag("schedule")
    if schedule is None:
        return
    tokens = [int(atom_config.max_num_batched_tokens)]
    for name in ("chunked_prefill_size", "max_prefill_tokens"):
        value = getattr(schedule, name, None)
        if value is not None and int(value) > 0:
            tokens.append(int(value))
    atom_config.max_num_batched_tokens = max(tokens)
    max_running_requests = getattr(schedule, "max_running_requests", None)
    if max_running_requests:
        atom_config.max_num_seqs = int(max_running_requests)
    # The builder sizes its score workspace from this fraction; the raw
    # ServerArgs value is None unless the operator passed it.
    mem_fraction_static = getattr(schedule, "mem_fraction_static", None)
    if mem_fraction_static:
        atom_config.gpu_memory_utilization = float(mem_fraction_static)
    elif atom_config.gpu_memory_utilization is None:
        raise ValueError(
            "DeepSeek-V4.1 cannot resolve SGLang's mem_fraction_static; pass "
            "--mem-fraction-static"
        )


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------
# `atom.plugin.vllm.v41_core_overrides` holds the same publication wrapper,
# but importing anything under `atom.plugin.vllm` imports vLLM.


def _v41_publication_owner(variables, device):
    from atom.utils import CpuGpuBuffer
    from atom.utils.h2d import PublicationOwner

    event = torch.cuda.Event() if torch.device(device).type == "cuda" else None
    owner = PublicationOwner(device, event)
    members = {}
    for name, buffer in variables.items():
        if isinstance(buffer, CpuGpuBuffer):
            binding = owner.bind(buffer, name, unit=buffer.publication_unit)
            if buffer.publication_group:
                members.setdefault(buffer.publication_group, []).append(binding)
    groups = {name: owner.group(name, bindings) for name, bindings in members.items()}
    if event is not None:
        event.record()
    return owner, groups


@contextmanager
def _v41_publication_epoch(owner, *, resume=False):
    """One H2D publication epoch; ``resume`` reopens the epoch a step's
    preparation sealed (its tentative Engram prefixes are written at
    preparation and read by the commit after sampling)."""
    if resume:
        owner.resume()
    else:
        owner.begin()
    try:
        yield
    except BaseException:
        owner.fail()
        raise
    else:
        owner.finish()


def _publish_v41_metadata(prepare):
    """Run a step's staging inside one H2D publication epoch.

    ATOM's runner owns the publication registry natively; the shim runner here
    gets one built over its ``forward_vars`` on first use.
    """

    @wraps(prepare)
    def publish(self, *args, **kwargs):
        runner = self.model_runner
        if getattr(runner, "h2d_owner", None) is None:
            owner, groups = _v41_publication_owner(runner.forward_vars, self.device)
            for name, names in self.h2d_group_members().items():
                groups[name] = owner.group(
                    name, [runner.forward_vars[item]._publication for item in names]
                )
            runner.h2d_owner, runner.h2d_groups = owner, groups
        with _v41_publication_epoch(runner.h2d_owner):
            return prepare(self, *args, **kwargs)

    return publish


def _make_builder_cls():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    class SGLangDeepseekV41MetadataBuilder(DeepseekV41MetadataBuilder):
        """ATOM's V4.1 builder behind a shim runner, staged once per step."""

        def prepare_model_inputs(self, input_ids, metadata):
            cache = metadata.cache
            if getattr(cache, "_sglang_h2d_owner", None) is None:
                variables = {
                    name: getattr(cache, name)
                    for name in (
                        "_cursor_staging",
                        "tentative_staging",
                        "_reset_slots",
                        "_reset_histories",
                    )
                }
                cache._sglang_h2d_owner, _ = _v41_publication_owner(
                    variables, self.device
                )
            with _v41_publication_epoch(cache._sglang_h2d_owner):
                super().prepare_model_inputs(input_ids, metadata)

        # Whether the prefill being staged gets a decoder tail: the bridge's
        # verdict on its ForwardBatch (`v41_extend_replays`), carried on the
        # scheduled batch as ``decoder_replays``.
        _sglang_decoder_replays = True

        @_publish_v41_metadata
        def _prepare(self, batch, running_bs, running_tokens, **kwargs):
            # No runner token inputs carry the query metadata on this path.
            self.publish_query_metadata(
                np.asarray(batch.num_scheduled_tokens, np.int32),
                running_bs,
                running_tokens,
            )
            self._sglang_decoder_replays = bool(getattr(batch, "decoder_replays", True))
            try:
                return super()._prepare(batch, running_bs, running_tokens, **kwargs)
            finally:
                self._sglang_decoder_replays = True

        def _publish_decoder(self, step, rows, cache, running_bs):
            if not self._sglang_decoder_replays:
                # `step.decoder` stays None, so the runtime model's
                # `replayable` is False and every row runs every layer.
                return None
            return super()._publish_decoder(step, rows, cache, running_bs)

        def build_for_sglang_capture(self, bs, *, input_ids, slots, width=1):
            """The step a decode graph of ``bs`` requests is recorded on.

            Core ``build_for_cudagraph_capture`` with two differences: the
            STATE slots are the caller's (the proxy pool's reserved capture
            slots, never a live request's), and ``input_ids`` is SGLang's
            graph buffer, the address every replay of this bucket refills.
            Engram's stage, gather and cursor advance stay inside the forward
            (``staged_outside_forward`` unset), gated on a device live count
            of 0 here and of the real rows on replay.

            ``width`` 1 is a plain decode (``max_q_len=1``). On a DSpark pool
            the only graph shape is the verify, ``width`` =
            ``speculative_tokens + 1`` rows a request: a tentative step, as
            native's ``build_for_cudagraph_capture(bs, max_q_len=K+1)``. The
            step is dummy, so ``prepare_state`` is skipped and no
            ``TentativeState`` is left pending.
            """
            from atom.utils.forward_context import RoundService

            cache = self.cache
            if cache is None:
                raise RuntimeError("Allocate the serving cache before graph capture")
            geometry = cache.geometry
            speculative = int(geometry.speculative_tokens or 0)
            bs, width = int(bs), int(width)
            if speculative and width != speculative + 1:
                raise NotImplementedError(
                    "DeepSeek-V4.1 SGLang graphs on a speculative pool capture "
                    f"the verify step ({speculative + 1} rows a request) only, "
                    f"not width {width}"
                )
            if not speculative and width != 1:
                raise NotImplementedError(
                    "DeepSeek-V4.1 SGLang decode graphs capture one row a "
                    f"request, not {width}"
                )
            tokens = bs * width
            if bs < 1 or bs > self.max_bs or tokens > self.max_num_batched_tokens:
                raise ValueError(
                    f"DeepSeek-V4.1 capture of {bs} requests x {width} exceeds the "
                    f"builder's {self.max_bs} / {self.max_num_batched_tokens}"
                )
            slots = tuple(int(slot) for slot in slots)
            if len(slots) != bs or len(set(slots)) != bs:
                raise ValueError(
                    f"DeepSeek-V4.1 capture needs {bs} distinct STATE slots; "
                    f"got {slots}"
                )
            if int(input_ids.shape[0]) != tokens:
                raise ValueError(
                    f"DeepSeek-V4.1 capture input_ids rows {int(input_ids.shape[0])} "
                    f"!= {bs} requests x {width}"
                )
            # A full window in, as core does: at position 0 the compressor
            # reads no history and records a cold branch replay never takes
            # (and a tentative step needs a prefix).
            start = geometry.window_size
            pages = -(-(start + width) // self.block_size)
            batch = SimpleNamespace(
                is_dummy_run=False,
                req_ids=slots,
                num_scheduled_tokens=(width,) * bs,
                context_lens=(start + width,) * bs,
                state_slots_committed=slots,
                # SGLang's padding page for every entry: capture's compressed
                # rows land there, and no request is ever handed page 0.
                block_tables=((0,) * pages,) * bs,
                total_seqs_num=bs,
                total_tokens_num=tokens,
                is_final_chunk=None,
                replaying={},
            )
            tentative = bool(speculative)
            metadata, _ = self._prepare(
                batch,
                bs,
                tokens,
                max_q_len=width,
                tentative=tentative,
                engram_live=False,
            )
            # No slot reset, host Engram lookup or cursor advance for it.
            metadata.dummy = True
            self.prepare_model_inputs(input_ids, metadata)
            if metadata.cache is not cache:
                raise RuntimeError("DeepSeek-V4.1 capture bound a scratch cache")
            if metadata.served_by_round != RoundService.NONE:
                raise NotImplementedError(
                    "DeepSeek-V4.1 SGLang decode graphs do not run a model "
                    "decode round"
                )
            if tentative and getattr(cache, "pending", None) is not None:
                raise RuntimeError(
                    "DeepSeek-V4.1 verify capture left a tentative state pending"
                )
            return metadata

        def build_for_sglang_token_graph_capture(self, tokens, *, input_ids, slots):
            """The prefill a token-graph bucket of ``tokens`` rows is recorded
            on: core ``build_for_token_graph_capture`` (finishing requests of
            one window each from position 0, the last taking the remainder,
            every row on SGLang's padding PAGE 0, Engram on no token), on the
            proxy pool's reserved capture STATE slots instead of slots
            ``0..n-1`` (live requests' rows in SGLang's table), and over
            ``input_ids``, the buffer every padded prefill stages its ids in.
            Returns ``(metadata, Context)``; publication as `_prepare` and
            `prepare_model_inputs` do it for any step."""
            from atom.utils.forward_context import Context

            if self.cache is None:
                raise RuntimeError("Allocate the serving cache before graph capture")
            tokens = int(tokens)
            if not 0 < tokens <= self.max_num_batched_tokens:
                raise ValueError(
                    f"DeepSeek-V4.1 token-graph bucket {tokens} exceeds the "
                    f"builder's {self.max_num_batched_tokens} tokens"
                )
            window = int(self.geometry.window_size)
            lengths = (window,) * (tokens // window) + (
                (tokens % window,) * bool(tokens % window)
            )
            if len(lengths) > self.max_bs:
                lengths = (tokens,)
            count = len(lengths)
            slots = tuple(int(slot) for slot in slots[:count])
            if len(slots) != count or len(set(slots)) != count:
                raise ValueError(
                    f"DeepSeek-V4.1 token-graph capture of {count} requests "
                    f"needs as many distinct STATE slots; got {slots}"
                )
            if int(input_ids.shape[0]) < tokens:
                raise ValueError(
                    f"DeepSeek-V4.1 token-graph ids buffer of {int(input_ids.shape[0])} "
                    f"rows under the {tokens} bucket"
                )
            pages = -(-max(lengths) // self.block_size)
            batch = SimpleNamespace(
                is_dummy_run=False,
                req_ids=slots,
                num_scheduled_tokens=lengths,
                context_lens=lengths,
                state_slots_committed=slots,
                block_tables=((0,) * pages,) * count,
                total_seqs_num=count,
                total_tokens_num=tokens,
                is_final_chunk=None,
                replaying={},
            )
            metadata, positions = self._prepare(
                batch, count, tokens, is_prefill=True, engram_live=False
            )
            # No slot reset, host Engram lookup or cursor advance for it.
            metadata.dummy = True
            self.prepare_model_inputs(input_ids[:tokens], metadata)
            context = Context(
                positions=positions[:tokens],
                is_prefill=True,
                is_dummy_run=False,
                scheduled_bs=count,
                scheduled_tokens=tokens,
                running_bs=count,
                running_tokens=tokens,
            )
            return metadata, context

    return SGLangDeepseekV41MetadataBuilder


def compact_v41_moe_buffers(model: Any, role: str = "target") -> int:
    """Re-materialize every MoE layer's expert tensors, one layer at a time,
    once SGLang has post-processed the weights and before its memory profile.
    Returns the layers compacted.

    The load frees and reallocates multi-GiB expert buffers layer by layer
    (ATOM's online FP4 -> FP8 quant, then SGLang's ``postprocess_weights``
    shuffling each of them). The caching allocator serves later buffers out
    of earlier, larger free blocks and leaves their tails split off inside
    segments that still hold live weights; ``empty_cache`` cannot return
    those (about 5 GiB a rank on DeepSeek-V4.1-Flash TP4), and the profile,
    which reads the device's free bytes, takes them from the KV pool.
    Copying each layer's tensors into fresh allocations and emptying the
    cache after each layer hands back whole segments instead.

    Values and Parameter objects are unchanged (``.data`` is swapped); no
    graph or kernel has recorded their addresses yet. The fused-MoE quant
    config ``init_prepare_finalize`` built during the load is rebuilt over
    the layer's current tensors, so it keeps none of the old ones alive."""
    import gc

    def stranded() -> int:
        stats = torch.cuda.memory_stats()
        return int(stats.get("inactive_split_bytes.all.current", 0))

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DeepSeek-V4.1 MoE buffers compacted under a capture")
    gc.collect()
    torch.cuda.empty_cache()
    stranded_before = stranded()
    layers = 0
    for module in model.modules():
        if not isinstance(getattr(module, "w13_weight", None), torch.nn.Parameter):
            continue
        for param in module.parameters(recurse=False):
            fresh = torch.empty_like(param.data)
            fresh.copy_(param.data)
            param.data = fresh
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "moe_quant_config", None) is not None:
            quant_method.moe_quant_config = quant_method.get_fused_moe_quant_config(
                module
            )
        layers += 1
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang: %s MoE buffers compacted in %d layers "
        "(reserved %.2f GiB, stranded %.2f -> %.2f GiB)",
        role,
        layers,
        torch.cuda.memory_reserved() / (1 << 30),
        stranded_before / (1 << 30),
        stranded() / (1 << 30),
    )
    return layers


def init_deepseek_v41_builder(atom_config: Any, device: Any = None):
    """Create this process's V4.1 metadata builder. Idempotent.

    Called once the weights are loaded and post-processed and before SGLang
    profiles free memory: the builder sizes its score workspace from what the
    weights leave, and the KV budget then already counts it.
    """
    global _V41_BUILDER
    if _V41_BUILDER is not None:
        return _V41_BUILDER

    from atom.utils import CpuGpuBuffer

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    max_num_batched_tokens = int(atom_config.max_num_batched_tokens)
    runner = SimpleNamespace(
        block_size=_sglang_page_size(),
        device=device,
        config=atom_config,
        max_bs=max(1, int(atom_config.max_num_seqs)),
        max_num_batched_tokens=max_num_batched_tokens,
        h2d_owner=None,
        h2d_groups=None,
        forward_vars={
            # `prepare_batch_step` takes its request-start dtype from here.
            "positions": CpuGpuBuffer(
                max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
                pin_memory=torch.device(device).type != "cpu",
            )
        },
    )
    builder = _make_builder_cls()(runner)
    geometry = builder.geometry
    speculative = getattr(atom_config, "speculative_config", None)
    if speculative is not None:
        _check_dspark_geometry(builder, atom_config)
    _V41_BUILDER = builder
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang builder: role=target block_size=%d max_bs=%d "
        "max_num_batched_tokens=%d kv=%s index=%s spec=%d layers=%d "
        "ring_slots=%d compress_ring=%d layout_id=%s paged_bytes=%d state_bytes=%d",
        runner.block_size,
        runner.max_bs,
        max_num_batched_tokens,
        atom_config.kv_cache_dtype,
        atom_config.index_cache_dtype,
        int(geometry.speculative_tokens),
        int(geometry.layers),
        int(geometry.ring_slots),
        int(geometry.compress_ring_slots),
        geometry.layout_id,
        int(geometry.paged_bytes),
        int(geometry.state_bytes),
    )
    return _V41_BUILDER


def _check_dspark_geometry(builder, atom_config) -> None:
    """The pool a DSpark deployment needs: five drafts of window slack and the
    draft stages' rings in every STATE slot, and a builder able to stage a
    verify of every running request at once."""
    geometry = builder.geometry
    hf = atom_config.hf_config
    stages = int(getattr(hf, "num_nextn_predict_layers", 0) or 0)
    expected_layers = int(hf.num_hidden_layers) + stages
    if (
        geometry.speculative_tokens != V41_DSPARK_DRAFT_TOKENS
        or stages != V41_DSPARK_STAGES
        or geometry.layers != expected_layers
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark needs a pool of "
            f"{V41_DSPARK_DRAFT_TOKENS} speculative tokens over "
            f"{expected_layers} layers; got spec={geometry.speculative_tokens} "
            f"layers={geometry.layers}"
        )
    rows = builder.max_bs * V41_DSPARK_VERIFY_TOKENS
    if builder.max_num_batched_tokens < rows:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark verifies {V41_DSPARK_VERIFY_TOKENS} rows a "
            f"request: max_num_batched_tokens {builder.max_num_batched_tokens} "
            f"< {builder.max_bs} requests x {V41_DSPARK_VERIFY_TOKENS}"
        )


def get_deepseek_v41_builder():
    if _V41_BUILDER is None:
        raise RuntimeError(
            "DeepSeek-V4.1 metadata builder is not initialized; it is created "
            "before SGLang profiles memory for the KV pool"
        )
    return _V41_BUILDER


def _sglang_page_size() -> int:
    page_size = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE
    try:
        from atom.plugin.config import get_sglang_server_args

        page_size = int(getattr(get_sglang_server_args(), "page_size", 0) or page_size)
    except Exception:  # noqa: BLE001, S110 - server args are optional in unit tests
        pass
    if page_size != ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin needs page_size "
            f"{ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE} (one SGLang page per V4.1 "
            f"PAGE); got {page_size}"
        )
    return page_size


# ---------------------------------------------------------------------------
# KV pool
# ---------------------------------------------------------------------------


# Where each of a quantized pool's source buffers starts in the arena: what
# a fresh allocation of its own would be aligned to, at least.
_V41_SOURCE_ALIGN = 256


def _v41_source_buffers(geometry) -> int:
    """How many buffers ``PagedAttentionCache`` allocates for a quantized pool
    (`_allocate_quantized`): the auxiliary one (index rows, compressor state,
    cursor, padding), one main source per owner and one window source per
    layer. A BF16 pool is one flat arena."""
    if not geometry.window_fp8:
        return 1
    return 1 + len(geometry.owners) + int(geometry.layers)


def _arena_bytes(geometry, num_pages: int, num_slots: int) -> int:
    size = int(geometry.paged_extents(num_pages)[1]) + num_slots * int(
        geometry.state_bytes
    )
    if geometry.window_fp8:
        # Room to start every source buffer on its own boundary.
        size += _V41_SOURCE_ALIGN * _v41_source_buffers(geometry)
    return size


def _capture_slot_count(builder) -> int:
    """STATE slots reserved for decode-graph capture: one per request of the
    widest step the builder stages. Reserved whether or not graphs are on
    (64 x 5.3 MB on Flash bf16), so the pool layout never depends on it."""
    return max(1, int(builder.max_bs))


def _make_backed_cache_cls():
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    class SGLangBackedPagedAttentionCache(PagedAttentionCache):
        """``PagedAttentionCache`` over the proxy pool's arena."""

        def __init__(self, *args, backing: torch.Tensor, **kwargs):
            # `super().__init__` calls `_allocate_backing`.
            self._external_backing = backing
            super().__init__(*args, **kwargs)

        def _checked_backing(self) -> torch.Tensor:
            backing = self._external_backing
            if backing.dtype != torch.uint8 or backing.dim() != 1:
                raise ValueError("A CSA2 backing must be flat uint8")
            if not backing.is_contiguous():
                raise ValueError("A CSA2 backing must be contiguous")
            return backing

        def _allocate_backing(self, size, device):
            # A BF16 pool: the whole cache is one flat arena.
            return self._allocate_buffers([size], device)[0]

        def _allocate_buffers(self, sizes, device):
            # A quantized pool (`_allocate_quantized`): its source buffers,
            # carved in order out of the arena, the first at its base, each
            # on a `_V41_SOURCE_ALIGN` boundary (`_arena_bytes` has the room).
            backing = self._checked_backing()
            views, offset = [], 0
            for size in sizes:
                offset = -(-offset // _V41_SOURCE_ALIGN) * _V41_SOURCE_ALIGN
                if offset + int(size) > backing.numel():
                    raise ValueError(
                        f"A CSA2 backing of {backing.numel()} bytes is short of "
                        f"the {offset + int(size)} this geometry needs"
                    )
                view = backing[offset : offset + int(size)]
                # Core's own allocations are zeroed; the cursor reset and
                # every cold read assume it.
                view.zero_()
                views.append(view)
                offset += int(size)
            return tuple(views)

    return SGLangBackedPagedAttentionCache


def _make_proxy_pool_cls():
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    class ATOMDeepSeekV41ProxyKVPool(BaseSWAKVPool):
        """SGLang-visible DSV4 pool whose bytes are ATOM's CSA2 arena.

        Installed under SGLang's ``DeepSeekV4TokenToKVPool`` name, so every
        ``isinstance`` check SGLang makes for a DSV4 pool sees this class.
        Constructed with ``_build_dsv4_kv_pool``'s keyword arguments.
        """

        is_atom_v41_proxy_pool = True
        # The DSpark draft runner's role view (`_init_draft_role`).
        is_atom_v41_draft_proxy = False
        atom_cache = None
        # SGLang reads these off any DSV4 pool: no unified-kv ring, no
        # encoder-replay request window, no SWA allocator (see module doc).
        _unified_kv = False
        _unified_kv_fp8 = False
        needs_paged_swa_allocator = False
        request_window = None
        unified_kv_pool = None
        full_kv_pool = None
        swa_kv_pool = None
        c4_kv_pool = None
        c128_kv_pool = None
        custom_mem_pool = None
        enable_custom_mem_pool = False
        layer_transfer_counter = None

        def __init__(
            self,
            max_num_reqs: int,
            swa_size: int | None,
            page_size: int,
            swa_page_size: int,
            dtype: torch.dtype | None = None,
            layer_num: int = 0,
            device: str = "cuda",
            start_layer: int | None = None,
            end_layer: int | None = None,
            num_req_slots: int | None = None,
            full_size: int | None = None,
            is_draft_worker: bool = False,
            **_unused_kwargs: Any,
        ) -> None:
            if is_draft_worker:
                self._init_draft_role(
                    max_num_reqs=max_num_reqs,
                    swa_size=swa_size,
                    page_size=page_size,
                    swa_page_size=swa_page_size,
                    dtype=dtype,
                    layer_num=layer_num,
                    device=device,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    full_size=full_size,
                )
                return
            self.is_atom_v41_draft_proxy = False
            builder = _V41_BUILDER
            if builder is None:
                # The sizing hook did not run (another SGLang entry point):
                # the builder's buffers land outside the profiled budget.
                from atom.config import get_current_atom_config

                logger.warning(
                    "DeepSeek-V4.1 builder created after the memory profile; "
                    "its buffers are not in the KV budget"
                )
                builder = init_deepseek_v41_builder(get_current_atom_config())
            geometry = builder.geometry
            if int(page_size) != geometry.block_size:
                raise ValueError(
                    f"SGLang page_size {page_size} != V4.1 PAGE "
                    f"{geometry.block_size}"
                )
            full_tokens = full_size if full_size is not None else swa_size
            if not full_tokens:
                raise ValueError(
                    "DeepSeek-V4.1 proxy pool needs SGLang's full token count; "
                    "do not launch with --disable-hybrid-swa-memory"
                )
            self.max_num_reqs = int(max_num_reqs)
            self.page_size = int(page_size)
            self.swa_page_size = int(swa_page_size)
            self.dtype = dtype
            self.store_dtype = torch.uint8
            self.device = device
            self.start_layer = 0 if start_layer is None else int(start_layer)
            self.end_layer = int(end_layer) if end_layer is not None else int(layer_num)
            self.layer_num = int(layer_num)
            # `size` is the token capacity SGLang's allocator schedules against.
            self.size = int(full_tokens)
            self.full_size = self.size
            self.swa_size = int(swa_size or 0)
            self.size_swa = self.swa_size
            self.full_to_swa_index_mapping = None

            # Page 0 is SGLang's padding page, never handed to a request, but
            # it is still addressable: size the PAGE region to cover it.
            self.num_pages = self.size // self.page_size + 1
            # One STATE slot per req_to_token row (row 0 is padding), then
            # one reserved slot per request a decode graph captures: capture
            # (at startup or a mid-serving recapture) never writes a live
            # request's state, and no req_pool_idx can name a capture slot.
            self.num_req_slots = max(
                1,
                (
                    int(num_req_slots)
                    if num_req_slots is not None
                    else self.max_num_reqs + 1
                ),
            )
            self.num_capture_slots = _capture_slot_count(builder)
            self.capture_slot_base = self.num_req_slots
            self.num_slots = self.num_req_slots + self.num_capture_slots
            self.capture_slots = tuple(range(self.capture_slot_base, self.num_slots))
            # Per request-row STATE slot, the `req_to_token_pool.req_generation`
            # of the tenant its state was last rebuilt or started for. Written
            # only by the radix replay seam (`deepseek_v41_radix`), read by the
            # extend guard; the generation SGLang bumps on every row allocation
            # is how a new tenant is told from a chunked continuation.
            self.state_gen = np.zeros(self.num_req_slots, dtype=np.int64)
            total_bytes = _arena_bytes(geometry, self.num_pages, self.num_slots)
            self.raw_arena = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            # Reported by /get_internal_state and the scheduler metrics, in GiB.
            self.mem_usage = total_bytes / (1 << 30)

            builder.num_blocks = self.num_pages
            builder.cache = _make_backed_cache_cls()(
                geometry,
                self.num_pages,
                self.num_slots,
                builder.device,
                max_tokens=builder.max_num_batched_tokens,
                workspace=builder.score_workspace,
                # A quantized pool's prefill window source, which the builder
                # reserved before SGLang's memory profile (native hands its
                # serving cache the same buffer).
                prefill_window_staging=getattr(
                    builder, "_prefill_window_staging", None
                ),
                backing=self.raw_arena,
            )
            self.atom_cache = builder.cache
            logger.info(
                "ATOM DeepSeek-V4.1 proxy pool: role=target %d PAGEs x %d B + "
                "%d STATE slots (%d request rows + %d capture slots from %d) "
                "x %d B = %.2f GiB (SGLang tokens %d)",
                self.num_pages,
                int(geometry.paged_bytes),
                self.num_slots,
                self.num_req_slots,
                self.num_capture_slots,
                self.capture_slot_base,
                int(geometry.state_bytes),
                self.mem_usage,
                self.size,
            )

        def _init_draft_role(
            self,
            *,
            max_num_reqs,
            swa_size,
            page_size,
            swa_page_size,
            dtype,
            layer_num,
            device,
            start_layer,
            end_layer,
            full_size,
        ) -> None:
            """The DSpark draft runner's pool: a role view of the target arena.

            The draft's whole KV is its context window, rings 40-42 of the
            requests' own STATE slots, which the target pool already holds
            and prices (`geometry.state_bytes` at ``spec=5``). So this pool
            allocates nothing and binds the target's ATOM cache. Same class
            as the target's, so SGLang's DSV4 ``isinstance`` checks hold;
            ``is_atom_v41_draft_proxy`` tells the roles apart. It defines no
            ``set_swa_key_buffer_radix_fused_norm_rope``, so SGLang's injector
            hands the target hidden to the draft model
            (``write_target_hidden_kv``) instead of writing MLA rows here.
            """
            builder = _V41_BUILDER
            if builder is None or builder.cache is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 DSpark draft pool built before the target's "
                    "ATOM arena"
                )
            geometry = builder.geometry
            if geometry.speculative_tokens != V41_DSPARK_DRAFT_TOKENS:
                raise ValueError(
                    "DeepSeek-V4.1 DSpark draft pool needs the target arena "
                    f"built for {V41_DSPARK_DRAFT_TOKENS} speculative tokens; "
                    f"it has {geometry.speculative_tokens}"
                )
            if int(page_size) != geometry.block_size:
                raise ValueError(
                    f"SGLang page_size {page_size} != V4.1 PAGE "
                    f"{geometry.block_size}"
                )
            self.is_atom_v41_draft_proxy = True
            self.max_num_reqs = int(max_num_reqs)
            self.page_size = int(page_size)
            self.swa_page_size = int(swa_page_size)
            self.dtype = dtype
            self.store_dtype = torch.uint8
            self.device = device
            self.start_layer = 0 if start_layer is None else int(start_layer)
            self.end_layer = int(end_layer) if end_layer is not None else int(layer_num)
            self.layer_num = int(layer_num)
            tokens = full_size if full_size is not None else swa_size
            self.size = self.full_size = int(tokens or 0)
            self.swa_size = self.size_swa = int(swa_size or 0)
            self.full_to_swa_index_mapping = None
            cache = builder.cache
            self.num_pages, self.num_slots = cache.num_pages, cache.num_slots
            self.atom_cache = cache
            self.raw_arena = torch.empty(0, dtype=torch.uint8, device=device)
            self.mem_usage = 0.0
            logger.info(
                "ATOM DeepSeek-V4.1 proxy pool: role=draft bound to the target "
                "arena (%d STATE slots, window rings of layers %s), 0 B allocated",
                self.num_slots,
                tuple(
                    range(
                        geometry.layers - V41_DSPARK_STAGES,
                        geometry.layers,
                    )
                ),
            )

        def get_kv_size_bytes(self):
            return self.raw_arena.nbytes

        def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
            if self.is_atom_v41_draft_proxy:
                return
            self.full_to_swa_index_mapping = full_to_swa_index_mapping

        def translate_loc_from_full_to_swa(
            self, kv_indices: torch.Tensor
        ) -> torch.Tensor:
            raise NotImplementedError(
                "ATOM V4.1 proxy pool has no SWA token pool; the window lives "
                "in ATOM's STATE ring"
            )

        def set_swa_loc(self, loc: torch.Tensor) -> None:
            pass

        def get_state_buf_infos(self):
            return ([], [], [])

        def get_contiguous_buf_infos(self):
            return ([self.raw_arena.data_ptr()], [self.raw_arena.nbytes], [1])

        def get_kv_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def set_kv_buffer(self, *args, **kwargs) -> None:
            raise NotImplementedError("ATOM V4.1 proxy pool is written by ATOM kernels")

        def get_key_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def get_value_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
            raise NotImplementedError(
                "DeepSeek-V4.1 on the SGLang plugin does not support KV "
                "relocation (--enable-unified-memory) yet"
            )

    return ATOMDeepSeekV41ProxyKVPool


def _is_atom_v41_target(kvc) -> bool:
    """Whether SGLang's KV configurator is building this plugin's V4.1 pool."""
    return (
        not getattr(kvc, "is_draft_worker", False)
        and is_deepseek_v41_config(getattr(kvc.model_config, "hf_config", None))
        and getattr(getattr(kvc, "model", None), "atom_config", None) is not None
    )


def _make_pool_configurator_cls():
    from sglang.srt.model_executor.pool_configurator import (
        MemoryPoolConfig,
        MemoryPoolConfigurator,
    )

    class ATOMDeepSeekV41PoolConfigurator(MemoryPoolConfigurator):
        """Size SGLang's V4.1 token pool by what ATOM's arena costs.

        SGLang's ``DSV4PoolConfigurator`` prices its own layout (FlashMLA KV
        rows of the KV-source layers plus a paged SWA pool). The arena is
        ``geometry.paged_bytes`` per 256-token PAGE, one STATE slot per
        request row plus the decode-graph capture slots, and the padding PAGE;
        ``req_to_token`` is charged here too since nothing else budgets it.
        """

        def __init__(self, kvc) -> None:
            builder = get_deepseek_v41_builder()
            geometry = builder.geometry
            self.geometry = geometry
            # The proxy pool's reserved decode-graph capture slots.
            self.capture_slots = _capture_slot_count(builder)
            self.page_size = int(kvc.page_size)
            self.context_len = int(kvc.model_config.context_len)
            schedule = _sglang_bag("schedule")
            disagg = _sglang_bag("disagg")
            requested = getattr(schedule, "max_running_requests", None)
            self.requested_reqs = (
                None
                if requested is None
                else max(1, int(requested) // max(1, int(kvc.attn_dp_size)))
            )
            self.extra_slots = (
                int(getattr(disagg, "disaggregation_decode_extra_slots", 0) or 0)
                if str(getattr(disagg, "disaggregation_mode", "null")) == "decode"
                else 0
            )

        def _pool_bytes(self, pages: int, slots: int) -> int:
            # `pages` usable PAGEs plus the padding PAGE, the STATE slots of
            # the request rows and of decode-graph capture, and the int32
            # req_to_token rows SGLang allocates beside them.
            return (
                _arena_bytes(self.geometry, pages + 1, slots + self.capture_slots)
                + slots * self.context_len * 4
            )

        def _reqs(self, available_bytes: int) -> int:
            if self.requested_reqs is not None:
                return self.requested_reqs
            # As ModelRunner resolves it when --max-running-requests is unset.
            tokens = available_bytes * self.page_size // self.geometry.paged_bytes
            estimated = max(min(int(tokens / self.context_len * 512), 4096), 2048)
            return max(1, min(estimated, tokens // 2))

        def _config(self, tokens: int) -> MemoryPoolConfig:
            tokens = tokens // self.page_size * self.page_size
            return MemoryPoolConfig(
                max_total_num_tokens=tokens,
                full_max_total_num_tokens=tokens,
                swa_max_total_num_tokens=0,
            )

        def calculate_pool_sizes(
            self, available_bytes: int, page_size: int
        ) -> MemoryPoolConfig:
            assert page_size == self.page_size
            slots = self._reqs(available_bytes) + self.extra_slots + 1
            fixed = self._pool_bytes(0, slots)
            pages = max(0, (available_bytes - fixed) // int(self.geometry.paged_bytes))
            # The extents are aligned per owner; step back until they fit.
            while pages > 0 and self._pool_bytes(pages, slots) > available_bytes:
                pages -= 1
            logger.info(
                "ATOM DeepSeek-V4.1 pool sizing: available=%.2f GiB, "
                "fixed=%.2f GiB (%d request slots + %d capture slots), "
                "%d PAGEs x %d B -> %d tokens",
                available_bytes / (1 << 30),
                fixed / (1 << 30),
                slots,
                self.capture_slots,
                pages,
                int(self.geometry.paged_bytes),
                pages * self.page_size,
            )
            return self._config(pages * self.page_size)

        def calculate_pool_sizes_from_max_tokens(
            self, max_total_num_tokens: int, page_size: int
        ) -> MemoryPoolConfig:
            assert page_size == self.page_size
            return self._config(max_total_num_tokens)

    return ATOMDeepSeekV41PoolConfigurator


def _install_pool_sizing_patch() -> None:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor import pool_configurator

    if getattr(pool_configurator, "_atom_v41_sizing_patched", False):
        return
    configurator_cls = _make_pool_configurator_cls()
    original_factory = pool_configurator.create_memory_pool_configurator

    def create_memory_pool_configurator(kvc):
        if _is_atom_v41_target(kvc):
            return configurator_cls(kvc)
        return original_factory(kvc)

    # kv_cache_configurator imports the factory inside its methods.
    pool_configurator.create_memory_pool_configurator = create_memory_pool_configurator

    original_profile = KVCacheConfigurator._profile_available_bytes

    def profile_available_bytes(self, pre_model_load_memory):
        # SGLang post-processes the weights after `load_weights` returns, so
        # this is the first point where the builder sees the memory they
        # leave; its buffers then come out of the profiled budget. The MoE
        # buffers are compacted first, so the profile sees no stranded bytes.
        if _is_atom_v41_target(self):
            if _V41_BUILDER is None:
                compact_v41_moe_buffers(self.model)
                # The DSpark draft, loaded, post-processed and attached by
                # now (SGLang wires it before the target's profile), strands
                # its own share.
                surface = getattr(self.model, "_v41_dspark", None)
                draft = getattr(surface, "draft", None)
                if draft is not None:
                    compact_v41_moe_buffers(draft, role="draft")
            init_deepseek_v41_builder(self.model.atom_config)
        return original_profile(self, pre_model_load_memory)

    KVCacheConfigurator._profile_available_bytes = profile_available_bytes
    pool_configurator._atom_v41_sizing_patched = True


def _install_graph_variants_patch() -> None:
    """Keep SGLang's V4.1 candidate-indexer graph variants off this model.

    SGLang (gfx95x / sm100+) would capture several decode graphs per bucket,
    keyed on its own candidate-indexer kernels' limits; the ATOM model runs
    none of them, and the backend stages one graph per bucket. Inert on gfx942.
    """
    try:
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner
    except Exception:  # noqa: BLE001 - graphs are optional
        return
    original = getattr(
        decode_cuda_graph_runner, "create_dsv41_candidate_graph_variants", None
    )
    if original is None or getattr(original, "_atom_v41_patched", False):
        return

    @wraps(original)
    def create_dsv41_candidate_graph_variants(model_runner, *args, **kwargs):
        model_config = getattr(model_runner, "model_config", None)
        if is_deepseek_v41_config(getattr(model_config, "hf_config", None)):
            return None
        return original(model_runner, *args, **kwargs)

    create_dsv41_candidate_graph_variants._atom_v41_patched = True
    decode_cuda_graph_runner.create_dsv41_candidate_graph_variants = (
        create_dsv41_candidate_graph_variants
    )


def _is_v41_runner(model_runner) -> bool:
    cached = getattr(model_runner, "_atom_is_v41", None)
    if cached is None:
        model_config = getattr(model_runner, "model_config", None)
        cached = is_deepseek_v41_config(getattr(model_config, "hf_config", None))
        try:
            model_runner._atom_is_v41 = cached
        except AttributeError:
            pass
    return cached


def _install_host_mirror_patch() -> None:
    """Carry a decode or verify batch's host request rows to the V4.1 staging.

    SGLang keeps ``req_pool_indices_cpu`` on the ScheduleBatch through
    decode (filter / merge maintain it) but hands it to the ForwardBatch for
    EXTEND only, and the decode graph runner's replay view carries neither it
    nor the request ids. Both ride along as ``atom_v41_*`` attributes, on
    V4.1 DECODE and TARGET_VERIFY batches only (DSpark's verify is built from
    the ScheduleBatch by ``prepare_for_verify``), so
    `build_v41_graph_replay_batch` can stage without reading the device back.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if not getattr(ForwardBatch, "_atom_v41_host_mirror_patched", False):
        original_init_new = ForwardBatch.init_new.__func__

        @wraps(original_init_new)
        def init_new(cls, batch, model_runner, *args, **kwargs):
            forward_batch = original_init_new(cls, batch, model_runner, *args, **kwargs)
            mode = forward_batch.forward_mode
            if (mode.is_decode() or _is_target_verify(mode)) and _is_v41_runner(
                model_runner
            ):
                forward_batch.atom_v41_req_pool_indices_cpu = getattr(
                    batch, "req_pool_indices_cpu", None
                )
            return forward_batch

        ForwardBatch.init_new = classmethod(init_new)
        ForwardBatch._atom_v41_host_mirror_patched = True

    try:
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner
    except Exception:  # noqa: BLE001 - graphs are optional
        return
    original_view = decode_cuda_graph_runner.build_replay_fb_view
    if getattr(original_view, "_atom_v41_patched", False):
        return

    @wraps(original_view)
    def build_replay_fb_view(*args, **kwargs):
        view = original_view(*args, **kwargs)
        forward_batch = kwargs.get("forward_batch", args[0] if args else None)
        host_slots = getattr(forward_batch, "atom_v41_req_pool_indices_cpu", None)
        if host_slots is not None:
            view.atom_v41_req_pool_indices_cpu = host_slots
            view.atom_v41_rids = getattr(forward_batch, "rids", None)
        return view

    build_replay_fb_view._atom_v41_patched = True
    decode_cuda_graph_runner.build_replay_fb_view = build_replay_fb_view


def install_deepseek_v41_proxy_pool_patch() -> None:
    """Make SGLang size and build the V4.1 proxy pool instead of its DSV4 pool,
    keep its own V4.1 decode-graph variants off the ATOM model, carry decode
    batches' host request rows to the replay staging, and make the
    scheduler's request broadcast poll instead of sleep."""
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as dsv4_pool

    from atom.plugin.sglang.patches.v41_spin_broadcast_patch import (
        install_v41_spin_broadcast_patch,
    )

    install_v41_spin_broadcast_patch()
    _install_pool_sizing_patch()
    _install_graph_variants_patch()
    _install_host_mirror_patch()
    from atom.plugin.sglang.patches.v41_extend_rows_at_alloc_patch import (
        install_v41_extend_rows_at_alloc_patch,
    )
    from atom.plugin.sglang.patches.v41_prefill_token_graph_patch import (
        install_v41_prefill_token_graph_patch,
    )

    install_v41_extend_rows_at_alloc_patch()
    install_v41_prefill_token_graph_patch()
    original_pool_cls = getattr(dsv4_pool, "DeepSeekV4TokenToKVPool", None)
    if original_pool_cls is None:
        raise RuntimeError("SGLang DeepSeekV4TokenToKVPool is not available")
    if getattr(original_pool_cls, "is_atom_v41_proxy_pool", False):
        return
    proxy_cls = _make_proxy_pool_cls()
    dsv4_pool.DeepSeekV4TokenToKVPool = proxy_cls
    dsv4_pool.ATOMDeepSeekV41ProxyKVPool = proxy_cls
    # The configurator and kv_cache_builder import the class as a module-local
    # symbol; later local imports read the patched module attribute.
    for module in list(sys.modules.values()):
        if getattr(module, "DeepSeekV4TokenToKVPool", None) is original_pool_cls:
            module.DeepSeekV4TokenToKVPool = proxy_cls


# ---------------------------------------------------------------------------
# Per-forward translation
# ---------------------------------------------------------------------------


def _cpu_ints(cpu_value, gpu_value, n: int) -> np.ndarray:
    value = cpu_value if cpu_value is not None else gpu_value
    if value is None:
        raise ValueError("DeepSeek-V4.1 forward batch is missing a length field")
    if torch.is_tensor(value):
        value = value[:n].detach().cpu().numpy()
    return np.asarray(value[:n], dtype=np.int64)


def _req_to_token() -> torch.Tensor:
    # ForwardBatch carries no pool references in 0.5.22; ModelRunner publishes
    # its attention backend, which holds them, around every forward.
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        has_forward_context,
    )

    if not has_forward_context():
        raise RuntimeError("DeepSeek-V4.1 forward ran outside SGLang's forward context")
    return get_req_to_token_pool().req_to_token


def _block_rows(req_to_token, req_pool_indices, ends, block_size: int):
    widths = [-(-int(end) // block_size) for end in ends]
    widest = max(widths, default=0)
    if widest == 0:
        return [np.zeros(0, dtype=np.int32) for _ in widths]
    table = (
        req_to_token[req_pool_indices, : widest * block_size : block_size] // block_size
    ).to(torch.int32)
    table = table.cpu().numpy()
    rows = [table[i, : widths[i]].copy() for i in range(len(widths))]
    _refuse_padding_page(rows)
    return rows


class HostPageRows:
    """Host copies of live requests' PAGE rows, so a decode or DSpark verify
    replay can stage without reading ``req_to_token`` back.

    That read is a device sync on the forward stream: it waits for everything
    queued before it (the previous step's graph; under DSpark, this round's
    draft block) and serializes the step's host staging behind it. A
    request's PAGE row only grows, one page every 256 tokens, and only
    SGLang's allocator writes it -- except that with radix cache on a
    checkpoint repoints a live row at the tree's pages in place, which drops
    that slot's copy (``forget``, called from the radix seam's
    ``checkpoint_into_tree`` hook; ``host_page_rows_supported``). Under
    DSpark the pages a verify's draft rows enter are SGLang's spec-v2
    over-allocation, reserved whole and kept until the request's KV is
    released (never per rejection), so they too only grow; with radix cache
    on, a checkpoint inserts only page-aligned committed rows
    (``[cache_protected_len, extend_end)``), never those, and drops the copy
    like any other. An EXTEND (every
    request's first forward, and again after a retraction) reads the row back
    anyway and records it here; a decode or verify reads it again only when a
    request has entered a page its copy does not hold, the slot's request
    changed, or a checkpoint dropped the copy.

    Keyed by STATE slot (``req_pool_idx``) and checked against the request id,
    so a copy is never used for another tenant of the slot. Under
    ``ATOM_SGLANG_V41_CHECK_HOST_ROWS=1`` every replay also reads the rows
    back and compares (a device sync per step; validation only).
    """

    def __init__(self):
        self._rows = {}
        # Per slot, how many times its row was rewritten in place (`forget`):
        # a copy read before a rewrite and recorded after it (a prefetch,
        # `VerifyRowPrefetch`) is refused by comparing this.
        self._generations = {}

    def generation(self, slot) -> int:
        return self._generations.get(int(slot), 0)

    def record(self, slots, rids, rows, generations=None):
        """Record each slot's row. ``generations``: the slot generations when
        the rows were read; a slot rewritten since keeps no copy."""
        if generations is None:
            generations = [None] * len(slots)
        for slot, rid, row, gen in zip(slots, rids, rows, generations):
            if gen is not None and gen != self.generation(slot):
                self._rows.pop(int(slot), None)
                continue
            self._rows[int(slot)] = (rid, row)

    def forget(self, slot) -> None:
        """Drop ``slot``'s copy: its row was rewritten in place."""
        slot = int(slot)
        self._rows.pop(slot, None)
        self._generations[slot] = self._generations.get(slot, 0) + 1

    def lookup(self, slots, rids, widths):
        """Each request's first ``widths[i]`` pages, or None if any request's
        copy is missing, stale or too short."""
        rows = []
        for slot, rid, width in zip(slots, rids, widths):
            entry = self._rows.get(int(slot))
            if entry is None or entry[0] != rid or len(entry[1]) < width:
                return None
            rows.append(entry[1][:width])
        return rows


_HOST_ROWS = HostPageRows()


_HOST_ROWS_SUPPORTED = None
# Set once the radix seam invalidates a slot's copy on every in-place rewrite
# of its row (`deepseek_v41_radix._install_checkpoint_invalidation`).
_HOST_ROWS_RADIX_SAFE = False


def forget_host_page_rows(slot) -> None:
    """The tree rewrote ``slot``'s ``req_to_token`` row: drop its host copy."""
    _HOST_ROWS.forget(slot)


def mark_host_page_rows_radix_safe() -> None:
    global _HOST_ROWS_RADIX_SAFE, _HOST_ROWS_SUPPORTED
    _HOST_ROWS_RADIX_SAFE = True
    _HOST_ROWS_SUPPORTED = None


def host_page_rows_supported() -> bool:
    """Whether a live request's host PAGE row can be trusted: either its row
    changes only through its own forwards (radix cache off), or every
    in-place rewrite -- a radix checkpoint repointing the row at the tree's
    pages -- drops the copy (``mark_host_page_rows_radix_safe``). Fixed for
    the process once the server is up, so resolved once."""
    global _HOST_ROWS_SUPPORTED
    if _HOST_ROWS_SUPPORTED is None:
        from atom.plugin.config import get_sglang_server_args

        memory = _sglang_bag("memory")
        if memory is not None and hasattr(memory, "disable_radix_cache"):
            radix_off = bool(memory.disable_radix_cache)
        else:
            radix_off = bool(
                getattr(get_sglang_server_args(), "disable_radix_cache", False)
            )
        _HOST_ROWS_SUPPORTED = radix_off or _HOST_ROWS_RADIX_SAFE
    return _HOST_ROWS_SUPPORTED


_CHECK_HOST_ROWS = os.environ.get("ATOM_SGLANG_V41_CHECK_HOST_ROWS", "0") == "1"

# Kill switch: "0" stops prefetching a DSpark round's PAGE rows before its
# draft launch (`VerifyRowPrefetch`); a verify that enters a page its host
# copy does not hold then reads the rows back behind the draft block.
V41_DSPARK_PREFETCH_ROWS_ENV = "ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS"


class VerifyRowPrefetch:
    """The PAGE rows a DSpark round's verify will miss in `HostPageRows`,
    copied to the host before the round's draft block is launched.

    A verify replay that finds a request's host copy shorter than the pages
    of ``p..p+5`` reads the rows back with a blocking D2H. Staged after the
    draft launch, that read waits for the whole draft block, then the rest of
    the staging runs with the GPU idle. Which rows will miss is known before
    the draft: SGLang's ``prepare_for_decode`` reserved every request's KV up
    to ``spec_info.nxt_kv_lens_cpu`` (page-aligned, at least the round's
    ``p + 6``; ``alloc_for_spec_decode`` wrote those ``req_to_token`` entries
    and the forward stream is ordered after them), so a request whose host
    copy does not cover ``nxt / 256`` pages gets those pages read on a side
    stream into pinned memory, without waiting. The verify's staging takes
    them (`take`, one event wait: the copy depends only on the allocation,
    not on the draft) and records them in `HostPageRows`, which then holds
    them the way an EXTEND or a read-back would: keyed by STATE slot and
    request id, so a copy is never used for another tenant. Under
    ``ATOM_SGLANG_V41_CHECK_HOST_ROWS=1`` the staging still reads every row
    back and compares.

    With radix cache on, a checkpoint (``checkpoint_into_tree``) repoints a
    live row at the tree's pages and drops its copy (``forget``). It runs in
    the scheduler's result processing (a chunk's or a release's), never
    between this ``issue`` and the same round's ``take``, and a checkpoint
    before ``issue`` makes the row a miss read afresh: the forward stream
    the side copy waits on waits on the schedule stream that rewrote it. The
    rows still carry their slots' rewrite generation from ``issue``, and
    ``take`` records none whose slot was rewritten since, so a stale copy can
    never enter `HostPageRows` whatever the order.
    """

    def __init__(self):
        self.pending = None
        self.stream = None
        self.event = None
        self.pinned = None

    def _host(self, rows: int, pages: int) -> torch.Tensor:
        """A contiguous pinned ``[rows, pages]`` (an async D2H needs one)."""
        pinned = self.pinned
        if pinned is None or pinned.numel() < rows * pages:
            pinned = torch.empty(
                (max(rows * pages, 64 * 64),),
                dtype=torch.int32,
                pin_memory=torch.cuda.is_available(),
            )
            self.pinned = pinned
        return pinned[: rows * pages].view(rows, pages)

    def issue(self, slots, rids, bounds, req_to_token, block_size: int) -> int:
        """Start the copy of the rows `HostPageRows` does not cover up to
        each request's reserved length ``bounds``; returns how many."""
        self.drop()
        # The table's columns bound a row (a reservation at the context cap).
        widths = np.minimum(
            -(-np.asarray(bounds, dtype=np.int64) // block_size),
            -(-int(req_to_token.shape[1]) // block_size),
        )
        miss = [
            i
            for i, (slot, rid, width) in enumerate(zip(slots, rids, widths))
            if _HOST_ROWS.lookup([slot], [rid], [width]) is None
        ]
        if not miss:
            return 0
        miss_slots = [int(slots[i]) for i in miss]
        pages = int(max(widths[i] for i in miss))
        device = req_to_token.device
        module = _device_module(device)
        current = module.current_stream()
        if self.stream is None:
            self.stream = module.Stream(device=device)
            self.event = module.Event()
        host = self._host(len(miss), pages)
        index = torch.tensor(miss_slots, dtype=torch.int64)
        self.stream.wait_stream(current)
        with module.stream(self.stream):
            index = index.to(device, non_blocking=True)
            table = (
                req_to_token[index, : pages * block_size : block_size] // block_size
            ).to(torch.int32)
            host.copy_(table, non_blocking=True)
            self.event.record(self.stream)
        self.pending = (
            miss_slots,
            [rids[i] for i in miss],
            [int(widths[i]) for i in miss],
            host,
            # A radix checkpoint that rewrites one of these rows before
            # `take` (`forget_host_page_rows`) voids that row's copy.
            [_HOST_ROWS.generation(slot) for slot in miss_slots],
        )
        return len(miss)

    def take(self) -> bool:
        """Record the prefetched rows in `HostPageRows` (one event wait);
        False if none are pending."""
        if self.pending is None:
            return False
        slots, rids, widths, host, generations = self.pending
        self.pending = None
        self.event.synchronize()
        rows = [host[i, :width].numpy().copy() for i, width in enumerate(widths)]
        _refuse_padding_page(rows)
        _HOST_ROWS.record(slots, rids, rows, generations)
        return True

    def drop(self) -> None:
        """Forget a prefetch no verify took (an eager round, or one whose
        copies already covered it): its pinned rows are reused next."""
        if self.pending is not None:
            self.pending = None
            self.event.synchronize()


_VERIFY_PREFETCH = VerifyRowPrefetch()


# Kill switch: "0" makes an EXTEND read its slots and PAGE rows back from the
# device at its staging (`_read_replay_rows`), behind whatever the forward
# stream still runs, instead of taking them from `ExtendRowRead`.
V41_EXTEND_ROWS_AT_ALLOC_ENV = "ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC"


def v41_extend_rows_at_alloc_enabled() -> bool:
    return os.environ.get(V41_EXTEND_ROWS_AT_ALLOC_ENV, "1") != "0"


class ExtendRowRead:
    """An EXTEND's PAGE rows, copied to the host where SGLang wrote them.

    The staging of an EXTEND needs every request's PAGE row on the host.
    Read back at staging, the copy is queued on the forward stream behind the
    step in flight -- under overlap a whole DSpark verify graph (~60 ms at
    c32) or decode graph -- and every host launch of the prefill waits for
    it. The rows exist much earlier: ``alloc_for_extend`` wrote them into
    ``req_to_token`` on the scheduler's stream, which does not wait for the
    forward stream. `issue` queues the copy right there, on that stream,
    after the allocator's own write; the staging of the same batch `take` s
    it (one event wait, for those few kernels). A copy is only used by the
    batch whose allocation made it: same STATE slots in the same order, same
    request ids, at least the widths the staging needs; anything else (a
    MIXED batch, a batch that never ran) falls back to the read-back.
    """

    def __init__(self):
        self.pending = None

    def issue(self, req_pool_indices, slots, rids, ends, req_to_token, block_size):
        self.pending = None
        ends = np.asarray(ends, dtype=np.int64)
        widths = np.minimum(
            -(-ends // block_size), -(-int(req_to_token.shape[1]) // block_size)
        )
        pages = int(widths.max(initial=0))
        if pages == 0 or len(slots) != len(rids) or len(slots) != len(ends):
            return False
        device = req_to_token.device
        module = _device_module(device)
        # On the caller's (the scheduler's) stream, after the allocator's
        # write of these rows.
        table = (
            req_to_token[
                req_pool_indices[: len(slots)].to(device, torch.int64),
                : pages * block_size : block_size,
            ]
            // block_size
        ).to(torch.int32)
        host = torch.empty(
            tuple(table.shape), dtype=torch.int32, pin_memory=device.type == "cuda"
        )
        host.copy_(table, non_blocking=True)
        event = module.Event()
        event.record()
        self.pending = ([int(s) for s in slots], list(rids), widths, host, event)
        return True

    def take(self, slots, rids, widths):
        """The rows of the batch ``(slots, rids)`` (first ``widths[i]`` pages
        each), or None when the pending copy is not that batch's."""
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        read_slots, read_rids, read_widths, host, event = pending
        if (
            [int(s) for s in slots] != read_slots
            or list(rids) != read_rids
            or np.any(np.asarray(widths, np.int64) > read_widths)
        ):
            return None
        event.synchronize()
        table = host.numpy()
        rows = [table[i, : int(width)].copy() for i, width in enumerate(widths)]
        _refuse_padding_page(rows)
        return rows

    def drop(self) -> None:
        self.pending = None


_EXTEND_ROWS = ExtendRowRead()


def read_v41_extend_rows_at_alloc(batch, req_pool_indices, req_pool_indices_cpu):
    """After SGLang's ``alloc_for_extend`` for a V4.1 target batch: queue the
    D2H of its requests' PAGE rows on the current (scheduler) stream
    (`ExtendRowRead`). From the batch's own allocation: its host slots,
    request ids and ``seq_lens_cpu``."""
    if _V41_BUILDER is None or _V41_BUILDER.cache is None:
        _EXTEND_ROWS.drop()
        return False
    reqs = getattr(batch, "reqs", None) or []
    ends = getattr(batch, "seq_lens_cpu", None)
    pool = getattr(batch, "req_to_token_pool", None)
    if ends is None or pool is None or req_pool_indices_cpu is None:
        _EXTEND_ROWS.drop()
        return False
    bs = len(reqs)
    if len(req_pool_indices_cpu) != bs:
        _EXTEND_ROWS.drop()
        return False
    return _EXTEND_ROWS.issue(
        req_pool_indices,
        [int(s) for s in req_pool_indices_cpu],
        [req.rid for req in reqs],
        _cpu_ints(ends, None, bs),
        pool.req_to_token,
        _V41_BUILDER.block_size,
    )


def _device_module(device):
    return torch.get_device_module(torch.device(device))


def v41_dspark_prefetch_rows_enabled() -> bool:
    return os.environ.get(V41_DSPARK_PREFETCH_ROWS_ENV, "1") != "0"


def prefetch_v41_dspark_verify_rows(batch, req_to_token, block_size: int) -> int:
    """Before a DSpark decode round's draft launch: prefetch the PAGE rows
    its verify would otherwise read back behind the draft
    (`VerifyRowPrefetch`). From the round's own ScheduleBatch: host slots,
    request ids and SGLang's reservation ``spec_info.nxt_kv_lens_cpu``."""
    if not host_page_rows_supported():
        # A round that prefetches nothing must not leave an earlier round's
        # copy for its verify to take.
        _VERIFY_PREFETCH.drop()
        return 0
    slots = getattr(batch, "req_pool_indices_cpu", None)
    bounds = getattr(getattr(batch, "spec_info", None), "nxt_kv_lens_cpu", None)
    reqs = getattr(batch, "reqs", None)
    if slots is None or bounds is None or reqs is None:
        _VERIFY_PREFETCH.drop()
        return 0
    bs = len(reqs)
    if len(slots) != bs or len(bounds) < bs:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark row prefetch: {bs} requests, {len(slots)} host "
            f"slots, {len(bounds)} reserved lengths"
        )
    slots = [int(slot) for slot in slots]
    bounds = _cpu_ints(bounds, None, bs)
    count = _VERIFY_PREFETCH.issue(
        slots, [req.rid for req in reqs], bounds, req_to_token, block_size
    )
    if count:
        count_v41_dspark("rows_prefetched", count)
    return count


def _refuse_padding_page(rows) -> None:
    # SGLang's paged allocator never hands out page 0 (its padding page, also
    # where decode-graph capture rows point), so a live row naming it read a
    # req_to_token entry that was never written.
    if any(np.any(row <= 0) for row in rows):
        raise ValueError(
            "DeepSeek-V4.1 block table names SGLang's padding page 0 in a live "
            "request's row"
        )


def _is_target_verify(mode) -> bool:
    return bool(getattr(mode, "is_target_verify", lambda: False)())


def _take_late_prefix(bs: int, host_slots):
    """A late-mode DSpark round's ``(committed prefix, STATE slots)``, taken
    by its verify staging (`dspark_deferred_seq_lens_patch.take_late_seq_lens`:
    the host waits here, after SGLang's own verify preparation), or None when
    none is pending. ``host_slots``, when the staging has them, must name the
    copy's slots; otherwise the caller compares the returned ones with the
    slots it stages (`_require_late_slots`)."""
    from atom.plugin.sglang.patches.dspark_deferred_seq_lens_patch import (
        take_late_seq_lens,
    )

    taken = take_late_seq_lens(
        bs, None if host_slots is None else _cpu_ints(host_slots, None, bs)
    )
    if taken is None:
        return None
    lens, slots = taken
    count_v41_dspark("late_seq_lens")
    return _cpu_ints(lens, None, bs), slots


def _require_late_slots(late_slots, slots) -> None:
    """The slots a verify staged are those its late host lengths were
    resolved for."""
    if late_slots is not None and tuple(int(s) for s in late_slots) != tuple(
        int(s) for s in slots
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark verify staged STATE slots "
            f"{[int(s) for s in slots]}, its late host lengths are those of "
            f"{list(late_slots)}"
        )


def _committed_prefix(spec_info, ends, bs: int, width: int, host_slots):
    """``(prefix, ends, late_slots)`` of a verify of ``width`` rows a request.

    ``live_seq_lens_cpu`` is the committed prefix the verify caller recorded
    before widening, ``ends`` (``seq_lens_cpu``) the widened lengths, which
    must agree. Without it, SGLang ran its ``needs_cpu_seq_lens=False``
    contract (the late host lengths, `_take_late_prefix`): ``ends`` is then
    SGLang's reservation bound, the exact ends are the taken prefix plus the
    width, which must fit inside it, and ``late_slots`` are the slots the
    staging must turn out to stage (`_require_late_slots`); None otherwise."""
    live = getattr(spec_info, "live_seq_lens_cpu", None)
    if live is None:
        taken = _take_late_prefix(bs, host_slots)
        if taken is None:
            from atom.plugin.sglang.patches.dspark_deferred_seq_lens_patch import (
                deferred_state_summary,
            )

            # `seq_lens` may be SGLang's over-allocated bound here, so the
            # prefix cannot be derived from it.
            raise ValueError(
                "DeepSeek-V4.1 TARGET_VERIFY needs spec_info.live_seq_lens_cpu "
                f"(bs={bs}; {deferred_state_summary()})"
            )
        late, late_slots = taken
        if np.any(late + width > ends):
            raise ValueError(
                "DeepSeek-V4.1 TARGET_VERIFY lengths exceed SGLang's reservation: "
                f"prefix={late.tolist()} width={width} bound={ends.tolist()}"
            )
        return late, late + width, late_slots
    prefix = _cpu_ints(live, None, bs)
    if prefix.shape != ends.shape or np.any(prefix + width != ends):
        raise ValueError(
            "DeepSeek-V4.1 TARGET_VERIFY lengths disagree: "
            f"prefix={prefix.tolist()} width={width} seq_lens={ends.tolist()}"
        )
    return prefix, ends, None


def _verify_lengths(
    forward_batch, ends, bs: int, speculative_tokens: int, host_slots=None
):
    """``(prefix, width, ends, late_slots)`` of a TARGET_VERIFY: every request verifies
    ``spec_info.draft_token_num`` rows, the bonus token and its drafts, on top
    of its committed prefix.

    The width is SGLang's official verify width, never the extend lengths.
    ``seq_lens_cpu`` arrives already widened by it (this backend has no
    ``make_forward_metadata_from_raw_verify``), or as SGLang's reservation
    bound under the late host lengths (`_committed_prefix`).
    """
    spec_info = getattr(forward_batch, "spec_info", None)
    width = int(getattr(spec_info, "draft_token_num", 0) or 0)
    if speculative_tokens <= 0:
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang bridge does not translate TARGET_VERIFY "
            "without a speculative (DSpark) pool"
        )
    if width != speculative_tokens + 1:
        raise ValueError(
            f"DeepSeek-V4.1 TARGET_VERIFY width {width} != the pool's "
            f"{speculative_tokens} drafts + 1"
        )
    prefix, ends, late_slots = _committed_prefix(spec_info, ends, bs, width, host_slots)
    if np.any(prefix <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 TARGET_VERIFY needs a committed prefix; got {prefix}"
        )
    return prefix, width, ends, late_slots


def _radix_stamps():
    """The proxy pool's per-slot tenant stamps and SGLang's row generations,
    off the pools of the current forward context."""
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    return (
        get_token_to_kv_pool().state_gen,
        get_req_to_token_pool().req_generation,
    )


def _check_radix_resumes(slots, prefix, fused=None) -> None:
    """With radix cache on, refuse an extend resuming a slot that is neither
    stamped for its current tenant nor rebuilt by the fused record this
    forward carries. Runs while the metadata is built, before any kernel."""
    from atom.plugin.sglang import deepseek_v41_radix as radix

    if radix.radix_replay_rows() is None or not np.any(prefix > 0):
        return
    state_gen, generation = _radix_stamps()
    carried = () if fused is None else fused.slots
    radix.check_state_stamps(state_gen, generation, slots, prefix, carried)


def _stamp_fused_hits(fused) -> None:
    """Stamp on consume: this forward carries ``fused``, checked against its
    ForwardBatch, so its hits' slots now belong to their tenants."""
    from atom.plugin.sglang import deepseek_v41_radix as radix

    if radix.radix_replay_rows() is None:
        return
    state_gen, generation = _radix_stamps()
    radix.stamp_fused_hits(state_gen, generation, fused)


def _replaying(forward_batch, prefix, ends, bs: int, block_size: int, history):
    """``ScheduledBatch.replaying`` for a radix replay side forward: every row
    rebuilds its fresh slot over ``[start, P)`` (``deepseek_v41_radix``)."""
    from atom.model_engine.sequence import Replay

    mode = forward_batch.forward_mode
    if getattr(mode, "name", None) != "EXTEND":
        raise NotImplementedError(
            f"DeepSeek-V4.1 radix replay runs as an EXTEND, not {mode}"
        )
    if getattr(forward_batch, "spec_info", None) is not None:
        raise NotImplementedError("DeepSeek-V4.1 radix replay carries no spec_info")
    ids = getattr(forward_batch, "engram_history", None)
    if (
        ids is None
        or ids.dim() != 2
        or int(ids.shape[0]) != bs
        or (history is not None and int(ids.shape[1]) != int(history))
    ):
        raise ValueError(
            "DeepSeek-V4.1 radix replay needs engram_history of "
            f"[{bs}, {history}], got "
            f"{None if ids is None else tuple(ids.shape)}"
        )
    if np.any(ends % block_size) or np.any(prefix % 2) or np.any(prefix >= ends):
        raise ValueError(
            "DeepSeek-V4.1 radix replay spans must start even and end on a "
            f"PAGE boundary: starts={prefix.tolist()} ends={ends.tolist()}"
        )
    ids = ids.detach().cpu().numpy()
    width = ids.shape[1]
    replaying = {}
    for i in range(bs):
        start = int(prefix[i])
        # SGLang left-pads a history shorter than its width; native ATOM
        # carries only the tokens that exist (`Scheduler._resume_prefill`).
        keep = min(width, start)
        replaying[i] = Replay(
            until=int(ends[i]),
            rebuilds=True,
            history_ids=tuple(int(t) for t in ids[i, width - keep :]),
            start=start,
        )
    return replaying


def _fused_replaying(fused, prefix, slots, rids, block_size: int, history):
    """``ScheduledBatch.replaying`` for a step carrying a fused radix replay
    (``deepseek_v41_radix.FusedReplay``): each hit's span starts at
    ``P - R`` in this forward, its slot rebuilt over ``[P - R, P)``. The
    record is checked against the ForwardBatch it rides on."""
    from atom.model_engine.sequence import Replay

    rows = fused.rows
    if rows.size and (rows[0] < 0 or rows[-1] >= len(slots)):
        raise ValueError(
            f"DeepSeek-V4.1 fused replay rows {rows.tolist()} outside a batch "
            f"of {len(slots)}"
        )
    if (
        not np.array_equal(fused.slots, slots[rows])
        or not np.array_equal(fused.ends, prefix[rows])
        or [rids[i] for i in rows] != list(fused.rids)
    ):
        raise ValueError(
            "DeepSeek-V4.1 fused replay does not describe this forward: "
            f"rows={rows.tolist()} slots={fused.slots.tolist()} vs "
            f"{slots[rows].tolist()}, ends={fused.ends.tolist()} vs "
            f"prefix {prefix[rows].tolist()}"
        )
    if np.any(fused.ends % block_size) or np.any(fused.starts % 2):
        raise ValueError(
            "DeepSeek-V4.1 fused replay spans must start even and end on a "
            f"PAGE boundary: starts={fused.starts.tolist()} "
            f"ends={fused.ends.tolist()}"
        )
    width = int(fused.histories.shape[1])
    if history is not None and width != int(history):
        raise ValueError(
            f"DeepSeek-V4.1 fused replay history width {width} != {int(history)}"
        )
    replaying = {}
    for k, i in enumerate(rows):
        start = int(fused.starts[k])
        # Left-padded as SGLang pads; native carries only the real tokens.
        keep = min(width, start)
        replaying[int(i)] = Replay(
            until=int(fused.ends[k]),
            rebuilds=True,
            history_ids=tuple(int(t) for t in fused.histories[k, width - keep :]),
            start=start,
        )
    return replaying


def build_v41_scheduled_batch(
    forward_batch, block_size: int, speculative_tokens=0, history=None
):
    """ATOM's scheduled-batch protocol for one SGLang ``ForwardBatch``.

    ``DECODE``, ``EXTEND`` (the eager runner turns ``MIXED`` into ``EXTEND``)
    and, on a speculative pool (``speculative_tokens`` drafts), DSpark's
    ``TARGET_VERIFY`` are translated; every other mode is refused. A radix
    replay side forward (``encoder_swa_replay``) carries its rows'
    ``replaying``; ``history``: the Engram history width it must carry. A
    step carrying a fused radix replay (``deepseek_v41_radix.FUSED_ATTR``)
    schedules each hit from ``P - R``: its span gains the replay rows
    (``replay_rows``), the same rows ``expand_v41_forward_inputs`` prepends
    to its inputs.
    """
    mode = forward_batch.forward_mode
    verify = _is_target_verify(mode)
    if not (mode.is_decode() or verify or mode.is_context_parallel_extend()):
        raise NotImplementedError(
            f"DeepSeek-V4.1 SGLang bridge does not translate {mode} yet"
        )
    if getattr(forward_batch, "rids", None) is None:
        # SGLang's `_dummy_run` (flashinfer autotune, graph warmup) builds
        # batches over its own buffers with no request ids and repeated
        # req_pool_idx 0; staging them would reset live STATE slots.
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang bridge does not run synthetic (dummy) batches"
        )
    bs = int(forward_batch.batch_size)
    ends = _cpu_ints(
        getattr(forward_batch, "seq_lens_cpu", None), forward_batch.seq_lens, bs
    )
    host_slots = getattr(forward_batch, "req_pool_indices_cpu", None)
    if host_slots is None:
        # A decode's / verify's ScheduleBatch mirror (`_install_host_mirror_patch`).
        host_slots = getattr(forward_batch, "atom_v41_req_pool_indices_cpu", None)
    late_slots = None
    if verify:
        prefix, width, ends, late_slots = _verify_lengths(
            forward_batch, ends, bs, speculative_tokens, host_slots
        )
        lengths = np.full(bs, width, dtype=np.int64)
    elif mode.is_decode():
        # positions = seq_lens - 1: seq_lens already counts the decode token.
        lengths = np.ones(bs, dtype=np.int64)
        prefix = ends - 1
    else:
        lengths = _cpu_ints(
            getattr(forward_batch, "extend_seq_lens_cpu", None),
            getattr(forward_batch, "extend_seq_lens", None),
            bs,
        )
        prefix = _cpu_ints(
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            getattr(forward_batch, "extend_prefix_lens", None),
            bs,
        )
    if np.any(prefix + lengths != ends):
        raise ValueError(
            "DeepSeek-V4.1 forward batch lengths disagree: "
            f"prefix={prefix.tolist()} lengths={lengths.tolist()} "
            f"ends={ends.tolist()}"
        )
    # Slots and PAGE rows in one device read (the eager path reads the rows
    # anyway), or, for an EXTEND, off the copy its allocation queued; the host
    # mirror, when present, must name the same slots. A verify's pages
    # include the rows SGLang allocated for it at p..p+5. A fused radix
    # replay's rows sit below P, inside these pages (SGLang's allocation
    # writes a hit's prefix pages into the row with the extend's own).
    widths = -(-ends // block_size)
    rids = list(forward_batch.rids[:bs])
    rows = None
    if not verify and not mode.is_decode():
        # The batch's own allocation queued these rows' copy on the
        # scheduler's stream (`ExtendRowRead`); taken, or dropped, here.
        early = None
        if host_slots is not None:
            early = _EXTEND_ROWS.take(_cpu_ints(host_slots, None, bs), rids, widths)
        else:
            _EXTEND_ROWS.drop()
        if early is not None:
            rows = early
            slots = _cpu_ints(host_slots, None, bs).astype(np.int32)
            _refuse_shared_or_padding_slots(slots, bs)
            count_v41_dspark("extend_rows_early")
    if rows is None or _CHECK_HOST_ROWS:
        read_slots, read_rows = _read_replay_rows(
            forward_batch, _req_to_token(), bs, widths, block_size
        )
        if rows is not None and (
            not np.array_equal(np.asarray(read_slots, np.int32), slots)
            or any(not np.array_equal(a, b) for a, b in zip(rows, read_rows))
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 EXTEND rows read at allocation disagree with the "
                f"device: slots {slots.tolist()} vs {read_slots.tolist()}"
            )
        if rows is None and not verify and not mode.is_decode():
            count_v41_dspark("extend_rows_read")
        slots, rows = read_slots, read_rows
    slots = np.asarray(slots, dtype=np.int32)
    _require_late_slots(late_slots, slots)
    if host_slots is not None and not np.array_equal(
        _cpu_ints(host_slots, None, bs).astype(np.int32), slots
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 host request slots disagree with the device: "
            f"{_cpu_ints(host_slots, None, bs).tolist()} vs {slots.tolist()}"
        )
    from atom.plugin.sglang.deepseek_v41_radix import FUSED_ATTR

    fused = getattr(forward_batch, FUSED_ATTR, None)
    replay_rows = np.zeros(bs, dtype=np.int64)
    if getattr(forward_batch, "encoder_swa_replay", False):
        if fused is not None:
            raise ValueError("DeepSeek-V4.1 radix side forward carries a fused replay")
        replaying = _replaying(forward_batch, prefix, ends, bs, block_size, history)
    else:
        replaying = {}
        if fused is not None and getattr(mode, "name", None) != "EXTEND":
            raise NotImplementedError(
                f"DeepSeek-V4.1 fused radix replay rides on an EXTEND, not {mode}"
            )
        if not (mode.is_decode() or verify):
            _check_radix_resumes(slots, prefix, fused)
        if fused is not None:
            replaying = _fused_replaying(
                fused, prefix, slots, rids, block_size, history
            )
            _stamp_fused_hits(fused)
            # Each hit's span now starts at P - R; context_lens stay.
            replay_rows = fused.replay_rows(bs)
            lengths = lengths + replay_rows
    # Graph replays stage off these copies (`HostPageRows`); a verify's rows
    # are valid copies too (over-allocated pages stay until release).
    _HOST_ROWS.record(slots, rids, rows)
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=rids,
        num_scheduled_tokens=lengths.astype(np.int32),
        context_lens=ends,
        # SGLang's ForwardBatch does not say whether a chunk finishes its
        # prompt, so every request counts as finishing (None): under decoder
        # bounded replay a chunk SGLang continues runs the decoder on its last
        # window rows too, whose ring the next chunk then reads if it is
        # shorter than the window (`v41_extend_replays`). A radix replay side
        # forward (``replaying``) is prefill-only, so its decoder computes
        # every row of ``[start, P)``: the ring rows below the hit that a main
        # extend shorter than the window reads, as the next chunk does above.
        # A fused replay is one span from P - R, at least the window long.
        is_final_chunk=None,
        replaying=replaying,
        # Rows ATOM runs ahead of each request's own (a fused radix replay).
        replay_rows=replay_rows,
        state_slots_committed=[int(s) for s in slots],
        block_tables=rows,
        total_seqs_num=bs,
        total_tokens_num=int(lengths.sum()),
    )


def v41_extend_replays(
    forward_batch, batch, window: int, *, dspark_context_in_forward: bool = False
) -> bool:
    """Whether this EXTEND may run V4.1's decoder on each request's last
    ``window`` rows only (decoder SWA bounded replay), which leaves every
    output row but each request's last zero.

    No where SGLang reads other rows than each request's last -- input
    logprobs, a FULL hidden-state capture, a prefill-only (scoring) batch --
    and where a request resumes its prompt (a prefix) with fewer than
    ``window`` rows: native backs such a forward off to start ``window`` rows
    before its end (`BlockManager.replay_start`), which SGLang's chunk cannot,
    and a tail of fewer rows would truncate their window at the chunk start.
    That covers a MIXED batch's decode rows too. Such a step computes every
    row, reading the decoder rings the previous chunk's tail wrote (every
    chunk counts as finishing, `build_v41_scheduled_batch`).

    A fused radix replay's rows (``batch.replay_rows``) are ATOM's: they
    count toward the tail (the span ATOM runs starts at ``P - R``, at least
    the window long) but not toward SGLang's logprob rows.

    ``dspark_context_in_forward``: a DSpark target, whose FULL capture feeds
    only SGLang's context injection, a checked no-op on this path
    (``write_target_hidden_kv``): the draft context is written inside the
    forward, from the tail's rows under a replay
    (`V41DSparkTargetSurface.finish_forward`), as native's drafter does.
    With a fused radix hit the tail is the last ``window`` rows of the
    widened span ``[P - R, L)`` (``R >= window``), i.e. ``[L - window, L)``,
    exactly the context the first draft block reads; the replay rows below
    it are not needed by any draft block. A radix side forward is
    prefill-only, so it never replays and writes every row of ``[P - R, P)``,
    and its main extend replays only when its own rows reach the window."""
    if getattr(forward_batch, "is_prefill_only", False):
        return False
    capture = getattr(forward_batch, "capture_hidden_mode", None)
    is_full = getattr(capture, "is_full", None)
    if is_full is not None and is_full() and not dspark_context_in_forward:
        return False
    lengths = np.asarray(batch.num_scheduled_tokens, np.int64)
    if getattr(forward_batch, "return_logprob", False):
        # As SGLang's `LogitsMetadata.from_forward_batch` decides it, over
        # SGLang's own rows.
        starts = getattr(forward_batch, "extend_logprob_start_lens_cpu", None)
        if starts is None:
            return False
        own = lengths - np.asarray(
            getattr(batch, "replay_rows", np.zeros_like(lengths)), np.int64
        )
        if np.any(own - np.asarray(starts[: len(own)], np.int64) > 0):
            return False
    prefix = np.asarray(batch.context_lens, np.int64) - lengths
    return not bool(np.any((prefix > 0) & (lengths < int(window))))


def build_deepseek_v41_forward_metadata(forward_batch, positions):
    """Stage one CSA2 step for this forward and return ATOM's metadata.

    Runs ATOM's ``_prepare`` (spans, compress plans, ``begin_step``) and
    ``prepare_model_inputs`` (slot resets, Engram, cursor advance) once, before
    the model, and marks the metadata ``staged_outside_forward`` so the
    model's own ``v41_begin_forward`` does not stage it again.

    A TARGET_VERIFY is staged tentatively (``max_q_len`` = the verify width):
    the cursor is not advanced, every prefix the sampler may accept is kept in
    ``cache.pending``, and `commit_deepseek_v41_dspark_verify` writes the
    accepted one after SGLang's accept. Not `prepare_decode`: its start
    positions come from native's token processor, which SGLang does not run.
    """
    mode = forward_batch.forward_mode
    if mode.is_idle():
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang bridge does not run IDLE batches yet "
            "(DP attention is not enabled for this model)"
        )
    builder = get_deepseek_v41_builder()
    if builder.cache is None:
        raise RuntimeError("DeepSeek-V4.1 proxy KV pool is not initialized")
    speculative_tokens = int(builder.geometry.speculative_tokens)
    if mode.is_decode() and speculative_tokens:
        raise NotImplementedError(
            "DeepSeek-V4.1 with DSpark runs no plain DECODE: every decode "
            "round is a TARGET_VERIFY"
        )
    batch = build_v41_scheduled_batch(
        forward_batch,
        builder.block_size,
        speculative_tokens=speculative_tokens,
        history=getattr(getattr(builder, "geometry", None), "history_size", None),
    )
    input_ids = forward_batch.input_ids
    from atom.plugin.sglang.deepseek_v41_radix import FUSED_ATTR

    fused = getattr(forward_batch, FUSED_ATTR, None)
    if fused is not None:
        # The inputs ATOM executes (`SGLangPluginRuntime`, through the
        # adapter's `expand_forward_inputs`): each hit's [P - R, P) first.
        if fused.expanded is None:
            raise RuntimeError(
                "DeepSeek-V4.1 fused radix replay reached the bridge without "
                "its widened inputs"
            )
        input_ids = fused.expanded.input_ids
    running_tokens = int(input_ids.shape[0])
    if running_tokens != int(positions.shape[0]):
        raise ValueError(
            f"DeepSeek-V4.1 input_ids rows {running_tokens} != positions rows "
            f"{int(positions.shape[0])}"
        )
    running_bs = batch.total_seqs_num
    if running_bs > builder.max_bs or running_tokens > builder.max_num_batched_tokens:
        raise ValueError(
            f"DeepSeek-V4.1 step of {running_bs} requests / {running_tokens} "
            f"tokens exceeds the builder's {builder.max_bs} / "
            f"{builder.max_num_batched_tokens}; align --max-running-requests "
            "and --chunked-prefill-size / --max-prefill-tokens"
        )
    verify = _is_target_verify(mode)
    if verify:
        width = int(batch.num_scheduled_tokens[0])
        if running_tokens != running_bs * width:
            raise ValueError(
                f"DeepSeek-V4.1 TARGET_VERIFY has {running_tokens} rows for "
                f"{running_bs} requests x {width}"
            )
        metadata, step_positions = builder._prepare(
            batch, running_bs, running_tokens, max_q_len=width, tentative=True
        )
    elif mode.is_decode():
        metadata, step_positions = builder.prepare_decode(
            batch, running_bs, running_tokens, 1
        )
    else:
        if getattr(builder, "_decoder_replays", False):
            batch.decoder_replays = v41_extend_replays(
                forward_batch,
                batch,
                builder.geometry.window_size,
                # Only DSpark speculates on V4.1 here (`v41_dspark_capability_error`).
                dspark_context_in_forward=speculative_tokens > 0,
            )
        bucket = prefill_token_graph_bucket(running_tokens)
        if bucket is None:
            metadata, step_positions = builder.prepare_prefill(
                batch, running_bs, running_tokens
            )
        else:
            # Padded to the recorded bucket, as native pads a prefill
            # (`ForwardMode._pad_to_token_graph`): the builder lays the spans
            # out and publishes positions out to ``bucket`` rows; the ids
            # past the scheduled ones are zero.
            metadata, step_positions = builder.prepare_prefill(
                batch, running_bs, bucket
            )
            input_ids = _pad_prefill_ids(input_ids, bucket)
            metadata.sglang_padded_inputs = PaddedPrefillInputs(
                input_ids=input_ids,
                positions=step_positions[:bucket],
                scheduled_tokens=running_tokens,
                scheduled_tokens_gpu=_PREFILL_GRAPHS.scheduled_tokens(running_tokens),
            )
            count_v41_dspark("prefill_token_graph")
    builder.prepare_model_inputs(input_ids, metadata)
    pending = getattr(getattr(metadata, "cache", None), "pending", None)
    if verify:
        if pending is None or pending.step is not metadata.step:
            raise RuntimeError("DeepSeek-V4.1 TARGET_VERIFY staged no tentative state")
        if pending.step.scheduled_bs != running_bs:
            raise RuntimeError(
                f"DeepSeek-V4.1 tentative step holds {pending.step.scheduled_bs} "
                f"requests, the verify {running_bs}"
            )
        count_v41_dspark("verify_eager")
    elif pending is not None:
        raise RuntimeError(f"DeepSeek-V4.1 {mode} left a tentative step behind")
    # The Engram staging too: the cursor advance rides on it, and the
    # forward's own stage/join is skipped for a step staged here.
    rows = getattr(metadata, "engram_embeddings", None)
    stage = getattr(rows, "stage", None)
    if stage is not None:
        stage()
        join = getattr(rows, "join", None)
        if join is not None:
            join()
    metadata.staged_outside_forward = True
    if _CHECK_POSITIONS:
        n = batch.total_tokens_num
        if not torch.equal(step_positions[:n], positions[:n].to(step_positions.dtype)):
            raise ValueError(
                "DeepSeek-V4.1 staged positions disagree with SGLang's positions"
            )
    return metadata


# ---------------------------------------------------------------------------
# Prefill token graphs
# ---------------------------------------------------------------------------
# Native ATOM records the runtime model's token-graph unit (`_EarlyLayers`
# under decoder bounded replay, `_Backbone` without it) once per prefill token
# bucket (`ATOM_TOKEN_GRAPH_SIZES`) and replays it on every prefill padded to
# the bucket that holds it, its eager regions (attention, Engram staging)
# running between the recorded segments on the live step
# (`atom/utils/region_graph.py`). The plugin records the same unit at the same
# buckets after SGLang's own graph capture (`record_deepseek_v41_prefill_
# token_graphs`) and pads an EXTEND to its bucket here; the forward context
# then replays (`ForwardContext.tr_graph`), and SGLang sees the scheduled rows
# only (`SGLangPluginRuntime.trim_output`). Larger prefills run eagerly.

# Kill switch: "0" records no prefill token graph; every EXTEND runs eagerly.
V41_PREFILL_TOKEN_GRAPHS_ENV = "ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS"


def v41_prefill_token_graphs_enabled() -> bool:
    return os.environ.get(V41_PREFILL_TOKEN_GRAPHS_ENV, "1") != "0"


@dataclass
class PaddedPrefillInputs:
    """A prefill padded to a token-graph bucket: the ids and positions the
    model runs (``bucket`` rows), the rows SGLang scheduled, and their count
    on the device (`Context.scheduled_tokens_gpu`, which the recording reads
    in place)."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    scheduled_tokens: int
    scheduled_tokens_gpu: torch.Tensor


class _PrefillTokenGraphs:
    """What the recording left for the serving prefills: its buckets, the ids
    buffer every padded prefill stages into, and the device row count."""

    def __init__(self) -> None:
        self.sizes: tuple[int, ...] = ()
        self.ids: torch.Tensor | None = None
        self._scheduled: torch.Tensor | None = None

    def ids_buffer(self, rows: int, device) -> torch.Tensor:
        ids = self.ids
        if ids is None or int(ids.shape[0]) < rows:
            # SGLang's ids are int64 (`flatten_arrays_to_pinned_cpu`).
            ids = torch.zeros(rows, dtype=torch.int64, device=device)
            self.ids = ids
        return ids

    def scheduled_tokens(self, tokens: int) -> torch.Tensor:
        """`[1]` int32 on the device holding ``tokens``, written in stream
        order (a fill, not a copy: nothing for the host to keep alive)."""
        if self._scheduled is None:
            self._scheduled = torch.zeros(1, dtype=torch.int32, device=self.ids.device)
        self._scheduled.fill_(int(tokens))
        return self._scheduled


_PREFILL_GRAPHS = _PrefillTokenGraphs()


def prefill_token_graph_bucket(tokens: int) -> int | None:
    """The recorded bucket an EXTEND of ``tokens`` rows pads to, or None
    (none recorded, or none holds it)."""
    if not _PREFILL_GRAPHS.sizes:
        return None
    from atom.utils.region_graph import TokenGraph

    bucket = TokenGraph.bucket(int(tokens), "prefill")
    if bucket is None or bucket not in _PREFILL_GRAPHS.sizes:
        return None
    return bucket


def _pad_prefill_ids(input_ids: torch.Tensor, bucket: int) -> torch.Tensor:
    """``input_ids`` in the leading rows of the padded-prefill ids buffer,
    zeros after them, ``bucket`` rows."""
    tokens = int(input_ids.shape[0])
    ids = _PREFILL_GRAPHS.ids_buffer(bucket, input_ids.device)[:bucket]
    ids[:tokens].copy_(input_ids)
    if bucket > tokens:
        ids[tokens:].zero_()
    return ids


def record_deepseek_v41_prefill_token_graphs(model_runner) -> tuple[int, ...]:
    """Record the target's prefill token graphs, every rank at once, after
    SGLang's graph capture: per bucket of ``ATOM_TOKEN_GRAPH_SIZES`` that a
    chunk can fill (largest first: a module's buckets share one input buffer
    and one pool), a fresh capture step (`build_for_sglang_token_graph_
    capture`), one warmup forward and one recording forward, under aiter's
    ``graph_capture`` (a capture stream; its custom all-reduce armed for
    capture), as native's `ModelRunner._record_tr_graphs`. Returns the
    buckets recorded."""
    from aiter.dist.parallel_state import graph_capture

    from atom.plugin.sglang.runtime.context import plugin_runtime_scope
    from atom.utils import envs
    from atom.utils.forward_context import (
        DPMetadata,
        get_forward_context,
        reset_forward_context,
        set_forward_context,
    )
    from atom.utils.region_graph import TokenGraph

    if not v41_prefill_token_graphs_enabled():
        logger.info(
            "ATOM DeepSeek-V4.1: prefills run eagerly (%s=0)",
            V41_PREFILL_TOKEN_GRAPHS_ENV,
        )
        return ()
    builder = _graph_builder()
    if not getattr(builder, "supports_token_graph", False):
        return ()
    wrapper = model_runner.model
    atom_config = wrapper.atom_config
    if getattr(atom_config, "enable_dp_attention", False) or getattr(
        atom_config, "enable_tbo", False
    ):
        # A padded bucket is the DP group's decision (`_pad_to_token_graph`),
        # and a TBO prefill splits into micro-batches: neither is wired here.
        logger.info("ATOM DeepSeek-V4.1: no prefill token graphs under DP / TBO")
        return ()
    pool = model_runner.token_to_kv_pool
    limit = min(
        int(builder.max_num_batched_tokens),
        int(getattr(model_runner.server_args, "chunked_prefill_size", 0) or 0)
        or int(builder.max_num_batched_tokens),
    )
    sizes = sorted(
        {int(n) for n in envs.ATOM_TOKEN_GRAPH_SIZES if 0 < int(n) <= limit},
        reverse=True,
    )
    if not sizes:
        return ()
    device = builder.cache.backing.device
    ids = _PREFILL_GRAPHS.ids_buffer(sizes[0], device)
    surface = getattr(wrapper, "_v41_dspark", None)
    parallel = atom_config.parallel_config
    start = time.perf_counter()
    with plugin_runtime_scope(framework="sglang", atom_config=atom_config):
        try:
            with graph_capture():
                for tokens in sizes:
                    for mode in (None, "record"):
                        metadata, context = (
                            builder.build_for_sglang_token_graph_capture(
                                tokens, input_ids=ids, slots=pool.capture_slots
                            )
                        )
                        context.scheduled_tokens_gpu = _PREFILL_GRAPHS.scheduled_tokens(
                            tokens
                        )
                        set_forward_context(
                            attn_metadata=metadata,
                            atom_config=atom_config,
                            context=context,
                            running_tokens_per_dp_rank=(
                                DPMetadata.gather_running_tokens(parallel, tokens)
                            ),
                            in_hipgraph=True,
                        )
                        get_forward_context().tr_graph = mode
                        if surface is not None:
                            # The DSpark taps fire in the decoder; nothing
                            # takes them from a recording forward.
                            surface._rows.clear()
                        wrapper.model(
                            **wrapper._filter_model_forward_kwargs(
                                {
                                    "input_ids": ids[:tokens],
                                    "positions": context.positions,
                                    "intermediate_tensors": None,
                                    "inputs_embeds": None,
                                }
                            )
                        )
                        if surface is not None:
                            surface._rows.clear()
                    torch.cuda.synchronize()
        finally:
            reset_forward_context()
    recorded = tuple(n for n in TokenGraph.captured_sizes("prefill") if n in set(sizes))
    if recorded != tuple(sorted(sizes)):
        raise RuntimeError(
            f"DeepSeek-V4.1 prefill token graphs recorded {recorded}, not "
            f"{sorted(sizes)}"
        )
    _PREFILL_GRAPHS.sizes = recorded
    torch.cuda.empty_cache()
    logger.info(
        "ATOM DeepSeek-V4.1: prefill token graphs recorded for buckets %s in "
        "%.1f s (%s=0 to disable)",
        list(recorded),
        time.perf_counter() - start,
        V41_PREFILL_TOKEN_GRAPHS_ENV,
    )
    return recorded


# ---------------------------------------------------------------------------
# Decode CUDA graphs
# ---------------------------------------------------------------------------
# SGLang captures one full decode graph per batch-size bucket and replays it
# with no model Python. Every step's ATOM metadata is staged on the host by the
# backend's `init_forward_metadata_out_graph`, outside the captured region, into
# the builder's fixed-address buffers; Engram's stage / gather / cursor advance
# are recorded inside the graph, as native ATOM records them. Prefill, mixed
# and larger-than-bucket decode batches keep the eager path above.

# The forward batch attribute a capture's metadata rides on into the forward.
GRAPH_METADATA_ATTR = "atom_v41_graph_metadata"


def _graph_builder():
    builder = get_deepseek_v41_builder()
    if builder.cache is None:
        raise RuntimeError("DeepSeek-V4.1 proxy KV pool is not initialized")
    return builder


def check_deepseek_v41_graph_capacity(
    pool, req_to_token: torch.Tensor, max_bs: int, max_num_tokens: int
):
    """Refuse a decode-graph runner the builder and pool cannot serve."""
    builder = _graph_builder()
    if builder.cache.backing.data_ptr() != pool.raw_arena.data_ptr():
        raise RuntimeError(
            "DeepSeek-V4.1 builder cache is not backed by the SGLang proxy pool"
        )
    engram = getattr(builder, "engram", None)
    if engram is not None and engram.host.overlap is None:
        # The host-lookup Engram path stages rows and advances the cursor on
        # the host each step; only the device staging is recorded in a graph.
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang decode graphs need the device Engram "
            "staging (UVA overlap); launch with --cuda-graph-backend-decode "
            "disabled"
        )
    if getattr(builder.score_workspace, "packed", False):
        # Packed logits plan a step in score bands on the host; a captured
        # graph replays one band (native gates this with `step_needs_eager`).
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang decode graphs need the plane score "
            "workspace (FP8 index cache); packed logits are not supported"
        )
    max_bs, max_num_tokens = int(max_bs), int(max_num_tokens)
    if max_bs > builder.max_bs or max_num_tokens > builder.max_num_batched_tokens:
        raise ValueError(
            f"DeepSeek-V4.1 decode graphs of {max_bs} requests / {max_num_tokens} "
            f"tokens exceed the builder's {builder.max_bs} / "
            f"{builder.max_num_batched_tokens}; align --max-running-requests "
            "with the cuda graph batch sizes"
        )
    if max_bs > int(pool.num_capture_slots):
        raise ValueError(
            f"DeepSeek-V4.1 decode graphs of {max_bs} requests exceed the pool's "
            f"{pool.num_capture_slots} capture slots"
        )
    rows = int(req_to_token.shape[0])
    if int(pool.capture_slot_base) < rows:
        raise ValueError(
            f"DeepSeek-V4.1 capture slots start at {pool.capture_slot_base}, "
            f"inside the {rows} req_to_token rows a request may hold"
        )
    return builder


def stage_deepseek_v41_graph_capture(forward_batch, pool):
    """Stage the step a graph bucket is recorded on and hand it to the
    forward on ``forward_batch`` (the object SGLang passes to both warmups and
    the capture). Runs outside the captured region.

    A plain pool records DECODE; a DSpark pool records only the target's
    TARGET_VERIFY of ``spec_info.draft_token_num`` rows a request."""
    mode = forward_batch.forward_mode
    builder = _graph_builder()
    speculative = int(builder.geometry.speculative_tokens or 0)
    if speculative:
        if not _is_target_verify(mode):
            raise NotImplementedError(
                "DeepSeek-V4.1 DSpark target graphs capture TARGET_VERIFY only, "
                f"not {getattr(mode, 'name', mode)}"
            )
        width = int(getattr(forward_batch.spec_info, "draft_token_num", 0) or 0)
        if width != speculative + 1:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark verify capture width {width} != the "
                f"pool's {speculative} drafts + 1"
            )
    elif not mode.is_decode():
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang graphs capture DECODE only, not "
            f"{getattr(mode, 'name', mode)}"
        )
    else:
        width = 1
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DeepSeek-V4.1 graph step must be staged before capture")
    bs = int(forward_batch.batch_size)
    if bs > len(pool.capture_slots):
        raise ValueError(
            f"DeepSeek-V4.1 capture of {bs} requests exceeds the pool's "
            f"{len(pool.capture_slots)} capture slots"
        )
    if int(forward_batch.input_ids.shape[0]) != bs * width:
        raise ValueError(
            f"DeepSeek-V4.1 capture of {bs} requests x {width} over "
            f"{int(forward_batch.input_ids.shape[0])} input rows"
        )
    kwargs = {} if width == 1 else {"width": width}
    metadata = builder.build_for_sglang_capture(
        bs, input_ids=forward_batch.input_ids, slots=pool.capture_slots[:bs], **kwargs
    )
    setattr(forward_batch, GRAPH_METADATA_ATTR, metadata)
    return metadata


def _read_replay_rows(fb_view, req_to_token, raw_bs, widths, block_size):
    """The live rows' slots and PAGE rows, read back from the device: one D2H,
    a sync on the forward stream."""
    pages = int(widths.max())
    indices = fb_view.req_pool_indices[:raw_bs]
    host = (
        torch.cat(
            (
                indices[:, None].to(torch.int32),
                (
                    req_to_token[indices, : pages * block_size : block_size]
                    // block_size
                ).to(torch.int32),
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    slots, table = host[:, 0], host[:, 1:]
    _refuse_shared_or_padding_slots(slots, raw_bs)
    rows = [table[i, : widths[i]].copy() for i in range(raw_bs)]
    _refuse_padding_page(rows)
    return slots, rows


def _refuse_shared_or_padding_slots(slots, count: int) -> None:
    if np.unique(slots).size != count or np.any(slots <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 needs one live req_pool_idx per request; got {slots}"
        )


def build_v41_graph_replay_batch(
    fb_view, req_to_token, block_size: int, width=1, ends=None
):
    """ATOM's scheduled batch for the live rows of a padded graph replay.

    The lengths come off SGLang's host seq_lens mirror (for a verify: widened
    by its ``width`` rows, ``p + width``, so the block tables cover the pages
    of ``p..p+width-1``). When the forward batch carried them
    (`_install_host_mirror_patch`: a decode, or a DSpark verify), the slots
    come off its host ``req_pool_indices`` mirror and the PAGE rows off
    `HostPageRows`; else, or when a request entered a page those copies do
    not hold, one D2H reads the slots and rows back and refreshes the copies.
    The padding rows (``req_pool_idx`` 0, ``seq_len`` 1) are not requests:
    ATOM pads them itself up to the bucket width.
    """
    width = int(width)
    bs = int(fb_view.batch_size)
    raw_bs = bs - int(getattr(fb_view, "num_padding", 0) or 0)
    if not 1 <= raw_bs <= bs:
        raise ValueError(f"DeepSeek-V4.1 replay of {raw_bs} live rows in {bs}")
    if ends is None:
        lengths = fb_view.seq_lens_cpu
        if lengths is None:
            raise ValueError(
                "DeepSeek-V4.1 graph replay needs SGLang's host seq_lens mirror"
            )
        if torch.is_tensor(lengths):
            lengths = lengths[:raw_bs].numpy()
        ends = lengths[:raw_bs]
    ends = np.asarray(ends, dtype=np.int64)
    if ends.shape != (raw_bs,):
        raise ValueError(
            f"DeepSeek-V4.1 replay of {raw_bs} live rows got {ends.size} lengths"
        )
    if np.any(ends < width):
        raise ValueError(
            f"DeepSeek-V4.1 graph replay lengths {ends.tolist()} under {width} rows"
        )
    widths = -(-ends // block_size)
    host_slots = getattr(fb_view, "atom_v41_req_pool_indices_cpu", None)
    rids = getattr(fb_view, "atom_v41_rids", None)
    rows = None
    slots = None
    if host_slots is not None and rids is not None:
        if len(host_slots) != raw_bs or len(rids) != raw_bs:
            raise ValueError(
                f"DeepSeek-V4.1 replay of {raw_bs} live rows got "
                f"{len(host_slots)} host slots and {len(rids)} request ids"
            )
        slots = np.asarray(host_slots, dtype=np.int32)
        rids = list(rids)
        if host_page_rows_supported():
            rows = _HOST_ROWS.lookup(slots, rids, widths)
            if rows is None and _VERIFY_PREFETCH.take():
                rows = _HOST_ROWS.lookup(slots, rids, widths)
            if rows is None and width > 1:
                # A DSpark verify reading back behind its draft block.
                count_v41_dspark("rows_read_sync")
    if rows is None or _CHECK_HOST_ROWS:
        read_slots, read_rows = _read_replay_rows(
            fb_view, req_to_token, raw_bs, widths, block_size
        )
        if slots is not None and (
            not np.array_equal(read_slots, slots)
            or (
                rows is not None
                and any(not np.array_equal(a, b) for a, b in zip(rows, read_rows))
            )
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 host request rows disagree with the device: "
                f"slots {slots.tolist()} vs {read_slots.tolist()}"
            )
        slots, rows = read_slots, read_rows
        if rids is not None:
            _HOST_ROWS.record(slots, rids, rows)
    _refuse_shared_or_padding_slots(slots, raw_bs)
    slot_list = [int(slot) for slot in slots]
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=slot_list,
        num_scheduled_tokens=np.full(raw_bs, width, dtype=np.int32),
        context_lens=ends,
        is_final_chunk=None,
        replaying={},
        state_slots_committed=slot_list,
        block_tables=rows,
        total_seqs_num=raw_bs,
        total_tokens_num=raw_bs * width,
    )


def stage_deepseek_v41_graph_replay(fb_view, req_to_token):
    """Stage one decode replay into the addresses its bucket recorded.

    ``prepare_decode`` at the bucket width (capacity-shaped plans, padding
    sentinels), then ``prepare_model_inputs`` over SGLang's graph
    ``input_ids``: the deferred cursor probe and the Engram step, whose
    lookup and cursor advance the graph runs. No Engram ``stage`` / ``join``
    here and no ``staged_outside_forward``: no model Python runs on replay.
    """
    from atom.utils.forward_context import RoundService

    mode = getattr(fb_view, "actual_forward_mode", None) or fb_view.forward_mode
    if not mode.is_decode():
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang decode graphs do not replay "
            f"{getattr(mode, 'name', mode)}"
        )
    builder = _graph_builder()
    bs = int(fb_view.batch_size)
    if bs > builder.max_bs or int(fb_view.input_ids.shape[0]) != bs:
        raise ValueError(
            f"DeepSeek-V4.1 replay of {bs} requests over "
            f"{int(fb_view.input_ids.shape[0])} input rows does not fit the "
            f"builder's {builder.max_bs}"
        )
    batch = build_v41_graph_replay_batch(fb_view, req_to_token, builder.block_size)
    metadata, _ = builder.prepare_decode(batch, bs, bs, 1)
    builder.prepare_model_inputs(fb_view.input_ids, metadata)
    step = metadata.step
    if (
        step.running_tokens != bs
        or not step.decode
        or step.tentative
        or metadata.cache is not builder.cache
        or metadata.served_by_round != RoundService.NONE
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 decode replay staged a step its graph did not record"
        )
    return metadata


def stage_deepseek_v41_verify_replay(fb_view, req_to_token):
    """Stage one DSpark target-verify replay into its bucket's addresses.

    The eager verify's step (`build_deepseek_v41_forward_metadata`) at the
    bucket width: ``_prepare(max_q_len=width, tentative=True)`` over the
    live rows -- the same ``(running_bs, max_q_len)`` the capture used, so
    plans, indptrs and state slots have capacity shape and ATOM pads the
    tail (zero-length requests, slot 0) -- then ``prepare_model_inputs``
    over SGLang's graph ``input_ids``: the device Engram step, the deferred
    probe and ``cache.pending`` (staged on the device). No cursor advance:
    the accepted prefix is committed after the replay, once, from SGLang's
    ``commit_lens`` (`commit_deepseek_v41_dspark_verify`).

    Lengths: the committed prefix from ``spec_info.live_seq_lens_cpu`` and
    the ends from the widened ``seq_lens_cpu`` (both host mirrors), or, under
    the late host lengths, the prefix taken here from the round's deferred
    copy and the ends that prefix plus the width (`_committed_prefix`).
    """
    from atom.utils.forward_context import RoundService

    mode = getattr(fb_view, "actual_forward_mode", None) or fb_view.forward_mode
    if not _is_target_verify(mode):
        raise NotImplementedError(
            "DeepSeek-V4.1 DSpark verify graphs do not replay "
            f"{getattr(mode, 'name', mode)}"
        )
    builder = _graph_builder()
    speculative = int(builder.geometry.speculative_tokens or 0)
    spec_info = getattr(fb_view, "spec_info", None)
    width = int(getattr(spec_info, "draft_token_num", 0) or 0)
    if not speculative or width != speculative + 1:
        raise ValueError(
            f"DeepSeek-V4.1 verify replay of width {width} on a pool with "
            f"{speculative} drafts"
        )
    bs = int(fb_view.batch_size)
    raw_bs = bs - int(getattr(fb_view, "num_padding", 0) or 0)
    if bs > builder.max_bs or int(fb_view.input_ids.shape[0]) != bs * width:
        raise ValueError(
            f"DeepSeek-V4.1 verify replay of {bs} requests x {width} over "
            f"{int(fb_view.input_ids.shape[0])} input rows does not fit the "
            f"builder's {builder.max_bs}"
        )
    if not 1 <= raw_bs <= bs:
        raise ValueError(f"DeepSeek-V4.1 verify replay of {raw_bs} live rows in {bs}")
    if fb_view.seq_lens_cpu is None:
        raise ValueError(
            "DeepSeek-V4.1 verify replay needs SGLang's host seq_lens mirror"
        )
    prefix, ends, late_slots = _committed_prefix(
        spec_info,
        _cpu_ints(fb_view.seq_lens_cpu, None, raw_bs),
        raw_bs,
        width,
        getattr(fb_view, "atom_v41_req_pool_indices_cpu", None),
    )
    if np.any(prefix <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 verify replay needs a committed prefix; got {prefix}"
        )
    batch = build_v41_graph_replay_batch(
        fb_view, req_to_token, builder.block_size, width=width, ends=ends
    )
    _require_late_slots(late_slots, batch.state_slots_committed)
    metadata, positions = builder._prepare(
        batch, bs, bs * width, max_q_len=width, tentative=True
    )
    builder.prepare_model_inputs(fb_view.input_ids, metadata)
    step = metadata.step
    pending = getattr(builder.cache, "pending", None)
    if (
        not step.tentative
        or int(step.max_q_len) != width
        or int(step.running_tokens) != bs * width
        or int(step.scheduled_bs) != raw_bs
        or metadata.cache is not builder.cache
        or metadata.served_by_round != RoundService.NONE
        or getattr(step, "replay", None) is not None
        or getattr(step, "decoder", None) is not None
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 verify replay staged a step its graph did not record"
        )
    if pending is None or pending.step is not step or not pending.staged_on_device:
        raise RuntimeError(
            "DeepSeek-V4.1 verify replay left no device-staged tentative state"
        )
    if _CHECK_POSITIONS:
        n = raw_bs * width
        if not torch.equal(positions[:n], fb_view.positions[:n].to(positions.dtype)):
            raise ValueError(
                "DeepSeek-V4.1 verify replay positions disagree with SGLang's"
            )
    return metadata


# ---------------------------------------------------------------------------
# DSpark graph stages: the stage gate and the runtime counters
# ---------------------------------------------------------------------------
# SGLang has one decode-graph switch for both DSpark runners. The gate keeps
# one of them from replaying (it is still captured) so each stage can be
# validated alone; the counters are the runtime proof of which stage ran.

V41_DSPARK_GRAPH_STAGES_ENV = "ATOM_SGLANG_V41_DSPARK_GRAPH_STAGES"
V41_DSPARK_GRAPH_STAGES = ("verify", "draft")
# The stages a launch replays unless the env says otherwise: both (validated
# alone, E9 / E10, and together, E11).
V41_DSPARK_GRAPH_STAGES_DEFAULT = "verify,draft"
V41_DSPARK_COUNTER_NAMES = (
    "verify_graph",
    "verify_eager",
    "draft_graph",
    "draft_eager",
    "accept_folded",
    "accept_eager",
    "commits",
    # graph replays of a bucket wider than the batch (padding rows)
    "verify_padded",
    "draft_padded",
    # PAGE rows copied before the draft launch (`VerifyRowPrefetch`), and
    # verify replays that still read rows back behind the draft
    "rows_prefetched",
    "rows_read_sync",
    # EXTEND PAGE rows taken from the copy queued at allocation
    # (`ExtendRowRead`), and EXTENDs that read them back at staging
    "extend_rows_early",
    "extend_rows_read",
    # EXTENDs padded to a recorded prefill token-graph bucket
    "prefill_token_graph",
    # verifies whose staging took the round's host lengths
    # (`_take_late_prefix`)
    "late_seq_lens",
)
V41_DSPARK_COUNTERS = dict.fromkeys(V41_DSPARK_COUNTER_NAMES, 0)
_V41_DSPARK_COUNTERS_EXIT = False


def v41_dspark_graph_stages() -> frozenset:
    """The graph stages allowed to replay: ``verify,draft``, ``verify``,
    ``draft`` or ``none`` (default `V41_DSPARK_GRAPH_STAGES_DEFAULT`)."""
    raw = os.environ.get(V41_DSPARK_GRAPH_STAGES_ENV, V41_DSPARK_GRAPH_STAGES_DEFAULT)
    parts = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if parts == {"none"}:
        return frozenset()
    if not parts or not parts <= set(V41_DSPARK_GRAPH_STAGES):
        raise ValueError(
            f"{V41_DSPARK_GRAPH_STAGES_ENV}={raw!r}: expected a comma list of "
            f"{V41_DSPARK_GRAPH_STAGES} or 'none'"
        )
    return frozenset(parts)


def _v41_dspark_counter_every() -> int:
    return max(1, int(os.environ.get("ATOM_SGLANG_V41_DSPARK_COUNTER_EVERY", "1000")))


def log_v41_dspark_counters(reason: str) -> None:
    logger.info(
        "ATOM V4.1 DSpark graph counters (%s): %s",
        reason,
        " ".join(
            f"{name}={V41_DSPARK_COUNTERS[name]}" for name in V41_DSPARK_COUNTER_NAMES
        ),
    )


def count_v41_dspark(name: str, count: int = 1) -> None:
    """``count`` more ``name`` events on this rank (eager-side: a graph
    replay is counted where it is staged, never inside the graph)."""
    global _V41_DSPARK_COUNTERS_EXIT
    V41_DSPARK_COUNTERS[name] += count
    if not _V41_DSPARK_COUNTERS_EXIT:
        import atexit

        _V41_DSPARK_COUNTERS_EXIT = True
        atexit.register(log_v41_dspark_counters, "exit")
    if (
        name == "commits"
        and V41_DSPARK_COUNTERS[name] % _v41_dspark_counter_every() == 0
    ):
        log_v41_dspark_counters(f"every {_v41_dspark_counter_every()} commits")


def _v41_dspark_graph_role(runner):
    """``verify`` / ``draft`` for a decode-graph runner whose backend is this
    plugin's V4.1 shim on a DSpark pool, None for every other runner."""
    backend = getattr(runner, "attn_backend", None)
    kind = getattr(backend, "_v41_graph_kind", None)
    return kind if kind in V41_DSPARK_GRAPH_STAGES else None


def install_v41_dspark_graph_stage_gate() -> frozenset:
    """Mask the DSpark graph stages ``ATOM_SGLANG_V41_DSPARK_GRAPH_STAGES``
    leaves out: their runner's ``can_run_graph`` answers False, so the round
    runs eager (a masked draft also turns SGLang's folded proposal and the
    folded accept off; a masked verify forces the eager accept). Scoped to
    runners of the V4.1 shim on a DSpark pool; idempotent."""
    stages = v41_dspark_graph_stages()
    try:
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner
    except Exception:  # noqa: BLE001 - graphs are optional
        return stages
    cls = decode_cuda_graph_runner.DecodeCudaGraphRunner
    if getattr(cls, "_atom_v41_dspark_stage_gate", None) is not None:
        if cls._atom_v41_dspark_stage_gate != stages:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark graph stage gate installed for "
                f"{sorted(cls._atom_v41_dspark_stage_gate)}, now {sorted(stages)}"
            )
        return stages
    original = cls.can_run_graph

    @wraps(original)
    def can_run_graph(self, forward_batch):
        role = _v41_dspark_graph_role(self)
        if role is not None and role not in stages:
            return False
        return original(self, forward_batch)

    cls.can_run_graph = can_run_graph
    cls._atom_v41_dspark_stage_gate = stages
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark graph stages: replay %s (masked: %s; %s)",
        sorted(stages) or "none",
        sorted(set(V41_DSPARK_GRAPH_STAGES) - stages) or "none",
        V41_DSPARK_GRAPH_STAGES_ENV,
    )
    return stages


# ---------------------------------------------------------------------------
# DSpark: the accepted prefix, committed once per round
# ---------------------------------------------------------------------------


def dspark_commit_anchors(commit_lens: torch.Tensor, verify_width: int):
    """Each request's flat row of its last accepted verify input.

    A request's verify rows are ``[bonus, d1..d5]`` at ``i*W .. i*W+W-1``;
    ``commit_lens`` (SGLang's ``accept_lens``) counts the bonus token, so
    ``c`` accepted tokens end at row ``i*W + c - 1`` -- native's
    ``seg_start + num_accepted_drafts``.
    """
    bs = int(commit_lens.numel())
    return (
        torch.arange(bs, device=commit_lens.device, dtype=torch.int32)
        * int(verify_width)
        + commit_lens.reshape(-1).to(torch.int32)
        - 1
    )


_V41_DSPARK_TP_CHECK_LOGGED = False


def _sglang_tp_group():
    """SGLang's TP group. Resolution errors propagate: a check that cannot
    find its group must not switch itself off silently."""
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:  # no SGLang at all
        return None
    return get_parallel().tp_group


def _check_commit_lens(commit_lens: torch.Tensor, verify_width: int) -> None:
    """Bring-up: lengths in range and identical on every TP rank (the ranks'
    tentative states must take the same prefix)."""
    lens = commit_lens.reshape(-1).to(torch.int64)
    if lens.numel() and bool(((lens < 1) | (lens > verify_width)).any()):
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark commit lengths out of [1, {verify_width}]: "
            f"{lens.tolist()}"
        )
    group = _sglang_tp_group()
    if group is None or int(getattr(group, "world_size", 1)) <= 1:
        return
    gathered = group.all_gather(lens.view(1, -1).contiguous(), dim=0)
    global _V41_DSPARK_TP_CHECK_LOGGED
    if not _V41_DSPARK_TP_CHECK_LOGGED:
        _V41_DSPARK_TP_CHECK_LOGGED = True
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: TP commit-length agreement check active "
            "(tp world=%d)",
            int(group.world_size),
        )
    if not bool((gathered == gathered[:1]).all()):
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark TP ranks disagree on commit lengths: "
            f"{gathered.tolist()}"
        )


# Kill switch: "0" commits every verify inside its preparation's resumed H2D
# epoch, a device-staged one too (the cache's reuse event then lands after
# the verify, and the next forward's staging waits for it).
V41_DSPARK_COMMIT_UNSEALED_ENV = "ATOM_SGLANG_V41_DSPARK_COMMIT_UNSEALED"


def v41_dspark_commit_unsealed() -> bool:
    return os.environ.get(V41_DSPARK_COMMIT_UNSEALED_ENV, "1") != "0"


def commit_deepseek_v41_dspark_verify(
    pool, *, req_slots, commit_lens: torch.Tensor, verify_width: int
) -> None:
    """Commit this round's accepted prefix into ATOM's Engram cursor.

    Everything comes from the current round: the pool from the target runner
    that owns it, the tentative step the verify staged in that pool's cache
    (``cache.pending``), and SGLang's official ``commit_lens``. Exactly once:
    the commit clears ``pending``, and the next forward's ``begin_step``
    refuses to start while one is left (`require_committed`).
    """
    if not getattr(pool, "is_atom_v41_proxy_pool", False) or getattr(
        pool, "is_atom_v41_draft_proxy", False
    ):
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark commit needs the target's ATOM proxy pool, "
            f"got {type(pool).__name__}"
        )
    cache = pool.atom_cache
    pending = getattr(cache, "pending", None)
    if pending is None:
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark: the verify forward left no tentative step "
            "to commit"
        )
    step = pending.step
    bs = int(commit_lens.numel())
    width = int(verify_width)
    if step.scheduled_bs != bs or step.max_q_len != width:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark commit of {bs} requests x {width} does not "
            f"match the tentative step's {step.scheduled_bs} x {step.max_q_len}"
        )
    slots = [int(span.slot) for span in step.requests]
    if slots != [int(slot) for slot in req_slots]:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark commit for requests in slots {list(req_slots)} "
            f"but the verify staged {slots}"
        )
    if v41_dspark_checks_enabled():
        _check_commit_lens(commit_lens, width)
    anchors = dspark_commit_anchors(commit_lens, width).to(step.cu_seqlens_q.device)
    owner = getattr(cache, "_sglang_h2d_owner", None)
    if owner is None or (pending.staged_on_device and v41_dspark_commit_unsealed()):
        # A verify staged on the device (every graph replay) commits from
        # device rows alone: no host source of the cache's publication is
        # read, so the epoch its preparation sealed stays sealed, and the
        # cache's reuse event stays where that preparation recorded it --
        # before the verify. Resumed, `finish` would re-record it after the
        # verify, and the next forward's `begin` (an arriving EXTEND's
        # staging) would wait for the whole verify graph.
        cache.commit_tentative(step, anchors)
        return
    # Host-staged Engram prefixes were written in the epoch the verify's
    # preparation sealed; their H2D copy happens here, in that same epoch.
    with _v41_publication_epoch(owner, resume=True):
        cache.commit_tentative(step, anchors)


def install_deepseek_v41_dspark_commit_patch() -> None:
    """Commit the accepted prefix at SGLang's model-local verify-commit hook.

    ``DSparkWorkerV2._commit_target_mamba_states_after_verify`` runs once per
    non-idle decode round, after the accept (and its TP sync) and before the
    hidden commit and the next draft, with the round's official
    ``commit_lens``; SGLang provides it for model-local state (mamba). The
    wrapper keeps the original and adds the commit only for this plugin's
    V4.1 target pool. Idempotent.
    """
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    cls = dspark_worker_v2.DSparkWorkerV2
    if getattr(cls, "_atom_v41_dspark_commit_patched", False):
        return
    original = cls._commit_target_mamba_states_after_verify

    @wraps(original)
    def commit_after_verify(
        self, *, batch, seq_lens_pre_verify, seq_lens_post_verify, commit_lens
    ):
        original(
            self,
            batch=batch,
            seq_lens_pre_verify=seq_lens_pre_verify,
            seq_lens_post_verify=seq_lens_post_verify,
            commit_lens=commit_lens,
        )
        pool = self.target_worker.model_runner.token_to_kv_pool
        if getattr(pool, "is_atom_v41_proxy_pool", False):
            # Which accept produced these lengths: the verify graph's folded
            # epilogue hands out its own static buffer.
            epilogue = getattr(self, "_verify_epilogue", None)
            folded = getattr(epilogue, "commit_lens_buf", None)
            count_v41_dspark(
                "accept_folded"
                if folded is not None
                and commit_lens.numel()
                and commit_lens.data_ptr() == folded.data_ptr()
                else "accept_eager"
            )
            # ScheduleBatch's own CPU mirror of the request rows (kept in
            # step with filter/merge); SGLang 0.5.22 Req has no req_pool_idx.
            req_slots_cpu = batch.req_pool_indices_cpu
            if req_slots_cpu is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 DSpark commit needs ScheduleBatch."
                    "req_pool_indices_cpu"
                )
            commit_deepseek_v41_dspark_verify(
                pool,
                req_slots=req_slots_cpu.tolist(),
                commit_lens=commit_lens,
                verify_width=int(self.verify_num_draft_tokens),
            )
            count_v41_dspark("commits")

    cls._commit_target_mamba_states_after_verify = commit_after_verify
    cls._atom_v41_dspark_commit_patched = True
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: accepted-prefix commit installed on "
        "DSparkWorkerV2._commit_target_mamba_states_after_verify"
    )


def log_deepseek_v41_runtime_once(model: Any) -> None:
    """Startup proof: what actually got constructed for this rank."""
    if getattr(model, "_atom_v41_runtime_logged", False):
        return
    model._atom_v41_runtime_logged = True
    attn_classes = sorted(
        {type(getattr(layer, "attn", None)).__name__ for layer in model.layers}
    )
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang runtime: model=%s replay=%s attention=%s",
        type(model).__name__,
        getattr(model, "replay", None),
        attn_classes,
    )
