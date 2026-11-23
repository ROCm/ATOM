# SPDX-License-Identifier: MIT
"""Run DeepSeek-V4.1 (CSA2) on SGLang's plugin path against an ATOM-owned pool.

SGLang 0.5.22 renames V4.1 to ``DeepseekV4ForCausalLM`` (``model_type`` stays
``deepseek_v41``), forces ``page_size=256`` and builds a DSV4 pool.

SGLang owns scheduling, page allocation and sampling; ATOM owns the model and
the CSA2 pool: ``ATOMDeepSeekV41ProxyKVPool`` is one uint8 arena (PAGEs, then
STATE slots) sized by ``ATOMDeepSeekV41PoolConfigurator``. Each ``ForwardBatch``
is translated into ATOM's scheduled batch and staged by ATOM's own builder
once per step, outside the forward. Decode CUDA graphs are staged by the
backend shim outside the captured region (see "Decode CUDA graphs").

The translation relies on: SGLang page id == V4.1 PAGE id (block tables are
``req_to_token[req, ::256] // 256``; page 0 is SGLang's padding page); STATE
slot == ``req_pool_idx`` (unique among live requests, row 0 is padding; a
request starting at position 0 resets its slot; with radix cache on, a prefix
hit at ``P`` rebuilds its slot in the step's own forward, its span widened to
start at ``P - R`` (``deepseek_v41_radix``, translated into ATOM's
``ScheduledBatch.replaying``); the window lives in the STATE ring, so SGLang
gets a plain paged allocator (``needs_paged_swa_allocator = False``).

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
- There is no DRAFT_EXTEND and no draft DECODE: the draft block runs as a
  TARGET_VERIFY of width 5 on the draft runner.
- Draft blocks and verifies run eagerly: decode CUDA graphs are refused
  with DSpark for now. Prefill stays eager.

DSpark under the overlap schedule (SGLang 0.5.22 ``event_loop_overlap``):

- Overlap does not split a round. ``DSparkWorkerV2._forward_decode`` runs the
  whole round -- draft block, verify, accept, ``on_publish``, the commit hook,
  the hidden commit -- in one ``forward_batch_generation`` call on the
  scheduler thread under ``forward_stream``. What overlaps the GPU is the
  scheduler's own work: processing round t-1's result and forming round
  t+1's batch. Round t+1's ``run_batch`` then waits for round t's publish
  (``FutureMap.resolve_seq_lens_cpu``) before it stages anything, so a verify
  is never staged over an uncommitted one (``require_committed`` and
  ``begin_step`` refuse it if it ever were).
- The tentative step's publication epoch is opened by the verify's staging
  and resumed by its commit on the same thread and stream
  (``PublicationOwner.resume`` checks both); the next ``begin`` waits on the
  commit's completion event before the pinned sources are rewritten.
- Host lengths: the ScheduleBatch's ``seq_lens_cpu`` is seeded one round late
  by ``prepare_for_decode`` and overwritten with the published exact lengths
  (this backend's ``needs_cpu_seq_lens``) before the forward, so
  ``live_seq_lens_cpu`` is the committed prefix as without overlap.
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
import sys
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)

from atom.model_engine.sequence import Replay
from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang import deepseek_v41_radix as radix

logger = logging.getLogger("atom")

DEEPSEEK_V41_ARCH = "DeepseekV41ForCausalLM"
DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
# SGLang forces page_size 256 for DeepSeek-V4 off NPU (its model override);
# V4.1's PAGE follows it.
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
# This process's metadata builder (`_install_pool_sizing_patch`).
_V41_BUILDER = None


def is_deepseek_v41_config(config: Any) -> bool:
    """Whether a flat HF config is DeepSeek-V4.1. Every caller passes SGLang's
    normalized config (``deepseek_v41``, renamed to a V4 architecture) or
    ATOM's text config (``deepseek_v41_text``), so the model type decides.
    ``resolve_model_arch_spec`` asks this first for every plugin model and
    reads its config with ``getattr`` throughout; this check keeps that."""
    return getattr(config, "model_type", None) in DEEPSEEK_V41_MODEL_TYPES


def v41_dspark_capability_error(config: Any) -> str | None:
    """Why ``config`` is not the V4.1 + bundled DSpark checkpoint this bridge
    serves, or None when it is.

    The capability is read off the checkpoint's own contract rather than the
    family name: a five-wide block, the published target taps, a noise token,
    a Markov head, three draft stages, and those stages window-only (no
    compressor), which is the layout the draft rings 40-42 assume."""
    if not is_deepseek_v41_config(config):
        return "not a DeepSeek-V4.1 checkpoint"
    block = int(getattr(config, "dspark_block_size", 0) or 0)
    if not block:
        return "the checkpoint bundles no DSpark head (dspark_block_size)"
    if block != V41_DSPARK_DRAFT_TOKENS:
        return f"dspark_block_size {block} != {V41_DSPARK_DRAFT_TOKENS}"
    if not list(getattr(config, "dspark_target_layer_ids", None) or ()):
        return "no dspark_target_layer_ids"
    if getattr(config, "dspark_noise_token_id", None) is None:
        return "no dspark_noise_token_id"
    if int(getattr(config, "dspark_markov_rank", 0) or 0) <= 0:
        return "dspark_markov_rank is not positive"
    stages = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    if stages != V41_DSPARK_STAGES:
        return f"num_nextn_predict_layers {stages} != {V41_DSPARK_STAGES}"
    layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    ratios = [int(r) for r in (getattr(config, "compress_ratios", None) or ())]
    draft_ratios = ratios[layers : layers + stages]
    if len(draft_ratios) != stages or any(draft_ratios):
        return (
            "the draft stages are not window-only "
            f"(compress_ratios[{layers}:] = {draft_ratios})"
        )
    return None


def is_v41_dspark_atom_config(atom_config: Any) -> bool:
    """Whether an ATOM ``Config`` is the V4.1 target of a DSpark deployment
    (what `build_v41_dspark_speculative_config` produced)."""
    speculative = atom_config.speculative_config
    return (
        speculative is not None
        and speculative.method == "dspark"
        and is_deepseek_v41_config(atom_config.hf_config)
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
    from sglang.srt.runtime_context import get_spec

    # The algorithm is DSPARK: SGLang refuses any other for V4.1
    # (`validate_deepseek_v41_features`) while it resolves the server args,
    # and its DSpark resolution (`_handle_dspark`) sets the verify width and
    # the draft path or refuses the launch.
    spec = get_spec()
    reason = v41_dspark_capability_error(hf_config)
    if reason is not None:
        raise ValueError(f"DeepSeek-V4.1 DSpark is not servable: {reason}")
    verify_tokens = spec.speculative_num_draft_tokens
    if int(verify_tokens) != V41_DSPARK_VERIFY_TOKENS:
        raise ValueError(
            "DeepSeek-V4.1 DSpark verifies the bonus token and five drafts: "
            f"launch with --speculative-num-draft-tokens {V41_DSPARK_VERIFY_TOKENS}"
            f" (got {verify_tokens})"
        )
    model_path = server_args.model_path
    draft_path = spec.speculative_draft_model_path
    if draft_path != model_path:
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


def sglang_v41_kv_cache_dtype(kv_cache_dtype: str) -> str:
    """The V4.1 pool format for the operator's ``--kv-cache-dtype`` (the raw
    ServerArgs value). SGLang's DSV4 hook admits only ``bfloat16`` and
    ``fp8_e4m3`` and resolves ``auto`` to ``fp8_e4m3`` in its own bags, so
    ``auto`` is refused: the operator names the pool's format."""
    if kv_cache_dtype == "bfloat16":
        return "bf16"
    if kv_cache_dtype == "fp8_e4m3":
        return "fp8"
    resolved = " (SGLang resolves auto to fp8_e4m3 for DeepSeek-V4.1)"
    raise ValueError(
        "DeepSeek-V4.1 on the SGLang plugin needs --kv-cache-dtype bfloat16 "
        f"or fp8_e4m3, got {kv_cache_dtype!r}"
        + (resolved if kv_cache_dtype == "auto" else "")
    )


# Native's FP4PTPC_RECIPE in SGLang's spelling.
FP4PTPC_SGLANG_RECIPE = (
    "DeepSeek-V4.1 FP4PTPC needs its routed experts converted to per-channel "
    "FP8 (recipes/atom_sglang/DeepSeek-V4.1-Flash.md): "
    '--model-loader-extra-config \'{"online_quant_config": '
    '{"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}}\''
)


def refuse_fp4ptpc_without_online_quant(atom_config: Any) -> None:
    """Refuse FP4PTPC without its experts converted to FP8, as native does."""
    if atom_config.hf_config.fp4ptpc and atom_config.online_quant_config is None:
        raise ValueError(FP4PTPC_SGLANG_RECIPE)


def fp8_kv_attention_backend(atom_config: Any) -> str | None:
    """AITER's backend for the FP8 pool's sparse attention on this device, or
    None. It asks for the call the pool's attention makes
    (``fp8_attention.sparse_attn_v41_paged_2src``): ATOM's FP8 dtype for Q
    and the CSA / SWA planes, E8M0 scales, contiguous, this rank's heads (DP
    attention is refused). Native FP8 drafting asks the same function
    (``draft_kv.validate_fp8_draft_support``), with ``float8_e4m3fn``."""
    from aiter.ops.mla_sparse_prefill import get_mla_sparse_prefill_v41_backend

    from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE

    model = atom_config.hf_config
    return get_mla_sparse_prefill_v41_backend(
        torch.device("cuda"),
        (
            1,
            model.num_attention_heads // atom_config.tensor_parallel_size,
            model.head_dim,
        ),
        unified_kv_dtype=FP8_DTYPE,
        softmax_scale=model.head_dim**-0.5,
        q_dtype=FP8_DTYPE,
        q_scale_dtype=torch.float8_e8m0fnu,
        unified_kv_scale_dtype=torch.float8_e8m0fnu,
        kv_dtype=FP8_DTYPE,
        kv_scale_dtype=torch.float8_e8m0fnu,
        layout="contiguous",
    )


def refuse_fp8_kv_without_backend(atom_config: Any) -> None:
    """Refuse the FP8 pool where AITER cannot attend over it; the forward
    would only fail at the first decode-graph capture."""
    if atom_config.kv_cache_dtype != "fp8" or fp8_kv_attention_backend(atom_config):
        return
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    raise ValueError(
        "FP8 KV needs aiter's FP8 mla_sparse_prefill_v41 backend, which this "
        f"device ({arch.split(':', 1)[0]}) lacks; launch with --kv-cache-dtype "
        "bfloat16"
    )


# --- Launch contract ---


def sglang_radix_cache_enabled() -> bool:
    """Whether SGLang serves this launch with its radix (prefix) cache."""
    from sglang.srt.runtime_context import get_memory

    return not get_memory().disable_radix_cache


def _radix_refusals(memory, schedule) -> tuple:
    """What radix cache cannot run beside (``deepseek_v41_radix``).

    Disabled chunked prefill is refused for every launch, and SGLang refuses
    unified memory (the only KV relocation) for this model itself. A chunk
    always holds a hit's ``R`` replay rows (``R`` <= chunk): SGLang asserts
    ``--chunked-prefill-size`` is a multiple of the page size (256 for
    DeepSeek-V4) before the plugin runs."""
    return (
        # The replay seam serves EXTEND batches only.
        ("mixed chunk with radix cache", schedule.enable_mixed_chunk),
        (
            "session radix cache",
            memory.enable_session_radix_cache,
        ),
        (
            "a custom radix cache backend",
            memory.radix_cache_backend is not None,
        ),
        # Trees that rewrite a live request's row outside
        # `UnifiedRadixCache.checkpoint_into_tree`, whose rewrites drop the
        # decode staging's host rows (`deepseek_v41_radix`).
        (
            (
                "a KV storage backend with radix cache (LMCache, FlexKV, "
                "external cache linker)"
            ),
            memory.enable_lmcache
            or memory.enable_flexkv
            or memory.enable_unified_cache_external_linker,
        ),
    )


def refuse_unsupported_sglang_features() -> None:
    """Fail before weights load on SGLang features this bridge cannot serve."""
    from sglang.srt import runtime_context as sglang_config
    from sglang.srt.environ import envs as sglang_envs

    from atom.utils import envs as atom_envs

    memory = sglang_config.get_memory()
    exec_ = sglang_config.get_exec()
    disagg = sglang_config.get_disagg()
    parallel = sglang_config.get_parallel()
    schedule = sglang_config.get_schedule()
    chunk = schedule.chunked_prefill_size
    graphs = exec_.graph.cuda_graph_config
    decode_graphs = graphs.decode.backend != "disabled"
    unsupported = (
        # Unchunked, SGLang admits a first prompt whole, past the builder's
        # buffers: the forward would refuse it and take the server down.
        (
            "disabled chunked prefill (--chunked-prefill-size must be > 0)",
            chunk is None or int(chunk) <= 0,
        ),
        ("hierarchical cache", memory.enable_hierarchical_cache),
        # SGLang refuses it itself on hardware other than CUDA and gfx950.
        (
            "encoder SWA bounded replay",
            exec_.features.enable_encoder_swa_bounded_replay,
        ),
        # Decoder SWA bounded replay is on by default, as natively; SGLang's
        # flag restates it, and contradicts the variable that turns it off.
        (
            (
                "--enable-decoder-swa-bounded-replay with "
                "ATOM_SGLANG_V41_DECODER_REPLAY off"
            ),
            exec_.features.enable_decoder_swa_bounded_replay
            and not deepseek_v41_envs.ATOM_SGLANG_V41_DECODER_REPLAY,
        ),
        # The pool is sized by SGLang's hybrid-SWA full token count.
        ("--disable-hybrid-swa-memory", schedule.disable_hybrid_swa_memory),
        # The ATOM arena is allocated outside SGLang's memory-saver region.
        ("--enable-memory-saver", exec_.features.enable_memory_saver),
        ("PD disaggregation", str(disagg.disaggregation_mode) != "null"),
        ("DP attention", parallel.enable_dp_attention),
        # Either makes SGLang gather attention inputs across attention-TP
        # ranks (`require_attn_tp_gather`), which the ATOM model does not
        # read, and round its decode-graph buckets up to the attention-TP
        # size, past --max-running-requests (the builder's request rows).
        # SGLang also sets the backend from --moe-runner-backend megamoe
        # (an alias) and --enable-waterfill (forces deepep): name the value.
        (
            (
                "a MoE all-to-all backend (resolved "
                f"moe_a2a_backend={exec_.moe.moe_a2a_backend}, from "
                "--moe-a2a-backend, --moe-runner-backend megamoe or "
                "--enable-waterfill)"
            ),
            str(exec_.moe.moe_a2a_backend) != "none",
        ),
        ("--moe-dense-tp-size", parallel.moe_dense_tp_size is not None),
        ("decode context parallelism (--dcp-size > 1)", parallel.dcp_size > 1),
        (
            "prefill CUDA graphs (launch with --cuda-graph-backend-prefill disabled)",
            graphs.prefill.backend != "disabled",
        ),
        # Decode graphs stage every replay on the host, outside the captured
        # region (the backend shim); the glue graph would record that staging
        # once and replay it.
        (
            "SGLANG_ENABLE_METADATA_GLUE_GRAPH with decode CUDA graphs",
            decode_graphs and sglang_envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get(),
        ),
        # A decode graph records Engram's device staging; with either variable
        # off Engram stages its rows on the host every step instead.
        (
            "ATOM_ENGRAM_UVA=0 or ATOM_ENGRAM_OVERLAP=0 with decode CUDA graphs",
            decode_graphs
            and not (atom_envs.ATOM_ENGRAM_UVA and atom_envs.ATOM_ENGRAM_OVERLAP),
        ),
        # Image rows need ATOM's vision path, which the wrapper does not feed.
        (
            "multimodal requests (launch with --language-model-only)",
            not disagg.language_model_only,
        ),
    )
    if not memory.disable_radix_cache:
        unsupported += _radix_refusals(memory, schedule)
    refused = [name for name, enabled in unsupported if enabled]
    spec = sglang_config.get_spec()
    if spec.speculative_algorithm is not None:
        refused.extend(_dspark_launch_refusals(spec, exec_))
    if refused:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin does not support " + ", ".join(refused)
        )


def _dspark_launch_refusals(spec, exec_) -> list:
    """What a DSpark V4.1 launch asks for that the DSpark path does not
    serve: static verify, with or without the overlap schedule; decode graphs
    off, prefill eager. SGLang's own variables are read through SGLang's
    ``envs``, parsed as SGLang parses them.

    The algorithm is DSPARK: SGLang refuses any other for V4.1, and its
    unified-kv DSV4 ring, while it resolves the server args
    (`validate_deepseek_v41_features`). The five drafts and the target's
    checkpoint are checked when ATOM's ``Config`` is built, before this runs
    (`build_v41_dspark_speculative_config`)."""
    from sglang.srt.environ import envs as sglang_envs

    refused = []
    mode = sglang_envs.SGLANG_RAGGED_VERIFY_MODE.get().strip().lower()
    if mode != "static":
        refused.append(
            f"SGLANG_RAGGED_VERIFY_MODE={mode} (DSpark verify must be static)"
        )
    decode = exec_.graph.cuda_graph_config.decode.backend
    if decode != "disabled":
        refused.append(
            f"DSpark with the {decode} decode CUDA graph backend (launch with "
            "--cuda-graph-backend-decode disabled)"
        )
    # The overlap schedule is served (see "DSpark under the overlap schedule"
    # in the module docstring): a round's verify, accept and commit are one
    # `forward_batch_generation` call on the scheduler thread and the forward
    # stream, so the verify's tentative step is committed, in the epoch it
    # opened, before the next round is staged.
    if not sglang_envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        # The ATOM draft embeds its own block; SGLang would pass input_embeds.
        refused.append("SGLANG_DSPARK_EMBED_IN_GRAPH=0 (the ATOM draft embeds)")
    # SGLang's parse: unset or unparsable is -1, off; it simulates above 0.
    simulated = float(sglang_envs.SGLANG_SIMULATE_ACC_LEN.get())
    if (
        simulated > 0
        and simulated != 1.0
        and not deepseek_v41_envs.ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC
    ):
        # A fixed acceptance length commits draft tokens the target did not
        # choose: the text is not the model's. Benchmarks at a fixed mean
        # accept length (native's --spec-decode-acceptance-length) opt in.
        refused.append(
            f"SGLANG_SIMULATE_ACC_LEN={simulated} (only 1.0, the verify-only "
            "test, unless ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC=1 for a "
            "perf run)"
        )
    if spec.speculative_use_block_verification:
        refused.append("DSpark block verification")
    if exec_.features.enable_decoder_swa_bounded_replay:
        # DSpark runs without decoder replay (`_deepseek_v41_decoder_replays`):
        # SGLang's flag would be accepted and then overridden.
        refused.append(
            "--enable-decoder-swa-bounded-replay with DSpark (DSpark runs "
            "without decoder SWA bounded replay)"
        )
    return refused


def align_atom_config_with_sglang(atom_config: Any) -> None:
    """Size the builder by SGLang's resolved schedule (``Config`` saw the raw
    ServerArgs): its fixed buffers must hold every step SGLang can form.

    No step exceeds these: SGLang runs at most ``max_running_requests``
    requests, a decode step has one token per request, and its
    ``PrefillAdder`` charges mixed-in decode rows against the chunk budget, so
    a step never has more tokens than the largest of ``max_running_requests``,
    ``chunked_prefill_size`` and ``max_prefill_tokens`` (``Config`` also keeps
    a 16384-token floor)."""
    from sglang.srt.runtime_context import get_schedule

    schedule = get_schedule()
    atom_config.max_num_batched_tokens = max(
        int(atom_config.max_num_batched_tokens),
        int(schedule.chunked_prefill_size),
        int(schedule.max_prefill_tokens),
        int(schedule.max_running_requests),
    )
    atom_config.max_num_seqs = int(schedule.max_running_requests)
    # The builder sizes its score workspace from this fraction.
    atom_config.gpu_memory_utilization = float(schedule.mem_fraction_static)


# --- Metadata builder ---


# H2D publication: native's ModelRunner owns a registry over its
# ``forward_vars``; here the shim runner and the cache's host sources do.

# `PagedAttentionCache`'s host sources, published by `prepare_model_inputs`.
_V41_CACHE_SOURCES = (
    "_cursor_staging",
    "tentative_staging",
    "_reset_slots",
    "_reset_histories",
)


def _v41_publication_owner(variables, device):
    """A publication owner over the ``CpuGpuBuffer``s in ``variables``."""
    from atom.utils import CpuGpuBuffer
    from atom.utils.h2d import PublicationOwner

    event = torch.cuda.Event()
    owner = PublicationOwner(device, event)
    members = {}
    for name, buffer in variables.items():
        if isinstance(buffer, CpuGpuBuffer):
            binding = owner.bind(buffer, name, unit=buffer.publication_unit)
            if buffer.publication_group:
                members.setdefault(buffer.publication_group, []).append(binding)
    groups = {name: owner.group(name, bindings) for name, bindings in members.items()}
    event.record()
    return owner, groups


@contextmanager
def _v41_publication_epoch(owner, *, resume=False):
    """One H2D publication epoch; ``resume`` reopens the one a step's
    preparation sealed (a DSpark verify's commit)."""
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


def _init_v41_publication(runner, builder):
    """The shim runner's registry: `_v41_publication_owner` over its
    ``forward_vars`` plus one group per phase the builder declares."""
    owner, groups = _v41_publication_owner(runner.forward_vars, builder.device)
    for name, names in builder.h2d_group_members().items():
        groups[name] = owner.group(
            name, [runner.forward_vars[item]._publication for item in names]
        )
    return owner, groups


def _make_builder_cls():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    class SGLangDeepseekV41MetadataBuilder(DeepseekV41MetadataBuilder):
        """ATOM's V4.1 builder, staged once per step in one publication epoch."""

        # Whether the prefill being staged gets a decoder tail: the bridge's
        # verdict on its ForwardBatch (`v41_extend_replays`), carried on the
        # scheduled batch as ``decoder_replays``.
        _sglang_decoder_replays = True

        def prepare_model_inputs(self, input_ids, metadata):
            cache = metadata.cache
            if cache._sglang_h2d_owner is None:
                sources = {name: getattr(cache, name) for name in _V41_CACHE_SOURCES}
                cache._sglang_h2d_owner, _ = _v41_publication_owner(
                    sources, self.device
                )
            with _v41_publication_epoch(cache._sglang_h2d_owner):
                super().prepare_model_inputs(input_ids, metadata)

        def _prepare(self, batch, running_bs, running_tokens, **kwargs):
            runner = self.model_runner
            if runner.h2d_owner is None:
                runner.h2d_owner, runner.h2d_groups = _init_v41_publication(
                    runner, self
                )
            with _v41_publication_epoch(runner.h2d_owner):
                # No runner token inputs carry the query metadata on this path.
                self.publish_query_metadata(
                    np.asarray(batch.num_scheduled_tokens, np.int32),
                    running_bs,
                    running_tokens,
                )
                self._sglang_decoder_replays = batch.decoder_replays
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

        def build_for_sglang_capture(self, bs, *, input_ids, slots):
            """Native ``build_for_cudagraph_capture`` (decode) for a bucket of
            ``bs`` requests, on the proxy pool's reserved capture STATE
            ``slots`` instead of ``0..bs-1`` (live requests' rows here), over
            SGLang's graph ``input_ids`` (the address every replay of the
            bucket refills) instead of the native runner's. Engram's stage,
            gather and cursor advance stay inside the forward."""
            # A full window in, as native, every row on SGLang's padding PAGE
            # 0, which no request is handed.
            start = self.cache.geometry.window_size
            pages = -(-(start + 1) // self.block_size)
            slots = tuple(int(slot) for slot in slots)
            batch = SimpleNamespace(
                is_dummy_run=False,
                req_ids=slots,
                num_scheduled_tokens=(1,) * bs,
                context_lens=(start + 1,) * bs,
                state_slots_committed=slots,
                block_tables=((0,) * pages,) * bs,
                total_seqs_num=bs,
                total_tokens_num=bs,
                is_final_chunk=None,
                replaying={},
                decoder_replays=True,
            )
            metadata, _ = self._prepare(
                batch,
                bs,
                bs,
                max_q_len=1,
                tentative=bool(self.cache.geometry.speculative_tokens),
                engram_live=False,
            )
            metadata.dummy = True  # No host Engram lookup for synthetic tokens.
            self.prepare_model_inputs(input_ids, metadata)
            return metadata

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


def init_deepseek_v41_builder(atom_config: Any):
    """Create this process's builder (idempotent) before SGLang's memory
    profile, so the KV budget counts its score workspace."""
    global _V41_BUILDER
    if _V41_BUILDER is not None:
        return _V41_BUILDER

    from atom.utils import CpuGpuBuffer

    device = torch.device("cuda", torch.cuda.current_device())
    max_num_batched_tokens = int(atom_config.max_num_batched_tokens)
    runner = SimpleNamespace(
        block_size=ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
        device=device,
        config=atom_config,
        max_bs=int(atom_config.max_num_seqs),
        max_num_batched_tokens=max_num_batched_tokens,
        h2d_owner=None,
        h2d_groups=None,
        forward_vars={
            # `prepare_batch_step` takes its request-start dtype from here.
            "positions": CpuGpuBuffer(
                max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
                pin_memory=True,
            )
        },
    )
    builder = _make_builder_cls()(runner)
    geometry = builder.geometry
    if atom_config.speculative_config is not None:
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
    stages = int(hf.num_nextn_predict_layers)
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


# --- KV pool ---


# Where each of a quantized pool's source buffers starts in the arena: what
# a fresh allocation of its own would be aligned to, at least.
_V41_SOURCE_ALIGN = 256


def _v41_source_buffers(geometry) -> int:
    """Buffers ``PagedAttentionCache`` allocates for a quantized pool: one
    auxiliary, one main source per owner, one window source per layer."""
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
    return int(builder.max_bs)


def _make_backed_cache_cls():
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    class SGLangBackedPagedAttentionCache(PagedAttentionCache):
        """``PagedAttentionCache`` over the proxy pool's arena."""

        def __init__(self, *args, backing: torch.Tensor, **kwargs):
            # `super().__init__` calls `_allocate_backing`.
            self._external_backing = backing
            # The publication owner over its host sources, made at the first
            # staged step (`SGLangDeepseekV41MetadataBuilder`).
            self._sglang_h2d_owner = None
            super().__init__(*args, **kwargs)

        def _allocate_backing(self, size, device):
            # A BF16 pool: the whole cache is one flat arena.
            return self._allocate_buffers([size], device)[0]

        def _allocate_buffers(self, sizes, device):
            # A quantized pool (`_allocate_quantized`): its source buffers,
            # carved in order out of the arena, the first at its base, each
            # on a `_V41_SOURCE_ALIGN` boundary (`_arena_bytes` has the room).
            spans, end = [], 0
            for size in sizes:
                start = -(-end // _V41_SOURCE_ALIGN) * _V41_SOURCE_ALIGN
                end = start + int(size)
                spans.append((start, end))
            backing = self._external_backing
            if backing.numel() < end:
                raise ValueError(
                    f"A CSA2 backing of {backing.numel()} bytes is short of the "
                    f"{end} this geometry needs"
                )
            views = []
            for start, stop in spans:
                view = backing[start:stop]
                # Core's own allocations are zeroed; the cursor reset and
                # every cold read assume it.
                view.zero_()
                views.append(view)
            return tuple(views)

    return SGLangBackedPagedAttentionCache


def _make_proxy_pool_cls():
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    class ATOMDeepSeekV41ProxyKVPool(BaseSWAKVPool):
        """SGLang's DSV4 pool (installed under its name) over ATOM's CSA2
        arena. SGLang builds it only in ``_build_dsv4_kv_pool``, which passes
        every argument below."""

        is_atom_v41_proxy_pool = True
        # The DSpark draft runner's role view (`_init_draft_role`).
        is_atom_v41_draft_proxy = False
        atom_cache = None
        # SGLang reads these off any DSV4 pool (no SWA allocator, no rings).
        _unified_kv = False
        needs_paged_swa_allocator = False
        request_window = None
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
            swa_size: int,
            page_size: int,
            swa_page_size: int,
            dtype: torch.dtype,
            layer_num: int,
            device: str,
            start_layer: int,
            end_layer: int,
            num_req_slots: int,
            full_size: int,
            is_draft_worker: bool,
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
            if _V41_BUILDER is None:
                # No sizing hook ran: the builder is outside the profiled budget.
                raise RuntimeError(
                    "DeepSeek-V4.1 proxy pool built before the metadata builder; "
                    "SGLang did not profile memory through "
                    "KVCacheConfigurator._profile_available_bytes"
                )
            builder = _V41_BUILDER
            geometry = builder.geometry
            self.max_num_reqs = int(max_num_reqs)
            self.page_size = int(page_size)
            self.swa_page_size = int(swa_page_size)
            self.dtype = dtype
            self.store_dtype = torch.uint8
            self.device = device
            self.start_layer = int(start_layer)
            self.end_layer = int(end_layer)
            self.layer_num = int(layer_num)
            # `size` is the token capacity SGLang's allocator schedules against:
            # its hybrid-SWA full count (--disable-hybrid-swa-memory, which
            # leaves it None, is refused at launch).
            self.size = int(full_size)
            self.full_size = self.size
            self.swa_size = int(swa_size)
            self.size_swa = self.swa_size
            self.full_to_swa_index_mapping = None

            # SGLang's padding page 0 is still addressable: count it.
            self.num_pages = self.size // self.page_size + 1
            # One STATE slot per req_to_token row (row 0 is padding), then
            # one reserved slot per request a decode graph captures: capture
            # (at startup or a mid-serving recapture) never writes a live
            # request's state, and no req_pool_idx can name a capture slot.
            self.num_req_slots = int(num_req_slots)
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
                prefill_window_staging=builder._prefill_window_staging,
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
            # SGLang builds the target worker (and its pool, which binds the
            # builder's cache) before the draft worker; the builder checked
            # the arena's DSpark geometry (`_check_dspark_geometry`), and
            # SGLang's page size is the target's.
            builder = get_deepseek_v41_builder()
            geometry = builder.geometry
            self.is_atom_v41_draft_proxy = True
            self.max_num_reqs = int(max_num_reqs)
            self.page_size = int(page_size)
            self.swa_page_size = int(swa_page_size)
            self.dtype = dtype
            self.store_dtype = torch.uint8
            self.device = device
            self.start_layer = int(start_layer)
            self.end_layer = int(end_layer)
            self.layer_num = int(layer_num)
            # SGLang sizes the draft from the target's pool config: the same
            # hybrid-SWA full count (DeepseekV4ForCausalLMDSpark is hybrid-SWA).
            self.size = self.full_size = int(full_size)
            self.swa_size = self.size_swa = int(swa_size)
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
    # Installed from `prepare_model`'s V4.1 branch only, so ``kvc.model`` is
    # the ATOM wrapper. The DSpark draft worker's pool is a role view of the
    # target's arena: it is not sized or profiled here.
    return not kvc.is_draft_worker and is_deepseek_v41_config(
        kvc.model_config.hf_config
    )


def _make_pool_configurator_cls():
    from sglang.srt.model_executor.pool_configurator import (
        MemoryPoolConfig,
        MemoryPoolConfigurator,
    )

    class ATOMDeepSeekV41PoolConfigurator(MemoryPoolConfigurator):
        """Size SGLang's V4.1 token pool by what ATOM's arena costs instead of
        SGLang's DSV4 layout: ``geometry.paged_bytes`` per PAGE plus the
        padding PAGE, one STATE slot per request row and per decode-graph
        capture slot, and the ``req_to_token`` rows nothing else budgets."""

        def __init__(self, kvc) -> None:
            builder = get_deepseek_v41_builder()
            geometry = builder.geometry
            self.geometry = geometry
            # The proxy pool's reserved decode-graph capture slots.
            self.capture_slots = _capture_slot_count(builder)
            self.page_size = int(kvc.page_size)
            self.context_len = int(kvc.model_config.context_len)
            from sglang.srt.runtime_context import get_schedule

            # Resolved before the model is built; DP attention is refused.
            self.reqs = int(get_schedule().max_running_requests)

        def _pool_bytes(self, pages: int, slots: int) -> int:
            # `pages` usable PAGEs plus the padding PAGE, the STATE slots of
            # the request rows and of decode-graph capture, and the int32
            # req_to_token rows SGLang allocates beside them.
            return (
                _arena_bytes(self.geometry, pages + 1, slots + self.capture_slots)
                + slots * self.context_len * 4
            )

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
            # One req_to_token row per request plus SGLang's padding row 0.
            slots = self.reqs + 1
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
                surface = self.model._v41_dspark
                if surface is not None and surface.draft is not None:
                    compact_v41_moe_buffers(surface.draft, role="draft")
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
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner

    original = decode_cuda_graph_runner.create_dsv41_candidate_graph_variants
    if getattr(original, "_atom_v41_patched", False):
        return

    @wraps(original)
    def create_dsv41_candidate_graph_variants(model_runner, *args, **kwargs):
        if is_deepseek_v41_config(model_runner.model_config.hf_config):
            return None
        return original(model_runner, *args, **kwargs)

    create_dsv41_candidate_graph_variants._atom_v41_patched = True
    decode_cuda_graph_runner.create_dsv41_candidate_graph_variants = (
        create_dsv41_candidate_graph_variants
    )


def _is_v41_runner(model_runner) -> bool:
    cached = getattr(model_runner, "_atom_is_v41", None)
    if cached is None:
        cached = is_deepseek_v41_config(model_runner.model_config.hf_config)
        model_runner._atom_is_v41 = cached
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
                forward_batch.atom_v41_req_pool_indices_cpu = batch.req_pool_indices_cpu
            return forward_batch

        ForwardBatch.init_new = classmethod(init_new)
        ForwardBatch._atom_v41_host_mirror_patched = True

    from sglang.srt.model_executor.runner import decode_cuda_graph_runner

    original_view = decode_cuda_graph_runner.build_replay_fb_view
    if getattr(original_view, "_atom_v41_patched", False):
        return

    @wraps(original_view)
    def build_replay_fb_view(*, forward_batch, **kwargs):
        # Both SGLang callers (the decode and NPU graph runners) pass keywords.
        view = original_view(forward_batch=forward_batch, **kwargs)
        host_slots = getattr(forward_batch, "atom_v41_req_pool_indices_cpu", None)
        if host_slots is not None:
            view.atom_v41_req_pool_indices_cpu = host_slots
            view.atom_v41_rids = forward_batch.rids
        return view

    build_replay_fb_view._atom_v41_patched = True
    decode_cuda_graph_runner.build_replay_fb_view = build_replay_fb_view


def install_deepseek_v41_proxy_pool_patch() -> None:
    """Make SGLang size and build the V4.1 proxy pool instead of its DSV4 pool,
    keep its own V4.1 decode-graph variants off the ATOM model, carry decode
    batches' host request rows to the replay staging.

    First refuses an SGLang other than the one this rewiring is written
    against (`sglang_version_guard`), before any of it is installed."""
    from atom.plugin.sglang.patches.sglang_version_guard import (
        refuse_unsupported_sglang,
    )

    refuse_unsupported_sglang()
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as dsv4_pool

    _install_pool_sizing_patch()
    _install_graph_variants_patch()
    _install_host_mirror_patch()
    original_pool_cls = dsv4_pool.DeepSeekV4TokenToKVPool
    if getattr(original_pool_cls, "is_atom_v41_proxy_pool", False):
        return
    proxy_cls = _make_proxy_pool_cls()
    dsv4_pool.DeepSeekV4TokenToKVPool = proxy_cls
    dsv4_pool.ATOMDeepSeekV41ProxyKVPool = proxy_cls
    # Modules that imported the class by name (the configurator) too.
    for module in list(sys.modules.values()):
        if getattr(module, "DeepSeekV4TokenToKVPool", None) is original_pool_cls:
            module.DeepSeekV4TokenToKVPool = proxy_cls


# --- Per-forward translation ---


def _cpu_ints(cpu_value, gpu_value, n: int) -> np.ndarray:
    # SGLang's host mirror when it has one (never req_pool_indices on DECODE).
    value = cpu_value if cpu_value is not None else gpu_value
    if torch.is_tensor(value):
        value = value[:n].detach().cpu().numpy()
    return np.asarray(value[:n], dtype=np.int64)


def _req_to_token() -> torch.Tensor:
    # ForwardBatch carries no pools; SGLang's forward context does.
    return get_req_to_token_pool().req_to_token


def _block_rows(req_to_token, req_pool_indices, ends, block_size: int):
    # DECODE / EXTEND / TARGET_VERIFY rows only (IDLE is refused), each ending
    # at >= 1 token. A verify's pages include the rows SGLang allocated for it
    # at p..p+5; a fused radix replay's rows sit below P, inside these pages.
    widths = [-(-int(end) // block_size) for end in ends]
    widest = max(widths)
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
    ``checkpoint_into_tree`` hook). Under DSpark (radix cache off) the pages
    a verify's draft rows enter are SGLang's spec-v2 over-allocation,
    reserved whole and kept until the request's KV is released (never per
    rejection), so they too only grow. An EXTEND (every request's first
    forward, and again after a retraction) reads the row back anyway and
    records it here; a decode or verify reads it again only when a request
    has entered a page its copy does not hold, the slot's request changed, or
    a checkpoint dropped the copy.

    Keyed by STATE slot (``req_pool_idx``) and checked against the request id,
    so a copy is never used for another tenant of the slot.
    """

    def __init__(self):
        self._rows = {}

    def record(self, slots, rids, rows):
        for slot, rid, row in zip(slots, rids, rows):
            self._rows[int(slot)] = (rid, row)

    def forget(self, slot) -> None:
        """Drop ``slot``'s copy: its row was rewritten in place."""
        self._rows.pop(int(slot), None)

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


def forget_host_page_rows(slot) -> None:
    """The tree rewrote ``slot``'s ``req_to_token`` row: drop its host copy."""
    _HOST_ROWS.forget(slot)


def _refuse_padding_page(rows) -> None:
    # SGLang's paged allocator never hands out page 0 (its padding page, also
    # where decode-graph capture rows point), so a live row naming it read a
    # req_to_token entry that was never written.
    if any(np.any(row <= 0) for row in rows):
        raise ValueError(
            "DeepSeek-V4.1 block table names SGLang's padding page 0 in a live "
            "request's row"
        )


def _refuse_shared_or_padding_slots(slots, count: int) -> None:
    if np.unique(slots).size != count or np.any(slots <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 needs one live req_pool_idx per request; got {slots}"
        )


def _is_target_verify(mode) -> bool:
    return mode.is_target_verify()


def _verify_lengths(forward_batch, bs: int):
    """``(prefix, width)`` of a TARGET_VERIFY: every request verifies
    ``spec_info.draft_token_num`` rows (6, the bonus token and its drafts;
    ATOM's speculative config refuses any other count), on top of its
    committed prefix.

    The width is SGLang's official verify width, never the extend lengths.
    ``seq_lens_cpu`` arrives already widened by it (this backend has no
    ``make_forward_metadata_from_raw_verify``); ``live_seq_lens_cpu`` is the
    committed prefix the verify caller recorded before widening.
    """
    spec_info = forward_batch.spec_info
    width = int(spec_info.draft_token_num)
    live = spec_info.live_seq_lens_cpu
    if live is None:
        # `seq_lens` may be SGLang's over-allocated bound here, so the prefix
        # cannot be derived from it.
        raise ValueError(
            "DeepSeek-V4.1 TARGET_VERIFY needs spec_info.live_seq_lens_cpu"
        )
    return _cpu_ints(live, None, bs), width


def _radix_stamps():
    """The proxy pool's per-slot tenant stamps and SGLang's row generations,
    off the pools of the current forward context."""
    return (
        get_token_to_kv_pool().state_gen,
        get_req_to_token_pool().req_generation,
    )


def _check_radix_resumes(slots, prefix, fused=None) -> None:
    """With radix cache on, refuse an extend resuming a slot that is neither
    stamped for its current tenant nor rebuilt by the fused record this
    forward carries. Runs while the metadata is built, before any kernel."""
    if radix.radix_replay_rows() is None or not np.any(prefix > 0):
        return
    state_gen, generation = _radix_stamps()
    carried = () if fused is None else fused.slots
    radix.check_state_stamps(state_gen, generation, slots, prefix, carried)


def _stamp_fused_hits(fused) -> None:
    """Stamp on consume: this forward carries ``fused``, so its hits' slots
    now belong to their tenants."""
    state_gen, generation = _radix_stamps()
    radix.stamp_fused_hits(state_gen, generation, fused)


def _fused_replaying(fused):
    """``ScheduledBatch.replaying`` for a step carrying a fused radix replay
    (``deepseek_v41_radix.FusedReplay``): each hit's span starts at
    ``P - R`` in this forward, its slot rebuilt over ``[P - R, P)``. The seam
    planned the record off the ScheduleBatch this ForwardBatch was built from
    in the same ``forward_batch_generation`` call: its rows, slots, request
    ids and ``P`` are this forward's, and its history is the builder's
    width."""
    width = int(fused.histories.shape[1])
    replaying = {}
    for k, i in enumerate(fused.rows):
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


def build_v41_scheduled_batch(forward_batch, block_size: int):
    """ATOM's scheduled-batch protocol for one SGLang ``ForwardBatch``.

    ``DECODE``, ``EXTEND`` (the eager runner turns ``MIXED`` into ``EXTEND``)
    and DSpark's ``TARGET_VERIFY`` are translated; every other mode is
    refused. A
    step carrying a fused radix replay (``deepseek_v41_radix.FUSED_ATTR``)
    schedules each hit from ``P - R``: its span gains the replay rows
    (``replay_rows``), the same rows ``expand_v41_forward_inputs`` prepends
    to its inputs."""
    mode = forward_batch.forward_mode
    verify = _is_target_verify(mode)
    if not (mode.is_decode() or verify or mode.is_context_parallel_extend()):
        raise NotImplementedError(
            f"DeepSeek-V4.1 SGLang bridge does not translate {mode} yet"
        )
    bs = int(forward_batch.batch_size)
    ends = _cpu_ints(forward_batch.seq_lens_cpu, forward_batch.seq_lens, bs)
    if verify:
        prefix, width = _verify_lengths(forward_batch, bs)
        lengths = np.full(bs, width, dtype=np.int64)
    elif mode.is_decode():
        # positions = seq_lens - 1: seq_lens already counts the decode token.
        lengths = np.ones(bs, dtype=np.int64)
        prefix = ends - 1
    else:
        lengths = _cpu_ints(
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens, bs
        )
        prefix = _cpu_ints(
            forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens, bs
        )
    if np.any(prefix + lengths != ends):
        raise ValueError(
            "DeepSeek-V4.1 forward batch lengths disagree: "
            f"prefix={prefix.tolist()} lengths={lengths.tolist()} "
            f"ends={ends.tolist()}"
        )
    # SGLang mirrors req_pool_indices on the host for EXTEND only.
    slots = _cpu_ints(
        forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices, bs
    ).astype(np.int32)
    _refuse_shared_or_padding_slots(slots, bs)
    rids = list(forward_batch.rids[:bs])
    fused = getattr(forward_batch, radix.FUSED_ATTR, None)
    replay_rows = np.zeros(bs, dtype=np.int64)
    replaying = {}
    # A fused record rides on an EXTEND only: the seam plans it for EXTEND
    # batches and refuses a MIXED one.
    if not (mode.is_decode() or verify):
        _check_radix_resumes(slots, prefix, fused)
    if fused is not None:
        replaying = _fused_replaying(fused)
        _stamp_fused_hits(fused)
        # Each hit's span now starts at P - R; context_lens stay.
        replay_rows = fused.replay_rows(bs)
        lengths = lengths + replay_rows
    rows = _block_rows(
        _req_to_token(), forward_batch.req_pool_indices[:bs], ends, block_size
    )
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
        # shorter than the window (`v41_extend_replays`). A fused radix
        # replay is one span from P - R, at least the window long.
        is_final_chunk=None,
        replaying=replaying,
        # Rows ATOM runs ahead of each request's own (a fused radix replay).
        replay_rows=replay_rows,
        # Whether a prefill gets a decoder tail (the builder reads it for
        # every batch, publishes a tail for a prefill only); an EXTEND's is
        # decided on its ForwardBatch (`v41_extend_replays`).
        decoder_replays=True,
        state_slots_committed=[int(s) for s in slots],
        block_tables=rows,
        total_seqs_num=bs,
        total_tokens_num=int(lengths.sum()),
    )


def v41_extend_replays(forward_batch, batch, window: int) -> bool:
    """Whether this EXTEND may run V4.1's decoder on each request's last
    ``window`` rows only (decoder SWA bounded replay), which leaves every
    output row but each request's last zero.

    Not where SGLang reads other rows than each request's last -- input
    logprobs, a FULL hidden-state capture, a prefill-only (scoring) batch --
    nor where a request resumes its prompt (a prefix) with fewer than
    ``window`` rows: native backs such a forward off to start ``window`` rows
    before its end (`BlockManager.replay_start`), which SGLang's chunk cannot,
    and a tail of fewer rows would truncate their window at the chunk start.
    That covers a MIXED batch's decode rows too. Such a step computes every
    row, reading the decoder rings the previous chunk's tail wrote (every
    chunk counts as finishing, `build_v41_scheduled_batch`).

    A fused radix replay's rows (``batch.replay_rows``) are ATOM's: they
    count toward the tail (the span ATOM runs starts at ``P - R``, at least
    the window long) but not toward SGLang's logprob rows."""
    if forward_batch.is_prefill_only:
        return False
    capture = forward_batch.capture_hidden_mode
    if capture is not None and capture.is_full():
        return False
    lengths = np.asarray(batch.num_scheduled_tokens, np.int64)
    if forward_batch.return_logprob:
        # As SGLang's `LogitsMetadata.from_forward_batch` decides it, over
        # SGLang's own rows.
        starts = forward_batch.extend_logprob_start_lens_cpu
        if starts is None:
            return False
        own = lengths - np.asarray(batch.replay_rows, np.int64)
        if np.any(own - np.asarray(starts[: len(own)], np.int64) > 0):
            return False
    prefix = np.asarray(batch.context_lens, np.int64) - lengths
    return not bool(np.any((prefix > 0) & (lengths < int(window))))


def build_deepseek_v41_forward_metadata(forward_batch):
    """Stage this forward's CSA2 step once, before the model (``_prepare``,
    ``prepare_model_inputs``, Engram), marked ``staged_outside_forward`` so
    the model's ``v41_begin_forward`` does not stage it again.

    A TARGET_VERIFY is staged tentatively (``max_q_len`` = the verify width):
    the cursor is not advanced, every prefix the sampler may accept is kept in
    ``cache.pending``, and `commit_deepseek_v41_dspark_verify` writes the
    accepted one after SGLang's accept. Not `prepare_decode`: its start
    positions come from native's token processor, which SGLang does not run.
    """
    from atom.model_ops.engram.device.staging import EngramStagedRows

    mode = forward_batch.forward_mode
    builder = get_deepseek_v41_builder()
    batch = build_v41_scheduled_batch(forward_batch, builder.block_size)
    input_ids = forward_batch.input_ids
    fused = getattr(forward_batch, radix.FUSED_ATTR, None)
    if fused is not None:
        # The inputs ATOM executes (`SGLangPluginRuntime`, through the
        # adapter's `expand_forward_inputs`): each hit's [P - R, P) first,
        # widened by `SGLangPluginRuntime.__enter__`
        # (`expand_v41_forward_inputs`) before the forward context builds
        # this metadata.
        input_ids = fused.expanded.input_ids
    running_tokens = int(input_ids.shape[0])
    running_bs = batch.total_seqs_num
    if _is_target_verify(mode):
        width = int(batch.num_scheduled_tokens[0])
        metadata, _ = builder._prepare(
            batch, running_bs, running_tokens, max_q_len=width, tentative=True
        )
    elif mode.is_decode():
        metadata, _ = builder.prepare_decode(batch, running_bs, running_tokens, 1)
    else:
        if builder._decoder_replays:
            batch.decoder_replays = v41_extend_replays(
                forward_batch, batch, builder.geometry.window_size
            )
        metadata, _ = builder.prepare_prefill(batch, running_bs, running_tokens)
    # ATOM's cache keeps the tentative step (`cache.pending`) until the
    # commit, and refuses any other step before it (`require_committed`).
    builder.prepare_model_inputs(input_ids, metadata)
    # Engram's staging too (its cursor advance rides on it).
    rows = metadata.engram_embeddings
    if isinstance(rows, EngramStagedRows):
        rows.stage()
        rows.join()
    metadata.staged_outside_forward = True
    return metadata


# --- Decode CUDA graphs ---
# SGLang captures one full decode graph per batch-size bucket and replays it
# with no model Python. Every step's ATOM metadata is staged on the host by the
# backend's `init_forward_metadata_out_graph`, outside the captured region, into
# the builder's fixed-address buffers; Engram's stage / gather / cursor advance
# are recorded inside the graph, as native ATOM records them. Prefill, mixed
# and larger-than-bucket decode batches keep the eager path above.

# The forward batch attribute a capture's metadata rides on into the forward.
GRAPH_METADATA_ATTR = "atom_v41_graph_metadata"


def check_deepseek_v41_graph_capacity(max_bs: int, max_num_tokens: int):
    """Refuse a decode-graph runner the builder cannot serve. The pool's
    capture slots follow the builder (`_capture_slot_count`), past the
    ``req_to_token`` rows SGLang hands it as ``num_req_slots``."""
    builder = get_deepseek_v41_builder()
    engram = builder.engram
    if engram is not None and engram.host.overlap is None:
        # ATOM_ENGRAM_UVA=0 / ATOM_ENGRAM_OVERLAP=0 are refused at launch;
        # this is the page-lock fallback at load (`EngramHost._enable_uva`),
        # where Engram stages its rows on the host each step and a graph
        # would record one step's.
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang decode graphs need Engram's device staging, "
            "and its tables could not be page-locked for it (see the "
            "'engram:' log lines); launch with --cuda-graph-backend-decode "
            "disabled"
        )
    max_bs, max_num_tokens = int(max_bs), int(max_num_tokens)
    if max_bs > builder.max_bs or max_num_tokens > builder.max_num_batched_tokens:
        # An invariant, checked once per runner: SGLang captures at most its
        # req_to_token pool's size (--max-running-requests, the builder's
        # max_bs), and every batch-size alignment that would round that up
        # (two-batch overlap, DP attention, a MoE all-to-all backend,
        # --moe-dense-tp-size) is refused at launch.
        raise ValueError(
            f"DeepSeek-V4.1 decode graphs of {max_bs} requests / {max_num_tokens} "
            f"tokens exceed the builder's {builder.max_bs} / "
            f"{builder.max_num_batched_tokens}; align --max-running-requests "
            "with the cuda graph batch sizes"
        )
    return builder


def stage_deepseek_v41_graph_capture(forward_batch, pool):
    """Stage the step a decode bucket is recorded on and hand it to the
    forward on ``forward_batch`` (the object SGLang passes to both warmups and
    the capture). Runs outside the captured region."""
    bs = int(forward_batch.batch_size)
    metadata = get_deepseek_v41_builder().build_for_sglang_capture(
        bs, input_ids=forward_batch.input_ids, slots=pool.capture_slots[:bs]
    )
    setattr(forward_batch, GRAPH_METADATA_ATTR, metadata)
    return metadata


def build_v41_graph_replay_batch(fb_view, req_to_token, block_size: int):
    """ATOM's scheduled batch for the live rows of a padded decode replay.

    The lengths come off SGLang's host seq_lens mirror, the slots and request
    ids off the host rows the forward batch carried
    (`_install_host_mirror_patch`), the PAGE rows off `HostPageRows`; a
    request that entered a page those copies do not hold, or a new tenant of
    a slot, costs one D2H of the rows. The padding rows (``req_pool_idx`` 0,
    ``seq_len`` 1) are not requests: ATOM pads them itself up to the bucket
    width.
    """
    raw_bs = int(fb_view.batch_size) - int(fb_view.num_padding)
    ends = np.asarray(fb_view.seq_lens_cpu[:raw_bs], dtype=np.int64)
    slots = np.asarray(fb_view.atom_v41_req_pool_indices_cpu, dtype=np.int32)
    _refuse_shared_or_padding_slots(slots, raw_bs)
    rids = list(fb_view.atom_v41_rids)
    rows = _HOST_ROWS.lookup(slots, rids, -(-ends // block_size))
    if rows is None:
        rows = _block_rows(
            req_to_token, fb_view.req_pool_indices[:raw_bs], ends, block_size
        )
        _HOST_ROWS.record(slots, rids, rows)
    slot_list = [int(slot) for slot in slots]
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=slot_list,
        num_scheduled_tokens=np.ones(raw_bs, dtype=np.int32),
        context_lens=ends,
        is_final_chunk=None,
        replaying={},
        decoder_replays=True,
        state_slots_committed=slot_list,
        block_tables=rows,
        total_seqs_num=raw_bs,
        total_tokens_num=raw_bs,
    )


def stage_deepseek_v41_graph_replay(fb_view, req_to_token):
    """Stage one decode replay into the addresses its bucket recorded.

    ``prepare_decode`` at the bucket width (capacity-shaped plans, padding
    sentinels), then ``prepare_model_inputs`` over SGLang's graph
    ``input_ids``: the deferred cursor probe and the Engram step, whose
    lookup and cursor advance the graph runs. No Engram ``stage`` / ``join``
    here and no ``staged_outside_forward``: no model Python runs on replay.
    """
    builder = get_deepseek_v41_builder()
    bs = int(fb_view.batch_size)
    batch = build_v41_graph_replay_batch(fb_view, req_to_token, builder.block_size)
    metadata, _ = builder.prepare_decode(batch, bs, bs, 1)
    builder.prepare_model_inputs(fb_view.input_ids, metadata)
    return metadata


# --- DSpark: the accepted prefix, committed once per round ---


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
    cache = pool.atom_cache
    # ATOM's `commit_tentative` refuses a missing tentative step and anchors
    # of another request count; the anchors are positional, so the requests
    # must also be the ones the verify staged, in its order.
    step = cache.pending.step
    slots = [int(span.slot) for span in step.requests]
    if slots != [int(slot) for slot in req_slots]:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark commit for requests in slots {list(req_slots)} "
            f"but the verify staged {slots}"
        )
    anchors = dspark_commit_anchors(commit_lens, verify_width).to(
        step.cu_seqlens_q.device
    )
    # Host-staged Engram prefixes were written in the epoch the verify's
    # preparation sealed; their H2D copy happens here, in that same epoch.
    with _v41_publication_epoch(cache._sglang_h2d_owner, resume=True):
        cache.commit_tentative(step, anchors)


def install_deepseek_v41_dspark_commit_patch() -> None:
    """Commit the accepted prefix at SGLang's model-local verify-commit hook.

    ``DSparkWorkerV2._commit_target_mamba_states_after_verify`` runs once per
    non-idle decode round, after the accept (and its TP sync) and before the
    hidden commit and the next draft, with the round's official
    ``commit_lens``; SGLang provides it for model-local state (mamba). The
    wrapper keeps the original and adds the commit. Installed only by the
    V4.1 DSpark target surface (`install_v41_dspark_target_surface`), so the
    target's pool is this plugin's proxy pool. Idempotent.
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
        # ScheduleBatch's own CPU mirror of the request rows (every writer
        # of the rows sets it, filter/merge keep it in step); SGLang 0.5.22
        # Req has no req_pool_idx.
        commit_deepseek_v41_dspark_verify(
            self.target_worker.model_runner.token_to_kv_pool,
            req_slots=batch.req_pool_indices_cpu.tolist(),
            commit_lens=commit_lens,
            verify_width=int(self.verify_num_draft_tokens),
        )

    cls._commit_target_mamba_states_after_verify = commit_after_verify
    cls._atom_v41_dspark_commit_patched = True
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: accepted-prefix commit installed on "
        "DSparkWorkerV2._commit_target_mamba_states_after_verify"
    )


def log_deepseek_v41_runtime(model: Any) -> None:
    """Startup proof: what actually got constructed for this rank (the
    wrapper's ``install_adapters``, once per wrapper)."""
    attn_classes = sorted({type(layer.attn).__name__ for layer in model.layers})
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang runtime: model=%s replay=%s attention=%s",
        type(model).__name__,
        model.replay,
        attn_classes,
    )
