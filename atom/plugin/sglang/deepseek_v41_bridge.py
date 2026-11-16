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
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
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
        resolve_replay_mode,
        resolve_replay_rows,
    )

    rows = resolve_replay_rows()
    resolve_replay_mode()
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
        # The DSpark draft rings and the verify lifecycle are not proved
        # under a replay yet.
        (
            "speculative decoding with radix cache",
            _sglang_setting(spec, "speculative_algorithm", None) is not None,
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
            "speculative decoding",
            setting(spec, "speculative_algorithm", None) is not None,
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

        unsupported += (
            (
                (
                    "radix cache without ATOM_V41_RADIX=1 (opt-in until the "
                    "fused replay is validated on GPU; or launch with "
                    "--disable-radix-cache)"
                ),
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
    if refused:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin does not support " + ", ".join(refused)
        )


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
def _v41_publication_epoch(owner):
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

        def build_for_sglang_capture(self, bs, *, input_ids, slots):
            """The step a decode graph of ``bs`` requests is recorded on.

            Core ``build_for_cudagraph_capture`` (decode, ``max_q_len=1``)
            with two differences: the STATE slots are the caller's (the proxy
            pool's reserved capture slots, never a live request's), and
            ``input_ids`` is SGLang's graph buffer, the address every replay
            of this bucket refills. Engram's stage, gather and cursor advance
            stay inside the forward (``staged_outside_forward`` unset), gated
            on a device live count of 0 here and of the real rows on replay.
            """
            from atom.utils.forward_context import RoundService

            cache = self.cache
            if cache is None:
                raise RuntimeError("Allocate the serving cache before graph capture")
            geometry = cache.geometry
            if geometry.speculative_tokens:
                raise NotImplementedError(
                    "DeepSeek-V4.1 SGLang decode graphs do not capture "
                    "speculative steps"
                )
            bs = int(bs)
            if bs < 1 or bs > self.max_bs or bs > self.max_num_batched_tokens:
                raise ValueError(
                    f"DeepSeek-V4.1 capture of {bs} requests exceeds the "
                    f"builder's {self.max_bs} / {self.max_num_batched_tokens}"
                )
            slots = tuple(int(slot) for slot in slots)
            if len(slots) != bs or len(set(slots)) != bs:
                raise ValueError(
                    f"DeepSeek-V4.1 capture needs {bs} distinct STATE slots; "
                    f"got {slots}"
                )
            if int(input_ids.shape[0]) != bs:
                raise ValueError(
                    f"DeepSeek-V4.1 capture input_ids rows {int(input_ids.shape[0])} "
                    f"!= {bs} requests"
                )
            # A full window in, as core does: at position 0 the compressor
            # reads no history and records a cold branch replay never takes.
            start = geometry.window_size
            pages = -(-(start + 1) // self.block_size)
            batch = SimpleNamespace(
                is_dummy_run=False,
                req_ids=slots,
                num_scheduled_tokens=(1,) * bs,
                context_lens=(start + 1,) * bs,
                state_slots_committed=slots,
                # SGLang's padding page for every entry: capture's compressed
                # rows land there, and no request is ever handed page 0.
                block_tables=((0,) * pages,) * bs,
                total_seqs_num=bs,
                total_tokens_num=bs,
                is_final_chunk=None,
                replaying={},
            )
            metadata, _ = self._prepare(
                batch, bs, bs, max_q_len=1, tentative=False, engram_live=False
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
    _V41_BUILDER = _make_builder_cls()(runner)
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang builder: block_size=%d max_bs=%d "
        "max_num_batched_tokens=%d kv=%s index=%s paged_bytes=%d state_bytes=%d",
        runner.block_size,
        runner.max_bs,
        max_num_batched_tokens,
        atom_config.kv_cache_dtype,
        atom_config.index_cache_dtype,
        int(_V41_BUILDER.geometry.paged_bytes),
        int(_V41_BUILDER.geometry.state_bytes),
    )
    return _V41_BUILDER


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
                raise NotImplementedError(
                    "DeepSeek-V4.1 DSpark draft pools are not enabled on the "
                    "SGLang plugin yet"
                )
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
            logger.info(
                "ATOM DeepSeek-V4.1 proxy pool: %d PAGEs x %d B + %d STATE slots "
                "(%d request rows + %d capture slots from %d) x %d B = %.2f GiB "
                "(SGLang tokens %d)",
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

        def get_kv_size_bytes(self):
            return self.raw_arena.nbytes

        def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
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
    """Carry a decode batch's host request rows to the V4.1 replay staging.

    SGLang keeps ``req_pool_indices_cpu`` on the ScheduleBatch through
    decode (filter / merge maintain it) but hands it to the ForwardBatch for
    EXTEND only, and the decode graph runner's replay view carries neither it
    nor the request ids. Both ride along as ``atom_v41_*`` attributes, on
    V4.1 batches only, so `build_v41_graph_replay_batch` can stage without
    reading the device back.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if not getattr(ForwardBatch, "_atom_v41_host_mirror_patched", False):
        original_init_new = ForwardBatch.init_new.__func__

        @wraps(original_init_new)
        def init_new(cls, batch, model_runner, *args, **kwargs):
            forward_batch = original_init_new(cls, batch, model_runner, *args, **kwargs)
            if forward_batch.forward_mode.is_decode() and _is_v41_runner(model_runner):
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
    """Host copies of live requests' PAGE rows, so a decode replay can stage
    without reading ``req_to_token`` back.

    That read is a device sync on the forward stream: under overlap
    scheduling it waits for the previous step's graph and serializes the next
    step's host staging behind it. A request's PAGE row only grows, one page
    every 256 tokens, and only SGLang's allocator writes it -- except that
    with radix cache on a checkpoint repoints a live row at the tree's pages
    in place, which drops that slot's copy (``forget``, called from the radix
    seam's ``checkpoint_into_tree`` hook; ``host_page_rows_supported``). An
    EXTEND (every request's first forward, and again after a retraction)
    reads the row back anyway and records it here; a decode reads it again
    only when a request has entered a page its copy does not hold, the slot's
    request changed, or a checkpoint dropped the copy.

    Keyed by STATE slot (``req_pool_idx``) and checked against the request id,
    so a copy is never used for another tenant of the slot. Under
    ``ATOM_SGLANG_V41_CHECK_HOST_ROWS=1`` every replay also reads the rows
    back and compares (a device sync per step; validation only).
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


def _refuse_padding_page(rows) -> None:
    # SGLang's paged allocator never hands out page 0 (its padding page, also
    # where decode-graph capture rows point), so a live row naming it read a
    # req_to_token entry that was never written.
    if any(np.any(row <= 0) for row in rows):
        raise ValueError(
            "DeepSeek-V4.1 block table names SGLang's padding page 0 in a live "
            "request's row"
        )


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


def build_v41_scheduled_batch(forward_batch, block_size: int, history=None):
    """ATOM's scheduled-batch protocol for one SGLang ``ForwardBatch``.

    Only ``DECODE`` and ``EXTEND`` (the eager runner turns ``MIXED`` into
    ``EXTEND``) are translated; speculative and other modes are refused. A
    radix replay side forward (``encoder_swa_replay``) carries its rows'
    ``replaying``; ``history``: the Engram history width it must carry. A
    step carrying a fused radix replay (``deepseek_v41_radix.FUSED_ATTR``)
    schedules each hit from ``P - R``: its span gains the replay rows
    (``replay_rows``), the same rows ``expand_v41_forward_inputs`` prepends
    to its inputs.
    """
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_context_parallel_extend()):
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
    if mode.is_decode():
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
    slots = _cpu_ints(
        getattr(forward_batch, "req_pool_indices_cpu", None),
        forward_batch.req_pool_indices,
        bs,
    ).astype(np.int32)
    if np.unique(slots).size != bs or np.any(slots <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 needs one live req_pool_idx per request; got {slots}"
        )
    rids = list(forward_batch.rids[:bs])
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
        if not mode.is_decode():
            _check_radix_resumes(slots, prefix, fused)
        if fused is not None:
            replaying = _fused_replaying(
                fused, prefix, slots, rids, block_size, history
            )
            _stamp_fused_hits(fused)
            # Each hit's span now starts at P - R; context_lens stay.
            replay_rows = fused.replay_rows(bs)
            lengths = lengths + replay_rows
    rows = _block_rows(
        _req_to_token(), forward_batch.req_pool_indices[:bs], ends, block_size
    )
    # Decode replays stage off these copies (`HostPageRows`).
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


def v41_extend_replays(forward_batch, batch, window: int) -> bool:
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
    the window long) but not toward SGLang's logprob rows."""
    if getattr(forward_batch, "is_prefill_only", False):
        return False
    capture = getattr(forward_batch, "capture_hidden_mode", None)
    is_full = getattr(capture, "is_full", None)
    if is_full is not None and is_full():
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
    batch = build_v41_scheduled_batch(
        forward_batch,
        builder.block_size,
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
    if mode.is_decode():
        metadata, step_positions = builder.prepare_decode(
            batch, running_bs, running_tokens, 1
        )
    else:
        if getattr(builder, "_decoder_replays", False):
            batch.decoder_replays = v41_extend_replays(
                forward_batch, batch, builder.geometry.window_size
            )
        metadata, step_positions = builder.prepare_prefill(
            batch, running_bs, running_tokens
        )
    builder.prepare_model_inputs(input_ids, metadata)
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
    """Stage the step a decode bucket is recorded on and hand it to the
    forward on ``forward_batch`` (the object SGLang passes to both warmups and
    the capture). Runs outside the captured region."""
    mode = forward_batch.forward_mode
    if not mode.is_decode():
        raise NotImplementedError(
            "DeepSeek-V4.1 SGLang graphs capture DECODE only, not "
            f"{getattr(mode, 'name', mode)}"
        )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DeepSeek-V4.1 graph step must be staged before capture")
    bs = int(forward_batch.batch_size)
    if bs > len(pool.capture_slots):
        raise ValueError(
            f"DeepSeek-V4.1 capture of {bs} requests exceeds the pool's "
            f"{len(pool.capture_slots)} capture slots"
        )
    metadata = _graph_builder().build_for_sglang_capture(
        bs, input_ids=forward_batch.input_ids, slots=pool.capture_slots[:bs]
    )
    setattr(forward_batch, GRAPH_METADATA_ATTR, metadata)
    return metadata


def _read_replay_rows(fb_view, req_to_token, raw_bs, widths, block_size):
    """The live rows' slots and PAGE rows, read back from the device: one D2H,
    a sync on the forward stream."""
    width = int(widths.max())
    indices = fb_view.req_pool_indices[:raw_bs]
    host = (
        torch.cat(
            (
                indices[:, None].to(torch.int32),
                (
                    req_to_token[indices, : width * block_size : block_size]
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


def build_v41_graph_replay_batch(fb_view, req_to_token, block_size: int):
    """ATOM's scheduled batch for the live rows of a padded decode replay.

    The lengths come off SGLang's host seq_lens mirror and, when the forward
    batch carried them (`_install_host_mirror_patch`), the slots off its host
    ``req_pool_indices`` mirror and the PAGE rows off `HostPageRows`; else, or
    when a request entered a page those copies do not hold, one D2H reads the
    slots and rows back. The padding rows (``req_pool_idx`` 0, ``seq_len`` 1)
    are not requests: ATOM pads them itself up to the bucket width.
    """
    bs = int(fb_view.batch_size)
    raw_bs = bs - int(getattr(fb_view, "num_padding", 0) or 0)
    if not 1 <= raw_bs <= bs:
        raise ValueError(f"DeepSeek-V4.1 replay of {raw_bs} live rows in {bs}")
    lengths = fb_view.seq_lens_cpu
    if lengths is None:
        raise ValueError(
            "DeepSeek-V4.1 decode replay needs SGLang's host seq_lens mirror"
        )
    if torch.is_tensor(lengths):
        lengths = lengths[:raw_bs].numpy()
    ends = np.asarray(lengths[:raw_bs], dtype=np.int64)
    if np.any(ends < 1):
        raise ValueError(f"DeepSeek-V4.1 decode replay lengths {ends.tolist()}")
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
        num_scheduled_tokens=np.ones(raw_bs, dtype=np.int32),
        context_lens=ends,
        is_final_chunk=None,
        replaying={},
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
