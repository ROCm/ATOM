# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import json
import logging
import os
import time

import safetensors
import safetensors.torch
import torch
from torch import nn
from transformers import AutoConfig

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `safetensors.torch.load(bytes)` path used when `ATOM_DISABLE_MMAP=true`
# raises `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu

from aiter.dist.parallel_state import get_tp_group

from atom.model_loader.loading_core import load_weights_into_model, rank_tag
from atom.model_loader.online_quant_streaming import OnlineQuantStreamer
from atom.model_loader.weight_iterator import (
    safetensors_weights_iterator,
)

# Re-exported so the many `from atom.model_loader.loader import WeightsMapper`
# call sites (models, vLLM/SGLang/RTP-LLM plugins) keep working.
from atom.model_loader.weight_names import WeightsMapper, WeightsMapping  # noqa: F401
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.moe import FusedMoEMethodBase
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.plugin.prepare import is_sglang
from atom.utils import envs

logger = logging.getLogger("atom")


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if loaded_weight.numel() == param.data.numel():
        param.data.copy_(loaded_weight)
    elif loaded_weight.numel() // get_tp_group().world_size == param.data.numel():
        loaded_weight_per_rank = loaded_weight.numel() // get_tp_group().world_size
        # Offset MUST use the TP-group-local rank (rank_in_group), NOT the global
        # rank: `.world_size` above is the TP group size, so the two must be from
        # the same (TP-group) frame. `.rank` is torch.distributed.get_rank()
        # (global). They coincide only when world == tp (pure TP); under PCP/DP/PP
        # the world splits into multiple TP groups, so a group's global ranks
        # (e.g. PCP rank 1 = global 4..7) exceed its world_size (4), making this
        # slice out of bounds → empty → copy_ fails.
        tp_rank_start = loaded_weight_per_rank * get_tp_group().rank_in_group
        tp_rank_end = tp_rank_start + loaded_weight_per_rank
        param.data.copy_(loaded_weight.view(-1)[tp_rank_start:tp_rank_end])
    else:
        # Shape mismatch we cannot resolve — leaving the destination at its init
        # value is almost always a bug. The post-load check in load_model() will
        # catch this and warn (param will be in `unloaded` set since this loader
        # never wrote to it). Raise here so the failure is loud at copy time
        # too, instead of being masked by the default ones-init of RMSNorm etc.
        raise RuntimeError(
            f"default_weight_loader: shape mismatch — param={tuple(param.shape)} "
            f"loaded={tuple(loaded_weight.shape)}. Cannot copy."
        )


# when plugin mode, model loader method is bind to model implementation
# thus call this interface to load the model, which leverages the load_model
# method
def load_model_in_plugin_mode(
    model,
    config,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
    spec_decode: bool = False,
    hf_config_override: AutoConfig | None = None,
    model_name_or_path_override: str | None = None,
) -> set[str]:

    # during loading model, the outplace operation may consume more
    # GPU mem, which cached in torch caching allocator, here actively
    # call empty cache to free the extra reserved but not used memory
    def _empty_cache():
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    assert (
        config.plugin_config is not None and config.plugin_config.is_plugin_mode
    ), "ATOM is not running in plugin mode"
    if model_name_or_path_override is not None:
        model_name_or_path = model_name_or_path_override
    elif config.plugin_config.is_vllm:
        model_name_or_path = config.plugin_config.model_config.model
    elif config.plugin_config.is_sglang:
        model_name_or_path = config.plugin_config.model_config.model_path
    elif config.plugin_config.is_rtpllm:
        model_name_or_path = config.plugin_config.model_config.ckpt_path

    _empty_cache()
    if hf_config_override is not None:
        config_for_loading = getattr(
            hf_config_override, "hf_config", hf_config_override
        )
        if hasattr(config_for_loading, "text_config"):
            config_for_loading = config_for_loading.text_config
    else:
        config_for_loading = (
            config.hf_config.text_config
            if hasattr(config.hf_config, "text_config")
            else config.hf_config
        )
    loaded_weights_record = load_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=config_for_loading,
        load_dummy=config.load_dummy,
        spec_decode=spec_decode,
        prefix=prefix,
        is_plugin_mode=True,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
    )
    _empty_cache()
    return loaded_weights_record


def _save_online_quant_info(
    oq_layers: list[dict],
    model_name_or_path: str,
    elapsed_seconds: float,
    online_quant_config: dict,
    timing_scope: str,
    peak_gpu_memory_gb: float | None,
):
    """Save online quantization info to a JSON file (rank 0 only)."""
    if get_tp_group().rank_in_group != 0:
        return
    output_dir = envs.ATOM_TORCH_PROFILER_DIR or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_ns = time.time_ns() % 1_000_000_000
    filepath = os.path.join(
        output_dir, f"online_quant_info_{timestamp}_{timestamp_ns:09d}.json"
    )

    payload = {
        "model": model_name_or_path,
        "online_quant_config": online_quant_config,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timing_scope": timing_scope,
        "peak_gpu_memory_gb": (
            round(peak_gpu_memory_gb, 3) if peak_gpu_memory_gb is not None else None
        ),
        "num_layers": len(oq_layers),
        "layers": oq_layers,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Online quantization info saved to %s", filepath)


# Dummy-weight init constants (see initialize_dummy_weights).
_DUMMY_WEIGHT_STD = 2.0**-4  # ~0.0625, a plausible transformer weight magnitude
_FP4_UNIT_BYTE = 0x22  # e2m1 fp4x2: both nibbles = 0b0010 = 1.0
_E8M0_UNIT_CODE = 123  # e8m0 exponent code for 2^(123-127) = 2^-4 = _DUMMY_WEIGHT_STD


def initialize_dummy_weights(model: nn.Module, mode: str) -> None:
    """Fill skipped-load (``--load_dummy``) params with finite values in place.

    ``mode="zero"``   -> every param zeroed (works for fp4/fp8/int/bf16 alike).
    ``mode="xavier"`` -> constant-magnitude init that keeps the forward finite and
    roughly at real-weight scale:

    - bf16/fp16/fp32 2D weight        -> ``xavier_uniform_``
    - 1D norm weight (non-bias)        -> 1.0
    - bias                             -> 0.0
    - float weight_scale              -> ``_DUMMY_WEIGHT_STD``
    - input_scale                      -> 1.0
    - fp8 packed weight               -> 1.0
    - fp4x2 packed weight (uint8-view) -> ``_FP4_UNIT_BYTE`` (each fp4 = 1.0)
    - e8m0 (uint8) block scale        -> ``_E8M0_UNIT_CODE`` (= 2^-4)

    Quantized weights are filled with a *constant* magnitude (not a true random
    distribution), so the effective weights survive the shuffle/swizzle in each
    quant method's ``process_weights_after_loading`` (a permutation of identical
    bytes is a no-op). FP4 (MXFP4) is the validated path; FP8 and other formats
    are made finite but not distribution-realistic.
    """
    for name, param in model.named_parameters():
        data = param.data
        if mode == "zero":
            # zero_() is the fast path: valid for every shape and every
            # standard dtype (fp8/int/bf16/...). Packed sub-byte dtypes
            # (e.g. Float4_e2m1fn_x2) have no CUDA fill kernel, so zero_()
            # raises ("fill_cuda" not implemented for that dtype); fall back
            # to zeroing the raw bytes instead (all-zero bytes == zero-valued
            # weights). Only packed weights reach the fallback, and those are
            # contiguous and >=1D, so the uint8 view is always valid there.
            try:
                data.zero_()
            except (NotImplementedError, RuntimeError):
                data.view(torch.uint8).zero_()
            continue
        # mode == "xavier"
        dt = data.dtype
        if "input_scale" in name:
            data.fill_(1.0)
        elif "scale" in name:
            if dt == torch.uint8:  # e8m0 block scale (fp4)
                data.fill_(_E8M0_UNIT_CODE)
            else:  # fp8/bf16 float scale
                data.fill_(_DUMMY_WEIGHT_STD)
        elif dt in (torch.float32, torch.float16, torch.bfloat16):
            if data.dim() >= 2:
                nn.init.xavier_uniform_(data)
            elif "bias" in name:
                data.zero_()
            else:  # 1D norm weight etc.
                data.fill_(1.0)
        elif dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
            data.fill_(1.0)  # fp8 packed weight
        else:  # fp4x2 packed weight, viewable as uint8
            data.view(torch.uint8).fill_(_FP4_UNIT_BYTE)


def load_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: str | None = None,
    spec_decode: bool = False,
    prefix: str = "",
    is_plugin_mode: bool = False,
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
):
    """Load a checkpoint into `model` and run post-load weight processing.

    The checkpoint -> parameter logic lives in `loading_core`, which is kept
    free of AITER so it can be unit-tested without a GPU build; this wrapper
    supplies the pieces that do need AITER (TP group, quant-config-driven
    shared-expert fusion) and owns everything that happens after the weights
    have landed.

    `is_plugin_mode` is unused and kept for call-site compatibility.
    """

    def _fuse_shared_expert(
        shared_expert_prefix: str, routed_expert_prefix: str
    ) -> bool:
        model_quant_config = getattr(
            getattr(model, "atom_config", None), "quant_config", None
        )
        if model_quant_config is None:
            model_quant_config = getattr(model, "quant_config", None)
        if model_quant_config is not None and hasattr(
            model_quant_config, "get_layer_quant_config"
        ):
            return is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                model_quant_config,
                shared_expert_prefix=shared_expert_prefix,
                routed_expert_prefix=routed_expert_prefix,
            )
        return is_rocm_aiter_fusion_shared_expert_enabled(
            shared_expert_prefix=shared_expert_prefix,
            routed_expert_prefix=routed_expert_prefix,
        )

    def _is_rank0() -> bool:
        # Diagnostics must not be the thing that breaks loading, and the TP
        # group may not exist yet (single-process tools, plugin hosts), so any
        # failure here degrades to "report from this rank".
        try:
            return get_tp_group().rank == 0
        except Exception:  # noqa: BLE001
            return True

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Quantize eligible modules as soon as their source weights complete.
    # This must also run for speculative draft loads: those modules were built
    # with the same meta-backed streaming parameters as the target. `spec_decode`
    # only changes checkpoint-name selection; it must not disable materialization.
    online_quant_streamer = OnlineQuantStreamer.maybe_create(model, load_dummy)

    loaded_weights_record = load_weights_into_model(
        model=model,
        model_name_or_path=model_name_or_path,
        hf_config=hf_config,
        load_dummy=load_dummy,
        spec_decode=spec_decode,
        prefix=prefix,
        weights_mapper=weights_mapper,
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
        default_weight_loader=default_weight_loader,
        fuse_shared_expert=_fuse_shared_expert,
        is_rank0=_is_rank0,
        weights_iterator=safetensors_weights_iterator,
        online_quant_streamer=online_quant_streamer,
    )

    # Dummy modes other than "empty" fill the skipped-load params with finite
    # values before post-processing, so shuffle/swizzle runs on clean constants.
    if load_dummy and load_dummy != "empty":
        initialize_dummy_weights(model, load_dummy)

    if online_quant_streamer is not None:
        online_quant_streamer.replay_stragglers_and_report(_is_rank0())

    has_online_quant = any(
        getattr(m, "online_quant", False)
        or (
            getattr(m, "quant_config", None) is not None
            and getattr(m.quant_config, "online_quant", False)
        )
        for _, m in model.named_modules()
    )
    streamed_done = (
        online_quant_streamer.done_module_ids
        if online_quant_streamer is not None
        else frozenset()
    )
    stream_candidate_count = (
        len(online_quant_streamer.candidates)
        if online_quant_streamer is not None
        else 0
    )
    stream_fallback_count = stream_candidate_count - len(streamed_done)
    if online_quant_streamer is not None:
        logger.info(
            "[%s] Streaming online quantization: %d/%d eligible modules were "
            "quantized while loading; the post-load pass remains enabled to "
            "quantize %d fallback module(s) and finish weight processing",
            rank_tag(),
            len(streamed_done),
            stream_candidate_count,
            stream_fallback_count,
        )
    elif has_online_quant:
        logger.info(
            "[%s] Post-load online quantization and weight processing started",
            rank_tag(),
        )
    pp_start = time.perf_counter()

    # Parent-first traversal is significant for streaming-deferred children:
    # their parent first combines source weights, then the child's normal hook
    # online-quantizes the final fused weight.
    for module_name, module in model.named_modules():
        # Avoid repeating module post-processing already run by the streamer.
        if (
            hasattr(module, "process_weights_after_loading")
            and id(module) not in streamed_done
        ):
            module.process_weights_after_loading()
        quant_method = getattr(module, "quant_method", None)

        # when running plugin mode for sglang, don't do the post process here
        # since sglang will call this func automatically after finishing loading
        if isinstance(quant_method, QuantizeMethodBase) and not is_sglang():
            quant_method.process_weights_after_loading(module)
        if isinstance(quant_method, FusedMoEMethodBase):
            quant_method.init_prepare_finalize(module)

        # Online quantization creates new params (e.g. weight_scale) that are
        # not present in the source checkpoint. Record them as "loaded" so the
        # plugin host's strict weight tracking (e.g. vLLM's default loader)
        # does not flag them as uninitialized.
        if getattr(module, "_online_quant_info", None) is not None:
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                loaded_weights_record.add(prefix + full_name)

    # Post-processing (AITER shuffle/swizzle, per-quant-method hooks) runs inside
    # the load time the caller reports, so it is timed unconditionally: without
    # this line a shuffle-dominated load is indistinguishable from a slow read.
    pp_elapsed = time.perf_counter() - pp_start
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / (1 << 30)
        if torch.cuda.is_available()
        else None
    )
    if not has_online_quant:
        logger.info(
            "[%s] Weight post-processing done: %.2f seconds",
            rank_tag(),
            pp_elapsed,
        )
    else:
        oq_layers = []
        raw_online_quant_config = None
        for _, module in model.named_modules():
            info = getattr(module, "_online_quant_info", None)
            if info is not None:
                oq_layers.append(info)
            if raw_online_quant_config is None:
                qc = getattr(module, "quant_config", None)
                if qc is not None and hasattr(qc, "online_quant_config_raw"):
                    raw_online_quant_config = qc.online_quant_config_raw
        if online_quant_streamer is not None:
            logger.info(
                "[%s] Post-stream fallback and weight processing done: %.2f "
                "seconds; %d module(s) quantized while loading, %d fallback "
                "module(s) quantized after loading, %d layers online-quantized "
                "in total",
                rank_tag(),
                pp_elapsed,
                len(streamed_done),
                stream_fallback_count,
                len(oq_layers),
            )
            timing_scope = "post_stream_fallback_and_weight_processing"
        else:
            logger.info(
                "[%s] Post-load online quantization and weight processing done: "
                "%.2f seconds, %d layers online-quantized",
                rank_tag(),
                pp_elapsed,
                len(oq_layers),
            )
            timing_scope = "post_load_online_quantization_and_weight_processing"
        _save_online_quant_info(
            oq_layers,
            model_name_or_path,
            pp_elapsed,
            raw_online_quant_config or {},
            timing_scope,
            peak_gpu_memory_gb,
        )

    # Measure both loading and post-processing for comparable peak memory.
    if peak_gpu_memory_gb is not None:
        if online_quant_streamer is not None:
            logger.info(
                "[%s] Peak GPU memory during streaming weight loading and "
                "online quantization: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )
        elif has_online_quant:
            logger.info(
                "[%s] Peak GPU memory during weight loading and post-load "
                "online quantization: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )
        else:
            logger.info(
                "[%s] Peak GPU memory during weight loading and "
                "post-processing: %.2f GB",
                rank_tag(),
                peak_gpu_memory_gb,
            )

    return loaded_weights_record
