"""ROCm workarounds for Flash EP decode and MTP on SGLang 0.5.20.

Ling's #2385 owns the Qwen4Exp MTP adapter (draft rewrite, QSA bridge,
NextN wrapper). These patches stay in the plugin and only cover shapes
that break on this ROCm stack after the 0.5.20 upgrade:
- EP decode MoE (2560x640 per-token FP8, token<=16) picks CK ``device_gemm``
  which rejects the problem; force the asm 1-stage path prefill already uses.
- HIP ``sgl_kernel`` tree-build and greedy-verify segfault on Flash's
  topk=1 chain. Divert only after a Qwen4Exp model is recognized.
  R1, DSV4, and GLM stay on ``sgl_kernel``; their topk=1 MTP already
  completes there.
- gfx950 BF16 hipBLASLt solutions added for Flash raise
  ``HIPBLAS_STATUS_INTERNAL_ERROR`` on the first prefill. Those shapes
  used to miss the tuned table and run the torch GEMM this accuracy job
  passed with. Other models keep their hipBLASLt rows.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")

# Set when this process loads Qwen4Exp. Kernel wrappers read it per call so
# a non-Flash server never leaves sgl_kernel.
_qwen4_exp_hip = False


def note_qwen4_exp_loaded() -> None:
    """Enable the Flash-only HIP tree/verify divert."""
    global _qwen4_exp_hip
    if _qwen4_exp_hip:
        return
    _qwen4_exp_hip = True
    logger.info(
        "Qwen4Exp HIP topk=1 tree build and greedy verify use Triton; "
        "other models stay on sgl_kernel"
    )


def note_qwen4_exp_from_identity(*labels: object) -> None:
    """Arm the HIP divert from the model being loaded.

    ``#2427`` wraps ``is_qwen_qsa`` later and returns False for Qwen4Exp
    without calling the inner hook, so this cannot live only there.
    """
    for label in labels:
        text = str(label or "")
        if "Qwen4Exp" in text or text.startswith("qwen4_exp"):
            note_qwen4_exp_loaded()
            return


def _install_ep_decode_asm_moe() -> None:
    """Use the asm 1-stage kernel for Flash EP decode."""
    try:
        import aiter.fused_moe as fused_moe_mod
        from aiter import QuantType, dtypes
    except Exception:  # noqa: BLE001 - aiter is optional until MoE loads
        return

    if getattr(fused_moe_mod, "_atom_ep_decode_asm", False):
        return
    original = fused_moe_mod.get_2stage_cfgs

    def get_2stage_cfgs(token, model_dim, inter_dim, expert, topk, *args, **kwargs):
        metadata = original(token, model_dim, inter_dim, expert, topk, *args, **kwargs)
        # Positional order after topk matches get_2stage_cfgs:
        # dtype, q_dtype_a, q_dtype_w, q_type, ...
        q_type = kwargs.get("q_type", args[3] if len(args) > 3 else None)
        q_dtype_w = kwargs.get("q_dtype_w", args[2] if len(args) > 2 else None)
        if (
            not getattr(metadata, "run_1stage", True)
            and int(token) <= 16
            and int(model_dim) == 2560
            and int(inter_dim) == 640
            and q_type == QuantType.per_Token
            and q_dtype_w == dtypes.fp8
        ):
            # Re-query just above the heuristic cutoff. 17 is not a
            # nextPow2 bucket, so a tuned CK row keyed at 16/32 cannot
            # match; the untuned path is the asm kernel prefill already
            # uses once token>16. block_m for that path ignores token.
            metadata = original(
                17,
                model_dim,
                inter_dim,
                expert,
                topk,
                *args,
                **kwargs,
            )
            if not getattr(fused_moe_mod, "_atom_ep_decode_asm_logged", False):
                fused_moe_mod._atom_ep_decode_asm_logged = True
                logger.info(
                    "Flash EP decode MoE: per-token FP8 2560x640 token<=16 "
                    "uses asm 1-stage (CK device_gemm rejects this shape)"
                )
        return metadata

    fused_moe_mod.get_2stage_cfgs = get_2stage_cfgs
    fused_moe_mod._atom_ep_decode_asm = True


def _hip_topk1_tree_builder(original, triton_impl, bitpack_mode: int):
    """Route HIP topk=1 tree builds to Triton."""

    def sgl_build_tree_kernel_efficient(*args, **kwargs):
        topk = (
            kwargs["topk"] if "topk" in kwargs else args[8] if len(args) > 8 else None
        )
        mode = (
            kwargs["tree_mask_mode"]
            if "tree_mask_mode" in kwargs
            else (args[11] if len(args) > 11 else 0)
        )
        if _qwen4_exp_hip and topk == 1 and int(mode) != bitpack_mode:
            return triton_impl(*args, **kwargs)
        return original(*args, **kwargs)

    return sgl_build_tree_kernel_efficient


def _hip_verify_tree_greedy(original, triton_impl):
    """Route Qwen4Exp HIP greedy verify to Triton; leave every other model."""

    def verify_tree_greedy_func(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        topk: int = -1,
    ):
        if not _qwen4_exp_hip:
            return original(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                target_predict,
                topk,
            )
        del topk
        triton_impl(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
        return predicts, accept_index, accept_token_num

    return verify_tree_greedy_func


def _patch_hip_topk1_tree_kernel() -> None:
    try:
        from sglang.srt.speculative import eagle_utils
    except Exception:  # noqa: BLE001 - SGLang tree builder is optional
        return
    if not getattr(eagle_utils, "_is_hip", False):
        return
    if getattr(eagle_utils, "_atom_hip_topk1_tree_triton", False):
        return
    eagle_utils.sgl_build_tree_kernel_efficient = _hip_topk1_tree_builder(
        eagle_utils.sgl_build_tree_kernel_efficient,
        eagle_utils.sgl_build_tree_kernel_triton,
        int(eagle_utils.TreeMaskMode.QLEN_ONLY_BITPACKING),
    )
    eagle_utils._atom_hip_topk1_tree_triton = True
    if not getattr(eagle_utils, "_atom_hip_verify_tree_triton", False):
        eagle_utils.verify_tree_greedy_func = _hip_verify_tree_greedy(
            eagle_utils.verify_tree_greedy_func,
            eagle_utils.verify_tree_greedy_triton,
        )
        eagle_utils._atom_hip_verify_tree_triton = True


def bf16_config_avoiding_gfx950_hipblaslt(config, *, qwen4: bool, gfx: str):
    """Return ``config`` with Flash's broken gfx950 hipBLASLt row replaced.

    The replacement is a copy. The tuned-config cache must keep the original
    row so a non-Flash model in another process, and a replay of this dict,
    still see what AITER selected.
    """
    if (
        qwen4
        and gfx == "gfx950"
        and isinstance(config, dict)
        and config.get("libtype") == "hipblaslt"
    ):
        replaced = dict(config)
        replaced["libtype"] = "torch"
        replaced["solidx"] = 0
        return replaced
    return config


def _live_gfx() -> str:
    """Arch of the GPU actually running, not AITER's ``GPU_ARCHS`` list.

    ``get_gfx()`` returns the last entry of ``gfx942;gfx950``. On MI308 that
    is gfx950, which would send this bypass down the gfx942 hipBLASLt path.
    """
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _install_gfx950_bf16_hipblaslt_bypass() -> None:
    """Send Qwen4Exp gfx950 BF16 hipBLASLt GEMMs back to torch."""
    try:
        from aiter import tuned_gemm
    except Exception:  # noqa: BLE001 - aiter is optional until GEMM loads
        return
    if getattr(tuned_gemm, "_atom_qwen4_gfx950_hipblaslt_bypass", False):
        return
    original = tuned_gemm.get_GEMM_A16W16_config

    def get_GEMM_A16W16_config(*args, **kwargs):
        config = original(*args, **kwargs)
        replaced = bf16_config_avoiding_gfx950_hipblaslt(
            config, qwen4=_qwen4_exp_hip, gfx=_live_gfx()
        )
        if replaced is not config and not getattr(
            tuned_gemm, "_atom_qwen4_gfx950_hipblaslt_logged", False
        ):
            tuned_gemm._atom_qwen4_gfx950_hipblaslt_logged = True
            logger.info(
                "Qwen4Exp gfx950 BF16 GEMM skips hipBLASLt "
                "(HIPBLAS_STATUS_INTERNAL_ERROR) and uses torch"
            )
        return replaced

    tuned_gemm.get_GEMM_A16W16_config = get_GEMM_A16W16_config
    tuned_gemm._atom_qwen4_gfx950_hipblaslt_bypass = True


def apply_qwen4_exp_rocm_patch() -> None:
    """Install Flash EP / MTP ROCm workarounds that stay outside atom core."""

    _install_ep_decode_asm_moe()
    _patch_hip_topk1_tree_kernel()
    _install_gfx950_bf16_hipblaslt_bypass()
