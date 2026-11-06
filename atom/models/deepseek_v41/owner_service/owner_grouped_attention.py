"""Launch four original H16 native attention tiles together on gfx950."""

import functools
import hashlib
from pathlib import Path

import torch
from aiter.jit.core import AITER_CSRC_DIR, compile_ops
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.pa_sparse_prefill_opus import _auto_kv_splits


@functools.lru_cache(maxsize=1)
def build_arguments():
    """Unique private module; preserve every native compiler/dispatch setting."""
    native = Path(AITER_CSRC_DIR)
    source = Path(__file__).resolve().parent / "native"
    files = (
        source / "owner_mla_v4_prefill_opus_kernels.cu",
        source / "owner_grouped_attention_pybind.cu",
        source / "owner_grouped_attention.h",
        native / "py_itfs_cu/mla_v4_prefill_opus_kernels.cu",
        native / "include/mla_v4_prefill_opus.h",
        native / "cpp_itfs/pa/pa_ps.cuh",
        native.parent / "aiter/jit/optCompilerConfig.json",
        native / "include/opus/opus.hpp",
        native / "include/opus/dtypes.hpp",
        native / "include/opus/hip_minimal.hpp",
    )
    sha = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    # compile_ops inherits the base op's flags, including its per-source
    # *mla_v4_prefill_opus_kernels.cu early-inline exception. This source's
    # suffix matches that exact pattern; neither native source nor config is
    # edited. A different wrapper/header/config gets a different module name.
    return {
        "md_name": "module_owner_grouped_attention_" + sha[:16],
        "srcs": [str(files[0]), str(files[1])],
        "extra_include": [str(source), str(native / "py_itfs_cu")],
    }


def _build_for_call(*args, **kwargs):
    return build_arguments()


@compile_ops("module_mla_v4_prefill_opus", gen_func=_build_for_call, develop=True)
def owner_grouped_attention_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    partial_o: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int,
) -> None: ...


def attend_grouped_head_shards(
    query,
    pool,
    prefix_ids,
    prefix_indptr,
    window,
    window_ids,
    window_indptr,
    sink,
    softmax_scale,
    shard_heads,
):
    from .owner_head_attention import attend_original_head_shards

    if not (
        shard_heads == 16
        and query.ndim == 3
        and query.shape[1:] == (64, 512)
        and query.dtype == torch.bfloat16
        and query.is_contiguous()
        and get_gfx_runtime() == "gfx950"
    ):
        return attend_original_head_shards(
            query,
            pool,
            prefix_ids,
            prefix_indptr,
            window,
            window_ids,
            window_indptr,
            sink,
            softmax_scale,
            shard_heads,
        )
    rows = query.shape[0]
    # The public H64 policy is different. Keep each original H16 shard's
    # split count so each head observes the identical native reduction tree.
    splits = _auto_kv_splits("gfx950", rows, shard_heads)
    if splits not in (1, 4):
        raise ValueError("The native H16 split contract changed")
    output = torch.empty_like(query)
    shape = (rows, splits, 64) if splits > 1 else (0,)
    partial_o = torch.empty((*shape, 512), dtype=torch.float32, device=query.device)
    partial_max = torch.empty(shape, dtype=torch.float32, device=query.device)
    partial_sum = torch.empty_like(partial_max)
    owner_grouped_attention_fwd(
        query,
        pool,
        prefix_ids,
        prefix_indptr,
        window,
        window_ids,
        window_indptr,
        sink,
        partial_o,
        partial_max,
        partial_sum,
        output,
        float(softmax_scale),
        splits,
    )
    return output
