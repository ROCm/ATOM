# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER tuned GEMM tables for Qwen3.8-Flash-Next shapes.

The tables under `configs/` were produced with AITER's CK a8w8 bpreshuffle
tuner and gradlib's hipBLASLt tuner on MI308X (gfx942, 80 CUs). Rows are keyed
by gfx / CU count, so they are inert on other GPUs.

AITER merges its default table with `configs/model_configs/*` into a file
shared by every process under /tmp/aiter_configs. Processes that never build
this model (tokenizer, detokenizer) rewrite that file without our rows, and a
scheduler waiting on the merge lock reads whatever the holder wrote. So the
merge is done here, per process, into a private file, and AITER is pointed at
that single file (which it uses as is).
"""

import glob
import logging
import os
import tempfile

logger = logging.getLogger("atom")

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# (env var, AITER table name, file in _CONFIG_DIR, key columns)
_TABLES = (
    (
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
        "a8w8_bpreshuffle_tuned_gemm",
        "qwen38_flash_next_a8w8_bpreshuffle_tuned_gemm.csv",
        ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"],
    ),
    (
        "AITER_CONFIG_GEMM_BF16",
        "bf16_tuned_gemm",
        "qwen38_flash_next_bf16_tuned_gemm.csv",
        ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB",
         "bpreshuffle"],
    ),
)

_registered = False


def _source_files(env: str, aiter_configs: str, table: str) -> list[str]:
    current = os.environ.get(env)
    if current:
        return current.split(os.pathsep)
    return [os.path.join(aiter_configs, f"{table}.csv")] + sorted(
        p
        for p in glob.glob(os.path.join(aiter_configs, "model_configs", f"*{table}*.csv"))
        if "untuned" not in os.path.basename(p)
    )


def _merge(files: list[str], keys: list[str], out_path: str) -> None:
    import pandas as pd
    from aiter.jit.utils.chip_info import gfx_from_cu_num

    frames = []
    for path in files:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "gfx" not in df.columns and "cu_num" in df.columns:
            df["gfx"] = df["cu_num"].map(gfx_from_cu_num)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    # Later files win: ours is last.
    merged = merged.drop_duplicates(subset=[k for k in keys if k in merged.columns], keep="last")
    tmp = out_path + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def register_qwen4_exp_tuned_configs() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    try:
        from aiter.jit import core
    except ImportError:
        return
    aiter_configs = os.path.join(core.AITER_ROOT_DIR, "aiter", "configs")
    out_dir = os.path.join(tempfile.gettempdir(), "atom_qwen4_exp_configs", str(os.getpid()))
    os.makedirs(out_dir, exist_ok=True)
    for env, table, name, keys in _TABLES:
        ours = os.path.join(_CONFIG_DIR, name)
        if not os.path.exists(ours):
            continue
        out_path = os.path.join(out_dir, f"{table}.csv")
        _merge(_source_files(env, aiter_configs, table) + [ours], keys, out_path)
        os.environ[env] = out_path
    # AITER resolves (and caches) the BF16 table when `aiter.tuned_gemm` is
    # imported, which happens before any model is built. Registration runs
    # before the first GEMM, so dropping the cached tables is enough for the
    # next lookup to read ours.
    core.AITER_CONFIG.get_config_file.cache_clear()
    try:
        from aiter import tuned_gemm
        from aiter.ops import gemm_op_a8w8

        tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config.cache_clear()
        gemm_op_a8w8._GEMM_QUANT_TYPE_CACHE.clear()
        gemm_op_a8w8.get_GEMM_config_with_quant_type.cache_clear()
    except (ImportError, AttributeError) as e:
        logger.warning("Could not refresh AITER GEMM config caches: %s", e)
    logger.info("Registered Qwen3.8-Flash-Next AITER tuned GEMM tables in %s", out_dir)
