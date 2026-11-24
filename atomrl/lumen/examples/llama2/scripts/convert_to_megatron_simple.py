# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Convert HF LLaMA checkpoint to Megatron-LM-AMD format.

Simplified version that skips RMSNorm and TE checks for compatibility
with the lumen_unit_test container.
"""

import importlib
import math
import os
import sys

megatron_root = os.environ.get("MEGATRON_ROOT", "/workspace/megatron_lm")
ckpt_tools = os.path.join(megatron_root, "tools", "checkpoint")
if ckpt_tools not in sys.path:
    sys.path.insert(0, ckpt_tools)

rope_utils = importlib.import_module(
    "megatron.core.models.common.embeddings.rope_utils"
)
fused_apply_rotary_pos_emb = importlib.import_module(
    "apex.transformer.functional.fused_rope"
).fused_apply_rotary_pos_emb

rope_utils.fused_apply_rotary_pos_emb = fused_apply_rotary_pos_emb
print("[convert] Registered apex fused_apply_rotary_pos_emb into rope_utils")

MegatronCheckpointSaverBase = importlib.import_module(
    "saver_base"
).MegatronCheckpointSaverBase
_ConverterFakeProcessGroup = importlib.import_module("utils")._ConverterFakeProcessGroup

_orig_insert = MegatronCheckpointSaverBase.insert_megatron_path_and_check_te


def _patched_insert(self):
    _megatron_root = os.environ.get("MEGATRON_ROOT", "/workspace/megatron_lm")
    sys.path.append(os.path.abspath(_megatron_root))
    if self.args.megatron_path is not None:
        sys.path.insert(0, self.args.megatron_path)


MegatronCheckpointSaverBase.insert_megatron_path_and_check_te = _patched_insert

_orig_init_env = MegatronCheckpointSaverBase.initialize_megatron_env


def _patched_init_env(self):
    _orig_init_env(self)
    # megatron.core.mpu was renamed to megatron.core.parallel_state in newer versions
    try:
        mpu = importlib.import_module("megatron.core.mpu")
    except ModuleNotFoundError:
        mpu = importlib.import_module("megatron.core.parallel_state")

    fake_dp = _ConverterFakeProcessGroup(size=1)
    mpu._DATA_PARALLEL_GROUP = fake_dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = fake_dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp
    mpu._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = fake_dp


MegatronCheckpointSaverBase.initialize_megatron_env = _patched_init_env

_orig_load_ckpt_args = MegatronCheckpointSaverBase._load_checkpoint_args


def _patched_load_ckpt_args(self, margs):
    margs = _orig_load_ckpt_args(self, margs)
    if margs.tensor_model_parallel_size > 1 and getattr(margs, "vocab_size", None):
        divisor = getattr(margs, "make_vocab_size_divisible_by", 128)
        multiple = divisor * margs.tensor_model_parallel_size
        recalc = int(math.ceil(margs.vocab_size / multiple) * multiple)
        if getattr(margs, "padded_vocab_size", None) != recalc:
            print(
                f"[convert] Fixing padded_vocab_size: {margs.padded_vocab_size} -> {recalc}"
            )
            margs.padded_vocab_size = recalc
    # TE is not installed; force local spec so gpt_builder uses get_gpt_layer_local_spec().
    # The checkpoint args may carry transformer_impl="transformer_engine" which would crash.
    margs.transformer_impl = "local"
    print("[convert] Forced transformer_impl=local in checkpoint args")
    return margs


MegatronCheckpointSaverBase._load_checkpoint_args = _patched_load_ckpt_args

_mcore_utils = importlib.import_module("megatron.core.utils")
_pstate = importlib.import_module("megatron.core.parallel_state")

_orig_get_tp_if_none = _mcore_utils.get_tensor_model_parallel_group_if_none


def _patched_get_tp_if_none(tp_group, is_expert=False, check_initialized=True):
    if tp_group is not None:
        return tp_group
    grp = _pstate._TENSOR_MODEL_PARALLEL_GROUP
    if grp is not None:
        return grp
    return _orig_get_tp_if_none(tp_group, is_expert, check_initialized)


_mcore_utils.get_tensor_model_parallel_group_if_none = _patched_get_tp_if_none

_orig_get_pg_size = _mcore_utils.get_pg_size
_orig_get_pg_rank = _mcore_utils.get_pg_rank


def _patched_get_pg_size(group=None):
    if group is not None and hasattr(group, "size"):
        return group.size()
    return _orig_get_pg_size(group)


def _patched_get_pg_rank(group=None):
    if group is not None and hasattr(group, "rank"):
        return group.rank()
    return _orig_get_pg_rank(group)


_mcore_utils.get_pg_size = _patched_get_pg_size
_mcore_utils.get_pg_rank = _patched_get_pg_rank

_tp_layers = importlib.import_module("megatron.core.tensor_parallel.layers")

_tp_layers.get_pg_size = _patched_get_pg_size
_tp_layers.get_pg_rank = _patched_get_pg_rank
_tp_layers.get_tensor_model_parallel_group_if_none = _patched_get_tp_if_none
print("[convert] Patched TP group resolution for non-distributed saver")

_mega_args = importlib.import_module("megatron.training.arguments")

_orig_validate = _mega_args.validate_args


def _patched_validate(args, defaults=None):
    if defaults is None:
        defaults = {}
    saved_sp = args.sequence_parallel
    args.sequence_parallel = True
    # Force local (non-TE) transformer spec so model building uses get_gpt_layer_local_spec().
    # TE is not installed in this container; using local spec avoids TESpecProvider entirely.
    args.transformer_impl = "local"
    result = _orig_validate(args, defaults)
    target = result if result is not None else args
    target.sequence_parallel = saved_sp
    target.transformer_impl = "local"
    return target


_mega_args.validate_args = _patched_validate
print("[convert] Patched saver: skip TE check + fake DP groups + auto SP")

_loader_mod = importlib.import_module("loader_llama_mistral")


def _noop_verify():
    pass


_loader_mod.verify_transformers_version = _noop_verify
print("[convert] Bypassed transformers version check")

print("[convert] No output-layer padding (using --make-vocab-size-divisible-by 1)")

if __name__ == "__main__":
    importlib.import_module("convert").main()
