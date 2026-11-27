# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""HF checkpoint → Megatron torch_dist conversion (Lumen-local)."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from mbridge import AutoBridge
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    save_checkpoint,
)
from megatron.training.training import get_model

TOOLS_DIR = Path(__file__).resolve().parent
LUMEN_DIR = TOOLS_DIR.parent.parent.parent
ATOMRL_DIR = LUMEN_DIR.parent
if str(ATOMRL_DIR) not in sys.path:
    sys.path.insert(0, str(ATOMRL_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

MEGATRON_PATH = os.environ.get("MEGATRON_PATH")
if MEGATRON_PATH and MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

import lumen_mbridge  # noqa: F401 — register DeepseekV4Bridge
from megatron_convert import (
    get_convert_model_provider,
    init_megatron_for_convert,
    set_default_megatron_args,
)

logger = logging.getLogger(__name__)


class ROCmFileSystemWriterAsync:
    """HIP: avoid non_blocking pinned memory that breaks fork after save."""

    @staticmethod
    def preload_tensors(*args, **kwargs):
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )

        if torch.version.hip:
            print("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)
        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)


def add_conversion_args(parser):
    parser.add_argument(
        "--hf-checkpoint", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="Weight export mode for downstream inference stacks.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except argparse.ArgumentError:
        pass
    return parser


def get_convert_args():
    args = parse_args(add_conversion_args)
    args = set_default_megatron_args(args)

    args.debug_deterministic_collective = False
    args.enable_witness = False
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = world_size

    assert args.pipeline_model_parallel_size <= args.num_layers, (
        f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be <= "
        f"num_layers {args.num_layers}."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if (
        args.pipeline_model_parallel_size == 1
        and world_size > 1
        and getattr(args, "expert_model_parallel_size", 1) == 1
    ):
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)
            if args.decoder_last_pipeline_num_layers > 0:
                break
            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find valid PP size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using PP={args.pipeline_model_parallel_size}, "
        f"decoder_last_pipeline_num_layers={args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def _print_memory(msg: str) -> None:
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    logger.info(
        "[Rank %s] Memory-Usage %s: gpu=%s total_GB=%.2f free_GB=%.2f",
        dist.get_rank(),
        msg,
        device,
        total / (1024**3),
        free / (1024**3),
    )


def main() -> None:
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        import megatron.core.dist_checkpointing.strategies.torch as torch_strategy_module

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        torch_strategy_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    args = get_convert_args()
    init_megatron_for_convert(args)
    model = get_model(
        get_convert_model_provider(args),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )

    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    _print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)

    dist.barrier()

    if dist.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
