# SPDX-License-Identifier: MIT
"""TP4 image requests through the production scheduler, cache and model runner."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.entrypoints.openai.chat_encoders import load_custom_message_encoder
from atom.model_engine.model_runner import ModelRunner
from atom.models.deepseek_v41.image_processing import DeepseekV41ImageProcessor

from .benchmark_runtime import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--expert-backend", choices=("eager", "aiter"), default="eager")
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    config = Config(
        model=args.model,
        tensor_parallel_size=size,
        enable_expert_parallel=True,
        enforce_eager=False,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, cudagraph_capture_sizes=[1, 2, 4]
        ),
        kv_cache_dtype="fp4",
        index_cache_dtype="fp4",
        max_num_batched_tokens=1024,
        max_model_len=2048,
        max_num_seqs=4,
        long_prefill_token_threshold=args.chunk_size,
        state_checkpoint_interval_tokens=128,
        enable_log_stats=False,
        port=port,
    )
    config.hf_config.expert_backend = args.expert_backend
    config.parallel_config.data_parallel_base_port = port
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor = DeepseekV41ImageProcessor(
        config, tokenizer, load_custom_message_encoder(args.model)
    )
    runner = ModelRunner(rank, config)
    report = {
        "tp": size,
        "chunk_size": args.chunk_size,
        "expert_backend": args.expert_backend,
        "cases": [],
    }
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(512)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(512)
        runner.capture_cudagraph()
        cases = [
            (
                "single",
                ["red"],
                [
                    "What color fills this image? Answer with only the color name in English."
                ],
            ),
            (
                "multiple",
                ["blue", "red"],
                [
                    "Name the colors in these two images, in order. Answer briefly in English."
                ],
            ),
            (
                "interleaved",
                ["green", "blue"],
                [
                    "Image A:",
                    "Image B:",
                    "Give the color of A, then B. Answer briefly in English.",
                ],
            ),
        ]
        for name, colors, texts in cases:
            images = [Image.new("RGB", (256, 256), color) for color in colors]
            parts = []
            for i, image in enumerate(images):
                if len(texts) > 1:
                    parts.append({"type": "text", "text": texts[i]})
                parts.append({"type": "image", "image": image})
            parts.append({"type": "text", "text": texts[-1]})
            ids, data = processor.prepare(
                [{"role": "user", "content": parts}], images, {"thinking_mode": "chat"}
            )
            encoded_before = runner.vision_embeddings.encodes
            result = run_case(runner, [ids], 48, multimodal_data=[data])
            assert runner.vision_embeddings.encodes - encoded_before == len(images)
            assert not runner.vision_embeddings.entries
            assert not runner.vision_embeddings.leases
            result["vision_encodes"] = runner.vision_embeddings.encodes - encoded_before
            hashes = [None] * size
            torch.distributed.all_gather_object(hashes, result["output_sha256"])
            assert len(set(hashes)) == 1
            tokens = result["outputs"][0]
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(tokens)
            positions = [text.lower().find(color) for color in colors]
            passed = all(p >= 0 for p in positions) and positions == sorted(positions)
            record = {
                "name": name,
                "colors": colors,
                "output": text,
                "prompt_tokens": len(ids),
                "image_spans": data["embedding_spans"],
                "passed": passed,
                **result,
            }
            report["cases"].append(record)
            if rank == 0:
                print("VISION", json.dumps(record), flush=True)
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            assert passed, text
        report["passed"] = True
        if rank == 0:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
