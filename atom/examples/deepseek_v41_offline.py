# SPDX-License-Identifier: MIT
"""Eager DeepSeek-V4.1 text completion with native weights and host Engram.

Run with torchrun, e.g. --nproc_per_node=8 -m atom.examples.deepseek_v41_offline.
This raw-prefix example has no chat template, paging, graph or speculation.
"""

import argparse
import os
from contextlib import contextmanager

import numpy as np
import torch
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from transformers import AutoTokenizer

from atom.config import get_hf_config
from atom.model_engine.engram_runtime import EngramHost, EngramPrefetcher, EngramRequest
from atom.model_ops.engram import CompressedTokenizer, EngramConfig, NgramHashMapping
from atom.models.deepseek_v41.config import IndexTieBreak
from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM
from atom.models.deepseek_v41.weights import (
    CheckpointReader,
    build_weight_manifest,
    checkpoint_schema,
)


@contextmanager
def load_offline_model(directory, max_length, *, index_topk_tie_break=None):
    """Keep mapped Engram tables alive for the entire offline model lifetime."""
    config = get_hf_config(directory)
    if index_topk_tie_break is not None:
        config.index_topk_tie_break = IndexTieBreak(index_topk_tie_break).value
        config._multimodal_config.text_config.index_topk_tie_break = (
            config.index_topk_tie_break
        )
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    group = get_tp_group()
    schema = checkpoint_schema(config)
    manifest = build_weight_manifest(
        schema,
        tp_rank=group.rank_in_group,
        tp_size=group.world_size,
        ep_rank=group.rank_in_group,
        ep_size=group.world_size,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = DeepseekV41ForCausalLM(config, max_length=max_length)
    finally:
        torch.set_default_dtype(previous)
    with CheckpointReader(directory, schema) as reader:
        reader.load_parameters(model, manifest)
        model.process_weights_after_loading()
        engram_config = EngramConfig.from_hf(config.to_dict())
        mapping = NgramHashMapping(
            engram_config,
            CompressedTokenizer(
                tokenizer, expected_size=engram_config.compressed_vocab_size
            ),
        )
        prefetcher = EngramPrefetcher(mapping, reader.engram_tables(config))
        host = EngramHost(
            prefetcher,
            max_length,
            engram_config.num_hash_heads,
            engram_config.head_dim,
            model.embed.weight.device,
        )
        try:
            yield model, tokenizer, mapping, host
        finally:
            host.shutdown()


def prepare_engram(token_ids, position, history, mapping, host):
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
    request = EngramRequest(0, 0, position, tuple(tokens[0]), tuple(history[0]))
    host.stage_embeddings([request])
    host.wait_for_embeddings()
    embeddings = {
        layer: host.embeddings(layer).unsqueeze(0) for layer in host.layer_ids
    }
    next_history = mapping.advance_history(history, mapping.compress_tokens(tokens))
    return embeddings, next_history


def initialize_parallel():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    initialize_model_parallel(tensor_model_parallel_size=size)
    return rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--index-topk-tie-break",
        choices=[policy.value for policy in IndexTieBreak],
        help="Equal index scores prefer the smaller or larger position; defaults to the model config",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    rank = initialize_parallel()
    try:
        with load_offline_model(
            args.model, args.max_length, index_topk_tie_break=args.index_topk_tie_break
        ) as (model, tokenizer, mapping, host):
            tokens = tokenizer.encode(args.prompt, add_special_tokens=True)
            if not tokens or len(tokens) + args.max_new_tokens > args.max_length:
                raise ValueError(
                    "Prompt and generation exceed the offline context capacity"
                )
            cache = model.new_cache(1)
            history = np.full(
                (1, mapping.config.max_ngram_size - 1), -1, dtype=np.int64
            )
            generated = []
            for _ in range(args.max_new_tokens):
                embeddings, next_history = prepare_engram(
                    tokens, cache.position, history, mapping, host
                )
                logits = model(torch.tensor([tokens], device="cuda"), cache, embeddings)
                history = next_history
                token = int(logits.argmax(-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
                tokens = [token]
            if rank == 0:
                print(tokenizer.decode(generated), flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
