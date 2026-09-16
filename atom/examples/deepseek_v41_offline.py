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
    init_distributed_environment,
    initialize_model_parallel,
)
from transformers import AutoTokenizer

from atom.config import Config, use_custom_atom_config
from atom.model_engine.engram_runtime import EngramHost, EngramPrefetcher, EngramRequest
from atom.model_loader.deepseek_v41 import engram_tables
from atom.model_loader.loader import load_model
from atom.model_ops.engram import CompressedTokenizer, EngramConfig, NgramHashMapping
from atom.models.deepseek_v41.config import IndexTieBreak
from atom.models.deepseek_v41.multimodal import DeepseekV41MultimodalModel
from atom.utils.forward_context import (
    Context,
    reset_forward_context,
    set_forward_context,
)


@contextmanager
def offline_forward_context(atom_config):
    """Ask the shared LM head for a row per token, not per sequence.

    `ParallelLMHead` keeps only the last row of each sequence on a prefill
    context, and every offline caller has already chosen its own rows.
    Positions stay empty: offline layers take theirs through `step`.
    """
    set_forward_context(
        None,
        atom_config,
        Context(torch.empty(0, dtype=torch.int64), is_prefill=False),
    )
    try:
        yield
    finally:
        reset_forward_context()


@contextmanager
def load_offline_model(directory, max_length, *, index_topk_tie_break=None):
    """Keep mapped Engram tables alive for the entire offline model lifetime."""
    # The serving config, so this path is held to the same admission rules.
    # Everything here -- eager, whole-expert EP, no paging or speculation --
    # is inside that envelope rather than beside it.
    atom_config = Config(
        model=directory,
        tensor_parallel_size=torch.distributed.get_world_size(),
        enable_expert_parallel=True,
        enforce_eager=True,
        max_model_len=max_length,
    )
    config = atom_config.hf_config
    if index_topk_tie_break is not None:
        config.index_topk_tie_break = IndexTieBreak(index_topk_tie_break).value
        config._multimodal_config.text_config.index_topk_tie_break = (
            config.index_topk_tie_break
        )
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    # The shared MoE reads its config off this global, both while the layers are
    # built and while they run. `ModelRunner` sets it once; with no runner here,
    # scope it to the model's lifetime.
    with use_custom_atom_config(atom_config):
        try:
            with torch.device("cuda"):
                # The multimodal class unconditionally, as serving builds it:
                # it is the one that owns the vision tensors, and a text-only
                # backbone would leave them with no parameter to land in --
                # reported as unroutable, or worse, skipped quietly.
                model = DeepseekV41MultimodalModel(config, max_length=max_length)
        finally:
            torch.set_default_dtype(previous)
        load_model(model, directory, config)
        with engram_tables(directory, config) as tables:
            engram_config = EngramConfig.from_hf(config.to_dict())
            mapping = NgramHashMapping(
                engram_config,
                CompressedTokenizer(
                    tokenizer, expected_size=engram_config.compressed_vocab_size
                ),
            )
            prefetcher = EngramPrefetcher(mapping, tables)
            host = EngramHost(
                prefetcher,
                max_length,
                engram_config.num_hash_heads,
                engram_config.head_dim,
                model.embed.weight.device,
            )
            try:
                with offline_forward_context(atom_config):
                    yield model, tokenizer, mapping, host
            finally:
                host.shutdown()


def prepare_engram(token_ids, position, history, mapping, host, *, token_mask=None):
    tokens = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
    mask = (
        None
        if token_mask is None
        else np.asarray(token_mask, dtype=np.bool_).reshape(1, -1)
    )
    request = EngramRequest(
        0,
        0,
        position,
        tuple(tokens[0]),
        tuple(history[0]),
        None if mask is None else tuple(mask[0]),
    )
    host.stage_embeddings([request])
    host.wait_for_embeddings()
    embeddings = {
        layer: host.embeddings(layer).unsqueeze(0) for layer in host.layer_ids
    }
    next_history = mapping.advance_history(
        history, mapping.compress_tokens(tokens, mask)
    )
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
