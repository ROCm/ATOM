"""Shared Qwen4Exp checkpoint-name mapping for model and plugin loaders."""

from typing import Any


def qwen4_exp_packed_modules_mapping(
    base_mapping: dict[str, Any], split_ngram_parts: int
) -> dict[str, Any]:
    return {
        **base_mapping,
        **{
            f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
            for shard in range(int(split_ngram_parts))
        },
    }
