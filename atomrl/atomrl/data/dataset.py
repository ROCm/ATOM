"""Dataset preprocessing pipeline with multiprocessing and caching.

Preprocesses ShareGPT-format datasets using KimiK25Parser to produce
input_ids + packed_loss_mask, cached as .pt files for fast reloading.
"""

import hashlib
import logging
import multiprocessing as mp
import os

import torch
from tqdm import tqdm

from atomrl.data.kimi_k25_parser import (
    KimiK25Parser,
    has_thinking_content,
    normalize_conversation,
    pack_loss_mask,
    serialize_packed_loss_mask,
)

logger = logging.getLogger(__name__)

_worker_state = {}


def _init_worker(tokenizer_path, max_length, last_turn_loss_only, min_loss_tokens):
    """Per-worker initializer — loads tokenizer once."""
    from transformers import AutoTokenizer

    _worker_state["tokenizer"] = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    _worker_state["parser"] = KimiK25Parser(_worker_state["tokenizer"])
    _worker_state["max_length"] = max_length
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens


def _resolve_last_turn_loss_only(messages):
    """Resolve last_turn_loss_only, supporting "auto" mode.

    When "auto", returns True if the conversation contains real thinking
    content (non-empty <think> blocks), so that loss is only computed on
    the last assistant turn where the model actually reasons.
    """
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        return has_thinking_content(messages)
    return bool(ltlo)


def _tokenize_single(messages):
    """Worker function — tokenize one sample, return dict or None."""
    parser = _worker_state["parser"]
    max_length = _worker_state["max_length"]
    min_loss_tokens = _worker_state.get("min_loss_tokens", 1)

    messages = normalize_conversation(messages)
    last_turn_only = _resolve_last_turn_loss_only(messages)
    input_ids, loss_mask = parser.parse(
        messages, max_length, last_turn_only=last_turn_only
    )

    if loss_mask.sum() < max(1, min_loss_tokens):
        return None

    packed = pack_loss_mask(loss_mask)
    return {
        "input_ids": input_ids.tolist(),
        "packed_loss_mask": serialize_packed_loss_mask(packed),
    }


def load_and_preprocess_dataset(
    dataset_path: str,
    tokenizer_path: str,
    max_length: int,
    chat_template: str = "kimi-k25",
    seed: int = 42,
    last_turn_loss_only: str = "false",
    min_loss_tokens: int = 0,
    num_workers: int = 16,
    cache_dir: str = "/dev/shm/atomrl_cache",
) -> list:
    """Load, tokenize, and cache a conversation dataset.

    Args:
        last_turn_loss_only: "true", "false", or "auto". When "auto", only
            computes loss on the last assistant turn for samples that contain
            real <think> content.
        min_loss_tokens: Skip samples with fewer supervised tokens than this.

    Returns list of dicts with keys: input_ids (List[int]), packed_loss_mask (str).
    """
    if isinstance(last_turn_loss_only, bool):
        last_turn_loss_only = str(last_turn_loss_only).lower()

    file_stat = ""
    if os.path.isfile(dataset_path):
        st = os.stat(dataset_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    cache_params = (
        f"{os.path.basename(dataset_path)}-{dataset_path}{file_stat}"
        f"-{tokenizer_path}-{max_length}-{chat_template}"
        f"-ltlo={last_turn_loss_only}-mlt={min_loss_tokens}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_subdir = os.path.join(cache_dir, "tokenized_dataset")
    cache_path = os.path.join(cache_subdir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        logger.info("Loading preprocessed dataset from cache: %s", cache_path)
        data = torch.load(cache_path, weights_only=False)
        logger.info("Loaded %d cached samples", len(data))
        return data

    logger.info("Preprocessing dataset (cache will be saved to %s)", cache_path)

    from datasets import load_dataset as _load_dataset

    if os.path.isfile(dataset_path):
        if dataset_path.endswith((".jsonl", ".json")):
            ds = _load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            ds = _load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            ds = _load_dataset(dataset_path, split="train")
    elif os.path.isdir(dataset_path):
        ds = _load_dataset(dataset_path, split="train")
    else:
        ds = _load_dataset(dataset_path, split="train")

    ds = ds.shuffle(seed=seed)

    raw_conversations = []
    for sample in tqdm(ds, desc="Loading samples"):
        convs = sample.get("conversations")
        if convs and isinstance(convs, list):
            raw_conversations.append(convs)

    logger.info(
        "Loaded %d samples, tokenizing with %d workers...",
        len(raw_conversations),
        num_workers,
    )

    if num_workers <= 1:
        _init_worker(tokenizer_path, max_length, last_turn_loss_only, min_loss_tokens)
        results = [
            _tokenize_single(c) for c in tqdm(raw_conversations, desc="Tokenizing")
        ]
    else:
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_path, max_length, last_turn_loss_only, min_loss_tokens),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_tokenize_single, raw_conversations, chunksize=64),
                    total=len(raw_conversations),
                    desc="Tokenizing",
                )
            )

    data = [r for r in results if r is not None]
    skipped = len(results) - len(data)
    if skipped:
        logger.warning("Skipped %d samples (empty or zero loss mask)", skipped)

    os.makedirs(cache_subdir, exist_ok=True)
    torch.save(data, cache_path)
    logger.info("Saved %d preprocessed samples to %s", len(data), cache_path)

    return data
