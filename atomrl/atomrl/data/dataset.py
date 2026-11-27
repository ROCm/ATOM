# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Dataset preprocessing through the model's official HF chat template."""

import hashlib
import json
import logging
import multiprocessing as mp
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from atomrl.data.sft_dataset import normalize_messages, tokenize_chat_messages

logger = logging.getLogger(__name__)

_PREPROCESSING_VERSION = "hf-chat-template-v2"
_worker_state = {}


def _dataset_fingerprint(dataset_path: str) -> str:
    """Fingerprint directory datasets so modified shards cannot reuse a cache."""
    if not os.path.isdir(dataset_path):
        return ""
    digest = hashlib.sha256()
    for root, _, filenames in os.walk(dataset_path):
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            relative = os.path.relpath(path, dataset_path).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_fingerprint(tokenizer, tokenizer_path: str) -> str:
    """Identify the effective template and tokenizer revision for cache safety."""
    template = tokenizer.chat_template or ""
    commit = tokenizer.init_kwargs.get("_commit_hash", "")
    local_files = []
    if os.path.isdir(tokenizer_path):
        for name in (
            "chat_template.jinja",
            "tokenizer_config.json",
            "tokenizer.json",
            "tiktoken.model",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
        ):
            path = os.path.join(tokenizer_path, name)
            if os.path.isfile(path):
                stat = os.stat(path)
                local_files.append((name, stat.st_size, stat.st_mtime_ns))
    payload = {
        "version": _PREPROCESSING_VERSION,
        "path": tokenizer_path,
        "commit": commit,
        "template": template,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "assistant_id": tokenizer.convert_tokens_to_ids("<|im_assistant|>"),
        "middle_id": tokenizer.convert_tokens_to_ids("<|im_middle|>"),
        "end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "local_files": local_files,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _init_worker(
    tokenizer_path: str,
    max_length: int,
    last_turn_loss_only: str,
    min_loss_tokens: int,
) -> None:
    _worker_state["tokenizer"] = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    _worker_state["max_length"] = max_length
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens


def _last_turn_only(messages: list[dict]) -> bool:
    configured = _worker_state["last_turn_loss_only"]
    if configured != "auto":
        return configured == "true"
    normalized = normalize_messages(messages)
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        if message.get("reasoning_content"):
            return True
        content = message.get("content")
        if isinstance(content, str):
            start = content.find("<think>")
            end = content.find("</think>", start + len("<think>"))
            if (
                start >= 0
                and end >= 0
                and content[start + len("<think>") : end].strip()
            ):
                return True
    return False


def _tokenize_single(messages: list[dict]) -> dict | None:
    input_ids, loss_mask, _ = tokenize_chat_messages(
        _worker_state["tokenizer"],
        messages,
        _worker_state["max_length"],
        last_turn_only=_last_turn_only(messages),
    )
    if int(loss_mask.sum()) < max(1, _worker_state["min_loss_tokens"]):
        return None
    return {
        "input_ids": input_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
    }


def load_and_preprocess_dataset(
    dataset_path: str,
    tokenizer_path: str,
    max_length: int,
    chat_template: str = "hf",
    seed: int = 42,
    last_turn_loss_only: str = "false",
    min_loss_tokens: int = 0,
    num_workers: int = 16,
    cache_dir: str = "/dev/shm/atomrl_cache",
) -> list:
    """Load conversations, render the official template, and cache token masks."""
    if isinstance(last_turn_loss_only, bool):
        last_turn_loss_only = str(last_turn_loss_only).lower()
    if last_turn_loss_only not in {"true", "false", "auto"}:
        raise ValueError("last_turn_loss_only must be true, false, or auto")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer, tokenizer_path)
    file_stat = ""
    if os.path.isfile(dataset_path):
        stat = os.stat(dataset_path)
        file_stat = f"-{stat.st_size}-{stat.st_mtime_ns}"
    cache_payload = (
        f"{dataset_path}{file_stat}-{max_length}-{chat_template}"
        f"-ltlo={last_turn_loss_only}-mlt={min_loss_tokens}"
        f"-seed={seed}-dataset={_dataset_fingerprint(dataset_path)}"
        f"-tokenizer={tokenizer_fingerprint}"
    )
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    cache_subdir = os.path.join(cache_dir, "tokenized_dataset")
    cache_path = os.path.join(cache_subdir, f"{cache_key}.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        logger.info("Loaded %d cached samples from %s", len(data), cache_path)
        return data

    from datasets import load_dataset

    if os.path.isfile(dataset_path) and dataset_path.endswith((".jsonl", ".json")):
        source = load_dataset("json", data_files=dataset_path, split="train")
    elif os.path.isfile(dataset_path) and dataset_path.endswith(".parquet"):
        source = load_dataset("parquet", data_files=dataset_path, split="train")
    else:
        source = load_dataset(dataset_path, split="train")
    source = source.shuffle(seed=seed)
    conversations = [
        sample["conversations"]
        for sample in tqdm(source, desc="Loading samples")
        if isinstance(sample.get("conversations"), list)
    ]

    worker_args = (
        tokenizer_path,
        max_length,
        last_turn_loss_only,
        min_loss_tokens,
    )
    if num_workers <= 1:
        _init_worker(*worker_args)
        results = [
            _tokenize_single(conversation)
            for conversation in tqdm(conversations, desc="Tokenizing")
        ]
    else:
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=worker_args,
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(_tokenize_single, conversations, chunksize=64),
                    total=len(conversations),
                    desc="Tokenizing",
                )
            )

    data = [result for result in results if result is not None]
    skipped = len(results) - len(data)
    if skipped:
        logger.warning("Skipped %d samples with empty loss masks", skipped)
    os.makedirs(cache_subdir, exist_ok=True)
    torch.save(data, cache_path)
    logger.info("Saved %d preprocessed samples to %s", len(data), cache_path)
    return data
