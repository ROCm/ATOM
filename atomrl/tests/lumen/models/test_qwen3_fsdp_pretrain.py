# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for the Qwen3 FSDP full-pretraining entrypoint."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# Load the example by path. ``examples`` is not a package on the ATOM pytest
# root, and the image also ships another ``examples`` tree.
_ENTRY = (
    Path(__file__).resolve().parents[3] / "lumen/examples/qwen3/train_qwen3_fsdp.py"
)
_spec = importlib.util.spec_from_file_location("train_qwen3_fsdp", _ENTRY)
_train = importlib.util.module_from_spec(_spec)
sys.modules["train_qwen3_fsdp"] = _train
assert _spec.loader is not None
_spec.loader.exec_module(_train)
_pretrain_loss = _train._pretrain_loss
_set_fsdp2_gradient_sync = _train._set_fsdp2_gradient_sync
_validation_batch_count = _train._validation_batch_count
parse_args = _train.parse_args

_BASE = [
    "--model-name-or-path",
    "Qwen/Qwen3-8B",
    "--train-data-path",
    "train.jsonl",
]


def test_pretrain_mxfp4_defaults_to_full_parameters_and_bf16_tail():
    args = parse_args(_BASE + ["--task", "pretrain", "--mode", "mxfp4"])

    assert args.lora_rank == 0
    assert args.linear_fp4 is True
    assert args.linear_fp8 is False
    assert args.first_last_layers_bf16 is True
    assert args.num_layers_at_start_in_bf16 == 0
    assert args.num_layers_at_end_in_bf16 == 5


def test_sft_defaults_remain_lora_and_no_bf16_tail():
    args = parse_args(_BASE)

    assert args.task == "sft"
    assert args.lora_rank == 16
    assert args.first_last_layers_bf16 is False


def test_pretrain_rejects_lora():
    with pytest.raises(ValueError, match="requires --lora-rank 0"):
        parse_args(_BASE + ["--task", "pretrain", "--lora-rank", "16"])


def test_init_from_scratch_is_pretrain_only():
    with pytest.raises(ValueError, match="only valid with --task pretrain"):
        parse_args(_BASE + ["--init-from-scratch"])


def test_fsdp2_accumulation_sync_recurses():
    model = MagicMock()

    _set_fsdp2_gradient_sync(model, False)
    _set_fsdp2_gradient_sync(model, True)

    assert model.set_requires_gradient_sync.call_args_list[0].args == (False,)
    assert model.set_requires_gradient_sync.call_args_list[0].kwargs == {
        "recurse": True
    }
    assert model.set_requires_gradient_sync.call_args_list[1].args == (True,)


def test_pretrain_loss_does_not_shift_dataset_labels_twice():
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 4), -20.0)
    logits.scatter_(-1, labels.unsqueeze(-1), 20.0)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.item() < 1e-6
    model.assert_called_once_with(input_ids=batch["input_ids"])


def test_pretrain_loss_upcasts_bf16_logits_for_cross_entropy():
    logits = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    model = MagicMock(return_value=SimpleNamespace(logits=logits))
    batch = {
        "input_ids": torch.randint(0, 8, (2, 3)),
        "labels": torch.randint(0, 8, (2, 3)),
    }

    loss = _pretrain_loss(model, batch, "cpu")

    assert loss.dtype == torch.float32


def test_eval_batches_must_be_positive():
    with pytest.raises(ValueError, match="--eval-batches must be >= 1"):
        parse_args(_BASE + ["--eval-batches", "0"])


def test_validation_batch_count_rejects_empty_local_loader():
    with pytest.raises(ValueError, match="no full micro-batch"):
        _validation_batch_count(0, 10, "cpu")


def test_validation_batch_count_uses_global_minimum():
    def set_remote_min(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(3)

    with (
        patch("train_qwen3_fsdp.dist.is_initialized", return_value=True),
        patch(
            "train_qwen3_fsdp.dist.all_reduce",
            side_effect=set_remote_min,
        ),
    ):
        count = _validation_batch_count(8, 10, "cpu")

    assert count == 3


def test_pretrain_dataset_import_does_not_require_megatron():
    from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

    assert PretrainTextDataset.__name__ == "PretrainTextDataset"


def test_pretrain_dataset_tokenization_is_sharded_by_input_line(tmp_path):
    from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("1\n2\n3\n4\n", encoding="utf-8")
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        encode=lambda text, add_special_tokens=False: [int(text)],
    )

    rank0 = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        rank=0,
        world_size=2,
    )
    rank1 = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        rank=1,
        world_size=2,
    )

    assert [rank0[i]["input_ids"].item() for i in range(len(rank0))] == [1, 3]
    assert [rank1[i]["input_ids"].item() for i in range(len(rank1))] == [2, 4]


def test_pretrain_dataset_stops_reading_once_sample_budget_is_met(tmp_path):
    from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

    data_path = tmp_path / "train.txt"
    data_path.write_text("".join(f"{i}\n" for i in range(1, 1001)), encoding="utf-8")
    seen = []

    def encode(text, add_special_tokens=False):
        seen.append(text)
        return [int(text)]

    tokenizer = SimpleNamespace(eos_token_id=0, encode=encode)

    ds = PretrainTextDataset(
        str(data_path),
        seq_length=1,
        tokenizer=tokenizer,
        is_hf_tokenizer=True,
        max_samples=2,
    )

    assert len(ds) == 2
    # Two samples need 4 tokens; the scan must not walk the whole file.
    assert len(seen) <= 4


def test_train_samples_is_pretrain_only():
    with pytest.raises(ValueError, match="--train-samples"):
        parse_args(_BASE + ["--task", "sft", "--train-samples", "8"])
