"""Real checkpoint TP4 acceptance of the ordinary decode and layer routes."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.models.deepseek_v41.mono.indexer_dispatch import MonoIndexerModel
from atom.sampling_params import SamplingParams

rank = int(os.environ["RANK"])
port = int(os.environ["MASTER_PORT"])
model_path = "/data/models/deepseek-ai/DeepSeek-V4.1-Flash"
test_sizes = (1, 8)
result_file = os.environ.get(
    "TP4_TEST_OUTPUT", "tp4-cross-graph-level3-acceptance.json"
)
cfg = Config(
    model=model_path,
    tensor_parallel_size=4,
    enable_expert_parallel=False,
    enforce_eager=False,
    compilation_config=CompilationConfig(
        level=3,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=list(test_sizes),
    ),
    kv_cache_dtype="bf16",
    index_cache_dtype="fp4",
    index_cache_layout="mfma",
    max_model_len=32768,
    max_num_seqs=8,
    max_num_batched_tokens=4096,
    long_prefill_token_threshold=4096,
    enable_log_stats=False,
    port=port,
)
cfg.parallel_config.data_parallel_base_port = port
preflight_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
cfg.eos_token_id = preflight_tokenizer.eos_token_id
assert all(
    isinstance(x, int)
    for x in preflight_tokenizer.encode(
        preflight_tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 17 plus 25?"}],
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode="chat",
        )
    )
)
runner = ModelRunner(rank, cfg)
runner.get_num_blocks()
runner.pool_plan = runner.pool_plan.with_paged_entries(2048)
cfg.pool_entries = dict(runner.pool_plan.entries)
runner.allocate_kv_cache(2048)
wrapped = runner.model
installed = wrapped
raw = installed.model.model
native_indexer = MonoIndexerModel(raw, raw, cfg)
# DSV4.1 supports runtime levels 0/3 and explicitly rejects torch.compile.
compiled_native = raw
compiled_native_indexer = native_indexer

# Check each successful Mono indexer's IW against the same real model input.
# Capture GPU comparisons with the forward; read the small record after replay.
from atom.models.deepseek_v41.mono.indexer import MonoIndexer

head_checks = torch.zeros((65, len(raw.layers)), dtype=torch.int32, device="cuda")
head_refs = {}
head_records = torch.zeros((65, len(raw.layers), 5), dtype=torch.int32, device="cuda")
for block in raw.layers:
    idx = getattr(block.attn, "indexer", None)
    if idx is not None and hasattr(idx, "weights_proj"):
        head_refs[id(idx)] = idx.weights_proj.weight.float().T


def check_head_weights(indexer, hidden, actual, rows, step):
    expected = (hidden[0].float() @ head_refs[id(indexer)]).bfloat16().float() / 64
    bad = (
        (~torch.isfinite(actual))
        | (~torch.isfinite(expected))
        | ((actual - expected).abs() > 1e-6 + 0.008 * expected.abs())
    )
    live = (step.batch_ids >= 0)[:, None]
    head_checks[rows, indexer.spec.layer_id].copy_((bad & live).sum().int() + 1)
    head_records[rows, indexer.spec.layer_id].copy_(
        torch.stack(
            [
                bad.sum(),
                (bad & live).sum(),
                live.sum(),
                (~torch.isfinite(actual)).sum(),
                (~torch.isfinite(expected)).sum(),
            ]
        ).int()
    )


_original_indexer_run = MonoIndexer.run


def checked_indexer_run(self, indexer, hidden, qr, qr_scale, cache, step, rope):
    success = _original_indexer_run(
        self, indexer, hidden, qr, qr_scale, cache, step, rope
    )
    if success:
        check_head_weights(indexer, hidden, self.iw[: self.rows], self.rows, step)
    return success


MonoIndexer.run = checked_indexer_run
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner

_original_layer_indexer = V41MonoDecodeRunner._indexer


def checked_layer_indexer(self, spec, attn, weights, cache, step, rope):
    _original_layer_indexer(self, spec, attn, weights, cache, step, rope)
    check_head_weights(attn.indexer, self.normed, self.iw, self.normed.shape[1], step)


V41MonoDecodeRunner._indexer = checked_layer_indexer
# Both wrappers bind hooks to the same loaded indexers. The ordinary wrapper
# activates its LazyRunner only inside its own forward, so raw/low-Mono runs
# see the hook inactive.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
prompts = [
    "What is 17 plus 25? Answer with the number.",
    "Write a Python function that returns the larger of two numbers.",
    "Name the planet closest to the Sun.",
    "A store sells 3 boxes of 8 pencils. How many pencils is that?",
]
filler = "Background notes: The room contains a desk, a lamp, and a chair.\n" * 1600
long_prompts = [
    filler + "\nIgnore the background notes and answer this question:\n" + p
    for p in prompts
]
assert min(len(tokenizer.encode(p)) for p in long_prompts) > 16384
assert max(len(tokenizer.encode(p)) for p in long_prompts) + 128 < cfg.max_model_len
reports = {}
serial = 100


def generate(model, bs, prompts, long_case, name):
    global serial
    runner.model = model
    head_checks.zero_()
    head_records.zero_()
    scheduler = Scheduler(cfg, state_runtime=runner.state_runtime)
    scheduler.block_manager._state_checkpoint_cache.clear_index()
    seqs = []
    for i in range(bs):
        serial += 1
        seq = Sequence(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompts[i % len(prompts)]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    thinking_mode="chat",
                )
            ),
            runner.block_size,
            sampling_params=SamplingParams(
                temperature=0, max_tokens=8, ignore_eos=False
            ),
            has_per_req_cache=True,
            id=serial,
        )
        scheduler.add(seq)
        seqs.append(seq)
    decode_ms = []
    for tick in range(1000 if long_case else 150):
        item = scheduler.schedule()
        if item is None:
            break
        batch, batch_seqs = item
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = runner.forward(batch)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        scheduler.postprocess(list(batch_seqs.values()), output, batch=batch)
        if batch.total_tokens_num_prefill == 0:
            decode_ms.append(elapsed)
            if int(head_checks.max().item()) > 1:
                print(
                    "HEAD_CHECK_FAILURE",
                    rank,
                    tick,
                    head_records.cpu().tolist(),
                    flush=True,
                )
                raise AssertionError(("head-weight mismatch", rank))
            if (
                rank == 0
                and len(decode_ms) <= 3
                and not name.startswith("native_repeat")
                and name != "native"
            ):
                nonzero = head_records[bs].cpu()
                print(
                    "HEAD_CHECK_RECORDS",
                    model.__class__.__name__,
                    tick,
                    [(i, v.tolist()) for i, v in enumerate(nonzero) if int(v[2])],
                    flush=True,
                )
    assert scheduler.is_finished()
    for s in seqs:
        new_tokens = list(s.token_ids)[s.num_prompt_tokens :]
        if cfg.eos_token_id in new_tokens:
            assert new_tokens.index(cfg.eos_token_id) == len(new_tokens) - 1
    seen = int((head_checks[bs] > 0).sum().item())
    if name not in ("native", "native_repeat"):
        assert seen == len(head_refs), (
            "head-weight check did not cover every indexer",
            rank,
            bs,
            seen,
            len(head_refs),
        )
    return {
        "head_weight_layers_checked": seen,
        "head_weight_max_record": int(head_checks.max().item()),
        "prompt_tokens": [s.num_prompt_tokens for s in seqs],
        "tokens": [list(s.token_ids)[s.num_prompt_tokens :] for s in seqs],
        "decode_ms": decode_ms,
        "text": [
            tokenizer.decode(list(s.token_ids)[s.num_prompt_tokens :]) for s in seqs
        ],
    }


try:
    for name, model in (
        ("native", compiled_native),
        ("native_mono_indexer", compiled_native_indexer),
        ("layer_mono_indexer", wrapped),
        ("native_repeat", compiled_native),
    ):
        hook_owner = (
            native_indexer
            if name == "native_mono_indexer"
            else installed if name == "layer_mono_indexer" else None
        )
        for block in raw.layers:
            if block.attn.indexer is not None:
                block.attn.indexer._mono = (
                    hook_owner._mono if hook_owner is not None else None
                )
        runner.model = model
        runner.req_graph = None
        stats_before = dict(torch._dynamo.utils.counters["stats"])
        runner.capture_cudagraph()
        stats_after = dict(torch._dynamo.utils.counters["stats"])
        if rank == 0:
            print("LEVEL3_CAPTURE", name, stats_before, stats_after, flush=True)
        cases = [("short", bs, prompts, False) for bs in test_sizes] + [
            ("long", bs, long_prompts, True) for bs in test_sizes
        ]
        for tag, bs, case_prompts, case_long in cases:
            result = generate(model, bs, case_prompts, case_long, name)
            result.update(
                compilation_level=3,
                cudagraph_mode="FULL",
                dynamo_stats_after_capture=stats_after,
            )
            reports[f"{tag}_{name}_bs{bs}"] = result
            if rank == 0:
                print("TP4_GRAPH_CASE", tag, name, bs, json.dumps(result), flush=True)
                Path("/app/dsv41-mono-indexer/validation", result_file).write_text(
                    json.dumps(reports, indent=2)
                )
            if name == "native_mono_indexer":
                assert native_indexer._mono.runner.widths._built
                assert not native_indexer._mono.runner.widths._refused
            if name == "layer_mono_indexer":
                assert installed.model._mono.runner.widths._built
                assert not installed.model._mono.runner.widths._refused
        if name == "native_mono_indexer":
            assert native_indexer._mono.runner is not None
        if name == "layer_mono_indexer":
            assert installed.model._mono.runner.indexer_apart
    if rank == 0:
        comparisons = {}
        for tag in ("short", "long"):
            for bs in test_sizes:
                ref = reports[f"{tag}_native_bs{bs}"]["tokens"]
                comparisons[f"{tag}_bs{bs}"] = {
                    name: {
                        "identical_sequences": sum(
                            a == b
                            for a, b in zip(
                                ref, reports[f"{tag}_{name}_bs{bs}"]["tokens"]
                            )
                        ),
                        "sequences": bs,
                    }
                    for name in (
                        "native_repeat",
                        "native_mono_indexer",
                        "layer_mono_indexer",
                    )
                }
        reports["token_comparison"] = comparisons
        Path("/app/dsv41-mono-indexer/validation", result_file).write_text(
            json.dumps(reports, indent=2)
        )
        print("TOKEN_COMPARISON", json.dumps(comparisons), flush=True)
        print(
            "HEAD_CHECK_FINAL_RECORDS",
            head_records[test_sizes[0]].cpu().tolist(),
            flush=True,
        )
        print("TP4_GRAPH_HEAD_WEIGHT_ACCEPTANCE_COMPLETE", flush=True)
finally:
    runner.exit()
