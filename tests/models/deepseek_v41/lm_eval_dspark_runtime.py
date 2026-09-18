# SPDX-License-Identifier: MIT
"""Paired lm_eval on the scheduler path, including teacher-forced verification.

Likelihood requests force their supplied continuation through the sampler and
draft proposals, while scoring the untouched target logits. Generation requests
use the native drafter and sampler. State commit and scheduling are unchanged.
"""

import argparse
import copy
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from atom.models.deepseek_v41.config import validate_runtime_config
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from transformers import AutoTokenizer

from atom.config import (
    CompilationConfig,
    Config,
    CUDAGraphMode,
    DSparkConfig,
    SpeculativeConfig,
)
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template as render_chat_prompt,
)
from atom.entrypoints.openai.chat_encoders import (
    load_custom_message_encoder,
)
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils.forward_context import get_forward_context

from .reference import MANIFEST
from .validate_dspark_runtime import diagnostic_config


class RuntimeLM(LM):
    def __init__(
        self,
        runner,
        tokenizer,
        directory,
        *,
        chat=False,
        state_audit=None,
        generation_progress=False,
    ):
        super().__init__()
        self.runner, self.tokenizer, self.directory = runner, tokenizer, directory
        # This harness instantiates ModelRunner directly. Mirror the tokenizer
        # initialization normally owned by LLMEngine before creating Scheduler.
        if tokenizer.eos_token_id is None:
            raise ValueError("Evaluation requires the model's EOS token")
        runner.config.bos_token_id = tokenizer.bos_token_id
        runner.config.eos_token_id = tokenizer.eos_token_id
        runner.config.stop_token_ids = [
            token
            for token in runner.config.stop_token_ids
            if token != tokenizer.eos_token_id
        ]
        self.generation_progress = generation_progress
        self.progress_records = []
        self.state_audit = state_audit
        self.generation_records = []
        self.chat = chat
        self.encoder = (
            load_custom_message_encoder(runner.config.model) if chat else None
        )
        if chat and (self.encoder is None or self.encoder.name != "encoding_dsv41"):
            raise ValueError("Chat evaluation requires the published V4.1 encoder")
        self.scored = 0

    @property
    def tokenizer_name(self):
        return self.runner.config.model

    def chat_template(self, chat_template=False):
        return "encoding_dsv41:chat" if self.chat else ""

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        if not add_generation_prompt:
            raise NotImplementedError(
                "Assistant-prefill chat evaluation is not supported"
            )
        return render_chat_prompt(
            self.tokenizer, self.encoder, chat_history, thinking_mode="chat"
        )

    def loglikelihood(self, requests):
        results = []
        for start in range(0, len(requests), 4):
            group = requests[start : start + 4]
            prompts, labels = [], []
            for request in group:
                context, continuation = request.args
                spaces = len(context) - len(context.rstrip())
                if spaces:
                    context, continuation = (
                        context[:-spaces],
                        context[-spaces:] + continuation,
                    )
                if context:
                    prefix = self.tokenizer.encode(
                        context, add_special_tokens=not self.chat
                    )
                    full = self.tokenizer.encode(
                        context + continuation, add_special_tokens=not self.chat
                    )
                else:
                    prefix = [self.tokenizer.bos_token_id]
                    full = prefix + self.tokenizer.encode(
                        continuation, add_special_tokens=False
                    )
                suffix = full[len(prefix) :]
                if not suffix or len(full) > self.runner.config.max_model_len:
                    raise ValueError(
                        "Invalid likelihood continuation or context capacity"
                    )
                prompts.append(full[: len(prefix)])
                labels.append(suffix)
            _, scores = self.execute(prompts, [len(x) for x in labels], labels=labels)
            for request, prefix, suffix, logits in zip(group, prompts, labels, scores):
                expected = torch.tensor(suffix, dtype=torch.long)
                log_probs = logits.log_softmax(-1)
                score = float(log_probs.gather(-1, expected[:, None]).sum())
                greedy = bool((logits.argmax(-1) == expected).all())
                results.append((score, greedy))
                key = hashlib.sha256(
                    json.dumps(request.args, ensure_ascii=False).encode()
                ).hexdigest()
                if self.runner.rank == 0:
                    torch.save(
                        {"prompt": prefix, "labels": suffix, "logits": logits},
                        self.directory / f"{key}.pt",
                    )
                self.cache_hook.add_partial("loglikelihood", request.args, results[-1])
            self.scored += len(group)
            if self.runner.rank == 0:
                print(json.dumps({"scored_requests": self.scored}), flush=True)
        return results

    def generate_until(self, requests):
        results = []
        for start in range(0, len(requests), 4):
            group = requests[start : start + 4]
            prompts = [
                self.tokenizer.encode(req.args[0], add_special_tokens=not self.chat)
                for req in group
            ]
            limits = [req.args[1].get("max_gen_toks", 256) for req in group]
            outputs, _ = self.execute(
                prompts, limits, stops=[req.args[1].get("until", []) for req in group]
            )
            for req, output in zip(group, outputs):
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                stops = req.args[1].get("until", [])
                for stop in [stops] if isinstance(stops, str) else stops:
                    if stop:
                        text = text.split(stop)[0]
                results.append(text)
                self.cache_hook.add_partial("generate_until", req.args, text)
            if self.runner.rank == 0:
                if self.generation_progress:
                    self.progress_records.extend(
                        {
                            "doc_id": req.doc_id,
                            "arguments": req.args,
                            "response": response,
                            **record,
                        }
                        for req, response, record in zip(
                            group,
                            results[-len(group) :],
                            self.generation_records[-len(group) :],
                        )
                    )
                    path = self.directory.parent / "generation_progress.json"
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(
                            {
                                "completed": False,
                                "note": "Generation progress only; final lm_eval scoring is pending.",
                                "records": self.progress_records,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    temporary.replace(path)
                print(json.dumps({"generated_requests": len(results)}), flush=True)
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Use explicit continuations for paired verification")

    @torch.inference_mode()
    def execute(self, prompts, limits, labels=None, stops=None):
        runner = self.runner
        if any(
            len(prompt) + limit > runner.config.max_model_len
            for prompt, limit in zip(prompts, limits)
        ):
            raise ValueError("Evaluation exceeds runtime context capacity")
        scheduler = Scheduler(runner.config, state_runtime=runner.state_runtime)
        terminators = []
        for values in stops or [[] for _ in prompts]:
            values = [values] if isinstance(values, str) else values
            encoded = [
                self.tokenizer.encode(value, add_special_tokens=False)
                for value in values
                if value
            ]
            terminators.append(
                [[token] for token in runner.config.stop_token_ids]
                + [tokens for tokens in encoded if tokens]
            )
        seqs = [
            Sequence(
                prompt,
                runner.block_size,
                SamplingParams(
                    temperature=0, max_tokens=limit, ignore_eos=labels is not None
                ),
                has_per_req_cache=True,
                num_draft_tokens=runner.num_spec_tokens,
                stop_token_sequences=None if labels is not None else terminators[i],
            )
            for i, (prompt, limit) in enumerate(zip(prompts, limits))
        ]
        if self.state_audit is not None:
            self.state_audit.bind(seqs)
        sequences = {seq.id: i for i, seq in enumerate(seqs)}
        saved = [{} for _ in seqs]
        original_logits = runner.model.compute_logits
        original_propose = (
            None if not hasattr(runner, "drafter") else runner.drafter.propose
        )

        def force_logits(hidden):
            logits = original_logits(hidden)
            fc = get_forward_context()
            forced = torch.full_like(logits, -10000)
            for i, span in enumerate(fc.attn_metadata.step.requests):
                index = sequences[span.request_id]
                full = prompts[index] + labels[index]
                offsets = (
                    [span.length - 1] if fc.context.is_prefill else range(span.length)
                )
                for offset in offsets:
                    row = i if fc.context.is_prefill else span.offset + offset
                    position = span.position + offset + 1
                    token = (
                        full[position]
                        if position < len(full)
                        else self.tokenizer.eos_token_id
                    )
                    forced[row, token] = 0
                    label_index = position - len(prompts[index])
                    if 0 <= label_index < len(labels[index]):
                        value = logits[row].detach().float().cpu()
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(
                                "Non-finite target verification logits"
                            )
                        saved[index][label_index] = value
            return forced

        def force_draft(*args, **kwargs):
            proposed = original_propose(*args, **kwargs)
            metadata = get_forward_context().attn_metadata
            positions = kwargs["target_positions"][
                kwargs["last_token_indices"]
            ].tolist()
            for i, (span, position) in enumerate(
                zip(metadata.step.requests, positions)
            ):
                index = sequences[span.request_id]
                full = prompts[index] + labels[index]
                tokens = [
                    full[p] if p < len(full) else self.tokenizer.eos_token_id
                    for p in range(position + 2, position + 2 + proposed.shape[1])
                ]
                proposed[i] = torch.tensor(
                    tokens, device=proposed.device, dtype=proposed.dtype
                )
            return proposed

        if labels is not None:
            runner.model.compute_logits = force_logits
            if original_propose is not None:
                runner.drafter.propose = force_draft
        try:
            for seq in seqs:
                scheduler.add(seq)
            while not scheduler.is_finished():
                item = scheduler.schedule()
                if item is None:
                    raise RuntimeError("Evaluation scheduler stalled")
                batch, active = item
                output = runner.forward(batch)
                scheduler.postprocess(list(active.values()), output, batch=batch)
                if self.state_audit is not None:
                    self.state_audit.observe()
            scheduler.block_manager.complete_previous_state_batch()
            if self.state_audit is not None:
                self.state_audit.finish()
            outputs = [list(seq.completion_token_ids) for seq in seqs]
            if labels is not None:
                assert (
                    outputs == labels
                ), "Teacher-forced sampler/proposal alignment failed"
                scores = [
                    torch.stack([row[i] for i in range(len(suffix))])
                    for row, suffix in zip(saved, labels)
                ]
            else:
                scores = None
                self.generation_records.extend(
                    {"output_ids": output, "leave_reason": str(seq.leave_reason)}
                    for seq, output in zip(seqs, outputs)
                )
            ranks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ranks, outputs)
            assert all(value == outputs for value in ranks)
            return outputs, scores
        finally:
            runner.model.compute_logits = original_logits
            if original_propose is not None:
                runner.drafter.propose = original_propose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--audit-state", action="store_true")
    parser.add_argument("--generation-progress", action="store_true")
    parser.add_argument("--calibration-profile")
    parser.add_argument("--production", action="store_true")
    parser.add_argument(
        "--serial-verify-ops",
        nargs="+",
        choices=("engram", "attention"),
        default=[],
        help="Diagnostic only: run selected verification operations one row at a time",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use the published V4.1 chat encoder through lm_eval",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "arc_easy",
            "hellaswag",
            "agieval_logiqa_zh",
            "bigbench_code_line_description_multiple_choice",
        ],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    # The arm FULL replaced, kept so the two are one flag apart on one harness.
    # PIECEWISE at `level=0` has no compiled pieces to replay, so this arm runs
    # the target eagerly -- which is exactly the control an accuracy comparison
    # wants, and what it is named after is where its graphs would go.
    parser.add_argument("--piecewise", action="store_true")
    args = parser.parse_args()
    if args.audit_state and (args.serial_verify_ops or args.tasks != ["gsm8k"]):
        parser.error("State audit requires unmodified greedy GSM8K generation")
    intervention_stats = None
    if args.serial_verify_ops:
        if args.production:
            parser.error("Production validation cannot patch model arithmetic")
        from .dspark_projection_probe import serial_verify_operations

        intervention_stats = serial_verify_operations(args.serial_verify_ops)
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])

    def check(config):
        checked = copy.copy(config)
        checked.dspark = copy.copy(config.dspark)
        checked.dspark.confidence_schedule = False
        diagnostic_config(checked)

    profile_root = os.environ.get("ATOM_V41_PROFILE")
    guard = (
        nullcontext()
        if args.production
        else patch(
            "atom.models.deepseek_v41.config.validate_runtime_config",
            validate_runtime_config if args.baseline else check,
        )
    )
    with guard:
        config = Config(
            model=args.model,
            tensor_parallel_size=size,
            enable_expert_parallel=True,
            enforce_eager=False,
            compilation_config=CompilationConfig(
                cudagraph_mode=(
                    CUDAGraphMode.PIECEWISE if args.piecewise else CUDAGraphMode.FULL
                ),
                cudagraph_capture_sizes=[1, 2, 4],
            ),
            speculative_config=(
                None
                if args.baseline
                else SpeculativeConfig(
                    method="dspark",
                    model=args.model,
                    # DIAGNOSTIC175: draft width is the dial that separates
                    # "stale rows left behind" from "the forward is multi-row".
                    # Production is 5. Revert: grep -rn DIAGNOSTIC175 tests/
                    num_speculative_tokens=int(
                        os.environ.get("ATOM_DSPARK_WIDTH") or 5
                    ),
                )
            ),
            dspark=DSparkConfig(
                confidence_schedule=bool(args.calibration_profile),
                ragged=bool(args.calibration_profile),
                calibration_profile=args.calibration_profile,
            ),
            torch_profiler_dir=profile_root,
            kv_cache_dtype="bf16",
            index_cache_dtype="fp8",
            max_num_batched_tokens=512,
            max_model_len=4096,
            max_num_seqs=4,
            long_prefill_token_threshold=128,
            # A multiple of every PAGE size CSA2 runs: the scheduler snaps an
            # interval that is not one to off, which would leave two arms of a
            # comparison with different checkpointing and nothing saying so.
            state_checkpoint_interval_tokens=256,
            enable_log_stats=False,
            port=port,
        )
    config.parallel_config.data_parallel_base_port = port
    runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(2048)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(2048)
        runner.capture_cudagraph()
        directory = args.output.parent / "logits"
        directory.mkdir(exist_ok=True)
        state_audit = None
        if args.audit_state:
            from .dspark_state_audit import RuntimeStateAudit

            state_audit = RuntimeStateAudit(runner)
        model = RuntimeLM(
            runner,
            AutoTokenizer.from_pretrained(args.model, local_files_only=True),
            directory,
            chat=args.chat,
            state_audit=state_audit,
            generation_progress=args.generation_progress,
        )
        # PERF185: ModelRunner already owns profiling -- per-rank directory,
        # `ATOM_PROFILER_MORE` for shapes/stack/memory, and a gzipped chrome
        # trace rather than an aggregate table. Keep --limit small when this is
        # on; the trace grows with generated tokens.
        if profile_root:
            runner.start_profiler("v41_generate")
        with torch.inference_mode():
            result = simple_evaluate(
                model=model,
                tasks=args.tasks,
                num_fewshot=args.fewshot,
                limit=args.limit,
                batch_size=4,
                bootstrap_iters=1000,
                log_samples=True,
                apply_chat_template=args.chat,
                gen_kwargs={"do_sample": False, "max_gen_toks": args.max_output_tokens},
            )
        if profile_root:
            print("profiler traces:", runner.stop_profiler())
        if intervention_stats is not None:
            for name in args.serial_verify_ops:
                if intervention_stats[name]["max_rows"] <= 1:
                    raise RuntimeError(
                        f"{name} intervention did not execute on multiple rows"
                    )
            if (
                "engram" in args.serial_verify_ops
                and intervention_stats["engram_graph_stage_bypasses"] == 0
            ):
                raise RuntimeError(
                    "Engram intervention never bypassed a captured stage"
                )
            ranks = [None] * size
            torch.distributed.all_gather_object(ranks, intervention_stats)
            result["arithmetic_intervention_stats"] = ranks
        if state_audit is not None:
            ranks = [None] * size
            torch.distributed.all_gather_object(ranks, state_audit.records)
            result["runtime_state_audit"] = ranks
            state_audit.close()
        result["reference_manifest"] = MANIFEST
        result["baseline"] = args.baseline
        result["production_config"] = args.production
        result["serial_verify_operations_diagnostic"] = args.serial_verify_ops
        result["chat"] = args.chat
        result["eos_token_id"] = config.eos_token_id
        result["runtime_generation"] = model.generation_records
        result["calibration_profile"] = args.calibration_profile
        result["completed"] = True
        if rank == 0:
            args.output.write_text(
                json.dumps(result, indent=2, default=str, allow_nan=False) + "\n"
            )
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
