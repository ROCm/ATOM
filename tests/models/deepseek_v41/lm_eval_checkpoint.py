# SPDX-License-Identifier: MIT
"""lm-evaluation-harness adapter for independent P04 target/reference runs.

Run both implementations with the same harness arguments for paired scoring.
Each process loads one model, bounding HBM without changing either graph.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

from atom.examples.deepseek_v41_offline import (
    initialize_parallel,
    load_offline_model,
    prepare_engram,
)

from .checkpoint_cache import evaluation_identity, response_cache
from .checkpoint_reference import reference_resources
from .reference import MANIFEST


class CheckpointLM(LM):
    def __init__(
        self, model, tokenizer, mapping, host, *, implementation, max_length, rank
    ):
        super().__init__()
        self.model, self.tokenizer, self.mapping, self.host = (
            model,
            tokenizer,
            mapping,
            host,
        )
        self.implementation, self.max_length, self.tp_rank = (
            implementation,
            max_length,
            rank,
        )

    def _new_request(self):
        cache = self.model.new_cache(1) if self.implementation == "target" else None
        return cache, np.full((1, 3), -1, dtype=np.int64)

    def _forward(self, tokens, position, cache, history, *, full=False, logits_start=0):
        rows, history = prepare_engram(
            tokens, position, history, self.mapping, self.host
        )
        ids = torch.tensor([tokens], device="cuda")
        if self.implementation == "target":
            logits = self.model(
                ids, cache, rows, full_logits=full, logits_start=logits_start
            )
        else:
            logits = self.model(
                ids, position, rows, full_logits=full, logits_start=logits_start
            )
        return logits, history

    def loglikelihood(self, requests):
        results = []
        for index, request in enumerate(requests):
            context, continuation = request.args
            spaces = len(context) - len(context.rstrip())
            if spaces:
                context, continuation = (
                    context[:-spaces],
                    context[-spaces:] + continuation,
                )
            if context:
                context_ids = self.tokenizer.encode(context, add_special_tokens=True)
                tokens = self.tokenizer.encode(
                    context + continuation, add_special_tokens=True
                )
            else:
                context_ids = [self.tokenizer.bos_token_id]
                tokens = context_ids + self.tokenizer.encode(
                    continuation, add_special_tokens=False
                )
            labels = tokens[len(context_ids) :]
            if not labels or len(tokens) > self.max_length:
                raise ValueError(
                    "Evaluation request has an empty continuation or exceeds capacity"
                )
            cache, history = self._new_request()
            logits, _ = self._forward(
                tokens[:-1],
                0,
                cache,
                history,
                full=True,
                logits_start=len(context_ids) - 1,
            )
            logits = logits.float()
            expected = torch.tensor([labels], device="cuda")
            score = (
                logits.log_softmax(-1).gather(-1, expected.unsqueeze(-1)).sum().item()
            )
            if not math.isfinite(score):
                raise FloatingPointError(
                    f"Non-finite likelihood for request {index}: {request.args!r}"
                )
            results.append((score, bool((logits.argmax(-1) == expected).all())))
            self.cache_hook.add_partial("loglikelihood", request.args, results[-1])
            if self.tp_rank == 0 and (index + 1) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "event": "scored_requests",
                            "count": index + 1,
                            "total": len(requests),
                        }
                    ),
                    flush=True,
                )
        return results

    def generate_until(self, requests):
        results = []
        for index, request in enumerate(requests):
            prompt, options = request.args
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            limit = options.get("max_gen_toks", 256)
            if len(tokens) + limit > self.max_length:
                raise ValueError("Evaluation generation exceeds capacity")
            stops = options.get("until", [])
            stops = [stops] if isinstance(stops, str) else stops
            cache, history = self._new_request()
            generated, position, current = [], 0, tokens
            for _ in range(limit):
                logits, history = self._forward(current, position, cache, history)
                position += len(current)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite generation logits for request {index}, position {position}"
                    )
                token = int(logits.argmax(-1).item())
                if token == self.tokenizer.eos_token_id:
                    break
                generated.append(token)
                text = self.tokenizer.decode(generated)
                if any(stop and stop in text for stop in stops):
                    break
                current = [token]
            text = self.tokenizer.decode(generated)
            for stop in stops:
                if stop:
                    text = text.split(stop)[0]
            results.append(text)
            self.cache_hook.add_partial("generate_until", request.args, text)
            if self.tp_rank == 0:
                print(
                    json.dumps({"event": "generation", "request": index, "text": text}),
                    flush=True,
                )
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Use a fixed continuation for this evaluator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation", choices=("target", "reference"), required=True
    )
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "hellaswag"])
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument(
        "--response-cache",
        type=Path,
        help="Resume finite harness responses from this build",
    )
    args = parser.parse_args()
    rank = initialize_parallel()
    torch.set_num_threads(4)
    loader = (
        load_offline_model if args.implementation == "target" else reference_resources
    )
    try:
        with loader(args.model, 4096) as resources:
            model = CheckpointLM(
                *resources,
                implementation=args.implementation,
                max_length=4096,
                rank=rank,
            )
            identity = (
                evaluation_identity(args.model, args.implementation, 4096)
                if args.response_cache is not None
                else None
            )
            with (
                response_cache(model, args.response_cache, identity) as cache_prefix,
                torch.inference_mode(),
            ):
                result = simple_evaluate(
                    model=model,
                    use_cache=cache_prefix,
                    tasks=args.tasks,
                    num_fewshot=args.fewshot,
                    limit=args.limit,
                    batch_size=1,
                    bootstrap_iters=1000,
                    log_samples=True,
                    gen_kwargs={"do_sample": False, "max_gen_toks": 256},
                )
            if rank == 0:
                result["reference_manifest"] = MANIFEST
                result["implementation"] = args.implementation
                result["tensor_parallel_size"] = get_tp_group().world_size
                result["custom_all_reduce"] = False
                args.output.write_text(
                    json.dumps(result, indent=2, default=str, allow_nan=False)
                )
                print(
                    json.dumps(
                        {
                            "event": "lm_eval_complete",
                            "implementation": args.implementation,
                            "results": result["results"],
                        }
                    ),
                    flush=True,
                )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
