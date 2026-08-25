# SPDX-License-Identifier: MIT
"""Minimal single-prompt greedy driver for MoE apply() dump comparison.

Runs ONE short prompt at temperature=0 (deterministic) for a few tokens, so the
MoE apply() boundary dump (ATOM_MOE_APPLY_DUMP_DIR) captures identical x /
router_logits across a default-path run and a triton-path run. Exits after.
"""

import argparse

from atom import SamplingParams
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
EngineArgs.add_cli_args(parser)
parser.add_argument("--prompt", type=str, default="1+2+3=?")
parser.add_argument("--max-tokens", type=int, default=4)
parser.add_argument("--num-requests", type=int, default=1)
parser.add_argument("--raw", action="store_true", help="skip chat template")


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.raw:
        prompt = args.prompt  # no chat template, matches /v1/completions
    else:
        custom_encoder = load_custom_message_encoder(args.model)
        prompt = apply_chat_template(
            tokenizer, custom_encoder, [{"role": "user", "content": args.prompt}]
        )
    # Greedy / deterministic so the two runs are comparable.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    # Send the SAME request twice sequentially to catch 2nd-request corruption.
    for req in range(args.num_requests):
        print(f"===== REQUEST {req} =====", flush=True)
        outputs = llm.generate([prompt], sampling_params)
        for output in outputs:
            print(f"Completion[req{req}]: {output['text']!r}", flush=True)


if __name__ == "__main__":
    main()
