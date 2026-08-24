# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stress prefix-cache MLA prefill to reproduce the OOM that motivated
ATOM_MLA_PREFILL_KV_THRESHOLD. Sends N concurrent requests sharing one
~12K-token cached prefix; each prefill chunk reads back the full prefix
times the number of in-batch requests, blowing past max_num_batched_tokens
sized intermediates if we go through the gather_kv_b_proj + materialized
[total_kv, H, 192] path.

Compare with ATOM_MLA_PREFILL_KV_THRESHOLD set:
  - very high (1<<31)  → forces old path, expect OOM
  - default (32768)    → new dequant + mla_prefill_fwd path, expect OK
"""

import argparse

from atom import SamplingParams
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Prefix-cache OOM stress probe",
)
EngineArgs.add_cli_args(parser)
parser.add_argument("--max-tokens", type=int, default=16)
parser.add_argument("--n-requests", type=int, default=32)
parser.add_argument("--prefix-repeat", type=int, default=70)


LONG_CONTEXT_BLOCK = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. Mitochondria generate most of the cell's supply of "
    "adenosine triphosphate (ATP), used as a source of chemical energy. "
    "Mitochondria have their own genetic material and the machinery to "
    "manufacture their own RNAs and proteins. The number of mitochondria "
    "per cell varies widely; for example, in humans, erythrocytes (red "
    "blood cells) do not contain any mitochondria, whereas liver cells and "
    "muscle cells may contain hundreds or even thousands. The organelle is "
    "composed of compartments that carry out specialized functions. These "
    "compartments or regions include the outer membrane, the intermembrane "
    "space, the inner membrane, and the cristae and matrix.\n\n"
)


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    custom_encoder = load_custom_message_encoder(args.model)

    shared_prefix = LONG_CONTEXT_BLOCK * args.prefix_repeat
    pre_tokens = len(tokenizer(shared_prefix).input_ids)
    print(f"\nShared prefix tokens: ~{pre_tokens}")
    print(f"N requests: {args.n_requests}")
    print(f"Approx total cached KV per prefill chunk: {pre_tokens * args.n_requests}")

    seed_prompt = apply_chat_template(
        tokenizer,
        custom_encoder,
        [{"role": "user", "content": shared_prefix + "\n\nQ: summarize."}],
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print("\n=== ROUND 1: seed (no prefix hit) ===")
    seed_out = llm.generate([seed_prompt], sp)
    print(f"  seed: {seed_out[0]['text'][:80]!r}")

    prompts = [
        apply_chat_template(
            tokenizer,
            custom_encoder,
            [
                {
                    "role": "user",
                    "content": shared_prefix + f"\n\nQ #{i}: name one fact.",
                }
            ],
        )
        for i in range(args.n_requests)
    ]

    print(f"\n=== ROUND 2: {args.n_requests} concurrent (prefix HIT) ===")
    outs = llm.generate(prompts, sp)
    ok = sum(1 for o in outs if o["text"].strip())
    print(f"  completed: {ok}/{len(outs)}")
    print(f"  sample[0]: {outs[0]['text'][:80]!r}")
    print(f"  sample[-1]: {outs[-1]['text'][:80]!r}")


if __name__ == "__main__":
    main()
