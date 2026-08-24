# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Prefix-cache correctness probe.

Round 1 seeds the cache with a long shared prefix (one request).
Round 2 sends N requests that all share that same prefix, each with a
short unique tail. Round 2 prefill therefore consumes the cache and
takes the prefix-cache code path under test (gfx950+fp8 MLA →
mla_prefill_ps_fwd). With temperature=0 the answers should be
deterministic and coherent — garbled or random output indicates the
prefix-cache attention path is producing wrong KV reads.
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
    description="Prefix-cache correctness probe",
)
EngineArgs.add_cli_args(parser)
parser.add_argument("--max-tokens", type=int, default=64)


# A long, factual passage. Repeated to push the shared prefix past the
# 4K-token mark so it occupies many blocks and exercises the paged path.
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
    "space, the inner membrane, and the cristae and matrix. Although most "
    "of a cell's DNA is contained in the cell nucleus, the mitochondrion "
    "has its own genome (mtDNA) that is substantially similar to bacterial "
    "genomes. This finding has led to general acceptance of the endosymbiotic "
    "hypothesis — that free-living prokaryotes were taken into another cell "
    "as endosymbionts and evolved into the mitochondria of today.\n\n"
)


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    custom_encoder = load_custom_message_encoder(args.model)

    # ~4.5K tokens of shared context. Adjust repetition if your model's
    # tokenizer compresses this more.
    shared_prefix = LONG_CONTEXT_BLOCK * 25

    questions = [
        "In one sentence, what is the endosymbiotic hypothesis?",
        "Which cells in the human body lack mitochondria?",
        "Name the four mitochondrial compartments mentioned above.",
        "What molecule do mitochondria primarily generate?",
    ]
    prompts_text = [shared_prefix + "\n\nQuestion: " + q for q in questions]
    prompts = [
        apply_chat_template(tokenizer, custom_encoder, [{"role": "user", "content": p}])
        for p in prompts_text
    ]

    seed_text = shared_prefix + "\n\nQuestion: Briefly summarize the passage."
    seed_prompt = apply_chat_template(
        tokenizer, custom_encoder, [{"role": "user", "content": seed_text}]
    )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    print("\n" + "=" * 70)
    print("ROUND 1: seed prefix cache (1 request, NO prefix-cache hit)")
    print("=" * 70)
    seed_out = llm.generate([seed_prompt], sp)
    print(f"  seed completion (first 120 chars): {seed_out[0]['text'][:120]!r}")

    print("\n" + "=" * 70)
    print("ROUND 2: 4 requests sharing the cached prefix (prefix-cache HIT)")
    print("=" * 70)
    outputs = llm.generate(prompts, sp)
    for q, out in zip(questions, outputs):
        print(f"\nQ: {q}")
        print(f"A: {out['text']!r}")

    llm.print_mtp_statistics()


if __name__ == "__main__":
    main()
