# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 prefix-cache HIT collapse/divergence probe (targets fix B').

WHY THIS EXISTS
---------------
The B' fix (scheduler tail re-forward, `_v4_swa_warmup_blocks`) repopulates the
per-request SWA *ring* by re-forwarding the last `ceil(win_with_spec/128)` hit
blocks. Open question this probe answers empirically:

  On current main (B' ON, per-req SWA ring, NO paged SWA #1417), does a
  cross-request prefix-cache HIT still corrupt output?

Two hypothesized residual gaps B' does NOT obviously cover:
  1. SWA window: for a SHORT fresh suffix (re-send an identical long prompt),
     the re-forward buffer is minimal, so a tail token's SWA window may still
     reach a not-re-forwarded region.
  2. CSA compressor ring: the re-forward starts at a 128-aligned boundary, but
     CSA (ratio=4, overlapping, K_pool=8) needs the previous ~32 raw tokens in
     its fp32 ring to emit the first few post-boundary committed entries. Those
     tokens are in the cached (not re-forwarded) region -> cold ring -> wrong
     boundary CSA entries. Low-frequency; only a 100%-hit token probe shows it.

DESIGN
------
Mirror the field bug report: a LONG prompt (~num_blocks*128 tokens), re-sent
IDENTICALLY (100% prefix-cache hit), greedy.

  SEED : first time this exact prompt is seen -> cold full forward (reference).
  HIT  : re-send SAME prompt -> cross-request prefix-cache hit -> exercises B'.

Signals, most-trustworthy first:
  * COLLAPSE (primary): HIT completion is empty / immediate-EOS / <=2 tokens
    while SEED was full. This is the field-report symptom and is IMMUNE to
    ATOM's greedy non-determinism (a cold recompute is never empty), so a
    single COLLAPSE is an unambiguous B' failure.
  * DIVERGE@k (secondary): first differing completion-token index between SEED
    and HIT. Because ATOM greedy is NOT bit-reproducible (see memory
    `project_atom_nondeterminism.md`), a late/unstable divergence is likely
    NOISE, not a bug. Establish the noise floor with --no-enable_prefix_caching
    (there both runs are full recomputes; any DIVERGE there is pure noise).
    A REAL B' bug shows as EARLY (k near 0) AND STABLE divergence across repeats
    that is absent from the caching-OFF noise floor.

USAGE
-----
  # prefix caching ON (exercises B' on the HIT):
  python -m atom.examples.probe_prefix_hit_collapse \
      --model <DeepSeek-V4-Flash-FP8> -tp 4 --kv_cache_dtype fp8 \
      --trust-remote-code --enforce-eager --enable_prefix_caching \
      --max-model-len 12288 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.6 --num-blocks 60

  # noise floor (same command, caching OFF) — compare DIVERGE@k distributions:
  python -m atom.examples.probe_prefix_hit_collapse ... --no-enable_prefix_caching

VERDICT
-------
  * Any COLLAPSE with caching ON            -> B' HAS A BUG (report-class).
  * DIVERGE@k with ON << DIVERGE@k with OFF -> B' has a subtle (CSA/SWA) bug.
  * ON matches OFF noise floor, no collapse -> B' is correct for this workload.
"""

import argparse

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="DeepSeek-V4 prefix-cache HIT collapse/divergence probe (B')",
)
EngineArgs.add_cli_args(parser)
parser.add_argument("--max-tokens", type=int, default=32)
parser.add_argument(
    "--num-blocks",
    type=int,
    default=60,
    help="Cached-prefix length in 128-token blocks (60 ~= 7680 tokens, "
    "matching the field report's ~7.7k prompt).",
)
parser.add_argument(
    "--remainders",
    type=str,
    default="21,4,64,127",
    help="Comma-separated tail token counts beyond num_blocks*128. Small values "
    "(e.g. 21, like the report) give B' the SMALLEST re-forward buffer -> worst "
    "case. The HIT boundary itself is always 128-aligned regardless.",
)
parser.add_argument(
    "--repeats",
    type=int,
    default=3,
    help="HIT resends per length, to separate a STABLE bug from random "
    "non-determinism noise.",
)
parser.add_argument(
    "--corpus",
    choices=["filler", "high", "issue"],
    default="filler",
    help="Prompt content:\n"
    "  filler = repeated mitochondria sentence (LOW entropy; robust collapse "
    "test, weak on subtle divergence).\n"
    "  high   = real diverse source text from the repo (HIGH entropy; a subtle "
    "CSA/SWA numeric error is far more likely to flip a token here).\n"
    "  issue  = the EXACT field-report prompt "
    "('the archival index records that fact again ' * 1100), max_tokens forced "
    "to 24, remainder sweep ignored.",
)

FILLER = (
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate, used as a source of chemical energy. Mitochondria have "
    "their own genetic material and machinery to manufacture RNAs and "
    "proteins. The number per cell varies widely across tissues and species. "
)

BLOCK_SIZE = 128

# Exact prompt from the field report (issue). Highly repetitive -> LOW entropy,
# so it's a collapse test, not a subtle-divergence test.
ISSUE_UNIT = "the archival index records that fact again "
ISSUE_REPEAT = 1100


def _high_entropy_text() -> str:
    """Concatenate real, diverse source text from the repo -> HIGH entropy.

    A subtle numeric error at the prefix-hit boundary (e.g. a cold CSA
    compressor ring) is far more likely to flip a greedy token on genuine,
    high-perplexity text than on a repeated filler sentence. Deterministic
    (sorted file walk), so SEED and HIT see identical input."""
    import glob
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # atom/
    parts: list[str] = []
    total = 0
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            with open(path, "r", errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        if len(txt) < 400:
            continue
        parts.append(txt)
        total += len(txt)
        if total > 2_000_000:  # plenty for ~8k tokens with unique markers
            break
    return "\n".join(parts)


def _tok_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _first_divergence(a: list[int], b: list[int]) -> int:
    """Index of first differing element; len(shorter) if one is a prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1, 2, 4, 8]"
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    caching_on = bool(getattr(engine_args, "enable_prefix_caching", False))

    # ---- corpus selection ----
    if args.corpus == "issue":
        # Exact field-report repro: one fixed prompt, resent; max_tokens=24.
        remainders = [0]
        max_tokens = 24
        source_text = None  # built inline below
    else:
        remainders = [int(x) for x in args.remainders.split(",") if x.strip() != ""]
        max_tokens = args.max_tokens
        source_text = _high_entropy_text() if args.corpus == "high" else (FILLER * 256)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    print("\n" + "=" * 78)
    print(
        f"V4 PREFIX-HIT COLLAPSE PROBE  (block_size={BLOCK_SIZE}, "
        f"num_blocks={args.num_blocks}, corpus={args.corpus}, "
        f"max_tokens={max_tokens}, prefix_caching={'ON' if caching_on else 'OFF'})"
    )
    print(
        "  ON  -> HIT exercises B'; a COLLAPSE = report-class bug.\n"
        "  OFF -> both runs are full recomputes = non-determinism NOISE FLOOR."
    )
    print("=" * 78)

    any_collapse = False
    worst_diverge = {}  # r -> min first-divergence across repeats

    for idx, r in enumerate(remainders):
        if args.corpus == "issue":
            # EXACT field-report prompt, verbatim. No unique marker, no trim —
            # reproduce the case as filed.
            prompt_text = ISSUE_UNIT * ISSUE_REPEAT
            L = len(_tok_ids(tokenizer, prompt_text))
        else:
            L = BLOCK_SIZE * args.num_blocks + r
            # Unique leading marker per length so its block hashes never collide
            # with another length's prompt (otherwise a "SEED" could hit a prior
            # length's cache and stop being a cold reference). Trimmed to exactly L.
            marker = f"COLLAPSEPROBE{idx}Z{idx}Z{idx}Z{idx}. "
            base_ids = _tok_ids(tokenizer, marker + source_text)
            assert len(base_ids) >= L, f"corpus too short: {len(base_ids)} < {L}"
            prompt_ids = base_ids[:L]
            prompt_text = tokenizer.decode(prompt_ids)

        # SEED: cold reference (first sighting -> no hit).
        seed_txt = llm.generate([prompt_text], sp)[0]["text"]
        seed_ids = _tok_ids(tokenizer, seed_txt)
        seed_collapsed = len(seed_ids) <= 2

        print(
            f"\n[r={r:>3} L={L}]  hit-boundary=128-aligned, "
            f"SWA-window-into-cached={BLOCK_SIZE - (r % BLOCK_SIZE)}"
        )
        print(
            f"  SEED (cold ref): ctoks={len(seed_ids):>3} "
            f"{'COLLAPSED!' if seed_collapsed else 'full'}  {seed_txt[:56]!r}"
        )

        min_div = max_tokens
        for j in range(args.repeats):
            hit_txt = llm.generate([prompt_text], sp)[0]["text"]
            hit_ids = _tok_ids(tokenizer, hit_txt)
            collapsed = len(hit_ids) <= 2 and not seed_collapsed
            div = _first_divergence(seed_ids, hit_ids)
            min_div = min(min_div, div)
            if collapsed:
                any_collapse = True
            tag = (
                "COLLAPSE <-- BUG"
                if collapsed
                else ("exact-match" if div >= min(len(seed_ids), len(hit_ids)) else "")
            )
            print(
                f"  HIT#{j+1:<2}       : ctoks={len(hit_ids):>3} "
                f"diverge@{div:<3} {tag}  {hit_txt[:56]!r}"
            )
        worst_diverge[r] = min_div

    print("\n" + "=" * 78)
    if caching_on:
        print("RESULT (prefix caching ON):")
        if any_collapse:
            print("  >>> COLLAPSE observed -> B' HAS A BUG (report-class corruption).")
        else:
            print("  >>> No collapse. Earliest seed-vs-hit divergence per length:")
            for r in remainders:
                print(f"        r={r:>3}: diverge@{worst_diverge[r]}")
            print(
                "      Compare these against a --no-enable_prefix_caching run:\n"
                "        * ON diverges MUCH earlier than OFF -> subtle B' bug\n"
                "          (likely CSA compressor-ring boundary).\n"
                "        * ON ~= OFF -> B' correct for this workload (divergence\n"
                "          is just greedy non-determinism)."
            )
    else:
        print("RESULT (prefix caching OFF = NOISE FLOOR):")
        print("  Earliest cold-vs-cold divergence per length (pure noise):")
        for r in remainders:
            print(f"        r={r:>3}: diverge@{worst_diverge[r]}")
        print("  Re-run WITHOUT --no-enable_prefix_caching and compare.")
    print("=" * 78)


if __name__ == "__main__":
    main()
