# DeepSeek-V4-Flash-Vision-Exp

The first multimodal model in the DeepSeek-V4 family: the V4-Flash language
stack (43 layers, DFlash sparse attention, mHC, FP8/FP4 MoE) plus a 32-layer ViT
and an aligner. ATOM serves it through the same `deepseek_v4.py` stack as the
text-only V4 checkpoints.

## Running

```bash
# Offline, single card — the weights are 156 GiB and a 288 GiB card fits them
# whole, so tp=1 is the simplest way to run it.
AITER_LOG_LEVEL=WARNING python -m atom.examples.multimodal_inference \
  --model /data/DeepSeek-V4-Flash-Vision-Exp \
  --image path/to/image.jpg \
  --prompt "What is in this image?" -tp 1

# OpenAI-compatible server
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model /data/DeepSeek-V4-Flash-Vision-Exp -tp 1
```

Images are passed as normal OpenAI `image_url` content parts.

## What is model-specific

**One architecture, optional vision tower.** Both checkpoints declare
`DeepseekV4ForCausalLM` because that is what they are: `DeepseekV4ForCausalLM`
builds the ViT + aligner iff `vision_n_layers > 0`, mirroring the reference's
own `Transformer.__init__`. No arch-string rewriting or config-based class
dispatch is involved. The registered multimodal input builder is keyed on the
same string and returns `None` for a text-only config so the caller falls back
to its text path.

**No HF processor.** The checkpoint ships no `preprocessor_config.json` and no
chat template. Image preprocessing is ported into
`atom/models/deepseek_v4_vl.py`, and the prompt template is loaded at
runtime from the checkpoint's own `encoding/encoding_dsv4.py` — versioned with
the weights, so it cannot drift the way a vendored copy would. That executes
code from the model directory, the same trust boundary
`AutoProcessor(trust_remote_code=True)` already crosses for Kimi-K3.

**Image tokens live above the vocabulary.** Each `<｜deepseek_image｜>` expands
into a run of `vocab_size + type` ids (129280..129284). Anything indexing a
vocab-sized table by `input_ids` must clamp; `embed_input_ids` does, and the
routing kernel treats `>= vocab_size` as the image predicate.

**Two router biases.** Every layer carries `gate.bias_vl` alongside `gate.bias`,
selected per token. `FusedMoE.select_experts` takes a single `[E]` bias, so
vision checkpoints route through `atom/model_ops/triton_mm_topk.py` on every
layer via the `custom_routing_function` hook.

**Bidirectional attention inside an image.** Tokens within an
`[IMAGE_START, IMAGE_END]` span attend to each other in both directions. ATOM's
prefill attention is purely index-driven with no causal mask, so this is
expressed by widening each token's index list
(`image_aware_extend_window`) rather than by changing the attention kernel. The
compressed (CSA/HCA) paths stay causal, matching the reference.

**Multimodal prefills are never chunked.** The vision embeddings are produced
for the whole prompt and scattered onto its placeholder positions, and in-image
attention reads across the whole block. The scheduler takes such a prompt whole
or waits; `_finalize_prefill_chunk` also skips the state-checkpoint rung cut
for a sequence carrying `multimodal_data` (#2110), which means **no state
checkpoint is kept for a multimodal request** — the cost is prefix-cache reuse
on image prompts, which are single-shot anyway.
`max_num_batched_tokens` therefore caps multimodal prompt length even with
chunked prefill enabled.

## Compilation

`--level` (default 3) drives the **language model**: `DeepseekV4Model` carries
`@support_torch_compile`, so it gets piecewise compilation plus CUDAGraph
capture on decode. Verified at level 3 — image answers are token-identical to
level 0, and GSM8K matches within noise.

The vision tower runs eager, like every other ATOM vision tower. It sits outside
`@support_torch_compile`, so `--level` does not reach it; wrapping it in
`torch.compile(dynamic=True)` measured 1.31x (42x61 grid) to 1.57x (30x40) on
the tower alone — ~3-5 ms per image against a prefill of a few hundred ms — which
did not justify a bespoke toggle. Revisit if image throughput ever matters.

CUDAGraph never covers the tower, but that is engine-wide rather than a vision
decision: `forward_context` gates capture on `not is_prefill`, and the tower runs
only during prefill.

## Not supported

- **PCP** (prefill context parallelism) with images — `inputs_embeds` would have
  to be round-robin split alongside `input_ids`. Raises rather than guessing.
- **DSpark / MTP speculative decoding** — the `mtp.*` gates carry `bias_vl`
  too and the draft path has no sentinel handling yet. Raises at startup rather
  than routing images with the text bias; run without `--method`.
- **Chunked multimodal prefill** — see above.
- **Images behind a KV connector (PD disaggregation).** Untested here; the
  resume path is not exercisable on this machine, so the interaction between a
  parked `multimodal_data` request and its resumed prefill is unverified.
- **TBO prefill micro-batching** with images — a token-split ubatch would cut an
  image span in half. Raises; start without `--enable-tbo`.
- **Images at TP > 1 are lightly covered.** The ChartQA and ZeroBench runs below
  were served at `tp=8`; the interactive/offline checks were `tp=1`. The tower is
  replicated per rank like Kimi-K3's, so nothing in the design is TP-specific,
  but no test pins per-rank image-embedding equality.

## Accuracy

GSM8K 3-shot, 1319 samples, `tp=8`, `--kv_cache_dtype fp8` (text-only, so this
covers the `bias_vl` routing kernel on all 43 layers):

| | flexible-extract | strict-match |
|---|---|---|
| `--level 0` | 0.9416 ± 0.0065 | 0.9424 ± 0.0064 |
| `--level 3` (default) | 0.9340 ± 0.0068 | 0.9348 ± 0.0068 |
| `--level 3`, re-measured after the rebase onto `5a9c2068` | 0.9393 ± 0.0066 | 0.9393 ± 0.0066 |

The 0.76pp level-0/3 gap is ~0.8σ, and the post-rebase re-measurement sits
0.53pp above its own earlier run — both inside one noise band, so neither is a
real difference. CI has no baseline for this model; V4-Pro's 0.96 is a larger
model (61 layers, hidden 7168) and not a fair comparison.

Vision, via `/app/scripts/run_vision_bench.sh` (lmms_eval against the OpenAI
server), `tp=8`, `--kv_cache_dtype fp8`, greedy:

| Task | Metric | Value |
|---|---|---|
| chartqa (2500 samples, 0 failed) | relaxed_overall | 0.8888 ± 0.0063 |
| | relaxed_augmented_split | 0.9576 ± 0.0057 |
| | relaxed_human_split | 0.8200 ± 0.0109 |

ZeroBench, via `/app/scripts/run_zerobench.py` (ungated `mm-eval/ZeroBench`
mirror; grading imports lmms_eval's own `_is_exact_match`), thinking mode at
`max` reasoning effort, `temperature=1.0 top_p=0.95`, `max_tokens=16384`,
0 failed requests in 2170 samples:

| Split | Pass@1 | Pass@5 |
|---|---|---|
| `zerobench` (100 main) | 1.0 | 3.0 |
| `zerobench_subquestions` (334) | 21.0 | **39.5** |
| `zerobench_subquestions`, thinking OFF | 14.7 | — |

**Card comparison, with caveats.** The card reports "ZeroBench (Pass@5) 35.0"
without naming a split. The main split cannot be it — 3.0 here, and published
frontier results sit at 0-7% Pass@1 by design — so it must mean the
subquestions, where we get 39.5 against the card's 35.0. Treat that as
"same magnitude, consistent", NOT a reproduction. DeepSeek Harness itself IS
open source (github.com/deepseek-ai/deepseek-harness, MIT, branch `master`), but
it ships no evaluation tooling: `BENCHMARK.md` is a three-line pointer to the
Python SDK guide, and a full-tree scan of its 10,294 files contains no task
definitions, prompts, graders or scores for any of the card's benchmarks. So the
prompt template and grader cannot be matched, and scoring *above* the card is as
likely to mean a more lenient exact-match as a better serve. The card's note 1
also scopes the harness to the TEXT rows only — it states no methodology at all
for the four multimodal ones.

ApexBench and Agents' Last Exam are **not** reproducible: no public task set,
prompts or grader exists for either. The public APEX-Agents (Mercor, arXiv
2601.14242) is a different benchmark — a score from it must not be reported as
ApexBench. Chartography has a public dataset (`surgeai/chartography`) but it
could not be confirmed as the card's benchmark. ChartQA above is a separate,
public benchmark — do not compare it to the card's Chartography row.

## Tests

```bash
python -m pytest tests/test_deepseek_v4_vl_cpu.py          # CPU, vs the reference impl
python -m pytest tests/test_prefill_indices_paged.py   # needs a GPU
```

`tests/test_deepseek_v4_vl_cpu.py` compares against the reference implementation
shipped inside the checkpoint (`inference/vision.py`,
`inference/image_processor.py`, `encoding/encoding_dsv4.py`) rather than against
hand-written expectations, and skips when the checkpoint is not present.
