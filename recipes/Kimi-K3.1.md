# Kimi-K3.1 original checkpoint

Native ATOM path for the original Kimi-K3.1 checkpoint
(`moonshotai/Kimi-K3.1-Pro`) on gfx950 / MI355X. The launch below scores
97.65% on GSM8K (5-shot, full set, mono on; see Validated scope).

## Precision

- Routed expert matrices keep the checkpoint's MXFP4. Attention, the dense MLP
  and the latent MoE down/up matrices keep their BF16 weights; KDA recurrence
  controls, convolutions and output norms keep their FP32 storage.
- MLA KV is BF16 (`--kv_cache_dtype bf16`). MLA K/V projections and the absorbed
  BMM weights stay BF16.
- The DSA index cache must be FP8: pass `--index-cache-dtype fp8`. Left unset it
  follows the KV cache dtype (BF16), which the NoPE index writer does not
  support; `fp4` is rejected while the model is built. FP8 applies to the
  indexer's Q/K activations for MQA scoring, not to weights.

## Model wiring

- `flatten_dsa_config` (`atom/config.py`) flattens the nested `dsa_config`
  before attention pools are chosen; only the `dsa_layers` own index planes.
- All 24 MLA layers carry a NoPE indexer: 32 heads, head dimension 128, causal
  top-k 2048, no index sharing. Queries come from the normalized q-lora latent.
- `FusedIndexerProjection` concatenates q_a, kv_a, index K and index head
  weights in one BF16 projection; all four checkpoint regions must load, and a
  quantization scheme (checkpoint or online) that would quantize it is
  rejected while the model is built.
- Loading rejects incomplete expert regions (unless
  `ATOM_LOADER_STRICT_COVERAGE=false`) and warns about parameters nothing
  loaded, checkpoint tensors nothing took, and float tensors converted to
  another dtype. Check the load log: a missed parameter keeps its init value
  and fails silently.

## AITER requirements

Validated with AITER `main` at `dbaffe23ed` plus
[ROCm/aiter#7067](https://github.com/ROCm/aiter/pull/7067). K3.1 relies on:

- Custom all-reduce summing ranks in a fixed order on every reduce path
  (#6954), so a token's sum does not depend on its row offset or the receiving
  rank.
- The D192/V128 OPUS attention scanning KV forward on both passes, so batch
  composition does not change BF16 attention order.
- The gfx950 BF16 MLA prefill code objects waiting for the first KV/index
  prefetch before reading it.
- From #7067: the register top-k path taking the FP32 `e_score_correction_bias`
  with BF16 router logits, and the MXFP4 MoE stage2 honoring the tuned reduce
  mode (bit-reproducible routed-expert output). Use AITER's normal merged
  tuning tables; pointing `AITER_CONFIG_FMOE` at a single CSV drops required
  shared entries.

After updating AITER, rebuild its JIT modules before starting the server: an
existing `aiter/jit/module_*.so` is loaded without being compared to its
sources. Clear `aiter/jit/` (modules and `build/`) and let the first start
rebuild them, or rebuild them in one process. Do not set `AITER_REBUILD=1` on
the TP server: every rank would rebuild every module it uses.

## Launch

TP4 cannot hold the resident checkpoint; use TP8.

```bash
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model /data/Kimi-K3.1-Pro --trust-remote-code -tp 8 \
  --decode-context-parallel-size 8 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --max-model-len 131072 --max-num-seqs 64 --block-size 128
```

- Mono decode is on by default (`ATOM_MONO_ENABLE=1`; set it to 0 to run
  the original decode path). It serves decode steps of up to 64 rows, a plain
  decode or a speculative verify (16-row token groups past 16): the 69 KDA
  layers run K1 (the attention seam through the KDA recurrence) and K2
  (`o_proj` plus the latent MoE), the 24 MLA layers keep their attention and
  run K2, and their DSA indexer runs as one mono launch: under DCP equal to TP
  over the ranks' own keys, without DCP over the replicated cache (a request's
  rows owned by one rank, a long one's keys split; not yet validated on
  K3.1). A wider step runs the
  original path. It works with ReplaySSM (`ATOM_ENABLE_REPLAYSSM=1`) and with
  a DSpark drafter (`--method dspark --draft-model /data/Kimi-K3-DSpark
  --num-speculative-tokens 7`, at `--max-num-seqs 8` so a step stays within 64
  rows). The kernels compile on each width's first step (each rank compiles
  its share); a cold FlyDSL cache adds a few minutes to the first requests.
- DCP is optional (`--decode-context-parallel-size`); it uses the shared FP8
  index-key exchange for prefill and the score-only candidate merge
  (`flydsl_dcp_topk_merge`) for decode. KDA keeps the complete token sequence
  on every rank.
- Local prefix caching (`--enable_prefix_caching`) reuses KDA's PAGE-backed
  state checkpoints. MLA and index keys share scheduler block identity, and a
  hit requires a restorable recurrent checkpoint.

## Validated scope

`moonshotai/Kimi-K3.1-Pro` (the checkpoint now published; same tensors as the
first K3.1 release, new values): GSM8K 5-shot on the full set (1319 questions)
with the launch above, mono on: flexible-extract 97.65%, strict-match 97.65%
(±0.42), no fault. The results below are from the first K3.1 release.

GSM8K 5-shot on the full set (1319 questions) with the launch above and mono
off: flexible-extract 97.19%, strict-match 97.19% (±0.45), no fault.

Mono, paired against the original path on the same runs (TP8 DCP8, ISL / OSL
1024): GSM8K 96.66% vs 96.13–96.51% at concurrency 16, disagreements within
the same-code band; throughput at concurrency 64 2713 -> 3038 tok/s (+12%).
With DSpark (pinned acceptance length 4.59) and ReplaySSM, concurrency 1 / 4 /
8: 321 -> 562, 919 -> 1354, 1656 -> 1931 tok/s; GSM8K mono with ReplaySSM
96.21%, the original path with ReplaySSM 96.21%; 5-shot with mono, DSpark
(no pinned acceptance) and ReplaySSM at `--max-num-seqs 8`: 96.97% (±0.47).

Only that configuration has been validated: TP8 with DCP8,
`--kv_cache_dtype bf16`, `--index-cache-dtype fp8`, CUDA graphs on, with mono
on or off, and with the DSpark drafter and ReplaySSM above.
PCP is rejected while the model is built (the indexer's projections live in
the fused q_a projection, and the PCP indexer projects k itself). Nothing
else is rejected at startup, but online quantization of the other layers,
PP, expert parallelism, DP attention, other speculative methods, plugins,
TBO and KV transfer are untested. The vision wrapper is registered, but image
inference has not been validated.
