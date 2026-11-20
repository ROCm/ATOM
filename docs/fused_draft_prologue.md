# Fused DSpark draft prologue (opt-in)

## Scope

`ATOM_DSPARK_FUSED_PROLOGUE=1` combines `draft_step_indices`, the local
embedding gather, and `build_block_state`: three launches become one.
Five outputs retain the same shape, dtype and arithmetic (copies and integer
indexing). Default off. A non-local/TP-sharded embedding, CPU table, missing
table or non-unit inner stride keeps the existing path.

This is separate from #3438's state-initialization fusion: this change also
reads anchors directly from the embedding table and writes the draft index
lists. Neither the head, confidence computation, attention nor verifier changes.
The new optional `state` argument avoids re-building the already prepared
state inside `draft_hidden`.

## Recorded GPU evidence (2026-11-20)

MI355X/gfx950, node crsuse2-m2m-v3-013, container xyt_c256, GPU7, cooperative
GPU window. Original r1 integration; synthetic embedding rows matching the
checkpoint dimensions (V=129280, H=5120, width=5, hc_mult=4, window=128).
These are **isolated CUDA-graph prologue timings**, not full draft or E2E.

| Requests/rank | Native, us | Fused, us | Reduction |
| --- | --- | --- | --- |
| 11 | 9.85 | 3.23 | 67.2% |
| 13 | 10.51 | 3.30 | 68.6% |
| 15 | 11.22 | 3.36 | 70.1% |

Each batch: eight input sets (empty/partial/full history, ring boundary,
first/last embedding row); all five outputs matched eagerly and after replays.
Original command: `python bench_prologue.py 11,13,15` in the worker environment.
Raw record and harness:
`/shared_nfs/xiaobizh/yuxu_c256/kopt/gpu7/r2/{bench_prologue.py,out_prologue.txt,prologue.json,NOTES.md}`.
Both submitted device/helper function ASTs were checked against the measured
integration copy. Surrounding ATOM code is rebased to main
`ecec81921da0c27a088d542fcefad51fe83e4d5a`.

## Reproduce and remaining gates

- CPU routing checks: `python tests/test_dsv41_fused_draft_prologue.py`.
- Standalone GPU check, on an idle locked ROCm GPU:
  `python tests/model_ops/check_fused_draft_prologue_gpu.py`.
  Covers eager, empty batch and graph replay after changing inputs.
  This new portable script has not yet been executed on hardware.
- Final rebased full-server accuracy/performance and hardware rerun remain
  pending. Existing integration GSM8K is not isolated attribution for this PR.
- No throughput, SA, B300, cross-architecture or model-quality claim.
