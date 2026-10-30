# Attention stride A0 — prepared, GPU unrun

The user approved implementing the next complete attention candidate. GPU
allocation **253987** is pending; see [RESOURCE.md](RESOURCE.md). This directory
is editable preparation. There is no SOURCE freeze, weight load, GPU result,
service result or model accuracy acceptance yet.

The candidate removes the four Q/scale copies and output concatenation from
the FP8 owner adapter. It keeps four native H16 calls, native split selection,
CSR, sinks, original projection arithmetic, four K2048 wo_b BF16 partials and
their original rank sum. `binding.py` changes only the adapter import in a
private AST binding of the production forward; production files are unchanged.

`weights.py` declares only two real native Attention modules and their mHC
weights, checks checkpoint headers/parameter mappings, and uses the production
weight dispatcher and postload order. No full-model load is needed. `fixture.py`
uses native cache geometry, ScorePlanner, tentative metadata, compressor/index
writers and FULL layer8 selection consumed by REUSE layer9. Hidden states and
finite historical FP8/FP4 records are **constructed**, not a recorded service
trajectory. Layer9 has a separate constructed residual input; no FFN between
these two windows is timed or claimed.

The measured window is `Block.prepare_attention -> full forward_owner ->
Block.prepare_ffn`, including preparation, cache/index, attention, inverse RoPE,
wo_a/wo_b and mHC. The main local capacity is384/global1536; the second numerical
case is221/global883 valid rows, with one padded row on rank3. Three input
updates, native-self, native/candidate eager/graph, all output bytes, full
private cache bytes and replay idempotence are checked before timing. The GPU
adapter precheck also verifies native kernel/split equality, direct input/output
pointers, sentinel-overwrite coverage including an empty CSR row, and graph
release, before weight loading. The main cache layout is prepared before the
weight load as well.

The full windows use three ABBA rounds in the same load. Preparation, four-H16
attention and projection/mHC costs use the retained two-device-marker helper;
each observer must show at most1% absolute perturbation of its complete window.
Separate observed intervals are not additive whole-service accounting. All
graphs are reset before cache/communicator retirement. `review.py` is new and
unrun; run it once only after a successful retired/exported run. Preserve any
first-run failure separately; do not edit frozen source to rerun it.

CPU preparation passed Python syntax, Black, Ruff, unchanged-native-math AST,
independent native C stride fields, full/ragged alignment and output partition
checks. `LOCAL_PREPARATION.json` records hashes. These are not GPU qualifications.
Local Python has no Torch; runtime imports, model bindings and kernel checks
remain for the exact GPU container.

Reuse: the exact existing image, r8 executable-cache restore/pool helpers,
native profile configuration and retained device clock. Direct new source
bindings are recorded by the new launcher, with unchanged dependency receipts;
no old experiment or reviewer runs. Supply1.03969x remains retained evidence:
include its qualified fixed-gate path (native fallback outside eligibility) in
the later combined service candidate if this attention A/B is beneficial. This
attention-only run does not claim that supply has been integrated into production.

## Resource and launch

253987 requests one exclusive 8-MI355X node for the permitted maximum three days,
with `exec sleep infinity`. Last checked02:54 UTC: PENDING/Resources, scheduler
estimate2026-10-30 04:10:41 UTC, planned v3-049. No alternative job was submitted.
The only currently idle nodes belong to a dedicated partition outside this
account's associations; the dev queue is quota constrained. Recheck the actual
allocation, enter interactively through slog-005 and srun, verify all8 idle,
and inspect the exact image before launch. If missing, the retained reference
is `rocm/atom-dev@sha256:32c81e56e4680065b6ea3e0ca8363a00cd3ccb932d8990e83d799b8bbb58a4b3`.

On the verified allocated node:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /home/yifehuan/dsv41_commfusedmoe/validation/dashboard_whole_20261029/attention_stride_r1/launch.py
```

The launcher rechecks job/user/node/Exclusive=1/8GPUs, idle GPUs, runtime HEAD,
image and fresh storage bounds; it freezes SOURCE immediately before execution.
Outputs are capped at2MB, with export duplication included, one976MiB temporary
pool, and retirement before SHA-verified export. The allocation is retained.

CPU storage estimate before final docs: validation3,552,344,064B; recent-group
bound including4,256,000B reserve139,194,112B<148MB; conservative total including
retained remote charges and976MiB pool9,120,318,976B<10GB. This estimate is not
launch admission; the launcher recalculates it. Never restore the stale
8,828,348,672B fixed base or rerun completed cleanup/review programs.
