# DEP8 kernel/model integration (draft)

Companion to plugin_poc's DEP8 kernel package and ATOM#3639. Kernel-library patches
and GEMM tables live in plugin_poc, not in this PR. The shared-activation FlyDSL
builder lives in ops.silu_medium.dep8_kernel; ATOM keeps its explicit call-site
selection and unchanged unfused fallback. Installing the package does not activate it.

This PR connects native-MegaMoE and its safe single-stream/shipped-BF16 return
rules, FFN seam prequant reuse, exact shared act/quant, attention-chain dispatch,
draft two-buffer attention, lean greedy head, single-argmax verifier, and the
serial maxima-grid choice. All experimental switches remain default off.
The lossless sampling branch and current-main BF16/scale guards are retained.
The rejected MXFP8 draft head/shadow experiment is not included.

Some compound Triton attention/verify code remains in ATOM because it operates on
ATOM-owned cache and speculative state, without an existing generic provider
seam. Those parts need review; they are not disguised as admitted plugin routes.
No new per-plugin patch of ATOM is shipped in plugin_poc.

Prerequisites: plugin kernel package (AITERbaseb9e08daca, FlyDSL0.5.0),
plugin#115 -> #112; #3639 supplies the prologue. Scheduler/prefix policy is
a separate ATOM PR, independently default-off. Current main is merged into
the #3639 base so recent fixes are preserved.

Validation: targeted combined CPU selection339 passed/44 skipped in the
no-driver environment. Moved activation66 GPU native-byte/replay cases pass.
The full local CPU suite hits92 collection errors in the ROCm/AITER mock
environment; it is NOT a pass. GitHub's CPU-only lane is the relevant CI follow-up.

The229.11/213385.82 AgentX result belongs to the pre-rebase frozen full stack,
not this branch. Raw point:
`/root/runs_night/v_sa_r4_all_i96_h1014/dep8-c256-20261120-112420`.
GSM8K500x3 of that stack .896/.894/.914 passed the frozen local paired screen.
The present rebase and relocated kernel library require a dedicated full-server
accuracy/performance rerun before merge. No isolated additive gain is claimed.
