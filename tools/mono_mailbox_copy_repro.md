# Mono mailbox backoff reproducer (gfx950)

This tool exercises the production V4.1 K1/StepMailboxes code, not a replacement
spin loop. It uses synthetic nonzero FP8 weights, 6 token rows, projection TP4
with a single local peer, a noncooperative 256-CTA x 512-thread K1 launch, and
32 captured K1 launches replayed 32 times per sample. A second full-256-CU HIP
mask stream (both flags0) runs native KV unpack with 724 blocks / 267,822,080 useful bytes. The current public
TransferSlotPool supplies one window-sized contiguous Slot.tensor; registration
is an explicit offline no-op, so this is not an RDMA/MR test. This differs from
the historical diagnostic 362 padded-slot/strided-unpack fixture. It does not
restore removed APIs or claim historical results for this new layout.
No model checkpoint or running Store service is needed.
It checks full restored KV bytes and the K1 Q/ring output against its initial
complete control run. This is a mailbox/copy liveness probe, not precision or
service performance validation, and events do not prove physical CU overlap.

Use the repository's matched ROCm PyTorch, AITER and FlyDSL development runtime
(including its real native launch extension). Expose exactly one idle, owned
MI355X. Do not run on a device with an existing service or after a timeout until
its prior process/KFD ownership has actually gone away.

```
# First validate the actual pool allocation, prepared ID upload and GPU pack/unpack:
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s python tools/mono_mailbox_copy_repro.py --setup-only
# Run from the checkout root. PYTHONPATH="$PWD" makes Python import this
# checkout's atom package, which the tool's ownership check requires.
# 128 rounds is a smoke run. To change POLL_SLEEP_UNITS use --repeats 1024
# (see "Before lowering" below).
# --deadline SECONDS (default 15) is how long a round may take on the GPU before
# the tool reports pending_deadline and waits for the external timeout.
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode side --repeats 128
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode mono-alone --repeats 3
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode copy-alone --repeats 3
```

For a rollback control, make a separate clean checkout of the same revision and
revert only the backoff there. Every site this PR changed must go back:

1. `atom/mono/polling.py`: set `POLL_SLEEP_UNITS = 1`. That restores the
   `s_sleep(1)` that main had at `atom/mono/device/sync.py` (two polls) and
   `atom/mono/runtime/step_begin.py` (three polls), which now use
   `POLL_SLEEP_UNITS`.
2. The three `spin_pause` definitions (`atom/mono/device/ops.py`,
   `atom/models/deepseek_v4/mono/kernels/primitives.py`,
   `atom/models/deepseek_v41/mono/hi/device/gfx950/ops.py`): emit `"s_nop 0"`
   instead of `f"s_sleep {POLL_SLEEP_UNITS}"`.
3. `atom/models/deepseek_v4/mono/kernels/helpers.py`, function `poll()` (two polls,
   `rocdl.s_sleep(polling.POLL_SLEEP_UNITS)`: the bounded loop under `poll_timeout_ticks`
   and the unbounded loop in the `else` branch): change both back to `rocdl.s_nop(0)`.

`sync.py::sleep_ticks` keeps `s_sleep(1)` (harness delay injection, unchanged by
this PR). The V4.1 K1 that this tool
exercises does not go through item 3, but the full recipe is what restores the
pre-backoff polling instructions. Repeat the commands above in that checkout with its own fresh JIT
cache and the same dependencies, GPU, masks and inputs. Do not monkeypatch the current wait functions or overwrite
the candidate checkout. Save both source identities, logs, process exit codes,
per-round busy preconditions and post-process KFD/VRAM ownership checks. The tool
logs the actual K1/sync import paths and sync SHA256. A side sample in which Mono
was already complete before the copy is not evidence of concurrent-copy stress.
A successful side run requires at least one busy sample and one positive GPU
event-window intersection; it still does not prove physical CU residency.

On a polling deadline the tool emits pending_deadline and retains all buffer,
metadata, graph and stream owners while waiting for the external watchdog.
On other exceptions it logs the original error/traceback, retains frame owners,
and waits for the watchdog too. A failed setup/import/submit is distinguished
from pending_deadline; watchdog exit124 alone is not evidence of GPU hang.
It deliberately does not report success or prematurely release in-flight data.
A successful run emits one complete_round per requested sample (Q/ring is checked
when Mono runs, restored bytes when the copy runs, so both in `--mode side`),
then emits complete and must also exit normally. Completion is still only
a sampled liveness result; a rare hang remains possible and the hardware root
cause and Store/Mono residency/ordering restrictions are not solved by backoff.

Before lowering `POLL_SLEEP_UNITS` (currently 16), 128-round runs are not enough:
8 passed 44/44 such runs but hung once in four 1024-round runs, while 16 passed
12/12 1024-round runs. Use `--mode side --repeats 1024` with `timeout` of at
least 300s, on several GPUs, at least 12 runs for the candidate and for 16 in the
same batch; any hang rejects the candidate. Passing is still only a sampled
liveness result on this synthetic workload, not a proof of safety.

Historical larger fixture: original sleep1 backcheck repeated the hang; sleep16
passed 544 concurrent mini cases and three private Store/native smoke cases.
This checked-in extracted harness must be validated separately on its exact
revision; those old results are not automatically results of this new tool.
