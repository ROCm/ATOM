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
# Install/run from a full checkout so Python imports its atom package.
# Use PYTHONPATH only for this checkout; the tool asserts module path ownership.
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode side --repeats 128
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode mono-alone --repeats 3
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=5s 300s   python tools/mono_mailbox_copy_repro.py --mode copy-alone --repeats 3
```

For a rollback control, prepare a separate clean checkout at the original PR's
parent (79149c7f10c0a58fce13997fcece3e1bd123bd27), copy these two tool files into
it unchanged, and repeat with its own fresh JIT cache and the same dependencies,
GPU, masks and inputs. Do not monkeypatch the current wait functions or overwrite
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
A successful run emits one complete_round per requested sample, checks bytes and
Q/ring, then emits complete and must also exit normally. Completion is still only
a sampled liveness result; a rare hang remains possible and the hardware root
cause and Store/Mono residency/ordering restrictions are not solved by backoff.

Historical larger fixture: original sleep1 backcheck repeated the hang; sleep16
passed 544 concurrent mini cases and three private Store/native smoke cases.
This checked-in extracted harness must be validated separately on its exact
revision; those old results are not automatically results of this new tool.
