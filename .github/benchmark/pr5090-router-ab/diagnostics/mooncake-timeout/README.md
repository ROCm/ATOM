# PP4 / DEP4 native timeout probe

Diagnostic only. Keep ten sender threads, twenty sender tasks and the native
deadline unchanged. This code does not constitute a fix or a new SA result.

The native `engine.so` stays byte-identical to SHA256
`540428ce937a5be29726195730246f053f88d3ddfabaa2af33ad79c7a92198a8`.
The launcher verifies that hash in the derived worker. The probe additionally
checks the intercepted function's ELF-relative address before reading its layout.
It interposes the existing `MultiTransport::freeBatchID` overload, forwarding
its original arguments and opaque C++ return unchanged. It never calls status
polling, modifies completion fields, inserts sleeps, or changes the deadline.

The snapshot is taken only when the caller is already freeing a batch whose
completion status was not cached. For the synchronous wrapper, that occurs after
a failure/timeout or a submit error. The first three nonempty such batches per
process are recorded. Successful frees only perform the guard/one-byte status
read after one-time hook initialization; output is bounded. Callback activity
can change during the snapshot, so a single mismatching count is evidence to
investigate, not proof of a race.

Batch/task offsets are from the exact installed ELF and the inspected public
0.3.14 header. Task stride is 152 bytes with metrics enabled. Slice samples use
only the common prefix; no unverified RDMA-union offsets are read. No KV tensor
contents, credentials, or request text are recorded. The connector records its
registration ranges/lifecycle and at most three failed descriptor lists, keyed
by PID/native TID to correlate with native snapshots.

CPU checks:

```sh
python3 test_probe.py
python3 test_connector_probe.py /path/to/pinned/mooncake_connector.py
```

The first builds a RTLD_LOCAL fake engine and exercises the actual C++ sret ABI,
callback forwarding, unchanged batch bytes, ten concurrent callers, successful
batch suppression, pending slices, stale-counter fixtures, and a mismatched-ELF
guard. It is a probe test; it does not reproduce the live transport bug.
The second extracts and executes the real pinned `_send_blocks` AST after
instrumentation, checking argument/return/statistics preservation, range edges,
wrong-source rejection and failed-output handling.

Live evidence must include `hook_active` with `layout_compatible=true`, four P
and four D registration records, the first native stalled batch, and the matching
Python failed-write record. Missing fields or truncated task/slice samples must
be disclosed. Keep original failed workload codes and verify GPU release.

The first live capture (`Lf13536`) found 5,005 unfinished tasks before cleanup;
the 128 sampled slices used RDMA and remained PENDING. A later 50,610-task
snapshot was taken after termination began and is not independent root-cause
evidence. Batches over 65,536 tasks exceed the native probe's read bound; the
current `unsupported_batch_layout` marker alone does not establish corruption.

The follow-up uses a separate, pinned-base Ubuntu debugger image. It shares only
the exact owned prefill container's PID namespace, with no network or GPU devices.
After the first correlated failure, before terminating P/D, it attaches to that
recorded worker PID and captures native stacks for at most 25 seconds. The model
worker's packages, native engine, sender count, and deadlines stay unchanged.
GDB uses one symbol worker and loads only the engine and basic runtime symbols;
it disables auto-load scripts and debuginfod. A capture-specific label scopes
cleanup even if the Docker client times out. A debugger pause can affect events
after attachment, so keep the original pre-attach snapshot and timestamps.

Validate native symbol loading, not merely the presence of Python/libc frames:
build the debugger image, then run `debugger/test_native_symbols.py IMAGE` on a
Linux CPU host with Docker and g++. It loads a real C++ `engine.so` fixture and
requires a named `mooncake::WorkerPool::performPostSend` frame, mappings,
registers, successful detach, and sidecar cleanup. The earlier compound
`sharedlibrary` filter did not load the native engine; the separate library
filters avoid that failure. Raw PCs from that capture can be recovered only
within the explicitly documented bounds, not treated as a complete backtrace.

Task details now retain at most eight completed controls and prioritize unfinished
tasks. The older first-128 sampling hid the unfinished tail behind a completed
prefix in `Lcffeae`; its aggregate completion counts remain usable. A native CPU
fixture with 140 completed tasks followed by 20 pending ones reproduces that
sampling failure and verifies the corrected sample without changing batch bytes.

The debugger also records the engine mapping and registers. Its optional queue
decoder reads only at two PCs whose instructions and layout were checked against
the exact engine hash: the full-queue metadata loop and the incoming-queue mutex
call return. It reports queue length/cursor or mutex owner plus worker counters.
It does not call functions in the inferior or change its memory. An absent sample
at those PCs is not evidence that the queue is empty; a CPU layout fixture checks
the decoder, while the next live capture must still establish the actual values.
