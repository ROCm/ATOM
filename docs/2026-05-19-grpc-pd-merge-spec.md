# PD Stream Merge — State Machine Spec

**Date**: 2026-05-19
**Companion to**: `2026-05-19-grpc-engine-extraction-design.md`
**Scope**: `merge_pd_streams` — the private function (in the engine PD-merge file; see main design §3.2 for the 3-file split) that combines prefill and decode worker streams in PD mode into a single `WorkerStream<TokenChunk>`.

This function is the highest-risk single piece of code in the refactor: today's equivalent (`process_dual_streaming_chunks` in `regular/streaming.rs:595`) has been the source of multiple historical bugs around stream lifecycles and partial-failure paths. This spec is the precondition for writing the implementation — code that doesn't conform to §3 should not pass review.

---

## 1. What PD merge actually does (today's behavior, decoded from code)

Read of `regular/streaming.rs:595-643` and `:829-878` and `proto_wrapper.rs::ProtoStream` shows the actual semantics are surprisingly narrow:

1. **Prefill stream produces zero token chunks** in PD mode. It is consulted for exactly one signal: a terminal `Complete` message (which may carry `input_logprobs`) or a terminal `Error` message.
2. **Decode stream is the only source of token output.**
3. **Drain-or-skip on prefill is decided by request**: if the client requested `input_logprobs`, prefill must be drained until its terminal message arrives *before* yielding any decode chunk. If not, prefill is left untouched.
4. **`input_logprobs` is on prefill, but belongs on the consumer's Complete chunk** — so the merger must extract it from prefill's Complete and stuff it into decode's Complete.
5. **Lifecycle**: on normal completion, the prefill stream is "marked completed" to suppress `AbortOnDropStream`'s cancellation message. On any failure path or consumer drop, both upstream streams are dropped and their `AbortOnDropStream` wrappers send cancellation.

The function is not a stream "merger" in the conventional sense — it's an **attach-prefill-to-decode** operation with conditional drain. Naming it `merge_pd_streams` is a mild misnomer carried for continuity with today's `process_dual_streaming_chunks`.

---

## 2. `TokenChunk` field-source semantics (binding)

The following must be documented as doc-comments on `TokenChunk::Complete` and enforced by tests:

| Field | Single mode source | PD mode source |
|---|---|---|
| `token_ids` | worker's Complete | decode's Complete |
| `finish_reason` | worker's Complete | decode's Complete |
| `matched_stop` | worker's Complete | decode's Complete |
| `usage` | worker's Complete | decode's Complete |
| `logprobs` (output) | worker's Complete | decode's Complete |
| **`input_logprobs`** | **worker's Complete** | **prefill's Complete (injected by merger)** |
| `meta.cached_tokens` | worker's Complete | decode's Complete |
| `meta.request_id` | worker's Complete | decode's Complete |
| `meta.weight_version` | worker's Complete | decode's Complete |

**The `input_logprobs` row is the entire reason this function is subtle.** In single mode, the worker fills it on its Complete and the engine proto-adapter file's sglang/vllm → `TokenChunk` converter copies it through. In PD mode, decode's Complete will have `input_logprobs = None`, and the merger must inject the value extracted from prefill's Complete.

If the proto-adapter's converter unconditionally copies `input_logprobs` from the proto Complete in both single and PD modes, PD mode will silently produce `None` even when requested. **Test T2 catches this.**

---

## 3. State machine

### States

```
WaitingPrefill            — entered only when need_input_logprobs == true
Streaming { pending_input_logprobs: Option<InputLogprobs> }
Terminal                  — absorbing; subsequent input is ignored
```

### Inputs

- `prefill.next() -> Option<Result<ProtoChunk, tonic::Status>>` where `ProtoChunk` is one of `Partial`, `Complete { input_logprobs, ... }`, `Error { message }`, `None` (proto-empty).
- `decode.next() -> Option<Result<ProtoChunk, tonic::Status>>` with the same variants.
- External: consumer drop of the produced `WorkerStream`.

### Outputs

- `WorkerStream::Item = Result<TokenChunk, EngineError>` — yielded to the consumer.

### Transitions

```
Entry:
  if !need_input_logprobs:
      → Streaming { pending_input_logprobs: None }
  else:
      → WaitingPrefill

State: WaitingPrefill
  ╭─ on prefill.next() = Some(Ok(Partial _)):
  │    loop (ignore; PD prefill should not emit Partial but tolerate it)
  ├─ on prefill.next() = Some(Ok(None)):
  │    loop (ignore proto-empty)
  ├─ on prefill.next() = Some(Ok(Complete { input_logprobs, .. })):
  │    → Streaming { pending_input_logprobs: input_logprobs }
  │    (do NOT yield anything; do NOT consume decode yet)
  ├─ on prefill.next() = Some(Ok(Error { message })):
  │    yield Err(EngineError::Prefill(message))
  │    drop decode  (AbortOnDrop fires)
  │    → Terminal
  ├─ on prefill.next() = Some(Err(status)):
  │    yield Err(EngineError::Transport(status))
  │    drop decode
  │    → Terminal
  ╰─ on prefill.next() = None:                          [stream closed without Complete or Error]
       yield Err(EngineError::PrefillEarlyClose)
       drop decode
       → Terminal

State: Streaming { pending_input_logprobs }
  ╭─ on decode.next() = Some(Ok(Partial { token_ids, logprobs })):
  │    yield Ok(TokenChunk::Partial { token_ids, logprobs })
  │    (no state change)
  ├─ on decode.next() = Some(Ok(Complete c)):
  │    let c = match pending_input_logprobs.take() {
  │        Some(lp) => c.with_input_logprobs(lp),       // PD injection
  │        None     => c,                                // Single-style passthrough
  │    };
  │    yield Ok(TokenChunk::Complete(c))
  │    prefill.mark_completed()                          // suppress upstream abort
  │    → Terminal (clean exit)
  ├─ on decode.next() = Some(Ok(Error { message })):
  │    yield Err(EngineError::DecodeError(message))
  │    drop prefill                                      // AbortOnDrop fires
  │    → Terminal
  ├─ on decode.next() = Some(Err(status)):
  │    yield Err(EngineError::Transport(status))
  │    drop prefill
  │    → Terminal
  ╰─ on decode.next() = None:                            [decode closed without Complete]
       yield Err(EngineError::DecodeIncomplete)
       drop prefill
       → Terminal

Cross-cutting: consumer drops the produced WorkerStream at any point
  drop decode upstream  (AbortOnDrop fires)
  drop prefill upstream (AbortOnDrop fires)
  → Terminal
```

### Invariants

- **I1**: prefill's token chunks never reach the consumer. In `WaitingPrefill`, `Partial` from prefill is silently dropped.
- **I2**: when `need_input_logprobs == false`, prefill stream is never polled. `prefill.next()` call count must be 0.
- **I3**: when `need_input_logprobs == true`, no decode item is yielded before prefill's terminal message is observed.
- **I4**: on clean exit (decode `Complete` received), `prefill.mark_completed()` is called *before* dropping prefill. This suppresses the abort message that `AbortOnDropStream::Drop` would otherwise send.
- **I5**: on any non-clean exit (`Err` yielded by the merger), prefill is dropped *without* `mark_completed()`. The abort is the intended signal to the upstream prefill worker.
- **I6**: there is no timeout in this function. If prefill never sends a terminal message and the consumer never drops, the merger hangs forever. *This is today's behavior;* see §5 deferred concerns.

---

## 4. Test obligations

Each test corresponds to a transition or invariant. Tests use a `MockWorkerStream` that yields scripted `Vec<ProtoChunk>` and tracks `next()` call counts and Drop events.

### T1 — `need_input_logprobs=false` path skips prefill entirely (invariant I2)

```
Setup:   prefill scripted to yield [Partial×100, Complete{input_logprobs=None}]
         decode  scripted to yield [Partial×3, Complete{...}]
         need_input_logprobs = false
Drive:   collect all yielded items
Assert:  - 3 Partial + 1 Complete yielded to consumer (decode's chunks only)
         - prefill.next() called 0 times
         - prefill dropped at end without mark_completed
```

### T2 — `input_logprobs` injection from prefill into decode (critical, prevents silent data loss)

```
Setup:   prefill scripted to yield [Complete{input_logprobs=Some(IL)}]
         decode  scripted to yield [Partial, Complete{input_logprobs=None, ...}]
         need_input_logprobs = true
Drive:   collect all yielded items
Assert:  - yielded Complete.input_logprobs == Some(IL)
         - prefill.mark_completed() called exactly once
         - no Partial from prefill leaks to consumer
```

### T3 — prefill Error fails fast, decode not touched (invariant I3 + WaitingPrefill error path)

```
Setup:   prefill scripted to yield [Error { message: "OOM" }]
         decode  scripted with assertions to fail if polled
         need_input_logprobs = true
Drive:   poll merger once
Assert:  - one Err(EngineError::Prefill("OOM")) yielded
         - decode.next() called 0 times
         - decode dropped (AbortOnDrop side effect observed)
         - subsequent polls return None (Terminal)
```

### T4 — prefill closure during Streaming state is silently ignored

```
Setup:   prefill scripted to yield [Complete{input_logprobs=Some(IL)}] then close
         decode  scripted to yield [Partial×5, Complete]
         need_input_logprobs = true
Drive:   collect all yielded items
Assert:  - 5 Partial + 1 Complete yielded
         - merger does NOT poll prefill again after transition to Streaming
         - prefill.mark_completed() called after decode Complete observed
```

### T5 — decode Transport error mid-stream propagates and aborts prefill (Streaming error path)

```
Setup:   prefill scripted to yield [Complete{input_logprobs=None}]   (drained quickly)
         decode  scripted to yield [Partial×2, Err(Status::aborted("..."))]
         need_input_logprobs = true
Drive:   collect until first Err
Assert:  - sequence: Ok(Partial), Ok(Partial), Err(EngineError::Transport(_))
         - prefill dropped (AbortOnDrop side effect observed)
         - prefill.mark_completed() NOT called
```

### T6 — consumer drop mid-stream propagates to both upstreams (cross-cutting)

```
Setup:   prefill scripted to yield [Complete{input_logprobs=None}]
         decode  scripted to yield [Partial×infinite]
         need_input_logprobs = true
Drive:   take 3 items, then drop the WorkerStream
Assert:  - both prefill and decode received Drop (AbortOnDrop side effects observed on both)
         - no late items after drop
```

### T7 — prefill never produces terminal (hang behavior is intentional, document with test)

```
Setup:   prefill scripted to be a pending forever stream (returns Poll::Pending always)
         decode  scripted to yield [Partial, Complete]
         need_input_logprobs = true
Drive:   poll merger with a 100ms tokio timeout
Assert:  - timeout elapses, merger has yielded 0 items
         - dropping the WorkerStream cleanly aborts both upstreams
         - (This documents that the merger has NO internal timeout; cancellation
            must come from the consumer side. If a future change adds a timeout,
            this test must be updated and the change called out in review.)
```

---

## 5. Connected design corrections (changes to main design)

The state machine in §3 implies five small corrections to `2026-05-19-grpc-engine-extraction-design.md`:

### 5.1 `TokenChunk::Complete.input_logprobs` source documentation

Add doc-comment to the `TokenChunk::Complete.input_logprobs` field (in the grpc-internal `TokenChunk` type-def file) describing the PD-mode injection (see §2 table). This is the *single* surprising behavior in the entire `TokenChunk` type and must be discoverable from the type definition.

### 5.2 `EngineError` enumeration

Defined alongside `TokenChunk` (same type-def file) since it's part of the engine's return type. Must define at minimum:

```rust
pub enum EngineError {
    Transport(tonic::Status),       // any tonic-layer failure
    Prefill(ProtoErrorMessage),     // prefill yielded a proto Error
    DecodeError(ProtoErrorMessage), // decode yielded a proto Error (during Streaming)
    PrefillEarlyClose,              // prefill stream closed without Complete or Error
    DecodeIncomplete,               // decode stream closed without Complete or Error
    ConnectionAcquireFailed(String),
    RequestBuildFailed(String),
}
```

The first five are the ones the state machine produces. The last two are for `dispatch_one` before any stream exists. **Status (2026-05-19 revision)**: main design §3.2 has been updated to reflect this complete enum; the "+ a few internal categories" hand-wave is gone. Keep both docs in sync on any future change.

### 5.3 `need_input_logprobs` lives on `GenerationPayload`

`GenerationPayload.logprob: LogprobConfig` must include an `input_logprobs: bool` field. `merge_pd_streams` reads it from the payload (passed in by `engine.dispatch`):

```rust
fn merge_pd_streams(
    prefill: WorkerStream,
    decode: WorkerStream,
    need_input_logprobs: bool,    // = payload.logprob.input_logprobs
) -> WorkerStream { ... }
```

This is the single source of truth: protocol decoding produces it, engine threads it through, merger reads it. No drift.

### 5.4 `WorkerStream::Drop` does not need its own `AbortHandle`

For single mode, `WorkerStream` holds the tonic `Streaming<T>` directly; tonic's own Drop closes the H2 stream and cancels the RPC. No `AbortHandle` field is needed.

For PD mode, `WorkerStream` (the merger's output) holds both prefill and decode upstreams as owned `WorkerStream` values. Its Drop propagates by dropping both, which fires their respective `AbortOnDropStream` wrappers.

**Strike "Holds the underlying tonic AbortHandle" from the main design §3.2 `WorkerStream` doc-comment.** Replace with: "Single mode holds one tonic stream; PD mode holds both upstreams. Drop in either case propagates cancellation via tonic-native H2 stream close."

### 5.5 `merge_pd_streams` location

The function is a private `fn` in its own file (the engine PD-merge file per main design §3.2 — not the dispatch file or the proto-adapter file). It is implementation detail of `GrpcEngine::dispatch`. **Not exposed at the module boundary** — consumers see only `WorkerStream`.

---

## 6. Deferred concerns (intentionally out of scope)

- **No timeout on PD merge**: today's behavior is "hang until consumer drops". §3 invariant I6 documents this; T7 tests for it. Adding a timeout is a separate change that requires deciding the value, the metric, and the user-facing error. Not in this refactor.
- **No bootstrap signal flow through `merge_pd_streams`**: today's PD bootstrap (KV handshake, host/port exchange) happens at the proto level via `inject_bootstrap_metadata` *before* the streams are created. The streams themselves don't carry bootstrap signals. If a future PD protocol moves bootstrap into the stream, this spec must be revised.
- **No `n>1` interaction**: today vLLM doesn't support `n>1`, SGLang does, and `n>1` chunks have an `index` field. The merger doesn't inspect `index` — both prefill and decode handle `n>1` natively in their respective Complete messages. If `input_logprobs` ever becomes per-index, the merger needs revision.
