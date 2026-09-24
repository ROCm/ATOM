# Kimi-K3 Wide-EP on gfx1250 (MI455) — 4 nodes, dp16ep16

Full-size **Kimi-K3** (2.78T, 93 layers: 24 full-attention MLA + 69 KDA linear,
896 routed experts top-16, MXFP4 routed experts / BF16 everything else) served
across **four gfx1250 nodes, 4 GPUs each, in SPX** — 16 ranks total, `-tp 1`
with data parallel 16 and expert parallel 16. The checkpoint is ~1.42 TiB and is
needed **in full on every node**; it is not sharded across the cluster.

This is a bring-up configuration, not a tuned deployment. Read the translation
section first — on this silicon it is the difference between 95% and 0%.

| | |
|---|---|
| Hardware | 4 × gfx1250 (MI455), 4 GPUs/node, SPX, 288 GiB/GPU |
| Parallelism | `-tp 1 --data-parallel-size 16 --data-parallel-size-local 4`, EP16, DP attention |
| Attention | Triton MLA (`ATOM_USE_TRITON_MLA=1`), unshuffled KV |
| Fabric | UALink within a node-set sharing one `PPOD_ID` + `VPOD_ID`; RCCL over the data-plane NIC |
| **GSM8K (lm_eval, 5-shot, full 1319)** | **strict-match 0.9545 ±0.0057**, flexible-extract 0.9538 ±0.0058 |
| Throughput / TTFT / TPOT | **not characterized yet** — see [Not yet measured](#not-yet-measured) |

---

## ⚠️ B0→A0 code-object translation is mandatory

**Read this before anything else. There is no configuration of this recipe that
works without it, and skipping it does not produce an error.**

These parts are **B0 revision silicon**, and the kernels in the bring-up image
are compiled for **A0**. `libhsa_hotswap_rocjitsu.so` attaches to the HSA tool
interface and rewrites each code object from B0 to A0 as it is loaded. Without
that layer the GPU executes code built for a different revision.

The result is not degraded accuracy. It is **every generated token being `!`**,
and **GSM8K 0%**. The failure is completely silent: HTTP 200, `finish_reason:
"stop"`, sane token counts, plausible throughput and TTFT in the metrics. A
benchmark will run to completion and produce a clean-looking report built
entirely from `!`.

> **`HOTSWAP=0` is not a fallback.** Its only use is reproducing this bug.
> Any accuracy or performance number produced without translation is void.

### The image's own rocjitsu is not a substitute

The image ships a hook and a translator, and they are the wrong ones. The hook
installs, logs `installed eager gfx1250 B0-to-A0 hook`, and then translates
nothing — because `rj_pretranslate` derives the translation store's location
from the **translator's own install prefix**, and the image's prefix has an
empty store.

| | image's own | required prefix |
|---|---|---|
| `libhsa_hotswap_rocjitsu.so` | 143217 B | **129112 B** |
| `librocjitsu_gfx1250_b0_to_a0.so` | 7531385 B | **8002216 B** (0.3.0) |
| translation store entries | 170 | **1976** |
| translations actually performed | **0** | **592** (484 translated + 108 reused) |

### Where to get it

**`j07-01:/home/zejchen/rocmjit.zip`** — 424 MB, unpacks to a 1.7 GB prefix.
An unpacked copy sits next to it at `j07-01:/home/zejchen/rocmjit/`.

The prefix originally lived at `j07-04:/tmp/rjprefix`. **That path is gone** —
`/tmp` is cleared on reboot. Its absence is expected and is not a reason to
conclude the node cannot serve; take the archive above. If you are outside this
cluster, ask your AMD contact for the gfx1250 B0→A0 rocjitsu prefix by the
checksums in the table.

### Install it on every node

`docker cp` puts it inside the container, so it is **lost when the container is
removed** (unlike a bind mount) and must be re-installed after any
`docker rm`. `docker stop` / `start` keeps it.

```bash
for n in <node0> <node1> <node2> <node3>; do
  scp -q ~/rocmjit.zip "$n:~/" &
done; wait

for n in <node0> <node1> <node2> <node3>; do
  ssh -q -o LogLevel=ERROR "$n" '
    cd ~ && [ -d rocmjit/share ] || unzip -q -o rocmjit.zip
    sudo docker cp ~/rocmjit k3ep16:/app/rjprefix
  ' &
done; wait
```

Then, before launching the server (inside the container):

```bash
export LD_LIBRARY_PATH=/app/rjprefix/lib:$LD_LIBRARY_PATH   # must be first
export HSA_TOOLS_LIB=/app/rjprefix/lib/libhsa_hotswap_rocjitsu.so
export HSA_HOTSWAP_VERBOSE=1
```

Putting the prefix's `lib` **first** on `LD_LIBRARY_PATH` is what makes the
populated store reachable — that is the mechanism, not a precaution.

> `librocjitsu_gfx1250_b0_to_a0.so.0` cannot be used as `HSA_TOOLS_LIB`
> directly: it exports only `rj_gfx1250_b0_to_a0_translate` / `_free` and has no
> `OnLoad`, so HSA ignores it silently. It is a `NEEDED` dependency of the hook,
> which loads it.

### Verify — before trusting any output

Check the three identifying numbers on every node:

```bash
sudo docker exec k3ep16 bash -c '
  stat -c%s /app/rjprefix/lib/libhsa_hotswap_rocjitsu.so             # 129112
  stat -c%s /app/rjprefix/lib/librocjitsu_gfx1250_b0_to_a0.so.0.3.0  # 8002216
  ls /app/rjprefix/share/rocjitsu/translations/gfx1250-b0-a0/v1 | wc -l  # 1976'
```

Then check that translation actually happened at runtime. Seeing the hook
install is **not** sufficient — that is exactly what the wrong prefix also does:

```bash
grep -c 'outcome=translated'       <log>   # expect several hundred
grep -c 'reused tier'              <log>   # expect over a hundred
grep -c 'translation_status=[^0]'  <log>   # must be 0
```

A correct run logs lines of this shape:

```
[hsa-hotswap-rj] eager translation source_id=fnv1a64:44173c6a13023c91
    input_revision=b0 output_revision=a0 outcome=translated changed=23 ...
[hsa-hotswap-rj] reused tier=aot input_bytes=501616 output_bytes=505712 status=0
```

---

## Prerequisites

**All four nodes in one fabric domain.** Wide EP requires identical `PPOD_ID`
*and* `VPOD_ID`. A mismatch does not report an error — the rendezvous hangs.

```bash
for n in <node0> <node1> <node2> <node3>; do
  printf "%-10s " "$n"
  ssh -q -o LogLevel=ERROR "$n" \
    "amd-smi fabric 2>/dev/null | grep -E 'PPOD_ID|VPOD_ID' | head -2 | tr -d ' ' | paste -sd' ' -"
done
```

`BANDWIDTH: 0 Mb/s`, `LATENCY: 0 ns` and `VERSION: 4294967295` in `amd-smi
fabric` are **not** faults; this firmware does not populate those fields.

**Other requirements**

- A **gfx1250 bring-up image** with ATOM, aiter, mori and FlyDSL.
- Passwordless SSH between all four nodes (launch is fanned out over it).
- The full checkpoint on **every** node (~1.42 TiB each), plus swap — weight
  loading peaks well above the 250 GB of host RAM on these boxes.
- After a host reboot: `amdgpu` is blacklisted on the kernel command line, so
  `/dev/kfd` will not exist until `sudo modprobe amdgpu`. UALink then takes tens
  of seconds to train; `unconfigured` in
  `/sys/class/drm/card*/device/ualink/accel_state` is normal during that window.
  Do not re-`modprobe` once it reads `active` — that retrains the links.

---

## Container

One container per node, entrypoint `sleep infinity`, driven by `docker exec`:

```bash
sudo docker run -d --name k3ep16 \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host --pid=host \
  --group-add 39 --group-add 105 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  --shm-size=128g \
  -v <model-dir>:/models:ro \
  <gfx1250-bringup-image>
```

`--network=host` is required: the ranks address each other by the node's
data-plane IP. Stale segments from a crashed run wedge the next rendezvous, so
clear them between attempts:

```bash
sudo rm -f /dev/shm/psm_* /dev/shm/nccl-*
```

---

## One upstream change this configuration needs

**`ATOM_UNFUSED_GATHER_KV_B_PROJ=1`** — an unfused torch fallback for
`gather_kv_b_proj`. On gfx1250 the fused Triton kernel cannot be compiled for
the shapes a *chunked* prefill produces; Triton emits a PHI node with mismatched
operand types and LLVM asserts
(`PHINode::setIncomingValue: getType() == V->getType()`), aborting the process.
FlyDSL is not an alternative — it is gfx950-only. The fallback does not
implement the shuffled-KV layout, hence `ATOM_USE_TRITON_MLA_SHUFFLE_KV=0`.

This is being upstreamed; until it lands, this recipe assumes a build that has
it.

---

## Launch

Every node runs the **same** command; each rank derives its index by matching
its own NIC addresses against the node list, which also fixes
`MORI_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`. **The
addresses must be the data-plane NIC, not the management port.**

Only the first node serves the API on `:8000`; the other three compute only.

```bash
python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --trust-remote-code \
  -tp 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 4 \
  --data-parallel-rank <0|4|8|12> \
  --data-parallel-master-ip <node0-data-plane-ip> \
  --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
  --enable-expert-parallel --enable-dp-attention \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --cudagraph-mode FULL \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --no-enable_prefix_caching \
  --disable_uvicorn_access_log
```

> The validated run above was launched with `--cudagraph-mode FULL_DECODE_ONLY`
> rather than `FULL`. On the native engine the two are equivalent (see
> [KV budget](#kv-budget)), and `FULL` is the default, so the recipe uses it.

Do **not** set `--max-model-len`; let ATOM use the model's own
`max_position_embeddings` (1048576). Confirm `'max_model_len': None` in the log.
A small value silently truncates the generation budget the accuracy run needs —
see [max_tokens](#k3-is-a-reasoning-model).

### Environment

```bash
# --- translation: without these, every token is `!` ---
export LD_LIBRARY_PATH=/app/rjprefix/lib:$LD_LIBRARY_PATH
export HSA_TOOLS_LIB=/app/rjprefix/lib/libhsa_hotswap_rocjitsu.so
export HSA_HOTSWAP_VERBOSE=1

# --- architecture ---
export PYTORCH_ROCM_ARCH=gfx1250 AITER_RUNTIME_GPU_ARCH=gfx1250
export GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0                        # CK is not ported to gfx1250

# --- attention ---
export ATOM_USE_TRITON_MLA=1              # unset: first decode SIGABRTs, silently
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0   # the unfused gather has no shuffled layout
export ATOM_UNFUSED_GATHER_KV_B_PROJ=1
export ATOM_USE_AITER_TRITON_ATTN=1 ATOM_USE_UNIFIED_ATTN=1

# --- MoE ---
export ATOM_MOE_GU_ITLV=1                 # required on gfx1250, not a tuning knob
export ATOM_USE_TRITON_MOE_DECODE=0       # K3 activation is situ, not SiLU: asserts
export MEGA_DISPATCH=mori MEGA_WIRE=fp4 MEGA_DISPATCH_WIRE=fp4
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1
export AITER_USE_GROUPED_GEMM=1 AITER_USE_OPUS_MOE_SORTING=1

# --- GEMM / quantization ---
export ATOM_USE_TRITON_GEMM=1 ATOM_WO_A_USE_FLYDSL=1
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1

# --- communication ---
export NCCL_MNNVL_ENABLE=1                # off: silently falls back to TCP (~4.5 GB/s)
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export ATOM_DP_LM_HEAD_MODE=allgather     # required for multi-node + hipGraph
# these two must be set together or not at all (see Known issues)
export ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1

# --- loading ---
export HSA_XNACK=1 HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export ATOM_LOADER_USE_THREADPOOL=1 ATOM_LOADER_NUM_THREADS=4
```

A cold start takes about **25 minutes** — 96 shards, translation, and graph
capture. `ncclCommInitRank` hangs intermittently on this stack with no known
fix; retry. Give any retry wrapper **at least 25 minutes** or it will kill
servers that are loading normally.

To tell a hang from progress, look at the ModelRunner (`ATOM::DPxTP0`), not the
EngineCore, which only ever shows as waiting on a queue:

```bash
py-spy dump --pid $(pgrep -f 'ATOM::DP0TP0' | head -1)
```

`MainThread (idle)` in `as_completed` is a healthy load. `(active)` in
`ncclCommInitRank` is the hang.

### KV budget

```bash
grep 'Memory budget' <log> | tail -1     # available_for_kv must be positive
grep -oE 'experts=[0-9]+' <log> | sort -u  # EP16 -> 56
```

With the default `--max-num-batched-tokens 16384`, `peak_torch` and the
cudagraph estimate together reach ~77 GB and `available_for_kv` goes to about
**−33 GB**, so the server never starts. **`--max-num-batched-tokens 2048` is the
lever**: the profile run is what sets `peak_torch`, and
`_estimate_cudagraph_overhead()` derives the pool estimate from that same
allocator high-water mark, so shrinking the profile batch shrinks both terms.

`--cudagraph-mode` is *not* a second lever here, despite appearances. ATOM's
manual capture is already decode-only by construction — `capture_cudagraph()`
iterates `max_schedulable_decode_bs(...)` and prefill never replays a graph — and
nothing in the native runtime calls `mixed_mode()`, which is the only predicate
where `FULL` and `FULL_DECODE_ONLY` differ. The two are interchangeable on this
path, so this recipe uses the default `FULL`. (`FULL_DECODE_ONLY` *is* honored
when ATOM runs as a vLLM plugin backend, where vLLM's own runner reads
`mixed_mode()`.)

---

## Accuracy gate — run this before any benchmark

The two highest-impact failures on this platform (missing translation, missing
`ATOM_USE_TRITON_MLA`) are silent, and one of them produces a service that
benchmarks beautifully. Check the *content* first:

1. `/v1/models` responds.
2. **Sanity gate — abort if it fails:** generate a few short completions and
   assert the output is not all `!`, not a single repeated token, and that
   logprobs are not near-uniform.
3. Compare decode against prefill at `temperature=0` for the same prompts.
4. Only then, GSM8K.

A quick manual version of step 2:

```bash
curl -sS http://<node0>:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is",
       "max_tokens":16,"temperature":0}' | python3 -m json.tool
```

## GSM8K

Use `lm-evaluation-harness` (validated with 0.4.13) with the standard 5-shot
task YAML. Numbers from hand-written clients are not comparable and should not
be reported.

```bash
pip install --user 'lm_eval[api]'   # the [api] extra is required
```

Without `[api]`, `tenacity` is missing and the failure only surfaces *after* the
task data has loaded.

```bash
EP=<node0>:8000
MODEL=moonshotai/Kimi-K3

lm_eval --model local-chat-completions \
  --apply_chat_template \
  --include_path ~/lmeval/tasks \
  --tasks gsm8k \
  --model_args "model=${MODEL},base_url=http://${EP}/v1/chat/completions,\
api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,\
tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 \
  --output_path ~/lmeval/out --log_samples
```

### K3 is a reasoning model

The chat endpoint puts the chain of thought in `reasoning_content` and the
answer in `content`. If `max_tokens` runs out mid-thought, **`content` is an
empty string** while `finish_reason` is merely `length` — no error. `lm_eval`
reads only `content`, so a too-small budget scores **0**, not "slightly worse":

```
max_tokens=32    -> content=''            reasoning_content='The user is asking...'
max_tokens=3500  -> content='...#### 72'  finish_reason=stop
```

12288 is the validated value. This is also why the server must not cap
`--max-model-len`.

### Concurrency

`--max-num-seqs` is **per DP rank**, so dp16 admits `max_num_seqs × 16`. Size
`num_concurrent` accordingly — too low measures the client, not the engine.

### Result (full 1319 questions)

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9538|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9545|±  |0.0057|
```

39 minutes at `num_concurrent=32`, no retries or disconnects.

---

## Known issues

| Symptom | Cause / fix |
|---|---|
| Every token is `!`, service otherwise perfect | Translation not in effect — B0 silicon running A0 kernels. See the top of this page. Most common failure by a wide margin |
| `FAIL: HOTSWAP=1 but /app/rjprefix not found` | The prefix is not in the container. Re-install it; `docker rm` removes it. **Do not "fix" this with `HOTSWAP=0`** |
| First decode request SIGABRTs, silently | `ATOM_USE_TRITON_MLA=1` not set |
| `available_for_kv` negative, server never starts | Lower `--max-num-batched-tokens` (2048 here). `--cudagraph-mode` does not affect this on the native engine — see [KV budget](#kv-budget) |
| LLVM PHI assertion on long input at concurrency | Triton `gather_kv_b_proj` codegen; set `ATOM_UNFUSED_GATHER_KV_B_PROJ=1` |
| `assert not ca_comm.disabled` kills the ModelRunner while HTTP stays up | `ATOM_USE_CUSTOM_ALL_GATHER` and `AITER_CUSTOM_AR_USE_SYMM_MEM` must be set together |
| MoE GUGU layout error | `ATOM_MOE_GU_ITLV=1` |
| `ATOM_USE_TRITON_MOE_DECODE=1` asserts | K3's activation is `situ`, not SiLU |
| Startup stops at `load RCCL version`, CPU spinning at ~110% | Intermittent `ncclCommInitRank` hang. Retry; allow ≥25 min |
| Multi-node rendezvous hangs with no error | Nodes are not in the same `PPOD_ID` / `VPOD_ID` domain |
| Log appears frozen | tqdm writes `\r`; pipe through `tr '\r' '\n'` |

---

## Not yet measured

- **Throughput, TTFT and TPOT have not been benchmarked.** The configuration
  above is the one that first served correct output at full size; it is not
  tuned, and `--max-num-seqs 8` in particular is conservative.
- Agentic (AgentX) has not been brought up on this configuration.
- Prefix caching is disabled here and untested on this platform.
