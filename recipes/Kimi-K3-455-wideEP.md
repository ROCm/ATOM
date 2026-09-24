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
| **Required ATOM patch** | [PR #2380](https://github.com/ROCm/ATOM/pull/2380) — apply the **diff** to the image's ATOM; stock ATOM aborts on chunked prefill here |
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
fabric` are **not** faults; this firmware does not populate those fields — they
are unpopulated, not measured. To actually measure the link, use ubench07 below.

### After a reboot: load the driver, then confirm the links trained

`amdgpu` is blacklisted on the kernel command line on these hosts, so after a
reboot the driver is simply not loaded and **`/dev/kfd` does not exist**. That
is configuration, not breakage.

```bash
sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1
```

Both parameters are recommended for bring-up. They make the GPU **stop and stay
diagnosable** on a hang instead of being reset out from under you: with the
defaults (`gpu_recovery=-1`, `halt_if_hws_hang=0`) a hardware-scheduler hang
triggers a recovery reset, and what you see afterwards is a run that died for no
visible reason. On a 25-minute cold start with as many silent failure modes as
this platform has, a silently reset GPU is an expensive thing to debug. Check
what is actually loaded:

```bash
cat /sys/module/amdgpu/parameters/gpu_recovery        # want 0 (default -1)
cat /sys/module/amdgpu/parameters/halt_if_hws_hang    # want 1 (default 0)
```

These are load-time parameters — if the driver is already up with the defaults,
they only take effect after an `rmmod` / `modprobe` cycle.

**Then wait for the links to train.** UALink takes tens of seconds after
`modprobe`, during which `accel_state` reads `unconfigured`. That is normal;
`inb-node-agent` finishes the configuration and it flips to `active`.

```bash
sudo cat /sys/class/drm/card*/device/ualink/accel_state              # all: active
sudo cat /sys/class/drm/card*/device/ualink/local_accels             # e.g. 3 2 1 0
sudo cat /sys/class/drm/card*/device/ualink/station_lane_en_bitmap   # non-zero, identical across cards
```

`accel_state` is the gate: **every** entry must read `active` before you try
anything multi-node. The other two corroborate *how* it came up rather than
merely that it did — `local_accels` lists the accelerators this card sees
locally, and `station_lane_en_bitmap` is the per-station enabled-lane mask, so a
partially-trained link shows up as a bitmap that differs from its peers or from
what the same host reported when healthy. Record the healthy values for your
nodes once and diff against them after a reboot; the readings vary with
partitioning, so there is no single correct string to match.

> Once the state reads `active`, **do not `modprobe` again** — that retrains the
> links and costs you the wait for no reason.

Do this before ubench07: an untrained link makes the fabric test fail in a way
that looks like a fabric fault.

### Validate the fabric first — ubench07

Strongly recommended before the first multi-node launch, and the first thing to
re-run when a launch hangs. The expert all-to-all rides this fabric, and **a bad
link hangs rather than errors** — indistinguishable at the server level from the
intermittent `ncclCommInitRank` hang. Ten minutes here saves a 25-minute cold
start that ends in a stuck rendezvous.

`07_ualoe` in the `ubench` suite tests **UALink-over-Ethernet between two
separate OS images** using HIP fabric VMM handles: one node exports a GPU
allocation as a fabric handle, ships it over a socket, the peer imports it and
drives traffic across the fabric. (`06_interconnect_bandwidth` cannot do this —
`hipMemcpyPeerAsync` only sees GPUs in the local process.) It needs ROCm ≥ 7.15
and `amd-smi` ≥ 26.2.1, and is not part of `run_all.sh` because it needs a peer.

```bash
# AMD-internal tarball; ask your AMD contact if the host is not reachable to you
wget http://dcgpuval-storage.amd.com/users/rexyap/MI450/Script/ubench-20260712.tar.gz
tar xzf ubench-20260712.tar.gz && cd ubench-20260712/07_ualoe
./rebuild.sh gfx1250          # -> build/ualoe_p2p.exe, build/ualoe_bw.exe
```

**Correctness first** — single GPU, seconds. Start the exporter first:

```bash
# node A (exporter, owns the memory and waits)
./build/ualoe_p2p.exe export -port=55559 -gpu=0
# node B (importer) -- peer IP is node A's DATA-PLANE address
./build/ualoe_p2p.exe import -peer_ip=<nodeA-data-plane-ip> -port=55559 -gpu=0
```

Both sides must print `RESULT OK: 1/1 pairs PASS`. Omitting `-gpu` uses **all**
local GPUs, pairing GPU *i* on one node with GPU *i* on the other and reporting
the aggregate.

**Then bandwidth.** `ualoe_bw` is symmetric — both sides export and import, so
`bidir` is true full duplex. Start the `listen` side first; the `connect` side
prints the table:

```bash
./build/ualoe_bw.exe listen  -port=55560                          # node A
./build/ualoe_bw.exe connect -peer_ip=<nodeA-ip> -port=55560      # node B
```

Measured on this cluster (4 GPU pairs aggregated, 1 GB transfers):

| Direction | GB/s |
|---|---|
| read | 2029 |
| write | 3192 |
| **bidirectional** | **3986** |

That is the same order as intra-node XGMI, which is what rules out cross-node
bandwidth as a concern for wide EP. **A result near ~4.5 GB/s instead means
MNNVL is not in effect and the traffic silently fell back to TCP** — check
`NCCL_MNNVL_ENABLE=1`. Two orders of magnitude, so this is not a subtle reading.

The suite's README says `ACCEL_STATE` must be `READY`; this firmware reports
`ACTIVE` for the same condition.

#### Three ways this wastes your afternoon

1. **Aggregate mode takes every GPU on both nodes.** On a shared box, confirm
   nobody else is running first.
2. **A dead peer leaves the `listen` side holding device memory and never
   exiting**, and the *next* round then fails silently — no table, no error.
   Clean up between rounds:
   ```bash
   ps -eo pid,args --no-headers \
     | awk '$2 ~ /ualoe_(bw|p2p)\.exe$/ {print $1}' | xargs -r kill -9
   ```
   Do **not** use `pkill -f ualoe`: your own command line contains that string,
   so it kills the shell you are typing in.
3. **Fabric failures land in `dmesg`, not on stdout.** The tool may just sit
   there. Look for `IMPORT: NPA-RSP timeout from remote AccId:<n>` or
   `LSDMA PIO error`. Map the id back with `accel 4*(N-1) .. +3` for node *N* —
   `ACCELERATOR_ID` from `amd-smi fabric` is the same global numbering — to find
   which node and which GPU is at fault.

**Other requirements**

- A **gfx1250 bring-up image** with ATOM, aiter, mori and FlyDSL, built from an
  ATOM that carries [PR #2380](https://github.com/ROCm/ATOM/pull/2380) — apply
  that PR's **diff** to the image's own ATOM rather than replacing it, see
  [Required patch](#-required-patch-atom-pr-2380). Stock ATOM will not serve
  this configuration.
- Passwordless SSH between all four nodes (launch is fanned out over it).
- The full checkpoint on **every** node (~1.42 TiB each), plus swap — weight
  loading peaks well above the 250 GB of host RAM on these boxes.

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

## ⚠️ Required patch: ATOM PR #2380

**This configuration does not run on stock ATOM.** Apply
[ROCm/ATOM#2380 — *feat(mla): unfused torch fallback for gather_kv_b_proj*](https://github.com/ROCm/ATOM/pull/2380)
before serving, and set `ATOM_UNFUSED_GATHER_KV_B_PROJ=1`. Without the patch the
env var does nothing and the server dies on the first long prompt at
concurrency.

### Apply the diff to the ATOM inside the image — do not swap in the PR's ATOM

> **Take the code change, not the branch.** The PR is based on `main`; the
> gfx1250 image carries its own bring-up build of ATOM, which is *not* `main`.
> Checking out the PR branch (or installing ATOM from it) replaces that build
> and silently drops whatever bring-up deltas the image was made with — you
> would be debugging a different engine than the one this recipe was validated
> against. **Apply the PR's diff on top of the image's own ATOM tree.**

Only three files matter at runtime — `atom/model_ops/mla_unfused_gather.py`
(new), `atom/model_ops/attention_mla.py`, `atom/utils/envs.py`. The other three
are tests, which the image does not ship; drop them or the hunks will not apply.

```bash
# on the host
curl -sSL https://github.com/ROCm/ATOM/pull/2380.diff -o /tmp/2380.diff
filterdiff -i 'atom/*' /tmp/2380.diff > /tmp/2380-runtime.diff   # patchutils
sudo docker cp /tmp/2380-runtime.diff k3ep16:/tmp/
```

```bash
# inside the container: patch the ATOM the server actually imports, whatever
# layout it was installed with (editable checkout or site-packages wheel)
ATOM_ROOT=$(python3 -c 'import atom, os; print(os.path.dirname(os.path.dirname(atom.__file__)))')
cd "$ATOM_ROOT"
patch -p1 --dry-run < /tmp/2380-runtime.diff   # read this before the real run
patch -p1           < /tmp/2380-runtime.diff
```

Without `filterdiff`, apply the whole diff and let the `tests/` hunks fail —
just confirm from the output that the three `atom/` files applied cleanly.

Verify, in the container:

```bash
python3 -c "import atom.model_ops.mla_unfused_gather as m; print(m.__file__)"
python3 -c "from atom.utils import envs; print(envs.ATOM_UNFUSED_GATHER_KV_B_PROJ)"
# -> the module path, then False (True once the env var is set)
```

If the second line raises `AttributeError`, `envs.py` did not get patched and
the env var will be ignored at runtime — which is the silent half of this
failure.

Like the rocjitsu prefix, this edits the container's filesystem, so **`docker
rm` discards it** and it must be re-applied on a rebuilt container. Do it on all
four nodes.

**Why it is needed.** On gfx1250 the fused Triton `gather_kv_b_proj` cannot be
compiled for the shapes a *chunked* prefill produces. Triton emits a PHI node
with mismatched operand types and LLVM asserts:

```
llvm/IR/Instructions.h: PHINode::setIncomingValue:
Assertion `getType() == V->getType()' failed.
```

That is a **compile-time abort with no fallback** — the process dies, so a long
prompt at concurrency takes the engine down. FlyDSL is not an alternative here:
it is gfx950-only and aborts in its own compiler if forced. The patch adds a
pure-torch implementation of the same chain (row gather, cache dequant,
`kv_b_proj`, the k_nope/v split, the k_pe concat), which needs no codegen and so
cannot miscompile.

It is off by default and checked after FlyDSL, so it changes nothing on
architectures where the fused kernel compiles. It does not implement the
shuffled-KV layout — hence `ATOM_USE_TRITON_MLA_SHUFFLE_KV=0` below, which the
patch enforces at construction rather than mid-prefill.

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
export ATOM_UNFUSED_GATHER_KV_B_PROJ=1    # requires ATOM PR #2380
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
| LLVM PHI assertion on long input at concurrency | Triton `gather_kv_b_proj` codegen. Needs [PR #2380](https://github.com/ROCm/ATOM/pull/2380) **and** `ATOM_UNFUSED_GATHER_KV_B_PROJ=1` — the env var alone does nothing on stock ATOM |
| `assert not ca_comm.disabled` kills the ModelRunner while HTTP stays up | `ATOM_USE_CUSTOM_ALL_GATHER` and `AITER_CUSTOM_AR_USE_SYMM_MEM` must be set together |
| MoE GUGU layout error | `ATOM_MOE_GU_ITLV=1` |
| `ATOM_USE_TRITON_MOE_DECODE=1` asserts | K3's activation is `situ`, not SiLU |
| `/dev/kfd` missing after a reboot | `amdgpu` is blacklisted on the kernel command line; `modprobe` it — see [After a reboot](#after-a-reboot-load-the-driver-then-confirm-the-links-trained) |
| `accel_state` reads `unconfigured` | Links are still training after `modprobe`. Wait; do not re-`modprobe` |
| A run dies with nothing in the log to explain it | Possibly a GPU recovery reset. Reload the driver with `gpu_recovery=0 halt_if_hws_hang=1` so the next one halts diagnosably |
| Startup stops at `load RCCL version`, CPU spinning at ~110% | Intermittent `ncclCommInitRank` hang. Retry; allow ≥25 min. Looks identical to a bad fabric — rule that out with [ubench07](#validate-the-fabric-first--ubench07) once, then retry |
| Multi-node rendezvous hangs with no error | Nodes are not in the same `PPOD_ID` / `VPOD_ID` domain, or the fabric itself is bad. Confirm with [ubench07](#validate-the-fabric-first--ubench07) before blaming the engine |
| Cross-node bandwidth ~4.5 GB/s instead of thousands | MNNVL not in effect, silently fell back to TCP. Set `NCCL_MNNVL_ENABLE=1` |
| Log appears frozen | tqdm writes `\r`; pipe through `tr '\r' '\n'` |

---

## Throughput

Fixed-length `benchmark_serving`, run from inside the container on the
coordinator. Config exactly as above — `--max-num-seqs 8`, prefix caching off,
`--max-num-batched-tokens 2048`, `max_model_len` unset.

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --backend openai --host <node0-ip> --port 8000 \
  --model /models/Kimi-K3 --tokenizer /models/Kimi-K3 --trust-remote-code \
  --served-model-name moonshotai/Kimi-K3 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --max-concurrency 64 --num-prompts 256 \
  --ignore-eos --percentile-metrics ttft,tpot,itl,e2el
```

> ⚠️ **`--served-model-name` is not optional.** Without it the client puts
> `--model`'s value (a filesystem path) in the request body, the server 404s
> every request, and the benchmark still "completes" — in under a second, with
> every metric printed as `0.00`. The only hint is a `UserWarning: All requests
> failed` buried above the table. `--random-range-ratio 1.0` pins the input
> length; `--ignore-eos` makes every request generate the full output length.

| | 1k/1k, conc 64 | 14k/500, conc 64 | 14k/32, conc 64 |
|---|---|---|---|
| Successful requests | 256 / 256 | 128 / 128 | 64 / 64 |
| Duration (s) | 481.48 | 233.10 | 45.81 |
| Output tok/s | **544.45** | **274.56** | 44.71 |
| Total tok/s | 1088.90 | 8146.84 | **20073.36** |
| Concurrency (actual) | 56.01 | 61.30 | 58.40 |
| TTFT median / p99 (ms) | 2080 / 33217 | 21986 / 46390 | 32158 / 41811 |
| TPOT mean (ms) | 96.94 | 178.22 | 321.75 |
| ITL mean / p99 (ms) | 96.84 / 101 | 177.87 / 176 | 311.69 / 6971 |
| E2EL median (ms) | 101642 | 110666 | 40581 |

⚠️ **These are cluster aggregates, not per-GPU.** `benchmark_serving` divides by
duration only (`sum(output_lens) / dur_s`), with no notion of device count.
Across 64 GPUs the 1k/1k figure is 8.51 output tok/s per GPU.

Reading these:

- **TPOT 96.94 ms is ~10.3 tok/s per stream, which is what a single stream gets
  on its own** (a warm-up request of 1024 tokens took 92.8 s). Concurrency 64
  costs almost nothing per stream, i.e. decode is nowhere near saturated.
  `--max-num-seqs 8` caps each DP rank at 8 sequences — 128 slots across dp16,
  and a decode batch of at most 8. There is headroom here that this recipe does
  not reach.
- **The 14k total of 8146 tok/s is not 7.5x better decode**; it counts input
  tokens, and that run is 1.84M input against 64k output. Compare
  `Output tok/s`: 544 → 275, i.e. decode roughly halves at long context.
- **14k TTFT is ~10x the 1k figure** because `--max-num-batched-tokens 2048`
  splits a 14336-token prompt into 7 chunked-prefill steps. That is the cost of
  the conservative setting, not a fault.

⚠️ ATOM's own `tput_per_gpu` divides by `tp × pp × pcp` — **DP is not counted**.
With TP=1 the denominator is 1 and any per-GPU figure is 16x too high here.
The table above is aggregate and unaffected.

---

## Profiling

The server writes torch profiler traces only if it was started with a profiler
directory; the client flag alone does nothing.

**Server** — ⚠️ **the CLI flag only. The environment variable does not work.**

```bash
--torch-profiler-dir /data/profile_run        # this one
# ATOM_TORCH_PROFILER_DIR=/data/profile_run   # does NOT work, see below
```

`envs.py` defines `ATOM_TORCH_PROFILER_DIR` and `config.py` appears to take it
as the default for `torch_profiler_dir`, but something downstream overrides it:
`atom/plugin/config.py` hardcodes `torch_profiler_dir=None` in three places.
Measured, on this image:

| | `Engine kwargs` | traces written |
|---|---|---|
| `ATOM_TORCH_PROFILER_DIR=<dir>` | `'torch_profiler_dir': None` | **none** |
| `--torch-profiler-dir <dir>` | `'torch_profiler_dir': '<dir>'` | yes |

**Check the log before spending a run on it:**

```bash
grep -o "'torch_profiler_dir': [^,]*" <server log> | head -1
```

`model_runner.py` appends the rank name to that path, so each rank writes its
own subdirectory — `dp0_tp0/`, `dp1_tp0/`, … one `.pt.trace.json.gz` each.

**Client** — add `--profile` to `benchmark_serving`. It POSTs
`/start_profile` before the run and `/stop_profile` after
(routes in `atom/entrypoints/openai/api_server.py`):

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --backend openai --host <node0-ip> --port 8000 \
  --model /models/Kimi-K3 --tokenizer /models/Kimi-K3 --trust-remote-code \
  --served-model-name moonshotai/Kimi-K3 \
  --dataset-name random --random-input-len 14336 --random-output-len 32 \
  --random-range-ratio 1.0 --max-concurrency 64 --num-prompts 64 \
  --ignore-eos --profile
```

> ⚠️ **A server without a working profiler directory is a silent no-op, and it
> is quieter than it sounds.** `POST /start_profile` still answers
> `HTTP 200 {"message":"Profiling started"}`, the client still prints
> `Starting profiler...` / `Stopping profiler...`, and the benchmark still
> reports a full set of metrics. The only symptom is an empty output directory.
> Point the directory at a bind-mounted path (`/data` here) so traces land on
> the host, and keep `--num-prompts` to a single concurrency wave — 14k/32 at
> concurrency 64 produced 162 MB per node, ~41 MB per rank.

Verified output, 16 ranks over 4 nodes:

```
profile_run/dp0_tp0/Kimi-K3_ts_20260924_100111_980.pt.trace.json.gz   41 MB
```

2.5M trace events, loadable in Perfetto. Top GPU time on a 14k-prefill-heavy
run looks like this — worth knowing before tuning the wrong thing:

| GPU time (us) | kernel |
|---|---|
| 11,160,833 | `_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_256_N_256_K_64` |
| 10,078,004 | `mori_ep_dispatch_tdm_fp4x2_ws16_h1792_k16_64x16` |
| 4,785,535 | `ep_combine_fused_sync_0` |
| 2,714,530 | `a8w4_tdm_fp4_t64x256x256_w1x4_b3_K3584_e56_act3_q1r4` |
| 2,290,119 | `a8w4_tdm_fp4_t64x256x256_w1x4_b3_K3072_e56_epscatter` |

**Expert all-to-all is the second-largest cost and close to the first**:
dispatch + combine together are 14.9M us against 11.2M for the biggest GEMM.
And that GEMM names itself `bandwidth_bound`.

---

## Agentic (AgentX)

Trace replay of real Claude Code sessions, not synthetic fixed lengths:
ISL median ~104k tokens, OSL median ~339, theoretical prefix cache hit 97.34%.

⚠️ **This needs `--enable_prefix_caching` on the server** — the opposite of the
launch above. Without it every turn re-prefills the whole history and the
numbers describe repeated prefill, not the engine.

```
aiperf          0.12.0  (SemiAnalysis fork, NOT PyPI upstream)
submodule       utils/aiperf @ 754356e9a39acc6cc6afb242d123bb57c3fb6f75
                git describe: agentx-v1.0.2-3-g754356e9
Python          3.11  (aiperf dropped 3.10)
dataset         semianalysisai/cc-traces-weka-062126 — 393 traces,
                traces.jsonl 1.85 GB, public (no HF_TOKEN needed)
```

```bash
git clone --recurse-submodules https://github.com/SemiAnalysisAI/InferenceX.git
git -C InferenceX/utils/aiperf checkout 754356e9a39acc6cc6afb242d123bb57c3fb6f75
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
uv venv --python 3.11 "$AIPERF_RUNTIME_DIR/venv"
uv pip install --python "$AIPERF_RUNTIME_DIR/venv/bin/python" \
    -r InferenceX/utils/agentic-benchmark/requirements.txt -e InferenceX/utils/aiperf
hf download --repo-type dataset semianalysisai/cc-traces-weka-062126
```

```bash
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://<node0-ip>:8000 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model moonshotai/Kimi-K3 \
  --tokenizer moonshotai/Kimi-K3 --tokenizer-trust-remote-code \
  --apply-chat-template \
  --concurrency 16 --benchmark-duration 1800 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --trace-idle-gap-cap-seconds 300 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir <artifacts>
```

Two things that bite before the first request:

- **`--tokenizer` must be the HF repo id, not the local model path.** The
  wire name passed to `--model` is not a repo id, and the model directory may
  not be readable by the account running aiperf. `moonshotai/Kimi-K3` is public
  and only the tokenizer files are fetched.
- **`tokenizer.chat_template` is `None` for K3, and that is fine.**
  `tokenization_kimi.py` implements `apply_chat_template()` as a method rather
  than shipping a Jinja template string, so `--apply-chat-template` works even
  though the attribute reads empty. Verify before a 30-minute run:
  ```bash
  python -c "from transformers import AutoTokenizer as A; \
    t=A.from_pretrained('moonshotai/Kimi-K3',trust_remote_code=True); \
    print(t.apply_chat_template([{'role':'user','content':'hi'}],tokenize=False))"
  ```

The scenario enforces `--benchmark-duration >= 900`; shorter needs
`--unsafe-override` and marks the result `submission_valid=false`.
Effective concurrency is far below `--concurrency` because lanes spend much of
the trace idle — read the Effective figures, not the nominal ones.

---

## Not yet measured

- Agentic (AgentX) numbers on this configuration — the setup above is
  validated, the run is not yet reported here.
- Prefix caching is disabled in the launch above and untested on this platform.
- `--max-num-seqs` has not been swept; see the note under
  [Throughput](#throughput) for why it is the first thing to try.
