# Kimi-K3 Wide-EP on gfx1250 (MI455) — 4 nodes, dp16ep16

Full-size **Kimi-K3** (2.78T, 93 layers: 24 full-attention MLA + 69 KDA linear,
896 routed experts top-16, MXFP4 routed experts / BF16 everything else) served
across **four gfx1250 nodes, 4 GPUs each, in SPX** — 16 ranks total, `-tp 1`
with data parallel 16 and expert parallel 16. The checkpoint is ~1.42 TiB and is
needed **in full on every node**; it is not sharded across the cluster.

## A0 and B0 silicon — same image, one difference

Both use the **same** bring-up image, `rocm/fw-bringup:gfx1250-atom-20260918-ep8`,
and both need [ATOM PR #2380](#-required-patch-atom-pr-2380). The image ships
**B0** code objects. The only difference is the silicon you run them on:

| silicon | code-object translation | status |
|---|---|---|
| **B0** | **none** — skip the rocjitsu prefix entirely | **the configuration this page is about** |
| A0 | **mandatory** — B0→A0, or every token is `!` and GSM8K is 0% | legacy path, see [Appendix B](#appendix-b-running-the-same-image-on-a0-silicon) |

The translation library is named `librocjitsu_gfx1250_**b0_to_a0**.so` for
exactly this reason: it rewrites the image's B0 objects down to A0. On B0
silicon there is nothing to rewrite.

**Everything below is the B0 path** unless a section says otherwise. Verified
end to end: the hook attached on all 16 ranks and performed **zero**
translations, and GSM8K still scored **0.9621** over the full 1319.

### Where to go

| you want | go to |
|---|---|
| bring the server up | [Setup](#container) → [Launch](#launch) → [Accuracy gate](#accuracy-gate--run-this-before-any-benchmark) |
| the exact validated commands and env | **[Appendix A](#appendix-a-the-exact-configuration-this-was-validated-on)** |
| agentic numbers and how they were taken | [Agentic (AgentX)](#agentic-agentx) → [AgentX result](#agentx-result) |
| why it is slow and what to try | [Where the time goes](#where-the-time-goes-and-what-to-try-next) |
| take a profile | [Profiling](#profiling) |
| try the optimization PR stack — **pending verification** | [Optimized source stack](#optimized-source-stack--pending-verification) |
| you are on A0 silicon | [Appendix B](#appendix-b-running-the-same-image-on-a0-silicon) |
| find a tray's IP or BMC | [The rack: hosts and addresses](#the-rack-hosts-and-addresses) |
| reboot a tray, or bring one back after any reboot | [Rebooting a tray](#rebooting-a-tray) → [perf_setup](#after-a-reboot-load-the-driver-then-confirm-the-links-trained) |
| AC-cycle a tray with helicop (the standard full reboot) | [With helicop](#with-helicop-ac-cycle-then-power-on) |

### Measured on B0, `rocm/fw-bringup:gfx1250-atom-20260918-ep8`

4 nodes × 4 gfx1250, **432 GiB/GPU**, dp16ep16, no translation.

| | |
|---|---|
| **Agentic (AgentX), interactivity p90** | **2.53 tok/s/user** |
| **Agentic, total throughput per GPU** | **182 tok/s** (2913 aggregate / 16) |
| **Agentic, prefix cache read** | **92.36%** (94.68% theoretical) |
| **GSM8K**, 5-shot, full 1319 | **strict-match 0.9621 ±0.0053** — 9 min at `num_concurrent=32` |
| Fixed-length 14k/16, con32 | 26,207 tok/s total, TTFT median 13.2 s, TPOT median 73.9 ms |
| Cold start | **under 6 min** |
| Bottleneck | **decode** — see [Where the time goes](#where-the-time-goes-and-what-to-try-next) |

### The rest of the configuration

| | |
|---|---|
| Hardware | 4 × gfx1250 (MI455), 4 GPUs/node, SPX. **Check HBM per GPU** — 288 GiB and 432 GiB boards both exist; every KV number below depends on which you have |
| Parallelism | `-tp 1 --data-parallel-size 16 --data-parallel-size-local 4`, EP16, DP attention |
| Attention | Triton MLA (`ATOM_USE_TRITON_MLA=1`), unshuffled KV |
| Fabric | UALink within a node-set sharing one `PPOD_ID` + `VPOD_ID`; RCCL over the data-plane NIC |
| **Required ATOM patch** | [PR #2380](https://github.com/ROCm/ATOM/pull/2380) — apply the **diff** to the image's ATOM; stock ATOM aborts on chunked prefill here |
| GSM8K on the original A0 run | strict-match 0.9545 ±0.0057, flexible-extract 0.9538 ±0.0058 |

---

## The rack: hosts and addresses

This page was written on one 18-tray HeliosM C-rack (`HELIOSM-DVT-CT1` …
`CT18`, 4 × gfx1250 per tray). You reach the trays from a jump host.

| | |
|---|---|
| Jump host | `nj-4050-genamd-console`, `10.223.239.5`. Not a GPU node. `/home` is NFS-shared with every tray |
| Tray login | `root@<OS IP>`. CT1, CT2, CT3 and CT18 do not accept the shared SSH key and need the root password. Ask the rack owner; it is not written here |
| Data plane | `enp1s0f1` on every tray. Its address **is** the SSH address, and it is the address to give ubench07 and RCCL |
| Fabric | every tray checked reports the same `PPOD_ID` and `VPOD_ID 2` |
| UALink switch management | `10.210.11.62` … `.73`: 12 × `Helios_switch` (DS6010, Redfish) |

| Tray | OS / data-plane IP | BMC IP | BMC address from | `accel_id` |
|---|---|---|---|---|
| CT1 | 10.210.11.19 | 10.210.11.44 | in-band | 0–3 |
| CT2 | 10.210.11.24 | 10.210.11.47 | in-band | 4–7 |
| CT3 | 10.210.11.16 | 10.210.11.48 | in-band | 8–11 |
| CT4 | 10.210.11.11 | 10.210.11.49 | login sheet | 12–15 |
| CT5 | 10.210.11.17 | 10.210.11.50 | in-band | 16–19 |
| CT6 | 10.210.11.15 | 10.210.11.51 | inferred | 20–23 |
| CT7 | 10.210.11.23 | 10.210.11.53 | login sheet | 24–27 |
| CT8 | 10.210.11.21 | 10.210.11.55 | login sheet | 28–31 |
| CT9 | 10.210.11.27 | 10.210.11.57 | login sheet | 32–35 |
| CT10 | 10.210.11.80 | 10.210.11.60 | login sheet, unverified | 36–39 |
| CT11 | 10.210.11.26 | 10.210.11.61 | in-band | 40–43 |
| CT12 | 10.210.11.22 | 10.210.11.59 | in-band | 44–47 |
| CT13 | 10.210.11.14 | 10.210.11.58 | login sheet | 48–51 |
| CT14 | 10.210.11.25 | 10.210.11.56 | in-band | 52–55 |
| CT15 | 10.210.11.13 | 10.210.11.54 | in-band | 56–59 |
| CT16 | 10.210.11.12 | 10.210.11.52 | login sheet | 60–63 |
| CT17 | 10.210.11.18 | 10.210.11.46 | login sheet | 64–67 |
| CT18 | 10.210.11.10 | 10.210.11.45 | in-band | 68–71 |

- `accel_id` is 4 × (tray − 1) … +3. It is the same global numbering that
  `amd-smi fabric` and the fabric's dmesg lines (`AccId:<n>`) use.
- BMC addresses are DHCP leases, so re-derive one before resetting a tray (see
  [Rebooting a tray](#rebooting-a-tray)). The "from" column means:
  - *in-band*: read on the tray with `ipmitool lan print 1`, or matched by host-NIC MAC, on 2026-10-08/09.
  - *login sheet*: the rack's login sheet; not re-checked.
  - *inferred*: the only BMC left unassigned.
  - CT10 and its listed BMC were both down when this was written.
- CT10's OS address is `.80`. The `.60` that some lists give for its OS is its BMC.
- helicop's rack file (`rack/MissionM.rack`, see [With helicop](#with-helicop-ac-cycle-then-power-on))
  lists the same OS and BMC addresses for all 18 trays.
- Tray ownership changes daily. Check [Is the rack actually free?](#is-the-rack-actually-free)
  before using or rebooting any tray.

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

**Load it through `perf_setup`, not with a bare `modprobe`.**
`perf_setup_BPC21.sh` is the platform's perf-setup script (AMD-internal; ask
your AMD contact). It loads `amdgpu` itself and applies tuning that **every
reboot discards**, including reboots nobody asked for (BMC watchdog, crash).
Part of that tuning only works while `amdgpu` is still unloaded, so the order
is fixed: reboot → `perf_setup` → anything else.

| when | what `perf_setup_BPC21.sh -enable-csc` does |
|---|---|
| before the driver | enables the CSC feature; reverts the unvalidated PPT boost settings |
| driver load | menu **D**: `modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1 mtype_local=0 noretry=1`. That is MTYPE=RW and XNACK off, plus the two bring-up parameters explained below |
| after the driver | SOC PCC off, KLL chicken bits, FCLK BW DPM, `GFX_ICG_TCP_CTRL2=0x2` on every XCD, CPU governor `performance`, NUMA balancing off, **−82 mV** voltage offset, CSC DVO **25 mV** |

**Why it is not optional.** Two data points from this rack:
- Same image, same workload (DSR1 offline, one node): a tuned tray measured
  14,713 tok/s/GPU and an untuned one 13,030. That is about 13%. They were two
  different trays, so chip-to-chip variance is in that number too.
- The tuning goes missing silently. One sweep of this rack found 8 of 9 trays had
  been rebooted since their last `perf_setup`.

⚠️ **If `amdgpu` is already loaded, `perf_setup` cannot fix it.** Its `modprobe`
is then a no-op: MTYPE and XNACK stay at the defaults, and the pre-driver steps
land too late. The report still says `D — MTYPE=RW, XNACK=disabled`. The only
fix is another reboot. So do not "just modprobe" first.

⚠️ **K3 note.** The K3 runs on this page document the bare `modprobe` shown at the
end of this subsection, and do not record any `perf_setup` state. `perf_setup`
changes MTYPE and XNACK, so run the
[accuracy gate](#accuracy-gate--run-this-before-any-benchmark) after switching.

**Running it.** The script reads its prompts from `/dev/tty`, so piping answers
in does not work. Drive it with `expect`, and match each prompt **exactly**: a
loose regex once answered the DVO prompt with `-82`.

```bash
# as root on the tray, from /root (the script writes its downloads and perf_env.sh to $PWD)
lsmod | grep -c '^amdgpu '     # must print 0 -- otherwise reboot first
lsmod | grep '^ifoe '          # must be loaded (see the ordering note below)
cat > /root/run_perf_setup.exp <<'EOF'
#!/usr/bin/expect -f
set timeout 3600
log_file -a /root/perf_setup_expect.log
cd /root
spawn ./perf_setup_BPC21.sh -enable-csc
expect {
    -ex {Choose [A-H]: }                                                { send -- "D\r";   exp_continue }
    -ex {Voltage offset in mV [default -82, or 'skip' to not set it]: } { send -- "-82\r"; exp_continue }
    -ex {CSC DVO offset in mV [default 25, or 'skip' to not set it]: }  { send -- "25\r";  exp_continue }
    timeout { exit 2 }
    eof
}
EOF
chmod +x /root/perf_setup_BPC21.sh
setsid -f nohup expect -f /root/run_perf_setup.exp > /root/perf_setup_run.out 2>&1 < /dev/null
```

It needs to reach AMD-internal hosts (it downloads its tuning scripts and tools)
and takes 2–7 minutes. Over `ssh`, `pgrep -f perf_setup` also matches your own
remote shell, so check progress by PID or with a `[p]erf_setup` pattern.

**Verifying it.**

```bash
tail -32 $(ls -1t /opt/amd-apps/perf_setup_BPC21_report_*.log | head -1)   # Perf steps: 8 passed, 0 failed
for p in mtype_local noretry gpu_recovery halt_if_hws_hang; do
  echo "$p=$(cat /sys/module/amdgpu/parameters/$p)"; done                    # 0 1 0 1
```

- `Result: [PARTIAL]` with 8/8 perf steps passed is fine **when the only failure
  is the TBP read**. That step SSHes to the tray's BMC with a default login, and
  several trays reject it (CT1, CT2, CT5, CT12, CT14).
- The `Could not read BKC version` and `restricted shell` warnings are normal.

A tray is tuned **for this boot** only if both of these hold:
- the newest `/opt/amd-apps/perf_setup_BPC21_report_*.log` is newer than `uptime -s`;
- `mtype_local=0` and `noretry=1`.

A driver loaded by hand with those two parameters passes the second check but
not the first, and it is missing all the other tuning.

**Without the script**, this loads the driver with the bring-up parameters but
none of the tuning:

```bash
lsmod | grep '^ifoe '                                    # must already be loaded
sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1
```

⚠️ **`accel_state` can stick at `unconfigured` permanently, and it looks
exactly like "still training".** UALink here is UALink-over-Ethernet through
Pensando AINIC parts (`lspci -d 1dd8:` → 56 functions, OXRP bridges at
`lspci -d 1022:1746` → 8), carried by the `ifoe` stack.

Observed on one boot: all four GPUs sat at `unconfigured` for **12+ hours**
with `ppod_id` all-zero, `vpod_id=0`, `accel_id=4294967295`,
`link_type=invalid`. Waiting does not fix it. On a later boot, with `ifoe`
already loaded and `amdgpu` not yet, a single `modprobe` brought all four to
`active` in **under 15 s**, and the same held across all four nodes.

The ordering above is the working hypothesis for the difference — `ifoe` was
loaded late (about an hour into the boot) on the run that got stuck — but it
has **not** been confirmed by a controlled test, so treat it as a checklist
item rather than a proven mechanism:

- if `accel_state` is `unconfigured` more than a minute after `modprobe`,
  confirm `ifoe` is loaded and do an `rmmod amdgpu` / `modprobe amdgpu` cycle;
- once it reads `active`, do **not** `modprobe` again (see below).

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

**A later run reads much higher.** On 2026-10-09 the binaries were built with the
trays' own `hipcc` (HIP 7.16, ROCm 10.1), every tray had just had `perf_setup`,
and all four GPU pairs moved 1 GB transfers. Seven pairs were measured:
CT1↔CT5, CT5↔CT12, CT12↔CT1, CT3↔CT11, CT11↔CT15, CT15↔CT3 and CT2↔CT3.

| Direction | GB/s, 4-pair aggregate | per GPU |
|---|---|---|
| read | 5,772–5,797 | ~1.45 TB/s |
| write | 6,219–6,228 | ~1.56 TB/s |
| **bidirectional** | **10,942–10,981** | ~2.74 TB/s |

That is 80–86% of the 1.8 TB/s uplink one way, and 76% of 3.6 TB/s
bidirectionally. The spread across pairs was under 0.5%. To judge one pair,
compare it against a known-good pair measured on the same day, not against
either table.

**Test every tray against two peers.** Use triangles: A↔B, B↔C, C↔A. A bad tray
then fails both of its pairs, while a bad peer fails only one. A pair takes
seconds. Pairs that share a tray must run one after the other.

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

   Two more signatures from the same family, seen on a rack where
   `accel_state` read `active` on **every** node:

   ```
   amdgpu 0001:01:00.0: LSDMA PIO error status bits 0x8000
   amdgpu 0001:01:00.0: LSDMA PIO failed to copy memory!
   amdgpu 0001:01:00.0: HELLO: Send failed to remote AccId:51
   amdgpu 0001:01:00.0: IMPORT: connection setup failed with remote AccId:51
   amdgpu 0001:01:00.0: IMPORT: XA import failed for handle:<hex>:<hex>
   ```

   and, from RCCL, `src/transport/p2p_tmp.cc:372 NCCL WARN HIP failure
   'invalid argument'` → `NCCL error: unhandled cuda error`.

   **Read dmesg on both nodes.** The importer logs all of this; the exporter
   logs *nothing*, because the HELLO never arrived. An empty exporter-side
   dmesg next to a failing importer is the clean signature that packets are not
   crossing the wire.

   `ualoe_p2p` surfaces it as
   `[FATAL] ualoe_p2p.cpp:49 hipMemSetAccess(...) -> invalid argument`.
   Line 49 is inside `fab_import`, **not** `fab_alloc` — the remote handle
   imported and mapped, and only making it accessible failed. It is not a local
   allocation problem.

> **`accel_state=active` is necessary, not sufficient.** It reports link
> training, not reachability. A whole rack can read `active` on every node while
> no pair exchanges a single packet. Run `ualoe_p2p` before trusting it.

#### Single-tray TDM gate (epcheck)

ubench07 needs a peer. To check one tray on its own, use the `epcheck` image. It
compiles three TDM/fabric micro-benchmarks, compares them against recorded
baselines (±5%), and exits non-zero on any miss. It takes about 30 s.

```bash
# private image, ~373 GB unpacked: check free space on the docker root first.
# Registry credentials come from your AMD contact.
docker run --rm --device /dev/kfd --device /dev/dri --ipc=host --network host --privileged \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
  --group-add video --shm-size 64g \
  --entrypoint /opt/epcheck/epcheck.sh rocm/aigmodels-private:epcheck-gfx1250-20260806
echo "exit=$?"
```

The line to read is `epsim mode0 grid=64` (MODE0, the per-warp TDM throughput
gate; band 1467.8–1690.5).

On 2026-10-09 six tuned trays (CT1, CT2, CT3, CT5, CT11, CT12) were checked:
- **all passed MODE0**, at 1,584–1,623;
- every one of them failed the same two items: `epsim mode9 grid=64 READ` at
  757–770 against a floor of 835, and warp density at 0.87–0.92 against 0.95.

So each run ends in `EPCHECK FAIL`. Identical misses on every tray point to a
systematic difference from the reference node, not to six bad trays.

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

## Is the rack actually free?

⚠️ **VRAM is not a free/busy signal.** A job whose containers exist but have
not yet loaded weights reads **0% on every GPU**. A pre-flight that only looks
at `rocm-smi` will call the rack idle, and you will start on top of someone
else. The collision surfaces minutes later, on your side, as a KV budget that
has gone negative:

```
RuntimeError: Per-request cache tensor (3.35GB for 8 slots) exceeds available
KV budget (-17.63GB) at --gpu-memory-utilization 0.94. Set
--gpu-memory-utilization >= 0.99 ...
```

That is the engine working correctly — it saw ~30% of each card already taken
and refused to start. Read it as *someone else's weights are loading*, not as
something to fix by raising `--gpu-memory-utilization`.

Check the container list and the event log, not the GPU counters:

```bash
for n in <nodes>; do
  printf '%-22s ' "$n"
  ssh -q "$n" "docker ps --format '{{.Names}}' | tr '\n' ' '"      # ALL of them
  ssh -q "$n" "docker events --since 30m --until 0s --filter event=create \
                 --format '{{.Actor.Attributes.name}}' | tr '\n' ' '"
  echo
done
```

`docker events` is the one that catches it: a `create` shows up there well
before any GPU counter moves. Filtering `docker ps` by your own container name
is the specific mistake to avoid — it reports "nothing of mine is running",
which is not the same question.

## Rebooting a tray

Reboot only a tray you have confirmed is free (see above). Every container on it,
yours and everyone else's, stops (`Exited (255)`) and does not come back on its
own. After any reboot, run [perf_setup](#after-a-reboot-load-the-driver-then-confirm-the-links-trained)
before anything else.

**When you have to.** With `gpu_recovery=0` a wedged GPU does not recover by
itself; that is the point of the parameter. The signature:
- `amdgpu …: MES(0, 0) failed to respond to msg=REMOVE_QUEUE` in dmesg;
- engine processes `<defunct>`;
- `/sys/class/kfd/kfd/proc` never drains;
- kworkers stuck in `D` state.

You usually get there by killing a hung engine, or when a multi-node job aborts
and kills its engines. After an abort, check **every** participant. A tray whose
engines are zombies but whose kfd list is empty has freed its GPUs and needs
nothing.

The other case is a host whose userland hangs: SSH authenticates (`ssh -v`
shows `Entering interactive session`), but no command ever runs.

**Record `boot_id` first** (`cat /proc/sys/kernel/random/boot_id`). A changed
`boot_id` is the only reliable proof that the tray rebooted.

| tray state | how | round trip measured here |
|---|---|---|
| OS healthy, nothing in `D` state | `systemctl reboot --no-wall` | ~3 min |
| GPU wedged, `D`-state workers, or userland hung | BMC Redfish `ForceRestart` | ~2.5 min |
| `ForceRestart` did not bring the GPUs back | Redfish `PowerCycle` | not tried here |
| a cold boot, or nothing above brought the tray back | helicop: AC-cycle the whole tray, BMC included, then power on. See [With helicop](#with-helicop-ac-cycle-then-power-on) | 7½–9 min to SSH |

Do not use `systemctl reboot` or Redfish `GracefulRestart` on a wedged GPU:
shutdown blocks while unloading `amdgpu`.

**The BMC** runs OpenBMC with Redfish and is reachable from the jump host.
Addresses are in [The rack](#the-rack-hosts-and-addresses). Credentials come from
the rack owner and are not written here. SSH to the BMC's own shell is
`admin@<bmc-ip>` on port **2200**; port 22 on the BMC address is proxied to the
host OS. Before resetting, make sure the BMC belongs to the tray you mean. The
Redfish `UUID` cannot tell you: it is an unfilled `$BOARD_UUID` placeholder.

```bash
B=<bmc-ip>
# 1. on the tray: its own BMC address and MAC (in-band, via /dev/ipmi0)
ipmitool lan print 1 | grep -E '^(IP Address|MAC Address) +:'
# 2. from the jump host: the BMC must report the same eth0 MAC ...
curl -sk -u "$BMC_CRED" https://$B/redfish/v1/Managers/bmc/EthernetInterfaces/eth0 | grep -o '"MACAddress": "[^"]*"'
#    ... or, if the tray has no usable shell, compare its data-plane MAC
#    (from a peer tray on the same subnet: ping <tray-ip>; ip neigh show <tray-ip>)
curl -sk -u "$BMC_CRED" https://$B/redfish/v1/Systems/system/EthernetInterfaces/0 | grep -o '"MACAddress": "[^"]*"'
# 3. reset; expect HTTP 200 and Base.1.13.0.Success
curl -sk -u "$BMC_CRED" -X POST -H 'Content-Type: application/json' \
  -d '{"ResetType":"ForceRestart"}' \
  https://$B/redfish/v1/Systems/system/Actions/ComputerSystem.Reset
```

- Allowed `ResetType`s: `On`, `ForceOff`, `ForceOn`, `ForceRestart`,
  `GracefulRestart`, `GracefulShutdown`, `PowerCycle`, `Nmi`.
- The jump host is routed to the trays and has no ARP entries for them, so take
  a tray's MAC from a peer tray.

**Waiting for it.**

```bash
OLD=<boot_id before>
until b=$(ssh -o ConnectTimeout=10 root@<tray-ip> cat /proc/sys/kernel/random/boot_id 2>/dev/null) \
      && [ -n "$b" ] && [ "$b" != "$OLD" ]; do sleep 10; done
```

If nothing answers after about 10 minutes, read the BMC's event log over IPMI:
`ipmitool -I lanplus -H <bmc-ip> -U … -P … sel elist last 20`.

- Once, a `ForceRestart` was followed about 2 minutes later by `Watchdog2 … Hard
  reset`: the watchdog the OS had armed fired while the tray was still booting,
  and reset it a second time.
- If the BMC does not answer either, the tray has probably lost power. That needs
  someone at the rack.

**Reading the boot logs.**

| seen at boot | meaning |
|---|---|
| `mce: … CPU 0/48: Machine Check: 0 Bank 62: a0002000003d0800`, plus a BERT "previous boot" record for MSR `0xc00023e1` | noise. It appears after **any** warm reboot, healthy trays included. It was absent after cold boots: a rack-wide power-on, and a helicop AC cycle (6 of 6 trays) |
| `mana … Failed to query link config: -71` | noise; SSH runs over that NIC |
| BERT `event severity: fatal`, `fru_text: Perr: CPU0`, bank 19 status `baa000000005080b` (the SEL's OEM records decode to `Perr: CPU0`) | **a real host crash**. Seen on CT1, CT2, CT3, CT12 and CT15. CT2 and CT12 were under load. CT15 crashed twice within 30 minutes, and later once while idle, about 4 minutes after `perf_setup` finished on a fresh boot from an AC cycle. Keep a tray that repeats it out of long multi-node runs |

### With helicop: AC cycle, then power on

`helicop` is the Helios bring-up team's controller for MI455X boards. It is
AMD-internal; ask your AMD contact for it. It is a Bash tool that drives a tray's
host and BMC over SSH. Its reboot is the standard one on these racks: **an AC
cycle of the whole tray, BMC included, then a power-on.** That is a cold boot.
It is also the method to reach for when a `ForceRestart` did not bring a tray
back. Validated with helicop v0.39 on CT1, CT2, CT3, CT5, CT11 and CT15
(2026-10-09).

**Setup, once.**
- helicop lives on an internal GitLab that the jump host cannot reach, so copy
  the repo in as an archive.
- Its `rack/MissionM.rack` describes this rack, one line per tray:
  `CTnn root:<pw>@<OS IP> admin:<pw>@<BMC IP>:2200`. The passwords are in plain
  text, so run `chmod -R go-rwx` on your copy.
- The file has a `jump` line. Comment it out when running on the jump host
  itself, which reaches the trays directly.
- helicop pipes privileged BMC commands through `sudo -S` itself.
- Copy the lines for the trays you need into `hosts.cfg` in the helicop
  directory, then `chmod 600` it. The first field becomes an alias, so passwords
  stay off the command line and out of `ps`: `./helicop CT03`.
- Two settings keep helicop from stalling on this rack. Keep both outside the
  helicop tree:

```bash
H=<helicop dir>; mkdir -p ~/hc/bin
# 1. the power-stage polls cannot succeed here (see below): 5 tries instead of 250, 3 s apart
sed -E 's/^(power_stage_poll_attempts *= *).*/\15/' $H/settings.cfg > ~/hc/settings.cfg
# 2. keep-alives and a connect timeout. Without them, helicop's persistent SSH connection
#    to a host that just lost power blocks its next command for minutes
cat > ~/hc/ssh_config <<'EOF'
Host *
    ConnectTimeout 10
    ServerAliveInterval 5
    ServerAliveCountMax 3
    UserKnownHostsFile /dev/null
    LogLevel ERROR
Include /etc/ssh/ssh_config
EOF
printf '#!/bin/sh\nexec /usr/bin/ssh -F %s "$@"\n' ~/hc/ssh_config > ~/hc/bin/ssh && chmod +x ~/hc/bin/ssh
export PATH=~/hc/bin:$PATH HELICOP_SETTINGS_FILE=~/hc/settings.cfg
```

**The sequence.** It takes two helicop sessions per tray, because the AC cycle
takes the BMC down with the host.

| | do | what happens | took here |
|---|---|---|---|
| 0 | check the tray is [free](#is-the-rack-actually-free); record `boot_id`; `sync` | | |
| 1 | `./helicop CT03` → `ac_cycle` → type `confirm` → `quit` | The BMC runs `gpioset gpiochip3 20=1` (line `VIRTUAL_RESEAT`). helicop prints `AC cycle issued — BMC connection lost after 1s` and returns straight to the menu | seconds |
| 2 | wait | The BMC comes back on the same address (6 of 6 trays) and its uptime restarts. Then wait **2 more minutes** (see the pitfalls) | 2¾–3 min |
| 3 | `./helicop CT03` → `prep_machine auto upto 13` → `quit` | Step 10 powers the host on. Step 13 waits for host SSH | 4½–6 min |
| 4 | check that `boot_id` changed, then run [perf_setup](#after-a-reboot-load-the-driver-then-confirm-the-links-trained) | | 2–7 min |

From the AC cycle to host SSH took 7½–9 min.

- **`upto 13` is required.** Step 19 (`CheckAmdgpuLoad`) has a fix that loads
  `amdgpu`. After that, `perf_setup` can no longer apply its pre-driver tuning.
- **Step 10 needs an `f` from you.** `auto` pauses at the first FAIL that has no
  fix (step 06 here) and stays paused.
  - Answer `s` to the FAILs that have no fix.
  - Answer `f` at `Suggested fix: PerformDcPowerOn`. That sends `ipmitool power on`.
  - The restore policy is `always-on`, so the host also comes up without the
    `f`, but later: about 3½ min after the BMC returns instead of just over 2.

**Expected FAILs on this rack.** On this BMC firmware (OpenBMC 26.8.4,
`m1120-wcs`), helicop cannot read the CPLD power-stage register.
- `i2cget` is not on the `admin` account's `PATH`.
- Called by its full path, it fails to read address `0x20` on bus 4 and on bus 8.

So every step that polls the stage fails, although the power commands
themselves work. Check the power state yourself instead:
- the power state: Redfish `PowerState` (`/redfish/v1/Systems/system`), or
  `ipmitool power status` on the BMC;
- the reboot itself: `boot_id`.

| step | shows | answer |
|---|---|---|
| Detect EVT revision (`ac_cycle` [02], `prep_machine` [03]) | WARN `unrecognised revision byte 0xEMPTY` | — |
| [05] Check PLDM version | WARN `26.11.58 > 26.11.2`: newer than helicop's pin | — |
| [06] Detect BIOS PIN | FAIL after about 50 s: `no Anacapa_BIOS_* object found` (a Helios-P name) | `s` |
| [07] Check SBIOS version | WARN `no sbios_version_heliosm in best_known_config.cfg` | — |
| [08] Detect HPM CPLD version | WARN `i2cdump failed` | — |
| [09] Wait for power-on (snapshot) | FAIL `I2C read failed (bus 4, dev 0x20)` | `s` |
| [10] Wait for power stage 4 | FAIL `0x??`. Its fix then shows `DC power-on issued … expected 0x12` as a FAIL, **but the power-on was sent**. Step 10 then fails once more | `f`, then `s` |
| [13] Wait for host boot | PASS `SSH connection established` | — |

**Driving it with `expect`.** The menu and the FAIL prompts read from the
terminal. This driver runs one program per session. Start it from the helicop
directory:

```tcl
#!/usr/bin/expect -f
# hc.exp <alias> ac|on     ac: ac_cycle     on: prep_machine auto upto 13
set alias [lindex $argv 0]; set mode [lindex $argv 1]
set timeout 2400; match_max 200000
spawn ./helicop $alias
set fixed 0
proc to_menu {} {
    global fixed
    expect {
        -ex {Type "confirm" to proceed} { send "confirm\r"; exp_continue }
        -re {nfo: $} {
            # a FAIL prompt. It ends in "[i]nfo: ", with colour codes inside the brackets
            set b $expect_out(buffer)
            if {[string first "running fix: PerformDcPowerOn" $b] >= 0} { set fixed 1 }
            if {!$fixed && [string first "Suggested fix: " $b] >= 0 && [string first "PerformDcPowerOn" $b] >= 0} {
                set fixed 1; send "f\r"
            } else { send "s\r" }
            exp_continue
        }
        -ex {Press Enter to continue...} { send "\r"; exp_continue }
        -ex {Program [options]: }        { return }
        -re {assword: ?$}                { send "\003"; exit 4 }
        timeout { exit 2 }
        eof     { exit 3 }
    }
}
to_menu
if {$mode eq "ac"} { send "ac_cycle\r" } else { send "prep_machine auto upto 13\r" }
to_menu
send "quit\r"; expect eof
```

```bash
cd <helicop dir>        # with PATH and HELICOP_SETTINGS_FILE exported as in the setup
T=CT03; TRAY=10.210.11.16; B=10.210.11.48
OLD=$(ssh root@$TRAY cat /proc/sys/kernel/random/boot_id); ssh root@$TRAY sync
expect -f ~/hc/hc.exp $T ac
sleep 30; until curl -sk -m 8 -o /dev/null https://$B/redfish/v1/; do sleep 10; done; sleep 120
expect -f ~/hc/hc.exp $T on
[ "$(ssh root@$TRAY cat /proc/sys/kernel/random/boot_id)" != "$OLD" ] && echo rebooted   # now perf_setup
```

**Pitfalls.**
- **Do not substitute `dc_power_off` followed by `prep_machine`.** That cycles
  only the host's DC power. On CT2 the host was still unreachable 7 minutes after
  the power-on. An AC cycle then brought it up in 3½.
- **Session 2 can spin at step 01** (`waiting for BMC SSH … attempt n/250`).
  - Cause: the session's first login to the BMC failed. Its retry then logs in as
    `root` with the `admin` account's password, which this BMC rejects every time.
  - Cost: 250 attempts take about an hour.
  - Fix: Ctrl-C and start session 2 again.
  - Seen on 2 of 6 trays whose session 2 started within the BMC's first minute.
    Both passed when started again about 4 minutes later, hence the 2-minute wait.
- `pkill -f 'helicop CT03'` also matches the shell that runs it. Use
  `pkill -f '[h]elicop CT03'`.
- Other people run helicop against this rack too. Check
  `pgrep -af '[h]elicop'` on the jump host first: two sessions acting on one tray
  will fight.
- The rack-mode screen (`./helicop rack/MissionM.rack`) offers the same steps as
  routines (`PowerManagement ▸ AC_Cycle`, `DC_PowerOn`). It was not used here.

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

### ⚠️ Check byte sizes — `import` passes on a truncated file

The check above is necessary and **not sufficient**: a **0-byte** `.py` imports
cleanly, prints its path, and defines nothing. A `docker cp` interrupted
mid-copy — host reset, BMC watchdog, rack power event — leaves exactly that
behind. Seen on all four nodes at once: the rocjitsu hook and all three patched
ATOM files were 0 bytes while every check in this section reported success.

For `gfx1250-atom-20260918-ep8` the sizes are:

```bash
sudo docker exec k3ep16 bash -c '
stat -c"%s %n" \
  /app/rjprefix/lib/libhsa_hotswap_rocjitsu.so             `# 129112`  \
  /app/rjprefix/lib/librocjitsu_gfx1250_b0_to_a0.so.0.3.0  `# 8002216` \
  /app/ATOM/atom/model_ops/mla_unfused_gather.py           `# 7499`    \
  /app/ATOM/atom/utils/envs.py                             `# 47967`   \
  /app/ATOM/atom/model_ops/attention_mla.py                `# 146235`
grep -c use_unfused_gather_kv_b_proj /app/ATOM/atom/model_ops/attention_mla.py  # 3'
```

Re-run it after **every** reboot or container rebuild, not just the first
install.

### The `envs.py` hunk does not apply to this image

`patch -p1` reports `Hunk #1 FAILED` on `atom/utils/envs.py`. The PR is based on
`main`, whose trailing context (`ATOM_USE_FLYDSL_FP8_PREFILL_ATTN`) does not
exist in the image's ATOM — which has `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION`
in that position instead. `attention_mla.py` applies, with fuzz 2 at offset −199.

Insert the `ATOM_UNFUSED_GATHER_KV_B_PROJ` block by hand immediately after
`ATOM_USE_FLYDSL_GATHER_KV_B_PROJ`, keeping the image's own following context.
Then verify the env var reads `False` by default **and** `True` under the env
var — both, not just one.

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
export MEGA_DISPATCH=mori MEGA_DISPATCH_WIRE=fp4   # MEGA_WIRE is dead, see below
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
capture. On `gfx1250-atom-20260918-ep8` with a warm page cache it has been
measured at **under 6 minutes**, so treat 25 as the budget, not the
expectation. `ncclCommInitRank` hangs intermittently on this stack with no
known fix; retry. Give any retry wrapper **at least 25 minutes** or it will
kill servers that are loading normally.

`py-spy` is **not in the image**; install it before you need it:

```bash
sudo docker exec k3ep16 pip install --break-system-packages py-spy
```

To tell a hang from progress, look at the ModelRunner (`ATOM::DPxTP0`), not the
EngineCore, which only ever shows as waiting on a queue:

```bash
py-spy dump --pid $(pgrep -f 'ATOM::DP0TP0' | head -1)
```

`MainThread (idle)` in `as_completed` is a healthy load. `(active)` in
`ncclCommInitRank` is the hang — in practice it shows up as:

```
synchronize (torch/cuda/streams.py:108)
__init__ (communicator_pynccl.py:143)
init_model_parallel_group (aiter/dist/parallel_state.py:1626)
```

### When retrying does not help

If it reproduces every time, it is not the intermittent hang and no amount of
retrying fixes it. **Changing the transport is not the lever** — the channels
build cleanly in all three cases and the hang is afterwards:

| setting | transport RCCL picks | outcome |
|---|---|---|
| recipe default | `P2P/CUMEMMNNVL` | hangs |
| `NCCL_MNNVL_ENABLE=0` | `P2P/CUMEM` | hangs |
| `+ NCCL_CUMEM_ENABLE=0` | `P2P/IPC` | hangs |

Work below ATOM instead. This takes a minute and tells you whether ATOM is
involved at all:

```python
# ncclmin.py
import os, torch, torch.distributed as dist, datetime
r=int(os.environ["RANK"]); w=int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(r)
dist.init_process_group("nccl", rank=r, world_size=w,
                        timeout=datetime.timedelta(seconds=45))
t=torch.ones(1024, device=f"cuda:{r}")*(r+1)
dist.all_reduce(t); torch.cuda.synchronize()
print(f"[{r}] {t[0].item()} expect {w*(w+1)//2}", flush=True)
```

```bash
for W in 1 2 4; do
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29800+W)) WORLD_SIZE=$W \
  sh -c 'for r in $(seq 0 $((WORLD_SIZE-1))); do RANK=$r python3 ncclmin.py & done; wait'
done
```

`world_size=1` passing while 2 and 4 time out with `last completed work: -1`
means the collective kernel never ran. That is a platform fault; nothing in
this recipe's configuration will move it.

### The MoE dispatch backend: mori, and why `MEGA_DISPATCH=flydsl` does nothing

`MEGA_DISPATCH` takes `flydsl` or `mori`, and **aiter's own default is
`flydsl`** (`mega_moe.py`: `os.environ.get("MEGA_DISPATCH", "flydsl")`). This
recipe names `mori` — but on the fp4 wire the environment variable is not what
decides it, and setting it to `flydsl` changes nothing.

ATOM passes the backend explicitly, and an explicit argument wins over the env:

```python
# atom/model_ops/fused_moe/mori_v2_prepare_finalize.py
# Only mori's dispatch carries the scale row, so a quantizing wire has
# no other backend to run on. Named here rather than left to
# $MEGA_DISPATCH, whose default is flydsl: otherwise asking for fp4 is
# rejected at the first MoE layer for a reason the operator did not set.
**({"dispatch_backend": "mori"} if _MEGA_DISPATCH_WIRE in ("fp8", "fp4") else {})
```

So with `MEGA_DISPATCH_WIRE=fp4`, `dispatch_backend="mori"` is hard-wired and
`$MEGA_DISPATCH` is dead. aiter's side agrees:
`# Only mori's kernel carries the scale row`.

**The reason is the scale row, not performance.** fp4 and fp8 are quantizing
wires: every token carries a per-token scale alongside its payload. FlyDSL's
dispatch kernel does not move that row, so a quantizing wire has nowhere else
to run.

| | mori | flydsl |
|---|---|---|
| carries the per-token scale row | **yes** | no |
| usable with `MEGA_DISPATCH_WIRE=fp4` / `fp8` | **yes** | no |
| usable with `MEGA_DISPATCH_WIRE=bf16` | yes | yes |
| what this recipe runs | **this one** | — |

> **Scope:** the table and the paragraph below describe the aiter inside
> `gfx1250-atom-20260918-ep8`. Since aiter #5814 (2026-09-25) flydsl's TDM
> dispatch carries the scale row too, so on newer aiter — including the
> [optimized source stack](#optimized-source-stack--pending-verification) — the
> flydsl column no longer holds. The environment variable stays inert either
> way: the ATOM comment above dates from 2026-09-01 (ATOM #2018), and ATOM
> still hard-wires `mori` for fp4/fp8.

**Switching to flydsl means giving up fp4 on the wire.** It is a pair, not a
single knob:

```bash
export MEGA_DISPATCH=flydsl
export MEGA_DISPATCH_WIRE=bf16     # fp4 is not available on this backend
```

That multiplies the dispatch payload by **4x** (fp4 → bf16). Expert
all-to-all is already the second-largest cost on this stack — the A0-path
profile puts `mori_ep_dispatch_tdm_fp4x2` at 10.08M us against 11.16M us for
the largest GEMM — so a 4x wider dispatch is the wrong direction unless
something else pays for it. **Not measured here**; recorded so nobody spends a
run discovering that the env var alone is inert.

The knob that *is* live on the fp4 wire is `ATOM_MORI_V2_FUSED`: `1` binds
aiter's `MegaMoEGfx1250`, which owns the fused dispatch/combine pair, `0` binds
mori's v2 op-layer running the plain path. This recipe uses `1`. It has not
been swept.

### `MEGA_WIRE` is deprecated — drop it

`MEGA_WIRE` was renamed to `MEGA_DISPATCH_WIRE` (combine now gets its own
wire). It is **no longer read**, and both ATOM and aiter raise if the two
disagree:

```
RuntimeError: MEGA_WIRE was renamed to MEGA_DISPATCH_WIRE; update the launch
script, the old name is no longer read
```

Older launch scripts that set `MEGA_WIRE=fp4 MEGA_DISPATCH_WIRE=fp4` survive
only because the values match. Remove `MEGA_WIRE`.

### KV budget

```bash
grep 'Memory budget' <log> | tail -1     # available_for_kv must be positive
grep -oE 'experts=[0-9]+' <log> | sort -u  # EP16 -> 56
```

**`available_for_kv` scales with the board, so re-derive it rather than reusing
the numbers here.** Measured on 432 GiB boards, same launch flags:

```
utilization=0.90  budget=388.80GB  peak_torch=194.67GB  available_for_kv=159.37GB
utilization=0.94  budget=406.08GB  peak_torch=194.67GB  available_for_kv=176GB+
```

i.e. ~4.7x the 33.53 GB the 288 GiB boards give at 0.90. The
`--max-num-batched-tokens 2048` lever below still matters — `peak_torch` is
~195 GB either way — but the pressure it relieves is much lower on 432 GiB
parts.

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
| LLVM PHI assertion on long input at concurrency | Triton `gather_kv_b_proj` codegen. Needs [PR #2380](https://github.com/ROCm/ATOM/pull/2380) **and** `ATOM_UNFUSED_GATHER_KV_B_PROJ=1` — the env var alone does nothing on stock ATOM. The [optimized source stack](#optimized-source-stack--pending-verification) fixes it in aiter instead (#6120) |
| `assert not ca_comm.disabled` kills the ModelRunner while HTTP stays up | `ATOM_USE_CUSTOM_ALL_GATHER` and `AITER_CUSTOM_AR_USE_SYMM_MEM` must be set together |
| MoE GUGU layout error | `ATOM_MOE_GU_ITLV=1` |
| `ATOM_USE_TRITON_MOE_DECODE=1` asserts | K3's activation is `situ`, not SiLU |
| `/dev/kfd` missing after a reboot | `amdgpu` is blacklisted on the kernel command line. Load it with `perf_setup`; a bare `modprobe` skips the tuning. See [After a reboot](#after-a-reboot-load-the-driver-then-confirm-the-links-trained) |
| `accel_state` reads `unconfigured` | Links are still training after `modprobe`. Wait; do not re-`modprobe` |
| A run dies with nothing in the log to explain it | Possibly a GPU recovery reset. Reload the driver with `gpu_recovery=0 halt_if_hws_hang=1` so the next one halts diagnosably |
| Startup stops at `load RCCL version`, CPU spinning at ~110% | Intermittent `ncclCommInitRank` hang. Retry; allow ≥25 min. Looks identical to a bad fabric — rule that out with [ubench07](#validate-the-fabric-first--ubench07) once, then retry |
| Multi-node rendezvous hangs with no error | Nodes are not in the same `PPOD_ID` / `VPOD_ID` domain, or the fabric itself is bad. Confirm with [ubench07](#validate-the-fabric-first--ubench07) before blaming the engine |
| `accel_state` stuck at `unconfigured` long after `modprobe` | `ifoe` was not loaded when `amdgpu` initialised. It never recovers on its own — see [After a reboot](#after-a-reboot-load-the-driver-then-confirm-the-links-trained) |
| `load RCCL version` hang that reproduces **every** time | Not the intermittent hang. Drop below ATOM with the minimal `all_reduce` above before touching any ATOM flag |
| Patch/prefix "installed" but behaves as if absent | Files may be 0 bytes from an interrupted `docker cp`; `import` still succeeds. Check byte sizes |
| OOM at load with `0 bytes is free` while `peak_torch` is small | Another job owns the GPUs. Check `rocm-smi --showmemuse` (idle reads 0% / ~177 MB) and `docker ps` on every node before launching |
| `available KV budget (-NN GB)` on a config that booted an hour ago | Same cause, later symptom. Someone else's weights are loading. See [Is the rack actually free?](#is-the-rack-actually-free) |
| Cross-node bandwidth ~4.5 GB/s instead of thousands | MNNVL not in effect, silently fell back to TCP. Set `NCCL_MNNVL_ENABLE=1` |
| Log appears frozen | tqdm writes `\r`; pipe through `tr '\r' '\n'` |
| Engine processes `<defunct>`, kfd process list never drains, dmesg `MES … failed to respond to msg=REMOVE_QUEUE` | GPU wedged; `gpu_recovery=0` keeps it that way by design. BMC `ForceRestart`, then `perf_setup`. See [Rebooting a tray](#rebooting-a-tray) |
| `perf_setup` report says mode D, but `mtype_local` / `noretry` read `-1` | `amdgpu` was already loaded when it ran. Reboot, then run `perf_setup` before anything else |
| A tray drops out mid-run; the next boot's BERT says `event severity: fatal`, `Perr: CPU0` | Host CPU fatal error, not the engine. See the boot-log table in [Rebooting a tray](#rebooting-a-tray) |
| helicop power steps FAIL with `0x??` or `I2C read failed (bus 4, dev 0x20)` | Expected on this rack: the CPLD stage register cannot be read on this BMC firmware. The power commands still work. Check Redfish `PowerState` and `boot_id`. See [With helicop](#with-helicop-ac-cycle-then-power-on) |
| helicop `prep_machine` spins at step 01 `waiting for BMC SSH` after an AC cycle | Its first BMC login failed, and the retry (as `root`) never succeeds. Ctrl-C, then start the session again 2 min after the BMC is back |
| Host never reaches the OS after `dc_power_off` and a power-on | Seen once (CT2). AC-cycle the tray with helicop instead |

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

### Server launch for AgentX

Same as [Launch](#launch) except for these four. **All four matter** — the
first two are what make it boot and stay up at all:

```bash
--enable_prefix_caching          # instead of --no-enable_prefix_caching
--gpu-memory-utilization 0.94    # instead of 0.90
--max-num-batched-tokens 2048    # on 288 GiB boards. On 432 GiB, raise it -- see below
--max-num-seqs 8                 # unchanged -- do NOT raise it either
```

Measured effect of the utilization bump, everything else equal:

```
0.90 -> available_for_kv=33.53GB  num_kvcache_blocks=146862
0.94 -> available_for_kv=50.97GB  num_kvcache_blocks=231173   (+52%)
```

⚠️ **Do not raise `--max-num-batched-tokens` to speed up prefill.** It looks
like the obvious fix for a 100k-token prompt being split into 51 chunks, and it
is wrong twice over:

1. **It will not boot** — *on a 288 GiB board*. At 16384 with
   `--gpu-memory-utilization 0.93`:
   ```
   RuntimeError: Per-request cache tensor (3.35GB for 8 slots) exceeds available
   KV budget (-4.67GB) at --gpu-memory-utilization 0.93. Set
   --gpu-memory-utilization >= 0.96 ... or reduce --max-num-seqs (currently 8).
   ```
   Raising utilization far enough to cover it is not possible — the deficit is
   ~40 GB against a 288 GB card already at 0.93.

   **On a 432 GiB board this does not apply.** Measured at 16384 with
   `--gpu-memory-utilization 0.94`, everything else as above:

   | | 2048 | 16384 |
   |---|---|---|
   | `peak_torch` | 194.67 GB | 236.08 GB |
   | `cudagraph_est` | 1.18 GB | 9.40 GB |
   | `available_for_kv` | 159.37 GB | **118.80 GB** |
   | TTFT, 5-token prompt | 2.73 s | **0.50 s** |

   It boots, and 118 GB of KV is still 3.5x what a 288 GiB board has at 0.90.
   **Re-derive the budget for your board instead of copying either number.**
2. **Chunk size is not the whole story.** At the *same* 2048 chunk size, a 14k
   fixed-length run prefills at 1252 tok/s per rank while the agentic trace
   manages 283 tok/s. The difference is context length: chunk *n* attends to
   everything before it, so the cost is quadratic in prompt length and a bigger
   chunk only amortises per-step overhead. On a 288 GiB board you would trade
   ~38 GB of KV for maybe 2-3x, and there KV is the thing you cannot spare.

   **Where KV is not scarce, take the 2-3x.** At 2048 the AgentX warmup on
   this rack needed **2 h 52 min** to clear 707 requests, and the 1800 s
   profiling window that followed completed **3 of 89** trajectories — aiperf
   then rejected the run (see below). At 16384 the same warmup's snapshot
   primers cleared roughly **8x faster**. On a 432 GiB board the conservative
   value is not a safe default; it is what makes AgentX unmeasurable.

⚠️ **`--enable_prefix_caching` on this platform deadlocked the cluster once, at
`--gpu-memory-utilization 0.90`.** Signature, for recognition:

```
[13:08:38] Engine 011: prompt throughput: 0.0 tok/s, Running: 1, KV usage: 98.6%
           ... 30 minutes of total silence from every rank ...
[13:38:47] Engine Core: _sync_dp_state failed: Timed out waiting 1800000ms
[13:38:47] Engine Core: All DP ranks agreed to shutdown, exiting busy_loop
```

**15 of 16 ranks logged the timeout; one did not.** That is the shape of a
collective one participant never joined, not of a slow cluster — and the rank
that never joined was the one at 98.6% KV occupancy. The client sees none of
this: aiperf sat at `errors=0` with 4 requests in flight for another 27 minutes
after the server was gone. It has not recurred at 0.94, which is the reason for
that setting, but the root cause is not proven — so watch for it, and if a run
goes quiet, take stacks before killing anything:

```bash
sudo docker exec k3ep16 py-spy dump --pid $(pgrep -f 'ATOM::DP0TP0' | head -1)
```

A ModelRunner stopped in `ncclCommInitRank` or inside a mori dispatch is a
collective hang; one spinning in the scheduler is a KV allocation problem.

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

### ⚠️ A server too slow for the trace fails the run outright

aiperf requires **95% metric coverage** over the profiling window. If too few
trajectories finish inside it, the whole run is discarded — you get no numbers
at all, not slow ones:

```
Phase profiling timed out, cancelling all credits.
  Stats: sent=89, completed=3, cancelled=0, in_flight=86
Profiling metric coverage below the required 95.0% for phase 'profiling':
  TTFT=0.2%, inter-token latency=0.2% over the configured 1800.0s duration
AIPerf System Exit Errors: ProfileMetricCoverageError
```

That was `--concurrency 64` with `--max-num-batched-tokens 2048` on 432 GiB
boards: 3 of 89 trajectories completed in 1800 s. The summary table it prints
before dying is misleading — `Benchmark Duration: 3.63 sec` is the span between
the three completed requests, **not** the length of the run.

Two things cause it, and both need fixing together:

- **Chunk size.** See [the batched-tokens note](#server-launch-for-agentx) —
  at 2048 a 100k-token turn is 51 chunked prefill steps.
- **Concurrency above what the server can retire.** A profiling "request" here
  is a whole multi-turn trajectory (`BranchOrchestrator: spawned=188` for 89
  sent), so nominal concurrency costs far more than it does in
  `benchmark_serving`. If nothing completes, halve it.

Watch `returned=` in the warmup progress lines before the profiling phase
starts. If the primers alone take tens of minutes, the profiling window will
not produce coverage either.

---

### AgentX result

16 GPUs (4 nodes, dp16ep16), 432 GiB boards, `gfx1250-atom-20260918-ep8`.
1790 s profiling window, 53 trajectories retired, 0 errors.
`coverage passed: TTFT=79.8%, inter-token latency=99.4%`.

**Headline three:**

| | |
|---|---|
| **Interactivity, p90** | **2.53 tok/s/user** (avg 1.07, p50 0.53, p99 4.21) |
| **Total throughput per GPU** | **182 tok/s** (2913 tok/s aggregate ÷ 16) |
| **Prefix cache hit rate** | **92.36%** measured (94.68% theoretical for the trace) |

**Latency and shape:**

| metric | avg | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT (ms) | 198,611 | 124,981 | 444,961 | 588,742 |
| Inter-token latency (ms) | 1,420 | 1,898 | 1,992 | 2,050 |
| Request latency (ms) | 524,555 | 419,238 | 1,005,916 | 1,409,498 |
| Input sequence length (tok) | 100,926 | 60,788 | 217,227 | 605,798 |
| Output sequence length (tok) | 212 | 159 | 420 | 823 |
| Effective concurrency | 15.11 | 14.00 | 25.00 | 28.00 |
| Prefill throughput per user (tok/s) | 1,386 | 411 | 4,012 | 10,675 |

**Aggregates:** input 2907 tok/s, output 6.12 tok/s, total 2913 tok/s,
0.029 req/s. 5,349,055 prompt tokens of which 4,940,224 served from cache.

**`--fake-eplb` is part of the method here, not an accident.** Agentic runs on
this configuration are measured with it so that expert load is uniform and
successive runs are comparable; the router is replaced with a uniform
distribution, so the generated text is not K3's and is not meant to be read.
Use the [accuracy gate](#accuracy-gate--run-this-before-any-benchmark) and
GSM8K for correctness, and these numbers for throughput.

One technical note for whoever reads them: uniform routing is **not** simply a
best case. It spreads a batch's tokens across all 896 experts, so the receiving
rank gets smaller per-expert batches than real, skewed routing would. Whether
that helps or hurts decode on this stack is not established here.

⚠️ **Two limits on what these numbers can be compared against.**

1. **The launch differs from the [AgentX launch](#server-launch-for-agentx)
   above** — `--max-num-batched-tokens 16384` (not 2048) and `--concurrency 32`
   (not 16). At 2048/con64 the run does not produce coverage at all.
2. **There is no baseline**, so nothing here can be attributed to `--fake-eplb`,
   `ATOM_DP_SESSION_AFFINITY=1` or the chunk size individually.

**What the numbers say.** Decode is the bottleneck, not prefill: 212 output
tokens take ~326 s of decode at 1.4 s per token, while prefill moves 2907 tok/s
in aggregate. Effective concurrency settles at 15.11 against a nominal 32 because lanes idle
inside the trace — size `--concurrency` by what retires, not by what you ask
for. Note what that implies: **`--concurrency 16` and `--concurrency 32` land
on nearly the same working point here**, so sweeping downward from 32 buys no
new information. To move the operating point, go up. The 92.36% cache read rate is the thing keeping ISL p90 of 217k
tokens affordable at all.

`ATOM_DP_SESSION_AFFINITY=1` was set for this run. It places a new session on
the lightest DP rank and pins every later request to that cache owner, which is
the behaviour prefix caching needs across turns — a follow-up landing on a
different rank re-prefills the whole context. It is off by default and costs
nothing to set; its effect here is unmeasured for want of a baseline.

---

## Where the time goes, and what to try next

From the AgentX run on `gfx1250-atom-20260918-ep8`, 16 GPUs, con32.

### Decode is the bottleneck, not prefill

| | measured |
|---|---|
| Prefill, aggregate | **2,907 tok/s** |
| Prefill, per user | 1,386 tok/s avg (p90 4,012) |
| Decode, aggregate | **6.12 tok/s** |
| Decode, per user | **1.07 tok/s** (p50 0.53, p90 2.53) |
| Inter-token latency | **1,420 ms** avg (p50 1,898, p90 1,992) |

An average trajectory is 100,926 input tokens and **212 output tokens**. The
input side costs ~35 s of prefill; the 212 output tokens cost **~326 s**. Over
90% of request latency is decode, generating three orders of magnitude fewer
tokens than prefill consumes.

An ITL near 1.9 s at p50 is not a batching artefact — decode batches are tiny
here. `--max-num-seqs 8` caps each DP rank at 8 sequences, and effective
concurrency across the cluster settled at **15.11**, i.e. roughly **one
sequence per rank**. Every decode step pays the full MoE all-to-all for a
batch of ~1.

### What that implies, in order of expected value

1. **Raise the decode batch.** `--max-num-seqs 8` is inherited from the A0 path
   and has never been swept. On 432 GiB boards at 0.94 there is
   118.80 GB of KV — 3.5x what the A0 path had at 0.90. This is the cheapest
   thing to try and the most likely to move ITL.
2. **Raise concurrency.** Effective 15.11 against nominal 32 means the client
   is not the limit — but see the warning in
   [AgentX result](#agentx-result): con16 and con32 land on the same working
   point, so go *up*, not down. A con64 attempt was made and not completed.
3. **EP32.** Per-GPU expert weight halves (14.64 GiB/layer ÷ 32 = 0.46, against
   0.92 at EP16), freeing roughly **41 GiB per GPU** for KV — which feeds
   directly into (1) and (2). Untested; needs 8 nodes with the full checkpoint
   on each.
4. **Real EPLB.** ATOM has `--eplb-enable` / `--enable-eplb` with
   `--eplb-config` (JSON: `num_redundant_experts`, `placement_policy`
   `naive`|`biased`, `rebalance_interval`, `load_window_size`, …). K3 is 896
   routed experts top-16, 56 per GPU at EP16; expert load is long-tailed, and
   replicating the hot ones costs KV that these boards have. **This is
   different from `--fake-eplb`**, which only uniformises the router for
   measurement and is not deployable.
5. **Prefill chunk is already handled.** `--max-num-batched-tokens 16384` took
   TTFT on a short prompt from 2.73 s to 0.50 s and made AgentX measurable at
   all. Going higher trades KV that (1) and (2) want more.

### What is already healthy

Prefix caching is doing its job and should not be the next thing tuned:
**92.36%** of 5,349,055 prompt tokens were served from cache, against a 94.68%
theoretical ceiling for the trace. `ATOM_DP_SESSION_AFFINITY=1` was set, which
pins a session to its cache owner so a later turn does not re-prefill on
another rank; its isolated contribution is unmeasured.

---

## Optimized source stack — pending verification

> ⚠️ **Pending verification. Nothing in this section has been run on this
> page's harness yet.** It merges XiaobingSuper's optimization work on the same
> rack (2026-10-01 → 10-07, branch
> [`xiaobingsuper/kimi-k3-mi455-b0-recipe`](https://github.com/ROCm/ATOM/tree/xiaobingsuper/kimi-k3-mi455-b0-recipe)).
> Every speedup quoted below is the PR's own **isolated operator** number
> (MI455 B0, CUDAGraph replay) — none is end-to-end, and they must not be added
> up. The stack has not passed the
> [accuracy gate](#accuracy-gate--run-this-before-any-benchmark) or produced an
> AgentX result. Until it does, run
> [Appendix A](#appendix-a-the-exact-configuration-this-was-validated-on).

Unlike the rest of this page, this is a **source build**, not the bring-up
image: ATOM and aiter at pinned commits plus eight PRs, all still open on
2026-10-08. The launch is Config B with the deltas in
[Environment and CLI](#environment-and-cli--delta-from-config-b).

### Why these PRs: TP1 shapes on gfx1250

Wide EP runs attention and the dense layers at `-tp 1`, so every per-rank shape
is the **full model width**: 96 heads where a TP8 deployment sees 12, a dense
intermediate of 33792 where TP8 sees 4224, and every rank resolving routes over
all 896 global experts. aiter's kernels were sized and validated on MI355 TP8
shapes, and at TP1 several of their limits are crossed — a 512-expert scan, a
64 KB LDS assumption, a grid-stride loop that only appears at 96 heads. Most of
the PRs below remove one of those limits; the rest retune for what is specific
to gfx1250: wave32, 320 KB LDS, FP8 WMMA, TDM.

### The PRs

#### Decode hot path — every MoE layer, every step (×92)

**[aiter #6131](https://github.com/ROCm/aiter/pull/6131) — router top-k.**
Adds a gfx1250 Triton kernel for sigmoid + bias top-k and dispatches it from
`biased_grouped_topk` when `num_expert_group == topk_group == 1`
(E ∈ {512, 768, 896, 900, 1024}, K ∈ {4, 8, 16}, bf16 logits). K3's router —
896 experts, top-16, `noaux_tc` — reaches exactly this call through ATOM's
`rocm_aiter_biased_grouped_topk`; on `main` it runs the generic HIP kernel. The
new kernel reproduces the HIP kernel's wave32 tie-break order, so the selected
expert ids are bit-identical. *Isolated:* E896/K16 34–36% faster than the best
generic path, M = 1…128.

**[aiter #6115](https://github.com/ROCm/aiter/pull/6115) — global→local expert LUT, 512 → 1024.**
On the fp4 wire ATOM forces the **mori** dispatch backend (see
[The MoE dispatch backend](#the-moe-dispatch-backend-mori-and-why-mega_dispatchflydsl-does-nothing)),
and with mori the MegaMoE receiver rebuilds a global→local expert LUT **every
layer, every step** — the same kernel also zeroes the per-expert route
counters. The FlyDSL LUT kernel was a single 512-thread workgroup with one
thread per global expert, so K3's 896 fell back to ~7 torch launches plus a
counter fill and a multiply. The PR scans in two levels — DPP inside each wave,
then one wave over the wave totals, 2 barriers instead of 9–10 — in a
1024-thread workgroup; 1024 = 32² is exactly what that covers on wave32.
*Isolated:* 13.4× at the K3 shape (896 global / 56 local / top-16);
1.33–1.42× for the existing ≤ 512 cases.

#### MXFP4 routed projections — `routed_expert_down/up_proj` (×184 GEMMs per step)

**[ATOM #2466](https://github.com/ROCm/ATOM/pull/2466) — MXFP4 scale layouts.**
On gfx1250 with preshuffled MXFP4 weights, ATOM's default GEMM (Opus F4GEMM)
expects `shuffle_scale_f4` scales while ATOM stores `e8m0_shuffle`. The PR
routes these layers to aiter's `gemm_afp4wfp4_preshuffle` instead — the gfx1250
Gluon TDM kernel, which reads `e8m0_shuffle` — and disables the routed
RMSNorm+MXFP4 fusion on gfx1250, whose padded/preshuffled `[256, 112]` scales do
not match the row-major `[M, 112]` that GEMM wants below M = 32. **This is what
makes `ATOM_USE_TRITON_GEMM=0` safe on gfx1250** — the switch this stack needs
so BF16 GEMMs take the tuned table and ptpc FP8 GEMMs take aiter's preshuffled
kernels. Without it, that switch silently pairs MXFP4 weights with the wrong
scale layout. Needs aiter #6097 (merged, already in the aiter base).
*Isolated:* none claimed — correctness.

#### Dense MLP — layer 0 only

**[aiter #6078](https://github.com/ROCm/aiter/pull/6078) — SiTUv2 + per-token FP8 at D = 33792.**
K3's only dense MLP has intermediate 33792, unsharded at `-tp 1`. Under ptpc its
`down_proj` is per-token FP8, so ATOM calls aiter's fused SiTUv2+quant kernel —
and on `main` that kernel's `AITER_CHECK(d <= 16376)` **aborts the process**;
the cap came from assuming 64 KB of LDS. The PR uses gfx1250's real 320 KB,
reads LDS size and wave size per device, removes the cap, adds a 1024-thread
`D = 33792` dispatch with packed BF16 math, and recomputes activations for rows
that do not fit in LDS. Required for ptpc at TP1, not merely faster; the shared
experts (D = 6144) already worked. *Isolated:* 2.36–2.57× against SiTUv2
followed by a standalone quant.

#### KDA prefill (69 layers)

**[aiter #6130](https://github.com/ROCm/aiter/pull/6130) — FlashKDA K2 schedule.**
K3 prefill runs `chunk_kimi_delta_attn` → `flash_kda_fwd`; the Gluon K1/K2 is
gfx950-only, so gfx1250 runs the Triton K2 — the serial, per-chunk delta-rule
recurrence. The PR publishes a gfx1250 schedule (`BW=16, num_warps=1,
num_stages=3`) and fixes the selection bug that made publishing necessary: with
autotune off, the production default, `autotune_configs` always returned the
fallback config, so no architecture's tuned shortlist was ever used. Only
gfx1250 changes (gfx950's first shortlisted entry equals the fallback). Output
and final state are bitwise equal. *Isolated:* K2 1.52× / 1.59× / 1.64× at
2K / 4K / 14K tokens (marked provisional in the PR).

**[ATOM #2458](https://github.com/ROCm/ATOM/pull/2458) — gated RMSNorm after KDA.**
Two parts. Prefill now hands the contiguous KDA output straight to `o_norm`
instead of `out.copy_()`-ing it into an identical buffer — about 100 MB of copy
traffic per KDA layer per 2048-token chunk. And a multi-row gated-RMSNorm kernel
replaces one program per 128-element (token, head) row — 196,608 programs for a
2048-token chunk at 96 heads — with 8 rows per program (32 at ≥ 900,000 rows,
which a 16384-token chunk does not reach). ⚠️ **The multi-row kernel only runs
when `o_proj` is BF16.** With ptpc, as in this stack, `o_norm` fuses per-token
FP8 quant in a different kernel (one program per token), and only the copy
removal applies. *Isolated:* copy + norm pipeline 1.32–4.97× over 256–16,384
tokens.

#### MLA cached-prefix expansion (24 layers; chunked-prefill chunk ≥ 2, or a prefix-cache hit)

**[aiter #6120](https://github.com/ROCm/aiter/pull/6120) — the LLVM PHI assertion, fixed at the root.**
This is the assertion [PR #2380](#-required-patch-atom-pr-2380) works around.
The flat (page-size-1) Triton `gather_kv_b_proj` caps its workers at
`CU × 6 / heads` and grid-strides over the remaining chunks; at 96 heads the cap
is 16 chunks (128 at TP8's 12 heads), so TP1 generates the loop on modest
prefixes, and gfx1250's LLVM asserts on it. On gfx1250 the PR launches one
workgroup per KV chunk, so the loop is never generated and the kernel stays
fused. With it, the unfused fallback is not needed. *Isolated:* none claimed —
compile fix.

**[aiter #6121](https://github.com/ROCm/aiter/pull/6121) — FlyDSL FP8 WMMA `gather_kv_b_proj`.**
Gathers the FP8 latent rows, runs an 8-wave **FP8 WMMA** projection whose
epilogue writes K-nope and V directly, then broadcasts K-PE across the heads —
no full projection temporary. At 96 heads each prefix row expands to ~61 KB of
K/V, so the op is output-bandwidth-bound and grows with context length
(quadratically over a long prompt split into chunks). It requires an FP8,
16×16-preshuffled `kv_b_proj` with per-row scale — which ptpc produces — and an
FP8, unshuffled KV cache. ATOM picks it up through
`ATOM_USE_FLYDSL_GATHER_KV_B_PROJ`, whose default is already `1`. *Isolated:*
1.07× / 2.07× / 2.70× / 2.80× at 2K / 4K / 8K / 16K prefix rows — against a
multi-kernel unfused baseline, not against the #6120-fixed fused kernel.

#### Also in the bundle, without a PR

The bundle's `atom.patch` carries five local changes that have no PR yet; their
tests are included.

| change | what | why |
|---|---|---|
| KDA decode recurrence | `fused_sigmoid_gating_delta_rule_update`: `num_warps` 4 → 8 | wider launch for the 96-head decode recurrence. ⚠️ unconditional — applies to every model and arch that uses this kernel |
| causal-conv1d decode | width-4, single-token update writes the shifted conv state from registers instead of re-reading columns through a 2-D tile; fork-safe | K3's normal decode step |
| causal-conv1d prefill | width-4 conv vectorised across 8 token rows instead of a serial rolling window; a new `has_state_fork` metadata flag keeps forks, APC, other layouts and short calls on the generic kernel | K3 long prefill |
| guarded MLA split8 | Triton MLA decode `num_kv_splits` 4 → 8, only for the exact K3 contract (gfx1250, bf16, fp8 KV, 96 heads, 512/128/64/128, page 1, max batch ≤ 8, unshuffled) **and** a live batch ≤ 2 | AgentX decodes about one sequence per rank; for wider batches split8 regresses on short tails |
| AttentionResidual long prefill | `(num_warps, num_stages, BL) = (4, 2, 1)` for H = 7168, T ≥ 2048 on gfx1250 | long-prefill launch tuning |

Deliberately **not** included: ATOM #2447, a SiTU shape guard that fell back to
the unfused path — superseded by aiter #6078, which fixes the kernel instead —
and a closed ATOM grouped-top-k candidate, superseded by aiter #6131. The
bundle's opt-in `--experimental-mori` overlay (MegaMoE TDM tile choice, direct
EP route and counter reset on the mori path) is **not** part of this stack: it
has only EP4/operator evidence and a page-fault history.

Xiaobing's measurement log — profiles, the `ATOM_MORI_V2_FUSED` A/B, the BF16
tuning, rejected candidates — is `recipes/Kimi-K3-455-wideEP-optimization.md` on
his branch. It describes the image-based stack as of 2026-10-02.

### Getting the source stack

```bash
git clone git@github.com:ROCm/ATOM.git ATOM-base
git -C ATOM-base checkout --detach a526f0d557eeed08396670249af58bd85fa57338
git clone git@github.com:ROCm/aiter.git AITER-base
git -C AITER-base checkout --detach 57b7cf03ad32f72247c151cff2933abd11ff77d1

git clone --branch xiaobingsuper/kimi-k3-mi455-b0-recipe \
  git@github.com:ROCm/ATOM.git xb-recipe
BUNDLE=$PWD/xb-recipe/experiments/kimi_k3_b0/all_optimizations
"$BUNDLE/apply.sh" "$PWD/ATOM-base" "$PWD/AITER-base"
```

`apply.sh` refuses to run unless both checkouts sit at those bases and every
artifact matches its checksum. It then applies `atom.patch` (ATOM #2458 @
`9f20a70`, #2466 @ `b52cc83`, plus the five local changes) and `aiter.patch`
(#6078 @ `3e2807d`, #6115 @ `2ca6fff`, #6120 @ `0acb6d6`, #6121 @ `d6d8777`,
#6130 @ `4fa13d4`, #6131 @ `45a599a`; #6097 is already in the base), and
installs the BF16 GEMM table as
`aiter/configs/k3_bf16_hot_gfx1250_production_safe.csv`. Re-running is safe.
**Do not substitute a newer `main`** — the patches apply only to these bases.

⚠️ **Not yet recorded — settle these on the first run:**

- **Runtime image and install.** The bundle patches source trees; which image
  they were installed into, and how, is not written down. #6078 changes
  `csrc/`, so aiter's `module_activation` must be rebuilt from the patched
  tree — a stale build keeps the `d <= 16376` abort.
- **mori version.** The aiter base contains aiter #5810, which imports
  `TokOffExt` from `mori.ops.dispatch_combine_v2.hip_backend` — with no
  fallback — whenever `MORI_EP_TOKOFF_EXT` is on, which is the default. mori
  made that class public on 2026-09-24 (ROCm/mori#708), after the `20260918`
  image was built. Either run a mori that has it, or set
  `MORI_EP_TOKOFF_EXT=0`, which leaves dispatch serialised on one cco-window
  atomic. *Inferred from the code, not observed.*

  ```bash
  python3 -c "from mori.ops.dispatch_combine_v2.hip_backend import TokOffExt; print('ok')"
  ```

### Environment and CLI — delta from Config B

Everything not in this table is identical to
[Config B](#config-b--agentic-throughput-agentx): architecture, attention, MoE
(`MEGA_DISPATCH=mori`, `MEGA_DISPATCH_WIRE=fp4`, `ATOM_MORI_V2_FUSED=1`, …),
`ATOM_WO_A_USE_FLYDSL`, `ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE`,
`AITER_ROPE_TRITON_BACKEND` / `AITER_USE_SYSTEM_TRITON`, communication, session
affinity, loading, and every other CLI flag.

| | Config B (validated) | source stack | why |
|---|---|---|---|
| software | image + PR #2380 | pinned bases + bundle | [above](#getting-the-source-stack) |
| translation lines | set, inert on B0 | dropped | B0 only |
| `ATOM_UNFUSED_GATHER_KV_B_PROJ` | `1` | `0` | #2380 is not in the ATOM base, where the variable is not even defined; #6120 removes the need |
| `ATOM_USE_FLYDSL_GATHER_KV_B_PROJ` | unset (default `1`; the image's FlyDSL gather is gfx950-only, so a no-op) | `1` tier B, `0` tier A | switches #6121. The default is `1`, so tier A must set `0` explicitly |
| `ATOM_USE_TRITON_GEMM` | `1` | `0` | BF16 GEMMs take the tuned table, ptpc FP8 GEMMs take aiter's preshuffled kernels. Safe only with ATOM #2466 |
| `AITER_CONFIG_GEMM_BF16` | — | the installed CSV | 75 hot K3 BF16 shapes tuned on gfx1250, restricted to kernel ids in the default production build; in isolation 64/75 are ≥ 3% faster, median +58.5% |
| `--online_quant_config` | — | ptpc_fp8, see below | attention, dense MLP and shared experts to per-token FP8 — gfx1250's FP8 WMMA runs at roughly 4× BF16. #6078 and #6121 depend on it |

Run on every node, `DPRANK` = 0 / 4 / 8 / 12:

```bash
#!/bin/bash
# --- architecture ---
export PYTORCH_ROCM_ARCH=gfx1250 AITER_RUNTIME_GPU_ARCH=gfx1250
export GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0
# --- attention ---
export ATOM_USE_TRITON_MLA=1
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0          # CHANGED 1 -> 0: aiter #6120
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1       # NEW: aiter #6121 (tier B); 0 = tier A
export ATOM_USE_AITER_TRITON_ATTN=1 ATOM_USE_UNIFIED_ATTN=1
# --- MoE ---
export ATOM_MOE_GU_ITLV=1
export ATOM_USE_TRITON_MOE_DECODE=0
export MEGA_DISPATCH=mori MEGA_DISPATCH_WIRE=fp4
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1
export AITER_USE_GROUPED_GEMM=1 AITER_USE_OPUS_MOE_SORTING=1
# --- GEMM / quantization ---
export ATOM_USE_TRITON_GEMM=0 ATOM_WO_A_USE_FLYDSL=1   # CHANGED 1 -> 0: needs ATOM #2466
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export AITER_CONFIG_GEMM_BF16=<AITER-base>/aiter/configs/k3_bf16_hot_gfx1250_production_safe.csv  # NEW
export ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*.f_b_proj","*block_sparse_moe.gate","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'  # NEW
# --- communication ---
export NCCL_MNNVL_ENABLE=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1
export MORI_SOCKET_IFNAME=enp1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1
# --- agentic: pin a session to its cache owner across turns ---
export ATOM_DP_SESSION_AFFINITY=1
# --- loading ---
export HSA_XNACK=1 HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export ATOM_LOADER_USE_THREADPOOL=1 ATOM_LOADER_NUM_THREADS=4

cd /tmp
exec python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --trust-remote-code \
  -tp 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 4 \
  --data-parallel-rank ${DPRANK} \
  --data-parallel-master-ip <node0-data-plane-ip> \
  --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
  --enable-expert-parallel --enable-dp-attention \
  --fake-eplb \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --cudagraph-mode FULL \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.94 \
  --enable_prefix_caching \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --disable_uvicorn_access_log
```

`--fake-eplb` is for AgentX only: **drop it for the accuracy gate**, as for
Config A. The bundle's own `launch_server.sh` defaults to `FAKE_EPLB=1`, so run
it with `FAKE_EPLB=0` for accuracy. It also exports a few scheduler knobs
(`ATOM_DP_LB_REQ_EQUIV=512`, the prefill-delayer settings,
`--state-checkpoint-interval-tokens 8192`, …) at ATOM's own defaults — they are
not changes.

⚠️ **Do not use the bundle's `ptpc_online_experimental.json` as is.** Its MoE
patterns are written `*.mlp.gate`, `*.mlp.experts`, `*.mlp.routed_expert_*`,
but ATOM names K3's MoE block `block_sparse_moe` (`atom/models/kimi_k3.py`), and
the online `exclude_layer` list is matched with `fnmatch` against those module
names (`_matches_exclude` in `atom/config.py`). Under that rule the routed
experts and both routed projections are **not** excluded, and
`will_online_requant` / `FusedMoE._online_quant` would re-quantize them from
MXFP4 to FP8 — established by replaying the matching rule on K3's layer names,
not by booting it. The list above keeps the bundle's intent with K3's names:
the patterns from [Kimi-K3.md](Kimi-K3.md) plus the bundle's `*.f_b_proj`
(K = 128). `kv_b_proj` stays quantized on purpose, because #6121 needs it in
FP8.

### What "verified" means for this section

Run in this order and stop at the first failure:

1. **Boot.** On every rank: `Created MegaMoE ... experts=896 ... dispatch=mori
   wire=fp4` — MegaMoE requires MXFP4 experts, so this also confirms the
   exclusion list held — a positive `available_for_kv`, and the `TokOffExt`
   check above. ptpc re-quantizes weights during load, so allow a longer cold
   start than Config B's.
2. **Tier A — correctness reference**, `ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=0`, no
   `--fake-eplb`: the [sanity gate](#accuracy-gate--run-this-before-any-benchmark)
   (no `!`, no single repeated token, no empty `content`), then full 5-shot
   [GSM8K](#gsm8k). Reference points: Config A scored 0.9621; on earlier cuts
   of this stack Xiaobing measured strict 0.9583 (before #6121, #6078 and
   #6131 were added) and 0.9598 (ptpc only).
3. **Tier B — full stack**, `ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1`: the same gate
   again.
4. **AgentX**, with `--fake-eplb` and **this page's** client exactly (con32,
   `--benchmark-duration 1800`, warmup 3/lane, seed 42), against Config B's p90
   **2.53 tok/s/user** and **182 tok/s per GPU**. The bundle's `run_agentx.sh`
   runs 900 s; its 2.97 baseline is not comparable with 2.53.
5. **dmesg** before and after each run. Xiaobing's two 1800 s attempts on this
   rack each lost a rank to a gfx1250 `no-retry page fault`
   (`Faulty UTCL2 client ID: TCP`), on different nodes, with the image-based
   configuration — so it is a platform issue to watch, not a property of these
   PRs. A node that afterwards reports `MES ring buffer is full` or SDMA fence
   timeouts is not healthy even at 0% VRAM.

Then record the numbers here and drop "pending" — or strike whatever failed.

---

## Appendix A: the exact configuration this was validated on

Everything below was run as written on a 4-node gfx1250 rack. Copy it rather
than reassembling the deltas scattered through the sections above.

### The rack

| | |
|---|---|
| Silicon | **B0** (site information — no revision field on the board reports it) |
| Nodes | 4 × 4 GPUs = **16**, SPX / NPS1, **432 GiB HBM per GPU** |
| Image | `rocm/fw-bringup:gfx1250-atom-20260918-ep8` |
| **B0→A0 translation** | **not needed** — this image's kernels are built for B0 |
| Fabric | UALoE, one PPOD/VPOD, `accel_state=active` on all 16 |
| Data-plane NIC | `enp1s0f1` |
| Node → DP rank | `10.210.11.14`→0, `.25`→4, `.12`→8, `.18`→12 (node0 serves :8000) |

> On translation: the runs below still had `HSA_TOOLS_LIB` and the rocjitsu
> prefix set, and it did **nothing** — `outcome=translated: 0`, `reused tier: 0`
> across all 16 ranks, while GSM8K still scored 0.9621. On B0 silicon
> those three lines can simply be dropped. They are kept in the listing so the
> record matches what actually ran.

### Per-node setup, before either launch

```bash
R=$(ls -d /opt/rocm-10.1.0a* | head -1)
rm -rf /root/rjprefix && mkdir -p /root/rjprefix/lib /root/rjprefix/share
cp -a $R/lib/libhsa_hotswap_rocjitsu.so $R/lib/librocjitsu*.so* /root/rjprefix/lib/
cp -a $R/share/rocjitsu /root/rjprefix/share/            # inert on a B0 image

docker run -d --name k3ep16 \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host --pid=host \
  --group-add 39 --group-add 105 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  --shm-size=128g \
  -v /mnt/k3:/models:ro -v /root/k3ep16-logs:/logs \
  --entrypoint sleep rocm/fw-bringup:gfx1250-atom-20260918-ep8 infinity

docker cp /root/rjprefix k3ep16:/app/rjprefix
# PR #2380, pre-applied to THIS image's ATOM (the upstream diff does not apply)
docker cp atom_patched/model_ops/attention_mla.py      k3ep16:/app/ATOM/atom/model_ops/
docker cp atom_patched/model_ops/mla_unfused_gather.py k3ep16:/app/ATOM/atom/model_ops/
docker cp atom_patched/utils/envs.py                   k3ep16:/app/ATOM/atom/utils/
```

Then **verify by byte size** — see
[Check byte sizes](#-check-byte-sizes--import-passes-on-a-truncated-file).
Expected: `129112 / 8002216 / 7499 / 47967 / 146235`, and
`grep -c use_unfused_gather_kv_b_proj attention_mla.py` → 3.

### Config A — accuracy (GSM8K). Stock launch.

Identical to [Launch](#launch): `--max-num-batched-tokens 2048`,
`--gpu-memory-utilization 0.90`, `--no-enable_prefix_caching`, **no**
`--fake-eplb`, **no** `ATOM_DP_SESSION_AFFINITY`.

Measured on boot: `total_gpu=432.00GB utilization=0.90 budget=388.80GB
peak_torch=194.67GB available_for_kv=159.37GB`, `experts=56`,
`'max_model_len': None`. Cold start **under 6 minutes**.

Sanity gate output was ` Paris. The Eiffel Tower is located in Paris. ...` —
coherent, i.e. no translation needed. Single-stream TPOT 30.5 ms (32.8 tok/s).

```bash
lm_eval --model local-chat-completions \
  --apply_chat_template \
  --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,\
base_url=http://<node0>:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,\
max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,\
max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 \
  --output_path ~/lmeval/out --log_samples
```

**Result** — lm_eval 0.4.13, 5-shot, full 1319, **9 min 07 s** at
`num_concurrent=32`:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9613|±  |0.0053|
|     |       |strict-match    |     5|exact_match|↑  |0.9621|±  |0.0053|
```

### Config B — agentic throughput (AgentX)

Four deltas from Config A: `--max-num-batched-tokens 16384`,
`--gpu-memory-utilization 0.94`, `--enable_prefix_caching`, `--fake-eplb`,
plus `ATOM_DP_SESSION_AFFINITY=1`.

Run on every node, `DPRANK` = 0 / 4 / 8 / 12:

```bash
#!/bin/bash
# --- translation: inert on B0 silicon, kept for the record ---
export LD_LIBRARY_PATH=/app/rjprefix/lib:$LD_LIBRARY_PATH
export HSA_TOOLS_LIB=/app/rjprefix/lib/libhsa_hotswap_rocjitsu.so
export HSA_HOTSWAP_VERBOSE=1
# --- architecture ---
export PYTORCH_ROCM_ARCH=gfx1250 AITER_RUNTIME_GPU_ARCH=gfx1250
export GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0
# --- attention ---
export ATOM_USE_TRITON_MLA=1
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
export ATOM_UNFUSED_GATHER_KV_B_PROJ=1
export ATOM_USE_AITER_TRITON_ATTN=1 ATOM_USE_UNIFIED_ATTN=1
# --- MoE ---
export ATOM_MOE_GU_ITLV=1
export ATOM_USE_TRITON_MOE_DECODE=0
export MEGA_DISPATCH=mori MEGA_DISPATCH_WIRE=fp4   # MEGA_WIRE is dead, see below
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1
export AITER_USE_GROUPED_GEMM=1 AITER_USE_OPUS_MOE_SORTING=1
# --- GEMM / quantization ---
export ATOM_USE_TRITON_GEMM=1 ATOM_WO_A_USE_FLYDSL=1
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
# --- communication ---
export NCCL_MNNVL_ENABLE=1
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1
export MORI_SOCKET_IFNAME=enp1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1
# --- agentic: pin a session to its cache owner across turns ---
export ATOM_DP_SESSION_AFFINITY=1
# --- loading ---
export HSA_XNACK=1 HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export ATOM_LOADER_USE_THREADPOOL=1 ATOM_LOADER_NUM_THREADS=4

cd /tmp
exec python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --trust-remote-code \
  -tp 1 \
  --data-parallel-size 16 \
  --data-parallel-size-local 4 \
  --data-parallel-rank ${DPRANK} \
  --data-parallel-master-ip <node0-data-plane-ip> \
  --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
  --enable-expert-parallel --enable-dp-attention \
  --fake-eplb \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --cudagraph-mode FULL \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.94 \
  --enable_prefix_caching \
  --disable_uvicorn_access_log
```

Measured on boot: `peak_torch=236.08GB cudagraph_est=9.40GB
available_for_kv=118.80GB`, cold start ~2 min, TTFT on a 5-token prompt
**0.50 s** (against 2.73 s at 2048).

Client — aiperf 0.12.0, dataset cached beforehand with
`hf download --repo-type dataset semianalysisai/cc-traces-weka-062126`:

```bash
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://<node0>:8000 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model moonshotai/Kimi-K3 \
  --tokenizer moonshotai/Kimi-K3 --tokenizer-trust-remote-code \
  --apply-chat-template \
  --concurrency 32 --benchmark-duration 1800 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 3 --warmup-grace-period 1800 \
  --trace-idle-gap-cap-seconds 300 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir <artifacts>
```

⚠️ `--warmup-requests-per-lane` is the wall-clock lever. At **10** the warmup
took **2 h 52 min** (707 requests); at **3** it took **13.5 min** (130). It
only primes the cache, so lowering it does not change what is measured — but
it is not free either, since less priming means a colder cache at the start of
the profiling window. Do not change it between runs you intend to compare.

Note the repo layout moved: aiperf is now at `inferencex-e2e/utils/aiperf`
(not `utils/aiperf`), and `agentic-benchmark/requirements.txt` no longer
exists — `uv pip install -e inferencex-e2e/utils/aiperf` is enough. The pinned
submodule commit `754356e9` is already checked out by
`--recurse-submodules`. `py-spy` and `lm_eval` both need
`pip install --break-system-packages` inside the image.

**Result** — see [AgentX result](#agentx-result) for the full table.
Headline: interactivity **p90 2.53 tok/s/user**, **182 tok/s per GPU** total
(2913 aggregate / 16), prefix cache read **92.36%**.

---

## Appendix B: running the same image on A0 silicon

**Only read this if your parts are A0.** On B0 the whole layer is inert and
should be left out.

The image ships **B0** code objects. On A0 silicon they must be rewritten down
to A0 as each one loads — that is what `librocjitsu_gfx1250_b0_to_a0.so` does,
and what `libhsa_hotswap_rocjitsu.so` hooks into the HSA tool interface to
drive. Without it the GPU executes objects built for a different revision, and
the result is not degraded accuracy: **every generated token is `!`** and
**GSM8K is 0%**, with HTTP 200, `finish_reason: "stop"`, sane token counts and
plausible throughput. A benchmark runs to completion and produces a
clean-looking report built entirely from `!`.

> **`HOTSWAP=0` is not a fallback on A0.** Any accuracy or performance number
> from A0 silicon without translation is void.

### Which case am I in?

No revision field on the board reports the stepping — PCI `revision=0x00`,
`rocminfo` `ASIC Revision: 0(0x0)`, `amd-smi` `REV_ID: 0x00`, marketing name
`AMD Eng Sample`, and no IP-discovery line in `dmesg`. Decide from behaviour:

```bash
curl -sS http://<node0>:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is",
       "max_tokens":16,"temperature":0}' | python3 -m json.tool
```

| what you see | what it means |
|---|---|
| ` Paris. The Eiffel Tower is located in ...` | B0 — **no translation needed**, stop here |
| `!!!!!!!!!!!!!!!!` | A0 — install the prefix below, then re-test |

```bash
grep -c 'installed eager'     <log>   # hook attached (says nothing about translating)
grep -c 'outcome=translated'  <log>   # 0 = inert; several hundred = doing the work
```

On B0 this reads `installed eager` on every rank and `outcome=translated: 0`,
and the model is correct anyway — that is the expected B0 signature, not a
fault.

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

## Not yet measured

- AgentX above `--concurrency 32`. Effective concurrency of 15 suggests
  headroom, but a con64 attempt at 16384 was not completed.
- Any A/B of `--fake-eplb`, `ATOM_DP_SESSION_AFFINITY` or
  `--max-num-batched-tokens` against each other — each run costs a full
  warmup, so a baseline was never captured alongside them.
- Anything at EP32. At EP32 the per-GPU expert weight halves
  (14.64 GiB/layer ÷ 32 = 0.46, against 0.92 at EP16), freeing roughly 41 GiB
  per GPU for KV, which is the obvious lever for pushing concurrency up.
- `--max-num-seqs` has not been swept; see the note under
  [Throughput](#throughput) for why it is the first thing to try.
- The [optimized source stack](#optimized-source-stack--pending-verification)
  as a whole: boot, accuracy and AgentX are all pending.
