# Kimi-K3 Wide-EP on gfx1250 (MI455) — dp8ep8 on two trays

Full-size **Kimi-K3** on **two gfx1250 B0 trays, 4 GPUs each (432 GiB/GPU, SPX)**:

- **Model:** 2.78T parameters. 93 layers: 24 full-attention MLA and 69 KDA linear. 896 routed experts, top-16.
  MXFP4 routed experts, BF16 everything else.
- **Parallelism:** `-tp 1`, DP 8, EP 8 (112 experts per GPU), DP attention.
- **Checkpoint:** about 1.42 TiB, needed **in full on every tray**.

Validated on CT2 + CT3 of the HeliosM rack below, 2026-10-10 and 10-11.

| | |
|---|---|
| **GSM8K**, 5-shot, all 1319 | **strict-match 0.9621 ±0.0053**, flexible-extract 0.9613 (flydsl dispatch, fp4 combine, 16384 batched tokens), with or without prefix caching (with it: #2532 + #2533) |
| **AgentX**, 4 concurrencies, 3600 s each | per GPU 123–447 tok/s, P90 interactivity 0.21–0.35 (E2E-normalized); see [Agentic](#agentic-agentx) |
| Cold start | about 3½ min with warm caches; 8 min on a tray's first run (aiter JIT) |

Contents:

- [A0 and B0](#a0-and-b0-silicon)
- [The rack](#the-rack-hosts-and-addresses)
- [Tray operations](#tray-operations)
- [Container and patches](#container-and-patches)
- [Launch](#launch-dp8ep8)
- [Accuracy](#accuracy-gate-and-gsm8k)
- [Agentic](#agentic-agentx)
- [Known issues](#known-issues)

---

## A0 and B0 silicon

**B0: nothing to do.** The image's code objects are B0.

**A0: translation is mandatory.** Without it the GPU runs B0 objects on A0, and **every generated token is
`!`**. GSM8K scores 0%, yet the server returns HTTP 200, `finish_reason: "stop"`, sane token counts and
plausible throughput. Any number from A0 without translation is void. This page's runs are all B0. The
atom-dev image below ships no rocjitsu; the A0 steps were validated with the gfx1250 bring-up images.

**Which one you have.** The board does not report the stepping: PCI `revision=0x00`, `rocminfo` `ASIC
Revision: 0`, `amd-smi` `REV_ID: 0x00`. Decide from behaviour, with the
[gate request](#accuracy-gate-and-gsm8k):

- ` Paris. The Eiffel Tower is …` means B0.
- `!!!!!!!!!!!!!!!!` means A0.

**The translator for A0** is a rocjitsu prefix:

- `libhsa_hotswap_rocjitsu.so` hooks the HSA tool interface;
- `librocjitsu_gfx1250_b0_to_a0.so` rewrites each B0 object to A0 as it loads.

A bring-up image's own rocjitsu installs and logs `installed eager gfx1250 B0-to-A0 hook`, then translates
nothing, because its translation store is empty. Use this prefix instead:

| | image's own | required prefix |
|---|---|---|
| `libhsa_hotswap_rocjitsu.so` | 143217 B | **129112 B** |
| `librocjitsu_gfx1250_b0_to_a0.so.0.3.0` | 7531385 B | **8002216 B** |
| translation store entries | 170 | **1976** |

Get it from `j07-01:/home/zejchen/rocmjit.zip` (424 MB; unpacked copy in `j07-01:/home/zejchen/rocmjit/`). Outside
that cluster, ask your AMD contact for the gfx1250 B0→A0 rocjitsu prefix by these sizes. Copy it into every
container. A `docker rm` loses it; `docker stop` and `start` keep it.

```bash
for n in <nodes>; do
  ssh "$n" 'cd ~ && [ -d rocmjit/share ] || unzip -q -o rocmjit.zip; sudo docker cp ~/rocmjit <container>:/app/rjprefix'   # zip in the NFS home
done
# inside the container, before launching -- the prefix's lib must be FIRST on LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/app/rjprefix/lib:$LD_LIBRARY_PATH
export HSA_TOOLS_LIB=/app/rjprefix/lib/libhsa_hotswap_rocjitsu.so
export HSA_HOTSWAP_VERBOSE=1
```

Verify on every node before trusting any output. The hook installing is not enough; the wrong prefix does
that too.

```bash
stat -c%s /app/rjprefix/lib/libhsa_hotswap_rocjitsu.so                  # 129112
stat -c%s /app/rjprefix/lib/librocjitsu_gfx1250_b0_to_a0.so.0.3.0       # 8002216
ls /app/rjprefix/share/rocjitsu/translations/gfx1250-b0-a0/v1 | wc -l   # 1976
grep -c 'outcome=translated' <server log>         # several hundred (0 on B0, which is fine there)
grep -c 'translation_status=[^0]' <server log>    # must be 0
```

Two traps:

- `librocjitsu_gfx1250_b0_to_a0.so.0` cannot be `HSA_TOOLS_LIB`: it has no `OnLoad`, so HSA ignores it
  silently.
- `HOTSWAP=0` is not a fallback on A0.

---

## The rack: hosts and addresses

One 18-tray HeliosM C-rack (`HELIOSM-DVT-CT1` … `CT18`, 4 × gfx1250 per tray), reached from a jump host.

| | |
|---|---|
| Jump host | `nj-4050-genamd-console`, `10.223.239.5`. Not a GPU node. `/home` is NFS-shared with every tray |
| Tray login | `root@<OS IP>`. CT1, CT2, CT3 and CT18 do not accept the shared SSH key; ask the rack owner for the root password |
| Data plane | `enp1s0f1` on every tray. Its address **is** the SSH address, and the one RCCL, mori and ubench use |
| Fabric | every tray checked reports the same `PPOD_ID` and `VPOD_ID 2` (`amd-smi fabric`) |
| UALink switch management | `10.210.11.62` … `.73`: 12 × `Helios_switch` (DS6010, Redfish) |

| Tray | OS / data-plane IP | BMC IP | `accel_id` |
|---|---|---|---|
| CT1 | 10.210.11.19 | 10.210.11.44 | 0–3 |
| CT2 | 10.210.11.24 | 10.210.11.47 | 4–7 |
| CT3 | 10.210.11.16 | 10.210.11.48 | 8–11 |
| CT4 | 10.210.11.11 | 10.210.11.49 | 12–15 |
| CT5 | 10.210.11.17 | 10.210.11.50 | 16–19 |
| CT6 | 10.210.11.15 | 10.210.11.51 | 20–23 |
| CT7 | 10.210.11.23 | 10.210.11.53 | 24–27 |
| CT8 | 10.210.11.21 | 10.210.11.55 | 28–31 |
| CT9 | 10.210.11.27 | 10.210.11.57 | 32–35 |
| CT10 | 10.210.11.80 | 10.210.11.60 | 36–39 |
| CT11 | 10.210.11.26 | 10.210.11.61 | 40–43 |
| CT12 | 10.210.11.22 | 10.210.11.59 | 44–47 |
| CT13 | 10.210.11.14 | 10.210.11.58 | 48–51 |
| CT14 | 10.210.11.25 | 10.210.11.56 | 52–55 |
| CT15 | 10.210.11.13 | 10.210.11.54 | 56–59 |
| CT16 | 10.210.11.12 | 10.210.11.52 | 60–63 |
| CT17 | 10.210.11.18 | 10.210.11.46 | 64–67 |
| CT18 | 10.210.11.10 | 10.210.11.45 | 68–71 |

- `accel_id` is 4 × (tray − 1) … +3. `amd-smi fabric` and the fabric's dmesg lines (`AccId:<n>`) use the same
  numbering.
- BMC addresses are DHCP leases. Re-check one before resetting a tray (see [Rebooting](#rebooting-a-tray)).
  CT4, CT7–CT10, CT13, CT16 and CT17 come from the rack's login sheet and were not re-checked; CT6 is inferred.
- Tray ownership changes daily. Check [Is the rack free](#is-the-rack-free) before using or rebooting a tray.

---

## Tray operations

### Is the rack free?

**VRAM is not a free/busy signal.** A job whose containers exist but whose weights have not loaded yet reads
0% on every GPU. Check the containers and the event log on every tray:

```bash
for n in <nodes>; do
  printf '%-14s ' "$n"
  ssh -q "$n" "docker ps --format '{{.Names}}' | tr '\n' ' '; \
    docker events --since 30m --until 0s --filter event=create --format '{{.Actor.Attributes.name}}' | tr '\n' ' '"
  echo
done
```

Starting on top of someone else shows up minutes later as a negative KV budget (`exceeds available KV budget
(-17.63GB)`). That means someone else's weights are loading. It is not something to fix with
`--gpu-memory-utilization`.

### After a reboot: perf_setup, then check the links

`amdgpu` is blacklisted on the kernel command line, so after any reboot `/dev/kfd` does not exist. Load it
through **`perf_setup_BPC21.sh`**, the platform's tuning script (AMD-internal; ask your AMD contact), not
with a bare `modprobe`. The script:

- loads `amdgpu` with menu **D**: `gpu_recovery=0 halt_if_hws_hang=1 mtype_local=0 noretry=1`;
- applies tuning that every reboot discards: −82 mV voltage offset, CSC DVO 25 mV, clocks, governor and more.

Part of that tuning only works while `amdgpu` is unloaded, so the order is fixed: **reboot → perf_setup →
anything else**. If `amdgpu` is already loaded, the report still says mode D but the parameters stay at
their defaults (`mtype_local=-1`); only another reboot fixes that.

The script reads its prompts from `/dev/tty`, so drive it with `expect` and match each prompt exactly:

```bash
# as root on the tray, from /root
lsmod | grep -c '^amdgpu '     # must print 0 -- otherwise reboot first
lsmod | grep '^ifoe '          # must be loaded
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

It takes 2–7 min and needs to reach AMD-internal hosts. Verify it:

```bash
tail -32 $(ls -1t /opt/amd-apps/perf_setup_BPC21_report_*.log | head -1)   # Perf steps: 8 passed, 0 failed
for p in mtype_local noretry gpu_recovery halt_if_hws_hang; do echo "$p=$(cat /sys/module/amdgpu/parameters/$p)"; done   # 0 1 0 1
```

`Result: [PARTIAL]` is fine when the only failure is the TBP read: it logs into the BMC with a default login,
which several trays reject. A tray is tuned **for this boot** only if both of these hold:

- the newest report is newer than `uptime -s`;
- `mtype_local=0` and `noretry=1`.

Without the script, `sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1` loads the driver with the two
bring-up parameters and none of the tuning. Those two parameters make a hung GPU stop and stay diagnosable
instead of being reset silently.

**Then wait for the links.** UALink trains for tens of seconds after the driver loads. Every card must read
`active` before anything multi-node, and once it does, do not `modprobe` again (that retrains the links):

```bash
cat /sys/class/drm/card*/device/ualink/accel_state              # all: active
cat /sys/class/drm/card*/device/ualink/station_lane_en_bitmap   # non-zero, identical across cards
```

If a card is still `unconfigured` a minute later, confirm `ifoe` is loaded and do one `rmmod amdgpu` /
`modprobe amdgpu` cycle. Left alone it stays `unconfigured` for good; once it lasted 12 hours, after `ifoe`
had loaded late.

### Rebooting a tray

Reboot only a tray you have confirmed is free. Every container on it stops (`Exited (255)`) and stays down.
Record `boot_id` (`cat /proc/sys/kernel/random/boot_id`) first: a changed `boot_id` is the only reliable proof
of a reboot. Afterwards, run [perf_setup](#after-a-reboot-perf_setup-then-check-the-links).

| tray state | how | took here |
|---|---|---|
| OS healthy | `systemctl reboot --no-wall` | ~3 min |
| GPU wedged (`MES … failed to respond to msg=REMOVE_QUEUE`, `<defunct>` engines, kfd list never drains, `D`-state kworkers) or userland hung | BMC Redfish `ForceRestart` | ~2½ min |
| cold boot, or nothing above brought it back | helicop: AC-cycle the whole tray, BMC included, then power on | 7½–9 min to SSH |

Do not use `systemctl reboot` or Redfish `GracefulRestart` on a wedged GPU: shutdown blocks while unloading
`amdgpu`.

**BMC (OpenBMC, Redfish).** Credentials come from the rack owner. The BMC's own shell is `admin@<bmc-ip>`
on port **2200**; port 22 is proxied to the host. The Redfish UUID is a placeholder, so match MACs to make sure
the BMC belongs to the tray you mean:

```bash
B=<bmc-ip>
ipmitool lan print 1 | grep -E '^(IP Address|MAC Address) +:'      # on the tray: its BMC address and MAC
curl -sk -u "$BMC_CRED" https://$B/redfish/v1/Managers/bmc/EthernetInterfaces/eth0 | grep -o '"MACAddress": "[^"]*"'
curl -sk -u "$BMC_CRED" -X POST -H 'Content-Type: application/json' -d '{"ResetType":"ForceRestart"}' \
  https://$B/redfish/v1/Systems/system/Actions/ComputerSystem.Reset         # expect HTTP 200
OLD=<boot_id before>
until b=$(ssh -o ConnectTimeout=10 root@<tray-ip> cat /proc/sys/kernel/random/boot_id 2>/dev/null) \
      && [ -n "$b" ] && [ "$b" != "$OLD" ]; do sleep 10; done
```

If nothing answers after about 10 min, read the BMC event log:
`ipmitool -I lanplus -H <bmc-ip> -U … -P … sel elist last 20`.

At boot:

- `Machine Check: 0 Bank 62` with a BERT record for MSR `0xc00023e1` is noise after any warm reboot.
- BERT `event severity: fatal`, `Perr: CPU0` is a real host crash. It was seen on CT1, CT2, CT3, CT12 and
  CT15 (twice in 30 min on CT15). Keep a tray that repeats it out of long runs.

**helicop: AC cycle, then power on.** helicop is the Helios bring-up team's controller (AMD-internal). It
drives a tray's host and BMC over SSH; validated with v0.39.

Setup:

- Copy the repo in as an archive; the jump host cannot reach its GitLab.
- `rack/MissionM.rack` describes this rack: `CTnn root:<pw>@<OS IP> admin:<pw>@<BMC IP>:2200`. Comment out
  its `jump` line when running on the jump host.
- Put the trays you need into `hosts.cfg` (`chmod 600`). The first field becomes an alias, so passwords stay
  off the command line: `./helicop CT03`.
- Keep two settings outside the helicop tree:

```bash
H=<helicop dir>; mkdir -p ~/hc/bin
sed -E 's/^(power_stage_poll_attempts *= *).*/\15/' $H/settings.cfg > ~/hc/settings.cfg   # the stage polls cannot succeed here
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

It takes two sessions per tray, because the AC cycle takes the BMC down too:

1. `ac_cycle`.
2. Wait for the BMC, plus 2 more minutes.
3. `prep_machine auto upto 13`. **`upto 13` is required**: step 19 loads `amdgpu`, after which `perf_setup`
   cannot apply its pre-driver tuning.

Driver for both sessions, run from the helicop directory:

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
T=CT03; TRAY=10.210.11.16; B=10.210.11.48
OLD=$(ssh root@$TRAY cat /proc/sys/kernel/random/boot_id); ssh root@$TRAY sync
expect -f ~/hc/hc.exp $T ac
sleep 30; until curl -sk -m 8 -o /dev/null https://$B/redfish/v1/; do sleep 10; done; sleep 120
expect -f ~/hc/hc.exp $T on
[ "$(ssh root@$TRAY cat /proc/sys/kernel/random/boot_id)" != "$OLD" ] && echo rebooted   # now perf_setup
```

On this BMC firmware helicop cannot read the CPLD power-stage register. Expect these, and judge the reboot
by Redfish `PowerState` and `boot_id` instead:

- FAILs at steps 06, 09 and 10;
- WARNs at 02, 03, 05, 07 and 08.

The driver answers `s` to each FAIL, except the `PerformDcPowerOn` suggestion at step 10, which needs `f`.

Pitfalls:

- Do not use `dc_power_off` plus `prep_machine` instead: CT2 stayed unreachable after it.
- If session 2 spins at step 01 (`waiting for BMC SSH … attempt n/250`), press Ctrl-C and start it again
  2 min later.
- Others run helicop against this rack too. Check `pgrep -af '[h]elicop'` first.

### Fabric check (ubench07)

Run this when a launch hangs at the rendezvous; a bad link hangs rather than errors.
`07_ualoe` drives UALink-over-Ethernet between two trays with HIP fabric handles:

```bash
wget http://dcgpuval-storage.amd.com/users/rexyap/MI450/Script/ubench-20260712.tar.gz   # AMD-internal
tar xzf ubench-20260712.tar.gz && cd ubench-20260712/07_ualoe && ./rebuild.sh gfx1250
./build/ualoe_p2p.exe export -port=55559 -gpu=0                              # tray A first
./build/ualoe_p2p.exe import -peer_ip=<trayA-ip> -port=55559 -gpu=0          # tray B: RESULT OK: 1/1 pairs PASS
./build/ualoe_bw.exe listen -port=55560                                      # tray A
./build/ualoe_bw.exe connect -peer_ip=<trayA-ip> -port=55560                 # tray B prints the table
```

Healthy on 2026-10-09 (4 GPU pairs, 1 GB transfers, perf_setup done): read 5.8, write 6.2, **bidirectional
10.9 TB/s**, with under 0.5% spread across seven pairs.

- About 4.5 GB/s means MNNVL is off and traffic fell back to TCP.
- Failures land in **dmesg on the importer**: `IMPORT: NPA-RSP timeout from remote AccId:<n>`,
  `LSDMA PIO error`.
- Kill leftover runs with `awk '$2 ~ /ualoe_(bw|p2p)\.exe$/'` on `ps` output. `pkill -f ualoe` would kill
  your own shell.

### Copying the checkpoint between trays

The checkpoint must be on every tray. Over the data plane it takes about 150 s (about 9 GB/s), with the jump
host's key forwarded (`ssh -A`):

```bash
# on the destination tray
SSH="ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no"
$SSH -n root@<src-ip> "ls -1A /mnt/k3/Kimi-K3" \
  | xargs -P 8 -I{} rsync -a --inplace -e "$SSH" "root@<src-ip>:/mnt/k3/Kimi-K3/{}" /data/k3/Kimi-K3/
sync
```

Then compare `find . -type f -printf '%P %s\n' | sort` on both sides.

---

## Container and patches

**Image:** `rocm/atom-dev:ub24-dcgpu-rk-10.2.0a20260922.bkc.20261002-pyt2.14.0-tri-main-atomsrc-gfx1250-1009`
(public), digest `sha256:961ec7e904aa4fcd7e1aad3067c4c0506be09ea91b6e573440878d5a12b41e29`.

- Stack: ROCm 10.2.0a20260922 (TheRock), torch 2.14.0, Triton 3.9.0 (upstream `main`), FlyDSL 0.3.4.1.
- ATOM, aiter and mori are `main` of 2026-10-08: `16cc652b` / `e965ebfe` / `066fd10e`.
- ATOM and aiter are **wheels** in the venv `/root/venv/atom-venv-gfx1250`. `docker exec` bypasses the
  entrypoint, so activate the venv yourself.
- aiter JIT-builds its modules at first use, so bind-mount the caches from the host.

One container per tray:

```bash
sudo docker run -d --name k3dp8 \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
  --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g \
  -v <model-dir>:/models:ro -v <work>/logs:/logs \
  -v <work>/aiter:/root/.aiter -v <work>/triton:/root/.triton -v <work>/cache:/root/.cache \
  --entrypoint sleep <image> infinity
```

`--network=host` is required: the ranks reach each other by data-plane IP. Between attempts, clear stale
segments with `rm -f /dev/shm/psm_* /dev/shm/nccl-*`. Do this only when no other container shares the host's
`/dev/shm`.

### Patches

| PR | needed for | what it fixes |
|---|---|---|
| [ATOM #2380](https://github.com/ROCm/ATOM/pull/2380) | **always** | unfused torch `gather_kv_b_proj`, enabled by `ATOM_UNFUSED_GATHER_KV_B_PROJ=1`. The env var does nothing without the patch. Without it, a chunked prefill aborts in an LLVM PHI assertion (fused Triton gather on gfx1250) |
| [aiter #5431](https://github.com/ROCm/aiter/pull/5431) | the allgather LM head (default) | custom all-reduce over torch.symm_mem (`AITER_CUSTOM_AR_USE_SYMM_MEM`). Without it, custom all-reduce is disabled on any group spanning trays, and the LM head's all-gather asserts `not ca_comm.disabled` at the first decode |
| [ATOM #2522](https://github.com/ROCm/ATOM/pull/2522) | `MEGA_DISPATCH=flydsl` | lets the flydsl dispatch run aiter's compact plan on the fp4/fp8 wire; this ATOM hard-wires mori there. `mori` stays the default |
| [ATOM #2532](https://github.com/ROCm/ATOM/pull/2532) | `--enable_prefix_caching` | the DP dummy decode advanced a live KDA state through a stale index. With prefix caching, that poisoned a rank's cached prefixes |
| [ATOM #2533](https://github.com/ROCm/ATOM/pull/2533) | `--enable_prefix_caching` (large, full KV pools) | the Triton MLA decode computed KV row offsets in int32. Rows ≥ 3,728,270 (block 233,016) read out of bounds and decoded NaN |

Apply the runtime parts into the venv, which is where the server imports them from. aiter's JIT reads its
`csrc` from `aiter_meta`.

```bash
# on the host
mkdir -p /tmp/k3patches
for pr in ATOM/pull/2380 aiter/pull/5431 ATOM/pull/2522 ATOM/pull/2532 ATOM/pull/2533; do
  curl -sSL https://github.com/ROCm/$pr.diff -o /tmp/k3patches/$(echo $pr | tr / _).diff; done
sudo docker cp /tmp/k3patches/. k3dp8:/tmp/

# in the container
SP=/root/venv/atom-venv-gfx1250/lib/python3.12/site-packages
cd $SP && for p in ATOM_pull_2380 ATOM_pull_2522 ATOM_pull_2532 ATOM_pull_2533; do
  git apply --include='atom/*' /tmp/$p.diff || echo "FAILED $p"; done
cd $SP && git apply --include='aiter/*' /tmp/aiter_pull_5431.diff
cd $SP/aiter_meta && git apply --include='csrc/*' /tmp/aiter_pull_5431.diff
# verify: each must reverse-apply cleanly (this also catches a truncated docker cp)
cd $SP && for p in ATOM_pull_2380 ATOM_pull_2522 ATOM_pull_2532 ATOM_pull_2533; do
  git apply --check -R --include='atom/*' /tmp/$p.diff && echo "ok $p"; done
rm -rf /root/.cache/atom/*     # after any code change
```

- `--include` drops each PR's tests and docs.
- #2532 and #2533 are each a few lines (`gdn_attn.py`, `triton_mla.py`). If a newer revision of either no
  longer applies to this image, apply the change by hand.
- `docker rm` discards every patch.
- After launch, every rank must log `Custom allreduce: using torch.symm_mem transport (mori backend)`. If the
  probe fails, aiter falls back to IPC without saying so, and the assert comes back.

---

## Launch (dp8ep8)

Both trays run the same script inside their container, CT2 with `DPRANK=0` and CT3 with `DPRANK=4`. Only
node0 serves the API on `:8000`.

```bash
#!/bin/bash
# run_server.sh <first DP rank: 0|4> <node0 data-plane IP>
DPRANK=$1 MASTER=$2
source /root/venv/atom-venv-gfx1250/bin/activate
# --- architecture (B0: no translation lines)
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0
# --- attention
export ATOM_USE_TRITON_MLA=1             # unset: first decode SIGABRTs
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0  # the unfused gather has no shuffled layout
export ATOM_UNFUSED_GATHER_KV_B_PROJ=1   # ATOM #2380
export ATOM_USE_UNIFIED_ATTN=1
# --- MoE
export ATOM_MOE_GU_ITLV=1                # required on gfx1250
export ATOM_USE_TRITON_MOE_DECODE=0      # K3's activation is situ, not SiLU
export MEGA_DISPATCH=flydsl MEGA_DISPATCH_WIRE=fp4   # flydsl needs ATOM #2522; mori works without it
export ATOM_MEGA_COMBINE_WIRE=fp4        # prefill steps only; decode combines in bf16
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1
export AITER_USE_GROUPED_GEMM=1
export MORI_EP_TOKOFF_EXT=0              # required on more than one tray
# --- GEMM / quantization
export ATOM_USE_TRITON_GEMM=1
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
# --- communication
export NCCL_MNNVL_ENABLE=1               # off: silently falls back to TCP
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=enp1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1   # the image says lo
export ATOM_DP_LM_HEAD_MODE=allgather ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1   # aiter #5431
# --- loading
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 ATOM_LOADER_NUM_THREADS=4
export HSA_XNACK=$([ "$(cat /sys/module/amdgpu/parameters/noretry)" = 1 ] && echo 0 || echo 1)   # follow the driver
export TRITON_CACHE_DIR=/root/.triton/cache

MODE_ARGS=(--max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 --no-enable_prefix_caching)   # accuracy
# AgentX: prefix caching (needs #2532 + #2533), sessions pinned to their cache owner
# export ATOM_DP_SESSION_AFFINITY=1
# MODE_ARGS=(--max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 --enable_prefix_caching)

cd /tmp
exec python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code \
  -tp 1 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-rank "$DPRANK" \
  --data-parallel-master-ip "$MASTER" --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
  --enable-expert-parallel --enable-dp-attention \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 --cudagraph-mode FULL --max-num-seqs 8 \
  "${MODE_ARGS[@]}" --disable_uvicorn_access_log
```

```bash
# from the jump host, both trays at once (CT2 = node0)
ssh root@10.210.11.24 "docker exec -d k3dp8 bash /logs/run_server.sh 0 10.210.11.24"
ssh root@10.210.11.16 "docker exec -d k3dp8 bash /logs/run_server.sh 4 10.210.11.24"
until curl -sf -m 5 http://10.210.11.24:8000/v1/models >/dev/null; do sleep 15; done
```

- **0.90, not 0.94, at 16384 batched tokens.** At 0.94, aiter's grouped GEMM allocates its `grouped_out`
  transient (14 GiB at EP 8: received tokens × top-k rows × hidden, bf16) on the first long prefill, and it
  runs out of memory. The budget does not reserve it.
- Expect `peak_torch` about 295 GB and `available_for_kv` 54–56 GB per rank.
- **16384 rather than 2048.** A 14k prompt prefills in one step instead of seven. Output throughput: +16% on
  flydsl, +11% on mori; TTFT median drops 30–40%.
- **flydsl vs mori dispatch.** flydsl (compact plan, #2522) gives 6% more output throughput than mori at 14k/500,
  con32, with 2048 batched tokens, and 10% more with 16384. To run without #2522, set `MEGA_DISPATCH=mori`.
- **The fp4 combine wire** is accurate, and gives no measurable throughput change at 14k/500.
- **The LM head.** The default allgather head (custom all-gather, #5431) hangs rarely; see
  [Known issues](#known-issues). The replicated head avoids it:
  `ATOM_DP_LM_HEAD_MODE=default ATOM_USE_CUSTOM_ALL_GATHER=0`, without `AITER_CUSTOM_AR_USE_SYMM_MEM`. That
  is also the only option at DP > 8, because aiter's gfx1250 custom all-reduce takes 2, 4, 6 or 8 ranks.
- **Do not set `--max-model-len`.** K3 needs the full generation budget; see
  [GSM8K](#accuracy-gate-and-gsm8k).
- **Cold start.** About 200 s with warm caches. The first run on a tray takes about 8 min, about 4 of them
  aiter JIT builds.
- **Hangs.** `py-spy` is not in the image (`pip install py-spy`). When the server goes quiet, take
  `py-spy dump` of the `ATOM::DP<r>TP0` processes and rocgdb `info dispatches` **before** killing anything;
  see [Known issues](#known-issues).

---

## Accuracy gate and GSM8K

Before any benchmark, check what the model actually writes. The worst failures here are silent.

```bash
curl -sS http://10.210.11.24:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
# B0 with everything right: " Paris. The Eiffel Tower is located in Paris. ..."   (all "!" = A0 untranslated)
```

Also compare decode against prefill on the same prompts at `temperature=0`. A chat `17 * 23` must answer
`content='391'`.

GSM8K, with `lm_eval` 0.4.13 (`pip install 'lm_eval[api]'`; without `[api]` it fails after loading the data):

```bash
lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,base_url=http://10.210.11.24:8000/v1/chat/completions,api_key=EMPTY,\
eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 --output_path <out> --log_samples
```

**`max_tokens=12288` is required.** K3 reasons first and puts the chain of thought in `reasoning_content`.
If `max_tokens` runs out mid-thought, `content` comes back empty with `finish_reason: length`. lm_eval reads
only `content`, so a small budget scores 0.

| server (dp8ep8, 0.90) | strict-match | flexible-extract | time |
|---|---|---|---|
| flydsl dispatch, fp4 combine, 16384, no prefix caching | **0.9621 ±0.0053** | 0.9613 | 7 min |
| the same with prefix caching (#2532 + #2533, replicated LM head) | **0.9621 ±0.0053** | 0.9613 | 12 min |
| mori dispatch, bf16 combine, 2048, no prefix caching | 0.9575 ±0.0056 | 0.9568 | 5½ min |

Without #2532 and #2533, the prefix-caching row drops to **0.8954**: 84 answers were empty or `!!!!` up to
`max_tokens`.

---

## Agentic (AgentX)

**Method: ATOM CI's agentic benchmark, unchanged.** That is `.github/scripts/aiperf_agentic.sh` from ATOM
`main`, run the way `.github/scripts/atom_test.sh` runs it. It installs SemiAnalysis aiperf at the pinned
commit `754356e9` and replays `semianalysis_cc_traces_weka_062126`: real Claude Code sessions, ISL median
about 100k tokens, scenario `inferencex-agentx-mvp`. Defaults:

- 3600 s of profiling;
- 10 warm-up requests per lane;
- session headers on, because `SERVER_ARGS` contains `--enable-dp-attention`.

Server: the launch above in **AgentX mode**, with prefix caching (#2532 and #2533 applied) and
`ATOM_DP_SESSION_AFFINITY=1`. No EPLB of any kind: neither `--fake-eplb` nor `--enable-eplb`. Start a fresh
server for every concurrency.

```bash
# inside node0's container, with ATOM main's .github/scripts/aiperf_agentic.sh copied to /logs
export MODEL_PATH=/models/Kimi-K3 AIPERF_MODEL=moonshotai/Kimi-K3 HF_HOME=/root/.cache/huggingface
export SERVER_ARGS="--model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code -tp 1 \
--data-parallel-size 8 --data-parallel-size-local 4 --enable-expert-parallel --enable-dp-attention \
--kv_cache_dtype fp8 --index-cache-dtype fp8 --cudagraph-mode FULL --max-num-seqs 8 \
--max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 --enable_prefix_caching --disable_uvicorn_access_log"
export AIPERF_DIR=/root/.cache/aiperf-src AIPERF_VENV=/root/.cache/aiperf-venv   # reuse the install between runs
CONC=32; OUT=/logs/agentic/c${CONC}_$(date +%m%d-%H%M%S); mkdir -p $OUT; cd /tmp
source /logs/aiperf_agentic.sh
agentic_prepare
run_aiperf_agentic http://127.0.0.1:8000 $CONC $OUT http://127.0.0.1:8000/metrics
```

**Validity.** aiperf discards a run unless its last first token or its last completed response lands in
the final 5% of the window (`ProfileMetricCoverageError`, `submission_valid=False`).

**Interactivity** follows ATOM's definitions (`.github/scripts/atomesh/interactivity.py`, per request, from
`profile_export.jsonl`):

- E2E-normalized, the dashboard default: `1 / p90(ITL + TTFT/OSL)`;
- also `1 / p90(ITL)`.

Neither is aiperf's own per-user p90.

### Results, dp8ep8 (2026-10-11)

flydsl dispatch, fp4 combine, 16384, 0.90, prefix caching, session affinity, no EPLB; default amdgpu
parameters (`HSA_XNACK=1`).

| concurrency | P90 interactivity, E2E-normalized | P90 interactivity, 1/p90(ITL) | throughput per GPU (total) | prefix cache hit | valid |
|---|---|---|---|---|---|
| 8 | 0.241 | 0.775 | 123.3 (987) tok/s | 96.65% | no: coverage TTFT 90.3%, ITL 94.6% |
| 16 | 0.237 | 0.531 | 155.4 (1243) tok/s | 89.85% | yes |
| 32 | **0.354** | 0.435 | 391.8 (3134) tok/s | 86.15% | yes |
| 48 | 0.213 | 0.286 | **447.2** (3577) tok/s | 74.23% | yes |

Throughput is aiperf's `total_token_throughput` (input plus output) over 8 GPUs. Prefix cache hit is cached
prompt tokens over prompt tokens.

| concurrency | warm-up | profiling: completed / cancelled, errors | TTFT p50 | ITL p50 |
|---|---|---|---|---|
| 8 | 87 requests, 10 min | 35 / 10, 0 | 192 s | 1.27 s |
| 16 | 177 requests, 27 min | 56 / 22, 0 | 96 s | 1.83 s |
| 32 | 354 requests, 31 min | 157 / 43, 0 | 21.5 s | 2.15 s |
| 48 | 531 requests, 59 min | 140 / 57, 0 | 42.5 s | 2.45 s |

- **These runs predate #2532 and #2533.** Some requests decoded garbage. aiperf does not read the text, so
  the numbers include whatever those requests cost. Re-measure with the patches.
- **Concurrency 8 is not a valid submission.** At 8 lanes, requests ran 465 s p50 and up to 2852 s.
  Nothing completed in the last 194 s of the window.
- **Warm-up grows with concurrency.** At 48 it took 59 min; 16 of those passed before the last cold primer
  (one long prefill per lane) returned.

---

## Known issues

| symptom | cause and fix |
|---|---|
| Every token is `!`, service otherwise fine | A0 silicon without translation. See [A0 and B0](#a0-and-b0-silicon) |
| With `--enable_prefix_caching`: answers end in `!!!!…` up to `max_tokens` or come back empty, per rank or in bursts | Apply [ATOM #2532](https://github.com/ROCm/ATOM/pull/2532) and [#2533](https://github.com/ROCm/ATOM/pull/2533). GSM8K goes from 0.8954 back to 0.9621. To check a server, send one prompt under two dozen session ids (`x-session-id`, `x-dynamo-session-id`) and look for `!!!!` |
| All DP ranks stop logging at once, mid-run, with no error. 30 min later all but one log `_sync_dp_state failed: … Timed out waiting 1800000ms`; meanwhile `[aiter] No available shared memory broadcast block found` repeats | The allgather LM head's custom all-gather hung on one rank. rocgdb shows that rank in `aiter::ag_gfx1250_lastdim<bf16, 8>` and the others in `tdm_compact_plan`. Seen twice in a few hours of GSM8K, never in the AgentX runs. Workaround: the replicated head. Watch the `Engine 00x: Avg` lines, not the log's mtime: they stop |
| `RuntimeError: MORI_EP_TOKOFF_EXT: hipIpcOpenMemHandle(pe N) returned hipError 17` at load | aiter's MegaMoE turns on mori's TokOffExt, whose IPC handles do not cross trays. `MORI_EP_TOKOFF_EXT=0` |
| `assert not ca_comm.disabled` kills the ModelRunner while HTTP stays up | aiter #5431 missing, or `ATOM_USE_CUSTOM_ALL_GATHER` / `AITER_CUSTOM_AR_USE_SYMM_MEM` not set together, or DP > 8. Use the replicated head otherwise |
| LLVM `PHINode::setIncomingValue` assertion on a long prompt at concurrency | ATOM #2380 plus `ATOM_UNFUSED_GATHER_KV_B_PROJ=1` |
| First decode request SIGABRTs, silently | `ATOM_USE_TRITON_MLA=1` not set |
| `torch.OutOfMemoryError … Tried to allocate 14.05 GiB` in `_grouped_a8w4_tdm_moe` | 16384 batched tokens at 0.94. Use 0.90 |
| Ranks on the two trays cannot connect at startup | the image's ENV sets `NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`. Set `NCCL_`, `GLOO_` and `MORI_SOCKET_IFNAME=enp1s0f1` |
| `MEGA_WIRE was renamed to MEGA_DISPATCH_WIRE` | drop `MEGA_WIRE` |
| MoE GUGU layout error / `ATOM_USE_TRITON_MOE_DECODE=1` asserts | `ATOM_MOE_GU_ITLV=1` / leave it 0 (K3 uses situ) |
| GSM8K near 0 with empty answers | `max_tokens` too small, or `--max-model-len` capped |
| `available KV budget (-NN GB)` or OOM with `0 bytes is free` at load | someone else's job; see [Is the rack free](#is-the-rack-free) |
| Patch "applied" but behaves as if absent | file truncated by an interrupted `docker cp`; reverse-apply check, clear `/root/.cache/atom/*` |
| Startup stops at `load RCCL version`, CPU at ~110% | intermittent `ncclCommInitRank` hang. Retry; if it reproduces every time, run the [fabric check](#fabric-check-ubench07) |
| Cross-tray bandwidth ~4.5 GB/s | `NCCL_MNNVL_ENABLE=1` missing |
| `/dev/kfd` missing after a reboot | expected; run [perf_setup](#after-a-reboot-perf_setup-then-check-the-links) |
| `accel_state` stuck at `unconfigured` | `ifoe` not loaded when `amdgpu` initialised. One `rmmod` / `modprobe amdgpu` cycle |
| perf_setup says mode D, but `mtype_local` / `noretry` read `-1` | `amdgpu` was loaded before it ran. Reboot, then perf_setup first |
| Engines `<defunct>`, kfd list never drains, `MES … REMOVE_QUEUE` in dmesg | GPU wedged (`gpu_recovery=0` keeps it so). BMC `ForceRestart`, then perf_setup |
| A tray drops out; next boot's BERT says `event severity: fatal`, `Perr: CPU0` | host CPU fatal error, not the engine |
| helicop FAILs `0x??` / `I2C read failed (bus 4, dev 0x20)` | expected on this BMC firmware. Check Redfish `PowerState` and `boot_id` |
| Log looks frozen | tqdm writes `\r`; pipe through `tr '\r' '\n'` |
