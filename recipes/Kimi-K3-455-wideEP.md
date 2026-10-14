# Kimi-K3 Wide-EP on gfx1250 (MI455) — dp8ep8 on two trays

Full-size **Kimi-K3** on **two gfx1250 B0 trays, 4 GPUs each (432 GiB/GPU, SPX)**:

- **Model:** 2.78T parameters. 93 layers: 24 full-attention MLA and 69 KDA linear. 896 routed experts, top-16.
  MXFP4 routed experts, BF16 everything else.
- **Parallelism:** `-tp 1`, DP 8, EP 8 (112 experts per GPU), DP attention.
- **Checkpoint:** about 1.42 TiB, needed **in full on every tray**.
- **Image:** `rocm/fw-bringup:gfx1250-atom-k3-20261014`. ATOM and aiter are at `main` of 2026-10-14 and mori is at a
  `main` commit. Two ATOM patches are applied, and the launch scripts are in `/opt/k3`. The ROCm, torch, Triton
  and FlyDSL stack comes from `rocm/fw-bringup-mlperf:gfx1250-atom-20261012`. See
  [Container and patches](#container-and-patches).

Validated on CT11 (node0) and CT10 of the HeliosM rack below, 2026-10-14.

| | |
|---|---|
| **GSM8K**, 5-shot, all 1319 | **strict-match 0.9659 ±0.0050**, flexible-extract 0.9651 ±0.0051 (flydsl compact dispatch, fp4 combine, 16384 batched tokens, no prefix caching) |
| **Throughput**, 14k/500, concurrency 32 | 477.9 output tok/s; median TTFT 7.7 s, median TPOT 50.5 ms (ATOM's standard serving benchmark, `/opt/k3/bench.sh`) |
| Gate | B0 text, decode matches prefill, chat `17 * 23` answers `391` |
| Prefix caching | the probe passes (`K3_MODE=agentic`) |
| Cold start | about 8 min in a fresh container (aiter JIT) |
| AgentX | not measured with this image yet; method in [Agentic](#agentic-agentx) |

Contents:

- [A0 and B0](#a0-and-b0-silicon)
- [The rack](#the-rack-hosts-and-addresses)
- [Tray operations](#tray-operations)
- [Container and patches](#container-and-patches)
- [Launch](#launch-dp8ep8)
- [Online FP8 quantization](#online-fp8-quantization-optional)
- [Accuracy](#accuracy-gate-and-gsm8k)
- [Agentic](#agentic-agentx)
- [Known issues](#known-issues)

---

## A0 and B0 silicon

**B0: nothing to do.** The image's code objects are B0.

**A0: translation is mandatory.** Without it the GPU runs B0 objects on A0, and **every generated token is
`!`**. GSM8K scores 0%, yet the server returns HTTP 200, `finish_reason: "stop"`, sane token counts and
plausible throughput. Any number from A0 without translation is void. This page's runs are all B0. The
image below ships its own rocjitsu in `_rocm_sdk_devel/lib`: the "image's own" column in the table below,
which translates nothing.

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

**Image:** `rocm/fw-bringup:gfx1250-atom-k3-20261014`, digest
`sha256:4da3dd983701f07caf2b3a2d24529b0f9fcb8e80c4df57323ddd2e96947e8d8d`. It is 150 GB unpacked.

**ATOM, aiter and mori are public `main` commits.** They are editable source checkouts in `/app`:

| | commit | on `main` |
|---|---|---|
| ATOM, `/app/ATOM` | `0adc6456` | `main` of 2026-10-14, plus the two patches below |
| aiter, `/app/aiter` | `1305f0c8` | `main` of 2026-10-14; aiter #5431 and #5844 are merged there |
| mori, `/app/mori` | `695d4c03` | `main` of 2026-10-10. This is the last commit before mori #745 (EPv2 1250x EP8), whose interface aiter `main` does not use yet (aiter #6430) |

The previous image, `rocm/fw-bringup:gfx1250-atom-k3-20261013`, ran ATOM, aiter and mori from the bring-up team's
private MLPerf branches. Use this one.

**Everything else comes from** the gfx1250 bring-up team's DeepSeek-R1 MLPerf image
`rocm/fw-bringup-mlperf:gfx1250-atom-20261012` (digest
`sha256:e7fcdb088de972009dcd02778eadd5552374a30a3edfd26a3b35218cb7df116d`). On top of it the image adds:
- **`/opt/k3`**: this page's launch (`run_server.sh`, `env_k3.sh`), the gate, GSM8K, benchmark and AgentX
  commands, and `verify_inside.sh`. It also holds `manifest.json`, which records the base digest, the
  commits and the patches;
- **aiperf** at `754356e9`, in `/opt/aiperf-venv` and on PATH;
- **one ENV default, `MORI_AMD_SMI_PATH`** (see [Known issues](#known-issues)).

Notes:
- **Private repository.** `docker login` with a Docker Hub account that can read `rocm/fw-bringup` (ask the
  gfx1250 bring-up team), then `docker pull`. A tray that already has the base image pulls only the top layers.
- **Stack** (from the base): ROCm 10.1.0a20260811 from TheRock's Python wheels (`ROCM_PATH` is
  `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core`), torch 2.11.0, Triton 3.8.0 (AMD-Triton
  `triton-mi450`, branch `shared/gfx1250`, in `/app/triton-mi450`), FlyDSL 0.3.3.dev886. aiter `main` pins FlyDSL
  0.3.4.1; the image keeps the base's FlyDSL, and this page's numbers are with it. System Python 3.12, no venv.
  py-spy is installed.
- **The base's ENV is set up for DeepSeek-R1.** The launch below overrides `NCCL_SOCKET_IFNAME=lo
  GLOO_SOCKET_IFNAME=lo` and `ATOM_MEGA_MASK_PAD_ROWS=1`. ATOM `main` does not read
  `ATOM_MLA_QH128_ASM_PAGE_SIZE=64`.
- **aiter JIT** builds its modules at first use into `/app/aiter/aiter/jit`, inside the container. A fresh
  container's first start takes about 8 min; restart the server in the same container to reuse them.

One container per tray:

```bash
sudo docker run -d --name k3dp8 \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
  --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g \
  -v <model-dir>:/models:ro -v <work>/logs:/logs \
  -v <work>/triton:/root/.triton -v <work>/cache:/root/.cache \
  --entrypoint sleep rocm/fw-bringup:gfx1250-atom-k3-20261014 infinity
```

`--network=host` is required: the ranks reach each other by data-plane IP. Between attempts, clear stale
segments with `rm -f /dev/shm/psm_* /dev/shm/nccl-*`. Do this only when no other container shares the host's
`/dev/shm`.

### Patches

| PR | what it fixes | in this image |
|---|---|---|
| [ATOM #2380](https://github.com/ROCm/ATOM/pull/2380) | unfused torch `gather_kv_b_proj`, enabled by `ATOM_UNFUSED_GATHER_KV_B_PROJ=1`. Without it, a chunked prefill aborts in an LLVM PHI assertion (fused Triton gather on gfx1250) | **patch 1** (the PR is closed, not merged) |
| [ATOM #2532](https://github.com/ROCm/ATOM/pull/2532) | with `--enable_prefix_caching`, the DP dummy decode advanced a live KDA state through a stale index and poisoned a rank's cached prefixes | **patch 1** |
| [ATOM #2533](https://github.com/ROCm/ATOM/pull/2533) | with `--enable_prefix_caching` and a large KV pool, the Triton MLA decode computed KV row offsets in int32 and decoded NaN past row 3,728,270 | **patch 1** |
| [ATOM #2522](https://github.com/ROCm/ATOM/pull/2522) | ATOM picks MegaMoE's dispatch kernel itself (`MEGA_DISPATCH` overrides the pick) and sizes the cco arena for the compact plan. Without it, ATOM `main` always runs mori's dispatch on the fp4 wire | **patch 2** |
| [aiter #5431](https://github.com/ROCm/aiter/pull/5431) | custom all-reduce over torch.symm_mem, for the allgather LM head; skips mori's shmem init on gfx125x, without which every rank aborts at MoE init across trays | in aiter `main` |
| [aiter #5844](https://github.com/ROCm/aiter/pull/5844) | the FlyDSL PS1 MLA decode (`AITER_MLA_DECODE_PS1_FLYDSL`) | in aiter `main` |

**Both patches are applied in this page's image.** `/opt/k3/verify_inside.sh` checks them. The patch files are
next to this page, in [`Kimi-K3-455-wideEP-patches/`](Kimi-K3-455-wideEP-patches/). Both are made for ATOM
`0adc6456` and leave out the PRs' tests and docs.
- **`atom-main-k3.diff`** (patch 1) carries #2380 at `dc83d9ef`, #2532 at `7bf8fc6b` and #2533 at `4229c757`.
- **`atom-main-2522.diff`** (patch 2) carries #2522 at `9cd096a2`. Apply it after patch 1.

To build the same tree from the base image instead, check out the three commits, rebuild mori, clear aiter's
JIT modules, then apply the patches. Run this in a container of `rocm/fw-bringup-mlperf:gfx1250-atom-20261012`
with network access:

```bash
cd /app/mori && git fetch -q https://github.com/ROCm/mori.git 695d4c03dc3dad56a61315eb3c47923028e20063 && git checkout -q --detach FETCH_HEAD
git submodule --quiet update --init --recursive -- $(git config -f .gitmodules --get-regexp '\.path$' | awk '{print $2}' | grep -v spdk)
python3 -m pip install --break-system-packages --no-build-isolation --no-deps -e .   # rebuilds mori's native libraries
cd /app/aiter && git fetch -q https://github.com/ROCm/aiter.git 1305f0c8bf78d1d1d435695f8ff6cbf20a70fc02 && git checkout -q --detach FETCH_HEAD
rm -rf aiter/jit/*.so aiter/jit/build      # modules JIT-built from the old sources
cd /app/ATOM && git fetch -q https://github.com/ROCm/ATOM.git 0adc6456c66ae186fb616c54e85479cf9267ff27 && git checkout -q --detach FETCH_HEAD
B=https://raw.githubusercontent.com/ROCm/ATOM/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP-patches
for p in atom-main-k3 atom-main-2522; do
  curl -sSfL $B/$p.diff -o /tmp/$p.diff
  git apply /tmp/$p.diff && git apply --check -R /tmp/$p.diff && echo "ok $p"; done
rm -rf /root/.cache/atom/*                 # after any code change
```

- `docker rm` discards changes made by hand. Those in the image stay.
- After launch, check two lines on every rank:
  - `Custom allreduce: using torch.symm_mem transport (mori backend)`. If the probe fails, aiter falls back
    to IPC without saying so, and `assert not ca_comm.disabled` comes back at the first decode.
  - `[MORI-V2] Created MegaMoE: … kernel=flydsl_compact (MEGA_DISPATCH) dispatch=flydsl stage1_fused=True
    wire=fp4 combine_quant=mxfp4 … cco_vmm_per_rank=7781MiB`. `dispatch=mori` although `MEGA_DISPATCH=flydsl`
    means patch 2 is missing.

---

## Launch (dp8ep8)

Both trays run the same launch inside their container, CT11 with `DPRANK=0` and CT10 with `DPRANK=4`. Only
node0 serves the API on `:8000`.
- **In the image.** It is `/opt/k3/run_server.sh <0|4> <node0 IP>`, which sources `/opt/k3/env_k3.sh`. The
  script below shows what it sets and runs. `env_k3.sh` also exports the names the bring-up team's MLPerf ATOM
  uses for the same switches (`ATOM_MEGA_STAGE1_FUSED`, `ATOM_MEGA_STAGE2_FUSED`, `ATOM_MLA_QH128_ASM_PAGE_SIZE`),
  so the same file serves that ATOM too. ATOM `main` ignores them.
- **Logs.** They go to `/logs/server_dp<rank>_<time>.log`, with a `server_dp<rank>.latest.log` link.
- **Knobs.** `K3_MODE=agentic` turns on prefix caching with session affinity. `K3_LMHEAD=safe` selects the
  replicated LM head.

```bash
#!/bin/bash
# run_server.sh <first DP rank: 0|4> <node0 data-plane IP>
DPRANK=$1 MASTER=$2
# --- architecture (B0: no translation lines)
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0
export ENABLE_CK=0
# --- attention
export ATOM_USE_TRITON_MLA=0
export AITER_MLA_DECODE_PS1_FLYDSL=1
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=0  # the unfused gather has no shuffled layout
export ATOM_UNFUSED_GATHER_KV_B_PROJ=1   # ATOM #2380, patch 1
export ATOM_USE_UNIFIED_ATTN=1
# --- MoE
export ATOM_MOE_GU_ITLV=1                # required on gfx1250
export ATOM_USE_TRITON_MOE_DECODE=0      # K3's activation is situ, not SiLU
export MEGA_DISPATCH_WIRE=fp4
export MEGA_DISPATCH=flydsl              # ATOM #2522, patch 2: flydsl dispatch with the compact plan; mori for an A/B
export ATOM_MEGA_COMBINE_WIRE=fp4        # prefill steps only; decode combines in bf16
export ATOM_MEGA_MASK_PAD_ROWS=0         # on by default in ATOM main (#2474); this page's numbers are with it off
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1   # aiter's fused MegaMoE transport
export AITER_USE_GROUPED_GEMM=1
export MORI_EP_TOKOFF_EXT=0              # required on more than one tray
export MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27   # see Known issues
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
# AgentX: prefix caching (#2532 + #2533, patch 1), sessions pinned to their cache owner
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
# from the jump host, both trays at once (CT11 = node0)
ssh root@10.210.11.26 "docker exec -d k3dp8 bash /opt/k3/run_server.sh 0 10.210.11.26"
ssh root@10.210.11.80 "docker exec -d k3dp8 bash /opt/k3/run_server.sh 4 10.210.11.26"
until curl -sf -m 5 http://10.210.11.26:8000/v1/models >/dev/null; do sleep 15; done
```

- **Expect** `peak_torch` 294 GB and `available_for_kv` about 56 GB per rank (55.8–56.8 GB on CT11).
- **0.90, not 0.94, at 16384 batched tokens.** At 0.94, aiter's grouped GEMM allocates its `grouped_out`
  transient (14 GiB at EP 8: received tokens × top-k rows × hidden, bf16) on the first long prefill, and it
  runs out of memory. The budget does not reserve it.
- **16384 rather than 2048.** A 14k prompt prefills in one step instead of seven. On an earlier stack it gave
  16% more output throughput and a 30–40% lower median TTFT.
- **The dispatch kernel** (ATOM #2522, patch 2). `MEGA_DISPATCH=flydsl` runs the flydsl dispatch with the
  compact plan, `mori` runs mori's, for an A/B. Unset, #2522 picks the compact plan for K3 on the fp4 wire at
  up to 16384 batched tokens, from its measured table.
  - Measured 2026-10-13 against mori at 14k/500, con32, 16384 batched tokens, on an earlier stack (ATOM `main`
    of 2026-10-08, aiter `e965ebfe`). The compact plan gave 2.6% more output throughput (279.2 against 272.2
    tok/s) and a 2.9% lower median TPOT (94.4 against 97.2 ms). Its median TTFT was 5% higher (9.0 against
    8.6 s).
  - Not re-measured on this image.
- **The fp4 combine wire** quantizes the combine on prefill steps only. On an earlier stack it was as accurate
  as bf16 (GSM8K 0.9621 against 0.9629) and changed no throughput.
- **Masking DP pad rows** (`ATOM_MEGA_MASK_PAD_ROWS`, ATOM #2474) is on by default in ATOM `main`: pad rows
  skip the MoE. Tested with K3 on 2026-10-13 under the compact plan: full GSM8K strict-match 0.9651, and GSM8K's
  requests took as long as without it. This page's runs keep it off.
- **The LM head.** The default allgather head (custom all-gather, #5431) hangs rarely; see
  [Known issues](#known-issues). The replicated head avoids it:
  `ATOM_DP_LM_HEAD_MODE=default ATOM_USE_CUSTOM_ALL_GATHER=0`, without `AITER_CUSTOM_AR_USE_SYMM_MEM`. That
  is also the only option at DP > 8, because aiter's gfx1250 custom all-reduce takes 2, 4, 6 or 8 ranks.
- **Do not set `--max-model-len`.** K3 needs the full generation budget; see
  [GSM8K](#accuracy-gate-and-gsm8k).
- **Cold start.** About 8 min in a fresh container, most of it aiter JIT builds.
- **Hangs.** When the server goes quiet, take `py-spy dump` of the `ATOM::DP<r>TP0` processes and rocgdb
  `info dispatches` **before** killing anything; see [Known issues](#known-issues).

---

## Online FP8 quantization (optional)

The launch above runs every non-MoE linear in BF16. With `--online_quant_config`, ATOM quantizes them to FP8 at
load time; the MXFP4 routed experts are unchanged. **Not validated on this page's image yet**; see
[Measured](#measured-online-fp8) for the stack it was measured on.

- **Linears:** 128x128 block FP8 (`per_block_fp8`) with **E8M0** (power-of-two) scales. On gfx1250, aiter's
  `gemm_a8w8_blockscale_bpreshuffle` takes the FlyDSL mxfp8_128 GEMM only when both the activation and the
  weight scale are E8M0. FP32 weight scales fall back to an untuned Triton GEMM that is slower than BF16.
- **MLA `kv_b_proj`:** per-token FP8 (`ptpc_fp8`), so aiter's FlyDSL `gather_kv_b_proj` can read it.

### PRs it needs

None of these is in this page's image.

| PR | why | status, 2026-10-14 |
|---|---|---|
| [ATOM #2554](https://github.com/ROCm/ATOM/pull/2554) | online `per_1x128` quantization writes E8M0 weight scales. K3's fused norm + quant producers emit FP32 scales, so they stand down when the consumer is E8M0. Pads N to 128: KDA `in_proj` 49376 -> 49408, `fused_qkv_a_proj` 2112 -> 2176 | open, draft |
| [ATOM #2466](https://github.com/ROCm/ATOM/pull/2466) | gfx1250 MXFP4 linears always take the Triton/Gluon GEMM. With `ATOM_USE_TRITON_GEMM=0` they otherwise take the ASM GEMM, whose scale layout needs ATOM #2505 + aiter #6294, and GSM8K showed runaway answers on that path | open |
| [aiter #6121](https://github.com/ROCm/aiter/pull/6121) | FlyDSL `gather_kv_b_proj` for the FP8 `kv_b_proj` | open |
| [aiter #6530](https://github.com/ROCm/aiter/pull/6530) | tuned table for K3's 8 blockscale shapes, M = 1..16384. Without it every call takes the untuned FlyDSL fallback, 2-36x slower at decode M | open |

### Launch changes

On top of [the launch](#launch-dp8ep8):

```bash
export ATOM_USE_TRITON_GEMM=0               # keeps the ptpc kv_b_proj preshuffled, which #6121 needs
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0      # #2380's unfused gather cannot read the preshuffled FP8 weight
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 # already set above
# ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE and ATOM_USE_FLYDSL_GATHER_KV_B_PROJ: leave at their default, 1
--online_quant_config '{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}'
```

Check that it took effect:
- With `AITER_LOG_LEVEL=WARNING`, a line `[gfx1250] gemm_a8w8_blockscale_bpreshuffle untuned M=…, N=…, K=…; falling
  back to flydsl kernel '…'` names a shape that the table does not cover.
- In a torch trace, the blockscale GEMMs are `gemm_a8w8_mx128_*` (FlyDSL). `_gemm_a8w8_blockscale_preshuffle_gfx1250*`
  (Triton) means the weight scales are FP32, i.e. #2554 is missing or `ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE` is off.

### Measured: online FP8

8k in / 1k out, concurrency 32, 128 prompts (`atom.benchmarks.benchmark_serving`, random, `--ignore-eos`), and
GSM8K 5-shot on all 1319 questions:

| | BF16 | online FP8 |
|---|---|---|
| output throughput, tok/s | 511.6 | **591.5** |
| mean TTFT, ms | 5874 | **2933** |
| mean TPOT, ms | 56.0 | **50.5** |
| GSM8K strict-match | 0.9644 ±0.0051 | 0.9598 ±0.0054 |

- Both on `rocm/atom-dev` 1009 (torch 2.14), two B0 trays without perf_setup, with that stack's launch: mori
  dispatch, bf16 combine, 2048 batched tokens, Triton MLA decode.
- BF16: ATOM `16cc652b` + #2380, aiter `6bf550b6` + #5431 + #6121 (`4b89b9d2`); CT2 + CT15, 2026-10-12.
- Online FP8: ATOM `main` `d67a79f4` + #2466 + #2554, aiter `main` `9c971f57` + #6121 (`7ce0a5de`) + a K3 table
  from the same tuning as #6530's; CT2 + CT3, 2026-10-14.
- The columns also differ in ATOM and aiter version and in one tray, so not all of the difference is the
  quantization.
- GSM8K: one question (doc 593) ran into a repeating answer under load. It answers correctly when sent alone.

---

## Accuracy gate and GSM8K

Before any benchmark, check what the model actually writes. The worst failures here are silent.

```bash
curl -sS http://10.210.11.26:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
# B0 with everything right: " Paris. The city is located in the northern part of the country. ..."   (all "!" = A0 untranslated)
```

Also compare decode against prefill on the same prompts at `temperature=0`. A chat `17 * 23` must answer
`content='391'`.

GSM8K, with `lm_eval` 0.4.13 (`pip install 'lm_eval[api]'`; without `[api]` it fails after loading the data):

```bash
lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,base_url=http://10.210.11.26:8000/v1/chat/completions,api_key=EMPTY,\
eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 --output_path <out> --log_samples
```

**`max_tokens=12288` is required.** K3 reasons first and puts the chain of thought in `reasoning_content`.
If `max_tokens` runs out mid-thought, `content` comes back empty with `finish_reason: length`. lm_eval reads
only `content`, so a small budget scores 0.

| server (dp8ep8, 0.90; CT11 + CT10, 2026-10-14) | strict-match | flexible-extract | time |
|---|---|---|---|
| flydsl compact dispatch, fp4 combine, 16384, no prefix caching | **0.9659 ±0.0050** | 0.9651 ±0.0051 | 6 min |

No answer was empty or `!!!!`. The previous image, with ATOM and aiter from the MLPerf branches, scored 0.9674.

---

## Agentic (AgentX)

**Method: ATOM CI's agentic benchmark, unchanged.** That is `.github/scripts/aiperf_agentic.sh` from ATOM
`main`, run the way `.github/scripts/atom_test.sh` runs it. It installs SemiAnalysis aiperf at the pinned
commit `754356e9` and replays `semianalysis_cc_traces_weka_062126`: real Claude Code sessions, ISL median
about 100k tokens, scenario `inferencex-agentx-mvp`. Defaults:

- 3600 s of profiling;
- 10 warm-up requests per lane;
- session headers on, because `SERVER_ARGS` contains `--enable-dp-attention`.

Server: the launch above in **AgentX mode**, with prefix caching (#2532 and #2533, in patch 1) and
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

### Results

Not measured with this image yet.

---


## Known issues

| symptom | cause and fix |
|---|---|
| Every token is `!`, service otherwise fine | A0 silicon without translation. See [A0 and B0](#a0-and-b0-silicon) |
| All ranks of a tray exit with code 255 at startup, without a traceback; the log has `[AMD-SMI] missing symbol amdsmi_init` | mori's topology loader opens `libamd_smi.so`. TheRock's wheels ship only `libamd_smi.so.27`, so it falls through to `librocm_smi64`, finds no `amdsmi_init` and calls `exit(-1)`. Set `MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27` |
| `[MORI-V2] Created MegaMoE: … dispatch=mori` although `MEGA_DISPATCH=flydsl` | Patch 2 (#2522) is missing: ATOM `main` alone always runs mori's dispatch on the fp4 wire |
| `RuntimeError: ccoMemAlloc failed: -1` at MoE init, after `Out of heap memory! Requested: 6222249984 bytes` | The compact plan runs, but the cco arena was sized for the token-major plan. Patch 2 sizes it |
| With `--enable_prefix_caching`: answers end in `!!!!…` up to `max_tokens` or come back empty, per rank or in bursts | #2532 and #2533 (patch 1) are missing. To check a server, send one prompt under two dozen session ids (`x-session-id`, `x-dynamo-session-id`) and look for `!!!!` |
| All DP ranks stop logging at once, mid-run, with no error. 30 min later all but one log `_sync_dp_state failed: … Timed out waiting 1800000ms`; meanwhile `[aiter] No available shared memory broadcast block found` repeats | The allgather LM head's custom all-gather hung on one rank. rocgdb shows that rank in `aiter::ag_gfx1250_lastdim<bf16, 8>` and the others in `tdm_compact_plan`. Seen twice in a few hours of GSM8K, never in AgentX runs. Workaround: the replicated head. Watch the `Engine 00x: Avg` lines, not the log's mtime: they stop |
| `RuntimeError: MORI_EP_TOKOFF_EXT: hipIpcOpenMemHandle(pe N) returned hipError 17` at load | aiter's MegaMoE turns on mori's TokOffExt, whose IPC handles do not cross trays. `MORI_EP_TOKOFF_EXT=0` |
| `assert not ca_comm.disabled` kills the ModelRunner while HTTP stays up | `ATOM_USE_CUSTOM_ALL_GATHER` and `AITER_CUSTOM_AR_USE_SYMM_MEM` are not set together, the symm_mem probe failed, or DP > 8. Use the replicated head otherwise |
| LLVM `PHINode::setIncomingValue` assertion on a long prompt at concurrency | Patch 1 (#2380) plus `ATOM_UNFUSED_GATHER_KV_B_PROJ=1` |
| `torch.OutOfMemoryError … Tried to allocate 14.05 GiB` in `_grouped_a8w4_tdm_moe` | 16384 batched tokens at 0.94. Use 0.90 |
| Ranks on the two trays cannot connect at startup | The image's ENV sets `NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`. Set `NCCL_`, `GLOO_` and `MORI_SOCKET_IFNAME=enp1s0f1` |
| `MEGA_WIRE was renamed to MEGA_DISPATCH_WIRE` | Drop `MEGA_WIRE` |
| MoE GUGU layout error / `ATOM_USE_TRITON_MOE_DECODE=1` asserts | `ATOM_MOE_GU_ITLV=1` / leave it 0 (K3 uses situ) |
| GSM8K near 0 with empty answers | `max_tokens` too small, or `--max-model-len` capped |
| `available KV budget (-NN GB)` or OOM with `0 bytes is free` at load | Someone else's job; see [Is the rack free](#is-the-rack-free) |
| Patch "applied" but behaves as if absent | File truncated by an interrupted `docker cp`; reverse-apply check, clear `/root/.cache/atom/*` |
| Startup stops at `load RCCL version`, CPU at ~110% | Intermittent `ncclCommInitRank` hang. Retry; if it reproduces every time, run the [fabric check](#fabric-check-ubench07) |
| Cross-tray bandwidth ~4.5 GB/s | `NCCL_MNNVL_ENABLE=1` missing |
| `/dev/kfd` missing after a reboot | Expected; run [perf_setup](#after-a-reboot-perf_setup-then-check-the-links) |
| `accel_state` stuck at `unconfigured` | `ifoe` not loaded when `amdgpu` initialised. One `rmmod` / `modprobe amdgpu` cycle |
| perf_setup says mode D, but `mtype_local` / `noretry` read `-1` | `amdgpu` was loaded before it ran. Reboot, then perf_setup first |
| Engines `<defunct>`, kfd list never drains, `MES … REMOVE_QUEUE` in dmesg | GPU wedged (`gpu_recovery=0` keeps it so). BMC `ForceRestart`, then perf_setup |
| A tray drops out; next boot's BERT says `event severity: fatal`, `Perr: CPU0` | Host CPU fatal error, not the engine |
| helicop FAILs `0x??` / `I2C read failed (bus 4, dev 0x20)` | Expected on this BMC firmware. Check Redfish `PowerState` and `boot_id` |
| Log looks frozen | tqdm writes `\r`; pipe through `tr '\r' '\n'` |
