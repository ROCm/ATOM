# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hardware probing from sysfs, model inspection, log diagnosis (no GPU)."""

import json
import os

import pytest

from atom.agent import hardware, models
from atom.agent.diagnose import analyze_log, clean, last_exception


@pytest.mark.parametrize(
    "version, name",
    [(90500, "gfx950"), (90402, "gfx942"), (90010, "gfx90a"), (120001, "gfx1201")],
)
def test_gfx_name(version, name):
    assert hardware.gfx_name(version) == name


def test_list_gpus_from_fake_sysfs(tmp_path, monkeypatch):
    topo = tmp_path / "topology"
    drm = tmp_path / "drm"
    procs = tmp_path / "proc"
    # Node 1 is a CPU; GPU nodes 3 and 2 are listed out of order on purpose:
    # the HIP index follows node order.
    nodes = {1: (0, 0, 0, 0), 2: (90500, 128, 0x7500, 11), 3: (90500, 136, 0x0500, 22)}
    for node, (ver, minor, loc, gpu_id) in nodes.items():
        d = topo / str(node)
        d.mkdir(parents=True)
        (d / "properties").write_text(
            f"simd_count {1024 if ver else 0}\ngfx_target_version {ver}\n"
            f"drm_render_minor {minor}\nlocation_id {loc}\ndomain 0\n"
        )
        (d / "gpu_id").write_text(str(gpu_id))
        if ver:
            dev = drm / f"renderD{minor}" / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(str(288 * 2**30))
            used = 283 if minor == 128 else 90_000
            (dev / "mem_info_vram_used").write_text(str(used * 2**20))
            (dev / "gpu_busy_percent").write_text("7")
    p = procs / str(os.getpid())
    p.mkdir(parents=True)
    (p / "vram_22").write_text(str(90_000 * 2**20))
    (p / "vram_11").write_text("0")
    monkeypatch.setattr(hardware, "KFD_TOPOLOGY", topo)
    monkeypatch.setattr(hardware, "DRM", drm)
    monkeypatch.setattr(hardware, "KFD_PROC", procs)
    monkeypatch.setattr(hardware, "_amd_smi_names", dict)
    gpus = hardware.list_gpus()
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].pci_bus == "0000:75:00.0" and gpus[0].free
    assert gpus[0].product == "AMD Instinct MI355X/MI350X"
    assert not gpus[1].free
    assert gpus[1].processes[0].pid == os.getpid()
    assert gpus[1].busy_percent == 7


def _write_model(path, cfg, nbytes=2048, index_total=None):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "model.safetensors").write_bytes(b"\0" * nbytes)
    if index_total:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": index_total}})
        )


def test_describe_config_nested_text_config():
    d = models.describe_config(
        {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "auto_map": {"AutoConfig": "x"},
            "text_config": {
                "num_hidden_layers": 61,
                "hidden_size": 7168,
                "num_attention_heads": 64,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
            },
        }
    )
    assert d["atom_supported"] and d["needs_trust_remote_code"]
    assert d["layers"] == 61 and d["moe_experts"] == 384 and d["experts_per_token"] == 8


def test_size_against_gpus_min_tp():
    gib = 2**30
    # 300 GiB: 150 GiB per GPU at TP2 fits 75% of 0.9 x 288 GiB; TP1 does not.
    s = models.size_against_gpus(300 * gib, 288 * 1024, {"model_type": "llama"})
    assert s["min_tp"] == 2
    assert not s["per_tp"][1]["fits"] and s["per_tp"][2]["fits"]


def test_inspect_and_find_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    _write_model(tmp_path / "data" / "Llama-Tiny", cfg, index_total=5 * 2**30)
    snap = tmp_path / "hub" / "models--org--Qwen3-Mini" / "snapshots" / "abc"
    _write_model(snap, {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
    (snap.parents[1] / "refs").mkdir()
    (snap.parents[1] / "refs" / "main").write_text("abc")
    _write_model(tmp_path / "data" / "bert", {"architectures": ["BertModel"]})

    info = models.inspect_model(str(tmp_path / "data" / "Llama-Tiny"), 288 * 1024)
    assert info["found"] and info["weights_gib"] == 5.0
    assert info["sizing"]["min_tp"] == 1
    assert models.resolve_model("org/Qwen3-Mini") == snap
    unsupported = models.inspect_model(str(tmp_path / "data" / "bert"))
    assert "warning" in unsupported

    found = models.find_models(roots=(str(tmp_path / "data"),))
    names = [m["name"] for m in found["models"]]
    assert names[:2] == ["Llama-Tiny", "org/Qwen3-Mini"]  # supported first
    assert names[-1] == "bert"
    assert [m["name"] for m in models.find_models("qwen3", roots=())["models"]] == [
        "org/Qwen3-Mini"
    ]
    # Bare directory names resolve through the search roots.
    assert models.resolve_model("Llama-Tiny", (str(tmp_path / "data"),)) == (
        tmp_path / "data" / "Llama-Tiny"
    )


HEALTHY = """\
[atom 16:19:48] Loading tokenizer from /m...
[atom 16:19:50] Initializing engine with model /m...
Loading safetensors shards[/m]:  10%|#  | 4/45 [00:01<00:10]\rLoading safetensors shards[/m]:  40%|#### | 18/45 [00:05<00:08]
[atom 16:21:47] [rank_0] Model load done: /m (weights loaded in 97.99s)
[atom 16:22:12] Model warmup done: /m
"""

CRASH = """\
[atom 03:44:30] Server started successfully and ready to accept requests
Traceback (most recent call last):
  File "x.py", line 1, in <module>
RuntimeError: boom
Memory access fault by GPU node-3 (Agent handle: 0x1) on address 0x7870. Reason: Unknown.
[atom 03:44:32] AsyncIOProcManager(ModelRunner): [ModelRunner1/4] proc died unexpectedly (exitcode=-6), shutting down.
"""


def test_analyze_log_phase_and_progress():
    loading = analyze_log(HEALTHY.split("[atom 16:21:47]")[0])
    assert loading["phase"] == "loading weights"
    assert loading["progress"] == "Loading safetensors shards 18/45"
    r = analyze_log(HEALTHY)
    assert r["phase"] == "warmup done"
    assert r["progress"] is None  # the shard counter belongs to an old phase
    assert r["findings"] == []
    assert "\r" not in clean(HEALTHY)


def test_analyze_log_cause_before_symptom():
    r = analyze_log(CRASH)
    keys = [f["key"] for f in r["findings"]]
    assert keys[0] == "gpu_fault"
    assert keys.index("worker_died") > keys.index("gpu_fault")
    assert keys[-1] == "traceback"  # non-fatal goes last
    assert last_exception(CRASH) == "RuntimeError: boom"


def test_analyze_log_kv_budget():
    r = analyze_log(
        "RuntimeError: Per-request cache tensor (0.47GB for 512 slots) exceeds "
        "available KV budget (-15.24GB) at --gpu-memory-utilization 0.04.\n"
        "proc died unexpectedly (exitcode=1)\n"
    )
    assert [f["key"] for f in r["findings"]][:2] == ["kv_cache_alloc", "worker_died"]


def test_analyze_log_argparse_error():
    r = analyze_log("openai_server.py: error: unrecognized arguments: --levle 3\n")
    assert r["findings"][0]["key"] == "bad_flag"


def test_partial_downloads_are_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    for name, present in (("Whole", 3), ("Partial", 1)):
        d = tmp_path / "data" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(cfg))
        shards = [f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)]
        (d / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 3 * 2**30},
                    "weight_map": {f"w{i}": s for i, s in enumerate(shards)},
                }
            )
        )
        for s in shards[:present]:
            (d / s).write_bytes(b"\0")
    # A HF-cache snapshot whose shard is a link to a blob that never arrived.
    snap = tmp_path / "hub" / "models--org--Linky" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(cfg))
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors"}})
    )
    (snap / "model.safetensors").symlink_to("../../blobs/missing")
    (snap / "extra.bin").write_bytes(b"\0")  # so the scan sees a weight file

    found = models.find_models(roots=(str(tmp_path / "data"),))
    by = {m["name"]: m for m in found["models"]}
    assert by["Whole"]["complete"] and not by["Partial"]["complete"]
    assert by["Partial"]["shards"] == "1/3" and not by["org/Linky"]["complete"]
    assert found["incomplete"] == 2 and found["total_found"] == 3
    assert next(m["name"] for m in found["models"]) == "Whole"  # complete first
    info = models.inspect_model(str(tmp_path / "data" / "Partial"))
    assert not info["complete"] and "incomplete checkpoint: 1 of 3" in info["warning"]
