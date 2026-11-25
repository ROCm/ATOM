# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""examples/run.sh: descriptions, judging, command construction and reports.

Nothing here touches Docker or a GPU; the runner is exercised through its
dry-run path, which builds every command a real run would execute.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from examples.harness import checks, cli, report
from examples.harness.manifest import (
    ATOMRL_DIR,
    EXAMPLES_DIR,
    TIERS,
    Checks,
    ManifestError,
    MetricCheck,
    load,
)
from examples.harness.runner import Options, Outcome, Runner
from examples.harness.site import Host, Rdma, Site, _split_rc, load_site

CATALOG = load()
RELEASE_TABLE = {"1", "2", "3", "4", "5", "6", "7", "9"}


def _site(n_hosts: int = 3) -> Site:
    hosts = [
        Host(f"h{i}", "local" if i == 0 else f"node{i}", f"192.0.2.{i + 1}")
        for i in range(n_hosts)
    ]
    return Site(
        data_root="/data",
        hosts=hosts,
        rdma=Rdma(interface="eth9", hca="mlx5_9", gid_index=3, env={}),
    )


def _log(step_lines: list[str], rc: int = 0, extra: str = "") -> str:
    body = "\n".join(step_lines)
    return f"RLTrainer.setup complete\n{body}\n{extra}\n=== LAUNCHER_EXIT={rc} ===\n"


def _metrics_line(step: int, **kv: float) -> str:
    pairs = " ".join(f"{k}={v}" for k, v in kv.items())
    return f"2026-11-17 [INFO] callbacks: step={step} {pairs} rollout_corr/kl=0.001"


# --------------------------------------------------------------- descriptions
def test_every_description_loads_and_ids_are_unique() -> None:
    ids = [ex.id for ex in CATALOG.examples]
    assert len(ids) == len(set(ids))
    assert {t for ex in CATALOG.examples for t in [ex.tier]} <= set(TIERS)
    for ex in CATALOG.examples:
        assert ex.image in CATALOG.images, ex.id
        for variant in filter(None, (ex.smoke, ex.full)):
            if variant.config:
                assert (ATOMRL_DIR / variant.config).is_file(), (ex.id, variant.config)


def test_release_table_numbers_map_to_judged_dapo_smokes() -> None:
    aliases = {a: ex for ex in CATALOG.examples for a in ex.aliases}
    assert RELEASE_TABLE <= set(aliases)
    for number in RELEASE_TABLE:
        ex = aliases[number]
        assert ex.tier == "t1" and ex.status == "validated", number
        names = {m.name for m in ex.smoke.checks.metrics}
        assert {"rollout_corr/k3_kl", "entropy", "rollout_corr/kl"} <= names, number
        assert ex.full is not None and ex.full.steps == 1000, number


def test_example_directories_all_carry_a_description() -> None:
    families = {ex.family.split("/")[0] for ex in CATALOG.examples}
    for d in EXAMPLES_DIR.iterdir():
        if d.is_dir() and d.name not in {"docs", "harness", "__pycache__"}:
            assert d.name in families, d.name


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "examples"
    (root / "X").mkdir(parents=True)
    (root / "X" / "examples.yaml").write_text(
        "images: {release: {ref: r, atomrl_mount: /a}}\n"
        "examples:\n"
        "  - {id: a, title: t, tier: t1, command: c, smoke: {stpes: 3}}\n"
    )
    with pytest.raises(ManifestError, match="stpes"):
        load(root)


def test_a_metric_needs_exactly_one_judgement(tmp_path: Path) -> None:
    root = tmp_path / "examples"
    root.mkdir()
    (root / "examples.yaml").write_text(
        "images: {release: {ref: r, atomrl_mount: /a}}\n"
        "examples:\n"
        "  - id: a\n    title: t\n    tier: t1\n    command: c\n"
        "    smoke: {checks: {metrics: [{name: m, ref: 1, rtol: 0.1, band: 10}]}}\n"
    )
    with pytest.raises(ManifestError, match="exactly one"):
        load(root)


# ------------------------------------------------------------------- judging
DAPO_CHECKS = Checks(
    metrics=(
        MetricCheck("rollout_corr/k3_kl", ref=0.001, rtol=0.3),
        MetricCheck("entropy", ref=0.6, rtol=0.25),
        MetricCheck("rollout_corr/ppl_ratio", info=True),
    )
)


def test_judge_passes_a_clean_run() -> None:
    text = _log(
        [
            _metrics_line(s, **{"rollout_corr/k3_kl": 0.0011, "entropy": 0.62})
            for s in (1, 2, 3)
        ]
    )
    v = checks.judge(text, DAPO_CHECKS, steps=3)
    assert v.status == "PASS", v.reasons
    assert [m.status for m in v.metrics] == ["PASS", "PASS", "INFO"]
    assert v.last_step == 3


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            _log(
                [_metrics_line(1, **{"rollout_corr/k3_kl": 0.002, "entropy": 0.6})], 0
            ),
            "k3_kl",
        ),
        (
            _log(
                [_metrics_line(1, **{"rollout_corr/k3_kl": 0.001, "entropy": 0.6})], 1
            ),
            "exit code 1",
        ),
        (
            _log([_metrics_line(2, **{"rollout_corr/k3_kl": 0.001, "entropy": 0.6})]),
            "step-1",
        ),
        (
            _log(
                [_metrics_line(1, **{"rollout_corr/k3_kl": 0.001, "entropy": 0.6})],
                extra="Traceback (x)",
            ),
            "Traceback",
        ),
        (
            _log(
                [_metrics_line(1, **{"rollout_corr/k3_kl": 0.001, "entropy": 0.6})],
                extra="bucket done - updated=10, skipped=96",
            ),
            "skipped tensors",
        ),
        ("step=1 rollout_corr/kl=0.1\n", "did not finish"),
    ],
)
def test_judge_fails(text: str, reason: str) -> None:
    v = checks.judge(text, DAPO_CHECKS, steps=1)
    assert v.status == "FAIL"
    assert any(reason in r for r in v.reasons), v.reasons


def test_judge_requires_the_final_step() -> None:
    text = _log([_metrics_line(1, **{"rollout_corr/k3_kl": 0.001, "entropy": 0.6})])
    v = checks.judge(text, DAPO_CHECKS, steps=3)
    assert any("final step 3" in r for r in v.reasons)


def test_metric_lines_take_the_first_metrics_line_of_a_step() -> None:
    text = "step=1 starting\nstep=1 a=1 rollout_corr/kl=0.5\nstep=1 a=2 rollout_corr/kl=0.6\n"
    assert checks.metric_lines(text, "rollout_corr/kl=")[1]["a"] == "1"


def test_band_and_bounds() -> None:
    band = MetricCheck("kl", ref=0.001, band=10)
    assert checks.judge_metric(band, "0.009").status == "PASS"
    assert checks.judge_metric(band, "0.02").status == "FAIL"
    assert checks.judge_metric(MetricCheck("loss", min=0.0), "nan").status == "FAIL"
    assert checks.judge_metric(MetricCheck("loss", min=0.0), "-1").status == "FAIL"


def test_pytest_sessions_are_judged_by_their_summary() -> None:
    c = Checks(pytest=True, forbid=(), weight_sync_buckets=False, last_step=False)
    ok = "....\n===== 999 passed, 6 skipped in 812.31s (0:13:32) =====\n=== LAUNCHER_EXIT=0 ===\n"
    bad = "F.\n===== 1 failed, 998 passed in 80.0s =====\n=== LAUNCHER_EXIT=1 ===\n"
    v = checks.judge(ok, c, None)
    assert v.status == "PASS" and v.counts["pytest_passed"] == 999
    assert checks.judge(bad, c, None).status == "FAIL"
    # -q prints the summary without the "=====" around it.
    quiet = "F.\n1 failed, 1071 passed, 13 skipped in 293.53s (0:04:53)\n=== LAUNCHER_EXIT=1 ===\n"
    v = checks.judge(quiet, c, None)
    assert v.counts["pytest_failed"] == 1 and any("1 failed" in r for r in v.reasons)


# ------------------------------------------------------- command construction
@pytest.mark.parametrize("ex", CATALOG.examples, ids=lambda e: e.id)
@pytest.mark.parametrize("full", [False, True])
def test_every_example_builds_its_commands(ex, full: bool) -> None:
    if full and ex.full is None:
        pytest.skip("no full variant")
    lines: list[str] = []
    runner = Runner(
        CATALOG, _site(), Options(dry_run=True, full=full), None, echo=lines.append
    )
    out = runner.run(ex)
    assert out.status == "SKIP" and out.reason == "dry run", out.reason
    text = "\n".join(lines)
    assert not re.search(r"\{[a-z_]+(\[\d\])?(\.[a-z_]+)?\}", text), text
    assert "LAUNCHER_EXIT" in text
    if ex.nodes > 1:
        assert (
            "RAY_ROLE=head" in text
            and "RAY_ROLE=worker" in text
            and "RAY_ROLE=driver" in text
        )


def test_dapo_command_carries_the_release_table_row() -> None:
    lines: list[str] = []
    Runner(CATALOG, _site(), Options(dry_run=True), None, echo=lines.append).run(
        CATALOG.get("4")
    )
    run = next(ln for ln in lines if ln.startswith("docker exec -d"))
    for flag in (
        "MODE=atomfp8",
        "TRAIN_FP8=1",
        "STEPS=3",
        "ATOM_FORCE_ATTN_TRITON=1",
        "CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml",
        "MODEL_PATH=/data/models/Qwen3-8B-Base",
    ):
        assert flag in run, flag
    assert "/opt/atomrl/ATOM/atomrl/examples/DAPO/run_dapo.sh" in run


def test_two_node_overrides_name_the_site_hosts_and_nic() -> None:
    site = _site()
    runner = Runner(CATALOG, site, Options(dry_run=True), None, echo=lambda s: None)
    ex = CATALOG.get("dapo-qwen3moe-fully-async-vllm-2node")
    ctx = runner.context(ex, ex.smoke, runner.image(ex.image), "/data/logs/x.log", 10)
    assert "controller.ray.actor.topology_tags.node_ip=192.0.2.1" in ctx["overrides"]
    assert "controller.ray.rollout.topology_tags.node_ip=192.0.2.2" in ctx["overrides"]
    assert "weight_sync.rdma.interface=eth9" in ctx["overrides"]
    env = runner.env(ex, ex.smoke, runner.image(ex.image), ctx, 10)
    assert (
        env["NCCL_SOCKET_IFNAME"] == "eth9"
        and env["EXTRA_OVERRIDE"] == ctx["overrides"]
    )


def test_full_runs_switch_wandb_off_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    runner = Runner(CATALOG, _site(), Options(full=True), None)
    ex = CATALOG.get("1")
    assert "logger.wandb_enabled=false" in runner.overrides(ex, ex.full).overrides


def test_examples_needing_more_hosts_skip() -> None:
    runner = Runner(CATALOG, _site(1), Options(), None)
    out = runner.run(CATALOG.get("dapo-qwen3moe-fully-async-vllm-2node"))
    assert out.status == "SKIP" and "needs 2 hosts" in out.reason


def test_image_references_follow_cli_then_site() -> None:
    site = _site()
    site.images = {"release": "site:tag"}
    assert Runner(CATALOG, site, Options(), None).image("release").ref == "site:tag"
    opts = Options(images={"release": "cli:tag"})
    assert Runner(CATALOG, site, opts, None).image("release").ref == "cli:tag"


@pytest.mark.parametrize(
    "script", ["DAPO/run_dapo.sh", "GRPO/run_grpo.sh", "GRPO/run_grpo_dsv4.sh"]
)
def test_launchers_start_ray_nodes_and_take_extra_overrides(script: str) -> None:
    text = (EXAMPLES_DIR / script).read_text(encoding="utf-8")
    assert 'case "${RAY_ROLE:-}" in' in text
    assert "ray start --head" in text and 'ray start --address="${HEAD_IP:?}' in text
    assert "${EXTRA_OVERRIDE:-}" in text


# ---------------------------------------------------------------------- site
def test_exit_status_comes_back_through_the_marker() -> None:
    # The trap prints "\n__ATOMRL_RC=<rc>\n" after whatever the command printed.
    r = _split_rc("hello\n\n__ATOMRL_RC=3\n", "", 0, "n")
    assert (r.rc, r.out) == (3, "hello\n")
    # A log chunk without a trailing newline must not swallow the marker.
    r = _split_rc("....... [ 19%]\n....\n__ATOMRL_RC=0\n", "", 0, "n")
    assert (r.rc, r.out) == (0, "....... [ 19%]\n....")
    assert _split_rc("hello\n", "", 0, "n").rc == 255
    assert _split_rc("hello\n", "", 7, "n").rc == 7


def test_remote_commands_report_status_on_exit() -> None:
    argv = Host("n", "node1", exec_template=("ssh", "{host}", "bash -c {qcmd}")).argv(
        "exit 3"
    )
    assert argv[:2] == ["ssh", "node1"]
    assert "trap" in argv[2] and "EXIT" in argv[2] and "exit 3" in argv[2]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_local_host_runs_directly() -> None:
    r = Host("me", "local").run("echo hi; exit 4")
    assert (r.rc, r.out) == (4, "hi\n")


def test_site_file(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(
        "data_root: /d\nhosts: [{name: a, address: local}, {name: b, address: x, ip: 1.2.3.4}]\n"
        "exec: [srun, '{host}', bash, -c, '{cmd}']\nrdma: {interface: ib0, hca: mlx5_1}\n"
    )
    site = load_site(path)
    assert site.data_root == "/d" and [h.name for h in site.hosts] == ["a", "b"]
    assert site.hosts[1].argv("true")[:2] == ["srun", "x"]
    assert site.rdma.nccl_env()["NCCL_IB_HCA"] == "mlx5_1"
    path.write_text("data_root: /d\nhost: []\n")
    with pytest.raises(ManifestError, match="host"):
        load_site(path)


def test_the_template_site_file_loads() -> None:
    site = load_site(EXAMPLES_DIR / "site.example.yaml")
    assert len(site.hosts) == 2 and site.rdma is not None


def test_site_container_prefix_can_be_scoped_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("ATOMRL_CONTAINER", "atomrl-ex-ci-123")
    assert load_site().container == "atomrl-ex-ci-123"


# ------------------------------------------------------------ suite, reports
def test_suite_defaults_to_t0_t1_t2_and_long_only_on_request(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_execute(catalog, site, args, selected, label):
        seen[label] = [ex.tier for ex in selected]
        return 0

    monkeypatch.setattr(cli, "_execute", fake_execute)
    monkeypatch.setattr(cli, "_site", lambda args: _site(1))
    assert cli.main(["suite", "--label", "default"]) == 0
    assert cli.main(["suite", "--label", "long", "--with-long"]) == 0
    assert set(seen["default"]) == {"t0", "t1", "t2"}
    assert "long" in seen["long"]
    assert seen["default"] == sorted(seen["default"], key=TIERS.index)


def test_reports(tmp_path: Path) -> None:
    ex = CATALOG.get("1")
    text = _log(
        [
            _metrics_line(s, **{"rollout_corr/k3_kl": 0.0011, "entropy": 0.62})
            for s in (1, 2, 3)
        ]
    )
    verdict = checks.judge(text, ex.smoke.checks, 3)
    good = Outcome(example=ex, status=verdict.status, verdict=verdict, duration_s=180)
    bad = Outcome(example=CATALOG.get("4"), status="ERROR", reason="image missing <x>")
    summary = report.write(tmp_path, {"label": "t", "started": "now"}, [good, bad])
    c = report.counts(summary)
    assert (c["PASS"], c["FAIL"], c["SKIP"], c["ERROR"]) == (1, 0, 0, 1)
    junit = (tmp_path / "junit.xml").read_text()
    assert 'tests="2"' in junit and "&lt;x&gt;" in junit
    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert loaded["results"][0]["step_metrics"]["1"]["entropy"] == "0.62"
    table = report.compare(loaded, loaded)
    assert "rollout_corr/k3_kl" in table and "+0.0%" in table


def test_known_issues_turn_failures_into_xfail(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "examples"
    root.mkdir()
    (root / "examples.yaml").write_text(
        "images: {release: {ref: r, atomrl_mount: /a}}\n"
        "examples:\n"
        "  - {id: a, title: t, tier: t1, command: c, known_issue: k,\n"
        "     smoke: {steps: 1}}\n"
    )
    catalog = load(root)
    ex = catalog.get("a")
    runner = Runner(catalog, _site(), Options(), None)
    monkeypatch.setattr(runner, "_run", lambda *a: setattr(a[-1], "status", "FAIL"))
    assert runner.run(ex).status == "XFAIL"
    monkeypatch.setattr(runner, "_run", lambda *a: setattr(a[-1], "status", "PASS"))
    assert runner.run(ex).status == "XPASS"


def test_rdma_examples_skip_without_the_site_nic() -> None:
    site = _site(1)
    site.rdma = None
    out = Runner(CATALOG, site, Options(), None).run(
        CATALOG.get("dapo-qwen3moe-vllm-fully-async")
    )
    assert out.status == "SKIP" and "rdma section" in out.reason


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_large_logs_come_back_as_an_excerpt(tmp_path: Path, monkeypatch) -> None:
    import examples.harness.runner as runner_mod

    monkeypatch.setattr(runner_mod, "FETCH_LIMIT", 1000)
    ex = CATALOG.get("1")
    log = tmp_path / "big.log"
    noise = "NCCL INFO channel 00/32 via NET/IB/0/GDRDMA\n" * 200
    metrics = _metrics_line(1, **{"rollout_corr/k3_kl": 0.0011, "entropy": 0.62})
    log.write_text(noise + metrics + "\n" + noise + "=== LAUNCHER_EXIT=0 ===\n")
    runner = Runner(CATALOG, _site(), Options(), tmp_path / "results")
    out = Outcome(example=ex, status="?")
    text = runner._fetch(
        Host("me", "local"), str(log), ex.id, "trainer.log", out, ex.smoke.checks
    )
    assert (
        "NCCL INFO" not in text and "LAUNCHER_EXIT=0" in text and "k3_kl=0.0011" in text
    )
    saved = (tmp_path / "results" / ex.id / "trainer.log").read_text()
    assert saved.startswith("# ") and "full log: me:" in saved
