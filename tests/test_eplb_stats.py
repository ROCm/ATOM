# SPDX-License-Identifier: MIT

import importlib

import torch


def _reload_eplb_stats_module():
    import atom.model_ops.eplb_stats as eplb_stats

    return importlib.reload(eplb_stats)


def test_record_and_step_updates_window_and_clears_pass(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_EPLB_LOAD_STATS", "1")
    monkeypatch.setenv("ATOM_EPLB_LOAD_WINDOW_SIZE", "3")
    monkeypatch.setenv("ATOM_EPLB_LOG_INTERVAL", "1000")
    monkeypatch.setenv("ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS", "0")

    eplb_stats = _reload_eplb_stats_module()
    monitor = eplb_stats.ExpertLoadMonitor()

    monitor.record_expert_load_pass("layer0", torch.tensor([2, 3], dtype=torch.int32))
    monitor.record_expert_load_pass("layer0", torch.tensor([1, 1], dtype=torch.int32))
    monitor.step()

    state = monitor._states["layer0"]  # internal state is expected in this unit test
    assert state.expert_load_window[0].tolist() == [3, 4]
    assert state.expert_load_pass.tolist() == [0, 0]
    assert state.window_step == 1


def test_auto_offline_rebalance_trigger_once(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_ENABLE_EPLB_LOAD_STATS", "1")
    monkeypatch.setenv("ATOM_EPLB_LOAD_WINDOW_SIZE", "4")
    monkeypatch.setenv("ATOM_EPLB_LOG_INTERVAL", "1000")
    monkeypatch.setenv("ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS", "2")

    eplb_stats = _reload_eplb_stats_module()
    monitor = eplb_stats.ExpertLoadMonitor()

    caplog.set_level("INFO", logger="atom")
    monitor.record_expert_load_pass("layerA", torch.tensor([1, 0, 2]))
    monitor.step()
    monitor.record_expert_load_pass("layerA", torch.tensor([2, 1, 0]))
    monitor.step()
    # One extra step should not emit the planning trigger again.
    monitor.record_expert_load_pass("layerA", torch.tensor([0, 1, 1]))
    monitor.step()

    trigger_logs = [
        r.message
        for r in caplog.records
        if "offline rebalance planning trigger" in r.message
    ]
    assert len(trigger_logs) == 1
    assert monitor._offline_rebalance_done is True


def test_dummy_steps_are_excluded_from_auto_trigger(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_ENABLE_EPLB_LOAD_STATS", "1")
    monkeypatch.setenv("ATOM_EPLB_LOAD_WINDOW_SIZE", "4")
    monkeypatch.setenv("ATOM_EPLB_LOG_INTERVAL", "1000")
    monkeypatch.setenv("ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS", "2")

    eplb_stats = _reload_eplb_stats_module()
    monitor = eplb_stats.ExpertLoadMonitor()

    caplog.set_level("INFO", logger="atom")
    monitor.record_expert_load_pass("layerA", torch.tensor([1, 0, 2]))
    monitor.step(is_dummy_run=True)
    monitor.record_expert_load_pass("layerA", torch.tensor([2, 1, 0]))
    monitor.step(is_dummy_run=False)
    monitor.record_expert_load_pass("layerA", torch.tensor([0, 1, 1]))
    monitor.step(is_dummy_run=False)

    trigger_logs = [
        r.message
        for r in caplog.records
        if "offline rebalance planning trigger" in r.message
    ]
    assert len(trigger_logs) == 1
    assert "real_step=2" in trigger_logs[0]
    assert monitor._offline_rebalance_done is True


def test_dummy_step_clears_pass_load_instead_of_leaking(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_EPLB_LOAD_STATS", "1")
    monkeypatch.setenv("ATOM_EPLB_LOAD_WINDOW_SIZE", "3")
    monkeypatch.setenv("ATOM_EPLB_LOG_INTERVAL", "1000")
    monkeypatch.setenv("ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS", "0")

    eplb_stats = _reload_eplb_stats_module()
    monitor = eplb_stats.ExpertLoadMonitor()

    monitor.record_expert_load_pass("layerA", torch.tensor([10, 0, 0]))
    monitor.step(is_dummy_run=True)

    monitor.record_expert_load_pass("layerA", torch.tensor([1, 2, 3]))
    monitor.step(is_dummy_run=False)

    state = monitor._states["layerA"]
    # If dummy pass leaked, this would become [11, 2, 3].
    assert state.expert_load_window[0].tolist() == [1, 2, 3]
    assert state.expert_load_pass.tolist() == [0, 0, 0]


def test_manual_offline_rebalance_trigger_sends_plan_log(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_ENABLE_EPLB_LOAD_STATS", "1")
    monkeypatch.setenv("ATOM_EPLB_LOAD_WINDOW_SIZE", "2")
    monkeypatch.setenv("ATOM_EPLB_LOG_INTERVAL", "1000")
    monkeypatch.setenv("ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS", "0")

    eplb_stats = _reload_eplb_stats_module()
    monitor = eplb_stats.ExpertLoadMonitor()

    monitor.record_expert_load_pass("layerB", torch.tensor([3, 1, 0]))
    monitor.step()

    caplog.set_level("INFO", logger="atom")
    monitor.trigger_offline_rebalance(reason="test")
    assert any(
        "offline rebalance planning trigger (test)" in r.message for r in caplog.records
    )
    assert any("offline plan layer=layerB" in r.message for r in caplog.records)
