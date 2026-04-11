#!/usr/bin/env python3
"""
Multi-channel notification dispatcher for experiment events.

Supports: Slack, Discord, Telegram, ntfy, Pushover, generic webhook, local file log.
Configure via environment variables or notify_config.json.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

CONFIG_FILE = "notify_config.json"
DEFAULT_CONFIG = {
    "enabled_channels": ["file"],
    "slack_webhook_url": "",
    "discord_webhook_url": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "ntfy_topic": "",
    "ntfy_server": "https://ntfy.sh",
    "pushover_token": "",
    "pushover_user": "",
    "generic_webhook_url": "",
    "email_smtp_host": "",
    "email_smtp_port": 587,
    "email_from": "",
    "email_to": "",
    "email_password": "",
    "file_log_path": "notifications.log",
    "min_interval_seconds": 30,
    "quiet_hours": "",  # e.g. "23:00-07:00"
}

# Notification priority: events that should bypass quiet hours / rate limits
HIGH_PRIORITY_EVENTS = {
    "new_pareto_point",
    "all_experiments_done",
    "early_stop_suggested",
    "server_failed",
    "pr_created",
}


class Notifier:
    """Dispatches formatted notifications to multiple channels."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(".")
        self.config = dict(DEFAULT_CONFIG)
        self._load_config()
        self._last_send_time = 0.0

    def _load_config(self):
        env_overrides = {
            "NOTIFY_SLACK_WEBHOOK": "slack_webhook_url",
            "NOTIFY_DISCORD_WEBHOOK": "discord_webhook_url",
            "NOTIFY_TELEGRAM_TOKEN": "telegram_bot_token",
            "NOTIFY_TELEGRAM_CHAT": "telegram_chat_id",
            "NOTIFY_NTFY_TOPIC": "ntfy_topic",
            "NOTIFY_NTFY_SERVER": "ntfy_server",
            "NOTIFY_PUSHOVER_TOKEN": "pushover_token",
            "NOTIFY_PUSHOVER_USER": "pushover_user",
            "NOTIFY_WEBHOOK_URL": "generic_webhook_url",
            "NOTIFY_CHANNELS": "enabled_channels",
        }

        # Load from file
        cfg_path = self.config_dir / CONFIG_FILE
        if cfg_path.exists():
            try:
                file_cfg = json.loads(cfg_path.read_text())
                self.config.update(file_cfg)
            except Exception:
                pass

        # Env vars override file config
        for env_key, cfg_key in env_overrides.items():
            val = os.environ.get(env_key)
            if val:
                if cfg_key == "enabled_channels":
                    self.config[cfg_key] = [c.strip() for c in val.split(",")]
                else:
                    self.config[cfg_key] = val

    def save_default_config(self, path: Optional[str] = None):
        """Write a template config file for the user to fill in."""
        out = Path(path) if path else self.config_dir / CONFIG_FILE
        out.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        return str(out)

    # ── main dispatch ──────────────────────────────────────────

    def send(self, payload: dict):
        """
        Send a notification to all enabled channels.
        payload is the dict from ExperimentTracker.build_notification().
        """
        event_type = payload.get("event_type", "unknown")
        is_high = event_type in HIGH_PRIORITY_EVENTS

        if not is_high and not self._rate_ok():
            return

        text = self._format_text(payload)
        markdown = self._format_markdown(payload)

        for channel in self.config.get("enabled_channels", ["file"]):
            try:
                if channel == "slack":
                    self._send_slack(markdown)
                elif channel == "discord":
                    self._send_discord(markdown)
                elif channel == "telegram":
                    self._send_telegram(text)
                elif channel == "ntfy":
                    self._send_ntfy(payload, text)
                elif channel == "pushover":
                    self._send_pushover(payload, text)
                elif channel == "webhook":
                    self._send_webhook(payload)
                elif channel == "file":
                    self._send_file(text)
            except Exception as e:
                self._send_file(f"[NOTIFY ERROR] {channel}: {e}")

        self._last_send_time = time.time()

    def _rate_ok(self) -> bool:
        interval = self.config.get("min_interval_seconds", 30)
        return (time.time() - self._last_send_time) >= interval

    # ── formatters ─────────────────────────────────────────────

    def _format_text(self, p: dict) -> str:
        lines = [
            f"[ATOM Experiment] {p['event_type'].upper()}",
            f"Progress: {p['progress_pct']:.0f}% | Phase: {p['phase']}",
            f"Message: {p['message']}",
        ]
        if p.get("best_throughput"):
            lines.append(
                f"Best: {p['best_throughput']:.0f} tok/s, "
                f"TPOT {p['best_tpot']:.1f}ms"
            )
        if p.get("pareto_changed"):
            lines.append("** Pareto frontier updated! **")

        shift = p.get("shift", {})
        if shift and shift.get("shift") != "no_data":
            tp = shift.get("throughput_improvement_pct", 0)
            lines.append(f"Throughput shift: {tp:+.1f}%")

        lines.append(f"Next: {p.get('next_step', '?')}")

        if p.get("suggest_stop"):
            lines.append("!! SUGGEST STOPPING !!")
        lines.append(f"GPU hours: {p.get('gpu_hours', 0):.2f}h")
        return "\n".join(lines)

    def _format_markdown(self, p: dict) -> str:
        emoji = {
            "experiment_started": ":rocket:",
            "batch_completed": ":white_check_mark:",
            "new_pareto_point": ":star:",
            "best_refreshed": ":chart_with_upwards_trend:",
            "no_progress": ":warning:",
            "early_stop_suggested": ":octagonal_sign:",
            "all_experiments_done": ":trophy:",
            "pr_created": ":tada:",
        }.get(p["event_type"], ":information_source:")

        blocks = [
            f"{emoji} *ATOM Experiment — {p['event_type'].replace('_', ' ').title()}*",
            f"> {p['message']}",
            "",
            f"*Progress*: {p['progress_pct']:.0f}% | *Phase*: `{p['phase']}`",
        ]

        if p.get("best_throughput"):
            blocks.append(
                f"*Best*: {p['best_throughput']:.0f} tok/s | "
                f"TPOT {p['best_tpot']:.1f}ms"
            )

        shift = p.get("shift", {})
        if shift and shift.get("shift") != "no_data":
            tp = shift.get("throughput_improvement_pct", 0)
            blocks.append(f"*Throughput shift*: {tp:+.1f}%")

        if p.get("pareto_changed"):
            blocks.append(":star: *Pareto frontier updated*")

        blocks.append(f"*Next*: {p.get('next_step', '?')}")

        if p.get("suggest_stop"):
            blocks.append(":octagonal_sign: *Suggest stopping experiment*")

        return "\n".join(blocks)

    # ── channel implementations ────────────────────────────────

    def _post_json(self, url: str, data: dict, headers: Optional[dict] = None):
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status

    def _send_slack(self, markdown: str):
        url = self.config.get("slack_webhook_url")
        if not url:
            return
        self._post_json(url, {"text": markdown})

    def _send_discord(self, markdown: str):
        url = self.config.get("discord_webhook_url")
        if not url:
            return
        self._post_json(url, {"content": markdown[:2000]})

    def _send_telegram(self, text: str):
        token = self.config.get("telegram_bot_token")
        chat_id = self.config.get("telegram_chat_id")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._post_json(url, {"chat_id": chat_id, "text": text[:4096]})

    def _send_ntfy(self, payload: dict, text: str):
        topic = self.config.get("ntfy_topic")
        server = self.config.get("ntfy_server", "https://ntfy.sh")
        if not topic:
            return
        url = f"{server}/{topic}"
        is_high = payload.get("event_type") in HIGH_PRIORITY_EVENTS
        headers = {
            "Title": f"ATOM: {payload['event_type'].replace('_', ' ').title()}",
            "Priority": "high" if is_high else "default",
            "Tags": f"atom,{payload['event_type']}",
        }
        req = urllib.request.Request(
            url,
            data=text.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)

    def _send_pushover(self, payload: dict, text: str):
        token = self.config.get("pushover_token")
        user = self.config.get("pushover_user")
        if not token or not user:
            return
        is_high = payload.get("event_type") in HIGH_PRIORITY_EVENTS
        self._post_json(
            "https://api.pushover.net/1/messages.json",
            {
                "token": token,
                "user": user,
                "message": text[:1024],
                "title": "ATOM Experiment",
                "priority": 1 if is_high else 0,
            },
        )

    def _send_webhook(self, payload: dict):
        url = self.config.get("generic_webhook_url")
        if not url:
            return
        self._post_json(url, payload)

    def _send_file(self, text: str):
        log_path = self.config_dir / self.config.get(
            "file_log_path", "notifications.log"
        )
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n{'='*60}\n")
