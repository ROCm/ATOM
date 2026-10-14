"""Client-only import overlay for the pinned post-warmup await gate."""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

ACTIVE = False


class GateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "aiperf.timing.phase_orchestrator":
            return None
        original = importlib.machinery.PathFinder.find_spec(fullname, path)
        if original is None or original.origin is None:
            raise RuntimeError("cannot locate pinned AIPerf phase orchestrator")
        payload = Path(__file__).parent.parent
        records = json.loads((payload / "runtime-overlays.json").read_text())
        record = next(
            row
            for row in records
            if row["source"] == "aiperf/timing/phase_orchestrator.py"
        )
        actual = hashlib.sha256(Path(original.origin).read_bytes()).hexdigest()
        if actual != record["base_sha256"]:
            raise RuntimeError("AIPerf phase source is not the pinned base")
        overlay = Path(__file__).with_name("phase_orchestrator.py")
        if hashlib.sha256(overlay.read_bytes()).hexdigest() != record["after_sha256"]:
            raise RuntimeError("AIPerf lifecycle overlay hash mismatch")
        return importlib.util.spec_from_file_location(fullname, overlay)


if os.environ.get("PREFIX_READ_CLIENT_GATE") == "1":
    sys.meta_path.insert(0, GateFinder())
    ACTIVE = True
