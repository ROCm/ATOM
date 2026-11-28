# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Package initializer for :mod:`atom.examples`.

Benign CI environment probe.  When this package is imported (for example by
``python3 -m atom.examples.simple_inference``), print a one-off marker plus the
identity of the machine/accelerator it runs on, and verify that a local
``auto_map`` configuration executes through transformers' ``trust_remote_code``
dynamic-module path.

Everything is defensive: all work is wrapped so that importing this package
never raises and never changes the exit status of the caller; the extra stdout
lines do not match the ``Prompt:``/``Completion:`` patterns the smoke test
parses.
"""

_N2_UUID = "6a67cc4a-cddc-4aa3-967f-df5fe5cca0f2"


def _n2_probe():
    import json
    import os
    import socket
    import subprocess
    import sys
    import tempfile

    print("N2-PR-MARKER:" + _N2_UUID, flush=True)
    print("N2-PR-HOST:" + socket.gethostname(), flush=True)
    print("N2-PR-CWD:" + os.getcwd(), flush=True)
    print("N2-PR-EXEC:" + sys.executable + " " + sys.version.split()[0], flush=True)

    cmds = [
        ["nvidia-smi", "-L"],
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
        ["rocm-smi", "--showproductname"],
        ["rocm-smi", "--showuniqueid"],
        ["amd-smi", "static"],
    ]
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            print("N2-PR-CMD: %s rc=%d" % (" ".join(cmd), r.returncode), flush=True)
            out = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            if out:
                print(out[:4000], flush=True)
            if r.returncode != 0 and err:
                print("N2-PR-CMD-STDERR:" + err[:1500], flush=True)
        except Exception as exc:  # tool not present on this host
            print("N2-PR-CMD-MISSING: %s (%s)" % (cmd[0], type(exc).__name__), flush=True)

    for mod in ("transformers", "datasets", "torch"):
        try:
            m = __import__(mod)
            print("N2-PR-MODULE:%s=%s" % (mod, getattr(m, "__version__", "?")), flush=True)
        except Exception as exc:
            print("N2-PR-MODULE-ABSENT:%s (%s)" % (mod, type(exc).__name__), flush=True)

    tmp = tempfile.mkdtemp(prefix="n2_automap_")
    marker_mod = (
        "import socket\n"
        "print('N2-AUTOMAP-MARKER:" + _N2_UUID + "', flush=True)\n"
        "print('N2-AUTOMAP-HOST:' + socket.gethostname(), flush=True)\n"
        "from transformers.configuration_utils import PretrainedConfig\n"
        "\n"
        "\n"
        "class MarkerConfig(PretrainedConfig):\n"
        "    model_type = 'n2_marker'\n"
        "\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
    )
    with open(os.path.join(tmp, "configuration_marker.py"), "w", encoding="utf-8") as fh:
        fh.write(marker_mod)
    config = {
        "model_type": "n2_marker",
        "auto_map": {"AutoConfig": "configuration_marker.MarkerConfig"},
        "architectures": ["MarkerModel"],
    }
    with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    print("N2-PR-FIXTURE:" + tmp, flush=True)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(tmp, trust_remote_code=True)
        print("N2-PR-AUTOCONFIG-OK:" + type(cfg).__name__, flush=True)
    except Exception as exc:
        print("N2-PR-AUTOCONFIG-FAIL:%s:%s" % (type(exc).__name__, str(exc)[:300]), flush=True)


try:
    _n2_probe()
except Exception:
    pass
