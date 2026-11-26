"""Apply the pinned PP backport and native-listener candidate to the fixed image."""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path("/usr/local/lib/python3.12/dist-packages")
control = Path("/opt/atomesh-mori")
manifest = json.loads((control / "manifest.json").read_text())
source = root / manifest["module_dir"]
before = control / "before"
before.mkdir()
for name, expected in manifest["base_files"].items():
    path = source / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    shutil.copyfile(path, before / name)
patch = control / "pp-production.patch"
assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
for name, expected in manifest["upstream_patched_files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
subprocess.run(
    ["python3", str(control / "test_pp_contract.py"), str(before), str(source)],
    check=True,
)
subprocess.run(
    ["python3", str(control / "test_write_contract.py"), str(source)], check=True
)
port_before = control / "port-before.py"
shutil.copyfile(source / "moriio_common.py", port_before)
port_patch = control / "engine-port.patch"
assert (
    hashlib.sha256(port_patch.read_bytes()).hexdigest()
    == manifest["engine_port_patch_sha256"]
)
subprocess.run(["git", "apply", "--check", str(port_patch)], cwd=root, check=True)
subprocess.run(["git", "apply", str(port_patch)], cwd=root, check=True)
subprocess.run(
    [
        "python3",
        str(control / "test_engine_port_contract.py"),
        str(port_before),
        str(source / "moriio_common.py"),
    ],
    check=True,
)
connector = source / "moriio_connector.py"
text = connector.read_text()
needle = "        self.moriio_engine = IOEngine(\n"
assert text.count(needle) == 1
text = text.replace(
    needle,
    "        from atomesh_mori_probe import engine_event\n        engine_event(self, False)\n"
    + needle,
)
needle = "        self.moriio_wrapper.notify_port = self.moriio_config.notify_port\n"
assert text.count(needle) == 1
text = text.replace(needle, "        engine_event(self, True)\n" + needle)
ast.parse(text)
connector.write_text(text)
engine = source / "moriio_engine.py"
text = engine.read_text()
needle = "        self.worker.moriio_wrapper.waiting_for_transfer_complete(transfer_statuses)\n"
assert text.count(needle) == 1
text = text.replace(
    needle,
    needle
    + "        from atomesh_mori_probe import completed\n        completed(self.worker, transfer_id, transfer_statuses)\n",
)
text += "\nfrom atomesh_mori_probe import install as _install_mori_probe\n_install_mori_probe(MoRIIOWriter)\n"
ast.parse(text)
engine.write_text(text)
shutil.copyfile(control / "atomesh_mori_probe.py", root / "atomesh_mori_probe.py")
