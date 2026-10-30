"""Verify the KDA loading fix mounted over exactly two pinned Python files."""
import hashlib
import json
from pathlib import Path


def verify_quant_patch(vllm_root: Path) -> dict:
    gap = Path(__file__).parent / '../../benchmark/c48-model-gap'
    manifest = json.loads((gap / 'online-quant-fix/manifest.json').read_text())
    # -INPAD cases mount the same fix with 128-row in_proj padding (pd_c48_port_mount.sh).
    pad128 = json.loads((gap / 'inproj-pad128/manifest.json').read_text())
    kda = hashlib.sha256((vllm_root / pad128['files'][0]['relative_path']).read_bytes()).hexdigest()
    # -N3966- cases mount the INPAD fix ported onto vLLM 3966924d.
    for candidate in (pad128, json.loads((gap / 'n3966924d/inproj-pad128/manifest.json').read_text())):
        if kda == candidate['files'][0]['patched_sha256']:
            manifest = candidate
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    observed = []
    for record in manifest['files']:
        path = vllm_root / record['relative_path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['patched_sha256']:
            raise RuntimeError(f'C48 KDA loading fix missing: {path}, sha256={digest}')
        readonly = any(line.split()[4].replace('\\040', ' ') == str(path)
                       and 'ro' in line.split()[5].split(',') for line in mounts)
        if not readonly:
            raise RuntimeError(f'C48 KDA fix must be a read-only mount: {path}')
        observed.append(dict(record, observed_sha256=digest, readonly_mount_observed=readonly))
    return dict(manifest, files=observed, deployment='two_python_source_readonly_bind_mounts')
