#!/usr/bin/env python3
"""Native Store allocation and data-path probe; no model weights or image edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = {'status': 'running', 'global_segment_size': config['global_segment_size'],
              'local_buffer_size': config['local_buffer_size'],
              'memlock': resource.getrlimit(resource.RLIMIT_MEMLOCK),
              'allocation_env': {key: os.environ.get(key) for key in
                  ('GLIBC_TUNABLES', 'MC_ENABLE_PARALLEL_REG_MR', 'MC_MAX_MR_SIZE')}}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    from mooncake.store import MooncakeDistributedStore
    # Match engine workers where HIP has been initialized before Store setup.
    import torch
    torch.cuda.init()
    store = MooncakeDistributedStore()
    started = time.monotonic()
    try:
        rc = store.setup(args.ip, config['metadata_server'], config['global_segment_size'],
                         config['local_buffer_size'], config['protocol'],
                         config['device_name'], config['master_server_address'])
        result['setup_rc'] = rc
        if rc != 0:
            raise RuntimeError(f'Native Store setup returned {rc}')
        key = 'atomesh-m31-preflight-' + uuid.uuid4().hex
        value = bytes(range(256)) * 4096
        rc = store.put(key, value)
        result['put_rc'] = rc
        if rc != 0:
            raise RuntimeError(f'Native Store PUT returned {rc}')
        received = store.get(key)
        if received != value:
            raise RuntimeError('Native Store GET mismatch')
        result.update(status='passed', roundtrip_bytes=len(value),
                      sha256=hashlib.sha256(received).hexdigest())
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        store.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
