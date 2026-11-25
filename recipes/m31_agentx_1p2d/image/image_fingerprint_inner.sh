#!/usr/bin/env bash
# Inside an image (GPU-less): print a content fingerprint of the aiter install so per-node images can be compared.
cd /app/aiter-test
echo "aiter_commit=$(cat AGX_AITER_COMMIT 2>/dev/null || git log -1 --format=%H)"
echo "so_count=$(ls aiter/jit/*.so | wc -l)"
echo "so_sha=$(sha256sum aiter/jit/*.so | sha256sum | cut -c1-16)"
echo "flydsl_cache_files=$(find aiter/jit/flydsl_cache -name '*.pkl' | wc -l) flydsl_aot_files=$(find aiter/jit/flydsl_aot -type f 2>/dev/null | wc -l)"
python3 -c "import importlib.metadata as m; print('versions', 'torch='+m.version('torch'), 'triton='+m.version('triton'), 'flydsl='+m.version('flydsl'), 'amd-aiter='+m.version('amd-aiter'))"
echo "mooncake_master=$(command -v mooncake_master) mooncake_client=$(command -v mooncake_client)"
python3 -c "import mooncake.store, mooncake.engine; print('mooncake imports ok')" 2>&1 | tail -1
