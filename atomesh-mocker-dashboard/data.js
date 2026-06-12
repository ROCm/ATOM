window.BENCHMARK_DATA = {
  "lastUpdate": 1781247311473,
  "repoUrl": "https://github.com/ROCm/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "zhen.wan@amd.com",
            "name": "zhen wan",
            "username": "wanzhenchn"
          },
          "committer": {
            "email": "zhen.wan@amd.com",
            "name": "zhen wan",
            "username": "wanzhenchn"
          },
          "distinct": true,
          "id": "4514fd35f2351bd0e593c7e6e959e7b0ecb8eba2",
          "message": "[ci][mesh] Enable mocker dashboard publishing workflow to run on zwan/feat-mesh-ci pushes.",
          "timestamp": "2026-06-11T14:52:41Z",
          "tree_id": "db2a1e6de62f5a6b15da0097e271086b08f3e4fa",
          "url": "https://github.com/ROCm/ATOM/commit/4514fd35f2351bd0e593c7e6e959e7b0ecb8eba2"
        },
        "date": 1781193210078,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2254.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405726 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.43,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405726 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405726 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405726 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=405726 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7723.05,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390149 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.03,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390149 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.78,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390149 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.81,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390149 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390149 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3594.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646987 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646987 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646987 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646987 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=646987 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5251.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945204 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945204 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945204 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945204 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945204 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6760.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1216885 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.15,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1216885 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.11,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1216885 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1216885 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1216885 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2279.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410251 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410251 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410251 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410251 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=410251 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7807.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1405357 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.01,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1405357 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1405357 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1405357 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1405357 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3714.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=668676 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=668676 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=668676 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.82,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=668676 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=668676 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5531.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=995685 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=995685 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=995685 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=995685 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=995685 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6867.06,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1236071 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.13,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1236071 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.08,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1236071 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1236071 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1236071 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2294.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412946 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.42,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412946 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412946 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412946 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=412946 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7725.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390666 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390666 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390666 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390666 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1390666 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3732,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671760 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671760 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.71,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671760 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.82,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671760 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=671760 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5486.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=987511 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=987511 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.16,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=987511 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=987511 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=987511 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6882.19,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1238795 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.13,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1238795 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1238795 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1238795 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1238795 Run: https://github.com/ROCm/ATOM/actions/runs/27355714447"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zhen.wan@amd.com",
            "name": "zhen wan",
            "username": "wanzhenchn"
          },
          "committer": {
            "email": "zhen.wan@amd.com",
            "name": "zhen wan",
            "username": "wanzhenchn"
          },
          "distinct": true,
          "id": "cbac91e93dea215a6443ada83e550910365b6f38",
          "message": "Polish Atomesh mocker dashboard legends",
          "timestamp": "2026-06-12T05:50:39Z",
          "tree_id": "3c27becfc70e5738dc2f31a10dc65f5a584eb9c5",
          "url": "https://github.com/ROCm/ATOM/commit/cbac91e93dea215a6443ada83e550910365b6f38"
        },
        "date": 1781247304207,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2140.53,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385296 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385296 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385296 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385296 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=385296 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7503.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350710 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350710 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.9,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350710 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.98,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350710 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1350710 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3468.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624403 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624403 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624403 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.88,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624403 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=624403 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5254.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945778 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945778 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.21,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945778 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945778 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=945778 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6659.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 2.14,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.71,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1198702 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2074.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373372 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373372 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373372 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373372 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=373372 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7570.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1362622 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1362622 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.86,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1362622 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1362622 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1362622 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3386.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=609577 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=609577 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.78,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=609577 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.9,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=609577 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=609577 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5108.72,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919570 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919570 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.25,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919570 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.56,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919570 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=919570 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6628.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193217 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193217 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193217 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.72,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193217 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1193217 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2123.51,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382232 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382232 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382232 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382232 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=382232 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7253.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1305717 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1305717 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 4.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1305717 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 5.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1305717 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1305717 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3427.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616963 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.56,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616963 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616963 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616963 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=616963 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5232.74,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=941894 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=941894 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.22,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=941894 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=941894 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=941894 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6603.95,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188711 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.17,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188711 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 2.15,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188711 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188711 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1188711 Run: https://github.com/ROCm/ATOM/actions/runs/27397399598"
          }
        ]
      }
    ]
  }
}