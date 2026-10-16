import asyncio
import hashlib
import json
import statistics
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import aiohttp

MODEL = '/shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8'
PROMPT = 'TP8 native baseline. Write the next eight integers after 100 in ascending order.'
async def main():
    tp, mode = int(sys.argv[1]), sys.argv[2]
    folder = Path(os.environ.get('GEN_ROOT', '/home/yifehuan/data/qwen38_mono_repro')) / f'tp{tp}_{mode}'
    if os.environ.get('GEN_SUFFIX'):
        folder = Path(str(folder)+'_'+os.environ['GEN_SUFFIX'])
    rows = []
    concs = tuple(map(int, os.environ.get('GEN_CONCS','1,2,4,8').split(',')))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
        async def request_one():
            payload = dict(model=MODEL, prompt=PROMPT, temperature=0., top_p=1., seed=1234,
                           max_tokens=128, ignore_eos=True, stream=False)
            start = time.perf_counter()
            async with session.post(os.environ.get('GEN_URL', 'http://127.0.0.1:8000') + '/v1/completions', json=payload) as response:
                body = await response.json()
                if response.status != 200: raise RuntimeError((response.status, body))
            wall = time.perf_counter() - start
            usage = body['usage']
            assert len(body['choices']) == 1 and usage['completion_tokens'] == 128, body
            return dict(wall_s=wall, usage=usage,
                        client_decode_ms=(wall-usage['ttft_s'])/127*1000,
                        hashes=[hashlib.sha256(c['text'].encode()).hexdigest() for c in body['choices']])
        async def one(conc, rep):
            start = time.perf_counter()
            requests = await asyncio.gather(*(request_one() for _ in range(conc)))
            return dict(conc=conc, rep=rep, wall_s=time.perf_counter()-start, requests=requests,
                        client_decode_ms=max(r['client_decode_ms'] for r in requests),
                        usage=dict(tpot_s=max(r['usage']['tpot_s'] for r in requests)))
        async def snapshot(label):
            parent = int(folder.joinpath('server.pid').read_text())
            process_rows = subprocess.check_output(['ps','-eo','pid=,ppid=,args='], text=True)
            parsed = [line.split(None,2) for line in process_rows.splitlines()]
            edges = [(int(parts[0]), int(parts[1])) for parts in parsed]
            eligible = {int(parts[0]) for parts in parsed if any(name in parts[2] for name in ('openai_server', 'spawn_main', 'ATOM::'))}
            owned = {parent}
            while True:
                more = {pid for pid, ppid in edges if ppid in owned}
                if more <= owned: break
                owned |= more
            for pid in owned & eligible:
                try: os.kill(pid, signal.SIGUSR1)
                except ProcessLookupError: pass
            await asyncio.sleep(.2)
            payload = {p.name:json.loads(p.read_text()) for p in folder.glob('telemetry_*.json') if p.stem.removeprefix('telemetry_').isdigit()}
            folder.joinpath(f'telemetry_{label}.json').write_text(json.dumps(payload, indent=2))
            mem = subprocess.check_output(['rocm-smi','--showmeminfo','vram','--json'], text=True)
            folder.joinpath(f'vram_{label}.json').write_text(mem)
        await snapshot('before')
        for conc in concs:
            await one(conc, -1)
            for rep in range(7):
                row = await one(conc, rep)
                rows.append(row)
                print(json.dumps(row), flush=True)
                folder.joinpath('benchmark_partial.json').write_text(json.dumps(rows, indent=2))
            await snapshot(f'c{conc}')
    summary = {}
    for conc in concs:
        selected = [r for r in rows if r['conc']==conc]
        client = statistics.median(r['client_decode_ms'] for r in selected)
        summary[str(conc)] = dict(client_decode_ms=client,
            server_tpot_ms=statistics.median(r['usage']['tpot_s']*1000 for r in selected),
            decode_tokens_s=conc*1000/client, decode_tokens_s_per_gpu=conc*1000/client/tp,
            end_to_end_tokens_s=statistics.median(conc*128/r['wall_s'] for r in selected))
    result = dict(tp=tp, mode=mode, prompt=PROMPT, output_tokens=128, repetitions=7,
                  admission='C concurrent greedy HTTP requests; verify actual widths from telemetry', rows=rows, summary=summary)
    folder.joinpath('benchmark.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
asyncio.run(main())
