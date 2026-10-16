"""Deterministic request and decode-dump comparison for native/Mono diagnosis."""
import argparse
import asyncio
import json
from pathlib import Path


async def request(args):
    import aiohttp
    prompts = args.prompt or [
        'TP8 native baseline. Write the next eight integers after 100 in ascending order.'
    ]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def one(prompt):
            payload = dict(model=args.model, prompt=prompt, temperature=0., top_p=1.,
                           seed=1234, max_tokens=args.tokens, ignore_eos=True, stream=False)
            async with session.post(args.url + '/v1/completions', json=payload) as response:
                body = await response.json()
                if response.status != 200:
                    raise RuntimeError((response.status, body))
                return dict(prompt=prompt, response=body)
        results = await asyncio.gather(*(one(p) for p in prompts))
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for result in results:
        print(result['response']['choices'][0]['text'])


def compare(args):
    import torch
    ref, target = Path(args.reference), Path(args.target)
    files = sorted(ref.glob('decode_*.pt')) if ref.is_dir() else [ref]
    rows = []
    for file in files:
        candidate = target / file.name if target.is_dir() else target
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        a, b = (torch.load(p, weights_only=True) for p in (file, candidate))
        row = dict(step=file.stem)
        for key in ('input_ids', 'positions'):
            if key in a or key in b:
                row[key + '_equal'] = key in a and key in b and torch.equal(a[key], b[key])
        for key in ('hidden', 'logits'):
            x, y = a[key].float(), b[key].float()
            if x.shape != y.shape:
                raise ValueError((file, key, x.shape, y.shape))
            delta = y - x
            row[key] = dict(finite=bool(torch.isfinite(y).all()),
                            rel_l2=float(delta.norm() / x.norm().clamp_min(1e-12)),
                            max_abs=float(delta.abs().max()),
                            cosine=float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)))
        row['argmax_equal'] = bool(torch.equal(a['argmax'], b['argmax']))
        row['reference_argmax'] = a['argmax'].tolist()
        row['target_argmax'] = b['argmax'].tolist()
        top = a['logits'].float().topk(2, dim=-1).values
        row['reference_top2_margin'] = (top[..., 0] - top[..., 1]).tolist()
        rows.append(row)
    report = dict(steps=len(rows), argmax_matches=sum(r['argmax_equal'] for r in rows), rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    req = commands.add_parser('request')
    req.add_argument('--url', default='http://127.0.0.1:8000')
    req.add_argument('--model', default='/shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8')
    req.add_argument('--prompt', action='append')
    req.add_argument('--tokens', type=int, default=128)
    req.add_argument('--output', required=True)
    cmp = commands.add_parser('compare')
    cmp.add_argument('reference')
    cmp.add_argument('target')
    cmp.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'request':
        asyncio.run(request(args))
    else:
        compare(args)


if __name__ == '__main__':
    main()
