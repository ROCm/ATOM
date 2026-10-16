"""Paired greedy GSM8K evaluation with bounded concurrency and sample logs."""
import argparse
import asyncio
import json
import re
import os
import tempfile
import stat
from decimal import Decimal
from pathlib import Path
import aiohttp


def extract(text):
    # Prefer the explicitly requested final answer, excluding reasoning text.
    matches = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if not matches:
        return None
    return str(Decimal(matches[-1].replace(',', '')).normalize())


def save_rescored_rows(output, rows):
    """Atomically persist normalized resume scores before appending new rows."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent,
                                         prefix=output.name + '.', delete=False) as file:
            temporary = Path(file.name)
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
            if output.exists():
                os.fchmod(file.fileno(), stat.S_IMODE(output.stat().st_mode))
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def main(args):
    samples = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()]
    if args.limit:
        samples = samples[:args.limit]
    selected = set(range(len(samples)))
    if args.indices:
        selected = {int(index) for index in args.indices.split(',')}
        if not selected or min(selected) < 0 or max(selected) >= len(samples):
            raise ValueError('Selected sample index is outside the dataset')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    for row in previous:
        if row['index'] not in selected:
            raise ValueError('Resume output includes an unselected sample')
        row['predicted'] = extract(row['content'])
        row['expected'] = extract(samples[row['index']]['answer'])
        if row.get('question') != samples[row['index']]['question']:
            raise ValueError('Resume dataset differs from saved sample')
        row['correct'] = row['expected'] is not None and row['predicted'] == row['expected']
    completed = {r['index'] for r in previous}
    if len(completed) != len(previous):
        raise ValueError('Resume output contains duplicate sample indices')
    if previous:
        save_rescored_rows(output, previous)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = list(previous)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        async def one(index, sample):
            async with semaphore:
                payload = dict(model=args.model, temperature=0., top_p=1., seed=1234,
                               max_tokens=args.tokens, stream=False,
                               messages=[dict(role='user', content=sample['question'] +
                                              '\nSolve the problem and finish with #### followed by the numerical answer.')])
                async with session.post(args.url + '/v1/chat/completions', json=payload) as response:
                    body = await response.json()
                    if response.status != 200:
                        raise RuntimeError((response.status, body))
                choice = body['choices'][0]
                message = choice['message']
                text = message.get('content') or ''
                predicted = extract(text)
                expected = extract(sample['answer'])
                row = dict(index=index, question=sample['question'], expected=expected,
                           predicted=predicted, correct=expected is not None and predicted == expected,
                           content=text, reasoning=message.get('reasoning_content'),
                           finish_reason=choice['finish_reason'], usage=body['usage'],
                           requested_max_tokens=args.tokens, concurrency=args.concurrency)
                async with lock:
                    with output.open('a') as file:
                        file.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows.append(row)
                    print(json.dumps(dict(index=index, correct=row['correct'],
                                          predicted=predicted, expected=expected,
                                          finish_reason=row['finish_reason'])), flush=True)
        await asyncio.gather(*(one(i, s) for i, s in enumerate(samples) if i in selected and i not in completed))
    summary = dict(samples=len(rows), correct=sum(r['correct'] for r in rows),
                   accuracy=sum(r['correct'] for r in rows)/len(rows),
                   unparsed=sum(r['predicted'] is None for r in rows),
                   truncated=sum(r['finish_reason'] == 'length' for r in rows),
                   concurrency=args.concurrency, max_tokens=args.tokens,
                   selected_indices=sorted(selected) if args.indices else None,
                   protocol='zero-shot greedy chat, explicit #### answer; verify Mono widths separately')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--indices', help='Comma-separated original dataset indices for failure retries')
    parser.add_argument('--concurrency', type=int, choices=(1, 2), default=1)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='/shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8')
    asyncio.run(main(parser.parse_args()))
