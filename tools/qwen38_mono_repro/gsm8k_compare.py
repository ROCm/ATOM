"""Rescore paired GSM8K sample logs and retain individual disagreements."""
import argparse
import json
from pathlib import Path

from gsm8k import extract


def load(path, dataset):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        index = row['index']
        if index in rows:
            raise ValueError(f'Duplicate index {index} in {path}')
        if row['question'] != dataset[index]['question']:
            raise ValueError(f'Dataset mismatch at {index} in {path}')
        expected = extract(dataset[index]['answer'])
        predicted = extract(row['content'])
        rows[index] = dict(index=index, predicted=predicted, expected=expected,
                           correct=expected is not None and predicted == expected,
                           truncated=row['finish_reason'] == 'length',
                           completion_tokens=row['usage'].get('completion_tokens'))
    return rows


def compare(reference, target):
    shared = sorted(reference.keys() & target.keys())
    disagreements = []
    for index in shared:
        a, b = reference[index], target[index]
        if a['predicted'] != b['predicted'] or a['truncated'] != b['truncated']:
            disagreements.append(dict(index=index, reference=a, target=b))
    return dict(
        paired_samples=len(shared),
        reference_only=sorted(reference.keys() - target.keys()),
        target_only=sorted(target.keys() - reference.keys()),
        reference_correct=sum(reference[i]['correct'] for i in shared),
        target_correct=sum(target[i]['correct'] for i in shared),
        reference_truncated=sum(reference[i]['truncated'] for i in shared),
        target_truncated=sum(target[i]['truncated'] for i in shared),
        both_correct=sum(reference[i]['correct'] and target[i]['correct'] for i in shared),
        both_wrong=sum(not reference[i]['correct'] and not target[i]['correct'] for i in shared),
        reference_numeric_wrong=sum(not reference[i]['correct'] and not reference[i]['truncated'] for i in shared),
        target_numeric_wrong=sum(not target[i]['correct'] and not target[i]['truncated'] for i in shared),
        reference_correct_target_wrong=[i for i in shared if reference[i]['correct'] and not target[i]['correct']],
        reference_wrong_target_correct=[i for i in shared if not reference[i]['correct'] and target[i]['correct']],
        disagreements=disagreements,
        limitation='Free-generation quality comparison; different request schedules and trajectories do not isolate kernel error.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dataset = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()]
    report = compare(load(args.reference, dataset), load(args.target, dataset))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'disagreements'}, indent=2))
