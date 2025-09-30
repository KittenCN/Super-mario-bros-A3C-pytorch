"""Parse env_stability_report.jsonl and print a short summary.

Run: python scripts/parse_env_report.py env_stability_report.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

def summarize(path: Path):
    if not path.exists():
        print("Report not found:", path)
        return
    total = 0
    success = 0
    errors = Counter()
    durations = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            if obj.get('reset_ok'):
                success += 1
            else:
                err = obj.get('error') or 'unknown'
                # shorten common error signatures
                if '\\n' in err:
                    err = err.splitlines()[-1]
                errors[err] += 1
            durations.append(obj.get('duration', 0.0))
    print(f'total trials: {total}')
    print(f'successes: {success} failures: {total-success}')
    if durations:
        print(f'min duration: {min(durations):.3f}s mean: {sum(durations)/len(durations):.3f}s max: {max(durations):.3f}s')
    print('\nTop errors:')
    for err, cnt in errors.most_common(10):
        print(f'{cnt:4d}  {err}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/parse_env_report.py env_stability_report.jsonl')
        sys.exit(1)
    summarize(Path(sys.argv[1]))
