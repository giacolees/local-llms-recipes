#!/usr/bin/env python3
"""Build a comparable CSV from successful vLLM benchmark JSON results."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ['profile', 'recipe', 'runtime', 'run', 'completed', 'failed',
          'output_throughput', 'total_token_throughput', 'request_throughput',
          'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms',
          'mean_itl_ms', 'mean_e2el_ms']


def summarize(root=ROOT):
    rows = []
    for path in sorted((root / 'results').glob('*/*/*/*/result.json')):
        result = json.loads(path.read_text())
        metadata = json.loads(path.with_name('metadata.json').read_text())
        row = {key: result.get(key, '') for key in FIELDS}
        row.update({key: metadata[key] for key in ('profile', 'recipe', 'runtime')})
        row['run'] = str(path.parent.relative_to(root))
        rows.append(row)
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    summarize()
