"""Record this session's verified prerequisite failures, never fake scores."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    methods = json.loads((ROOT/'methods.json').read_text(encoding='utf-8'))
    env = json.loads((ROOT/'audit/environment.json').read_text(encoding='utf-8'))
    fetch = json.loads((ROOT/'03_DSCMR/results/upstream.json').read_text(encoding='utf-8'))
    if env['modules']['torch'] or fetch['status'] != 'blocked':
        raise RuntimeError('Prerequisites changed: inspect manually instead of stamping blocked states')
    rows = []
    for m in methods:
        result = {'method': m['name'], 'dataset': m['dataset'], 'status': 'NOT_RUN_PREREQUISITES_BLOCKED',
                  'metrics': None, 'is_reproduction_verified': False,
                  'evidence': ['audit/environment.json', '03_DSCMR/results/upstream.json'],
                  'note': 'No model experiment ran. Shared prerequisites failed; not a model-specific training error.'}
        out = ROOT/m['folder']/'results'
        out.mkdir(parents=True, exist_ok=True)
        (out/'status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        for direction in ['I2T','T2I']:
            for metric in (['R@1','R@5','R@10'] if m['id'] in ['clip','vilt'] else ['mAP_protocol_pending']):
                rows.append({'method':m['name'],'dataset':m['dataset'],'direction':direction,'metric':metric,
                             'paper_value':'','measured_value':'','difference':'','status':result['status']})
    with (ROOT/'results_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Recorded blocked states; all benchmark values remain empty.')


if __name__ == '__main__':
    main()
