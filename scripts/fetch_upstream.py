"""Fetch selected upstream into an isolated folder; preserve real revision IDs.
No global Git configuration changes. Existing checkouts are never reset/pulled.
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['all', 'clip', 'vilt', 'dscmr', 'dcmh', 'dchmt'], default='all')
    args = p.parse_args()
    catalog = json.loads((ROOT / 'methods.json').read_text(encoding='utf-8'))
    failures = 0
    for method in catalog:
        if args.method not in ('all', method['id']):
            continue
        dest = ROOT / 'upstream' / method['id']
        dest.parent.mkdir(parents=True, exist_ok=True)
        record = {'method': method['id'], 'url': method['execution_repo'],
                  'timestamp': datetime.datetime.now().astimezone().isoformat()}
        try:
            if not dest.exists():
                proc = subprocess.run(['git', 'clone', '--depth', '1', method['execution_repo'], str(dest)],
                                      capture_output=True, text=True, timeout=600)
                record['clone_stdout'] = proc.stdout
                record['clone_stderr'] = proc.stderr
                if proc.returncode:
                    raise RuntimeError(f'git clone returned {proc.returncode}')
            origin = subprocess.check_output(['git', '-C', str(dest), 'remote', 'get-url', 'origin'], text=True).strip()
            if origin.rstrip('/').removesuffix('.git') != method['execution_repo'].rstrip('/').removesuffix('.git'):
                raise RuntimeError('Existing checkout origin differs; leaving it untouched')
            record['commit'] = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
            record['status'] = 'fetched'
        except Exception as exc:
            failures += 1
            record['status'] = 'blocked'
            record['error'] = str(exc)
        target = ROOT / method['folder'] / 'results' / 'upstream.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f"{method['id']}: {record['status']}")
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
