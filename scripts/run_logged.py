"""Execute a real command with an immutable run directory and full stdout.
Exit zero is not automatically a reproduced result. Metrics require review.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['clip', 'vilt', 'dscmr', 'dcmh', 'dchmt'])
    p.add_argument('--cwd', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('Supply executable and arguments after --')
    methods = json.loads((ROOT / 'methods.json').read_text(encoding='utf-8'))
    folder = next(x['folder'] for x in methods if x['id'] == a.method)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT / folder / 'results' / stamp
    out.mkdir(parents=True)
    meta = {'method': a.method, 'started_at': datetime.datetime.now().astimezone().isoformat(),
            'command': command, 'cwd': str(Path(a.cwd).resolve()), 'launcher_python': sys.executable,
            'platform': platform.platform(), 'requested_seed': a.seed, 'seed_note':
            'PYTHONHASHSEED is set; training RNGs must be set by the method itself.',
            'status': 'running', 'metrics': None, 'is_reproduction_verified': False}
    (out / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    env = dict(os.environ, PYTHONHASHSEED=str(a.seed), PYTHONUNBUFFERED='1')
    start = time.perf_counter()
    code = 1
    try:
        git = subprocess.run(['git', '-C', a.cwd, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        meta['git_commit'] = git.stdout.strip() if git.returncode == 0 else None
        diff = subprocess.run(['git', '-C', a.cwd, 'diff'], capture_output=True, text=True)
        (out / 'upstream_changes.patch').write_text(diff.stdout, encoding='utf-8')
        with (out / 'stdout.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(command, cwd=a.cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in proc.stdout:
                    print(line, end='')
                    log.write(line)
                    log.flush()
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
        meta['status'] = 'finished_requires_metric_review' if code == 0 else 'failed'
    except (Exception, KeyboardInterrupt) as exc:
        meta['status'] = 'failed'
        meta['error'] = repr(exc)
    finally:
        meta['exit_code'] = code
        meta['elapsed_seconds'] = time.perf_counter() - start
        (out / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(f'Run evidence: {out}')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
