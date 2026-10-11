"""Adapter for WendellGul/DCMH, a COMMUNITY implementation, not official code.
Old upstream APIs may need porting; runtime compatibility is not yet validated.
"""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default=str(ROOT/'upstream/dcmh'))
    p.add_argument('--data', required=True)
    p.add_argument('--vgg-weights', required=True)
    p.add_argument('--bits', type=int, choices=[16, 32, 64], default=16)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    repo = Path(a.upstream).resolve()
    data, weights = Path(a.data).resolve(), Path(a.vgg_weights).resolve()
    cmd = [sys.executable, str(ROOT/'scripts/seeded_entry.py'), '--seed', str(a.seed),
           '--entry', str(repo/'main.py'), '--', 'train', f'--data_path={data}',
           f'--pretrain_model_path={weights}', f'--bit={a.bits}',
           f'--max_epoch={a.epochs}', f'--batch_size={a.batch_size}']
    if a.dry_run:
        print(subprocess.list2cmdline(cmd))
        return
    for path in [repo/'main.py', data, weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for folder in ['checkpoints', 'result']:
        (repo/folder).mkdir(exist_ok=True)
    raise SystemExit(subprocess.call([sys.executable, str(ROOT/'scripts/run_logged.py'),
        '--method', 'dcmh', '--cwd', str(repo), '--seed', str(a.seed), '--'] + cmd))


if __name__ == '__main__':
    main()
