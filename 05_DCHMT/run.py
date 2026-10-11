"""Adapter for the original kalenforn/DCHMT repository, not successor configs.
Uses the upstream dataset layout; verify local parser against captured revision.
"""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default=str(ROOT/'upstream/dchmt'))
    p.add_argument('--clip-weights', required=True)
    p.add_argument('--bits', type=int, choices=[16, 32, 64], default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    repo = Path(a.upstream).resolve()
    weights = Path(a.clip_weights).resolve()
    cmd = [sys.executable, str(ROOT/'scripts/seeded_entry.py'), '--seed', str(a.seed),
           '--entry', str(repo/'main.py'), '--', '--is-train', '--hash-layer', 'select',
           '--dataset', 'flickr25k', '--caption-file', 'caption.mat', '--index-file', 'index.mat',
           '--label-file', 'label.mat', '--similarity-function', 'euclidean', '--loss-type', 'l2',
           '--vartheta', '0.75', '--lr', '0.0001', '--output-dim', str(a.bits),
           '--save-dir', f'./result/flickr25k/{a.bits}/seed{a.seed}', '--clip-path', str(weights),
           '--batch-size', str(a.batch_size)]
    if a.dry_run:
        print(subprocess.list2cmdline(cmd))
        return
    for path in [repo/'main.py', weights] + [repo/'dataset/flickr25k'/x for x in ['caption.mat', 'index.mat', 'label.mat']]:
        if not path.is_file():
            raise FileNotFoundError(path)
    print('Batch size 32 is a resource-oriented starting point, not a verified paper setting.')
    raise SystemExit(subprocess.call([sys.executable, str(ROOT/'scripts/run_logged.py'),
        '--method', 'dchmt', '--cwd', str(repo), '--seed', str(a.seed), '--'] + cmd))


if __name__ == '__main__':
    main()
