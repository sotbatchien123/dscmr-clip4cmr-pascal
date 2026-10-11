"""Launch the original ViLT retrieval trainer/evaluator with logged evidence."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default=str(ROOT / 'upstream/vilt'))
    p.add_argument('--arrow-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=['evaluate', 'finetune'], default='evaluate')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    repo = Path(a.upstream).resolve()
    arrow = Path(a.arrow_root).resolve()
    checkpoint = Path(a.checkpoint).resolve()
    cmd = [sys.executable, str(repo / 'run.py'), 'with', 'task_finetune_irtr_f30k_randaug',
           f'data_root={arrow}', 'num_gpus=1', 'num_nodes=1', f'per_gpu_batchsize={a.batch_size}',
           'precision=32', f'load_path={checkpoint}', f'seed={a.seed}']
    if a.mode == 'evaluate':
        cmd.append('test_only=True')
    if a.dry_run:
        print(subprocess.list2cmdline(cmd))
        return
    if not (repo / 'run.py').is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Fetch upstream and supply the correct checkpoint first')
    if not list(arrow.glob('*.arrow')):
        raise FileNotFoundError('No .arrow files in arrow-root; run the official data converter')
    raise SystemExit(subprocess.call([sys.executable, str(ROOT/'scripts/run_logged.py'),
         '--method', 'vilt', '--cwd', str(repo), '--seed', str(a.seed), '--'] + cmd))


if __name__ == '__main__':
    main()
