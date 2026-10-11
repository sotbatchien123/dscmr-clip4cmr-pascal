"""Seed Python/NumPy/Torch, then execute an unchanged upstream entry point."""
import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--entry', required=True)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    entry = Path(a.entry).resolve()
    if not entry.is_file():
        raise FileNotFoundError(entry)
    import numpy as np
    import torch
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.chdir(entry.parent)
    sys.path.insert(0, str(entry.parent))
    tail = a.args[1:] if a.args[:1] == ['--'] else a.args
    sys.argv = [str(entry)] + tail
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
