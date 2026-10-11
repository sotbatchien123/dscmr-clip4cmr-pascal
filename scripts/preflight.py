"""Read-only environment audit. Does not install packages or stop processes."""
import argparse
import datetime
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def capture(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=35)
        return {'command': command, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as exc:
        return {'command': command, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', action='store_true')
    parser.add_argument('--out', default='audit/environment.json')
    args = parser.parse_args()
    result = {'checked_at': datetime.datetime.now().astimezone().isoformat(),
              'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
              'modules': {x: importlib.util.find_spec(x) is not None for x in
                          ['numpy', 'scipy', 'torch', 'torchvision', 'transformers', 'clip']}}
    result['gpu'] = capture(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv'])
    if args.network:
        result['github'] = capture(['git', 'ls-remote', 'https://github.com/penghu-cs/DSCMR.git', 'HEAD'])
    if result['modules']['torch']:
        result['torch_runtime'] = capture([sys.executable, '-c',
            "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
