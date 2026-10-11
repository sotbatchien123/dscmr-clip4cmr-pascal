"""Run local infrastructure tests and syntax checks; never train a model."""
import datetime
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = ROOT/'audit'
    audit.mkdir(exist_ok=True)
    result = {'timestamp': datetime.datetime.now().astimezone().isoformat(),
              'scope': 'synthetic_software_tests_only_NOT_model_benchmarks', 'python': sys.executable}
    files = [p for p in ROOT.rglob('*.py') if 'upstream' not in p.relative_to(ROOT).parts and
             not any(x.startswith('.venv') for x in p.relative_to(ROOT).parts)]
    compile_errors = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append({'file': str(path.relative_to(ROOT)), 'error': str(exc)})
    result['syntax'] = {'files': len(files), 'errors': compile_errors}
    proc = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT,
                          capture_output=True, text=True)
    (audit/'tests.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
    result['tests_exit_code'] = proc.returncode
    checks = []
    for method in ['01_CLIP/evaluate.py', '02_ViLT/run.py', '03_DSCMR/train.py', '04_DCMH/run.py', '05_DCHMT/run.py']:
        cp = subprocess.run([sys.executable, method, '--help'], cwd=ROOT, capture_output=True, text=True)
        checks.append({'script': method, 'exit_code': cp.returncode, 'stderr': cp.stderr})
    result['cli_help_checks'] = checks
    (audit/'software_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(compile_errors or proc.returncode or any(x['exit_code'] for x in checks)))


if __name__ == '__main__':
    main()
