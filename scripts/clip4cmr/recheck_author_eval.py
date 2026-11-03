"""Independent re-check of scripts/clip4cmr/improve.py test mAP with the upstream CLIP4CMR evaluator.

Reloads every confirm/*/best.pt, recomputes test features, scores with upstream
fx_calc_map_label and compares with results.csv; also checks best.pt epoch = history argmax.
Usage: python -m scripts.clip4cmr.recheck_author_eval [out_dir]
"""
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.spatial  # noqa: F401  (upstream evaluator uses scipy.spatial.distance)
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.clip4cmr import improve as ci  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'reports/clip4cmr_improve_20261030'
if len(sys.argv) > 2:  # optional feature dir, e.g. CLIP4CMR/data/pascal_author_split_mirror
    ci.set_feats(sys.argv[2])
src = (ROOT/'CLIP4CMR/upstream/evaluate.py').read_text()
tree = ast.parse(src if src.lstrip().startswith('"""') else '"""'+src)  # same workaround as scripts/baseline/train_baseline.py
tree.body = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'fx_calc_map_label']
ns = {'np': np, 'scipy': sys.modules['scipy']}; exec(compile(tree, 'upstream_eval', 'exec'), ns)
fx = ns['fx_calc_map_label']

csvrows = {(r['backbone'], r['variant'], r['seed'], r['direction']): float(r['mAP@ALL'])
           for r in csv.DictReader(open(out/'results.csv', encoding='utf-8-sig'))}
datas = {k: ci.load_backbone(k, 'cuda') for k in ci.BACKBONES}
maxdiff, n, bad_epoch = 0., 0, 0
for k, d in datas.items():
    for run in sorted((out/'confirm'/k).iterdir()):
        variant, seed = run.name.rsplit('_seed', 1)
        res = json.loads((run/'result.json').read_text(encoding='utf-8'))
        mode, use_b = ci.VARIANTS[variant]
        model = ci.build_model(d, mode, res['alpha'], 'cuda')
        ck = torch.load(run/'best.pt', map_location='cuda'); model.load_state_dict(ck['state_dict']); model.eval()
        hist = json.loads((run/'history.json').read_text(encoding='utf-8'))
        best = max(hist, key=lambda h: (h['validation']['I2T']['mAP@ALL']+h['validation']['T2I']['mAP@ALL'])/2)
        bad_epoch += best['epoch'] != ck['epoch']
        with torch.no_grad():
            _, fi, ft = ci.forward(model, d['image'][d['test']], ci.eval_text(d, d['test'], use_b))
        y = d['labels'][d['test']]
        for dr, q, g in (('I2T', fi, ft), ('T2I', ft, fi)):
            v = float(fx(q.cpu().numpy(), g.cpu().numpy(), y))
            maxdiff = max(maxdiff, abs(v-csvrows[(k, variant, seed, dr)])); n += 1
print(f'checked {n} direction-runs; max |upstream - results.csv| = {maxdiff:.3g}; best.pt epoch mismatches = {bad_epoch}')
