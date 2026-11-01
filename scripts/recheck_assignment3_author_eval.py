"""Independent re-check of Assignment 3 test mAP using the author's evaluator.

Usage: python recheck_assignment3_author_eval.py <SEG source root> <workspace> [out dir] [main|confirm]
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy.io import loadmat

seg, ws = Path(sys.argv[1]), Path(sys.argv[2])
out = ws / (sys.argv[3] if len(sys.argv) > 3 else 'reports/assignment3_20261029')
sub = sys.argv[4] if len(sys.argv) > 4 else 'main'
spec = importlib.util.spec_from_file_location('up_model', seg / 'DSCMR/upstream/model.py')
um = importlib.util.module_from_spec(spec); spec.loader.exec_module(um)
spec = importlib.util.spec_from_file_location('up_eval', seg / 'DSCMR/upstream/evaluate.py')
ue = importlib.util.module_from_spec(spec); spec.loader.exec_module(ue)

d = seg / 'DSCMR/upstream/data/pascal'
ld = lambda s, f: loadmat(d / f'{s}_{f}.mat')[f'{s}_{f}']
classes = np.unique(ld('train', 'img_lab').reshape(-1))
x = torch.tensor(ld('test', 'img'), dtype=torch.float32)
t = torch.tensor(ld('test', 'txt'), dtype=torch.float32)
y = np.searchsorted(classes, ld('test', 'img_lab').reshape(-1))

csvrows = {(r['variant'].rstrip('*'), r['seed'], r['direction']): float(r['mAP@ALL'])
           for r in csv.DictReader(open(out / 'results.csv', encoding='utf-8-sig'))}
maxdiff, agg = 0.0, {}
for run in sorted((out / sub).iterdir()):
    variant, seed = run.name.rsplit('_seed', 1)
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    hist = json.load(open(run / 'history.json'))
    best = max(hist, key=lambda h: (h['validation']['I2T']['mAP@ALL'] + h['validation']['T2I']['mAP@ALL']) / 2)
    m = um.IDCM_NN(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(classes))
    m.load_state_dict(ck['model']); m.eval()
    with torch.no_grad():
        zi, zt = [a.numpy() for a in m(x, t)[:2]]
    i2t = ue.fx_calc_map_label(zi, zt, y); t2i = ue.fx_calc_map_label(zt, zi, y)
    for k, v in (('I2T', i2t), ('T2I', t2i)):
        maxdiff = max(maxdiff, abs(v - csvrows[(variant, seed, k)]))
        agg.setdefault((variant, k), []).append(v)
    print(f'{run.name:24s} ckpt_ep={ck["epoch"]:3d} hist_best={best["epoch"]:3d} ran={len(hist):3d} I2T={i2t*100:.2f} T2I={t2i*100:.2f}')
print('max |author_eval - results.csv| =', maxdiff)
for (v, k), a in sorted(agg.items()):
    print(f'{v:12s} {k} {np.mean(a)*100:.2f} ± {np.std(a, ddof=1)*100:.2f}')
