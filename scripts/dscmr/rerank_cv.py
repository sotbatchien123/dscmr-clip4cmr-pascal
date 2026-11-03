"""DSCMR on its own author features (VGG19 4096 / text 300, split 800/100): does post-hoc re-ranking help?

3-fold stratified CV on the 800 train rows (test untouched), DSCMR B0 (upstream loss, balanced
batches as Assignment 3), seed 42, max 300 epochs / patience 50; score held-out fold with the
improve_v3 re-rankers. Writes reports/improve_v3_20261030/cv/dscmr_own/B0/cv.json.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
from scripts.clip4cmr import improve as ci  # noqa: E402
from scripts.dscmr import loss_ablation as r1  # noqa: E402
from scripts.encoders import encoder_study as v  # noqa: E402
from common.paths import from_repo


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.1)
    Model, _ = r1.upstream_definitions(ROOT)
    x, t, y, _ = r1.load_split(ROOT/'DSCMR/upstream/data/pascal', 'train')
    X, T = torch.tensor(x, device='cuda'), torch.tensor(t, device='cuda')
    folds = ci.stratified_folds(y); base = v.OUT/'cv'/'dscmr_own'/'B0'; per = []
    for f in range(3):
        tr = np.concatenate([folds[g] for g in range(3) if g != f]); va = folds[f]
        res = r1.train_run(v.Args(base/f'fold{f}'), Model, (X, T), y, tr, va, 'B0_original', 42, '.', ci.TUNE_EPOCHS, ci.PATIENCE)
        m = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=20).cuda()
        m.load_state_dict(torch.load(from_repo(res['run'])/'best.pt', map_location='cuda', weights_only=False)['model']); m.eval()
        with torch.inference_mode():
            zi, zt = m(X[va], T[va])[:2]
        per.append({'val_best': res['best_validation_mean_map'], 'rerank': v.score(F.normalize(zi, dim=1), F.normalize(zt, dim=1), y[va])})
    summ = {k: [float(np.mean([p['rerank'][k][i] for p in per])) for i in (0, 1)] for k in per[0]['rerank']}
    ci.write_json(base/'cv.json', {'folds': per, 'cv_mean': summ})
    for k, s in summ.items():
        print(f'{k:8s} {s[0]*100:6.2f} {s[1]*100:6.2f} avg {np.mean(s)*100:6.2f}')


if __name__ == '__main__':
    main()
