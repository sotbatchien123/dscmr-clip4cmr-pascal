"""Feasibility probe (train-only 3-fold CV, test never touched): retrieval in label space.

Idea (Semantic Correlation Matching, Rasiwasia et al. 2010; DSCMR label layer): give each image and
each text a 20-class probability vector from a per-modality classifier, rank by similarity of those
vectors. Relevance = same class, so mAP@ALL tracks how well both modalities are classified.
Compares, on held-out folds of the 800 train rows:
  feature cosine (no training)  vs  softmax-regression probabilities (per modality)  vs
  + zero-shot CLIP prompt probabilities fused in (CLIP only).
Usage: python probe_label_space.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import clip4cmr_improve as ci  # noqa: E402
import dscmr_assignment3 as r1  # noqa: E402


def softmax_reg(x, y, nc, epochs=300, wd=1e-3):
    torch.manual_seed(0)
    W = torch.zeros(x.shape[1], nc, device=x.device, requires_grad=True)
    b = torch.zeros(nc, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=epochs, line_search_fn='strong_wolfe')
    yt = torch.as_tensor(y, device=x.device)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(x@W+b, yt) + wd*W.square().sum(); loss.backward(); return loss
    opt.step(closure)
    return lambda z: torch.softmax(z@W.detach()+b.detach(), 1)


def mapp(a, b, y):
    s = (F.normalize(a, dim=1) @ F.normalize(b, dim=1).T).cpu().numpy()
    return r1.metrics(s, y, y)['mAP@ALL'], r1.metrics(s.T, y, y)['mAP@ALL']


def cv(xi, xt, y, nc, prompts=None):
    folds = ci.stratified_folds(y)
    out = {}
    for f in range(3):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(3) if g != f])
        n = lambda z: F.normalize(z, dim=1)
        res = {'feature_cosine': mapp(xi[te], xt[te], y[te])} if xi.shape[1] == xt.shape[1] else {}
        pi = softmax_reg(n(xi[tr]), y[tr], nc)(n(xi[te])); pt = softmax_reg(n(xt[tr]), y[tr], nc)(n(xt[te]))
        res['label_space_probs'] = mapp(pi, pt, y[te])
        res['classifier_acc_img_txt'] = (float((pi.argmax(1).cpu().numpy() == y[te]).mean()), float((pt.argmax(1).cpu().numpy() == y[te]).mean()))
        if prompts is not None:
            P = n(torch.as_tensor(prompts, device=xi.device, dtype=torch.float32))
            zi, zt = torch.softmax(100*n(xi[te])@P.T, 1), torch.softmax(100*n(xt[te])@P.T, 1)
            res['zero_shot_probs'] = mapp(zi, zt, y[te])
            res['label_space+zero_shot'] = mapp(pi+zi, pt+zt, y[te])
        for k, v in res.items():
            out.setdefault(k, []).append(v)
    return {k: [round(float(np.mean([v[0] for v in vs]))*100, 2), round(float(np.mean([v[1] for v in vs]))*100, 2)] for k, vs in out.items()}


def main():
    report = {}
    for feats, keys in ((ROOT/'CLIP4CMR/data/pascal_author_split_mirror', ('rn50', 'vitl14')),):
        ci.set_feats(feats)
        for k in keys:
            d = ci.load_backbone(k, 'cuda'); tr = d['train']
            for tname in ('concat', 'capmean'):
                report[f'clip4cmr_author/{k}/{tname}'] = cv(d['image'][tr].float(), d[tname][tr].float(), d['labels'][tr], d['nc'], d['prompts'])
                print(f'clip4cmr_author/{k}/{tname}', report[f'clip4cmr_author/{k}/{tname}'], flush=True)
    x, t, y, _ = r1.load_split(ROOT/'DSCMR/upstream/data/pascal', 'train')
    report['dscmr/vgg19+text300'] = cv(torch.tensor(x, device='cuda'), torch.tensor(t, device='cuda'), y, 20)
    print('dscmr', report['dscmr/vgg19+text300'], flush=True)
    out = ROOT/'reports/probe_label_space_20261030.json'
    out.write_text(json.dumps({'note': __doc__, 'values': '[I2T, T2I] mAP@ALL % mean over 3 CV folds on train (acc: img, txt)',
                               'results': report}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
