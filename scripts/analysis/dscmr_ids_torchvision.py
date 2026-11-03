"""Recover Pascal image IDs for the DSCMR author features (train 800 / test 100) by matching.

DSCMR ships VGG19 fc7 (ReLU, 4096-D) features without IDs. We compute torchvision VGG19 fc7 for all
1000 UIUC images under a few preprocessing variants, then match each DSCMR row to a UIUC image of the
same class (assuming DSCMR class k = k-th class alphabetically) with the Hungarian algorithm on cosine.
The match is accepted only if it is near-exact and one-to-one; results + diagnostics go to
DSCMR/data_ids/. Usage: python -m scripts.analysis.dscmr_ids_torchvision
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('TORCH_HOME', str(ROOT/'checkpoints/torch'))
from scripts.dscmr import loss_ablation as r1  # noqa: E402

OUT = ROOT/'DSCMR/data_ids'
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
VARIANTS = {
    'resize256_crop224': T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    'resize224x224': T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM]),
}


def l2n(a):
    return a/np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT/'DSCMR/upstream/data/pascal'
    x, _, y, classes = r1.load_split(data, 'train'); xt, _, yt, _ = r1.load_split(data, 'test', classes)
    dsc = np.concatenate([x, xt]); dy = np.concatenate([y, yt])
    records = json.loads((ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))
    names = sorted({r['class'] for r in records})
    uy = np.array([names.index(r['class']) for r in records])
    torch.cuda.set_per_process_memory_fraction(0.2)
    vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1').cuda().eval().half()
    fc7 = torch.nn.Sequential(vgg.features, vgg.avgpool, torch.nn.Flatten(), *list(vgg.classifier.children())[:5])
    best = None; diag = {}
    for vname, tf in VARIANTS.items():
        feats = []
        with torch.inference_mode():
            for s in range(0, len(records), 16):
                px = torch.stack([tf(Image.open(ROOT/'CLIP4CMR/data/pascal_uiuc_seed42/images'/f"{r['id']}.jpg").convert('RGB'))
                                  for r in records[s:s+16]]).cuda().half()
                feats.append(fc7(px).float().cpu().numpy())
        u = np.concatenate(feats)
        cos = l2n(dsc) @ l2n(u).T
        match = np.full(len(dsc), -1)
        for c in range(len(names)):
            rows, cols = np.flatnonzero(dy == c), np.flatnonzero(uy == c)
            ri, ci = linear_sum_assignment(-cos[np.ix_(rows, cols)])
            match[rows[ri]] = cols[ci]
        mc = cos[np.arange(len(dsc)), match]
        # margin: best same-class cosine minus second best (uniqueness of the match)
        srt = np.sort(np.where(uy[None, :] == dy[:, None], cos, -1), axis=1)
        diag[vname] = {'matched_cos_mean': float(mc.mean()), 'matched_cos_min': float(mc.min()),
                       'n_cos>0.95': int((mc > .95).sum()), 'margin_median': float(np.median(srt[:, -1]-srt[:, -2])),
                       'argmax_agrees_hungarian': float((cos.argmax(1) == match).mean())}
        print(vname, diag[vname], flush=True)
        if best is None or mc.mean() > best[1].mean():
            best = (vname, mc, match)
    vname, mc, match = best
    ids = [records[i]['id'] for i in match]
    result = {'variant': vname, 'diagnostics': diag, 'class_order_assumed': names,
              'train_ids': ids[:800], 'test_ids': ids[800:], 'matched_cos': mc.round(4).tolist(),
              'unused_ids_(dscmr_validation?)': sorted(set(r['id'] for r in records) - set(ids))}
    (OUT/'recovered_ids.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('saved', OUT/'recovered_ids.json', 'unused', len(result['unused_ids_(dscmr_validation?)']))


if __name__ == '__main__':
    main()
