"""Fixed vs learnable semantic prototypes for the CLIP4CMR head (revision experiment).

A_sem keeps the class prototypes fixed at the CLIP text embedding of "a photo of a {class}.";
A_semL starts from the same prototypes but updates them with the adapter (same optimiser, same
learning rate). Everything else follows scripts/clip4cmr/improve.py: alpha = 0.3 (chosen on CV
in the original study), fixed 720/80 validation split, max 500 epochs, patience 50, seeds 42-46,
test opened once after all runs finished.

Two feature sets: RN50 on the author split (weak encoder) and frozen DFN5B ViT-H/14 (strong
encoder). Besides retrieval mAP the script reports nearest-prototype classification accuracy on
the test images/texts and how far the learnable prototypes moved from their initialisation.

    python -m scripts.clip4cmr.prototype_ablation
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.clip4cmr import improve as ci
from scripts.baseline import train_baseline as ts
from common.paths import from_repo

ROOT = ci.ROOT
FEATURES = {'rn50': (ROOT/'CLIP4CMR/data/pascal_author_split_mirror', 'rn50'),
            'dfn5b': (ROOT/'CLIP4CMR/data/author_split_v3', 'dfn5b_h14')}
VARIANTS = ('base', 'A_sem', 'A_semL', 'A_sem+B', 'A_semL+B')
ALPHA = 0.3


def class_accuracy(model, d, rows, use_b):
    """Nearest-prototype accuracy (cosine) for test images and texts."""
    model.eval()
    with torch.inference_mode():
        centers, fi, ft = ci.forward(model, d['image'][rows], ci.eval_text(d, rows, use_b))
        c = ci.l2n(centers.T)
        y = torch.as_tensor(d['labels'][rows], device=c.device)
        acc = lambda f: float((ci.l2n(f) @ c.T).argmax(1).eq(y).float().mean())
        return {'image': acc(fi), 'text': acc(ft)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'reports/revision_20261112/prototype_ablation')
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    pan = ts.prototype_loss()
    results = []
    for name, (feats, key) in FEATURES.items():
        ci.set_feats(feats); d = ci.load_backbone(key, args.device)
        tr = d['train']; inner, val = ci.fixed_split(d['labels'][tr])
        runs = [(v, s, ci.train_run(args, d, pan, tr[inner], tr[val], v, ALPHA, s,
                                     args.out/name/f'{v}_seed{s}', ci.CONFIRM_EPOCHS))
                for v in VARIANTS for s in ci.SEEDS]
        init = torch.as_tensor(d['prompts'], dtype=torch.float32, device=args.device)
        init = ci.l2n(init)
        for v, s, r in runs:  # test opened only now
            mode, use_b = {**ci.VARIANTS, **ci.EXTRA_VARIANTS}[v]
            ts.seed_all(s); model = ci.build_model(d, mode, ALPHA, args.device)
            model.load_state_dict(torch.load(from_repo(r['run'])/'best.pt', map_location=args.device)['state_dict'])
            m, _, _ = ci.evaluate(model, d['image'][d['test']], ci.eval_text(d, d['test'], use_b), d['labels'][d['test']])
            row = {'features': name, 'variant': v, 'seed': s, 'best_epoch': r['best_epoch'],
                   'I2T': m['I2T']['mAP@ALL'], 'T2I': m['T2I']['mAP@ALL'],
                   'accuracy': class_accuracy(model, d, d['test'], use_b)}
            if mode == 'semL':
                learned = ci.l2n(model.centers.detach().T)
                row['prototype_cos_to_init'] = float((learned*init).sum(1).mean())
            results.append(row)
    ci.write_json(args.out/'results.json', results)
    lines = ['| Features | Variant | I2T | T2I | Mean | Img acc | Txt acc | cos(init) |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in FEATURES:
        for v in VARIANTS:
            rs = [r for r in results if (r['features'], r['variant']) == (name, v)]
            mean = lambda f: np.mean([f(r) for r in rs])*100
            sd = np.std([(r['I2T']+r['T2I'])/2 for r in rs], ddof=1)*100
            cos = f"{np.mean([r['prototype_cos_to_init'] for r in rs]):.3f}" if 'prototype_cos_to_init' in rs[0] else ''
            lines.append(f"| {name} | {v} | {mean(lambda r: r['I2T']):.2f} | {mean(lambda r: r['T2I']):.2f} | "
                         f"{mean(lambda r: (r['I2T']+r['T2I'])/2):.2f} ± {sd:.2f} | {mean(lambda r: r['accuracy']['image']):.1f} | "
                         f"{mean(lambda r: r['accuracy']['text']):.1f} | {cos} |")
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
