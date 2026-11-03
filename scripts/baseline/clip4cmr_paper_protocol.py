"""Diagnostic: CLIP4CMR base vs A_sem+B under the paper's selection rule, author split (mirror).

Paper sec. 4.1: all 800 train pairs, max 200 epochs, early stopping 20; upstream code evaluates
TEST after every epoch and keeps the best test mean. Test-selected, hence optimistic; used only to
put our numbers on the same footing as the published 69.8 / 69.2. alpha = 0.3 (chosen by CV in
reports/clip4cmr_author_split_20261030). Seeds 42-46. Usage: python -m scripts.baseline.clip4cmr_paper_protocol
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.clip4cmr import improve as ci  # noqa: E402

OUT = ROOT/'reports/clip4cmr_paper_protocol_20261030'
SEEDS, EPOCHS, PATIENCE = (42, 43, 44, 45, 46), 200, 20


def main():
    ci.set_feats(ROOT/'CLIP4CMR/data/pascal_author_split_mirror')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    pan = ci.ts.prototype_loss()
    d = ci.load_backbone('rn50', 'cuda')
    rows, y_all = d['train'], torch.as_tensor(d['labels'], device='cuda')
    res = {}
    for variant in ('base', 'A_sem+B'):
        mode, use_b = ci.VARIANTS[variant]
        res[variant] = []
        for seed in SEEDS:
            ci.ts.seed_all(seed)
            model = ci.build_model(d, mode, .3, 'cuda')
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
            xt, tt, yt = d['image'][d['test']], ci.eval_text(d, d['test'], use_b), d['labels'][d['test']]
            best, best_ep = None, 0
            for epoch in range(1, EPOCHS+1):
                if epoch-1-best_ep >= PATIENCE: break
                model.train(); order = np.array(rows, copy=True); np.random.shuffle(order)
                if use_b:
                    crng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7]))
                    pick = {int(i): int(crng.choice(d['groups'][i])) for i in rows}
                for off in range(0, len(order), ci.BATCH):
                    ix = order[off:off+ci.BATCH]; it = torch.as_tensor(ix, device='cuda')
                    text = d['caps'][torch.as_tensor([pick[int(i)] for i in ix], device='cuda')] if use_b else d['concat'][it]
                    opt.zero_grad(set_to_none=True)
                    c, fi, ft = ci.forward(model, d['image'][it], text)
                    (pan(fi, c.T, y_all[it]) + pan(ft, c.T, y_all[it])).backward(); opt.step()
                m, _, _ = ci.evaluate(model, xt, tt, yt)
                sc = (m['I2T']['mAP@ALL']+m['T2I']['mAP@ALL'])/2
                if best is None or sc > best[0]:
                    best, best_ep = (sc, m['I2T']['mAP@ALL'], m['T2I']['mAP@ALL']), epoch
            res[variant].append({'seed': seed, 'best_epoch': best_ep, 'I2T': best[1], 'T2I': best[2]})
            print(variant, seed, best_ep, f'{best[1]*100:.2f}/{best[2]*100:.2f}', flush=True)
    agg = {v: {dr: [float(np.mean([r[dr] for r in rs])), float(np.std([r[dr] for r in rs], ddof=1))] for dr in ('I2T', 'T2I')}
           for v, rs in res.items()}
    ci.write_json(OUT/'result.json', {'protocol': __doc__, 'runs': res, 'aggregate_mean_sd': agg,
                                      'paper': {'I2T': .698, 'T2I': .692, 'source': 'arXiv 2201.02772v2 sec. 4.1'}})
    print(json.dumps(agg, indent=1))


if __name__ == '__main__':
    main()
