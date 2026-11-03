"""Diagnostic: DSCMR trained exactly like the upstream repo, to explain the gap to the paper.

Upstream main.py/train_model.py: all 800 train pairs, DataLoader batch 100 without shuffle,
Adam 1e-4 betas (.5,.999), alpha 1e-3, beta 1e-1, 500 epochs; test mAP (upstream
fx_calc_map_label, cosine) after every epoch and the checkpoint with the best TEST mean is kept.
That selection uses the test set, so it is optimistic; reported here only as a diagnostic next to
(a) the last epoch and (b) our validation-selected B0 numbers. Seeds 42-46.
Usage: python -m scripts.baseline.dscmr_author_protocol [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.spatial  # noqa: F401  (upstream evaluator)
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.dscmr import loss_ablation as r1  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
EPOCHS, BATCH = 500, 100


def upstream_eval():
    import ast
    tree = ast.parse((ROOT/'DSCMR/upstream/evaluate.py').read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fx_calc_map_label']
    ns = {'np': np, 'scipy': sys.modules['scipy']}
    exec(compile(tree, 'upstream_evaluate', 'exec'), ns)
    return ns['fx_calc_map_label']


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'reports/dscmr_author_protocol_20261030'
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    Model, calc_loss = r1.upstream_definitions(ROOT)
    fx = upstream_eval()
    data = ROOT/'DSCMR/upstream/data/pascal'
    x, t, y, classes = r1.load_split(data, 'train')
    xt, tt, yt, _ = r1.load_split(data, 'test', classes)
    dev = 'cuda'
    X, T = torch.tensor(x, device=dev), torch.tensor(t, device=dev)
    XT, TT = torch.tensor(xt, device=dev), torch.tensor(tt, device=dev)
    onehot = torch.nn.functional.one_hot(torch.tensor(y, device=dev), len(classes)).float()
    runs = []
    for seed in SEEDS:
        r1.seed_all(seed)
        model = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(classes)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
        hist = []
        for epoch in range(1, EPOCHS+1):
            model.train()
            for s in range(0, len(x), BATCH):  # upstream: shuffle=False
                opt.zero_grad()
                a, b, pa, pb = model(X[s:s+BATCH], T[s:s+BATCH])
                loss = calc_loss(a, b, pa, pb, onehot[s:s+BATCH], onehot[s:s+BATCH], 1e-3, 1e-1)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                fa, fb = [z.cpu().numpy() for z in model(XT, TT)[:2]]
            hist.append({'epoch': epoch, 'I2T': float(fx(fa, fb, yt)), 'T2I': float(fx(fb, fa, yt))})
        best = max(hist, key=lambda h: (h['I2T']+h['T2I'])/2)
        runs.append({'seed': seed, 'best_test_epoch': best['epoch'], 'best_test': best, 'last': hist[-1]})
        r1.write_json(out/f'history_seed{seed}.json', hist)
        print(f"seed {seed}: best-on-test ep{best['epoch']} {best['I2T']*100:.2f}/{best['T2I']*100:.2f}  "
              f"last {hist[-1]['I2T']*100:.2f}/{hist[-1]['T2I']*100:.2f}", flush=True)
    agg = {k: {d: [float(np.mean([r[k][d] for r in runs])), float(np.std([r[k][d] for r in runs], ddof=1))]
               for d in ('I2T', 'T2I')} for k in ('best_test', 'last')}
    r1.write_json(out/'result.json', {'protocol': __doc__, 'runs': runs, 'aggregate_mean_sd': agg,
                                      'paper': {'I2T': .710, 'T2I': .722, 'source': 'DSCMR/upstream/Result.jpg'}})
    print(json.dumps(agg, indent=1))


if __name__ == '__main__':
    main()
