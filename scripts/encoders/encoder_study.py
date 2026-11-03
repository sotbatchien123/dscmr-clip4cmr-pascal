"""Stronger encoders for CLIP4CMR and DSCMR heads on the author Pascal split (800/200).

Factors (all selected by 3-fold stratified CV on the 800 train rows; test opened once at the end):
  encoder  : frozen rn50 (author mirror), vitl14, vitl14_336, eva02l_336, siglip_so400m_384, dfn5b_h14,
             and concatenations of L2-normalised encoders (ens_*)
  head     : CLIP4CMR family  c4_base | c4_AB (residual adapter + semantic prototypes + per-caption, alpha)
             DSCMR family     ds_<text> (upstream IDCM_NN + calc_loss, text = concat | capmean)
  rerank   : post-hoc on embeddings: none | dual-softmax (tau) | database-side augmentation (k)
             (transductive: uses the whole gallery/query set; reported with and without)
Stages: data | cv | confirm   (confirm = 5 seeds 42-46 on the fixed 720/80 split, best val checkpoint)
Usage: python -m scripts.encoders.encoder_study --stage cv --encoders vitl14_336 eva02l_336 ... [--workers-slice i/n]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
from scripts.clip4cmr import improve as ci  # noqa: E402
from scripts.dscmr import loss_ablation as r1  # noqa: E402
from common.paths import to_repo

AUTHOR = ROOT/'CLIP4CMR/data/pascal_author_split_mirror'
ENC = ROOT/'CLIP4CMR/data/encoders'
DATA = ROOT/'CLIP4CMR/data/author_split_v3'
OUT = ROOT/'reports/improve_v3_20261030'
ALPHAS = (0.1, 0.3, 0.5)
SEEDS = (42, 43, 44, 45, 46)
RERANKS = [('none', None)] + [('dsl', t) for t in (0.01, 0.02, 0.05, 0.1)] + [('dba', k) for k in (2, 5, 10)]


# ---------------------------------------------------------------- data
def build_one(tag):
    """Write DATA/<tag>.npz in author-split row order (same keys as pascal_author_split_mirror)."""
    ref = np.load(AUTHOR/'rn50.npz')
    if tag in ('rn50', 'vitl14'):
        d = dict(np.load(AUTHOR/f'{tag}.npz'))
    else:
        e = np.load(ENC/f'{tag}.npz')
        lid = [str(x) for x in e['ids']]; pos = {k: i for i, k in enumerate(lid)}
        ids = [str(x) for x in ref['ids']]; idx = np.array([pos[i] for i in ids])
        assert (e['labels'][idx] == ref['labels']).all()
        new_of_old = {int(o): n for n, o in enumerate(idx)}
        newowner = np.array([new_of_old[int(o)] for o in e['caption_owner']])
        order = np.argsort(newowner, kind='stable')
        d = {'image': e['image'][idx], 'text_concat': e['text_author'][idx], 'captions': e['captions'][order],
             'caption_owner': newowner[order], 'prompts': e['prompts'], 'labels': ref['labels'], 'ids': ref['ids'],
             'train_index': ref['train_index'], 'test_index': ref['test_index']}
    assert (d['caption_owner'] == ref['caption_owner']).all()
    return d


def build_ensemble(tags):
    parts = [build_one(t) for t in tags]
    n = lambda a: a/np.linalg.norm(a, axis=1, keepdims=True)
    d = dict(parts[0])
    for k in ('image', 'text_concat', 'captions', 'prompts'):
        d[k] = np.concatenate([n(p[k].astype(np.float32)) for p in parts], 1)/np.sqrt(len(parts))
    return d


def ensure_data(tag):
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA/f'{tag}.npz').exists():
        d = build_ensemble(tag[4:].split('+')) if tag.startswith('ens_') else build_one(tag)
        np.savez(DATA/f'{tag}.npz', **{k: (v.astype(np.float32) if v.dtype.kind == 'f' else v) for k, v in d.items()})
    prov = DATA/'provenance.json'
    if not prov.exists():
        prov.write_text(json.dumps({'source': 'author split (HF mirror ids) + encoders in CLIP4CMR/data/encoders',
                                    'rows': 'mirror train 800 then test 200'}, indent=2), encoding='utf-8')


def load(tag, device='cuda'):
    ensure_data(tag); ci.set_feats(DATA)
    return ci.load_backbone(tag, device)


# ---------------------------------------------------------------- heads
def head_list():
    return ['c4_base'] + [f'c4_AB_a{a:g}' for a in ALPHAS] + ['ds_concat', 'ds_capmean']


class Args:  # minimal namespace for ci.train_run / r1.train_run
    def __init__(self, out, device='cuda'):
        self.out, self.device = Path(out), device


def train_head(d, head, rows, val_rows, seed, run, epochs, pan, Model):
    """Train one head; return (result, embed_fn(rows)->(fi, ft) on L2-normalised outputs)."""
    run = Path(run)
    if head.startswith('c4_'):
        variant, alpha = ('base', None) if head == 'c4_base' else ('A_sem+B', float(head.split('_a')[1]))
        res = ci.train_run(Args(OUT), d, pan, rows, val_rows, variant, alpha or .3, seed, run, epochs)
        mode, use_b = ci.VARIANTS[variant]
        def embed(rr):
            ci.ts.seed_all(seed); m = ci.build_model(d, mode, alpha or .3, 'cuda')
            m.load_state_dict(torch.load(run/'best.pt', map_location='cuda')['state_dict']); m.eval()
            with torch.inference_mode():
                _, fi, ft = ci.forward(m, d['image'][rr], ci.eval_text(d, rr, use_b))
            return F.normalize(fi, dim=1), F.normalize(ft, dim=1)
        return res, embed
    text = d['concat'] if head == 'ds_concat' else d['capmean']
    x, t, y = d['image'].float(), text.float(), d['labels']
    a = Args(run.parent); phase = '.'
    res = r1.train_run(a, Model, (x, t), y, np.asarray(rows), np.asarray(val_rows), 'B0_original', seed,
                       phase, epochs, ci.PATIENCE) if not (run/'training_result.json').exists() else json.loads((run/'training_result.json').read_text())
    src = run.parent/'.'/f'B0_original_seed{seed}'
    if src.exists() and src != run:  # r1 names its folder <variant>_seed<s>; move into our run name once
        src.rename(run)
        res['run'] = to_repo(run)  # keep the recorded path in sync with the renamed folder
        r1.write_json(run/'training_result.json', res)
    def embed(rr):
        m = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=d['nc']).cuda()
        m.load_state_dict(torch.load(run/'best.pt', map_location='cuda', weights_only=False)['model']); m.eval()
        with torch.inference_mode():
            zi, zt = m(x[rr], t[rr])[:2]
        return F.normalize(zi, dim=1), F.normalize(zt, dim=1)
    return res, embed


# ---------------------------------------------------------------- rerank + metric
def rerank(fi, ft, kind, p):
    if kind == 'dba':
        def aug(z):
            s = z @ z.T; w = torch.zeros_like(s); top = s.topk(p+1, dim=1).indices
            w.scatter_(1, top, 1.); return F.normalize(w @ z, dim=1)
        fi, ft = aug(fi), aug(ft)
    s = fi @ ft.T
    if kind == 'dsl':
        i2t = s * torch.softmax(s/p, dim=0)   # gallery text j normalised over image queries
        t2i = s.T * torch.softmax(s.T/p, dim=0)
        return i2t.cpu().numpy(), t2i.cpu().numpy()
    return s.cpu().numpy(), s.T.cpu().numpy()


def score(fi, ft, y):
    out = {}
    for kind, p in RERANKS:
        a, b = rerank(fi, ft, kind, p)
        out[f'{kind}{"" if p is None else p}'] = (r1.metrics(a, y, y)['mAP@ALL'], r1.metrics(b, y, y)['mAP@ALL'])
    return out


# ---------------------------------------------------------------- stages
def stage_cv(tags, heads, pan, Model):
    for tag in tags:
        d = load(tag); y_tr = d['labels'][d['train']]; folds = ci.stratified_folds(y_tr)
        for head in heads:
            f_out = OUT/'cv'/tag/head/'cv.json'
            if f_out.exists(): continue
            per = []
            for f in range(3):
                tr = d['train'][np.concatenate([folds[g] for g in range(3) if g != f])]; va = d['train'][folds[f]]
                res, embed = train_head(d, head, tr, va, 42, OUT/'cv'/tag/head/f'fold{f}', ci.TUNE_EPOCHS, pan, Model)
                fi, ft = embed(va)
                per.append({'val_best': res['best_validation_mean_map'], 'epochs': res['epochs'], 'rerank': score(fi, ft, d['labels'][va])})
            summ = {k: [float(np.mean([p['rerank'][k][0] for p in per])), float(np.mean([p['rerank'][k][1] for p in per]))]
                    for k in per[0]['rerank']}
            ci.write_json(f_out, {'tag': tag, 'head': head, 'folds': per, 'cv_mean': summ})
            print(f'CV {tag:28s} {head:12s} none {np.mean(summ["none"])*100:6.2f}  best-rerank '
                  f'{max(np.mean(v) for v in summ.values())*100:6.2f}', flush=True)


def stage_confirm(configs, pan, Model):
    """configs: list of (name, tag, head, rerank_key). Test opened once, after all training."""
    trained = []
    for name, tag, head, rk in configs:
        d = load(tag); inner, val = ci.fixed_split(d['labels'][d['train']])
        for s in SEEDS:
            run = OUT/'confirm'/name/f'seed{s}'
            res, embed = train_head(d, head, d['train'][inner], d['train'][val], s, run, ci.CONFIRM_EPOCHS, pan, Model)
            trained.append((name, tag, head, rk, s, res, embed))
    rows = []
    for name, tag, head, rk, s, res, embed in trained:
        d = load(tag); fi, ft = embed(d['test']); y = d['labels'][d['test']]
        sc = score(fi, ft, y)
        rows.append({'config': name, 'tag': tag, 'head': head, 'seed': s, 'epochs': res['epochs'],
                     'best_epoch': res['best_epoch'], 'rerank_key': rk,
                     'I2T': sc['none'][0], 'T2I': sc['none'][1], 'I2T_rr': sc[rk][0], 'T2I_rr': sc[rk][1]})
        np.savez_compressed(OUT/'confirm'/name/f'seed{s}'/'test_features.npz', image=fi.cpu().numpy(), text=ft.cpu().numpy(), label=y)
    ci.write_json(OUT/'confirm_results.json', rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('data', 'cv', 'confirm'), required=True)
    p.add_argument('--encoders', nargs='*', default=[])
    p.add_argument('--heads', nargs='*', default=None)
    p.add_argument('--configs', type=Path, help='json list of [name, tag, head, rerank_key] for confirm')
    p.add_argument('--vram', type=float, default=0.2)
    a = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(a.vram)
    OUT.mkdir(parents=True, exist_ok=True)
    pan = ci.ts.prototype_loss(); Model, _ = r1.upstream_definitions(ROOT)
    if a.stage == 'data':
        for t in a.encoders: ensure_data(t); print('data', t, flush=True)
    elif a.stage == 'cv':
        stage_cv(a.encoders, a.heads or head_list(), pan, Model)
    else:
        rows = stage_confirm([tuple(c) for c in json.loads(a.configs.read_text())], pan, Model)
        print(json.dumps(rows[:2], indent=1))


if __name__ == '__main__':
    main()
