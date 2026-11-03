"""Partial fine-tuning of the DFN5B ViT-H/14 image tower (last N blocks).

Tokens entering block L-N are cached once (frozen prefix, fp16, all 1000 author-split rows; no
labels used). Trainable: last N residual blocks + ln_post + proj (fp32 master weights, bf16
autocast, gradient checkpointing) jointly with a retrieval head:
  c4_AB : CLIP4CMR residual adapter + semantic prototypes + per-caption text (alpha 0.3), PAN loss
  ds    : DSCMR IDCM_NN + upstream loss (balanced batches), text = caption mean
Text side stays frozen (DFN5B text features from CLIP4CMR/data/author_split_v3/dfn5b_h14.npz).
Stages: cv (3-fold on train, seed 42) | confirm (5 seeds, 720/80 val selection, test once).
Usage: python -m scripts.encoders.finetune_image_tower --stage cv --head c4_AB --blocks 2
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.feat_libs'))
from scripts.clip4cmr import improve as ci  # noqa: E402
from scripts.dscmr import loss_ablation as r1  # noqa: E402
from scripts.encoders import encoder_study as v  # noqa: E402

TAG, MODEL, PRE = 'dfn5b_h14', 'ViT-H-14-quickgelu', 'dfn5b'
CACHE = ROOT/'CLIP4CMR/data/author_split_v3'
OUT = v.OUT/'finetune'
MAX_EPOCHS, PATIENCE, LR_TAIL, LR_HEAD = 100, 20, 1e-5, 1e-4


def cache_tokens(blocks):
    path = CACHE/f'{TAG}_tokens_L-{blocks}.pt'
    if path.exists():  # our own cache file (contains nn.Module copies of the tail)
        return torch.load(path, weights_only=False)
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRE, device='cpu')
    vis = model.visual.half().cuda().eval()
    ids = [str(x) for x in np.load(CACHE/f'{TAG}.npz')['ids']]
    toks, pooled = [], []
    with torch.inference_mode():
        for s in range(0, len(ids), 8):
            px = torch.stack([preprocess(Image.open(ROOT/'CLIP4CMR/data/pascal_uiuc_seed42/images'/f'{i}.jpg').convert('RGB'))
                              for i in ids[s:s+8]]).cuda().half()
            x = vis._embeds(px)
            for blk in vis.transformer.resblocks[:-blocks]: x = blk(x)
            toks.append(x.cpu())
            y = x
            for blk in vis.transformer.resblocks[-blocks:]: y = blk(y)
            p, _ = vis._pool(y); pooled.append((p @ vis.proj).float().cpu())
    tail = {'resblocks': copy.deepcopy(vis.transformer.resblocks[-blocks:]).float().cpu(),
            'ln_post': copy.deepcopy(vis.ln_post).float().cpu(), 'proj': vis.proj.detach().float().cpu().clone(),
            'pool_type': vis.pool_type, 'final_ln_after_pool': vis.final_ln_after_pool}
    out = {'tokens': torch.cat(toks), 'pooled_check': torch.cat(pooled), 'tail': tail}
    ref = torch.as_tensor(np.load(CACHE/f'{TAG}.npz')['image'])
    out['check_cos_min'] = float(F.cosine_similarity(out['pooled_check'], ref).min())
    torch.save(out, path); print('cached tokens', out['tokens'].shape, 'check cos min', out['check_cos_min'], flush=True)
    return out


class Tail(torch.nn.Module):
    def __init__(self, tail):
        super().__init__()
        self.blocks = copy.deepcopy(tail['resblocks']); self.ln_post = copy.deepcopy(tail['ln_post'])
        self.proj = torch.nn.Parameter(tail['proj'].clone()); self.pool_type = tail['pool_type']
        self.after = tail['final_ln_after_pool']

    def forward(self, x):
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.training else b(x)
        if self.after:
            p = x[:, 0] if self.pool_type == 'tok' else x[:, 1:].mean(1); p = self.ln_post(p)
        else:
            x = self.ln_post(x); p = x[:, 0] if self.pool_type == 'tok' else x[:, 1:].mean(1)
        return p @ self.proj


def run(head, blocks, rows, val_rows, seed, out, d, cache, pan, Model):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        return json.loads((out/'result.json').read_text())
    ci.ts.seed_all(seed)
    tail = Tail(cache['tail']).cuda()
    toks = cache['tokens']
    y_all = torch.as_tensor(d['labels'], device='cuda')
    if head == 'c4_AB':
        h = ci.build_model(d, 'sem', .3, 'cuda'); text_eval = d['capmean']
        fwd = lambda img, txt: h(img, txt)
    else:
        h = Model(img_input_dim=d['dim'], text_input_dim=d['dim'], output_dim=d['nc']).cuda(); text_eval = d['capmean']
        fwd = lambda img, txt: h(img, txt)
    opt = torch.optim.Adam([{'params': tail.parameters(), 'lr': LR_TAIL}, {'params': h.parameters(), 'lr': LR_HEAD}], betas=(.5, .999))

    def embed(rr):
        tail.eval(); h.eval(); feats = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for s in range(0, len(rr), 50):
                feats.append(tail(toks[rr[s:s+50]].cuda().float()).float())
        img = torch.cat(feats)
        with torch.inference_mode():
            o = fwd(img, text_eval[rr])
        zi, zt = (o[1], o[2]) if head == 'c4_AB' else (o[0], o[1])
        return F.normalize(zi.float(), dim=1), F.normalize(zt.float(), dim=1)

    best, best_ep, hist = -1., 0, []
    for epoch in range(1, MAX_EPOCHS+1):
        if epoch-1-best_ep >= PATIENCE: epoch -= 1; break
        tail.train(); h.train()
        if head == 'c4_AB':
            order = np.array(rows, copy=True); np.random.shuffle(order)
            crng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7]))
            pick = {int(i): int(crng.choice(d['groups'][i])) for i in rows}
            batches = [order[o:o+100] for o in range(0, len(order), 100)]
        else:
            batches = list(r1.balanced_batches(np.asarray(rows), d['labels'], seed, epoch))
        for ix in batches:
            it = torch.as_tensor(ix, device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):  # micro-chunks: checkpoint recompute per chunk, same loss
                img = torch.cat([tail(toks[ix[c:c+25]].cuda().float()) for c in range(0, len(ix), 25)])
            img = img.float()
            if head == 'c4_AB':
                txt = d['caps'][torch.as_tensor([pick[int(i)] for i in ix], device='cuda')]
                c, fi, ft = fwd(img, txt)
                loss = pan(fi, c.T, y_all[it]) + pan(ft, c.T, y_all[it])
            else:
                loss = (r1.components(fwd(img, d['capmean'][it]), y_all[it]) * torch.tensor([1., .001, .1], device='cuda')).sum()
            loss.backward(); opt.step()
        fi, ft = embed(val_rows); yv = d['labels'][val_rows]
        s = (fi @ ft.T).cpu().numpy()
        sc = (r1.metrics(s, yv, yv)['mAP@ALL'] + r1.metrics(s.T, yv, yv)['mAP@ALL'])/2
        hist.append({'epoch': epoch, 'val': sc})
        if sc > best + 1e-8:
            best, best_ep = sc, epoch
            torch.save({'tail': tail.state_dict(), 'head': h.state_dict(), 'epoch': epoch}, out/'best.pt')
    res = {'head': head, 'blocks': blocks, 'seed': seed, 'epochs': epoch, 'best_epoch': best_ep, 'best_val': best, 'history': hist}
    ci.write_json(out/'result.json', res)
    print(f'done {out.relative_to(OUT)} ep={epoch} best={best_ep} val={best*100:.2f}', flush=True)
    del tail, h, opt, embed; free()
    return res


def free():
    import gc
    gc.collect(); torch.cuda.empty_cache()


def reload(head, blocks, out, d, cache, Model):
    """rebuild embed() from best.pt"""
    tail = Tail(cache['tail']).cuda()
    h = ci.build_model(d, 'sem', .3, 'cuda') if head == 'c4_AB' else Model(img_input_dim=d['dim'], text_input_dim=d['dim'], output_dim=d['nc']).cuda()
    ck = torch.load(out/'best.pt', map_location='cuda'); tail.load_state_dict(ck['tail']); h.load_state_dict(ck['head'])
    tail.eval(); h.eval(); toks = cache['tokens']

    def embed(rr):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            img = torch.cat([tail(toks[rr[s:s+50]].cuda().float()).float() for s in range(0, len(rr), 50)])
        with torch.inference_mode():
            o = h(img.float(), d['capmean'][rr])
        zi, zt = (o[1], o[2]) if head == 'c4_AB' else (o[0], o[1])
        return F.normalize(zi.float(), dim=1), F.normalize(zt.float(), dim=1)
    return embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('cv', 'confirm'), required=True)
    p.add_argument('--head', choices=('c4_AB', 'ds'), required=True)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--vram', type=float, default=0.4)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.vram); torch.backends.cuda.matmul.allow_tf32 = False
    pan = ci.ts.prototype_loss(); Model, _ = r1.upstream_definitions(ROOT)
    d = v.load(TAG); cache = cache_tokens(a.blocks)
    name = f'{a.head}_L{a.blocks}'
    if a.stage == 'cv':
        y_tr = d['labels'][d['train']]; folds = ci.stratified_folds(y_tr); per = []
        for f in range(3):
            tr = d['train'][np.concatenate([folds[g] for g in range(3) if g != f])]; va = d['train'][folds[f]]
            o = OUT/'cv'/name/f'fold{f}'
            run(a.head, a.blocks, tr, va, 42, o, d, cache, pan, Model)
            fi, ft = reload(a.head, a.blocks, o, d, cache, Model)(va)
            per.append(v.score(fi, ft, d['labels'][va])); del fi, ft; free()
        summ = {k: [float(np.mean([q[k][i] for q in per])) for i in (0, 1)] for k in per[0]}
        ci.write_json(OUT/'cv'/name/'cv.json', {'cv_mean': summ, 'folds': per})
        print('CV', name, {k: round(np.mean(x)*100, 2) for k, x in summ.items()}, flush=True)
    else:
        inner, val = ci.fixed_split(d['labels'][d['train']]); rows = []
        for s in v.SEEDS:
            run(a.head, a.blocks, d['train'][inner], d['train'][val], s, OUT/'confirm'/name/f'seed{s}', d, cache, pan, Model)
        for s in v.SEEDS:  # test opened only after all seeds trained
            o = OUT/'confirm'/name/f'seed{s}'; fi, ft = reload(a.head, a.blocks, o, d, cache, Model)(d['test'])
            sc = v.score(fi, ft, d['labels'][d['test']]); rows.append({'seed': s, **{k: list(x) for k, x in sc.items()}})
            np.savez_compressed(o/'test_features.npz', image=fi.cpu().numpy(), text=ft.cpu().numpy(), label=d['labels'][d['test']])
            del fi, ft; free()
        ci.write_json(OUT/'confirm'/name/'test_results.json', rows)
        print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
