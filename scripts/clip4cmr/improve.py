"""CLIP4CMR improvement study: A (residual adapter + semantic CLIP prototypes), B (per-caption
text), D (backbone RN50 vs ViT-L/14). Frozen CLIP features from scripts/data/build_multicaption.py.

Baseline = the unchanged upstream CLIP4CMR head + PAN loss, trained exactly like
scripts/baseline/train_baseline.py (Adam 1e-4, betas (.5,.999), batch 300, global numpy shuffle after seeding).
  A_res : output f = l2norm((1-a)*l2norm(x) + a*l2norm(MLP(x))) stays in CLIP space; random
          fixed prototypes (upstream init). Control that isolates the residual adapter.
  A_sem : as A_res, prototypes = CLIP text embeddings of "a photo of a {class}." (fixed).
  B     : training text = one randomly sampled caption per row per epoch; validation/test
          text = mean of that image's raw caption embeddings (no 77-token truncation).
Stage tune   : a in {0.1, 0.3, 0.5} for A_sem, 3-fold stratified CV on 800 train (seed 42),
               per backbone; A_res and A_sem+B reuse the chosen a. Test never opened.
Stage confirm: per backbone {base, B, A_res, A_sem, A_sem+B} x seeds 42-46 on the fixed
               720/80 split of scripts/baseline/train_baseline.py (seed 42); checkpoint = best mean validation
               mAP, max 500 epochs, patience 50; test (200 pairs) opened once at the end.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
from scripts.baseline import train_baseline as ts  # noqa: E402
from common.metrics import label_relevance, map_all, recall_at_k  # noqa: E402
from common.paths import from_repo, to_repo

FEATS = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption'
BACKBONES = ('rn50', 'vitl14')
ALPHAS = (0.1, 0.3, 0.5)
SEEDS = (42, 43, 44, 45, 46)
TUNE_EPOCHS, CONFIRM_EPOCHS, PATIENCE, BATCH = 300, 500, 50, 300
VARIANTS = {'base': (None, False), 'B': (None, True), 'A_res': ('res', False),
            'A_sem': ('sem', False), 'A_sem+B': ('sem', True)}


def l2n(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


class ResidualProto(nn.Module):
    """Upstream ImgNN/TextNN as residual adapters in CLIP space; fixed prototypes."""
    def __init__(self, upstream, dim, prototypes, alpha):
        super().__init__()
        ns = upstream
        self.imgnn = ns.imgnn.__class__(input_dim=dim, mindum_dim=256, out_dim=dim)
        self.textnn = ns.textnn.__class__(input_dim=dim, mindum_dim=256, out_dim=dim)
        self.alpha = alpha
        self.register_buffer('centers', prototypes)  # (dim, classes), unit columns

    def forward(self, img, text):
        a = self.alpha
        fi = l2n((1-a)*l2n(img) + a*l2n(self.imgnn(img)))
        ft = l2n((1-a)*l2n(text) + a*l2n(self.textnn(text)))
        return self.centers, fi, ft


def write_json(path, value):
    for k in range(20):  # tolerate brief file locks (sync/AV); bounded retries
        try:
            return ts.write_json(Path(path), value)
        except PermissionError:
            if k == 19: raise
            time.sleep(.25*(k+1))


def set_feats(path):
    global FEATS
    FEATS = Path(path).resolve()


def load_backbone(key, device):
    d = np.load(FEATS/f'{key}.npz')
    caps = torch.as_tensor(d['captions'], device=device)
    owner = d['caption_owner']
    groups = [np.flatnonzero(owner == i) for i in range(len(d['ids']))]
    tr, te = d['train_index'], d['test_index']
    T = lambda a: torch.as_tensor(a, device=device)
    capmean = np.stack([d['captions'][g].mean(0) for g in groups])
    return {'image': T(d['image']), 'concat': T(d['text_concat']), 'capmean': T(capmean), 'caps': caps,
            'groups': groups, 'labels': d['labels'].astype(np.int64), 'train': tr, 'test': te, 'prompts': d['prompts'],
            'dim': d['image'].shape[1], 'nc': len(np.unique(d['labels']))}


def fixed_split(y_train):
    # Identical to scripts/baseline/train_baseline.py for seed 42 (inner 720 / validation 80).
    rng = np.random.default_rng(42); inner, val = [], []
    for c in range(len(np.unique(y_train))):
        ids = np.flatnonzero(y_train == c); rng.shuffle(ids)
        nv = max(1, int(round(len(ids)*0.1))); val.extend(ids[:nv]); inner.extend(ids[nv:])
    return np.sort(inner), np.sort(val)


def stratified_folds(y, k=3, seed=42):
    rng = np.random.default_rng(seed); folds = [[] for _ in range(k)]
    for c in np.unique(y):
        for f, part in enumerate(np.array_split(rng.permutation(np.flatnonzero(y == c)), k)):
            folds[f].extend(part.tolist())
    return [np.sort(np.array(f)) for f in folds]


def build_model(data, mode, alpha, device):
    base = ts.network('CLIP4CMR', data['nc'], data['dim'], data['dim'])
    if mode is None:
        return base.to(device)
    if mode == 'sem':
        protos = torch.as_tensor(data['prompts'], dtype=torch.float32).T
    else:  # same initialisation rule as upstream centers, in CLIP dimension
        protos = torch.randn(data['dim'], data['nc']); nn.init.kaiming_normal_(protos, mode='fan_out')
    protos = protos / protos.norm(dim=0, keepdim=True)
    return ResidualProto(base, data['dim'], protos, alpha).to(device)


def forward(model, x, t):
    out = model(x, t)
    return out[0], out[1], out[2]


def evaluate(model, x, t, y):
    model.eval()
    with torch.inference_mode():
        _, fi, ft = forward(model, x, t)
    fi, ft = fi.cpu().numpy(), ft.cpu().numpy()
    s = -cdist(fi, ft, 'euclidean'); rel = label_relevance(y, y)
    return {d: {**map_all(a, r), **recall_at_k(a, r)} for d, a, r in (('I2T', s, rel), ('T2I', s.T, rel.T))}, fi, ft


def eval_text(data, rows, use_b):
    return (data['capmean'] if use_b else data['concat'])[rows]


def train_run(args, data, pan, rows, val_rows, variant, alpha, seed, run, epochs):
    mode, use_b = VARIANTS[variant]
    run.mkdir(parents=True, exist_ok=True)
    if (run/'training_result.json').exists():
        return json.loads((run/'training_result.json').read_text(encoding='utf-8'))
    ts.seed_all(seed)
    model = build_model(data, mode, alpha, args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
    y_all = torch.as_tensor(data['labels'], device=args.device)
    xv, tv, yv = data['image'][val_rows], eval_text(data, val_rows, use_b), data['labels'][val_rows]
    best, best_epoch, history, start = -1., 0, [], time.perf_counter()
    epoch = 0
    for epoch in range(1, epochs+1):
        if (args.out/'STOP').exists():
            raise InterruptedError('STOP requested')
        if epoch-1-best_epoch >= PATIENCE:
            epoch -= 1; break
        model.train(); order = np.array(rows, copy=True); np.random.shuffle(order)
        if use_b:  # separate stream: never disturbs the global shuffle RNG
            crng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7]))
            pick = {int(i): int(crng.choice(data['groups'][i])) for i in rows}
        loss_sum = 0.
        for off in range(0, len(order), BATCH):
            ix = order[off:off+BATCH]; it = torch.as_tensor(ix, device=args.device)
            text = data['caps'][torch.as_tensor([pick[int(i)] for i in ix], device=args.device)] if use_b else data['concat'][it]
            opt.zero_grad(set_to_none=True)
            centers, fi, ft = forward(model, data['image'][it], text)
            loss = pan(fi, centers.T, y_all[it]) + pan(ft, centers.T, y_all[it])
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss {run} epoch {epoch}')
            loss.backward(); opt.step(); loss_sum += loss.item()*len(ix)
        vm, _, _ = evaluate(model, xv, tv, yv)
        score = (vm['I2T']['mAP@ALL']+vm['T2I']['mAP@ALL'])/2
        history.append({'epoch': epoch, 'train_loss': loss_sum/len(order), 'validation': vm})
        if score > best+1e-8:
            best, best_epoch = score, epoch
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, run/'best.pt')
    write_json(run/'history.json', history)
    result = {'variant': variant, 'alpha': alpha if mode else None, 'seed': seed, 'epochs': epoch,
              'best_epoch': best_epoch, 'best_validation_mean_map': best,
              'elapsed_seconds': time.perf_counter()-start, 'run': to_repo(run), 'test_evaluated': False}
    write_json(run/'training_result.json', result)
    print(f'done {run.relative_to(args.out)} epochs={epoch} best={best_epoch} val={best*100:.2f}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'reports/clip4cmr_improve_20261030')
    p.add_argument('--device', default='cuda')
    p.add_argument('--stage', choices=('smoke', 'all'), default='all')
    p.add_argument('--feats', type=Path, default=FEATS, help='feature dir (default: local UIUC seed-42 split)')
    args = p.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    set_feats(args.feats)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    lock = args.out/'run.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        pan = ts.prototype_loss()
        datas = {k: load_backbone(k, args.device) for k in BACKBONES}
        y_tr = datas['rn50']['labels'][datas['rn50']['train']]
        inner, val = fixed_split(y_tr)
        folds = stratified_folds(y_tr)
        for d in datas.values():
            assert (d['labels'][d['train']] == y_tr).all()
        protocol = {'variants': {k: list(v) for k, v in VARIANTS.items()}, 'alphas': list(ALPHAS), 'seeds': list(SEEDS),
                    'tune': '3-fold CV on 800 train, seed 42, A_sem only, max 300 / patience 50',
                    'confirm': 'fixed 720/80 split (train_selected seed 42), max 500 / patience 50, best mean val mAP',
                    'cv_folds': [f.tolist() for f in folds], 'validation_positions': val.tolist(),
                    'optimizer': 'Adam lr 1e-4 betas (0.5,0.999), batch 300', 'loss': 'upstream PAN on image + text',
                    'distance': 'euclidean on l2-normalised outputs (as train_selected)',
                    'features': json.loads((FEATS/'provenance.json').read_text(encoding='utf-8')),
                    'test_used_for_selection': False, 'runner_sha256': ts.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'torch': torch.__version__}
        proto = args.out/'protocol.json'
        if proto.exists() and json.loads(proto.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol/code changed; use a new --out')
        write_json(proto, protocol)
        tr = lambda d, pos: d['train'][pos]  # positions within the 800 train rows -> global row ids
        if args.stage == 'smoke':
            for k, d in datas.items():
                for v in VARIANTS:
                    train_run(args, d, pan, tr(d, inner), tr(d, val), v, .3, 42, args.out/'smoke'/k/v, 3)
            write_json(args.out/'status.json', {'state': 'smoke_completed'}); return
        tuning, chosen = {}, {}
        for k, d in datas.items():
            rows = []
            for a in ALPHAS:
                sc = [train_run(args, d, pan, tr(d, np.concatenate([folds[g] for g in range(3) if g != f])), tr(d, folds[f]),
                                'A_sem', a, 42, args.out/'tune'/k/f'alpha{a:g}'/f'fold{f}', TUNE_EPOCHS)['best_validation_mean_map']
                      for f in range(3)]
                rows.append({'alpha': a, 'fold_scores': sc, 'mean': float(np.mean(sc)), 'std': float(np.std(sc, ddof=1))})
            tuning[k] = rows
            chosen[k] = max(enumerate(rows), key=lambda e: (round(e[1]['mean'], 10), -e[0]))[1]['alpha']
            write_json(args.out/'status.json', {'state': 'tuned', 'backbone': k, 'alpha': chosen[k]})
        write_json(args.out/'tuning.json', {'rows': tuning, 'chosen_alpha': chosen})
        print('TUNING', chosen, flush=True)
        done = [train_run(args, d, pan, tr(d, inner), tr(d, val), v, chosen[k], s, args.out/'confirm'/k/f'{v}_seed{s}', CONFIRM_EPOCHS)
                | {'backbone': k} for s in SEEDS for k, d in datas.items() for v in VARIANTS]
        results = []
        for r in done:  # test opened only now, after every confirmation run finished
            d = datas[r['backbone']]; mode, use_b = VARIANTS[r['variant']]
            ts.seed_all(r['seed']); model = build_model(d, mode, r['alpha'], args.device)
            model.load_state_dict(torch.load(from_repo(r['run'])/'best.pt', map_location=args.device)['state_dict'])
            m, fi, ft = evaluate(model, d['image'][d['test']], eval_text(d, d['test'], use_b), d['labels'][d['test']])
            np.savez_compressed(from_repo(r['run'])/'test_features.npz', image=fi, text=ft, label=d['labels'][d['test']])
            r = {**r, 'test': m, 'test_evaluated': True}; write_json(from_repo(r['run'])/'result.json', r); results.append(r)
        report(args, tuning, chosen, results, datas)
        write_json(args.out/'status.json', {'state': 'completed', 'confirm_runs': len(results), 'tune_runs': 18})
        print('COMPLETED', flush=True)
    except InterruptedError as e:
        write_json(args.out/'status.json', {'state': 'paused', 'error': str(e)})
    except Exception as e:
        write_json(args.out/'status.json', {'state': 'failed', 'error': repr(e)}); raise
    finally:
        lock.unlink(missing_ok=True)


def zero_shot(d):
    y = d['labels'][d['test']]; rel = label_relevance(y, y); out = {}
    for name, T in (('concat', d['concat']), ('capmean', d['capmean'])):
        s = (l2n(d['image'][d['test']]) @ l2n(T[d['test']]).T).cpu().numpy()
        out[name] = {'I2T': map_all(s, rel)['mAP@ALL'], 'T2I': map_all(s.T, rel.T)['mAP@ALL']}
    return out


def report(args, tuning, chosen, results, datas):
    rows = [{'backbone': r['backbone'], 'variant': r['variant'], 'alpha': r['alpha'], 'seed': r['seed'], 'epochs': r['epochs'],
             'best_epoch': r['best_epoch'], 'direction': dr, **{k: v for k, v in r['test'][dr].items() if k != 'queries_without_positive'}}
            for r in results for dr in ('I2T', 'T2I')]
    with (args.out/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    get = lambda b, v, s, dr: next(x['mAP@ALL'] for x in rows if (x['backbone'], x['variant'], x['seed'], x['direction']) == (b, v, s, dr))
    names = {'rn50': 'RN50', 'vitl14': 'ViT-L/14'}
    split = json.loads((FEATS/'provenance.json').read_text(encoding='utf-8')).get('source', 'split cục bộ UIUC seed 42')
    L = ['# CLIP4CMR — cải tiến A (adapter phần dư + prototype ngữ nghĩa), B (từng caption), D (backbone)', '',
         f'Đặc trưng: `{FEATS.relative_to(ROOT)}` ({split}).', '',
         'Test 200 cặp. mAP@ALL %, trung bình ± SD mẫu qua 5 seed; checkpoint theo validation 720/80.', '',
         '## Tuning α cho A_sem (3-fold CV trên 800 train, không dùng test)', '', '| Backbone | α | CV mAP (%) | Fold 1 / 2 / 3 |', '|---|---:|---:|---|']
    for b, rs in tuning.items():
        for r in rs:
            L.append(f"| {names[b]} | {r['alpha']:g}{' **← chọn**' if r['alpha'] == chosen[b] else ''} | {r['mean']*100:.2f} ± {r['std']*100:.2f} | "
                     + ' / '.join(f'{v*100:.2f}' for v in r['fold_scores'])+' |')
    agg = {}
    L += ['', '## Kết quả test', '', '| Backbone | Cấu hình | I2T mAP | T2I mAP | TB 2 chiều | I2T R@1 | T2I R@1 |', '|---|---|---:|---:|---:|---:|---:|']
    for b in BACKBONES:
        zs = zero_shot(datas[b]); agg[f'{b}/zero_shot'] = zs
        for name, z in zs.items():
            L.append(f"| {names[b]} | *zero-shot CLIP ({name}, không huấn luyện)* | {z['I2T']*100:.2f} | {z['T2I']*100:.2f} | {(z['I2T']+z['T2I'])*50:.2f} | | |")
        for v in VARIANTS:
            st = {dr: {k: (float(np.mean(a)), float(np.std(a, ddof=1))) for k in ('mAP@ALL', 'R@1')
                       for a in [[x[k] for x in rows if (x['backbone'], x['variant'], x['direction']) == (b, v, dr)]]} for dr in ('I2T', 'T2I')}
            agg[f'{b}/{v}'] = st
            L.append(f"| {names[b]} | {v} | {st['I2T']['mAP@ALL'][0]*100:.2f} ± {st['I2T']['mAP@ALL'][1]*100:.2f} | "
                     f"{st['T2I']['mAP@ALL'][0]*100:.2f} ± {st['T2I']['mAP@ALL'][1]*100:.2f} | "
                     f"{(st['I2T']['mAP@ALL'][0]+st['T2I']['mAP@ALL'][0])*50:.2f} | {st['I2T']['R@1'][0]*100:.1f} | {st['T2I']['R@1'][0]*100:.1f} |")
    L += ['', '## Chênh lệch mAP theo cặp seed so với base cùng backbone (điểm %)', '',
          '| Backbone | Cấu hình | Chiều | '+' | '.join(f'S{s}' for s in SEEDS)+' | TB | Seed tăng |', '|---|---|---|'+'---:|'*(len(SEEDS)+2)]
    paired = {}
    for b in BACKBONES:
        for v in list(VARIANTS)[1:]:
            for dr in ('I2T', 'T2I'):
                diff = [(get(b, v, s, dr)-get(b, 'base', s, dr))*100 for s in SEEDS]; paired[f'{b}/{v}/{dr}'] = diff
                L.append(f'| {names[b]} | {v} | {dr} | '+' | '.join(f'{x:+.2f}' for x in diff)+f' | {np.mean(diff):+.2f} | {sum(x > 0 for x in diff)}/{len(diff)} |')
    L += ['', '## Hiệu ứng backbone (D): ViT-L/14 − RN50, cùng cấu hình, cùng seed (điểm %)', '', '| Cấu hình | I2T TB | T2I TB |', '|---|---:|---:|']
    for v in VARIANTS:
        dd = {dr: np.mean([(get('vitl14', v, s, dr)-get('rn50', v, s, dr))*100 for s in SEEDS]) for dr in ('I2T', 'T2I')}
        L.append(f"| {v} | {dd['I2T']:+.2f} | {dd['T2I']:+.2f} |")
    L += ['', '## Phạm vi', '',
          ('- Split 800/200 cục bộ từ UIUC, không phải split của bài báo; không so trực tiếp với số liệu bài báo hay với DSCMR (test 100).'
           if 'mirror' not in split else
           '- Split 800/200 lấy từ bản mirror bên thứ ba của clip_train/test.pkl (khớp pipeline tác giả, chưa được tác giả xác nhận). '
           'Bài báo chọn checkpoint bằng test; ở đây chọn bằng validation 80 mẫu tách từ train.'),
          '- Base giữ nguyên siêu tham số gốc (không tuning); A_sem được chọn α trên CV (3 giá trị). A_res, A_sem+B dùng α đó.',
          '- B đổi cả biểu diễn văn bản lúc đánh giá (trung bình caption) — đây chính là phương pháp, không phải đổi tập test.',
          '- Prototype từ prompt CLIP và adapter phần dư đã có trong các nghiên cứu CLIP (vd. CLIP-Adapter); đóng góp là áp dụng/phân tích cho CLIP4CMR.',
          '- Test 200 cặp; chênh lệch < ~1 điểm có thể là nhiễu. 5 seed không phải 5 fold.']
    write_json(args.out/'aggregate.json', {'aggregate': agg, 'paired_diff_points': paired})
    (args.out/'REPORT.md').write_text('\n'.join(L)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
