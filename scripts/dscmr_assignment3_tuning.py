"""DSCMR Assignment 3, round 2: equal-budget coefficient tuning, then confirmation.

Round 1 (dscmr_assignment3.py) used coefficients (1, 0.001, 0.1). With alpha=0.001
the common-space term is ~0.3-1% of the total loss, so replacing it (SupCon) or
re-weighting it barely changes training. This round tunes the coefficients with the
same budget for the baseline and for SupCon, then confirms with 5 seeds.

Stage tune:    3-fold stratified CV on the 800 train pairs (split seed 42), seed 42.
               B0 grid alpha x beta (9); B1 grid lambda x tau (9) with beta = B0 best.
               Score = mean over folds of the best mean I2T/T2I validation mAP.
Stage confirm: B0*, B1*, B2* (B0* + adaptive), B3* (B1* + adaptive) x seeds 42-46 on
               the round-1 720/80 split; checkpoint by validation; test opened only
               after every confirmation run has finished.

Reuses the verified loss, batching, metric and self-test code of round 1 unchanged.
All outputs go to --out. A STOP file pauses at the next epoch boundary.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import dscmr_assignment3 as r1

B0_GRID = [(a, b) for a in (0.001, 0.01, 0.1) for b in (0.03, 0.1, 0.3)]
B1_GRID = [(lam, tau) for lam in (0.03, 0.1, 0.3) for tau in (0.05, 0.1, 0.2)]
TUNE_SEED, TUNE_EPOCHS, TUNE_PATIENCE = 42, 300, 50
CONFIRM_SEEDS, CONFIRM_EPOCHS, CONFIRM_PATIENCE = (42, 43, 44, 45, 46), 500, 50


def retry_io(fn, *args, attempts=20):
    """OneDrive sync can briefly lock a file during replace; retry a bounded number of times."""
    for k in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if k == attempts-1:
                raise
            time.sleep(.25*(k+1))


def write_json(path, value):
    retry_io(r1.write_json, path, value)


def save_checkpoint(path, value):
    retry_io(r1.save_checkpoint, path, value)


def config(name, alpha, beta, contrastive=False, tau=0.1, adaptive=False):
    return {'name': name, 'weights': [1.0, alpha, beta], 'contrastive': contrastive,
            'tau': tau, 'adaptive': adaptive}


def tag(cfg):
    a, b = cfg['weights'][1:]
    t = f"_t{cfg['tau']:g}" if cfg['contrastive'] else ''
    return f"{cfg['name']}_a{a:g}_b{b:g}{t}"


def stratified_folds(y, k=3, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == c))
        for f, part in enumerate(np.array_split(ids, k)):
            folds[f].extend(part.tolist())
    return [np.sort(np.array(f)) for f in folds]


def train_run(args, Model, tensors, y, train, val, cfg, seed, run, epochs, patience):
    """Same loop as round 1, parameterised by coefficients/temperature/adaptive flag."""
    run.mkdir(parents=True, exist_ok=True)
    if (run/'training_result.json').exists():
        return json.loads((run/'training_result.json').read_text())
    r1.seed_all(seed)
    x, t = tensors
    model = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(np.unique(y))).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
    adaptive = r1.AdaptiveWeights(cfg['adaptive'], base=tuple(cfg['weights']))
    labels = torch.as_tensor(y, device=args.device)
    best, best_epoch, first, history, elapsed_before = -1., 0, 1, [], 0.
    if (run/'last.pt').exists():
        state = torch.load(run/'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        adaptive.restore(state['adaptive'])
        best, best_epoch, first = state['best'], state['best_epoch'], state['epoch']+1
        history, elapsed_before = state['history'], state['elapsed_seconds']
    start = time.perf_counter()

    def checkpoint(epoch):
        save_checkpoint(run/'last.pt', {'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'adaptive': adaptive.state(), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
            'history': history, 'elapsed_seconds': elapsed_before+time.perf_counter()-start})
    epoch = first-1
    for epoch in range(first, epochs+1):
        if (args.out/'STOP').exists():
            checkpoint(epoch-1)
            raise InterruptedError('STOP requested; checkpoint saved at epoch boundary')
        if patience and epoch-1-best_epoch >= patience:
            epoch -= 1
            break
        tick = time.perf_counter()
        model.train()
        sums = np.zeros(3)
        ws = adaptive.weights(epoch)
        wt = torch.tensor(ws, dtype=x.dtype, device=args.device)
        for batch in r1.balanced_batches(train, y, seed, epoch):
            ix = torch.as_tensor(batch, device=args.device)
            opt.zero_grad(set_to_none=True)
            terms = r1.components(model(x[ix], t[ix]), labels[ix], cfg['contrastive'], cfg['tau'])
            loss = (terms*wt).sum()
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss: {run}, epoch {epoch}')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite or missing gradients')
            opt.step()
            sums += terms.detach().cpu().numpy()*len(batch)
        raw = sums/len(train)
        adaptive.update(raw)
        vm = r1.evaluate(model, x[val], t[val], y[val])
        score = (vm['I2T']['mAP@ALL']+vm['T2I']['mAP@ALL'])/2
        if score > best+1e-8:
            best, best_epoch = score, epoch
            save_checkpoint(run/'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'validation': vm})
        contrib = raw*ws
        history.append({'epoch': epoch, 'raw_losses': raw.tolist(), 'weights': ws.tolist(),
                        'weighted_terms': contrib.tolist(), 'term_share': (contrib/contrib.sum()).tolist(),
                        'weighted_loss': float(contrib.sum()), 'validation': vm,
                        'seconds': time.perf_counter()-tick})
        if epoch % 10 == 0 or epoch == 1:
            write_json(run/'history.json', history)
            checkpoint(epoch)
            write_json(args.out/'status.json', {'state': 'training', 'run': run.name,
                'epoch': epoch, 'max_epochs': epochs, 'best_epoch': best_epoch,
                'best_validation_mean_map': best, 'pid': os.getpid(), 'updated_unix': time.time()})
    write_json(run/'history.json', history)
    share = np.array([h['term_share'] for h in history[:best_epoch]]).mean(0)
    result = {'config': cfg, 'seed': seed, 'epochs': epoch, 'budget': epochs, 'patience': patience,
              'best_epoch': best_epoch, 'best_validation_mean_map': best,
              'mean_term_share_until_best': share.tolist(),
              'elapsed_seconds': elapsed_before+time.perf_counter()-start,
              'run': str(run), 'test_evaluated': False}
    write_json(run/'training_result.json', result)
    (run/'last.pt').unlink(missing_ok=True)
    print(f"done {run.parent.name}/{run.name}: epochs={epoch} best={best_epoch} val={best*100:.2f}", flush=True)
    return result


def cv_score(args, Model, tensors, y, folds, cfg):
    scores = []
    for f, val in enumerate(folds):
        train = np.sort(np.concatenate([folds[g] for g in range(len(folds)) if g != f]))
        r = train_run(args, Model, tensors, y, train, val, cfg, TUNE_SEED,
                      args.out/'tune'/tag(cfg)/f'fold{f}', TUNE_EPOCHS, TUNE_PATIENCE)
        scores.append(r['best_validation_mean_map'])
    return {'tag': tag(cfg), 'config': cfg, 'fold_scores': scores,
            'mean': float(np.mean(scores)), 'std': float(np.std(scores, ddof=1))}


def pick(rows):
    # Highest mean CV score; ties keep the earlier (pre-registered order) entry.
    return max(enumerate(rows), key=lambda e: (round(e[1]['mean'], 10), -e[0]))[1]


def report(args, tuning, selected, chosen, results):
    pct = lambda v: f'{v*100:.2f}'
    names = [c['name'] for c in chosen]
    rows = [{'variant': r['config']['name'], 'seed': r['seed'], 'epochs': r['epochs'],
             'best_epoch': r['best_epoch'], 'direction': d, **r['test'][d]}
            for r in results for d in ('I2T', 'T2I')]
    with (args.out/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    get = lambda n, s, d, k='mAP@ALL': next(r[k] for r in rows if r['variant'] == n and r['seed'] == s and r['direction'] == d)
    agg, L = {}, ['# DSCMR Assignment 3 — vòng 2: tuning hệ số + xác nhận 5 seed', '']
    L += ['## Tuning (3-fold CV trên 800 mẫu train, seed 42; không dùng test)', '',
          '| Cấu hình | α / λ | β | τ | CV mAP trung bình (%) | Fold 1 / 2 / 3 |', '|---|---:|---:|---:|---:|---|']
    for t in tuning:
        c = t['config']; mark = ' **← chọn**' if t['tag'] in selected else ''
        L.append(f"| {c['name']}{mark} | {c['weights'][1]:g} | {c['weights'][2]:g} | "
                 f"{c['tau']:g}{'' if c['contrastive'] else ' (–)'} | {pct(t['mean'])} ± {pct(t['std'])} | "
                 + ' / '.join(pct(s) for s in t['fold_scores'])+' |')
    L += ['', '## Cấu hình xác nhận', '']
    for c in chosen:
        L.append(f"- **{c['name']}**: hệ số (1, {c['weights'][1]:g}, {c['weights'][2]:g})"
                 + (f", SupCon τ={c['tau']:g}" if c['contrastive'] else ', loss gốc')
                 + (', trọng số thích nghi' if c['adaptive'] else ', trọng số cố định'))
    L += ['', f'## Kết quả test (720/80 split, {len(CONFIRM_SEEDS)} seed, % trung bình ± độ lệch chuẩn mẫu)', '',
          '| Cấu hình | Chiều | mAP@ALL | R@1 | R@5 | R@10 |', '|---|---|---:|---:|---:|---:|']
    for n in names:
        agg[n] = {}
        for d in ('I2T', 'T2I'):
            sub = [r for r in rows if r['variant'] == n and r['direction'] == d]
            st = {k: {'mean': float(np.mean([r[k] for r in sub])), 'std': float(np.std([r[k] for r in sub], ddof=1))}
                  for k in ('mAP@ALL', 'R@1', 'R@5', 'R@10')}
            agg[n][d] = st
            L.append(f'| {n} | {d} | '+' | '.join(f"{pct(v['mean'])} ± {pct(v['std'])}" for v in st.values())+' |')
    L += ['', f'## Chênh lệch mAP theo cặp seed so với {names[0]} (điểm %)', '',
          '| Cấu hình | Chiều | '+' | '.join(f'Seed {s}' for s in CONFIRM_SEEDS)+' | Trung bình | Số seed tăng |',
          '|---|---|'+'---:|'*(len(CONFIRM_SEEDS)+2)]
    paired = {}
    for n in names[1:]:
        for d in ('I2T', 'T2I'):
            diff = [(get(n, s, d)-get(names[0], s, d))*100 for s in CONFIRM_SEEDS]
            paired[f'{n}/{d}'] = diff
            L.append(f'| {n} | {d} | '+' | '.join(f'{v:+.2f}' for v in diff)
                     + f' | {np.mean(diff):+.2f} | {sum(v > 0 for v in diff)}/{len(diff)} |')
    L += ['', '## Tỷ trọng từng thành phần loss (trung bình tới epoch tốt nhất, seed 42)', '',
          '| Cấu hình | Loss nhãn | Không gian chung | Cặp ảnh–văn bản |', '|---|---:|---:|---:|']
    for r in results:
        if r['seed'] == CONFIRM_SEEDS[0]:
            L.append(f"| {r['config']['name']} | "+' | '.join(f'{v*100:.1f}%' for v in r['mean_term_share_until_best'])+' |')
    L += ['', '## Phạm vi', '',
          '- Tuning dùng ngân sách bằng nhau: 9 cấu hình × 3 fold cho B0 và cho B1; B2*/B3* không tuning thêm.',
          '- CV score lấy epoch tốt nhất của mỗi fold nên hơi lạc quan, nhưng như nhau cho mọi cấu hình; chỉ dùng để xếp hạng.',
          '- Tập test 100 cặp; chênh lệch dưới ~1 điểm có thể là nhiễu. 5 seed không phải 5 fold; không tuyên bố ý nghĩa thống kê.',
          '- Tập test đã được xem ở các lượt trước; lượt này không dùng test để chọn cấu hình hay checkpoint.',
          '- Trọng số thích nghi: quy tắc EMA tốc độ giảm loss có giới hạn, không phải GradNorm; tính mới chưa được chứng minh.']
    write_json(args.out/'aggregate.json', {'aggregate': agg, 'paired_diff_points': paired})
    (args.out/'REPORT.md').write_text('\n'.join(L)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--stage', choices=('check', 'smoke', 'tune', 'all'), default='all')
    args = p.parse_args()
    args.source_root, args.out = args.source_root.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    lock = args.out/'run.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        Model, original_loss = r1.upstream_definitions(args.source_root)
        write_json(args.out/'verification.json', r1.self_test(args.source_root, Model, original_loss))
        print('VERIFICATION PASSED', flush=True)
        if args.stage == 'check': return
        data_root = args.source_root/'DSCMR/upstream/data/pascal'
        x, t, y, classes = r1.load_split(data_root, 'train')
        folds = stratified_folds(y)
        assert sorted(np.concatenate(folds).tolist()) == list(range(len(y)))
        train, val = r1.make_split(y)
        assert len(train) == 720 and len(val) == 80
        protocol = {'round': 2, 'b0_grid_alpha_beta': [list(g) for g in B0_GRID], 'b1_grid_lambda_tau': [list(g) for g in B1_GRID],
            'b1_beta': 'beta of selected B0', 'tune_seed': TUNE_SEED, 'tune_epochs': TUNE_EPOCHS,
            'tune_patience': TUNE_PATIENCE, 'cv': '3-fold stratified over 800 train, split seed 42',
            'cv_folds': [f.tolist() for f in folds],
            'cv_score': 'mean over folds of best mean I2T/T2I validation mAP; ties keep earlier grid entry',
            'confirm': 'B0*, B1*, B2*=B0*+adaptive, B3*=B1*+adaptive', 'confirm_seeds': list(CONFIRM_SEEDS),
            'confirm_epochs': CONFIRM_EPOCHS, 'confirm_patience': CONFIRM_PATIENCE,
            'confirm_split': 'round-1 720/80 split (seed 42)', 'validation_indices': val.tolist(),
            'optimizer': {'name': 'Adam', 'lr': .0001, 'betas': [.5, .999]},
            'adaptive': 'round-1 AdaptiveWeights with base = selected coefficients',
            'test_used_for_selection': False, 'device': args.device, 'torch_version': torch.__version__,
            'source_sha256': {n: r1.digest(args.source_root/'DSCMR/upstream'/n) for n in ('model.py', 'train_model.py')},
            'train_data_sha256': {f.name: r1.digest(f) for f in sorted(data_root.glob('train_*.mat'))},
            'runner_sha256': r1.digest(__file__), 'round1_runner_sha256': r1.digest(r1.__file__)}
        proto = args.out/'protocol.json'
        if proto.exists() and json.loads(proto.read_text()) != protocol:
            raise ValueError('Protocol/code changed; choose a new output directory')
        write_json(proto, protocol)
        tensors = (torch.as_tensor(x, device=args.device), torch.as_tensor(t, device=args.device))
        if args.stage == 'smoke':
            for cfg in (config('B0', .01, .1), config('B1', .1, .1, True, .1, True)):
                train_run(args, Model, tensors, y, train, val, cfg, 42, args.out/'smoke'/tag(cfg), 5, 0)
            write_json(args.out/'status.json', {'state': 'smoke_completed', 'updated_unix': time.time()})
            return
        b0 = [cv_score(args, Model, tensors, y, folds, config('B0_original', a, b)) for a, b in B0_GRID]
        b0_best = pick(b0)
        beta = b0_best['config']['weights'][2]
        b1 = [cv_score(args, Model, tensors, y, folds, config('B1_supcon', lam, beta, True, tau)) for lam, tau in B1_GRID]
        b1_best = pick(b1)
        tuning = b0+b1
        write_json(args.out/'tuning.json', {'rows': tuning, 'b0_best': b0_best['tag'], 'b1_best': b1_best['tag']})
        print('TUNING DONE', b0_best['tag'], b0_best['mean'], b1_best['tag'], b1_best['mean'], flush=True)
        if args.stage == 'tune': return
        c0, c1 = b0_best['config'], b1_best['config']
        chosen = [c0, c1, {**c0, 'name': 'B2_adaptive', 'adaptive': True}, {**c1, 'name': 'B3_combined', 'adaptive': True}]
        chosen = [{**c, 'name': c['name']+'*'} for c in chosen]
        done = [train_run(args, Model, tensors, y, train, val, c, s,
                          args.out/'confirm'/f"{c['name'].rstrip('*')}_seed{s}", CONFIRM_EPOCHS, CONFIRM_PATIENCE)
                for s in CONFIRM_SEEDS for c in chosen]
        # Open test arrays only after every confirmation run has selected its checkpoint.
        tx, tt, ty, _ = r1.load_split(data_root, 'test', classes)
        write_json(args.out/'test_manifest.json', {'pairs': len(ty), 'sha256':
            {f.name: r1.digest(f) for f in sorted(data_root.glob('test_*.mat'))}})
        tx, tt = torch.as_tensor(tx, device=args.device), torch.as_tensor(tt, device=args.device)
        results = []
        for r in done:
            path = Path(r['run'])/'result.json'
            if path.exists():
                results.append(json.loads(path.read_text())); continue
            model = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(classes)).to(args.device)
            model.load_state_dict(torch.load(Path(r['run'])/'best.pt', map_location=args.device, weights_only=False)['model'])
            res = {**r, 'test': r1.evaluate(model, tx, tt, ty), 'test_evaluated': True}
            write_json(path, res); results.append(res)
        report(args, tuning, (b0_best['tag'], b1_best['tag']), chosen, results)
        write_json(args.out/'status.json', {'state': 'completed', 'confirm_runs': len(results),
            'tune_runs': len(tuning)*3, 'report': str(args.out/'REPORT.md'), 'updated_unix': time.time()})
        print('COMPLETED '+str(args.out/'REPORT.md'), flush=True)
    except InterruptedError as exc:
        write_json(args.out/'status.json', {'state': 'paused', 'error': str(exc), 'updated_unix': time.time()})
    except Exception as exc:
        write_json(args.out/'status.json', {'state': 'failed', 'error': repr(exc), 'updated_unix': time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
