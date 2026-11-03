"""DSCMR loss ablation (round 1): original loss vs SupCon common-space loss vs adaptive weights.

Fixed author features, checkpoint selected on a 720/80 validation split, no refit.
--source-root is the repository root (default: this repository); upstream DSCMR code is read
from <source-root>/DSCMR/upstream. All outputs go to --out; a STOP file pauses at the next epoch.
Usage: python -m scripts.dscmr.loss_ablation --out reports/assignment3_20261029 [--stage check|smoke|all]
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.spatial.distance import cdist

VARIANTS = ('B0_original', 'B1_supcon', 'B2_adaptive', 'B3_combined')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def upstream_definitions(root):
    path = root / 'DSCMR/upstream/model.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in ('ImgNN', 'TextNN', 'IDCM_NN')]
    ns = {'nn': torch.nn, 'F': F}
    exec(compile(tree, str(path), 'exec'), ns)
    loss_path = root / 'DSCMR/upstream/train_model.py'
    tree = ast.parse(loss_path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('calc_loss', 'calc_label_sim')]
    ln = {'torch': torch}
    exec(compile(tree, str(loss_path), 'exec'), ln)
    return ns['IDCM_NN'], ln['calc_loss']


def common_original(zi, zt, labels):
    same = labels[:, None].eq(labels[None, :]).to(zi.dtype)
    def pair(a, b):
        # Preserve upstream denominator clamp and factor 1/2 exactly.
        den = a.square().sum(1).sqrt()[:, None] * b.square().sum(1).sqrt()[None, :]
        theta = a @ b.T / den.clamp_min(1e-6) / 2
        return (F.softplus(theta) - same * theta).mean()
    return pair(zi, zi) + pair(zi, zt) + pair(zt, zt)


def supervised_contrastive(zi, zt, labels, temperature=0.1):
    """Keep II, TT, and symmetric IT relations; exclude self only within modality.

    Average negative log probability over ALL same-class positives per anchor.
    Within-modality singleton anchors are skipped; paired cross positives remain.
    """
    zi, zt = F.normalize(zi, dim=1), F.normalize(zt, dim=1)
    same = labels[:, None].eq(labels[None, :])
    def pair(a, b, exclude_self=False):
        logits = a @ b.T / temperature
        allowed = torch.ones_like(same)
        if exclude_self:
            allowed.fill_diagonal_(False)
        positive = same & allowed
        valid = positive.sum(1) > 0
        if not valid.any():
            return (a.sum() + b.sum()) * 0
        logits = logits[valid]
        allowed, positive = allowed[valid], positive[valid]
        logp = logits - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True)
        return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
    return pair(zi, zi, True) + pair(zt, zt, True) + (pair(zi, zt) + pair(zt, zi)) / 2


def components(outputs, labels, contrastive=False, temperature=0.1):
    zi, zt, pi, pt = outputs
    target = F.one_hot(labels, pi.shape[1]).to(pi.dtype)
    label = torch.linalg.vector_norm(pi-target, dim=1).mean() + torch.linalg.vector_norm(pt-target, dim=1).mean()
    common = supervised_contrastive(zi, zt, labels, temperature) if contrastive else common_original(zi, zt, labels)
    paired = torch.linalg.vector_norm(zi-zt, dim=1).mean()
    return torch.stack((label, common, paired))


class AdaptiveWeights:
    """Bounded EMA loss-speed weighting, not GradNorm and not claimed novel.

    Epoch e uses only losses through e-1. Slower relative decrease gets a higher
    multiplier. No gradients through weights; baseline coefficients retained.
    """
    def __init__(self, enabled, base=(1., .001, .1), warmup=10):
        self.enabled = enabled
        self.base = np.asarray(base, dtype=float)
        self.warmup = warmup
        self.ema = None
        self.previous = None

    def weights(self, epoch):
        mult = np.ones(3)
        if self.enabled and epoch > self.warmup and self.previous is not None:
            ratio = self.ema / np.maximum(self.previous, 1e-8)
            ratio = np.clip(ratio, 0.1, 10.)
            mult = np.clip(np.exp(np.clip((ratio-ratio.mean()) / .5, -10, 10)), .5, 2.)
        return self.base * mult

    def update(self, values):
        self.previous = None if self.ema is None else self.ema.copy()
        self.ema = np.asarray(values) if self.ema is None else .9*self.ema + .1*np.asarray(values)

    def state(self):
        return {'ema': None if self.ema is None else self.ema.tolist(),
                'previous': None if self.previous is None else self.previous.tolist()}

    def restore(self, state):
        for key in ('ema', 'previous'):
            setattr(self, key, None if state[key] is None else np.array(state[key]))


def metrics(scores, qlabels, glabels):
    relevant = np.asarray(qlabels)[:, None] == np.asarray(glabels)[None, :]
    if not np.isfinite(scores).all() or not relevant.any(1).all():
        raise ValueError('Nonfinite scores or query without a relevant candidate')
    order = np.argsort(-scores, axis=1, kind='stable')
    ranked = np.take_along_axis(relevant, order, axis=1)
    precision = ranked.cumsum(1) / np.arange(1, ranked.shape[1]+1)
    ap = (precision*ranked).sum(1) / ranked.sum(1)
    return {'mAP@ALL': float(ap.mean()), **{f'R@{k}': float(ranked[:, :k].any(1).mean()) for k in (1, 5, 10)}}


def evaluate(model, x, t, y):
    model.eval()
    with torch.inference_mode():
        zi, zt = [a.cpu().numpy() for a in model(x, t)[:2]]
    scores = -cdist(zi, zt, 'cosine')
    return {'I2T': metrics(scores, y, y), 'T2I': metrics(scores.T, y, y)}


def load_split(data_root, split, classes=None):
    arrays = []
    for field in ('img', 'txt', 'img_lab'):
        arr = loadmat(data_root / f'{split}_{field}.mat')[f'{split}_{field}']
        arrays.append(arr)
    x, t, y = arrays
    y = y.reshape(-1)
    classes = np.unique(y) if classes is None else classes
    mapping = {v: i for i, v in enumerate(classes.tolist())}
    y = np.array([mapping[v] for v in y], dtype=np.int64)
    x, t = x.astype(np.float32), t.astype(np.float32)
    assert len(x) == len(t) == len(y) and np.isfinite(x).all() and np.isfinite(t).all()
    return x, t, y, classes


def make_split(y, seed=42):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        rng.shuffle(ids)
        n = max(1, round(len(ids)*.1))
        val.extend(ids[:n]); train.extend(ids[n:])
    return np.sort(train), np.sort(val)


def balanced_batches(indices, y, seed, epoch):
    """Each row once/epoch, up to 5 samples/class/batch; same for all variants."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    pools = [rng.permutation(indices[y[indices] == c]) for c in np.unique(y[indices])]
    for start in range(0, max(map(len, pools)), 5):
        batch = np.concatenate([p[start:start+5] for p in pools])
        rng.shuffle(batch)
        yield batch


def self_test(root, Model, reference):
    seed_all(7)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    out = [torch.randn(6, d, dtype=torch.double, requires_grad=True) for d in (8, 8, 3, 3)]
    oh = F.one_hot(y, 3).double()
    ours = (components(out, y) * torch.tensor([1., .001, .1], dtype=torch.double)).sum()
    theirs = reference(*out, oh, oh, .001, .1)
    torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-10)
    go = torch.autograd.grad(ours, out, retain_graph=True)
    gt = torch.autograd.grad(theirs, out)
    for a, b in zip(go, gt):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    # Brute force multi-positive SupCon reference, including self exclusion.
    a, b = F.normalize(out[0], dim=1), F.normalize(out[1], dim=1)
    def slow(a, b, intra):
        values = []
        for i in range(len(y)):
            eligible = [j for j in range(len(y)) if not (intra and i == j)]
            pos = [j for j in eligible if y[j] == y[i]]
            if pos:
                den = torch.logsumexp(torch.stack([a[i] @ b[j] / .1 for j in eligible]), 0)
                values.append(torch.stack([den - a[i] @ b[j] / .1 for j in pos]).mean())
        return torch.stack(values).mean()
    expected = slow(a, a, True) + slow(b, b, True) + (slow(a, b, False)+slow(b, a, False))/2
    actual = supervised_contrastive(out[0], out[1], y)
    torch.testing.assert_close(actual, expected)
    for labels in (y, torch.arange(6), torch.zeros(6, dtype=torch.long)):
        loss = supervised_contrastive(out[0], out[1], labels)
        grads = torch.autograd.grad(loss, out[:2], retain_graph=True)
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
    weights = AdaptiveWeights(True)
    for e in range(1, 40):
        w = weights.weights(e)
        assert np.all(w >= weights.base*.5) and np.all(w <= weights.base*2)
        if e <= 10: np.testing.assert_array_equal(w, weights.base)
        weights.update([1/(e+1), 1., 1/(e+1)**2])
    clone = AdaptiveWeights(True); clone.restore(weights.state())
    np.testing.assert_array_equal(clone.weights(40), weights.weights(40))
    # Known AP: relevant ranks 1 and 3 gives (1 + 2/3)/2.
    got = metrics(np.array([[3., 2., 1.]]), [0], [0, 1, 0])
    assert abs(got['mAP@ALL'] - 5/6) < 1e-12 and got['R@1'] == 1
    spec = importlib.util.spec_from_file_location('old_metrics', root/'common/metrics.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    scores = np.random.default_rng(8).normal(size=(6, 6))
    yy = y.numpy(); rel = old.label_relevance(yy, yy)
    for s, r in ((scores, rel), (scores.T, rel.T)):
        check = metrics(s, yy, yy)
        assert abs(check['mAP@ALL'] - old.map_all(s, r)['mAP@ALL']) < 1e-12
        assert all(check[k] == v for k, v in old.recall_at_k(s, r).items())
    for variant in VARIANTS:
        model = Model(img_input_dim=16, text_input_dim=7, img_output_dim=12,
                      text_output_dim=12, minus_one_dim=8, output_dim=3)
        loss = components(model(torch.randn(6, 16), torch.randn(6, 7)), y,
                          variant in (VARIANTS[1], VARIANTS[3])).sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
    return {'passed': True, 'checks': ['original_loss_value_and_gradient_equivalence',
        'supcon_bruteforce_equivalence', 'singleton_and_same_class_gradients',
        'adaptive_bounds_warmup_restore', 'metric_hand_example_and_existing_evaluator',
        'all_four_variants_gradient_flow']}


def train_run(args, Model, tensors, y, train, val, variant, seed, phase, epochs, patience):
    run = args.out / phase / f'{variant}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'training_result.json').exists():
        return json.loads((run/'training_result.json').read_text())
    seed_all(seed)
    x, t = tensors
    model = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(np.unique(y))).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
    adaptive = AdaptiveWeights(variant in (VARIANTS[2], VARIANTS[3]))
    contrastive = variant in (VARIANTS[1], VARIANTS[3])
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
        for batch in balanced_batches(train, y, seed, epoch):
            ix = torch.as_tensor(batch, device=args.device)
            opt.zero_grad(set_to_none=True)
            terms = components(model(x[ix], t[ix]), labels[ix], contrastive)
            loss = (terms*wt).sum()
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss: {run}, epoch {epoch}')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite or missing gradients')
            opt.step()
            sums += terms.detach().cpu().numpy()*len(batch)
        raw = sums/len(train)
        adaptive.update(raw)
        vm = evaluate(model, x[val], t[val], y[val])
        score = (vm['I2T']['mAP@ALL']+vm['T2I']['mAP@ALL'])/2
        if score > best+1e-8:
            best, best_epoch = score, epoch
            save_checkpoint(run/'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'validation': vm})
        history.append({'epoch': epoch, 'raw_losses': raw.tolist(), 'weights': ws.tolist(),
                        'weighted_loss': float(raw@ws), 'validation': vm,
                        'seconds': time.perf_counter()-tick})
        write_json(run/'history.json', history)
        status = {'state': 'training', 'phase': phase, 'variant': variant, 'seed': seed,
                  'epoch': epoch, 'max_epochs': epochs, 'best_epoch': best_epoch,
                  'best_validation_mean_map': best, 'pid': os.getpid(),
                  'updated_unix': time.time(), 'run': str(run)}
        write_json(args.out/'status.json', status)
        if epoch % 10 == 0: checkpoint(epoch)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(status), flush=True)
    elapsed = elapsed_before + time.perf_counter()-start
    result = {'phase': phase, 'variant': variant, 'seed': seed, 'epochs': epoch,
              'budget': epochs, 'patience': patience, 'best_epoch': best_epoch,
              'best_validation_mean_map': best, 'elapsed_seconds': elapsed,
              'seconds_per_epoch_median': float(np.median([h['seconds'] for h in history])),
              'run': str(run), 'test_evaluated': False}
    write_json(run/'training_result.json', result)
    # Completed runs retain best checkpoint, history and result; optimizer resume
    # state is no longer necessary. Only remove this run's own temporary state.
    (run/'last.pt').unlink(missing_ok=True)
    return result


def report(args, results):
    rows = []
    for r in results:
        for direction in ('I2T', 'T2I'):
            rows.append({'variant': r['variant'], 'seed': r['seed'], 'epochs': r['epochs'],
                         'best_epoch': r['best_epoch'], 'direction': direction, **r['test'][direction]})
    with (args.out/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = {}
    lines = ['# DSCMR Assignment 3 — controlled loss ablations', '',
             'Fixed author features; 720 train / 80 validation / 100 test. No full-train refit.',
             'Checkpoint selection: mean bidirectional validation mAP. Test evaluated after all training.',
             'R@K is query hit rate (at least one same-class candidate), NOT fraction of all relevant items.',
             'Values below are percentages, mean ± sample standard deviation over 3 training seeds.', '',
             '| Variant | Direction | mAP@ALL | R@1 | R@5 | R@10 |', '|---|---|---|---|---|---|']
    for variant in VARIANTS:
        aggregate[variant] = {}
        for d in ('I2T', 'T2I'):
            subset = [r for r in rows if r['variant'] == variant and r['direction'] == d]
            stats = {k: {'mean': float(np.mean([r[k] for r in subset])),
                         'std': float(np.std([r[k] for r in subset], ddof=1))}
                     for k in ('mAP@ALL', 'R@1', 'R@5', 'R@10')}
            aggregate[variant][d] = stats
            lines.append('| '+variant+' | '+d+' | '+' | '.join(f"{v['mean']*100:.2f} ± {v['std']*100:.2f}" for v in stats.values())+' |')
    lines += ['', '## Limits', '',
              '- This first ablation uses preregistered coefficients (1, 0.001, 0.1), temperature 0.1; no hyperparameter search.',
              '- Fixed-weight tuning is still required before claiming superiority over a tuned baseline.',
              '- Shared fixed split, three initialization/batch seeds; this is not cross-validation.',
              '- SupCon keeps image-image, text-text and symmetric cross-modal relations; excludes intra-modal self comparisons.',
              '- Adaptive method is bounded EMA relative-loss-speed weighting, not GradNorm; novelty is not established.',
              '- Author bundle has 100 test pairs. Results are not exact paper reproduction and not directly comparable with older refit runs.',
              '- Historical test results have already been seen; this test is not a newly blinded holdout. No test-driven tuning in this run.']
    write_json(args.out/'aggregate.json', aggregate)
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--stage', choices=('check', 'smoke', 'all'), default='all')
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
        Model, original_loss = upstream_definitions(args.source_root)
        tests = self_test(args.source_root, Model, original_loss)
        write_json(args.out/'verification.json', tests)
        print('VERIFICATION PASSED', flush=True)
        if args.stage == 'check': return
        data_root = args.source_root/'DSCMR/upstream/data/pascal'
        x, t, y, classes = load_split(data_root, 'train')
        train, val = make_split(y)
        assert len(train) == 720 and len(val) == 80 and not set(train)&set(val)
        assert len(set(train)|set(val)) == len(y)
        batch_ids = np.concatenate(list(balanced_batches(train, y, 42, 1)))
        np.testing.assert_array_equal(np.sort(batch_ids), train)
        protocol = {'variants': list(VARIANTS), 'training_seeds': [42, 43, 44], 'split_seed': 42,
            'train_indices': train.tolist(), 'validation_indices': val.tolist(),
            'feature_dimensions': [x.shape[1], t.shape[1]], 'classes': classes.tolist(),
            'smoke_epochs': 10, 'pilot_epochs': 200, 'main_epochs': 500, 'main_patience': 50,
            'selection': 'mean I2T/T2I validation mAP; min_delta=1e-8; no refit',
            'batching': 'up to 5 per class, each training row once per epoch, common to all variants',
            'optimizer': {'name': 'Adam', 'lr': .0001, 'betas': [.5, .999]},
            'base_weights': [1., .001, .1], 'supcon_temperature': .1,
            'adaptive': {'warmup': 10, 'ema_decay': .9, 'ratio_temperature': .5,
                         'multiplier_bounds': [.5, 2.], 'uses': 'previous training epoch EMA ratios only'},
            'hyperparameter_search': False, 'test_used_for_selection': False,
            'recall_definition': 'fraction of queries with >=1 same-class hit in top K',
            'source_root': str(args.source_root), 'torch_version': torch.__version__,
            'device': args.device, 'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
            'source_sha256': {str(f.relative_to(args.source_root)): digest(f) for f in
                [args.source_root/'DSCMR/upstream/model.py', args.source_root/'DSCMR/upstream/train_model.py']},
            'train_data_sha256': {f.name: digest(f) for f in sorted(data_root.glob('train_*.mat'))},
            'runner_sha256': digest(__file__)}
        proto_path = args.out/'protocol.json'
        if proto_path.exists() and json.loads(proto_path.read_text()) != protocol:
            raise ValueError('Protocol/code changed; choose a new output directory')
        write_json(proto_path, protocol)
        tensors = (torch.as_tensor(x, device=args.device), torch.as_tensor(t, device=args.device))
        smoke = [train_run(args, Model, tensors, y, train, val, v, 42, 'smoke', 10, 0) for v in VARIANTS]
        write_json(args.out/'timing.json', {'smoke': smoke,
                   'main_upper_bound_seconds': sum(r['seconds_per_epoch_median']*500*3 for r in smoke),
                   'pilot_upper_bound_seconds': sum(r['seconds_per_epoch_median']*200 for r in smoke)})
        if args.stage == 'smoke':
            write_json(args.out/'status.json', {'state': 'smoke_completed', 'pid': os.getpid(), 'updated_unix': time.time()})
            return
        for v in VARIANTS:
            train_run(args, Model, tensors, y, train, val, v, 42, 'pilot', 200, 0)
        completed = []
        for seed in (42, 43, 44):
            for v in VARIANTS:
                completed.append(train_run(args, Model, tensors, y, train, val, v, seed, 'main', 500, 50))
        # Open test arrays only after every main run has selected its checkpoint.
        tx, tt, ty, _ = load_split(data_root, 'test', classes)
        write_json(args.out/'test_manifest.json', {'pairs': len(ty), 'sha256':
            {f.name: digest(f) for f in sorted(data_root.glob('test_*.mat'))}})
        tx, tt = torch.as_tensor(tx, device=args.device), torch.as_tensor(tt, device=args.device)
        results = []
        for r in completed:
            path = Path(r['run'])/'result.json'
            if path.exists():
                results.append(json.loads(path.read_text())); continue
            model = Model(img_input_dim=x.shape[1], text_input_dim=t.shape[1], output_dim=len(classes)).to(args.device)
            ckpt = torch.load(Path(r['run'])/'best.pt', map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            result = {**r, 'test': evaluate(model, tx, tt, ty), 'test_evaluated': True}
            write_json(path, result); results.append(result)
        report(args, results)
        write_json(args.out/'status.json', {'state': 'completed', 'main_runs': len(results),
                   'total_main_epochs': sum(r['epochs'] for r in results),
                   'report': str(args.out/'REPORT.md'), 'updated_unix': time.time()})
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
