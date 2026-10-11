"""Adapter around the AUTHORS' model/loss/metric; upstream code is required.
Modes distinguish the upstream test-based model selection from fixed-last-epoch
evaluation. They must never be merged into a single paper-reproduction claim.
"""
import argparse
import datetime
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--selection', choices=['fixed-last', 'upstream-test'], default='fixed-last')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 2:
        p.error('epochs >= 1, batch-size >= 2 required')
    upstream = Path(a.upstream).resolve()
    for name in ['model.py', 'train_model.py', 'load_data.py', 'evaluate.py']:
        if not (upstream / name).is_file():
            raise FileNotFoundError(upstream / name)
    sys.path.insert(0, str(upstream))
    import torch
    import numpy as np
    from model import IDCM_NN
    from load_data import get_loader
    from train_model import train_model, calc_loss
    from evaluate import fx_calc_map_label
    from scipy.io import loadmat
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    if a.selection == 'upstream-test' and a.device == 'cpu' and torch.cuda.is_available():
        raise RuntimeError('Upstream loop forces CUDA when present. Use fixed-last for CPU or upstream-test with cuda.')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    data_path = str(Path(a.data).resolve()) + '/'
    loaders, info = get_loader(data_path, a.batch_size)
    # The original loader creates one-hot widths independently for each split.
    if info['label_train'].shape[1] != info['label_test'].shape[1]:
        raise ValueError('Inconsistent label dimensions in upstream loader; validate source features')
    device = torch.device(a.device)
    model = IDCM_NN(img_input_dim=info['img_dim'], text_input_dim=info['text_dim'], output_dim=info['num_class']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=0)
    if a.selection == 'upstream-test':
        print('NOTICE: replicating upstream checkpoint selection on its test split; not an independent held-out test.', flush=True)
        kwargs = {'num_epochs': a.epochs}
        if 'device' in inspect.signature(train_model).parameters:
            kwargs['device'] = device
        model, ih, th, lh = train_model(model, loaders, optimizer, 1e-3, 1e-1, **kwargs)
        history = {'I2T_history': [float(v) for v in ih], 'T2I_history': [float(v) for v in th], 'loss': [float(v) for v in lh]}
    else:
        history = []
        for epoch in range(a.epochs):
            model.train()
            total, count = 0., 0
            for imgs, txts, labels in loaders['train']:
                imgs, txts, labels = imgs.float().to(device), txts.float().to(device), labels.to(device)
                optimizer.zero_grad()
                vi, vt, pi, pt = model(imgs, txts)
                loss = calc_loss(vi, vt, pi, pt, labels, labels, 1e-3, 1e-1)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(imgs)
                count += len(imgs)
            row = {'epoch': epoch+1, 'train_loss': total/count}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    image, text = [], []
    with torch.inference_mode():
        for imgs, txts, labels in loaders['test']:
            vi, vt, _, _ = model(imgs.float().to(device), txts.float().to(device))
            image.append(vi.cpu().numpy())
            text.append(vt.cpu().numpy())
    iv, tv = np.concatenate(image), np.concatenate(text)
    labels = info['label_test'].argmax(1)
    metrics = {'I2T_mAP': float(fx_calc_map_label(iv, tv, labels)), 'T2I_mAP': float(fx_calc_map_label(tv, iv, labels))}
    git = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    result = {'method': 'DSCMR', 'status': 'evaluated', 'dataset': 'Pascal Sentence', 'metrics': metrics,
              'metrics_unit': 'fraction_0_1', 'selection': a.selection, 'is_reproduction_verified': False,
              'protocol_note': 'fixed-last changes checkpoint selection; upstream-test uses test for model selection.',
              'source_commit': git.stdout.strip(), 'args': vars(a), 'torch': torch.__version__,
              'timestamp': datetime.datetime.now().astimezone().isoformat()}
    torch.save({'state_dict': model.state_dict(), 'args': vars(a)}, out / 'model.pt')
    np.savez_compressed(out / 'embeddings.npz', image=iv, text=tv, labels=labels)
    (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
