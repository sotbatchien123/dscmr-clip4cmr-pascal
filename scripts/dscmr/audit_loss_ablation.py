"""Audit a completed Assignment 3 run and draw figures from measured logs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common.paths import from_repo


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    assert read(root/'status.json')['state'] == 'completed'
    p = read(root/'protocol.json')
    train, val = p['train_indices'], p['validation_indices']
    assert len(train) == 720 and len(val) == 80 and not set(train)&set(val)
    assert len(set(train)|set(val)) == 800
    source = from_repo(p['source_root'])
    for file, sha in p['source_sha256'].items():
        assert hashlib.sha256((source/file).read_bytes()).hexdigest() == sha
    for file, sha in p['train_data_sha256'].items():
        assert hashlib.sha256((source/'DSCMR/upstream/data/pascal'/file).read_bytes()).hexdigest() == sha
    assert read(root/'verification.json')['passed']
    variants = p['variants']
    records = []
    for phase, seeds, cap in [('smoke', [42], 10), ('pilot', [42], 200), ('main', [42, 43, 44], 500)]:
        for seed in seeds:
            for variant in variants:
                run = root/phase/f'{variant}_seed{seed}'
                r, history = read(run/'training_result.json'), read(run/'history.json')
                assert len(history) == r['epochs'] <= cap
                assert [h['epoch'] for h in history] == list(range(1, len(history)+1))
                scores = np.array([(h['validation']['I2T']['mAP@ALL']+h['validation']['T2I']['mAP@ALL'])/2 for h in history])
                assert abs(scores[r['best_epoch']-1]-r['best_validation_mean_map']) < 1e-12
                assert scores.max() <= r['best_validation_mean_map']+1e-8
                if phase == 'main' and r['epochs'] < cap:
                    assert r['epochs']-r['best_epoch'] == 50
                for h in history:
                    assert np.isfinite(h['raw_losses']).all()
                    multipliers = np.array(h['weights'])/np.array(p['base_weights'])
                    assert np.all(multipliers >= .5) and np.all(multipliers <= 2)
                if phase == 'main':
                    result = read(run/'result.json')
                    assert result['test_evaluated']
                    records.append(result)
                else:
                    assert not (run/'result.json').exists()
    # Adaptive is exactly inactive during warmup; this tests common seeds/batches.
    for phase in ('smoke', 'pilot', 'main'):
        for plain, adaptive in [(variants[0], variants[2]), (variants[1], variants[3])]:
            left = read(root/phase/f'{plain}_seed42'/'history.json')
            right = read(root/phase/f'{adaptive}_seed42'/'history.json')
            for a, b in zip(left[:10], right[:10]):
                np.testing.assert_array_equal(a['raw_losses'], b['raw_losses'])
                assert a['validation'] == b['validation']
    audit = {'passed': True, 'main_runs': len(records), 'main_epochs': sum(r['epochs'] for r in records),
             'smoke_epochs': 40, 'pilot_epochs': 800,
             'checks': ['source_and_training_data_unchanged', 'split_disjoint_complete',
                        'all_epoch_budgets', 'checkpoint_selection', 'early_stopping_patience',
                        'finite_losses_bounded_weights', 'same_initialization_batches_during_warmup',
                        'no_pilot_or_smoke_test_results'],
             'training_elapsed_seconds': sum(read(f)['elapsed_seconds'] for f in root.glob('*/*/training_result.json'))}
    (root/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, variant in zip(axes.flat, variants):
        for seed in (42, 43, 44):
            history = read(root/'main'/f'{variant}_seed{seed}'/'history.json')
            xs = [h['epoch'] for h in history]
            ys = [50*(h['validation']['I2T']['mAP@ALL']+h['validation']['T2I']['mAP@ALL']) for h in history]
            ax.plot(xs, ys, label=f'seed {seed}')
        ax.set(title=variant, xlabel='Epoch', ylabel='Mean validation mAP (%)')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('DSCMR: validation checkpoint selection (fixed 720/80 split)')
    fig.tight_layout(); fig.savefig(root/'validation_curves.png', dpi=160); plt.close(fig)
    aggregate = read(root/'aggregate.json')
    names = dict(zip(variants, ['DSCMR gốc', '+ Contrastive', '+ Trọng số thích nghi', '+ Cả hai']))
    vn = ['# Kết quả vòng 1 — DSCMR', '',
          'Gồm 4 lần kiểm tra × 10 epoch, 4 lần chạy thử × 200 epoch và 12 lần chạy chính (4 cấu hình × 3 seed).',
          'Mỗi lần chạy chính tối đa 500 epoch, early stopping 50 epoch; lấy checkpoint có mAP validation trung bình hai chiều cao nhất. Không refit.', '',
          '## Kết quả test', '',
          'Giá trị: phần trăm, trung bình ± độ lệch chuẩn mẫu qua 3 seed huấn luyện.', '',
          '| Cấu hình | I2T mAP@ALL | T2I mAP@ALL |', '|---|---:|---:|']
    for v in variants:
        cells = [f"{aggregate[v][d]['mAP@ALL']['mean']*100:.2f} ± {aggregate[v][d]['mAP@ALL']['std']*100:.2f}" for d in ('I2T', 'T2I')]
        vn.append('| '+names[v]+' | '+' | '.join(cells)+' |')
    vn += ['', 'Các chỉ số R@1/R@5/R@10 hai chiều nằm trong results.csv và REPORT.md.', '',
           '## Epoch thực tế', '', '| Cấu hình | Seed | Epoch đã chạy | Epoch checkpoint tốt nhất |', '|---|---:|---:|---:|']
    for r in records:
        vn.append(f"| {names[r['variant']]} | {r['seed']} | {r['epochs']} | {r['best_epoch']} |")
    vn += ['', f"Tổng epoch chạy chính: {audit['main_epochs']}. Tổng thời gian trong các vòng huấn luyện: {audit['training_elapsed_seconds']/60:.2f} phút (không gồm thời gian triển khai, khởi tạo môi trường và kiểm tra).", '',
           '## Phạm vi kết luận', '',
           '- Dữ liệu: đặc trưng tác giả, 720 train / 80 validation / 100 test; cùng split, kiến trúc và batch giữa bốn cấu hình.',
           '- R@K là tỷ lệ truy vấn tìm được ít nhất một kết quả cùng lớp trong top K, không phải tỷ lệ tất cả tài liệu liên quan được thu hồi.',
           '- Đây là thí nghiệm đối chứng với siêu tham số cố định, chưa tuning trọng số baseline hoặc temperature. Không tuyên bố tốt hơn một baseline đã tuning.',
           '- Trọng số thích nghi là quy tắc dựa trên tỷ lệ loss EMA có giới hạn; không phải GradNorm. Tính mới chưa được chứng minh.',
           '- Ba seed không phải ba fold. Chênh lệch nhỏ cần thêm kiểm chứng; không suy ra ý nghĩa thống kê chỉ từ bảng trung bình.',
           '- Dùng checkpoint validation, khác các báo cáo refit cũ; không đối chiếu trực tiếp như cùng protocol. Không chạy lại CLIP4CMR.',
           '- Tập test đã được dùng trong các lần chạy trước; lần chạy này không dùng test để chọn checkpoint hay điều chỉnh cấu hình.', '',
           '## Kiểm tra', '',
           'audit.json xác nhận ngân sách, early stopping, checkpoint, split, dữ liệu/mã nguồn gốc không đổi và tính nhất quán giai đoạn warmup.', '',
           '![Đường mAP validation](validation_curves.png)', '', '![mAP test](test_map.png)', '']
    # BAO_CAO.md is hand-extended after the first audit; never overwrite it on re-runs.
    report = root/'BAO_CAO.md'
    (report if not report.exists() else root/'BAO_CAO_audit_auto.md').write_text('\n'.join(vn), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(4)
    for i, direction in enumerate(('I2T', 'T2I')):
        mean = [aggregate[v][direction]['mAP@ALL']['mean']*100 for v in variants]
        sd = [aggregate[v][direction]['mAP@ALL']['std']*100 for v in variants]
        ax.bar(xs+(i-.5)*.36, mean, .36, yerr=sd, capsize=4, label=direction)
    ax.set(xticks=xs, xticklabels=variants, ylabel='Test mAP@ALL (%)', ylim=(0, 100),
           title='DSCMR loss ablations: 3 seeds, mean and sample SD')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(root/'test_map.png', dpi=160); plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
