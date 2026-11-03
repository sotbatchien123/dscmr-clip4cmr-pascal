"""Verify and summarise the long baseline runs (5,000-epoch validation tracking + 1,000-epoch refit).

Reads DSCMR/runs/pascal_selection5000_refit1000_seed42 and the CLIP4CMR equivalent (produced by
scripts/baseline/train_baseline.py), recomputes every saved metric from test_features.npz and writes
reports/RESULTS_5000_1000.md, reports/results_5000_1000.csv and reports/verification_5000_1000.json.
Usage: python -m scripts.baseline.report_5000_1000
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from common.metrics import label_relevance, map_all, recall_at_k

ROOT = Path(__file__).resolve().parents[2]
RUN = 'runs/pascal_selection5000_refit1000_seed42'
PREVIOUS = {'DSCMR': 'pascal_validation_refit_seed42', 'CLIP4CMR': 'pascal_uiuc_seed42'}
METRICS = ('mAP@ALL', 'R@1', 'R@5', 'R@10')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(method):
    run = ROOT/method/RUN
    result = read_json(run/'result.json')
    history, refit = read_json(run/'history.json'), read_json(run/'refit_history.json')
    assert [x['epoch'] for x in history] == list(range(1, 5001))
    assert [x['epoch'] for x in refit] == list(range(1, 1001))
    assert (result['validation_selection_epochs'], result['final_training_epochs'],
            result['total_optimizer_epochs']) == (5000, 1000, 6000)
    assert read_json(run/'verification.json')['status'] == 'passed'
    z = np.load(run/'test_features.npz')
    rel = label_relevance(z['label'], z['label'])
    scores = -cdist(z['image'], z['text'], 'cosine' if method == 'DSCMR' else 'euclidean')
    for direction, s in (('I2T', scores), ('T2I', scores.T)):
        for k, v in {**map_all(s, rel), **recall_at_k(s, rel)}.items():
            assert abs(v - result['metrics'][direction][k]) < 1e-10, (method, direction, k)
    check = {'validation_history_count': len(history), 'refit_history_count': len(refit),
             'saved_metrics_recomputed': True, 'upstream_mAP_matches': True}
    return result, check


def main():
    rows, notes, checks = [], [], {}
    for method, previous_run in PREVIOUS.items():
        result, checks[method] = verify(method)
        previous = read_json(ROOT/method/'runs'/previous_run/'result.json')
        for direction in ('I2T', 'T2I'):
            m = result['metrics'][direction]
            rows.append([method, direction, *[round(m[k]*100, 4) for k in METRICS],
                         round(previous['metrics'][direction]['mAP@ALL']*100, 4), result['test_pairs']])
        notes.append(f"- {method}: 5.000 epoch theo dõi validation + 1.000 epoch refit = 6.000 epoch trên hai mô hình "
                     f"khởi tạo riêng. Epoch tốt nhất trên validation: {result['best_validation_epoch']}. "
                     f"Thời gian chạy: {result['elapsed_seconds']:.1f} giây.")

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(['model', 'direction', 'mAP_percent', 'R1_percent', 'R5_percent', 'R10_percent',
                     'previous_mAP_percent', 'test_pairs'])
    writer.writerows(rows)
    (ROOT/'reports/results_5000_1000.csv').write_text(buf.getvalue(), encoding='utf-8')

    lines = ['# Baseline DSCMR và CLIP4CMR: 5.000 + 1.000 epoch', '',
             'Giai đoạn 1 huấn luyện 5.000 epoch trên 720 mẫu và ghi nhận checkpoint tốt nhất trên 80 mẫu validation. '
             'Giai đoạn 2 khởi tạo lại mô hình và huấn luyện đúng 1.000 epoch trên toàn bộ 800 mẫu train; '
             'checkpoint epoch 1.000 được đánh giá trên test.', '',
             '| Mô hình | Chiều | mAP (%) | R@1 (%) | R@5 (%) | R@10 (%) | mAP cấu hình ngắn (%) | Test |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    lines += ['', *notes, '',
              '- DSCMR: đặc trưng của tác giả, test 100 cặp. CLIP4CMR: đặc trưng CLIP RN50 đóng băng trích từ ảnh UIUC, '
              'split 800/200 (seed 42), 5 câu mô tả nối lại rồi cắt ở 77 token.',
              '- "mAP cấu hình ngắn" là kết quả của cùng mô hình với số epoch refit chọn theo validation; '
              'dữ liệu, seed, loss, learning rate và batch giữ nguyên.',
              '- Không so trực tiếp DSCMR với CLIP4CMR vì hai tập test khác nhau. Thêm epoch không bảo đảm tăng mAP.',
              '- Kiểm tra: đủ số epoch trong history.json và refit_history.json; mọi chỉ số được tính lại từ '
              'test_features.npz và mAP khớp evaluator của tác giả.',
              '- mAP và R@K tính theo %. R@K là tỷ lệ truy vấn có ít nhất một kết quả cùng lớp trong top K.']
    (ROOT/'reports/RESULTS_5000_1000.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT/'reports/verification_5000_1000.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(buf.getvalue())


if __name__ == '__main__':
    main()
