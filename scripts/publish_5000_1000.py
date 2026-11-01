"""Validate and report the requested 5000-selection/1000-refit experiments."""
import csv,io,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common.metrics import map_all,recall_at_k,label_relevance
old={'DSCMR':'pascal_validation_refit_seed42','CLIP4CMR':'pascal_uiuc_seed42'}
rows=[];details=[];checks={}
for method in old:
    run=ROOT/method/'runs/pascal_selection5000_refit1000_seed42'
    r=json.loads((run/'result.json').read_text())
    history=json.loads((run/'history.json').read_text());refit=json.loads((run/'refit_history.json').read_text())
    assert [x['epoch'] for x in history]==list(range(1,5001))
    assert [x['epoch'] for x in refit]==list(range(1,1001))
    assert r['validation_selection_epochs']==5000 and r['final_training_epochs']==1000 and r['total_optimizer_epochs']==6000
    assert json.loads((run/'verification.json').read_text())['status']=='passed'
    z=np.load(run/'test_features.npz');rel=label_relevance(z['label'],z['label'])
    scores=-cdist(z['image'],z['text'],'cosine' if method=='DSCMR' else 'euclidean')
    previous=json.loads((ROOT/method/'runs'/old[method]/'result.json').read_text())
    for direction,s in [('I2T',scores),('T2I',scores.T)]:
        for k,v in {**map_all(s,rel),**recall_at_k(s,rel)}.items(): assert abs(v-r['metrics'][direction][k])<1e-10
        m=r['metrics'][direction]
        rows.append([method,direction,*[round(m[k]*100,4) for k in ['mAP@ALL','R@1','R@5','R@10']],round(previous['metrics'][direction]['mAP@ALL']*100,4),r['test_pairs']])
    details.append(f"- {method}: đủ 5.000 epoch validation + 1.000 epoch refit = 6.000 epoch qua hai mô hình khởi tạo riêng. Best validation epoch: {r['best_validation_epoch']}. Thời gian runner: {r['elapsed_seconds']:.1f} giây.")
    checks[method]={'validation_history_count':len(history),'refit_history_count':len(refit),'saved_metrics_recomputed':True,'upstream_mAP_matches':True}
out=io.StringIO();w=csv.writer(out,lineterminator='\n');w.writerow(['model','direction','mAP_percent','R1_percent','R5_percent','R10_percent','previous_mAP_percent','test_pairs']);w.writerows(rows)
for p in ['reports/selected_results.csv','results_summary.csv','reports/results_5000_1000.csv']: (ROOT/p).write_text(out.getvalue(),encoding='utf-8')
lines=['# Kết quả DSCMR và CLIP4CMR: 5.000 + 1.000 epoch','',
 'Đã hoàn tất đúng cấu hình yêu cầu cho cả hai mô hình. Giai đoạn đầu ghi nhận checkpoint tốt nhất trên validation trong 5.000 epoch. Sau đó khởi tạo lại và huấn luyện đúng 1.000 epoch trên toàn bộ 800 mẫu train; đánh giá checkpoint epoch 1.000 trên test. Best validation epoch không quyết định thời lượng refit trong cấu hình này.','',
 '| Mô hình | Chiều | mAP (%) | R@1 (%) | R@5 (%) | R@10 (%) | mAP lượt trước (%) | Test |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
lines+=['| '+' | '.join(map(str,row))+' |' for row in rows]
lines+=['',*details,'',
 'Mỗi giai đoạn validation dùng 720 train/80 validation tách từ 800 mẫu train. Không dùng test để chọn checkpoint trong runner. Test của hai lượt cũ đã từng được báo cáo; đây là thí nghiệm tiếp theo trên cùng test, không phải một tập kiểm định mới chưa từng xem.',
 '', 'DSCMR: đặc trưng tác giả, test100. CLIP4CMR: đặc trưng CLIP RN50 đóng băng tự tạo từ UIUC, split800/200 seed42; ghép caption theo thứ tự nguồn và cắt77token. Không thay đổi dữ liệu, seed, loss, learning rate hoặc batch so với lượt trước.',
 '', 'Có thể đối chiếu lượt mới/lượt trước trong từng mô hình. Không xếp hạng trực tiếp DSCMR với CLIP4CMR vì test khác nhau; chưa xác minh tái lập đúng split/tiền xử lý của survey. Thêm epoch không bảo đảm tăng mAP.',
 '', 'Đã kiểm tra đủ số epoch trong history.json và refit_history.json, tính lại metrics từ test_features.npz và đối chiếu mAP hai chiều với evaluator tác giả.',
 '', 'Kết quả và checkpoint cũ vẫn được giữ. Kết quả mới:',
 '- DSCMR/runs/pascal_selection5000_refit1000_seed42/',
 '- CLIP4CMR/runs/pascal_selection5000_refit1000_seed42/',
 '', 'Trong mỗi thư mục: result.json, protocol.json, history.json, refit_history.json, validation_best.pt, model.pt, test_features.npz, verification.json. model.pt là checkpoint refit epoch1.000; validation_best.pt thuộc mạng giai đoạn720train.',
 '', 'Chỉ số mAP và R@K trong bảng là %. R@K là tỷ lệ truy vấn có ít nhất một kết quả cùng lớp ở top K.']
report='\n'.join(lines)+'\n'
for p in ['REPORT.md','CURRENT_STATUS.md','reports/SELECTED_MODELS_REPORT.md','reports/RESULTS_5000_1000.md']: (ROOT/p).write_text(report,encoding='utf-8')
(ROOT/'reports/verification_5000_1000.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(ROOT/'README.md').write_text('# DSCMR + CLIP4CMR\n\nĐã hoàn tất mỗi mô hình: 5.000 epoch validation và 1.000 epoch refit trên toàn bộ train.\n\n[Báo cáo mới](reports/RESULTS_5000_1000.md) · [CSV](reports/results_5000_1000.csv).\n\nDSCMR test100; CLIP4CMR test200 với dữ liệu UIUC tự tạo. Không so trực tiếp hai mô hình như cùng protocol. Kết quả cũ được giữ tại các thư mục runs.\n',encoding='utf-8')
print(out.getvalue())
