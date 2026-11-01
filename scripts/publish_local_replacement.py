"""Publish verified measured results; leave old run artifacts intact."""
import csv,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
runs={'DSCMR':'DSCMR/runs/pascal_validation_refit_seed42',
      'CLIP4CMR':'CLIP4CMR/runs/pascal_uiuc_seed42'}
rows=[];results={}
for name,run in runs.items():
    r=json.loads((ROOT/run/'result.json').read_text());results[name]=r
    assert json.loads((ROOT/run/'verification.json').read_text())['status']=='passed'
    for direction,m in r['metrics'].items():
        rows.append([name,direction,*[round(m[k]*100,4) for k in ['mAP@ALL','R@1','R@5','R@10']],r['train_pairs'],r['test_pairs'],r['final_training_epochs']])
header=['Model','Direction','mAP_percent','R1_percent','R5_percent','R10_percent','Train_pairs','Test_pairs','Final_epochs']
s=io.StringIO();w=csv.writer(s,lineterminator='\n');w.writerow(header);w.writerows(rows)
for path in ['results_summary.csv','reports/selected_results.csv']: (ROOT/path).write_text(s.getvalue(),encoding='utf-8')
lines=['# DSCMR và CLIP4CMR: kết quả local sau khi thay nguồn dữ liệu','',
 'Đã hoàn tất cả hai lượt chạy local và đo hai chiều. CLIP4CMR sử dụng dữ liệu tự trích từ UIUC thay cho link đặc trưng tác giả đã hết hạn. Hai mô hình hiện dùng tập test khác nhau; không dùng bảng này để xếp hạng trực tiếp hoặc khẳng định tái lập survey.','',
 '| Mô hình | Chiều | mAP (%) | R@1 (%) | R@5 (%) | R@10 (%) | Train | Test | Epoch mô hình cuối |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
lines+=['','CLIP4CMR: đủ 1.000 ảnh Pascal-Sentence, 20 lớp; mỗi lớp 40 ảnh train, 10 ảnh test, seed42. Ghép toàn bộ caption của mỗi ảnh theo thứ tự nguồn, tokenize với truncate=True, giới hạn77 token. CLIP RN50 đóng băng, đặc trưng1024 chiều; chỉ huấn luyện mạng chiếu truy hồi bằng PCL. Đây là quyết định tiền xử lý local, chưa xác minh trùng tác giả.','',
 'Mỗi mô hình tách 10% train để chọn epoch bằng validation, rồi khởi tạo lại và refit toàn bộ train. Test chỉ đánh giá sau refit; không chọn checkpoint bằng test.']
for name,r in results.items():
    lines+=['',f"{name}: chạy {r['validation_selection_epochs']} epoch giai đoạn validation; chọn {r['final_training_epochs']} epoch cho refit; tổng {r['total_optimizer_epochs']} epoch qua hai giai đoạn. Thời gian runner {r['elapsed_seconds']:.2f}s, chưa tính tải dữ liệu/trích đặc trưng."]
lines+=['','Mốc survey (chỉ tham khảo): DSCMR I2T67,4% / T2I68,2%; CLIP4CMR I2T69,8% / T2I69,2%. Chưa xác minh split và tiền xử lý trùng bài báo; không báo chênh lệch như kết quả tái lập.',
 '', 'DSCMR giữ kết quả từ đặc trưng tác giả: 800train/100test, ảnh4096 chiều và text300 chiều. CLIP4CMR mới dùng split800/200, vì vậy chưa là phép so sánh cùng dữ liệu.',
 '', 'Nguồn: https://vision.cs.uiuc.edu/pascal-sentences/ ; https://github.com/zhixiongz/CLIP4CMR ; https://arxiv.org/html/2201.02772v2',
 '', 'Dữ liệu, ID split, hash từng ảnh và checkpoint RN50, quy tắc caption: CLIP4CMR/data/pascal_uiuc_seed42/provenance.json.',
 '', 'Kết quả, lịch sử train, model.pt, protocol.json, test_features.npz và kiểm chứng evaluator gốc lưu tại các thư mục runs dưới đây. Chỉ một seed; chưa đo độ biến thiên nhiều seed.']
lines += ['',*[f'- {name}: {run}' for name,run in runs.items()]]
report='\n'.join(lines)+'\n'
for p in ['REPORT.md','reports/SELECTED_MODELS_REPORT.md','CURRENT_STATUS.md']: (ROOT/p).write_text(report,encoding='utf-8')
(ROOT/'README.md').write_text('# DSCMR + CLIP4CMR\n\nCả hai đã hoàn tất thí nghiệm local, đo I2T và T2I.\n\nĐọc [báo cáo](reports/SELECTED_MODELS_REPORT.md) và [CSV](reports/selected_results.csv). Hai mô hình dùng test khác nhau, chưa đối chiếu trực tiếp survey.\n\nCLIP4CMR dùng nguồn UIUC thay thế và đặc trưng RN50 tự tạo; dữ liệu tác giả vẫn chưa lấy được. Chạy show-status.cmd để xem trạng thái.\n',encoding='utf-8')
for p in ['reports/CLIP4CMR_INVESTIGATION.md','docs/PROJECT_LAYOUT.md']:
    path=ROOT/p
    original=path.read_text(encoding='utf-8')
    if not original.startswith('> Cập nhật: CLIP4CMR đã chạy xong'):
        path.write_text('> Cập nhật: CLIP4CMR đã chạy xong bằng nguồn UIUC thay thế. Các mô tả thiếu dữ liệu bên dưới là tình trạng trước phương án này. Xem báo cáo SELECTED_MODELS_REPORT.md.\n\n'+original,encoding='utf-8')
print('Published measured results: reports/SELECTED_MODELS_REPORT.md')
