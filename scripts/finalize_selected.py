"""Generate the active project index and report from measured JSON results."""
from pathlib import Path
import csv,io,json
ROOT=Path(__file__).resolve().parents[1]
def put(name,text):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
r=read('DSCMR/runs/pascal_validation_refit_seed42/result.json')
cleanup=read('reports/transition_20261014/cleanup_result.json')
availability={'CLIP4CMR':{'state':'blocked_missing_data','url':'https://pan.cstcloud.cn/s/JqKbqGfTRs','api_result':read('tmp/share_info.json'),'required_files':['clip_train.pkl','clip_test.pkl']},'DSCMR':{'available_train':800,'available_test':100,'reference_test':200}}
put('reports/transition_20261014/data_availability.json',json.dumps(availability,ensure_ascii=False,indent=2))
methods=[]
for name,repo in [('DSCMR','penghu-cs/DSCMR'),('CLIP4CMR','zhixiongz/CLIP4CMR')]:
    methods.append({'id':name,'folder':name,'official_repo':'https://github.com/'+repo,'dataset':'Pascal-Sentence','directions':['I2T','T2I'],'primary_metric':'mAP@ALL','additional_metrics':['R@1','R@5','R@10'],'status_file':name+'/status.json'})
put('methods.json',json.dumps(methods,indent=2))
output=io.StringIO();w=csv.writer(output)
w.writerow(['model','direction','local_mAP_percent','R1_percent','R5_percent','R10_percent','reference_mAP_percent','local_test_pairs','reference_test_pairs','direct_comparison_valid','state'])
for name,refs in [('DSCMR',[67.4,68.2]),('CLIP4CMR',[69.8,69.2])]:
    for direction,ref in zip(['I2T','T2I'],refs):
        m=r['metrics'][direction] if name=='DSCMR' else {}
        vals=[round(m[k]*100,6) if k in m else '' for k in ['mAP@ALL','R@1','R@5','R@10']]
        w.writerow([name,direction,*vals,ref,100 if m else '',200,False,'completed_local_protocol' if m else 'blocked_missing_data'])
put('reports/selected_results.csv',output.getvalue());put('results_summary.csv',output.getvalue())
report='''# DSCMR và CLIP4CMR — kết quả ngày 14/10/2026

Đã chốt hai mô hình trên Pascal-Sentence, đo cả ảnh → văn bản (I2T) và văn bản → ảnh (T2I). **Mới hoàn tất DSCMR theo protocol local. CLIP4CMR chưa thể huấn luyện vì thiếu dữ liệu; chưa hoàn thành thí nghiệm hai mô hình để đối chiếu survey.**

## Kết quả đo thật

Đơn vị %. mAP@ALL là chỉ số chính. Các giá trị R@K là tỷ lệ truy vấn có ít nhất một kết quả cùng lớp trong top K.

| Mô hình | Chiều | mAP@ALL | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| DSCMR | I2T | 64,6654 | 65 | 83 | 92 |
| DSCMR | T2I | 69,7477 | 72 | 96 | 99 |
| CLIP4CMR | I2T | Chưa có | — | — | — |
| CLIP4CMR | T2I | Chưa có | — | — | — |

DSCMR: mAP trung bình hai chiều 67,2065%. Thời gian huấn luyện và đánh giá trong runner khoảng 71,63 giây, không bao gồm chuẩn bị môi trường/tải dữ liệu. Không dùng thời gian này để so tốc độ với bài báo.

## Dữ liệu và epoch của DSCMR

- Sử dụng đặc trưng có sẵn trong repo tác giả: ảnh 4096 chiều, văn bản 300 chiều; 20 lớp. Huấn luyện mạng truy hồi trên đặc trưng cố định, không huấn luyện lại bộ mã hoá ảnh/văn bản.
- Repo cung cấp 800 cặp train và 100 cặp test. Nhãn cùng lớp xác định một kết quả phù hợp; không dùng tiêu chí caption ghép cặp của Flickr30K.
- Tách phân tầng từ 800 mẫu train: 720 train nội bộ, 80 validation. Chạy 500 epoch để chọn số epoch theo mAP trung bình hai chiều trên validation. Epoch tốt nhất: 61.
- Khởi tạo lại mạng cùng seed 42, huấn luyện trên toàn bộ 800 mẫu train trong 61 epoch. Tổng số epoch tối ưu qua hai giai đoạn là 561, không phải 561 epoch trên toàn bộ dữ liệu.
- Đánh giá một lần trên 100 mẫu test: 100 truy vấn và 100 ứng viên ở mỗi chiều. Không dùng test để chọn checkpoint. Toàn bộ test hiện có đã được dùng.
- Adam, learning rate 0,0001, betas (0,5; 0,999), batch 100. GPU RTX 5070 Laptop. Chỉ một seed, chưa có độ lệch chuẩn nhiều lần chạy.
- Lưu nguồn gốc/checksum dữ liệu, chỉ số từng epoch, checkpoint và đặc trưng test. mAP hai chiều khớp evaluator gốc; đã tính lại toàn bộ metric từ đặc trưng xuất ra.
- Nguồn gốc quá trình tạo đặc trưng của tác giả chưa được kiểm toán đầy đủ; việc tách validation áp dụng ở tầng mạng truy hồi.

## Mốc đối chiếu từ bài báo

| Mô hình | I2T mAP (%) | T2I mAP (%) | Test tham chiếu |
|---|---:|---:|---|
| DSCMR | 67,4 | 68,2 | 200 mẫu |
| CLIP4CMR | 69,8 | 69,2 | 200 mẫu |

Các số này là mốc đã công bố, **không phải kết quả local**. Nguồn: bảng 2 của survey `sources/survey.pdf`; bảng 1 và phần Pascal-Sentence trong [bài CLIP4CMR](https://arxiv.org/html/2201.02772v2). Chú thích DSCMR trong bài CLIP4CMR nêu tập test 200 mẫu. Bảng survey có nhãn chiều bị lặp; chiều T2I được kiểm tra với bảng bài gốc.

**Không kết luận DSCMR local tốt hơn/kém hơn mốc công bố:** tập test local 100 mẫu khác 200 mẫu tham chiếu; cách chọn epoch local cũng khác. Không tính chênh lệch để tuyên bố tái lập. Cần đúng split, đặc trưng và protocol tham chiếu trước khi đối chiếu trực tiếp.

## CLIP4CMR đang bị chặn ở đâu

[Repo tác giả](https://github.com/zhixiongz/CLIP4CMR) chỉ cung cấp hướng dẫn tải đặc trưng. [Link dữ liệu](https://pan.cstcloud.cn/s/JqKbqGfTRs) trả về `ERR_SHARE_EXPIRED` khi kiểm tra. Chưa tìm được bản thay thế có thể xác minh đúng bộ dữ liệu/split. Bằng chứng ở `reports/transition_20261014/data_availability.json`.

Cần `clip_train.pkl` và `clip_test.pkl` cho Pascal-Sentence (các trường image, text, label), kèm nguồn/split để xác minh. Đặt tại `CLIP4CMR/upstream/data/pascal/` rồi chạy:

```powershell
.\\.venv310\\Scripts\\python.exe scripts\\train_selected.py --method CLIP4CMR
```

Runner dự kiến tối đa 200 epoch chọn bằng validation, sau đó refit bằng số epoch được chọn. **Số epoch thực tế hiện tại: 0.** Đây là protocol local; chưa phải tái lập đúng early stopping của bài báo.

Đã kiểm tra forward/backward và evaluator bằng tensor giả, chỉ để kiểm tra mã. Không ghi các kiểm tra đó thành kết quả thực nghiệm. Giữ nguyên mã gốc; adapter nạp riêng hàm PAN cho nhánh PCL nhằm tránh docstring chưa đóng ở hàm Proxy_NCA không sử dụng trong repo. CLIP là thư viện phụ thuộc bên trong CLIP4CMR, không phải mô hình thứ ba được chọn.

## Dọn dự án và các tệp cần dùng

Đã xoá thư mục thực nghiệm CLIP/ViLT cũ, Flickr30K và checkpoint cũ; giữ môi trường Python, thư viện dùng chung, nguồn bài báo, hai repo mới và kết quả DSCMR. Mã/số liệu cũ cần truy nguyên được nén trong `reports/transition_20261014/retired_code_and_records.zip`; báo cáo cũ chuyển sang `reports/history_before_20261014/`. Không dùng số liệu cũ cho hai mô hình hiện tại. Không sửa lịch sử Git.

- `reports/selected_results.csv`: kết quả local và số tham chiếu nằm ở cột riêng.
- `DSCMR/runs/pascal_validation_refit_seed42/result.json`: kết quả thật.
- `DSCMR/runs/pascal_validation_refit_seed42/protocol.json`: cấu hình, split nội bộ, checksum.
- `DSCMR/runs/pascal_validation_refit_seed42/model.pt`: mô hình refit.
- `reports/verification_selected.json`: kiểm tra độc lập.
- `DSCMR/status.json`, `CLIP4CMR/status.json`: trạng thái mỗi mô hình.
- `show-status.cmd`, `watch-status.cmd`: xem trạng thái; `run-selected.cmd`: chạy phần còn thiếu.

Muốn hoàn thành đối chiếu đúng survey cần dữ liệu CLIP4CMR nêu trên và đặc trưng DSCMR cho đúng split test 200 mẫu. Không có tác vụ huấn luyện nền đang chờ tự chạy.
'''
report+=f"\nDung lượng tệp đã dọn theo nhật ký: {cleanup['removed_file_bytes']/1e9:.3f} GB (không phải dung lượng đĩa trống đo trực tiếp).\n"
put('reports/SELECTED_MODELS_REPORT.md',report)
put('REPORT.md',report)
summary='''# DSCMR + CLIP4CMR trên Pascal-Sentence

Đo cả ảnh → văn bản và văn bản → ảnh; chỉ số chính mAP@ALL.

| Mô hình | Trạng thái | I2T mAP | T2I mAP |
|---|---|---:|---:|
| DSCMR | Hoàn tất local, test 100 mẫu | 64,6654% | 69,7477% |
| CLIP4CMR | Thiếu dữ liệu từ link tác giả hết hạn | Chưa có | Chưa có |

Chưa đối chiếu trực tiếp được với survey dùng test 200 mẫu. DSCMR chọn epoch bằng 720 train/80 validation trong 500 epoch, rồi huấn luyện lại toàn bộ 800 train trong 61 epoch. Test không dùng chọn checkpoint.

Đọc [báo cáo](reports/SELECTED_MODELS_REPORT.md), [bảng kết quả](reports/selected_results.csv). Xem trạng thái bằng `show-status.cmd`. Chạy phần còn thiếu bằng `run-selected.cmd`; thiếu dữ liệu sẽ dừng và báo rõ.

Mã cũ lưu trong ZIP lịch sử ở `reports/transition_20261014/`, không thuộc thực nghiệm hiện tại. CLIP trong `CLIP4CMR/dependencies/` là thư viện phụ thuộc.
'''
put('README.md',summary);put('CURRENT_STATUS.md',summary)
put('docs/PROJECT_LAYOUT.md','''# Bố cục đang dùng

- DSCMR/: mã gốc, dữ liệu đặc trưng tác giả, checkpoint và kết quả local.
- CLIP4CMR/: mã gốc, thư viện phụ thuộc; chưa có dữ liệu thật.
- common/, tests/: tính và kiểm tra metric.
- scripts/train_selected.py: train/validation/refit/test, cả hai chiều.
- scripts/verify_selected.py: đối chiếu metric đã lưu, kiểm tra mã CLIP4CMR.
- reports/: báo cáo hiện tại; history_before_20261014 và transition_20261014 là lịch sử.
- sources/: bài báo nguồn.
- .venv310/: môi trường Python được giữ lại.
''')
put('STATUS_FILES.md','Trạng thái nguồn: DSCMR/status.json và CLIP4CMR/status.json.\nBáo cáo: reports/SELECTED_MODELS_REPORT.md.\nDùng show-status.cmd hoặc watch-status.cmd. Không diễn giải blocked_missing_data thành hoàn tất.\n')
put('CONTRIBUTIONS.md','Phạm vi đã chốt: DSCMR và CLIP4CMR trên Pascal-Sentence, cả I2T và T2I.\nChưa chốt lại phân công thành viên sau thay đổi mô hình. Phân công CLIP/ViLT cũ chỉ còn trong ZIP lịch sử.\n')
put('scripts/update_status.py','''import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def show():
    for name in ('DSCMR','CLIP4CMR'):
        p=ROOT/name/'status.json'
        d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'state':'not_started'}
        print(name, ':', d.get('state',d.get('status')),flush=True)
        for direction,m in d.get('metrics',{}).items(): print(' ',direction,'mAP@ALL:',round(m['mAP@ALL']*100,4),'%',flush=True)
        if 'error' in d: print(' ',d['error'],flush=True)
        if 'epoch' in d: print(' ',d['epoch'],'/',d.get('total_epochs','?'),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=float,default=0);a=parser.parse_args()
    while True:
        show()
        if not a.watch: break
        time.sleep(a.watch)
''')
put('run-selected.cmd','''@echo off
cd /d "%~dp0"
if exist "DSCMR\\runs\\pascal_validation_refit_seed42\\result.json" goto clip4cmr
".venv310\\Scripts\\python.exe" scripts\\train_selected.py --method DSCMR
if errorlevel 1 exit /b 1
:clip4cmr
if exist "CLIP4CMR\\runs\\pascal_validation_refit_seed42\\result.json" goto done
".venv310\\Scripts\\python.exe" scripts\\train_selected.py --method CLIP4CMR
if errorlevel 1 exit /b 1
:done
".venv310\\Scripts\\python.exe" scripts\\update_status.py
''')
print('Updated active report, methods, results and entry points.')
