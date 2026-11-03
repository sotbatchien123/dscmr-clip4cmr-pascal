# So sánh với bài báo gốc: CLIP4CMR và DSCMR (30/10/2026)

Mọi số liệu là mAP@ALL (%), trung bình ± độ lệch chuẩn qua 5 seed (42–46).

## 1. Dữ liệu gốc của CLIP4CMR: đã lấy lại từ bản sao

- Link dữ liệu của tác giả đã hết hạn. Bản sao `clip_train.pkl` / `clip_test.pkl` được lấy từ HF Space [shivangibithel/LCMI_T2I](https://huggingface.co/spaces/shivangibithel/LCMI_T2I) (nhánh main). Đây là bản của **bên thứ ba, chưa được tác giả xác nhận**.
- File lưu tại `CLIP4CMR/data/hf_mirror_lcmi/`. SHA-256 khớp mã băm LFS. Pickle chỉ chứa mảng numpy và được đọc bằng unpickler giới hạn.
- Các kiểm tra (`verification.json`, `text_rule_probe.json`):
  - Đúng 800 mẫu train và 200 mẫu test, gồm 1.000 ID khác nhau của Pascal UIUC.
  - Nhãn khớp lớp UIUC 100% (tác giả đánh số lớp từ 1).
  - Đặc trưng ảnh trùng với đặc trưng CLIP RN50 nhóm tự trích ở 922/1000 mẫu (cos > 0.999).
  - Đặc trưng văn bản trùng ở 938/1000 mẫu khi dùng đúng quy tắc trong `load_original_data` của tác giả: nối 5 caption rồi bỏ dấu câu và chữ số. Quy tắc nối caption thông thường không khớp mẫu nào.
  - → Gần như chắc chắn là dữ liệu do pipeline của tác giả sinh ra.
- Tập test này chỉ trùng 49/200 mẫu với tập test split local cũ.
- Đặc trưng dùng cho thí nghiệm: `CLIP4CMR/data/pascal_author_split_mirror/`.
  - RN50 dùng nguyên đặc trưng của bản sao.
  - ViT-L/14 được trích lại theo đúng quy tắc văn bản của tác giả.

## 2. CLIP4CMR so với bài báo (RN50, Pascal, test 200)

| Cấu hình | Chọn checkpoint | I2T | T2I |
|---|---|---:|---:|
| Bài báo (PCL, arXiv 2201.02772 mục 4.1) | theo test (code gốc) | 69.8 | 69.2 |
| Base, chạy lại theo cách của bài báo | theo test, tối đa 200 epoch, dừng sớm 20 | 69.23 ± 0.50 | 68.95 ± 0.34 |
| **A_sem + B**, cùng cách chọn | theo test | **71.14 ± 0.09** | **71.00 ± 0.09** |
| Base, cách của nhóm (kết quả chính) | validation 720/80, test mở 1 lần | 68.48 ± 0.70 | 67.87 ± 0.45 |
| **A_sem + B**, cách của nhóm | validation | **70.90 ± 0.29** | **70.61 ± 0.26** |

- **Tái lập:** bản gốc chạy lại theo cách của bài báo chỉ thấp hơn số công bố 0.6 / 0.3 điểm.
- **Cải tiến, theo đúng cách chọn checkpoint của bài báo:** cao hơn số công bố +1.3 (I2T) / +1.8 (T2I).
- **Cải tiến, theo cách chặt hơn của nhóm** (không dùng test để chọn):
  - Vẫn cao hơn số công bố +1.1 / +1.4.
  - Cao hơn base chạy lại +2.42 / +2.74, tăng ở 5/5 seed cho cả hai chiều.
- Nguồn số liệu:
  - `reports/clip4cmr_paper_protocol_20261030/result.json` (chẩn đoán: chọn theo test nên lạc quan, chỉ để so ngang với bài báo).
  - `reports/clip4cmr_author_split_20261030/REPORT.md` (đầy đủ: tuning, 5 cấu hình × 2 backbone × 5 seed). `recheck` khớp 2.2e-16 và best.pt đúng epoch ở 50/50 lượt.

### Khác biệt so với split local cũ (cần nói rõ trong paper)

| Backbone | Split | Base (TB 2 chiều) | A_sem+B | Chênh lệch (seed tăng) |
|---|---|---:|---:|---|
| RN50 | local cũ | 71.39 | 73.35 | +1.97 (5/5, 5/5) |
| RN50 | **tác giả** | 68.17 | 70.76 | **+2.58 (5/5, 5/5)** |
| ViT-L/14 | local cũ | 72.96 | 74.09 | +1.13 (5/5, 4/5) |
| ViT-L/14 | **tác giả** | 71.91 | 72.21 | **+0.30 (3/5, 2/5): không đáng kể** |

- Trên RN50 (backbone của bài báo), cải tiến **ổn định trên cả hai split**.
- Trên ViT-L/14 với split của tác giả, A_sem gần như không còn tác dụng. Riêng B vẫn có ích ở cả 5/5 seed (+0.39 / +1.16).
- A_res (prototype ngẫu nhiên) làm giảm mạnh trên ViT-L/14 (−3.2 / −2.4). Điều này vẫn ủng hộ kết luận: nếu đã dùng adapter phần dư thì prototype phải là prototype ngữ nghĩa.
- Có thể viết trong paper: "Phương pháp cải thiện rõ khi backbone yếu (RN50, đúng như bài báo). Với backbone mạnh (ViT-L/14), lợi ích thu hẹp và không còn ổn định."

## 3. DSCMR so với bài báo (Pascal, test 100, đặc trưng của tác giả trong repo)

| Cấu hình | Chọn checkpoint | I2T | T2I |
|---|---|---:|---:|
| Bài báo (CVPR 2019, `DSCMR/upstream/Result.jpg`) | theo test (code gốc) | 71.0 | 72.2 |
| Chạy đúng code gốc (800 train, batch 100, 500 epoch) | theo test | 68.84 ± 0.50 | 71.05 ± 0.45 |
| Cùng lần chạy đó, epoch 500 | không chọn | 64.22 ± 0.84 | 67.26 ± 0.80 |
| B0, cách của nhóm (Assignment 3 vòng 1, 3 seed) | validation 720/80 | 66.77 | 69.32 |

- DSCMR **không thiếu dữ liệu**: repo của tác giả có sẵn đặc trưng VGG19 4096 chiều và văn bản 300 chiều.
- Chọn checkpoint theo test làm kết quả cao hơn khoảng 2 điểm so với chọn theo validation. Đây là nguyên nhân chính khiến số của nhóm thấp hơn bài báo.
- Phần chênh còn lại so với bài báo (khoảng 2.2 / 1.2 điểm) chưa giải thích được (có thể do seed hoặc phần cứng); chưa kiểm chứng riêng.
- Số liệu: `reports/dscmr_author_protocol_20261030/result.json` và `history_seed*.json`.

## Cách trình bày trong paper

1. Báo cáo kết quả chính bằng cách chọn checkpoint theo validation (không dùng test).
2. Đặt số công bố và bảng "chạy lại theo cách của bài báo" cạnh nhau, để người đọc thấy (a) phần tái lập khớp và (b) khoảng chênh do cách chọn checkpoint.
3. Ghi rõ dữ liệu CLIP4CMR lấy từ bản sao bên thứ ba và cách đã xác minh. Nên email tác giả để xác nhận nếu kịp.

## Tái lập

```
.venv310\Scripts\python.exe -I scripts\scripts/data/verify_hf_mirror.py
.venv310\Scripts\python.exe -I scripts\scripts/data/probe_mirror_text_rule.py
.venv310\Scripts\python.exe -I scripts\scripts/data/build_author_split.py
.venv310\Scripts\python.exe scripts\scripts/clip4cmr/improve.py --stage all --feats CLIP4CMR/data/pascal_author_split_mirror --out reports/clip4cmr_author_split_20261030
.venv310\Scripts\python.exe scripts\scripts/clip4cmr/recheck_author_eval.py reports/clip4cmr_author_split_20261030 CLIP4CMR/data/pascal_author_split_mirror
.venv310\Scripts\python.exe scripts\scripts/baseline/clip4cmr_paper_protocol.py
.venv310\Scripts\python.exe scripts\scripts/baseline/dscmr_author_protocol.py
```
