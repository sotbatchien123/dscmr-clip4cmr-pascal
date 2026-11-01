# Cải tiến DSCMR và CLIP4CMR cho truy hồi ảnh – văn bản (Pascal Sentence)

> Ghi chú làm việc nội bộ: đọc và cập nhật `GHI_NHO.md` mỗi phiên. File này chỉ có trên máy, không đưa lên GitHub.

Dự án Assignment 3 có ba phần chính:
- Tái lập hai phương pháp truy hồi đa phương thức có giám sát: **DSCMR** (CVPR 2019) và **CLIP4CMR** (2022).
- Đề xuất và kiểm chứng các cải tiến cho từng phương pháp.
- Báo cáo trung thực, gồm cả các kết quả âm tính.

**Báo cáo tổng hợp (Word):** [reports/BAO_CAO_CAI_TIEN_20261031/Bao_cao_cai_tien_DSCMR_CLIP4CMR.docx](reports/BAO_CAO_CAI_TIEN_20261031/Bao_cao_cai_tien_DSCMR_CLIP4CMR.docx)

## Kết quả chính

mAP@ALL (%) trên tập test 200 cặp, trung bình 5 seed. Cấu hình được chọn bằng kiểm định chéo trên train; test chỉ mở một lần.

| Mô hình | Baseline I2T / T2I | Cải tiến I2T / T2I | Tăng (TB 2 chiều) |
|---|---|---|---|
| CLIP4CMR (split tác giả) | 68.48 / 67.87 | **76.63 / 75.68** | **+7.98**, 5/5 seed |
| DSCMR, đầu và loss gốc (split CLIP4CMR) | 68.86 / 69.89 | **74.78 / 72.70** | **+4.36**, 5/5 seed |

Các cải tiến đã dùng:
- (A) Prototype ngữ nghĩa từ câu nhắc CLIP thay cho tâm lớp ngẫu nhiên, cộng adapter phần dư.
- (B) Huấn luyện theo từng câu mô tả.
- (C) Encoder mạnh hơn (DFN5B ViT-H/14, EVA02, SigLIP).
- (D) Fine-tune 2–4 block cuối của encoder ảnh.
- Với DSCMR còn có SupCon cho không gian chung: chỉ cải thiện chiều I2T (+1.15 điểm).

CLIP4CMR cải tiến vượt số công bố của bài báo gốc (69.8 / 69.2). DSCMR không so cùng tập test với bài gốc, vì danh sách ảnh trong split của DSCMR không được công bố (xem [DSCMR/data_ids/NOTE.md](DSCMR/data_ids/NOTE.md)).

## Báo cáo chi tiết

| Nội dung | Báo cáo |
|---|---|
| So sánh với bài báo gốc | [reports/SO_SANH_BAI_BAO_20261030.md](reports/SO_SANH_BAI_BAO_20261030.md) |
| Baseline 5.000 + 1.000 epoch | [reports/RESULTS_5000_1000.md](reports/RESULTS_5000_1000.md) |
| DSCMR: cải tiến loss, vòng 1 và vòng 2 (tuning + 5 seed) | [vòng 1](reports/assignment3_20261029/BAO_CAO.md) · [vòng 2](reports/assignment3_tuning_20261030/BAO_CAO.md) · [protocol](reports/ASSIGNMENT3_PROTOCOL.md) |
| CLIP4CMR: cải tiến A + B | [split local](reports/clip4cmr_improve_20261030/BAO_CAO.md) · [split tác giả](reports/clip4cmr_author_split_20261030/REPORT.md) |
| Vòng 3: encoder mạnh + fine-tune | [reports/improve_v3_20261030/BAO_CAO.md](reports/improve_v3_20261030/BAO_CAO.md) |

## Cấu trúc thư mục

```
scripts/      mã huấn luyện, đánh giá, kiểm chứng và dựng báo cáo
common/       hàm tính metric dùng chung; tests/ kiểm tra metric
reports/      kết quả (JSON/CSV/log/hình) và báo cáo của từng vòng
DSCMR/        source.json (nguồn mã gốc), data_ids/NOTE.md; mã gốc tải về upstream/
CLIP4CMR/     source.json; mã gốc tải về upstream/, dữ liệu trong data/
logs/         log các lượt chạy vòng 3
docs/         ghi chú bố cục
```

Không có trong repo (xem [.gitignore](.gitignore)): mã gốc của tác giả, dữ liệu, đặc trưng, trọng số, checkpoint và môi trường Python.

## Tái lập

1. Tạo môi trường Python 3.10, cài PyTorch có CUDA, rồi chạy `pip install -r requirements-common.txt`.
2. Tải mã gốc DSCMR và CLIP4CMR (kèm đặc trưng Pascal của DSCMR): `python scripts/fetch_selected_20261014.py`.
3. Dữ liệu CLIP4CMR:
   - Tải 1.000 ảnh và câu mô tả Pascal Sentence từ trang UIUC: `python scripts/build_pascal_rn50_local.py` (đọc danh sách trong `reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json`).
   - Tải `clip_train.pkl` và `clip_test.pkl` từ HF Space `shivangibithel/LCMI_T2I` vào `CLIP4CMR/data/hf_mirror_lcmi/`.
   - Kiểm chứng bằng `scripts/verify_hf_mirror_clip4cmr.py`, dựng split bằng `scripts/build_clip4cmr_author_split.py`.
4. Chạy thí nghiệm (xem `--help` của từng script):
   - Cải tiến A + B: `python scripts/clip4cmr_improve.py --feats CLIP4CMR/data/pascal_author_split_mirror --out reports/clip4cmr_author_split_20261030`.
   - DSCMR: `python scripts/dscmr_assignment3_tuning.py --source-root . --out reports/assignment3_tuning_20261030`.
   - Vòng 3: `scripts/extract_encoders.py`, rồi `scripts/improve_v3.py --stage cv|confirm`, rồi `scripts/finetune_v3.py --stage cv|confirm`.

Bảng ánh xạ đầy đủ giữa script và kết quả nằm ở mục 8 của báo cáo Word.

## License

Mã nguồn và tài liệu của dự án dùng [MIT License](LICENSE). Mã gốc, trọng số mô hình và dữ liệu của bên thứ ba giữ điều khoản riêng, liệt kê trong [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
