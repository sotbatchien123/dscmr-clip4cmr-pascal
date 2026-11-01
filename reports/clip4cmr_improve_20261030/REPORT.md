# CLIP4CMR — cải tiến A (adapter phần dư + prototype ngữ nghĩa), B (từng caption), D (backbone)

Test 200 cặp (split cục bộ UIUC seed 42). mAP@ALL %, trung bình ± SD mẫu qua 5 seed; checkpoint theo validation 720/80.

## Tuning α cho A_sem (3-fold CV trên 800 train, không dùng test)

| Backbone | α | CV mAP (%) | Fold 1 / 2 / 3 |
|---|---:|---:|---|
| RN50 | 0.1 | 62.01 ± 3.91 | 62.11 / 65.88 / 58.06 |
| RN50 | 0.3 **← chọn** | 67.54 ± 4.09 | 67.54 / 71.64 / 63.46 |
| RN50 | 0.5 | 66.58 ± 3.81 | 66.92 / 70.20 / 62.61 |
| ViT-L/14 | 0.1 | 63.85 ± 3.89 | 63.50 / 67.89 / 60.14 |
| ViT-L/14 | 0.3 **← chọn** | 69.58 ± 4.12 | 69.18 / 73.88 / 65.66 |
| ViT-L/14 | 0.5 | 68.71 ± 4.11 | 68.36 / 72.98 / 64.78 |

## Kết quả test

| Backbone | Cấu hình | I2T mAP | T2I mAP | TB 2 chiều | I2T R@1 | T2I R@1 |
|---|---|---:|---:|---:|---:|---:|
| RN50 | *zero-shot CLIP (concat, không huấn luyện)* | 60.30 | 59.22 | 59.76 | | |
| RN50 | *zero-shot CLIP (capmean, không huấn luyện)* | 64.01 | 61.27 | 62.64 | | |
| RN50 | base | 70.63 ± 0.62 | 72.14 ± 0.41 | 71.39 | 78.8 | 79.2 |
| RN50 | B | 71.81 ± 0.22 | 72.94 ± 0.40 | 72.38 | 79.1 | 79.7 |
| RN50 | A_res | 70.79 ± 0.43 | 71.07 ± 0.29 | 70.93 | 95.6 | 95.4 |
| RN50 | A_sem | 72.48 ± 0.27 | 72.22 ± 0.17 | 72.35 | 94.1 | 93.1 |
| RN50 | A_sem+B | 73.57 ± 0.38 | 73.14 ± 0.52 | 73.35 | 95.4 | 90.1 |
| ViT-L/14 | *zero-shot CLIP (concat, không huấn luyện)* | 60.75 | 59.82 | 60.28 | | |
| ViT-L/14 | *zero-shot CLIP (capmean, không huấn luyện)* | 64.11 | 62.04 | 63.08 | | |
| ViT-L/14 | base | 72.40 ± 0.40 | 73.52 ± 0.65 | 72.96 | 79.8 | 81.7 |
| ViT-L/14 | B | 72.55 ± 0.34 | 73.52 ± 0.34 | 73.03 | 81.8 | 80.5 |
| ViT-L/14 | A_res | 71.60 ± 0.28 | 71.83 ± 0.29 | 71.71 | 99.0 | 98.2 |
| ViT-L/14 | A_sem | 73.78 ± 0.29 | 73.73 ± 0.39 | 73.76 | 98.5 | 96.3 |
| ViT-L/14 | A_sem+B | 74.13 ± 0.12 | 74.05 ± 0.16 | 74.09 | 98.1 | 95.3 |

## Chênh lệch mAP theo cặp seed so với base cùng backbone (điểm %)

| Backbone | Cấu hình | Chiều | S42 | S43 | S44 | S45 | S46 | TB | Seed tăng |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RN50 | B | I2T | +1.72 | +0.62 | +1.45 | +1.72 | +0.38 | +1.18 | 5/5 |
| RN50 | B | T2I | +0.75 | +0.06 | +1.41 | +1.66 | +0.14 | +0.80 | 5/5 |
| RN50 | A_res | I2T | +0.32 | +0.34 | -0.16 | +0.55 | -0.24 | +0.16 | 3/5 |
| RN50 | A_res | T2I | -1.36 | -1.08 | -0.98 | -0.73 | -1.19 | -1.07 | 0/5 |
| RN50 | A_sem | I2T | +2.82 | +1.23 | +1.79 | +2.52 | +0.89 | +1.85 | 5/5 |
| RN50 | A_sem | T2I | +0.17 | -0.34 | +0.38 | +0.73 | -0.52 | +0.08 | 3/5 |
| RN50 | A_sem+B | I2T | +3.81 | +2.81 | +2.32 | +3.42 | +2.35 | +2.94 | 5/5 |
| RN50 | A_sem+B | T2I | +1.37 | +0.99 | +0.30 | +1.39 | +0.94 | +1.00 | 5/5 |
| ViT-L/14 | B | I2T | +0.27 | -0.09 | -0.06 | +0.09 | +0.52 | +0.15 | 3/5 |
| ViT-L/14 | B | T2I | -0.42 | -0.14 | -0.11 | +0.16 | +0.50 | -0.00 | 2/5 |
| ViT-L/14 | A_res | I2T | -1.50 | -0.45 | -0.98 | -0.56 | -0.50 | -0.80 | 0/5 |
| ViT-L/14 | A_res | T2I | -3.12 | -1.63 | -1.65 | -1.19 | -0.88 | -1.69 | 0/5 |
| ViT-L/14 | A_sem | I2T | +0.71 | +1.22 | +1.08 | +1.46 | +2.44 | +1.38 | 5/5 |
| ViT-L/14 | A_sem | T2I | -0.71 | +0.07 | -0.42 | +0.51 | +1.60 | +0.21 | 3/5 |
| ViT-L/14 | A_sem+B | I2T | +1.18 | +1.94 | +1.36 | +1.89 | +2.25 | +1.73 | 5/5 |
| ViT-L/14 | A_sem+B | T2I | -0.50 | +0.80 | +0.20 | +1.03 | +1.09 | +0.52 | 4/5 |

## Hiệu ứng backbone (D): ViT-L/14 − RN50, cùng cấu hình, cùng seed (điểm %)

| Cấu hình | I2T TB | T2I TB |
|---|---:|---:|
| base | +1.77 | +1.38 |
| B | +0.74 | +0.57 |
| A_res | +0.81 | +0.76 |
| A_sem | +1.30 | +1.51 |
| A_sem+B | +0.55 | +0.91 |

## Phạm vi

- Split 800/200 cục bộ từ UIUC, không phải split của bài báo; không so trực tiếp với số liệu bài báo hay với DSCMR (test 100).
- Base giữ nguyên siêu tham số gốc (không tuning); A_sem được chọn α trên CV (3 giá trị). A_res, A_sem+B dùng α đó.
- B đổi cả biểu diễn văn bản lúc đánh giá (trung bình caption) — đây chính là phương pháp, không phải đổi tập test.
- Prototype từ prompt CLIP và adapter phần dư đã có trong các nghiên cứu CLIP (vd. CLIP-Adapter); đóng góp là áp dụng/phân tích cho CLIP4CMR.
- Test 200 cặp; chênh lệch < ~1 điểm có thể là nhiễu. 5 seed không phải 5 fold.
