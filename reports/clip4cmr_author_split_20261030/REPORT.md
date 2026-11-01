# CLIP4CMR — cải tiến A (adapter phần dư + prototype ngữ nghĩa), B (từng caption), D (backbone)

Đặc trưng: `CLIP4CMR\data\pascal_author_split_mirror` (https://huggingface.co/spaces/shivangibithel/LCMI_T2I (main), third-party mirror, not author-confirmed).

Test 200 cặp. mAP@ALL %, trung bình ± SD mẫu qua 5 seed; checkpoint theo validation 720/80.

## Tuning α cho A_sem (3-fold CV trên 800 train, không dùng test)

| Backbone | α | CV mAP (%) | Fold 1 / 2 / 3 |
|---|---:|---:|---|
| RN50 | 0.1 | 62.82 ± 0.76 | 63.68 / 62.51 / 62.26 |
| RN50 | 0.3 **← chọn** | 67.90 ± 0.97 | 68.60 / 66.80 / 68.31 |
| RN50 | 0.5 | 66.83 ± 1.23 | 67.66 / 65.41 / 67.42 |
| ViT-L/14 | 0.1 | 64.83 ± 0.30 | 64.80 / 65.15 / 64.55 |
| ViT-L/14 | 0.3 **← chọn** | 69.89 ± 0.29 | 70.03 / 69.57 / 70.09 |
| ViT-L/14 | 0.5 | 68.98 ± 0.68 | 69.46 / 68.20 / 69.28 |

## Kết quả test

| Backbone | Cấu hình | I2T mAP | T2I mAP | TB 2 chiều | I2T R@1 | T2I R@1 |
|---|---|---:|---:|---:|---:|---:|
| RN50 | *zero-shot CLIP (concat, không huấn luyện)* | 55.27 | 55.02 | 55.15 | | |
| RN50 | *zero-shot CLIP (capmean, không huấn luyện)* | 59.40 | 57.68 | 58.54 | | |
| RN50 | base | 68.48 ± 0.70 | 67.87 ± 0.45 | 68.17 | 75.8 | 74.6 |
| RN50 | B | 69.22 ± 0.55 | 68.38 ± 0.92 | 68.80 | 76.2 | 74.8 |
| RN50 | A_res | 68.44 ± 0.33 | 68.38 ± 0.46 | 68.41 | 97.2 | 97.5 |
| RN50 | A_sem | 70.35 ± 0.20 | 69.92 ± 0.16 | 70.13 | 93.8 | 95.3 |
| RN50 | A_sem+B | 70.90 ± 0.29 | 70.61 ± 0.26 | 70.76 | 95.0 | 95.2 |
| ViT-L/14 | *zero-shot CLIP (concat, không huấn luyện)* | 53.02 | 54.82 | 53.92 | | |
| ViT-L/14 | *zero-shot CLIP (capmean, không huấn luyện)* | 58.40 | 56.80 | 57.60 | | |
| ViT-L/14 | base | 72.14 ± 0.50 | 71.68 ± 0.29 | 71.91 | 78.9 | 78.7 |
| ViT-L/14 | B | 72.53 ± 0.51 | 72.85 ± 0.18 | 72.69 | 81.4 | 80.6 |
| ViT-L/14 | A_res | 68.90 ± 0.62 | 69.32 ± 0.77 | 69.11 | 98.6 | 97.0 |
| ViT-L/14 | A_sem | 72.42 ± 0.24 | 71.93 ± 0.17 | 72.17 | 97.0 | 93.7 |
| ViT-L/14 | A_sem+B | 72.62 ± 0.56 | 71.80 ± 0.42 | 72.21 | 98.4 | 94.5 |

## Chênh lệch mAP theo cặp seed so với base cùng backbone (điểm %)

| Backbone | Cấu hình | Chiều | S42 | S43 | S44 | S45 | S46 | TB | Seed tăng |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RN50 | B | I2T | +0.98 | +1.85 | -0.12 | +1.62 | -0.65 | +0.74 | 3/5 |
| RN50 | B | T2I | +0.75 | +1.33 | +0.10 | +1.65 | -1.24 | +0.52 | 4/5 |
| RN50 | A_res | I2T | -0.41 | +1.02 | -0.42 | +0.66 | -1.08 | -0.04 | 2/5 |
| RN50 | A_res | T2I | +0.16 | +1.14 | +1.33 | +0.82 | -0.90 | +0.51 | 4/5 |
| RN50 | A_sem | I2T | +1.93 | +2.98 | +1.12 | +2.19 | +1.13 | +1.87 | 5/5 |
| RN50 | A_sem | T2I | +1.78 | +2.24 | +2.33 | +2.47 | +1.43 | +2.05 | 5/5 |
| RN50 | A_sem+B | I2T | +1.99 | +3.84 | +1.73 | +2.74 | +1.79 | +2.42 | 5/5 |
| RN50 | A_sem+B | T2I | +2.26 | +3.21 | +2.85 | +3.23 | +2.17 | +2.74 | 5/5 |
| ViT-L/14 | B | I2T | +0.27 | +0.29 | +0.37 | +0.49 | +0.55 | +0.39 | 5/5 |
| ViT-L/14 | B | T2I | +0.99 | +0.78 | +1.56 | +1.20 | +1.29 | +1.16 | 5/5 |
| ViT-L/14 | A_res | I2T | -4.08 | -3.70 | -2.77 | -2.24 | -3.41 | -3.24 | 0/5 |
| ViT-L/14 | A_res | T2I | -3.71 | -2.68 | -1.97 | -1.32 | -2.16 | -2.37 | 0/5 |
| ViT-L/14 | A_sem | I2T | +0.26 | -0.44 | +1.11 | +0.24 | +0.22 | +0.28 | 4/5 |
| ViT-L/14 | A_sem | T2I | -0.07 | -0.12 | +0.61 | +0.47 | +0.34 | +0.25 | 3/5 |
| ViT-L/14 | A_sem+B | I2T | +0.41 | -0.02 | +1.39 | -0.13 | +0.75 | +0.48 | 3/5 |
| ViT-L/14 | A_sem+B | T2I | -0.07 | -0.19 | +0.53 | -0.24 | +0.54 | +0.11 | 2/5 |

## Hiệu ứng backbone (D): ViT-L/14 − RN50, cùng cấu hình, cùng seed (điểm %)

| Cấu hình | I2T TB | T2I TB |
|---|---:|---:|
| base | +3.66 | +3.82 |
| B | +3.31 | +4.46 |
| A_res | +0.46 | +0.94 |
| A_sem | +2.06 | +2.01 |
| A_sem+B | +1.72 | +1.19 |

## Phạm vi

- Split 800/200 lấy từ bản mirror bên thứ ba của clip_train/test.pkl (khớp pipeline tác giả, chưa được tác giả xác nhận). Bài báo chọn checkpoint bằng test; ở đây chọn bằng validation 80 mẫu tách từ train.
- Base giữ nguyên siêu tham số gốc (không tuning); A_sem được chọn α trên CV (3 giá trị). A_res, A_sem+B dùng α đó.
- B đổi cả biểu diễn văn bản lúc đánh giá (trung bình caption) — đây chính là phương pháp, không phải đổi tập test.
- Prototype từ prompt CLIP và adapter phần dư đã có trong các nghiên cứu CLIP (vd. CLIP-Adapter); đóng góp là áp dụng/phân tích cho CLIP4CMR.
- Test 200 cặp; chênh lệch < ~1 điểm có thể là nhiễu. 5 seed không phải 5 fold.
